"""
Character-level GPT with Block Attention Residuals (AttnRes).

Architecture:
- Standard Transformer (PreNorm, multi-head causal self-attention, MLP)
- Block AttnRes replacing standard residual connections, exactly per the paper
- Context length: 8192
- Character-level tokenization

AttnRes design (faithful to paper):
- Per-sub-layer learnable pseudo-query w_l ∈ R^d, initialized to zero
- RMSNorm on keys prevents magnitude-dominated attention
- L total sub-layers divided into N_blocks groups of S = L/N_blocks sub-layers
- Intra-block: outputs accumulated in partial_block
- Inter-block: completed blocks pushed to 'blocks' list; final block stays as partial_block
- We trigger N_blocks-1 boundaries (the last block never commits), so partial_block
  is always non-zero at the end → final readout AttnRes aggregates over all sources
- Total depth-attention sources = 1 (embedding) + (N_blocks-1) committed blocks
  + 1 current partial_block = N_blocks + 1
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    # model
    vocab_size: int = 108
    context_len: int = 8192
    n_layer: int = 12        # number of Transformer blocks (each = attn + mlp)
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.0

    # AttnRes: divide total sub-layers (= n_layer * 2) into n_attnres_blocks groups
    # We trigger n_attnres_blocks-1 boundaries → last block stays as partial_block
    # Total depth sources = 1 (emb) + (n_attnres_blocks-1) + 1 (partial) = n_attnres_blocks + 1
    n_attnres_blocks: int = 8

    # training
    batch_size: int = 4          # per-GPU micro batch
    grad_accum: int = 2
    max_iters: int = 20000
    lr: float = 3e-4
    lr_min: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    save_interval: int = 2000
    log_interval: int = 50
    data_path: str = '/apdcephfs/private_daisyjguo/dataset/corpus_crawled.txt'
    save_dir: str = '/apdcephfs/private_daisyjguo/attnres/checkpoints'


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


# ---------------------------------------------------------------------------
# AttnRes aggregation (one per sub-layer entry point)
# ---------------------------------------------------------------------------

class AttnResAggregator(nn.Module):
    """
    Computes depth-wise softmax attention over a list of block representations
    plus the current partial-block sum.

    Sources: [b_0, b_1, ..., b_{n-1}, partial_block]
    Query:   w_l ∈ R^d (learned, initialized to zero per paper)
    Key:     RMSNorm(v_i) for each source v_i (prevents magnitude dominance)
    Value:   v_i (same as key input, before norm)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNorm(d_model)

    def forward(self, blocks: list, partial_block: torch.Tensor) -> torch.Tensor:
        # sources: list of completed blocks + current partial_block
        all_sources = blocks + [partial_block]          # list of [B, T, D]
        V = torch.stack(all_sources, dim=0)             # [S, B, T, D]
        K = self.key_norm(V)                            # [S, B, T, D]
        # logits: dot product of pseudo-query with each key
        logits = torch.einsum('d, s b t d -> s b t', self.pseudo_query, K)  # [S, B, T]
        weights = logits.softmax(dim=0)                 # [S, B, T]
        h = torch.einsum('s b t, s b t d -> b t d', weights, V)             # [B, T, D]
        return h


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop_p = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop_p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Transformer Block with Block AttnRes
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One Transformer block = one Attention sub-layer + one MLP sub-layer.

    Each sub-layer:
      1. Computes h = AttnRes(blocks, partial_block)   — depth-wise attention input
      2. Applies sub-layer: out = f(norm(h))
      3. Accumulates: partial_block += out
      4. If this sub-layer index is a block boundary (set from outside):
           blocks = blocks + [partial_block]
           partial_block = zeros_like(partial_block)

    Args:
        attn_sub_idx: global 0-based index of the attention sub-layer
        mlp_sub_idx:  global 0-based index of the MLP sub-layer
        boundary_set: set of sub-layer indices that trigger a block commit+reset
    """

    def __init__(self, cfg: GPTConfig, attn_sub_idx: int, mlp_sub_idx: int,
                 boundary_set: set):
        super().__init__()
        self.attn_sub_idx = attn_sub_idx
        self.mlp_sub_idx  = mlp_sub_idx
        self.boundary_set = boundary_set

        self.attn_norm = RMSNorm(cfg.d_model)
        self.mlp_norm  = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)

        # One AttnResAggregator per sub-layer entry (pseudo-query + key_norm)
        self.attn_res = AttnResAggregator(cfg.d_model)
        self.mlp_res  = AttnResAggregator(cfg.d_model)

    def forward(self, blocks: list, partial_block: torch.Tensor):
        # --- Attention sub-layer ---
        h = self.attn_res(blocks, partial_block)
        attn_out = self.attn(self.attn_norm(h))
        partial_block = partial_block + attn_out

        if self.attn_sub_idx in self.boundary_set:
            blocks = blocks + [partial_block]
            partial_block = torch.zeros_like(partial_block)

        # --- MLP sub-layer ---
        h = self.mlp_res(blocks, partial_block)
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        if self.mlp_sub_idx in self.boundary_set:
            blocks = blocks + [partial_block]
            partial_block = torch.zeros_like(partial_block)

        return blocks, partial_block


# ---------------------------------------------------------------------------
# GPT with Block AttnRes
# ---------------------------------------------------------------------------

class AttnResGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        total_sub_layers = cfg.n_layer * 2
        assert total_sub_layers % cfg.n_attnres_blocks == 0, (
            f"n_layer*2={total_sub_layers} must be divisible by "
            f"n_attnres_blocks={cfg.n_attnres_blocks}"
        )
        sub_per_block = total_sub_layers // cfg.n_attnres_blocks  # S

        # Boundary sub-layer indices: commit partial_block to blocks list.
        # We trigger n_attnres_blocks-1 boundaries (exclude the last one at
        # total_sub_layers-1) so that partial_block is always non-zero at the end.
        # Boundaries occur after sub-layers: S-1, 2S-1, ..., (N-1)*S-1
        boundary_set = set()
        for k in range(1, cfg.n_attnres_blocks):           # 1 .. N-1
            boundary_set.add(k * sub_per_block - 1)
        # NOTE: total_sub_layers-1 (= N*S-1) is intentionally NOT in boundary_set

        self.sub_per_block = sub_per_block
        self.boundary_set  = boundary_set

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.context_len, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        self.transformer_blocks = nn.ModuleList()
        for i in range(cfg.n_layer):
            attn_sub = i * 2
            mlp_sub  = i * 2 + 1
            self.transformer_blocks.append(
                TransformerBlock(cfg, attn_sub, mlp_sub, boundary_set)
            )

        # Final readout: one AttnResAggregator that aggregates ALL depth sources
        # (completed blocks + final partial_block) into h_final for the LM head.
        self.final_attn_res = AttnResAggregator(cfg.d_model)
        self.final_norm     = RMSNorm(cfg.d_model)
        self.lm_head        = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: embedding and LM head share weights
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # pseudo_query params are already zero from AttnResAggregator init

        n_sources = 1 + (cfg.n_attnres_blocks - 1) + 1  # emb + committed + partial
        print(f"AttnResGPT: {self.num_params()/1e6:.2f}M parameters")
        print(f"  n_layer={cfg.n_layer}, n_head={cfg.n_head}, d_model={cfg.d_model}")
        print(f"  n_attnres_blocks={cfg.n_attnres_blocks}, sub_per_block={sub_per_block}")
        print(f"  boundary sub-layer indices: {sorted(boundary_set)}")
        print(f"  depth-attention sources at final readout: {n_sources}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        # pseudo_query left as zeros per paper requirement

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.cfg.context_len, f"T={T} > context_len={self.cfg.context_len}"

        # Embeddings
        tok = self.token_emb(idx)                      # [B, T, D]
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # [T, D]
        x = self.emb_drop(tok + pos)                   # [B, T, D]

        # AttnRes initial state:
        #   blocks = [b_0] where b_0 = token embedding (per paper)
        #   partial_block = zeros (no outputs accumulated yet)
        blocks = [x]
        partial_block = torch.zeros_like(x)

        # Forward through all Transformer blocks
        for blk in self.transformer_blocks:
            blocks, partial_block = blk(blocks, partial_block)

        # Final readout: aggregate over all depth sources
        # At this point:
        #   blocks[0]   = token embedding
        #   blocks[1..] = committed block representations (n_attnres_blocks-1 of them)
        #   partial_block = last block's accumulated outputs (non-zero, not committed)
        h_final = self.final_attn_res(blocks, partial_block)   # [B, T, D]
        h_final = self.final_norm(h_final)

        if targets is not None:
            logits = self.lm_head(h_final)              # [B, T, V]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            return logits, loss
        else:
            logits = self.lm_head(h_final[:, [-1], :])  # [B, 1, V]
            return logits, None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = (idx if idx.size(1) <= self.cfg.context_len
                        else idx[:, -self.cfg.context_len:])
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CharDataset:
    def __init__(self, data_path: str, context_len: int):
        print(f"Loading data from {data_path}...")
        raw = open(data_path, 'r', encoding='utf-8', errors='replace').read().lower()
        print(f"Dataset size: {len(raw):,} characters")

        chars = sorted(set(raw))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        print(f"Vocab size: {self.vocab_size}")

        data = torch.tensor([self.stoi[c] for c in raw], dtype=torch.long)
        n = int(0.95 * len(data))
        self.train_data = data[:n]
        self.val_data   = data[n:]
        self.context_len = context_len
        print(f"Train: {len(self.train_data):,} | Val: {len(self.val_data):,}")

    def get_batch(self, split: str, batch_size: int, device):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.context_len, (batch_size,))
        x = torch.stack([data[i     : i + self.context_len    ] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_len + 1] for i in ix])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# LR schedule: cosine with linear warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: GPTConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.lr * step / cfg.warmup_iters
    if step > cfg.max_iters:
        return cfg.lr_min
    ratio = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.lr_min + coeff * (cfg.lr - cfg.lr_min)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    cfg = GPTConfig()

    use_ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if use_ddp:
        dist.init_process_group(backend='nccl')
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device     = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_master  = rank == 0
    else:
        rank = 0; world_size = 1
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_master = True

    if is_master:
        os.makedirs(cfg.save_dir, exist_ok=True)
        print(f"Device: {device}, world_size: {world_size}")

    dataset = CharDataset(cfg.data_path, cfg.context_len)
    cfg.vocab_size = dataset.vocab_size

    torch.manual_seed(42 + rank)
    model = AttnResGPT(cfg).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if use_ddp else model

    # Separate weight decay groups
    decay_params    = []
    no_decay_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim < 2
                or 'emb' in name
                or 'norm' in name
                or 'bias' in name
                or 'pseudo_query' in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [{'params': decay_params,    'weight_decay': cfg.weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    if is_master:
        print(f"decay={sum(p.numel() for p in decay_params):,}  "
              f"no_decay={sum(p.numel() for p in no_decay_params):,}")

    scaler      = torch.amp.GradScaler('cuda')
    best_val    = float('inf')
    t0          = time.time()

    for step in range(cfg.max_iters + 1):
        # ---- evaluation ----
        if is_master and step % cfg.eval_interval == 0:
            raw_model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(20):
                    xv, yv = dataset.get_batch('val', cfg.batch_size, device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _, lv = raw_model(xv, yv)
                    val_losses.append(lv.item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"[step {step:5d}] val_loss={val_loss:.4f}")
            raw_model.train()
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'step': step, 'val_loss': val_loss,
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'cfg': cfg, 'stoi': dataset.stoi, 'itos': dataset.itos,
                }, os.path.join(cfg.save_dir, 'best.pt'))
                print(f"  >> saved best (val={val_loss:.4f})")

        if step == cfg.max_iters:
            break

        # ---- lr update ----
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ---- gradient accumulation ----
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(cfg.grad_accum):
            x, y = dataset.get_batch('train', cfg.batch_size, device)
            if use_ddp:
                model.require_backward_grad_sync = (micro == cfg.grad_accum - 1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ---- logging ----
        if is_master and step % cfg.log_interval == 0:
            t1 = time.time()
            tok_per_s = (cfg.batch_size * cfg.context_len * cfg.grad_accum
                         * world_size * cfg.log_interval) / (t1 - t0)
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"{tok_per_s/1e3:.1f}K tok/s")
            t0 = t1

        # ---- periodic checkpoint ----
        if is_master and step % cfg.save_interval == 0 and step > 0:
            torch.save({
                'step': step,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg, 'stoi': dataset.stoi, 'itos': dataset.itos,
            }, os.path.join(cfg.save_dir, f'ckpt_{step:06d}.pt'))
            print(f"  >> saved ckpt_{step:06d}.pt")

    if is_master:
        print("Training complete!")


if __name__ == '__main__':
    train()
