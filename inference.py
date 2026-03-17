"""
Inference script for AttnRes GPT.
Loads the best checkpoint and generates text from a prompt, 3 times.
"""

import sys
import torch

sys.path.insert(0, '/apdcephfs/private_daisyjguo/attnres')
from train_attnres_gpt import AttnResGPT, GPTConfig

CKPT_PATH   = '/apdcephfs/private_daisyjguo/attnres/checkpoints/best.pt'
OUTPUT_PATH = '/apdcephfs/private_daisyjguo/attnres/inference_output.txt'
PROMPT      = 'once upon a time, there was a girl in the forest.'
MAX_NEW     = 4096
TEMPERATURE = 0.8
TOP_K       = 64
NUM_RUNS    = 3

device = torch.device('cuda:0')

print(f"Loading checkpoint from {CKPT_PATH}...")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg: GPTConfig = ckpt['cfg']
stoi = ckpt['stoi']
itos = ckpt['itos']

model = AttnResGPT(cfg).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded checkpoint (step={ckpt.get('step','?')}, val_loss={ckpt.get('val_loss','?'):.4f})")

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    return ''.join(itos[i] for i in ids)

results = []
for run in range(1, NUM_RUNS + 1):
    print(f"\n--- Run {run}/{NUM_RUNS} ---")
    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=MAX_NEW,
                             temperature=TEMPERATURE, top_k=TOP_K)
    text = decode(out[0].tolist())
    results.append(text)
    # print first 300 chars as preview
    print(text[:300] + ('...' if len(text) > 300 else ''))

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for i, text in enumerate(results, 1):
        f.write(f"{'='*60}\n")
        f.write(f"RUN {i}  (prompt: {PROMPT!r})\n")
        f.write(f"  temperature={TEMPERATURE}, top_k={TOP_K}, max_new_tokens={MAX_NEW}\n")
        f.write(f"{'='*60}\n\n")
        f.write(text)
        f.write('\n\n')

print(f"\nOutput written to {OUTPUT_PATH}")
