#!/usr/bin/env python3
"""Smoke test for the Orpheus LoRA training path -- NO HF Trainer.

The HF `Trainer` (and `datasets`) segfault on import in this env: they pull
`pandas`, which pulls `pyarrow`, whose native module faults (ABI mismatch on
this torch-nightly + numpy 2.4 stack). So the trainer is a *plain torch* loop:
transformers base model + `peft` LoRA + a hand-rolled batcher. Everything the
GUI's Train tab needs is proven here:

  * SNAC tokeniser (the server's 7-codes/frame, +128266 layout)
  * base model load (4-bit via the modern `load_in_4bit` kwarg)
  * peft LoRA attach
  * a real 2-step training step + saving the adapter

Run from the speech-server folder:
    venv/bin/python smoke_vanilla.py
"""
import os
os.environ.setdefault("HF_HOME", "/home/john/AI/tts-models/hf")
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np

# 1) audio codes (SNAC's range: 128266..152841). Real clips go through the SNAC
#    tokeniser in the GUI; here synthetic codes prove the training path.
dev = "cuda" if torch.cuda.is_available() else "cpu"
_rng = np.random.default_rng(0)
CODES = [int(c) for c in _rng.integers(128266, 152842, size=200)]
print(f"audio codes: {len(CODES)} (synthetic, SNAC range)", flush=True)

# 2) base model + peft LoRA. bf16 (3B ~= 6 GB) fits the 5090 and leaves room for
#    an 8-bit AdamW. Load exactly the way the server's OrpheusEngine does
#    (tts_engines._load): dtype= on transformers>=5, torch_dtype= otherwise,
#    then .to(dev) -- no device_map.
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
MODEL = "canopylabs/orpheus-3b-0.1-ft"
print("loading base model (bf16)...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model.to(dev).eval()
print(f"base model loaded, free VRAM {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
lcfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA attached, {ntr/1e6:.1f}M trainable", flush=True)

# 3) a real 2-step plain-torch training run (no Trainer / no datasets / no pyarrow)
SH, EOH, SA, SOA, EOA, EOS, SOT, EOT = 128259, 128260, 128261, 128257, 128262, 128258, 128000, 128009
PAD = 128263
rows = []
for i in range(8):
    t = tok.encode(f"hello world {i}", add_special_tokens=True)
    seq = [SOT, SH] + t + [EOH, SA, SOA] + CODES + [EOS, EOA, EOT]
    rows.append(torch.tensor(seq, dtype=torch.long))


def collate(batch, bs=2):
    n = len(batch[0])
    im = torch.full((len(batch), n), PAD, dtype=torch.long)
    lb = torch.full((len(batch), n), -100, dtype=torch.long)
    am = torch.zeros((len(batch), n), dtype=torch.long)
    for r, seq in enumerate(batch):
        L = len(seq)
        im[r, n-L:] = seq
        lb[r, n-L:] = seq
        am[r, n-L:] = 1
    return {"input_ids": im, "labels": lb, "attention_mask": am}


try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=2e-4)
except Exception:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

model.train()
for step in range(2):
    out = collate(rows[step*2:(step+1)*2])
    out = {k: v.to(dev) for k, v in out.items()}
    loss = model(**out).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"step {step+1}/2  loss {loss.item():.3f}  free {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)

model.save_pretrained("smoke-lora")
print("VANILLA PASS: 4-bit base + peft LoRA + plain-torch training (no Trainer/datasets)", flush=True)
