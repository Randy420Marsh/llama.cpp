#!/usr/bin/env python3
"""Smoke test for the Orpheus LoRA training path, without the `datasets` lib.

`datasets` segfaults on import in this env (its pandas 3.0.6 wheel), so the
trainer must be fed a plain list of {input_ids, labels, attention_mask} rows
with a small collator. This script proves that works end-to-end:

  * the SNAC tokeniser (the notebook's 7-codes-per-frame layout)
  * a LoRA + 4-step Trainer run on 32 synthetic rows

Run from the speech-server folder (needs the venv):
    venv/bin/python smoke_train.py
"""
import os
os.environ.setdefault("HF_HOME", "/home/john/AI/tts-models/hf")
import numpy as np
import torch

# 1) SNAC tokeniser: same 7-codes/frame interleave as tts_engines._ref_codes
from snac import SNAC
dev = "cuda" if torch.cuda.is_available() else "cpu"
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(dev)
wav = torch.from_numpy(np.sin(2 * np.pi * 220 * np.arange(24000) / 24000).astype("float32")).to(dev).unsqueeze(0).unsqueeze(0)
with torch.inference_mode():
    l1, l2, l3 = [c[0].tolist() for c in snac.encode(wav)]
o = 128266
frames = []
for i in range(len(l1)):
    if 4 * i + 3 >= len(l3) or 2 * i + 1 >= len(l2):
        break
    frames.append([l1[i] + o, l2[2 * i] + o + 4096, l3[4 * i] + o + 2 * 4096,
                   l3[4 * i + 1] + o + 3 * 4096, l2[2 * i + 1] + o + 4 * 4096,
                   l3[4 * i + 2] + o + 5 * 4096, l3[4 * i + 3] + o + 6 * 4096])
codes = [c for f in frames for c in f]
print(f"snac tokenize OK: {len(l1)} frames, {len(codes)} codes")

# 2) base model + LoRA + a list-of-dicts Trainer (no `datasets` import)
import warnings; warnings.filterwarnings("ignore")
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="canopylabs/orpheus-3b-0.1-ft", load_in_4bit=True, max_seq_length=2048)
model = FastLanguageModel.get_peft_model(
    model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")

SOT, EOT = 128000, 128009
SH, EOH, SA, EOS, SOA, EOA = 128259, 128260, 128261, 128258, 128257, 128262
rows = []
for i in range(32):
    text_ids = tokenizer.encode(f"hello world {i}", add_special_tokens=True)
    seq = [SH] + text_ids + [EOH, SA, SOA] + codes + [EOS, EOA]
    rows.append({"input_ids": seq, "labels": seq, "attention_mask": [1] * len(seq)})

from transformers import Trainer, TrainingArguments

class Coll:
    def __init__(self, pad, eos):
        self.pad, self.eos = pad, eos
    def __call__(self, f):
        n = max(len(x["input_ids"]) for x in f)
        im, lb, am = [], [], []
        for x in f:
            L = len(x["input_ids"]); p = n - L
            im.append([self.pad] * p + x["input_ids"])
            am.append([0] * p + [1] * L)
            lb.append([-100] * p + x["labels"])
        return {"input_ids": torch.tensor(im, dtype=torch.long),
                "attention_mask": torch.tensor(am, dtype=torch.long),
                "labels": torch.tensor(lb, dtype=torch.long)}

args = TrainingArguments(output_dir="smoke-out", max_steps=4, per_device_train_batch_size=2,
                         gradient_accumulation_steps=1, optim="adamw_8bit", lr_scheduler_type="constant",
                         warmup_steps=0, logging_steps=2, report_to="none", seed=0,
                         max_grad_norm=1.0, bf16=torch.cuda.is_available())
tr = Trainer(model=model, train_dataset=rows, args=args, data_collator=Coll(128263, EOS))
res = tr.train()
print(f"train OK: loss={res.training_loss:.3f}")
model.save_pretrained("smoke-lora")
print("smoke PASS -- the `datasets`-free Trainer path works")
