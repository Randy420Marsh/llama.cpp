#!/usr/bin/env python3
"""Fine-tune Orpheus 3B on a voice dataset with LoRA -- plain PyTorch, no HF Trainer.

Why not the HF ``Trainer``? It (and ``datasets``) pulls in ``pandas`` -> ``pyarrow``
whose native module segfaults on this torch-nightly stack, so the whole training path
is a hand-rolled loop: transformers base model + ``peft`` LoRA + an 8-bit AdamW.
Everything else (SNAC tokeniser, the 7-codes/frame interleaving, the turn control
tokens) is exactly what the speech server's ``OrpheusEngine`` uses, so the adapters
it produces drop straight into ``llama-server --lora``.

    venv/bin/python orpheus/train_lora.py --dataset myvoice.jsonl --out myvoice-lora

Needs the speech-server venv (torch + peft + snac + bitsandbytes + soundfile).
The base model (canopylabs/orpheus-3b-0.1-ft, ~3 GB) is read from the HF cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/john/AI/tts-models/hf")

# The Orpheus token layout -- identical to the speech server's OrpheusEngine.
BASE = 128256
SOT, EOT, EOH = BASE + 3, BASE + 9, BASE + 4        # start human / end turn / end human
SOAI, EOAI = BASE + 5, BASE + 6                    # start ai / end ai
SOS, EOS = BASE + 1, BASE + 2                      # start audio / end of sequence
CODE_OFFSET = BASE + 10                            # 128266
CB = 4096                                          # one SNAC codebook
SNAC_REPO = "hubertsiuzdak/snac_24khz"
DEFAULT_BASE = os.environ.get("ORPHEUS_BASE_MODEL", "canopylabs/orpheus-3b-0.1-ft")


def _read_audio_24k(p: str) -> "list[float]":
    """Any clip -> a float32 24 kHz mono waveform (list of floats)."""
    import numpy as np
    p = Path(p)
    if p.suffix.lower() == ".wav":
        with wave.open(str(p), "rb") as wf:
            sr, ch = wf.getframerate(), wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype("float32") / 32768.0
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
    else:
        import soundfile as sf
        pcm, sr = sf.read(str(p), dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
    if sr != 24000:
        from math import gcd
        try:
            from scipy.signal import resample_poly
            g = gcd(sr, 24000)
            pcm = resample_poly(pcm, 24000 // g, sr // g)
        except Exception:
            x = np.linspace(0, len(pcm) - 1, int(len(pcm) * 24000 / sr))
            pcm = np.interp(x, np.arange(len(pcm)), pcm)
    return pcm


def _load_rows(path: Path):
    if path.is_file():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    from datasets import load_from_disk
    ds = load_from_disk(str(path))
    return [dict(r) for r in ds]


def _interleave(layers) -> list[int]:
    """SNAC (l1, l2, l3) -> the model's audio tokens, 7 per frame, in the order
    layer1, layer2, layer3[0..3], layer2, layer3 -- the server's reverse is _decode_codes."""
    out: list[int] = []
    for i in range(len(layers[0])):
        out += [layers[0][i] + CODE_OFFSET,
                layers[1][2 * i] + CODE_OFFSET + CB,
                layers[2][4 * i] + CODE_OFFSET + 2 * CB,
                layers[2][4 * i + 1] + CODE_OFFSET + 3 * CB,
                layers[1][2 * i + 1] + CODE_OFFSET + 4 * CB,
                layers[2][4 * i + 2] + CODE_OFFSET + 5 * CB,
                layers[2][4 * i + 3] + CODE_OFFSET + 6 * CB]
    return out


def _drop_dup_frames(codes: list[int]) -> list[int]:
    """The training data dropped a frame whose first code repeats the previous frame's
    (remove_duplicate_frames in the data prep): show the model the kind of sequence it learned from."""
    if not codes:
        return codes
    res = codes[:7]
    for i in range(7, len(codes) - 6, 7):
        if codes[i] != res[-7]:
            res.extend(codes[i:i + 7])
    return res


def build_examples(rows, snac, tokenizer, seq_len: int):
    """rows -> (input_ids, labels, attention_mask) per example, audio SNAC-encoded."""
    import numpy as np
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    multi = "source" in rows[0] if rows else False
    built = []
    for ex in rows:
        prompt = f"{ex['source']}: {ex['text']}" if multi and ex.get("source") else ex["text"]
        if not prompt.strip():
            continue
        text_ids = tokenizer.encode(prompt, add_special_tokens=True) + [EOT]
        try:
            wav = _read_audio_24k(ex["audio"])
            w = torch.from_numpy(np.ascontiguousarray(wav, dtype="float32"))
            w = w.unsqueeze(0).unsqueeze(0).to(dev)
            with torch.inference_mode():
                l1, l2, l3 = [c[0].tolist() for c in snac.encode(w)]
            codes = _drop_dup_frames(_interleave((l1, l2, l3)))
        except Exception as e:
            print(f"  skipping {ex.get('audio')}: {str(e).splitlines()[0]}", flush=True)
            continue
        ids = [SOT] + text_ids + [EOH, SOAI, SOS] + codes + [EOS, EOAI]
        if len(ids) > seq_len:
            print(f"  skipping (too long for seq-len {seq_len}): {ex.get('audio')}", flush=True)
            continue
        built.append({"input_ids": ids, "labels": ids, "attention_mask": [1] * len(ids)})
    return built


def collate(batch, pad_id: int, maxlen: int, dev):
    import torch
    n = min(maxlen, max(len(b["input_ids"]) for b in batch))
    im = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lb = torch.full((len(batch), n), -100, dtype=torch.long)
    am = torch.zeros((len(batch), n), dtype=torch.long)
    for r, b in enumerate(batch):
        L = min(n, len(b["input_ids"]))
        im[r, n - L:] = torch.tensor(b["input_ids"][-L:], dtype=torch.long)
        lb[r, n - L:] = torch.tensor(b["labels"][-L:], dtype=torch.long)
        am[r, n - L:] = 1
    return {k: v.to(dev) for k, v in {"input_ids": im, "labels": lb, "attention_mask": am}.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune Orpheus 3B (LoRA), plain-torch, no HF Trainer.")
    ap.add_argument("--dataset", required=True, help=".jsonl or a Hugging Face dataset folder")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--out", default="orpheus-lora", help="output folder for the LoRA adapters")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2, help="sequences per step (kept small: VRAM)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = all epochs)")
    ap.add_argument("--save-every", type=int, default=50, help="save a checkpoint every N steps")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--cpu", action="store_true", help="train on CPU (slow; for small data)")
    args = ap.parse_args()

    import numpy as np
    import torch
    from snac import SNAC
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"loading SNAC tokeniser on {dev}...", flush=True)
    snac = SNAC.from_pretrained(SNAC_REPO).eval().to(dev)

    rows = _load_rows(Path(args.dataset))
    print(f"{len(rows)} rows", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    examples = build_examples(rows, snac, tok, args.seq_len)
    if not examples:
        print("no usable examples", file=sys.stderr)
        return 1
    print(f"{len(examples)} usable examples", flush=True)

    print(f"loading base model ({args.base_model})...", flush=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 128263
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model.to(dev)
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    if dev == "cuda":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA attached, {ntr / 1e6:.1f}M trainable", flush=True)

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    except Exception:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    loss_sum = 0.0
    for epoch in range(args.epochs):
        np.random.shuffle(examples)
        for i in range(0, len(examples), args.batch):
            batch = collate(examples[i:i + args.batch], pad, args.seq_len, dev)
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            loss_sum += loss.item()
            if step % 5 == 0 or step == 1:
                print(f"epoch {epoch + 1}/{args.epochs} step {step} loss {loss.item():.3f} "
                      f"avg {loss_sum / step:.3f} free {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)
            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(out / f"ckpt-{step}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    model.save_pretrained(out)
    print(f"saved LoRA adapters to {out}", flush=True)
    print("next: convert to GGUF (llama.cpp convert_lora_to_gguf.py) and serve with llama-server --lora")
    return 0


if __name__ == "__main__":
    sys.exit(main())
