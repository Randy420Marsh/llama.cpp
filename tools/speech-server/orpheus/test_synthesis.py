#!/usr/bin/env python3
"""Prove a trained Orpheus LoRA actually speaks -- zero-shot clone + LoRA, HF side.

Loads the base model (bf16) + a trained LoRA adapter, builds the same zero-shot
voice-cloning prompt the speech server's OrpheusEngine builds (a finished example
turn in the reference voice, then the new text), samples audio codes, decodes them
through SNAC and writes a 24 kHz WAV. This is the in-GPU counterpart of what
``llama-server --lora`` does for the server, so the GUI can A/B the adapter against
the plain model by ear.

    venv/bin/python orpheus/test_synthesis.py --lora myvoice-lora \
        --ref ~/AI/tts-models/voices/<id>/reference.wav --ref-text "..." \
        --text "Read this out loud." --out out.wav

Needs the speech-server venv. The base model + SNAC come from the HF cache.
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import wave
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/john/AI/tts-models/hf")

BASE = 128256
SOT, EOT, EOH = BASE + 3, BASE + 9, BASE + 4
SOAI, EOAI = BASE + 5, BASE + 6
SOS, EOS = BASE + 1, BASE + 2
CODE_OFFSET = BASE + 10
CB = 4096
SNAC_REPO = "hubertsiuzdak/snac_24khz"
DEFAULT_BASE = os.environ.get("ORPHEUS_BASE_MODEL", "canopylabs/orpheus-3b-0.1-ft")


def _codes_from_snac(wav: Path, snac, dev) -> list[int]:
    import torch
    import numpy as np
    import soundfile as sf
    pcm, sr = sf.read(str(wav), dtype="float32")
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
    w = torch.from_numpy(pcm).to(dev).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        l1, l2, l3 = [c[0].tolist() for c in snac.encode(w)]
    frames: list[int] = []
    for i in range(len(l1)):
        if 4 * i + 3 >= len(l3) or 2 * i + 1 >= len(l2):
            break
        frames += [l1[i] + CODE_OFFSET,
                   l2[2 * i] + CODE_OFFSET + CB,
                   l3[4 * i] + CODE_OFFSET + 2 * CB,
                   l3[4 * i + 1] + CODE_OFFSET + 3 * CB,
                   l2[2 * i + 1] + CODE_OFFSET + 4 * CB,
                   l3[4 * i + 2] + CODE_OFFSET + 5 * CB,
                   l3[4 * i + 3] + CODE_OFFSET + 6 * CB]
    return frames


def _decode(codes, snac, dev) -> bytes:
    import numpy as np
    import torch
    frames = [codes[i:i + 7] for i in range(0, len(codes) // 7 * 7, 7)]
    frames = [f for f in frames if all(0 <= c - k * CB < CB for k, c in enumerate(f))]
    if not frames:
        return b""
    l1, l2, l3 = [], [], []
    for f in frames:
        l1.append(f[0])
        l2 += [f[1] - CB, f[4] - 4 * CB]
        l3 += [f[2] - 2 * CB, f[3] - 3 * CB, f[5] - 5 * CB, f[6] - 6 * CB]
    layers = [torch.tensor(x, device=dev).unsqueeze(0) for x in (l1, l2, l3)]
    with torch.no_grad():
        audio = snac.decode(layers)
    a = audio.squeeze().float().cpu().numpy()
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def _wav(pcm16: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize speech with an Orpheus LoRA (HF, zero-shot).")
    ap.add_argument("--lora", required=True, help="folder with the trained LoRA adapters")
    ap.add_argument("--ref", required=True, help="reference clip of the target voice")
    ap.add_argument("--ref-text", default="", help="what the reference clip says")
    ap.add_argument("--text", required=True, help="text to speak")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--out", default="orpheus-out.wav")
    ap.add_argument("--budget", type=int, default=0, help="max audio tokens (0 = from text length)")
    ap.add_argument("--no-lora", action="store_true", help="base model only (the A/B control)")
    args = ap.parse_args()

    import numpy as np
    import torch
    from snac import SNAC
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading SNAC + base model on {dev}...", flush=True)
    snac = SNAC.from_pretrained(SNAC_REPO).eval().to(dev)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model.to(dev)

    if not args.no_lora:
        from peft import PeftModel
        print(f"loading LoRA {args.lora}...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    ref_codes = _codes_from_snac(Path(args.ref), snac, dev)
    ref_ids = tok.encode(args.ref_text or "(speech)", add_special_tokens=True)
    tgt_ids = tok.encode(args.text, add_special_tokens=True)
    prompt = ([SOT] + ref_ids + [EOT, EOH, SOAI, SOS] + ref_codes + [EOS, EOAI]
              + [SOT] + tgt_ids + [EOT, EOH, SOAI, SOS])
    budget = args.budget or min(2000, 200 + 75 * len(args.text.split()))
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt], device=dev), max_new_tokens=budget,
                             do_sample=True, temperature=0.6, top_p=0.95,
                             repetition_penalty=1.1, eos_token_id=EOS)
    row = out[0].tolist()
    if SOS in row:
        row = row[len(row) - 1 - row[::-1].index(SOS) + 1:]
    codes = [t - CODE_OFFSET for t in row if t != EOS]
    pcm = _decode(codes, snac, dev)
    wav = _wav(pcm, 24000)
    Path(args.out).write_bytes(wav)
    print(f"wrote {args.out} ({len(pcm) // 2 / 24000:.1f} s, {len(codes)} audio tokens)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
