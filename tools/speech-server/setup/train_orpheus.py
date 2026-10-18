#!/usr/bin/env python3
"""
Fine-tune Orpheus 3B on your own voice, locally, with Unsloth.

Follows the official unsloth notebook (Orpheus_(3B)-TTS.ipynb) and the
canopylabs recommendations: start from the finetuned model, LoRA rank 64,
batch 1 with gradient accumulation 4, 8-bit AdamW, and (for a strong
result) about 300 clips of 6-15 s of natural speech per speaker.

    python train_orpheus.py --dataset myvoice.jsonl --out myvoice-lora
    python train_orpheus.py --dataset myvoice-dataset --epochs 3   # HF dataset folder

Needs:  pip install unsloth snac "datasets>=3.4.1,<4.0.0"
        (and a CUDA GPU; the unsloth kernel build wants a recent torch)
The model (unsloth/orpheus-3b-0.1-ft, ~3 GB) is downloaded from
Hugging Face on first run; with HF_TOKEN it can also pull the gated
canopylabs original.

What you get:
  <out>/            LoRA adapters (small, a few hundred MB)
  <out>-merged16/   the full model with the adapters merged, float16
  Then turn it into a GGUF the speech server can speak -- see the
  --gguf hint printed at the end (this repo's convert_hf_to_gguf.py +
  quantize), or upload the folder to Hugging Face and use
  unsloth's GGUF conversion notebook.

Data format (make_orpheus_dataset.py produces it): a .jsonl with
{"text": "...", "audio": "path.wav"} per line, optionally a "source"
(speaker) column for multi-speaker training, or a Hugging Face dataset
folder with `text`/`audio` columns.
"""
import argparse
import json
import os
import sys
import wave
from pathlib import Path

# The Orpheus token layout (see the speech server's Orpheus engine for the
# same constants): 128256 base tokens, then the control tokens, then the
# SNAC audio codes interleaved 7 per frame (layer1, layer2, layer3x3, layer2).
BASE = 128256
EOT = 128009
SOS, EOS_S = BASE + 1, BASE + 2          # start/end of speech
SOH, EOH = BASE + 3, BASE + 4           # start/end of human
SOAI, EOAI = BASE + 5, BASE + 6        # start/end of ai
CODE_OFFSET = BASE + 10
CB = 4096                                # one SNAC codebook
SNAC_REPO = "hubertsiuzdak/snac_24khz"
DEFAULT_BASE_MODEL = os.environ.get("ORPHEUS_BASE_MODEL", "unsloth/orpheus-3b-0.1-ft")


def _load_rows(path: Path):
    """jsonl -> list of dicts; a folder -> a Hugging Face dataset -> same."""
    if path.is_file():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    from datasets import load_from_disk
    ds = load_from_disk(str(path))
    return [dict(r) for r in ds]


def _read_audio_24k(p: str):
    """Any clip -> a float32 24 kHz mono waveform (numpy)."""
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
    return np.ascontiguousarray(pcm, dtype="float32")


def _interleave(codes) -> list[int]:
    """SNAC layers -> the model's audio tokens, 7 per frame, in the order
    layer1, layer2, layer3[0..3], layer2, layer3 -- as the model was
    trained (the speech server's _decode_codes does the reverse)."""
    out = []
    for i in range(codes[0].shape[1]):
        out.append(codes[0][0][i].item() + CODE_OFFSET)
        out.append(codes[1][0][2 * i].item() + CODE_OFFSET + CB)
        out.append(codes[2][0][4 * i].item() + CODE_OFFSET + 2 * CB)
        out.append(codes[2][0][4 * i + 1].item() + CODE_OFFSET + 3 * CB)
        out.append(codes[1][0][2 * i + 1].item() + CODE_OFFSET + 4 * CB)
        out.append(codes[2][0][4 * i + 2].item() + CODE_OFFSET + 5 * CB)
        out.append(codes[2][0][4 * i + 3].item() + CODE_OFFSET + 6 * CB)
    return out


def _drop_dup_frames(codes: list[int]) -> list[int]:
    """Consecutive duplicate audio frames add no information; the notebook
    drops them to keep sequences short."""
    if not codes:
        return codes
    res = codes[:7]
    for i in range(7, len(codes) - 6, 7):
        if codes[i] != res[-7]:
            res.extend(codes[i:i + 7])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune Orpheus 3B with Unsloth (LoRA).")
    ap.add_argument("--dataset", required=True, help=".jsonl or a Hugging Face dataset folder")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"base model (default {DEFAULT_BASE_MODEL})")
    ap.add_argument("--out", default="orpheus-finetune", help="output folder for the LoRA adapters")
    ap.add_argument("--merge", default="16", choices=["16", "4", "none"],
                    help="also merge the adapters into a full model (16-bit or 4-bit)")
    ap.add_argument("--rank", type=int, default=64, help="LoRA rank (default 64)")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="max sequence length (the authors train at 8192; 2048 covers ~20 s of speech)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = all epochs)")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.seq_len,
        dtype=None,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- dataset: text + audio -> input ids (the notebook's pipeline) ----
    from snac import SNAC
    import torchaudio.transforms as T
    snac = SNAC.from_pretrained(SNAC_REPO).to("cuda")

    rows = _load_rows(Path(args.dataset))
    multi = "source" in rows[0] if rows else False
    print(f"{len(rows)} rows" + (" (multi-speaker: uses `source`)" if multi else ""))
    if len(rows) < 50:
        print("note: the authors report ~50 clips for a usable result, 300 for a strong one")

    def prep(ex):
        prompt = f"{ex['source']}: {ex['text']}" if multi and ex.get("source") else ex["text"]
        text_ids = tokenizer.encode(prompt or "(speech)", add_special_tokens=True)
        text_ids.append(EOT)
        try:
            wav = _read_audio_24k(ex["audio"])
            import torch as _t
            w = _t.from_numpy(wav).unsqueeze(0).float()
            w = T.Resample(24000, 24000)(w).unsqueeze(0).cuda()
            with _t.inference_mode():
                codes = _interleave(snac.encode(w))
            codes = _drop_dup_frames(codes)
        except Exception as e:
            print(f"  skipping {ex.get('audio')}: {str(e).splitlines()[0]}")
            codes = []
        if len(text_ids) + len(codes) > args.seq_len - 16:
            print(f"  skipping (too long for seq-len {args.seq_len}): {ex.get('audio')}")
            codes = []
        if not codes:
            return None
        ids = ([SOH] + text_ids + [EOH] + [SOAI] + [SOS] + codes + [EOS_S, EOAI])
        ex["input_ids"] = ids
        ex["labels"] = ids
        ex["attention_mask"] = [1] * len(ids)
        return ex

    built, dropped = [], 0
    for ex in rows:
        r = prep(ex)
        if r is None:
            dropped += 1
        else:
            built.append(r)
    print(f"{len(built)} training sequences, {dropped} skipped")
    if not built:
        print("nothing to train on", file=sys.stderr)
        return 1

    from datasets import Dataset
    dataset = Dataset.from_list(built)

    from transformers import TrainingArguments, Trainer
    targs = TrainingArguments(
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.out,
        report_to="none",
    )
    if args.max_steps:
        targs.max_steps = args.max_steps
        targs.num_train_epochs = 0
    trainer = Trainer(model=model, train_dataset=dataset, args=targs)
    print("training...")
    trainer.train()

    # ---- save: adapters, and optionally the merged model ----
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"LoRA adapters -> {args.out}")
    if args.merge != "none":
        folder = f"{args.out}-merged{args.merge}bit"
        method = "merged_16bit" if args.merge == "16" else "merged_4bit"
        model.save_pretrained_merged(folder, tokenizer, save_method=method)
        print(f"merged {args.merge}-bit model -> {folder}")

    # ---- how to get a GGUF the speech server can speak ----
    merge_dir = f"{args.out}-merged{args.merge}bit" if args.merge != "none" else None
    print("\nNext: make a GGUF (this repo):")
    if merge_dir:
        print(f"  python convert_hf_to_gguf.py {merge_dir} --outname orpheus-myvoice-f16.gguf --f16")
        print("  ./build/bin/llama-quantize orpheus-myvoice-f16.gguf orpheus-myvoice-q8_0.gguf Q8_0")
    else:
        print("  merge first (--merge 16), or use the unsloth GGUF conversion notebook")
    print("then copy it next to the other Orpheus quants (Settings -> Voice -> Models shows the")
    print("folder), e.g. <cache>/gguf/orpheus/. The speech server picks up every orpheus*.gguf there;")
    print("a file without 'pretrained' in the name is a *finetuned* quant, so its presets (tara, ...)")
    print("and the emotion tags apply, and your voice becomes a named preset: put a clip of it in")
    print("Settings -> Voice -> My voices and name it whatever you like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
