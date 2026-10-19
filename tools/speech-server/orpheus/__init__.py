"""Orpheus 3B LoRA fine-tuning studio: build a voice dataset, train a LoRA,
publish it as a GGUF adapter, and A/B-test it against the base model.

Run with ``start-orpheus-studio.sh`` (or ``.bat``). The GUI itself is a Gradio
app that does no heavy work -- it shells out to the speech-server venv for
dataset building, LoRA training, GGUF conversion and synthesis.
"""
