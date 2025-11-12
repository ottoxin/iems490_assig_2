# IEMS 490 Assignment 2: Political Bias Fine-Tuning (LoRA on Qwen3-0.6B)

Classify news articles as **left / center / right** using a small LoRA adapter on `Qwen/Qwen3-0.6B`. This repo includes:

* data prep to build `train/val/test` JSONL splits from the Kaggle dataset,
* LoRA fine-tuning code with a label-only loss,
* base vs tuned evaluation scripts,
* a fast unit test that runs in ~5–10 minutes,
* a Dockerfile for reproducibility.
