# src/unit_test_lora_qwen.py
import argparse, json, inspect, time
from pathlib import Path

import torch
import datasets as ds
import yaml
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def format_row(ex):
    instr = ex["instruction"]; text = ex["input"]; label = ex["output"].strip().lower()
    prompt = f"{instr}\n\nText:\n{text}\n\nAnswer:"
    return {"prompt": prompt, "label": label}

def build_ds(data_dir: Path, n_train=None, n_val=None):
    tr = [format_row(x) for x in load_jsonl(data_dir/"train.jsonl")]
    va = [format_row(x) for x in load_jsonl(data_dir/"val.jsonl")]
    if n_train: tr = tr[:n_train]
    if n_val:   va = va[:n_val]
    return ds.Dataset.from_list(tr), ds.Dataset.from_list(va)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/lora_qwen3_0p6b.yaml or _unit.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir",  default=None)
    ap.add_argument("--n_train", type=int, default=0, help="optional small slice")
    ap.add_argument("--n_val",   type=int, default=0, help="optional small slice")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    data_dir = Path(args.data_dir or cfg.get("data_dir", "data/processed"))
    out_dir  = Path(args.out_dir  or cfg.get("output_dir", "adapters/qwen3_0p6b_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # === data ===
    train_ds, val_ds = build_ds(
        data_dir,
        n_train=args.n_train or None,
        n_val=args.n_val or None,
    )

    # === tokenizer/model ===
    model_id = cfg["model_id"]
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"  # keep the tail "... Answer: <label>"

    max_len = int(cfg.get("max_length", 384))

    # label-only loss: supervise only the final label tokens
    def tok_map(batch):
        prompt = batch["prompt"]
        label_text = " " + batch["label"]
        enc = tok(prompt, truncation=True, padding="max_length", max_length=max_len)
        input_ids = enc["input_ids"]
        labels = [-100] * len(input_ids)
        lab_ids = tok(label_text, add_special_tokens=False).input_ids
        L, K = len(labels), len(lab_ids)
        for i in range(1, min(K, L) + 1):
            labels[-i] = lab_ids[-i]
        enc["labels"] = labels
        return enc

    tr_tok = train_ds.map(tok_map, remove_columns=train_ds.column_names)
    va_tok = val_ds.map(tok_map,   remove_columns=val_ds.column_names)

    # dtype
    use_bf16 = bool(cfg.get("bf16", False))
    use_fp16 = bool(cfg.get("fp16", False))
    torch_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else None

    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True)
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    # LoRA
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    # training args
    common_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 1e-4),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 10),
        eval_steps=cfg.get("eval_steps", 40),
        save_steps=cfg.get("save_steps", 40),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=[],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        seed=cfg.get("seed", 42),
        data_seed=cfg.get("seed", 42),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        save_strategy="steps",
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        args_tr = TrainingArguments(
            evaluation_strategy=cfg.get("eval_strategy", "steps"),
            **{k: v for k, v in common_kwargs.items() if v is not None},
        )
    else:
        args_tr = TrainingArguments(
            eval_strategy=cfg.get("eval_strategy", "steps"),
            **{k: v for k, v in common_kwargs.items() if v is not None},
        )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=tr_tok,
        eval_dataset=va_tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience", 1))],
    )

    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    print(f"[TRAIN] Done in {secs:.1f}s. Saving to {out_dir}")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

if __name__ == "__main__":
    main()
