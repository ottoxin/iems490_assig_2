# src/train_lora_qwen.py
import argparse, json, inspect
from pathlib import Path
import datasets as ds
import yaml
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def format_row(ex):
    instr = ex["instruction"]; text = ex["input"]; label = ex["output"].strip()
    prompt = f"{instr}\n\nText:\n{text}\n\nAnswer:"
    return {"prompt": prompt, "label": label}

def build_ds(data_dir: Path):
    train = [format_row(x) for x in load_jsonl(data_dir / "train.jsonl")]
    val   = [format_row(x) for x in load_jsonl(data_dir / "val.jsonl")]
    return ds.Dataset.from_list(train), ds.Dataset.from_list(val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir",  default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    data_dir = Path(args.data_dir or cfg.get("data_dir", "data/processed"))
    out_dir  = Path(args.out_dir  or cfg.get("output_dir", "adapters/qwen3_0p6b_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    max_len = int(cfg.get("max_length", 1024))
    patience = int(cfg.get("early_stopping_patience", 2))

    # Data
    train_ds, val_ds = build_ds(data_dir)

    # Tokenizer
    model_id = cfg["model_id"]
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"

    # Label-only loss tokenization
    def tok_map(batch):
        prompt = batch["prompt"]
        label_text = " " + batch["label"]   # leading space helps tokenization
    
        # Encode separately (no special tokens)
        enc_prompt = tok(prompt, add_special_tokens=False)
        enc_label  = tok(label_text, add_special_tokens=False)
    
        prompt_ids = enc_prompt.input_ids
        label_ids  = enc_label.input_ids
    
        # Concatenate then LEFT-truncate to keep the tail "... Answer: <label>"
        input_ids = prompt_ids + label_ids
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
    
        # Figure out how many prompt tokens survived truncation
        # (label is always at the end after concat)
        # If truncation chopped into the prompt, kept_prompt shrinks; if it chopped into the label, we’ll reflect that below.
        kept_prompt = max(0, min(len(prompt_ids), len(input_ids) - len(label_ids)))
        kept_label  = len(input_ids) - kept_prompt  # how many label tokens remain in view
    
        # Build labels: ignore prompt, supervise only the visible label tail
        labels = [-100] * kept_prompt + input_ids[kept_prompt : kept_prompt + kept_label]
    
        # Attention mask
        attn = [1] * len(input_ids)
    
        # Pad on the LEFT to fixed max_len (matching left truncation)
        if len(input_ids) < max_len:
            pad = max_len - len(input_ids)
            pad_id = tok.pad_token_id
            input_ids = [pad_id] * pad + input_ids
            labels    = [-100]  * pad + labels
            attn      = [0]     * pad + attn
    
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    train_tok = train_ds.map(tok_map, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tok_map,   remove_columns=val_ds.column_names)

    # Choose dtype based on fp16/bf16 flags (avoid conflicting with AMP)
    use_bf16 = bool(cfg.get("bf16", False))
    use_fp16 = bool(cfg.get("fp16", False))
    torch_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else None
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True)

    # Optional: gradient checkpointing for memory
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

    # Training args
    common_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg.get("warmup_ratio", None),
        warmup_steps=cfg.get("warmup_steps", None),
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
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
        args_tr = TrainingArguments(evaluation_strategy=cfg.get("eval_strategy", "steps"), **{k: v for k, v in common_kwargs.items() if v is not None})
    else:
        args_tr = TrainingArguments(eval_strategy=cfg.get("eval_strategy", "steps"), **{k: v for k, v in common_kwargs.items() if v is not None})

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

if __name__ == "__main__":
    main()
