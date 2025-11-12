import argparse, json, random, re, time, inspect
from pathlib import Path

import torch
import datasets as ds
import yaml
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

VALID = {"left", "center", "right"}

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def format_row(ex):
    instr = ex["instruction"]; text = ex["input"]; label = ex["output"].strip().lower()
    prompt = f"{instr}\n\nText:\n{text}\n\nAnswer:"
    return {"prompt": prompt, "label": label}

def build_ds(data_dir: Path, n_train=None, n_val=None, n_test=None):
    tr = [format_row(x) for x in load_jsonl(data_dir/"train.jsonl")]
    va = [format_row(x) for x in load_jsonl(data_dir/"val.jsonl")]
    te = [format_row(x) for x in load_jsonl(data_dir/"test.jsonl")]
    if n_train: tr = tr[:n_train]
    if n_val:   va = va[:n_val]
    if n_test:  te = te[:n_test]
    return ds.Dataset.from_list(tr), ds.Dataset.from_list(va), ds.Dataset.from_list(te)

def normalize_pred(s: str) -> str:
    if not s: return "center"
    t = re.sub(r"[^a-z]", "", s.lower())
    if t in VALID: return t
    if "left" in t and "right" not in t:   return "left"
    if "right" in t and "left" not in t:   return "right"
    if "center" in t or "centre" in t or "neutral" in t: return "center"
    return "center"

@torch.inference_mode()
def generate_one(model, tok, prompt: str, max_new_tokens: int = 6):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tok.eos_token_id,
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    tail = txt.split("Answer:")[-1].strip()
    first = tail.split()[0] if tail else ""
    return normalize_pred(first)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/lora_qwen3_0p6b_unit.yaml")
    # optional overrides
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir",  default=None)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_val",   type=int, default=0)
    ap.add_argument("--n_test",  type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    seed = int(cfg.get("seed", 42)); set_seed(seed)
    model_id = cfg["model_id"]

    data_dir = Path(args.data_dir or cfg.get("data_dir", "data/processed_unit_test"))
    out_dir  = Path(args.out_dir  or cfg.get("output_dir", "adapters/unit_test_qwen3_0p6b_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # === data ===
    train_ds, val_ds, test_ds = build_ds(
        data_dir,
        n_train=args.n_train or None,
        n_val=args.n_val or None,
        n_test=args.n_test or None,
    )

    # === tokenizer/model ===
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"  # keep the tail "... Answer: <label>"

    max_len = int(cfg.get("max_length", 384))

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
    te_prompts = [r["prompt"] for r in test_ds]
    te_gold    = [r["label"]  for r in test_ds]

    dtype = torch.bfloat16 if (torch.cuda.is_available() and cfg.get("bf16", False)) else None
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    base.gradient_checkpointing_enable()

    # === LoRA ===
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    # === training args from YAML ===
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
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", True),
        report_to=[],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        args_tr = TrainingArguments(
            evaluation_strategy=cfg.get("eval_strategy", "steps"),
            save_strategy="steps",
            **common_kwargs,
        )
    else:
        args_tr = TrainingArguments(
            eval_strategy=cfg.get("eval_strategy", "steps"),
            save_strategy="steps",
            **common_kwargs,
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
