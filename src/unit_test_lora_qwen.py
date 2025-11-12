#!/usr/bin/env python
import argparse, json, os, random
from pathlib import Path

import yaml
import torch
import datasets as ds
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model

VALID = {"left","center","right"}

def set_seed(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for ln in f: yield json.loads(ln)

def format_row_train(ex):
    instr, text, label = ex["instruction"], ex["input"], ex["output"].strip().lower()
    prompt = f"{instr}\n\nText:\n{text}\n\nAnswer:"
    return {"prompt": prompt, "label": label}

def build_small_balanced(data_dir: Path, per_label: int, seed: int):
    random.seed(seed)
    pool = [format_row_train(x) for x in load_jsonl(data_dir/"train.jsonl") if x["output"].strip().lower() in VALID]
    by = {"left": [], "center": [], "right": []}
    for r in pool:
        if r["label"] in by: by[r["label"]].append(r)
    for k in by: random.shuffle(by[k])
    small = by["left"][:per_label] + by["center"][:per_label] + by["right"][:per_label]
    random.shuffle(small)
    return ds.Dataset.from_list(small)

def build_small_test(data_dir: Path, k: int, seed: int):
    random.seed(seed)
    pool = list(load_jsonl(data_dir/"test.jsonl"))
    random.shuffle(pool)
    pool = pool[:k]
    return [{"instruction": r["instruction"], "input": r["input"], "label": r["output"].strip().lower()} for r in pool]

def score_label_logprob(model, tok, prompt: str, label_text: str, max_length: int):
    """
    Sum token-level log-probs for the continuation `label_text` given `prompt`.
    No sampling; pure scoring. Works on 8GB GPUs.
    """
    with torch.inference_mode():
        ids_prompt = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        # We score label tokens step-by-step
        lab_ids = tok(label_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(model.device)
        total_logp = 0.0
        cur = ids_prompt["input_ids"][0]
        for tid in lab_ids:
            # forward to get logits for next token
            out = model(input_ids=cur.unsqueeze(0))
            next_logits = out.logits[0, -1]          # (vocab,)
            logp = torch.log_softmax(next_logits, dim=-1)[tid]
            total_logp += float(logp)
            # append token and continue
            cur = torch.cat([cur, tid.view(1)], dim=0)
            if cur.numel() > max_length: break
        return total_logp

def predict_one(model, tok, prompt: str, max_length: int):
    # Use a leading space for proper tokenization
    labels = [("left", " left"), ("center", " center"), ("right", " right")]
    scores = []
    for name, cont in labels:
        scores.append((name, score_label_logprob(model, tok, prompt, cont, max_length)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora_qwen3_0p6b_unit.yaml",
                    help="Tiny YAML for the unit test (keeps params out of code).")
    ap.add_argument("--data_dir", default=None, help="Override data dir if needed.")
    ap.add_argument("--out_dir",  default=None, help="Override output dir if needed.")
    # fast-run knobs (can be overridden by YAML)
    ap.add_argument("--per_label", type=int, default=None, help="Train examples per class.")
    ap.add_argument("--test_k", type=int, default=None, help="Test examples for quick eval.")
    ap.add_argument("--max_steps", type=int, default=None, help="Hard cap on training steps.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_dir = Path(args.data_dir or cfg.get("data_dir", "data/processed"))
    out_dir  = Path(args.out_dir  or cfg.get("output_dir", "adapters/unit_test_qwen3_0p6b_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    per_label = int(args.per_label or cfg.get("per_label", 40))
    test_k    = int(args.test_k or cfg.get("test_k", 180))
    max_steps = int(args.max_steps or cfg.get("max_steps", 120))
    max_len   = int(cfg.get("max_length", 384))
    model_id  = cfg.get("model_id", "Qwen/Qwen3-0.6B")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.truncation_side = "left"  # keep tail "... Answer: <label>"

    # Tiny train/val
    train_ds = build_small_balanced(data_dir, per_label, seed)
    val_ds   = train_ds.select(range(min(60, len(train_ds))))

    def tok_map(batch):
        prompt = batch["prompt"]; label_text = " " + batch["label"]
        enc = tok(prompt, truncation=True, padding="max_length", max_length=max_len)
        input_ids = enc["input_ids"]; labels = [-100] * len(input_ids)
        lab_ids = tok(label_text, add_special_tokens=False).input_ids
        L, K = len(labels), len(lab_ids)
        for i in range(1, min(K, L) + 1):
            labels[-i] = lab_ids[-i]
        enc["labels"] = labels
        return enc

    train_tok = train_ds.map(tok_map, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tok_map,   remove_columns=val_ds.column_names)

    # Model + LoRA (small r for speed)
    dtype = torch.float16 if torch.cuda.is_available() and cfg.get("fp16", True) else None
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    base.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 40),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 40),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        max_steps=max_steps,                        # hard cap
        gradient_checkpointing=True,
        eval_accumulation_steps=16,
        dataloader_pin_memory=False,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # ===== Quick evaluation: base vs tuned on a small slice of test =====
    test_small = build_small_test(data_dir, test_k, seed)

    def make_prompt(item):
        return f"{item['instruction']}\n\nText:\n{item['input']}\n\nAnswer:"

    # reload pristine base for fair compare
    base_eval = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).eval()
    tuned_eval = model.eval()

    y_true, y_pred_base, y_pred_tuned = [], [], []
    for ex in test_small:
        prm = make_prompt(ex); y_true.append(ex["label"])
        y_pred_base.append(predict_one(base_eval, tok, prm, max_len))
        y_pred_tuned.append(predict_one(tuned_eval, tok, prm, max_len))

    def pack_metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "report": classification_report(y_true, y_pred, labels=["left","center","right"], output_dict=True),
            "n": len(y_true),
        }

    m_base  = pack_metrics(y_true, y_pred_base)
    m_tuned = pack_metrics(y_true, y_pred_tuned)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("outputs/unit_test_metrics_base.json").write_text(json.dumps(m_base, indent=2), "utf-8")
    Path("outputs/unit_test_metrics_tuned.json").write_text(json.dumps(m_tuned, indent=2), "utf-8")
    # samples
    rows = []
    for i in range(min(20, len(test_small))):
        rows.append({
            "text_head": test_small[i]["input"][:240],
            "label": y_true[i],
            "base_pred": y_pred_base[i],
            "tuned_pred": y_pred_tuned[i],
        })
    Path("outputs/unit_test_samples.json").write_text(json.dumps(rows, indent=2), "utf-8")

    print("[DONE] Wrote:")
    print(" - outputs/unit_test_metrics_base.json")
    print(" - outputs/unit_test_metrics_tuned.json")
    print(" - outputs/unit_test_samples.json")
    print(f"per_label={per_label}, steps={max_steps}, test_k={test_k}")

if __name__ == "__main__":
    main()
