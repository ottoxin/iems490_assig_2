# src/infer_tuned.py
import argparse, json, re, random
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

VALID = {"left","center","right"}

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_pred(s: str) -> str:
    if not s: return "center"
    t = re.sub(r"[^a-z]", "", s.lower())
    if t in VALID: return t
    if "left" in t and "right" not in t:   return "left"
    if "right" in t and "left" not in t:   return "right"
    if "center" in t or "centre" in t or "neutral" in t: return "center"
    return "center"

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def build_prompt(instr: str, text: str) -> str:
    return f"{instr}\n\nText:\n{text}\n\nAnswer:"

@torch.inference_mode()
def generate_one(model, tok, prompt: str, max_new_tokens: int):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    txt = tok.decode(out[0], skip_special_tokens=True)
    tail = txt.split("Answer:")[-1].strip()
    first = tail.split()[0] if tail else ""
    return normalize_pred(first)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--eval_jsonl", required=True, help="data/processed/eval_small.jsonl")
    ap.add_argument("--out_jsonl",  required=True, help="outputs/tuned_eval.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)

    # optional: full test evaluation using generation + parsing
    ap.add_argument("--test_jsonl", default=None)
    ap.add_argument("--preds_jsonl", default=None)
    ap.add_argument("--metrics_json", default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else None

    print(f"[INFO] Loading base: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()

    print(f"[INFO] Loading LoRA adapter from: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()

    # qualitative small eval
    eval_path = Path(args.eval_jsonl)
    out_path  = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_examples = list(load_jsonl(eval_path))
    print(f"[INFO] Running tuned model on small eval set: {len(eval_examples)} items")
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(eval_examples, desc="Eval small (tuned)"):
            prompt = build_prompt(ex["instruction"], ex["input"])
            pred = generate_one(model, tok, prompt, args.max_new_tokens)
            f.write(json.dumps({
                "instruction": ex["instruction"],
                "input": ex["input"][:280],
                "tuned_pred": pred
            }, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote tuned qualitative outputs → {out_path}")

    # optional: labeled test set metrics (F1 only)
    if args.test_jsonl and args.metrics_json and args.preds_jsonl:
        from sklearn.metrics import f1_score

        test_path = Path(args.test_jsonl)
        preds_path = Path(args.preds_jsonl)
        metrics_path = Path(args.metrics_json)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        test_examples = list(load_jsonl(test_path))
        print(f"[INFO] Evaluating tuned model on test: {len(test_examples)} items")

        y_true, y_pred = [], []
        with preds_path.open("w", encoding="utf-8") as pf:
            for ex in tqdm(test_examples, desc="Test (tuned)"):
                prompt = build_prompt(ex["instruction"], ex["input"])
                pred = generate_one(model, tok, prompt, args.max_new_tokens)
                y_pred.append(pred)
                y_true.append(ex["output"].strip().lower())
                pf.write(json.dumps({
                    "pred": pred,
                    "gold": ex["output"].strip().lower()
                }, ensure_ascii=False) + "\n")

        report = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
            "n": len(test_examples),
        }
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] Tuned F1s → {metrics_path}: {report}")

if __name__ == "__main__":
    main()
