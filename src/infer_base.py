# src/infer_base.py
import argparse, json, re, random
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    ap.add_argument("--eval_jsonl", required=True, help="data/processed/eval_small.jsonl")
    ap.add_argument("--out_jsonl",  required=True, help="outputs/base_eval.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)

    # quick metrics on a small labeled subset of test.jsonl
    ap.add_argument("--quick_metrics_k", type=int, default=0,
                    help="If >0, evaluate on K samples from test.jsonl and save F1s.")
    ap.add_argument("--test_jsonl", default="data/processed/test.jsonl",
                    help="Path to test split for quick metrics.")
    ap.add_argument("--quick_metrics_out", default="outputs/base_metrics_quick.json",
                    help="Where to save quick metrics JSON.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else None

    print(f"[INFO] Loading base model: {args.model_id} (device={device})")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()

    # qualitative small eval
    eval_path = Path(args.eval_jsonl)
    out_path  = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_examples = list(load_jsonl(eval_path))
    print(f"[INFO] Running baseline on small eval set: {len(eval_examples)} items")
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(eval_examples, desc="Eval small"):
            prompt = build_prompt(ex["instruction"], ex["input"])
            pred = generate_one(model, tok, prompt, args.max_new_tokens)
            f.write(json.dumps({
                "instruction": ex["instruction"],
                "input": ex["input"][:280],
                "base_pred": pred
            }, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote qualitative outputs → {out_path}")

    # quick F1 metrics (no accuracy)
    if args.quick_metrics_k and args.quick_metrics_k > 0:
        from sklearn.metrics import f1_score

        test_path = Path(args.test_jsonl)
        test_examples = list(load_jsonl(test_path))
        if len(test_examples) == 0:
            print("[WARN] test.jsonl is empty; skipping quick metrics.")
            return

        k = min(args.quick_metrics_k, len(test_examples))
        random.shuffle(test_examples)
        test_examples = test_examples[:k]
        print(f"[INFO] Quick F1s on {k} test samples")

        y_true, y_pred = [], []
        for ex in tqdm(test_examples, desc="Quick test"):
            prompt = build_prompt(ex["instruction"], ex["input"])
            pred = generate_one(model, tok, prompt, args.max_new_tokens)
            y_pred.append(pred)
            y_true.append(ex["output"].strip().lower())

        report = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
            "n": k,
        }
        qm_path = Path(args.quick_metrics_out)
        qm_path.parent.mkdir(parents=True, exist_ok=True)
        qm_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[INFO] Quick baseline F1s → {qm_path}: {report}")

if __name__ == "__main__":
    main()
