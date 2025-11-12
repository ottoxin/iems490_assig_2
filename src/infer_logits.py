# src/infer_logits.py
import argparse, json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABELS = ["left","center","right"]

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

@torch.inference_mode()
def score_label_seq(model, ids, next_ids):
    logp = 0.0
    for tid in next_ids:
        out = model(input_ids=ids)
        lp = torch.log_softmax(out.logits[:, -1, :], dim=-1)[0, tid].item()
        logp += lp
        ids = torch.cat([ids, torch.tensor([[tid]], device=ids.device)], dim=1)
    # normalize by sequence length to avoid penalizing longer tokenizations
    return logp / max(len(next_ids), 1)

def predict_one(model, tok, instr, text, device, max_len=1024):
    # Keep the start of the article; prompt ends with "Answer:"
    tok.truncation_side = "left"
    prompt = f"{instr}\n\nText:\n{text}\n\nAnswer:"
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    ids = enc["input_ids"].to(device)

    best, best_lp = None, -1e18
    for lbl in LABELS:
        lab_ids = tok(" " + lbl, add_special_tokens=False).input_ids
        lp = score_label_seq(model, ids, lab_ids)
        if lp > best_lp:
            best, best_lp = lbl, lp
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter_dir", default=None, help="LoRA adapter path; if omitted, base only")
    ap.add_argument("--data_jsonl", required=True)   # labeled (test/val)
    ap.add_argument("--out_preds", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True
    ).to(device).eval()
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir).to(device).eval()

    data = list(load_jsonl(args.data_jsonl))
    y_true, y_pred = [], []
    outp = Path(args.out_preds); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as pf:
        for ex in tqdm(data, desc="Scoring"):
            pred = predict_one(model, tok, ex["instruction"], ex["input"], device, args.max_len)
            y_pred.append(pred)
            y_true.append(ex["output"].strip().lower())
            pf.write(json.dumps({"gold": ex["output"], "pred": pred}, ensure_ascii=False) + "\n")

    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(y_true, y_pred, output_dict=True),
        "n": len(y_true),
    }
    mj = Path(args.metrics_json); mj.parent.mkdir(parents=True, exist_ok=True)
    mj.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
