# src/prep_bias_data.py
import argparse, json, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

VALID = {"left", "center", "right"}

def normalize_label(x):
    if pd.isna(x): return None
    s = str(x).strip().lower()
    if s in {"-1","0","1","2"}:
        v = int(s)
        if v in {-1,0,1}: return {-1:"left",0:"center",1:"right"}[v]
        if v in {0,1,2}:  return {0:"left",1:"center",2:"right"}[v]
    if s in {"left","leaning-left"}: return "left"
    if s in {"centre","center"}:    return "center"
    if s in {"right","leaning-right"}: return "right"
    return None

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stratified_fraction(df, label_col, frac, seed):
    # sample same fraction within each label
    return (df.groupby(label_col, group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=seed))
              .reset_index(drop=True))

def stratified_cap(df, label_col, cap, seed):
    # split cap proportionally by label counts (at least 1 per class if present)
    n_total = len(df)
    if cap >= n_total: return df
    counts = df[label_col].value_counts()
    # initial quotas
    quotas = (counts * (cap / n_total)).astype(int)
    # ensure no zero for present classes
    quotas = quotas.where(quotas > 0, 1)
    # adjust to exact cap
    diff = cap - int(quotas.sum())
    # assign remainder to classes with largest fractional parts
    if diff != 0:
        fracs = (counts * (cap / n_total)) - (counts * (cap / n_total)).astype(int)
        order = fracs.sort_values(ascending=False).index.tolist()
        i = 0
        while diff != 0 and i < len(order):
            cls = order[i]
            step = 1 if diff > 0 else -1
            # don’t drop below 1
            if not (quotas[cls] + step < 1):
                quotas[cls] += step
                diff -= step
            i = (i + 1) % len(order)
    parts = []
    for cls, n in quotas.items():
        g = df[df[label_col] == cls]
        n = min(n, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)

def per_label_cap(df, label_col, per_cap, seed):
    parts = []
    for cls, g in df.groupby(label_col):
        n = min(per_cap, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path to bias_clean.csv (or equivalent).")
    ap.add_argument("--output_dir", required=True, help="Output folder for JSONL files.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_prompts_n", type=int, default=15, help="Size of small qualitative eval set.")
    # NEW: optional downsampling controls (train split only)
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="If <1, keep this fraction of TRAIN per class (stratified).")
    ap.add_argument("--train_cap", type=int, default=None,
                    help="If set, cap TRAIN to this many total examples (stratified).")
    ap.add_argument("--train_cap_per_label", type=int, default=None,
                    help="If set, cap TRAIN to this many per label (balanced).")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    if "bias" not in df.columns or "page_text" not in df.columns:
        raise ValueError("Expected columns: 'bias' and 'page_text' per dataset docs.")

    # Normalize labels and basic text cleanup (no length filter)
    df["_label"] = df["bias"].map(normalize_label)
    df["_text"]  = df["page_text"].astype(str).str.replace(r"\s+"," ", regex=True).str.strip()

    before = len(df)
    df = df[df["_label"].isin(VALID)]
    after = len(df)

    # Stable ID per article
    df["_id"] = pd.util.hash_pandas_object(df["_text"], index=False).astype("int64").astype(str)

    # Stratified split by label at ID level
    ids = df[["_id","_label"]].drop_duplicates()
    train_ids, test_ids = train_test_split(ids, test_size=0.15, stratify=ids["_label"], random_state=args.seed)
    train_ids, val_ids  = train_test_split(train_ids, test_size=0.15, stratify=train_ids["_label"], random_state=args.seed)

    def subframe(id_df): return df.merge(id_df[["_id"]], on="_id", how="inner").copy()
    train_df, val_df, test_df = subframe(train_ids), subframe(val_ids), subframe(test_ids)

    # Record pre-downsample sizes & label counts
    pre_sizes = {
        "train": len(train_df), "val": len(val_df), "test": len(test_df),
        "train_label_counts": train_df["_label"].value_counts().to_dict(),
        "val_label_counts":   val_df["_label"].value_counts().to_dict(),
        "test_label_counts":  test_df["_label"].value_counts().to_dict(),
    }

    # --- OPTIONAL: downsample TRAIN only (choose ONE scheme; priority: per_label_cap > frac > cap) ---
    ds_info = {"applied": False, "scheme": None, "params": {}}
    if args.train_cap_per_label:
        train_df = per_label_cap(train_df, "_label", args.train_cap_per_label, args.seed)
        ds_info = {"applied": True, "scheme": "per_label_cap", "params": {"train_cap_per_label": args.train_cap_per_label}}
    elif args.train_frac and args.train_frac < 1.0:
        train_df = stratified_fraction(train_df, "_label", args.train_frac, args.seed)
        ds_info = {"applied": True, "scheme": "train_frac", "params": {"train_frac": args.train_frac}}
    elif args.train_cap:
        train_df = stratified_cap(train_df, "_label", args.train_cap, args.seed)
        ds_info = {"applied": True, "scheme": "train_cap", "params": {"train_cap": args.train_cap}}

    # Small held-out qualitative eval (unlabeled), sampled from test
    eval_small = test_df.sample(n=min(args.eval_prompts_n, len(test_df)), random_state=args.seed)

    instr = "Classify the political leaning of this news article as one of {left|center|right}. Respond with a single word: left, center, or right."
    def rows_from(df_):
        return [{"instruction": instr, "input": t, "output": y} for t, y in zip(df_["_text"], df_["_label"])]

    write_jsonl(out/"train.jsonl", rows_from(train_df))
    write_jsonl(out/"val.jsonl",   rows_from(val_df))
    write_jsonl(out/"test.jsonl",  rows_from(test_df))
    write_jsonl(out/"eval_small.jsonl", [{"instruction": instr, "input": t} for t in eval_small["_text"].tolist()])

    # Meta
    post_sizes = {
        "train": len(train_df), "val": len(val_df), "test": len(test_df),
        "train_label_counts": train_df["_label"].value_counts().to_dict(),
        "val_label_counts":   val_df["_label"].value_counts().to_dict(),
        "test_label_counts":  test_df["_label"].value_counts().to_dict(),
    }
    meta = {
        "n_before_filter": int(before),
        "n_after_filter": int(after),
        "stratified": True,
        "pre_split": pre_sizes,
        "post_split": post_sizes,
        "downsample": ds_info,
        "columns_used": {"label": "bias", "text": "page_text"}
    }
    (out/"prep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
