# IEMS Assignment 2: Political Bias Classification with LoRA on Qwen 0.6B

This repo fine-tunes **Qwen/Qwen3-0.6B** with **LoRA** to classify news articles into `{left, center, right}` using the Kaggle dataset [*News Articles for Political Bias Classification*](https://www.kaggle.com/datasets/gandpablo/news-articles-for-political-bias-classification).

It includes:
- Data prep from the original CSV into train/val/test JSONL
- LoRA fine-tuning with label-only loss
- Free-generation inference for base and tuned models with robust parsing and F1 metrics
- A short unit test for quick grading
- Reproducible commands and configs

---

## Repo layout

```

iems490-assig-2/
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ configs/
│  ├─ lora_qwen3_0p6b.yaml
│  └─ lora_qwen3_0p6b_unit.yaml
├─ data/
│  ├─ bias_clean.csv              # original full dataset
│  └─ processed/                  # this run: TRAIN capped to 1,200 (stratified after split); VAL = 1,382 and TEST = 1,625 (each ~15% of 10,832)
│     ├─ train.jsonl
│     ├─ val.jsonl
│     ├─ test.jsonl
│     ├─ eval_small.jsonl         # 15 unlabeled samples for qualitative checks
│     └─ prep_meta.json           # sizes, label counts, split/downsample details
├─ adapters/
│  └─ .gitkeep                    # LoRA adapter output (starts empty)
├─ outputs/
│  └─ .gitkeep                    # predictions/metrics (starts empty)
└─ src/
├─ prep_bias_data.py
├─ train_lora_qwen.py
├─ infer_base.py
├─ infer_tuned.py
└─ unit_test_lora_qwen.py

````

### What is inside `data/processed/`
- `train.jsonl`, `val.jsonl`, `test.jsonl`: supervised triples with `instruction`, `input`, `output`
- `eval_small.jsonl`: 15 unlabeled examples sampled from `test.jsonl` for qualitative inspection
- `prep_meta.json`: metadata on counts, label balance, and any train downsampling

**Important**: To keep training time reasonable, this run uses `--train_cap 1200`. The cap is applied **after** creating stratified splits across three labels, so it reduces **train only**. Validation and test stay at their full split sizes and do not overlap with train (**val = 1,382**, **test = 1,625**; full dataset size = **10,832**).

---

## Data

**Source**: Kaggle — News Articles for Political Bias Classification  
Columns used:
- `bias` as the target
- `page_text` as the article body

Labels normalized to `{left, center, right}`. The repo assumes `data/bias_clean.csv` is present.

---

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# optional on shared servers
export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$HF_HOME"
````

GPU is recommended for training. Inference can run on CPU but will be slow.

---

## 1) Preprocess

This run caps **train** at 1,200 examples (stratified), keeps full val/test:

```bash
python src/prep_bias_data.py \
  --input_csv data/bias_clean.csv \
  --output_dir data/processed \
  --seed 42 \
  --train_cap 1200
```

Example console output:

```json
{
  "n_before_filter": 10832,
  "n_after_filter": 10832,
  "stratified": true,
  "pre_split": {"train": 7825, "val": 1382, "test": 1625, "...": "..."},
  "post_split": {"train": 1200, "val": 1382, "test": 1625, "...": "..."},
  "downsample": {"applied": true, "scheme": "train_cap", "params": {"train_cap": 1200}},
  "columns_used": {"label": "bias", "text": "page_text"}
}
```

Files written to `data/processed/`:

```
train.jsonl  val.jsonl  test.jsonl  eval_small.jsonl  prep_meta.json
```

---

## 2) Train LoRA on Qwen 0.6B

Config (edit `configs/lora_qwen3_0p6b.yaml` if needed):

```yaml
seed: 42
model_id: Qwen/Qwen3-0.6B

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]

# Training
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 5
weight_decay: 0.0
warmup_ratio: 0.03
logging_steps: 20
eval_strategy: "steps"
eval_steps: 200
save_steps: 200
early_stopping_patience: 2
lr_scheduler_type: linear
bf16: false
fp16: true

max_length: 512

data_dir: data/processed
output_dir: adapters/qwen3_0p6b_lora
```

Run:

```bash
python src/train_lora_qwen.py \
  --config configs/lora_qwen3_0p6b.yaml \
  --data_dir data/processed \
  --out_dir adapters/qwen3_0p6b_lora
```

Example console lines:

```
trainable params: 4,587,520 || all params: 600,637,440 || trainable%: 0.7638
{'loss': 2.91, 'grad_norm': 0.38, 'learning_rate': 9.20e-05, 'epoch': 0.33}
...
{'eval_loss': 2.78, 'eval_runtime': 250.1, 'epoch': 0.41}
...
{'train_runtime': 2197.65, 'train_steps_per_second': 0.341, 'train_loss': 2.98, 'epoch': 5.0}
```

Notes:

* **Label-only loss**: loss applied only to label tokens appended after `Answer:`
* `eval_small.jsonl` is for quick qualitative checks. Early stopping uses `eval_loss` on the validation set

---

## 3) Inference and metrics (macro-F1 and micro-F1)

Create the outputs folder if empty:

```bash
mkdir -p outputs
```

### Base model

```bash
python src/infer_base.py \
  --model_id Qwen/Qwen3-0.6B \
  --eval_jsonl data/processed/eval_small.jsonl \
  --out_jsonl outputs/base_eval.jsonl \
  --test_jsonl data/processed/test.jsonl \
  --quick_metrics_k 999999 \
  --quick_metrics_out outputs/base_metrics_test.json
```

Example `outputs/base_metrics_test.json`:

```json
{ "macro_f1": 0.23, "micro_f1": 0.34, "n": 1625 }
```

Example `outputs/base_eval.jsonl`:

```json
{"instruction":"Classify ...","input":"<truncated article...>","base_pred":"left"}
{"instruction":"Classify ...","input":"<truncated article...>","base_pred":"right"}
```

### Tuned model

```bash
python src/infer_tuned.py \
  --model_id Qwen/Qwen3-0.6B \
  --adapter_dir adapters/qwen3_0p6b_lora \
  --eval_jsonl data/processed/eval_small.jsonl \
  --out_jsonl outputs/tuned_eval.jsonl \
  --test_jsonl data/processed/test.jsonl \
  --preds_jsonl outputs/tuned_test_preds.jsonl \
  --metrics_json outputs/tuned_metrics_test.json
```

Example `outputs/tuned_metrics_test.json`:

```json
{ "macro_f1": 0.49, "micro_f1": 0.59, "n": 1625 }
```

Example `outputs/tuned_test_preds.jsonl`:

```json
{"pred":"right","gold":"right"}
{"pred":"left","gold":"left"}
{"pred":"center","gold":"left"}
```

---

## 4) (Optional) Join eval previews with gold labels

```bash
python src/join_eval_to_csv.py \
  --base_eval outputs/base_eval.jsonl \
  --tuned_eval outputs/tuned_eval.jsonl \
  --test_jsonl data/processed/test.jsonl \
  --out_csv outputs/eval_compare.csv \
  --join_words 100 \
  --preview_words 50
```

This matches rows using the first 100 normalized words and writes a readable CSV with a 50-word preview plus `gold`, `base_pred`, and `tuned_pred`.

---

## 5) Unit test (quick grading)

Runs a tiny end-to-end check in under 10 minutes.

```bash
python src/unit_test_lora_qwen.py \
  --config configs/lora_qwen3_0p6b_unit.yaml \
  --data_dir data/processed \
  --out_dir adapters/qwen3_0p6b_lora_unit \
  --n_train 64 --n_val 32 --n_test 64
```

What it does:

* Takes small slices from the capped splits
* Runs a short LoRA training
* Runs tuned inference on the tiny test slice
* Prints and saves macro-F1 and micro-F1

---

## Docker

Build:

```bash
docker build -t iems490-assig-2 .
```

Run:

```bash
docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  iems490-assig-2 bash
```

Inside the container, follow the same commands as above.

---

## Outputs folder

`outputs/` starts empty with `.gitkeep`. The scripts will populate it as you run inference:

* `base_eval.jsonl`, `base_metrics_test.json`
* `tuned_eval.jsonl`, `tuned_test_preds.jsonl`, `tuned_metrics_test.json`
* `eval_compare.csv` from the join step

---

## Discussion (to fill)

Add a short analysis here:

* Compare macro-F1 and micro-F1 of base vs tuned
* Identify which labels improved the most
* Show a few qualitative examples from `eval_small.jsonl` where tuned fixed base errors

---

## Notes

* Splits are stratified on stable text hashes. Any train downsampling happens after splitting, so there is no leakage into val/test.
* `eval_small.jsonl` is only for qualitative checks. F1 metrics always come from the labeled `test.jsonl`.
* If you keep adapters out of git, leave the folder and document where to download them.

```
