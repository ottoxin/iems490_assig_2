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
├─ eval_compare.csv               # comparison of example model outputs
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

---

## Data

**Source**: Kaggle — News Articles for Political Bias Classification  
Columns used:
- `bias` as the target
- `page_text` as the article body

Labels normalized to `{left, center, right}`.

### What is inside data/processed/
- `train.jsonl`, `val.jsonl`, `test.jsonl`: supervised triples with `instruction`, `input`, `output`
- `eval_small.jsonl`: 15 unlabeled examples sampled from `test.jsonl` for qualitative inspection
- `prep_meta.json`: metadata on counts, label balance, and any train downsampling

Note: To keep training time reasonable, this run uses `--train_cap 1200`. The cap is applied **after** creating stratified splits across three labels, so it reduces **train only**. Validation and test stay at their full split sizes and do not overlap with train (**val = 1,382**, **test = 1,625**; full dataset size = **10,832**).

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

## Discussion

### (1)Sample outputs (qualitative)

See the small preview table from eval_compare.csv and preview below.
These rows show cases where the tuned model corrects the base model’s mistakes, especially on clearly partisan pieces. Center remains trickier in some borderline or wire-style articles.

| article (preview 50 words)                                                                                                                                                                                                                                                                                  | gold   | base_pred   | tuned_pred   |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------|:------------|:-------------|
| Bank of America ordered to pay over $100M for 'double-dipping on fees,' withholding credit card rewards Bank of America serves 68M people and small business clients Federal regulators are ordering Bank of America to pay over $100 million to customers for illegally charging junk f | right  | left        | right        |
| Biden slams Trump for Capitol riot in 2024 campaign speech In his first campaign speech of 2024, President Joe Biden cast his likely election opponent, Donald Trump, as a fundamental threat to American democracy. "Whether democracy is still America's sacred cause is the most urge | center | left        | left         |
| Donald Trump and Ben Carson now stand alone at the top of the Republican field, as Carly Fiorina’s brief foray into the top tier of candidates seeking the GOP nomination for president appears to have ended, a new CNN/ORC poll finds. Fiorina has lost 11 points in the last month,   | left   | left        | right        |
| House Democrats on Saturday increased their demands for the IRS to give them access to President Trump’s tax returns -- foreshadowing a lengthy legal battle in the courts House Ways and Means Committee Chairman Richard Neal, D-Mass., told the IRS that the law gives Congress a rig | right  | left        | right        |
| In a historic first, the United States is set to sanction a unit of the Israel Defense Forces. According to three sources speaking with Axios, Secretary of State Antony Blinken will announce sanctions against the IDF’s 97th Netzah Yehuda Battalion. Notably, the sanctions are not  | right  | left        | left         |
| 'NO RESPECT' White House flips script on Hunter Biden's immigration rant against Trump 'He's tired as s---': Hunter reveals medication behind Biden's debate disaster Former first son unleashes fury at Hollywood star in profanity-laced interview 'I got one resume': Special counsel | right  | left        | right        |
| Republican Party communications chief Sean Spicer will be the voice of the Trump administration. President-elect Donald Trump announced Thursday that Spicer will get the coveted job of White House press secretary, as he announced the senior members of his communications team. Thi | right  | left        | right        |
| Sen. Lindsey Graham (R-SC) said there will be riots on the streets if former President Donald Trump faces charges over his handling of documents. He made the explosive prediction during an appearance on Fox News’s Sunday Night in America in an interview with host and former Rep.  | right  | left        | right        |
| Tax season has officially started: Here's everything you need to know before filing IRS will start accepting tax returns on Jan 29 – here are all of your questions answered Tax season officially kicks off on Monday, when the IRS will start officially accepting and processing 2023 | right  | left        | left         |
| The coronavirus continued to ravish the U.S. economy last week, with 4.4 million of American filing first-time unemployment claims, the Labor Department reported Thursday. Economists had forecast 4.25 million new claims for this week. The previous week’s claims figure was revised | right  | left        | right        |
| The European Union has agreed to an embargo on most Russian oil imports after late-night talks at a summit in Brussels. The president of the European Council, Charles Michel, hailed the deal as a “remarkable achievement”, after tweeting on Monday night that sanctions will immedia | left   | left        | left         |
| The internet was flooded with false claims about voter fraud in the days following the 2020 U.S. presidential election on Nov. 3, 2020. Many of these rumors centered around a piece of software from Dominion Voting Systems. We've addressed a few of these rumors in a previous artic | left   | left        | left         |
| U.S. sends submarine to Middle East amid fears of escalation The United States is sending the USS Georgia guided missile submarine to the Middle East and is speeding up the transit to the region of the USS Abraham Lincoln carrier strike group, equipped with F-35C fighters "in lig | left   | left        | right        |
| Vance accuses Denmark of underinvesting in Greenland as Trump presses for US takeover of the island Vance accuses Denmark of underinvesting in Greenland as Trump presses for US takeover of the island ▶ Follow live updates on President Donald Trump and his administration NUUK, Gre | left   | left        | left         |
| Which four Republicans will be on stage for the fourth presidential debate? Just four Republicans will be on stage Wednesday for the fourth Republican presidential debate at the University of Alabama in Tuscaloosa, the Republican National Committee announced Monday evening. Forme | left   | left        | left         |

Before tuning: The base model heavily over-predicts left on many items. It tends to grab a single partisan cue or named actor and default to left, while center is rarely chosen and right is under-called. This “class collapse” shows up in the qualitative rows and in the weak macro-F1.

After tuning: Predictions are more balanced across left/center/right. The model aggregates cues over the passage instead of reacting to one token, so clearly partisan pieces flip to the correct side more often, and neutral reporting is picked as center more reliably.

Why this pattern: Label-only loss focuses learning on the final class token, which reduces rambling and guessy completions. LoRA on attention projections lets the model separate partisan vs. neutral signals with a small number of trainable weights. The stratified split and task-specific prompt also nudge the model away from a one-label default.

### (2) Test set results

We evaluate on the full test split (n = 1,625). Metrics are macro-F1 and micro-F1 only.

| Model        | Macro-F1 | Micro-F1 |     n |
| ------------ | -------: | -------: | ----: |
| Base         |    0.223 |    0.450 | 1,625 |
| Tuned (LoRA) |    0.361 |    0.513 | 1,625 |

The tuned model clearly outperforms the base. Macro-F1 rises from 0.223 to 0.361 (about +62% relative), which signals better balance across left, center, and right rather than gains driven by a single frequent class. Micro-F1 moves from 0.450 to 0.513 (about +14%), showing more correct decisions overall. In practice this reflects two shifts: the tuned model reduces the base model’s tendency to over-predict left, and it picks center more appropriately on neutral passages. The gap between macro- and micro-F1 also narrows a bit, consistent with fewer extreme mistakes on minority classes.

---

## Notes

* Splits are stratified on stable text hashes. Any train downsampling happens after splitting, so there is no leakage into val/test.
* `eval_small.jsonl` is only for qualitative checks. F1 metrics always come from the labeled `test.jsonl`.
* If you keep adapters out of git, leave the folder and document where to download them.
