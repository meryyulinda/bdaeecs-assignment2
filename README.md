# Predicting GitHub User Activity with Relational Deep Learning

**Big Data Analytics Techniques and Applications — Assignment 2 | NYCU, Fall 2026**

**Notebook:** `bda_assignment2_20261118.ipynb` (a `.py` export of the same notebook is included)

## Overview

For each test user, the task is to predict whether the user performs each of six activities during the week
2026/11/22 00:00 – 11/29 00:00 (Taiwan time):

| Column | The user, during the week … |
| --- | --- |
| `open_pr` | opens a pull request |
| `open_issue` | opens an issue (not a pull request) |
| `issue_comment` | comments in an issue or pull request conversation |
| `pr_review` | submits a review of a pull request |
| `pr_review_comment` | comments on specific lines of a pull request's code diff |
| `push` | pushes commits to a branch or tag |

The eight relational tables are turned into a temporal heterogeneous graph (one node per row, one edge type per
foreign key plus its reverse). The selected model combines a heterogeneous GraphSAGE branch with a user-activity
history branch and uses per-activity tuned decision thresholds.

## Data Source

The data used in this project is sourced from [GH Archive](https://www.gharchive.org/), a publicly available archive
of GitHub events. The raw hourly files were converted into the eight anonymized tables with the course-provided
preprocessing script (`scripts/preprocess.py` from the
[course repository](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2)). No other data was used.
This project is for educational use within the course.

**Raw data period used:** GH Archive hour files `2026-06-01-0` to `2026-11-11-23` (UTC), 3,936 files.

## Repository contents

| Path | Content |
| --- | --- |
| `bda_assignment2_20261118.ipynb` | Full pipeline: data loading, features, graph snapshots, models E0–E3+, benchmark, final retraining, submission, report figures |
| `bda_assignment2_20261118.py` | Plain-Python export of the notebook |
| `assignment2/output_315540042.csv` | Submitted predictions for the 4,465 test users |
| `assignment2/` | Copy of the course repository (`scripts/preprocess.py`, `scripts/evaluate.py`, `scripts/cohort.txt`, `submission/test_input.csv`, `README.md`, `SCHEMA.md`) |
| `assignment2/benchmark_runs/` | Per-week benchmark scores (`*.csv`), run configs (`*_config.joblib`) and E3+ weights (`*_e3plus.pt`) for each E3+ variant |
| `assignment2/final_model_artifacts/` | Final model: `final_weights.pt`, `final_temporal_scaler.joblib`, `final_config.joblib` (variant, thresholds, feature columns, cutoffs), calibration results (`*.csv`) |
| `report_figures/` | Q1–Q4 figures used in the slides (`.png` and `.svg`) |

The preprocessed tables (`assignment2/data/`) are not committed (see `.gitignore`); rebuild them with the steps below.

## Models

| Model | Approach |
| --- | --- |
| **E0** | Copy Last Week: 1 if the user did the activity in the 7 days before the cutoff |
| **E1** | LightGBM on user-history features (threshold 0.5) |
| **E2** | Heterogeneous GraphSAGE on a temporal graph snapshot with base node features (node age + basic attributes) |
| **E3** | E2 graph branch + temporal MLP on 48 user-history features (7/30/90-day counts, unique repositories, recency) |
| **E3+** | E3 + extra numerical node encodings in the graph branch; three variants compared |

E3+ variants: **cyclical** (weekday/hour sin–cos on every node), **cyclical + repository activity** (12 per-repository
activity counts), **cyclical + workflow flags** (`is_cross_repo`, `is_draft` on PRs; `has_assignee` on issues).
The selected model is **E3+ cyclical + workflow flags with tuned per-activity thresholds**.

**Leakage rules.** At cutoff `t`, a row is used only if its row time (`first_seen_at`, or `created_at` for comments
and pushes) is before `t`. History features place each row at its activity time (`created_at` / `submitted_at`),
which must also be before `t`. Later-known columns (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`,
`dismissed_at`) are used only when their value is before `t`.

## Time splits

All cutoffs are 00:00 Taiwan time (converted to UTC for the tables). Each cutoff predicts the following 7 days.

| Stage | Cutoffs |
| --- | --- |
| Benchmark — train | Aug 16, 23, 30, Sep 6, 13 |
| Benchmark — threshold tuning | Sep 20 |
| Benchmark — held-out evaluation | Sep 27, Oct 4, Oct 11, Oct 18 |
| Final — retrain | Aug 16 – Oct 11 (9 weekly cutoffs) |
| Final — threshold calibration | Oct 18 |
| Final — test (submission) | Nov 22 |

## How to reproduce

**Environment:** Google Colab with a GPU runtime. Step 1 of the notebook installs `torch-geometric`, `pyg_lib`
(wheel for `torch-2.14.0+cu130`) and `graphviz`; restart the runtime after the installs, as noted in the notebook.
Random seeds are fixed (`torch.manual_seed(42)`).

1. **Get the course scripts** (the notebook does this in Step 1):

   ```bash
   git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
   ```

2. **Download the raw GH Archive hours** (2026-06-01-0 to 2026-11-11-23 UTC):

   ```bash
   wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-06-{01..30}-{0..23}.json.gz
   wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-07-{01..31}-{0..23}.json.gz
   wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-08-{01..31}-{0..23}.json.gz
   wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-09-{01..30}-{0..23}.json.gz
   wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-10-{01..31}-{0..23}.json.gz
   wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-11-{01..11}-{0..23}.json.gz
   ```

3. **Build the eight tables** with the course script (output read by the notebook from `assignment2/data/db/`):

   ```bash
   python assignment2/scripts/preprocess.py --raw RAW_DATA_FOLDER --out assignment2/data/
   ```

4. **Run the notebook in order.** Keep `RUN_MODE = "benchmark"` (do not change it midway).
   - Steps 1–10: data loading, labels, features, E0, E1, graph snapshots, E2, E3.
   - Steps 11–12: E3+ and the four-week benchmark. To reproduce all E3+ ablations, set `E3PLUS_VARIANT` to
     `"cyclical_only"`, `"cyclical_repo_activity"` and `"cyclical_workflow_flags"` in turn and rerun Steps 11–12
     for each; results are saved to `assignment2/benchmark_runs/`.
   - Step 13: set `FINAL_MODEL_CHOICE = "E3PLUS_TUNED:cyclical_workflow_flags"`. This retrains on Aug 16 – Oct 11,
     calibrates thresholds on Oct 18, predicts the Nov 22 cutoff, writes `assignment2/output_315540042.csv`, runs
     the official format check and saves the final artifacts to `assignment2/final_model_artifacts/`.
   - Step 14: generates the report figures in `report_figures/`.

5. **Check the submission format** (from the repository root, for the committed copy):

   ```bash
   python assignment2/scripts/evaluate.py --pred output_315540042.csv --input assignment2/submission/test_input.csv
   ```

   Expected: `Format check: no errors (4465 users)`.

GPU training is not bit-for-bit deterministic (random neighbour sampling and non-deterministic GPU kernels), so a
retrain can differ slightly from the committed output. The exact weights, scaler and thresholds behind the
submitted file are saved in `assignment2/final_model_artifacts/`.

## Results

Score = (macro-F1 + micro-F1) / 2, computed with `scores()` from the official `scripts/evaluate.py`.

**Held-out benchmark** (mean of the four weekly scores, Sep 27 – Oct 18):

| Model | Macro-F1 | Micro-F1 | Score |
| --- | ---: | ---: | ---: |
| E0 — Copy Last Week | 0.4123 | 0.5275 | 0.4699 |
| E1 — LightGBM | 0.3979 | 0.4171 | 0.4075 |
| E2 — Snapshot HetGNN | 0.3429 | 0.4199 | 0.3814 |
| E3 — Hybrid HetGNN (tuned) | 0.4487 | 0.5715 | 0.5101 |
| E3+ — Cyclical (tuned) | 0.4492 | 0.5630 | 0.5061 |
| E3+ — Cyclical + Repo Activity (tuned) | 0.4538 | 0.5661 | 0.5099 |
| **E3+ — Cyclical + Workflow Flags (tuned)** | **0.4528** | **0.5679** | **0.5104** |

**Course reference week only** (Sep 27 – Oct 4, the week of the reference scores in the course README):

| Model | Macro-F1 | Micro-F1 | Score |
| --- | ---: | ---: | ---: |
| Course reference — Copy last week | 0.4301 | 0.5620 | 0.4961 |
| Course reference — GNN with time encoding | 0.5090 | 0.5874 | 0.5482 |
| E0 — Copy Last Week | 0.4235 | 0.5490 | 0.4862 |
| E3 — Hybrid HetGNN (tuned) | 0.4636 | 0.5975 | 0.5305 |
| E3+ — Cyclical + Workflow Flags (tuned) | 0.4807 | 0.6013 | 0.5410 |

The E3+ gain over E3 is small (single seed), and threshold tuning accounts for most of the improvement over 0.5.

## Limitations

- The raw data used for the submitted file ends on 2026-11-11 23:59 UTC, ten days before the Nov 22 prediction
  cutoff, so the most recent history windows are empty at test time and the predicted positive rates are low
  (e.g. push 7.28%).
- Official test labels are withheld, so no test-week F1 is reported.
