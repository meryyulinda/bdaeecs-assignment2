# Predicting GitHub User Activity — Assignment 2

**Course:** Big Data Analytics Techniques and Applications, Fall 2026 (NYCU)  
**Student ID:** `315540042`  
**Method:** E3+ Column-Enhanced Hybrid Temporal Heterogeneous Graph Neural Network (HetGNN)

> **Data source:** This project uses **[GH Archive](https://www.gharchive.org/)** as the source of GitHub event records. GH Archive data were processed **only using the course-provided `scripts/preprocess.py`**, which constructs and anonymizes the eight permitted relational tables.

## 1. Task

For each anonymized user in the official `assignment2/submission/test_input.csv`, predict whether that user will perform each of the six activities at least once during **2026-11-22 00:00 through 2026-11-29 00:00, Taiwan time**:

| Output column | Meaning |
| --- | --- |
| `open_pr` | Open a pull request |
| `open_issue` | Open an issue that is **not** a pull request |
| `issue_comment` | Add an issue or pull-request discussion comment |
| `pr_review` | Submit a pull-request review |
| `pr_review_comment` | Add a line/code review comment |
| `push` | Push commits to a repository |

The output must include exactly **one row per official test UserID** and binary `0`/`1` values for these six labels. `UserID` corresponds to `Users.github_user_id`, **not** the internal table key `Users.user_id`.

## 2. Source data and time windows

The executed notebook supplied with this project uses historical GH Archive data downloaded for **1 June–11 November 2026 (UTC)**. The last observed event timestamps in the preprocessed tables fall on 11 November. The source database contains **689,354 rows across eight tables**:

| Preprocessed table | Rows in notebook run |
| --- | ---: |
| `Users` | 14,834 |
| `Repositories` | 48,563 |
| `PullRequests` | 60,262 |
| `Issues` | 38,320 |
| `IssueComments` | 20,329 |
| `PRReviews` | 16,387 |
| `PRReviewComments` | 12,061 |
| `Pushes` | 478,598 |

The final-mode notebook uses Taiwan-local weekly prediction cutoffs:

- **Training labels:** 9 weekly cutoffs, **Week 16 August – Week 11 October 2026** (inclusive, local midnight cutoffs). Each cutoff uses historical features available before that date and labels from the following 7 days. Therefore, the final training label window covers 11–18 October 2026.
- **Threshold-tuning labels:** one later cutoff, **Week 18 October 2026**, with an outcome window through 25 October.
- **Official test prediction cutoff:** **Week 22 November 2026, 00:00 Taiwan** (2026-11-21 16:00 UTC).

The raw timestamp columns are UTC-naive in the preprocessed Parquet files; code defines local Taiwan cutoffs and converts them to UTC-naive timestamps. The late-October portion of the archive is known to have incomplete records; the training-label window ends before that region.

### Permitted input tables only

All features come from the eight tables created by the official preprocessing script. No GitHub API data, additional event categories, or external enrichment data are used for modeling.

## 3. Model and features

The notebook compares:

- **E0 — Copy Last Week:** simple activity-history baseline.
- **E1 — LightGBM:** one binary classifier per activity using engineered historical counts, distinct-repository counts and recency.
- **E2 — Snapshot-based Temporal HetGNN:** eight node types, 20 PK–FK relationship types and their reverse edges (40 directed relation types), two relation-specific heterogeneous GraphSAGE layers, and 32-dimensional user graph embeddings. Neighbor sampling uses **`[16, 8]`**. Temporal information enters E2 through time-filtered snapshots and `log1p(node age)` features; GraphSAGE itself is not a temporal-specific convolution.
- **E3 — Hybrid HetGNN + Temporal MLP:** E2 graph features combined with a 48-dimensional historical activity vector, projected through a `48 → 64 → 32` temporal MLP. Graph and temporal embeddings are concatenated to 64 dimensions and passed through a fusion head.
- **E3+ — Column-Enhanced Hybrid:** E3 plus table-specific numerical features and selected categorical embeddings, combined with the base node projections **by addition** before message passing. Selected categorical fields are review `state`, review-comment `subject_type`, and push `ref_type`.

### Temporal features

For each of six activities, the implementation computes:

- Event counts and unique repository counts over each of the **7-, 30-, and 90-day** lookback windows: **36 features**.
- Days since last activity: **6 raw features**.
- Missing-recency indicators: **6 additional features**.

Thus the temporal MLP receives **48 features**. Counts and recency are transformed with `log1p`; the `StandardScaler` is fitted only on training samples. `activity_time` are activity timestamps (i.e. `created_at` / `submitted_at`) and  `row_time` are row-availability timestamps (i.e. `first_seen_at` where applicable) are distinguished when constructing historical features.

### Data-leakage precautions

At cutoff `t`, the graph includes only nodes available before `t` and only edges whose endpoints exist by that time. The historical feature builder requires both `(row_time < t)` and `(activity_time < t)`. Links known later, such as a PR's `merged_by_id`, are guarded by the merge timestamp. Future event labels are never included as features. As with any event archive, incomplete records and fields backfilled by the provider remain data-quality limitations.

### Training

The executed configuration initializes the graph branch, then trains E2 for **8 epochs**, E3 for **5 epochs**, and E3+ for **3 epochs**. Optimization uses AdamW and weighted binary cross entropy (`BCEWithLogitsLoss`) to mitigate class imbalance. The notebook tunes one threshold per activity by maximizing its F1 on the **18–25 October tuning week**, then applies these fixed thresholds during November inference.

## 4. Reproduce the pipeline (Google Colab / GPU)

The primary implementation is the notebook **`bda_assignment2.ipynb`**. Upload this notebook to the repository (rename the accompanying executed notebook to this name if needed). It contains the pipeline's functions and definitions; there is no separate model-training Python file required.

**Environment used in the saved run:** Python 3.13.16, PyTorch 2.14.1+cu130, PyTorch Geometric 2.8.1, `pyg_lib` 0.9.0+pt214cu130, Linux and an available CUDA GPU. Other versions may work but have not been verified with these saved outputs. Also needed: `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `joblib`, `graphviz` and Jupyter/Colab. The notebook's first cells install the relevant graph dependencies; a matching `pyg_lib` wheel must be selected for the installed Torch/CUDA build.

### Step A — Obtain the official assignment code

From a fresh Colab working directory (the notebook assumes `/content`):

```bash
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

Use the **current** `assignment2/submission/test_input.csv`; it contains 4,465 test users in the executed run. The official repository also supplies `scripts/preprocess.py`, `scripts/evaluate.py`, `cohort.txt`, and `SCHEMA.md`.

### Step B — Download the allowed raw archive and preprocess it

Put GH Archive hourly `.json.gz` files for the intended historical coverage into a directory such as `/content/RAW_DATA_FOLDER`. The notebook demonstrates hourly downloads for June through November 11, 2026. The archive can be large; reuse the same downloaded hourly files if exact reproducibility is desired. **Do not mix Parquet outputs from different preprocessing runs**, because internal row IDs may be regenerated.

Recreate the eight tables using the required course script:

```bash
python /content/assignment2/scripts/preprocess.py \
  --raw /content/RAW_DATA_FOLDER \
  --out /content/assignment2/data/
```

The resulting files must exist under `/content/assignment2/data/db/` (one Parquet file per table). The notebook also contains a commented option for extracting an already-preprocessed archive; for a fresh permitted reproduction, the official preprocessing command above is the canonical route.

### Step C — Run the notebook

1. Open **`bda_assignment2.ipynb`** in Google Colab with a **GPU runtime**. Install the dependencies in the first notebook cells and restart the runtime if needed.
2. Ensure the working directory is `/content` and that the eight Parquet tables are in `assignment2/data/db/`.
3. In **Step 3**, set `RUN_MODE = "final"`.
4. Run all code cells **in order from Step 1 through Step 11**. Rerunning from Step 3 is sufficient only if the eight loaded DataFrames and dependent helper definitions are already present in the current kernel.
5. The notebook reconstructs the nine-week training samples, fits the train-only scaler, trains E2 → E3 → E3+, selects October thresholds, constructs the official test features, runs E3+ inference, and writes `assignment2/output_315540042.csv`.
6. The `model_artifacts/` subdirectory, created by Step 12, contains trained weights, the scaler and configuration/threshold data. Rerunning Step 12 also generates the question diagrams.

The training and inference stages require substantial memory and compute. The notebook's reproducibility assumes the same code, source data, preprocessing script and random-seed configuration; stochastic GPU operations can produce small score differences across environments.

### Step D — Validate the submission file

From `/content`, run:

```bash
python assignment2/scripts/evaluate.py \
  --pred assignment2/output_315540042.csv \
  --input assignment2/submission/test_input.csv
```

**Result in the supplied executed notebook:** `Format check: no errors (4465 users)`.

This check verifies formatting and test-user coverage. The official November test labels were not available when this notebook was executed, so **no November F1 can be reported yet**. For the E3 submission, the file must be named `output_315540042.csv`.

> **Before the deadline:** The supplied run only incorporates historical records through November 11. If additional permitted GH Archive files are downloaded before the November 21, 2026 23:55 Taiwan submission deadline, rerun official preprocessing, reload all tables, regenerate the features/graph cache, retrain the pipeline, calibrate thresholds using appropriately held-out labeled history, and regenerate/validate the output. Do **not** use future data that was unavailable by the deadline.

## 5. Saved experiment results

On the **18–25 October 2026 threshold-tuning week** (7,745 observable cohort users), the saved notebook reports the following combined score, `(macro-F1 + micro-F1) / 2`:

| Experiment | Macro-F1 | Micro-F1 | Combined score |
| --- | ---: | ---: | ---: |
| E0 Copy Last Week | 0.3821 | 0.4939 | 0.4380 |
| E1 LightGBM | 0.3594 | 0.4753 | 0.4174 |
| E2 snapshot-based Temporal HetGNN | 0.3598 | 0.4591 | 0.4095 |
| E3 Hybrid HetGNN + Temporal | 0.4402 | 0.5378 | 0.4890 |
| E3+ with threshold 0.5 | 0.4343 | 0.5359 | 0.4851 |
| **E3+ with tuned thresholds** | **0.4545** | **0.5523** | **0.5034** |

**Important:** The last row is evaluated on the **same October week used to choose thresholds**. It is useful as calibration diagnostics but is **optimistically biased** as an estimate of unseen future performance. It should **not** be described as an independently held-out benchmark or an official November score.

Selected October thresholds:

| `open_pr` | `open_issue` | `issue_comment` | `pr_review` | `pr_review_comment` | `push` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 | 0.57 | 0.60 | 0.55 | 0.48 | 0.53 |

## 6. Final output diagnostics in the saved notebook

The executed candidate prediction file has **4,465 rows**, **all UserIDs matched**, **zero absent-table cold starts**, and passed the official format validation. Predicted-positive rates are:

| Activity | Predicted positive rate |
| --- | ---: |
| `open_pr` | 0.90% |
| `open_issue` | 1.72% |
| `issue_comment` | 0.38% |
| `pr_review` | 1.61% |
| `pr_review_comment` | 1.41% |
| `push` | 6.43% |

These are **model outputs**, not observed official test outcomes. Their magnitude may be affected by the late-year archive-coverage issue and the fact that the run's source history ends November 11.

## References

- [GH Archive — data source](https://www.gharchive.org/)
- [NYCU BigData2026 Assignment 2 — course repository](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2)

