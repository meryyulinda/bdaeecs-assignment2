# Predicting GitHub User Activity

**Big Data Analytics — Assignment 2 | NYCU, Fall 2026**  
**Notebook:** `bda_assignment2_20261118.ipynb`

## Overview

This project predicts whether a GitHub user will perform each of six activities in the next seven days: **open a pull request, open an issue, comment on an issue, submit a PR review, comment on a PR review, or push commits**.

The data comes from [GH Archive](https://www.gharchive.org/) and is converted into eight relational tables using the [course-provided preprocessing script](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2). Features and graph snapshots include only information available before each prediction cutoff.

## Models

Five approaches are compared:

| Model | Approach |
| --- | --- |
| **E0** | Copy Last Week baseline |
| **E1** | LightGBM with historical activity features |
| **E2** | Heterogeneous GraphSAGE using relational graph snapshots and base node features |
| **E3** | E2 combined with a temporal MLP for 7-, 30-, and 90-day user activity history |
| **E3+** | E3 with additional node-level feature encoding |

For E3+, we compare **cyclical weekday/hour**, **cyclical time + repository activity counts**, and **cyclical time + workflow flags**. The selected model is **E3+ Cyclical + Workflow Flags**, using `is_cross_repo`, `is_draft`, and `has_assignee` in addition to cyclical time features. Its predictions use activity-specific tuned thresholds.

## Results

Models were trained on weekly cutoffs from **August 16–September 13, 2026**, calibrated on **September 20**, and evaluated on four later weekly cutoffs (**September 27–October 18**). The score is the average of macro-F1 and micro-F1.

| Model | Macro-F1 | Micro-F1 | Combined score |
| --- | ---: | ---: | ---: |
| E0 — Copy Last Week | 0.4123 | 0.5275 | 0.4699 |
| E1 — LightGBM | 0.3979 | 0.4171 | 0.4075 |
| E2 — Snapshot HetGNN | 0.3429 | 0.4199 | 0.3814 |
| E3 — Hybrid HetGNN (tuned) | 0.4487 | 0.5715 | 0.5101 |
| **E3+ — Cyclical + Workflow Flags (tuned)** | **0.4528** | **0.5679** | **0.5104** |

For the E3+ ablation, the combined scores after tuning were **0.5061** (cyclical only), **0.5099** (cyclical + repository activity), and **0.5104** (cyclical + workflow flags). The improvement over E3 is small; threshold tuning is important to the reported performance.

After model selection, the final E3+ pipeline was retrained using weekly cutoffs from **August 16–October 11** and recalibrated on **October 18** for the official **November 22** prediction cutoff.

## Reproduction

1. Open `bda_assignment2_20261118.ipynb` in **Google Colab with GPU** and follow its dependency setup cells.
2. Clone the [assignment repository](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2), download the required GH Archive files, and preprocess them with the course script:

   ```bash
   git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
   python assignment2/scripts/preprocess.py --raw RAW_DATA_FOLDER --out assignment2/data/
   ```

3. Keep `RUN_MODE = "benchmark"`. Run the notebook in order. To reproduce all E3+ ablations, change `E3PLUS_VARIANT` and rerun Steps 11–12 for each variant.
4. In Step 13, use `FINAL_MODEL_CHOICE = "E3PLUS_TUNED:cyclical_workflow_flags"` for final retraining and submission generation. Step 14 generates report figures.
5. Validate the submission file:

   ```bash
   python assignment2/scripts/evaluate.py \
     --pred assignment2/output_315540042.csv \
     --input assignment2/submission/test_input.csv
   ```

The saved submission has **4,465 users** and passed the format check. **Official November F1 is not available** because test labels are withheld. The saved run's historical data ends on **November 11**, before the November 22 prediction cutoff, which is a limitation of that submission candidate.
