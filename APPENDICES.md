# Thesis Appendices — paste-ready text

These four appendix sections are written to drop into the thesis document with minimal editing. Square-bracketed `[…]` markers are the only placeholders that need filling in (the GitHub URL and commit hash).

---

## Appendix A — Source Code Repository

The complete source code for this thesis is publicly available on GitHub:

**Repository:** `[https://github.com/<username>/<repo-name>]`
**Branch / commit referenced in this thesis:** `[main @ <short-commit-hash>]`
**License:** `[MIT / academic-use / specify]`

The repository contains:

- The full Python source tree under `src/`, organised into eleven modules (data generation, feature engineering, training, evaluation, statistics, robustness analysis, prediction, visualisation, diagrams, and the orchestrator). Each module is documented in its docstring.
- A pinned `requirements.txt` listing every third-party dependency (`numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`).
- The trained Random Forest model serialised to `models/cheat_detector.pkl` (regenerated on every run).
- Every figure and table referenced in this thesis, regenerated on each run into `reports/` so reviewers can verify reproducibility.
- A README (Appendix D) describing setup and usage.

The implementation is approximately 1,400 source lines of Python across the eleven modules.

---

## Appendix B — Dataset Availability and Reproduction

### Why there is no static dataset

The classifier in this thesis is trained on procedurally generated data rather than a fixed pre-collected corpus. This is a deliberate methodological choice (see Section II.1.4 and III.4): real labelled triggerbot data for Valorant is not publicly available, and synthesising the data from the public Valorant data schema preserves reproducibility in a way that scraped or proprietary datasets cannot.

Each run of the generator produces 5,000 match sessions, yielding a 5,000-row × 25-feature table with a balanced `is_cheating` label (approximately 2,500 triggerbot rows and 2,500 legitimate rows).

### Reproducibility under a fixed seed

The generator is seeded via `seed_everything()` in `validation.py`, which fixes Python's `random` module and NumPy's RNG with `RANDOM_STATE = 42`. Every scikit-learn component is also constructed with the same `random_state`. Given identical source code, the entire dataset is bit-for-bit reproducible. Running the pipeline twice produces identical class counts (`triggerbots = 2,501`, `legits = 2,499`) and identical metrics to four decimal places.

### Reproducing the canonical dataset

The canonical dataset can be regenerated with a single command:

```bash
python src/dataset_export.py
```

This writes `data/Synthetic/canonical_dataset.csv` — a 5,000-row, 28-column file (≈910 KB) containing every feature listed in Section III.3 plus the `puuid`, `rank_name`, and `is_cheating` columns.

### Dataset schema summary

| Group | Columns | Source |
|---|---|---|
| Identity | `puuid` (random per match), `rank_name`, `tier` (1–27) | Generator |
| Player aggregates | `kills`, `deaths`, `assists`, `score` | Rank-relative Gaussian draws |
| Ratios | `kda`, `kd_ratio`, `score_per_kill` | Derived |
| Kill-event signals | `hit_accuracy`, `time_on_target`, `aim_deviation`, `reaction_time`, `shots_fired`, `shots_hit`, `wallbang`, `kill_distance` | Generator |
| Reaction-time flags | `sub_100ms_reaction`, `sub_50ms_reaction`, `precision_score` | Derived |
| Tier-relative anomalies | `tier_relative_{kills,deaths,score}`, `{kills,deaths,score}_z` | Derived against `RANK_STAT_AVERAGES` |
| Label | `is_cheating` (0 = legit, 1 = triggerbot) | Generator |

A full description of how each column is constructed appears in Sections III.3 and III.4 of this thesis.

---

## Appendix C — Detailed Results and Additional Tables

This appendix contains the extended versions of the tables summarised in Chapter IV. All tables are reproducible by running:

```bash
python src/detailed_results.py
```

which writes the full tables to `reports/detailed_results.txt` (≈12.5 KB plain text). The five tables are:

### Table C.1 — Full feature importance ranking

All 25 features in the model with their Random Forest Gini-based importance values, sorted by descending importance. This is the unabridged version of Figure 4 in Chapter IV, including the small-importance and zero-importance features (`wallbang`, `sub_50ms_reaction`, etc.) that confirm the model correctly ignores non-informative inputs.

### Table C.2 — Mann-Whitney U + effect sizes (all 25 features)

The complete per-feature statistical test, extending Table 4 of Chapter IV from the curated 10 features to every feature in the model. Columns: `mean_legit`, `mean_triggerbot`, `cohens_d`, `rank_biserial_r`, `p_value`. Rows sorted by `|rank_biserial_r|` descending. This shows that beyond the headline features, even the redundant tier-relative variants (e.g. `kills_z` vs. `tier_relative_kills`) have identical effect sizes by construction, and confirms that the dead-weight features (`wallbang`, `sub_50ms_reaction`) carry no discriminative signal alongside `aim_deviation`.

### Table C.3 — Per-fold cross-validation breakdown

The fold-by-fold metric values that produce the `mean ± std` figures in Table 2 of Chapter IV, presented separately for each of the three candidate models (Random Forest, Gradient Boosting, Logistic Regression). Each table has five rows (one per fold) and six columns (accuracy, precision, recall, F1, ROC-AUC, average precision). The low cross-fold variance is the direct empirical justification for the small standard deviations reported in Chapter IV.

### Table C.4 — Per-feature descriptive statistics by class

For each of the 25 features, summary statistics (`n`, `mean`, `std`, `min`, `q25`, `median`, `q75`, `max`) computed separately for the legitimate and triggerbot subsets of the canonical dataset. This is the dataset-level analogue of Table 4 and supports any quantitative claim about distributional shape made elsewhere in the thesis (e.g. the tail overlap visible in Figure 5).

### Table C.5 — Candidate model hyperparameters

The exact estimator class, preprocessing pipeline, and key hyperparameter values for each of the three classifiers compared in Chapter IV. Reported for full reproducibility: the same hyperparameters are used in both the cross-validation comparison and the final saved model.

---

## Appendix D — Installation Guide / README

The repository ships with a `README.md` reproducing this guide. Excerpt:

### Requirements

- Python 3.11 or later (developed against 3.11.9)
- The packages pinned in `src/requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`

Tested versions: `Python 3.11.9`, `scikit-learn 1.8.0`, `pandas 3.0.1`, `numpy 2.4.2`, `scipy 1.17.0`, `matplotlib 3.10.8`, `joblib 1.5.3`.

### Installation steps

```bash
# 1. Clone the repository
git clone [<repository-url>] thesis-project
cd thesis-project

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r src/requirements.txt
```

### Running the full pipeline

The complete pipeline — data generation, feature engineering, statistical analysis, multi-model cross-validation, final model training, robustness sweep, demo predictions, and per-player diagnostics — is driven by a single script:

```bash
python src/main.py
```

The total run time is approximately ten to fifteen seconds on a standard desktop machine. The script writes the trained model to `models/cheat_detector.pkl` and every figure and table referenced in this thesis to `reports/`.

### Running individual phases

| Command | Purpose |
|---|---|
| `python src/train.py` | Train + evaluate the classifier without the predict / visualise demo |
| `python src/predict.py` | Load the saved model and classify a fresh batch of players |
| `python src/visualize.py` | Render per-player diagnostic PNGs from the saved model |
| `python src/dataset_export.py` | Export the canonical 5,000-row CSV referenced in Appendix B |
| `python src/detailed_results.py` | Write the extended results tables in Appendix C |
| `python src/diagrams.py` | Re-render the pipeline / architecture / verdict-workflow diagrams |

### Verifying reproducibility

Running `python src/main.py` twice should produce identical console output (class counts, cross-validation metrics, classification report, robustness sweep) because of the fixed-seed reproducibility controls described in Section III.8. If results differ, the seeding routine has not been applied — check that `seed_everything()` is the first call inside `main()`.

### Project structure (top level)

- `src/` — eleven Python modules implementing every phase of the pipeline
- `data/Synthetic/` — produced by `dataset_export.py`; contains the canonical CSV
- `models/` — produced by `train.py` or `main.py`; contains the persisted `.pkl` classifier
- `reports/` — produced by every analysis script; contains all figures and the detailed-results table

A full directory listing and per-module description appears in the `README.md`.
