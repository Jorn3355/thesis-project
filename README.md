# Triggerbot Detection on Synthetic Valorant Data

A bachelor-thesis project that trains a supervised classifier to distinguish legitimate Valorant players from triggerbot cheats using engineered behavioural features extracted from synthetic kill-event data.

The positive class is specifically **triggerbot** (manual aim + automated firing the instant the crosshair touches a target). The project frames cheat detection as supervised binary classification over 25 engineered per-event features, evaluated under 5-fold stratified cross-validation with multi-model comparison and a robustness sweep.

---

## Requirements

- Python **3.11+** (developed against 3.11.9)
- Packages pinned in `src/requirements.txt`:
  - `numpy >= 1.21`
  - `pandas >= 1.3`
  - `scikit-learn >= 1.3`
  - `scipy >= 1.10`
  - `matplotlib` (installed transitively or via pip)
  - `joblib >= 1.3`

Tested versions: Python 3.11.9, scikit-learn 1.8.0, pandas 3.0.1, numpy 2.4.2, scipy 1.17.0, matplotlib 3.10.8, joblib 1.5.3.

---

## Installation

```bash
# 1. Clone the repository
git clone <repository-url> thesis-project
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

---

## Usage

Every entry point is a Python script in `src/`.

### Full pipeline (recommended)

```bash
python src/main.py
```

Runs Phases 1 + 2 (generate + featurise) → 2b (Mann-Whitney + effect-size plots) → 3a (multi-model CV) → 3b (final RF train + save + ROC/PR + importance plot) → 3c (subtle-triggerbot robustness sweep) → 4 (5-player demo predictions with explanations) → 5 (per-player diagnostic PNGs).

Outputs land in:
- `models/cheat_detector.pkl` — persisted Random Forest
- `reports/*.png` — analytical figures + per-player diagnostics

### Individual scripts

| Script | Purpose |
|---|---|
| `python src/train.py` | Train + evaluate the model, no demo predictions |
| `python src/predict.py` | Load saved model, classify a fresh batch with explanations |
| `python src/visualize.py` | Generate per-player diagnostic PNGs from the saved model |
| `python src/diagrams.py` | Re-render the pipeline / architecture / verdict-workflow diagrams |
| `python src/dataset_export.py` | Export the canonical 5000-row dataset to `data/Synthetic/canonical_dataset.csv` |
| `python src/detailed_results.py` | Write extended results tables to `reports/detailed_results.txt` |

---

## Project structure

```
thesis-project/
├── src/
│   ├── config.py            # paths, knobs, hyperparameters, thresholds
│   ├── classes.py           # PlayerSession / KillEvent / MatchSession schema, rank tables
│   ├── data_gen.py          # generate_match_session()
│   ├── features.py          # matches_to_dataframe, add_features, FEATURE_COLUMNS (25 features)
│   ├── stats.py             # Mann-Whitney, Cohen's d, rank-biserial, effect-size plot, distribution plot
│   ├── validation.py        # seed, model factories, multi-model CV, ROC/PR + importance plots
│   ├── robustness.py        # subtle-triggerbot sweep + degradation curve
│   ├── predict.py           # classify + 3-tier verdict + per-feature explanations
│   ├── visualize.py         # per-player diagnostic figure
│   ├── diagrams.py          # pipeline / architecture / verdict-workflow diagrams
│   ├── train.py             # standalone trainer
│   ├── main.py              # full pipeline orchestrator
│   ├── dataset_export.py    # canonical CSV exporter
│   ├── detailed_results.py  # extended results tables
│   └── requirements.txt
├── data/
│   └── Synthetic/
│       └── canonical_dataset.csv   # produced by dataset_export.py
├── models/
│   └── cheat_detector.pkl
└── reports/
    ├── pipeline.png
    ├── architecture.png
    ├── verdict_workflow.png
    ├── effect_sizes.png
    ├── time_on_target_distributions.png
    ├── feature_importances.png
    ├── roc_pr_curves.png
    ├── robustness_curve.png
    ├── detailed_results.txt
    └── diagnostic_player_*.png
```

---

## Reproducibility

A single `seed_everything()` call (in `validation.py`) seeds Python's `random` and `numpy.random`. Every scikit-learn component is constructed with `random_state=42`. Given identical code, the full pipeline produces bit-for-bit identical results across runs.

Verify by running `python src/main.py` twice — class counts (`triggerbots=2501  legits=2499`) and all metrics will match.

---

## License / Citation

Academic thesis project — see thesis document for citation. No real player data, telemetry, or cheating software is used or produced.
