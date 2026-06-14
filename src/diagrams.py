import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import REPORTS_DIR



PALETTE = {
    "data":     "#3498db",
    "feat":     "#1abc9c",
    "model":    "#9b59b6",
    "eval":     "#e67e22",
    "viz":      "#34495e",
    "pass":     "#2ecc71",
    "review":   "#f39c12",
    "trigger":  "#e74c3c",
    "sidebar":  "#ecf0f1",
    "muted":    "#7f8c8d",
}


def _box(ax, x, y, w, h, text, fill, fontsize=10, fontweight="bold", text_color="white"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.4, edgecolor="#2c3e50", facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            wrap=True)


def _arrow(ax, p_from, p_to, color="#2c3e50", style="arc3,rad=0.0", lw=1.8):
    ax.add_patch(FancyArrowPatch(
        p_from, p_to, arrowstyle="->",
        mutation_scale=20, linewidth=lw, color=color,
        connectionstyle=style,
    ))


def _label(ax, x, y, text, fontsize=9, color="#2c3e50", ha="center", style="italic"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fontsize, color=color, style=style)





def plot_pipeline(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")


    boxes = [
        (0.3, 5.0, 2.4, 1.6, "Synthetic\nMatch Generator\n(classes.py, data_gen.py)", PALETTE["data"]),
        (3.1, 5.0, 2.4, 1.6, "Feature\nEngineering\n(features.py, 25 features)", PALETTE["feat"]),
        (5.9, 5.0, 2.4, 1.6, "Stratified\nTrain/Test Split\n(80% / 20%)", PALETTE["model"]),
        (8.7, 5.0, 2.4, 1.6, "5-fold CV\n+ Multi-model\n(validation.py)", PALETTE["model"]),
        (11.5, 5.0, 2.4, 1.6, "Final RF Training\n+ persistence\n(cheat_detector.pkl)", PALETTE["model"]),
    ]
    for x, y, w, h, txt, color in boxes:
        _box(ax, x, y, w, h, txt, color, fontsize=9)


    for (x1, _, w1, _, _, _), (x2, _, _, _, _, _) in zip(boxes[:-1], boxes[1:]):
        _arrow(ax, (x1 + w1, 5.8), (x2, 5.8))


    bot = [
        (0.3, 1.6, 3.4, 1.6, "Statistical Tests\n(Mann-Whitney U,\nCohen's d, rank-biserial)\n(stats.py)", PALETTE["eval"]),
        (4.1, 1.6, 3.4, 1.6, "Robustness Sweep\n(subtle-triggerbot\nhold_time sweep)\n(robustness.py)", PALETTE["eval"]),
        (7.9, 1.6, 3.4, 1.6, "Per-player Diagnostics\n(diagnostic PNGs,\nfeature explanations)\n(visualize.py)", PALETTE["eval"]),
        (11.7, 1.6, 4.0, 1.6, "Per-kill Verdict\nPASS / REVIEW / TRIGGERBOT\n+ probability\n(predict.py)", PALETTE["viz"]),
    ]
    for x, y, w, h, txt, color in bot:
        _box(ax, x, y, w, h, txt, color, fontsize=9)


    _arrow(ax, (12.7, 5.0), (13.7, 3.2))
    _arrow(ax, (12.7, 5.0), (9.6,  3.2), style="arc3,rad=0.15")
    _arrow(ax, (12.7, 5.0), (5.8,  3.2), style="arc3,rad=0.25")
    _arrow(ax, (4.3,  5.0), (2.0,  3.2), style="arc3,rad=-0.15")


    ax.text(8, 7.5, "Figure 1.  Data and Modelling Pipeline",
            ha="center", fontsize=15, fontweight="bold", color="#2c3e50")
    ax.text(8, 0.6,
            "A single run flows left-to-right through the top row (generation, features, splitting, training);\n"
            "evaluation modules (bottom row) consume the trained model and produce the per-kill verdict and diagnostics.",
            ha="center", fontsize=9, style="italic", color="#34495e")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)





def plot_architecture(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")


    layer_x = 1.0
    layer_w = 10.0
    layer_h = 1.55
    layers = [
        (8.05, "Data Generation Layer",
         "classes.py  (PlayerSession, KillEvent, MatchSession, rank tables)\n"
         "data_gen.py  (generate_match_session)   →   List[MatchSession]",
         PALETTE["data"]),
        (6.20, "Feature Engineering Layer",
         "features.py  (matches_to_dataframe, add_features, FEATURE_COLUMNS)\n"
         "raw events + tier-relative anomalies   →   25-column feature DataFrame",
         PALETTE["feat"]),
        (4.35, "Modelling Layer",
         "validation.py  (Random Forest, Gradient Boosting, Logistic Regression — 5-fold CV)\n"
         "train.py / main.py  →   cheat_detector.pkl   via joblib",
         PALETTE["model"]),
        (2.50, "Evaluation Layer",
         "stats.py  (Mann-Whitney U, Cohen's d, rank-biserial)   ·   robustness.py  (subtlety sweep)\n"
         "predict.py  (3-tier verdict)   ·   visualize.py  (per-player diagnostic PNGs)",
         PALETTE["eval"]),
    ]
    for y, title, body, color in layers:
        _box(ax, layer_x, y, layer_w, layer_h, "", color)
        ax.text(layer_x + 0.3, y + layer_h - 0.30, title,
                ha="left", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(layer_x + 0.3, y + 0.45, body,
                ha="left", va="center", fontsize=9, color="white")


    for y_top in [8.05, 6.20, 4.35]:
        _arrow(ax, (layer_x + layer_w / 2, y_top), (layer_x + layer_w / 2, y_top - 0.30))


    rx, ry, rw, rh = 11.6, 6.20, 3.1, 3.40
    _box(ax, rx, ry, rw, rh, "", PALETTE["sidebar"], text_color="#2c3e50")
    ax.text(rx + rw / 2, ry + rh - 0.30, "Reproducibility",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.text(rx + 0.2, ry + rh - 1.20,
            "•  seed_everything()\n"
            "•  RANDOM_STATE = 42\n"
            "•  sklearn random_state set\n"
            "    on every component\n"
            "•  StratifiedKFold(shuffle=True,\n"
            "    random_state=42)\n"
            "•  Identical code = identical\n"
            "    result, end-to-end",
            ha="left", va="top", fontsize=9, color="#2c3e50")


    sx, sy, sw, sh = 11.6, 2.50, 3.1, 3.40
    _box(ax, sx, sy, sw, sh, "", PALETTE["sidebar"], text_color="#2c3e50")
    ax.text(sx + sw / 2, sy + sh - 0.30, "Software Stack",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#2c3e50")
    ax.text(sx + 0.2, sy + sh - 1.20,
            "•  Python 3.11.9\n"
            "•  scikit-learn 1.8.0\n"
            "•  pandas 3.0.1\n"
            "•  numpy 2.4.2\n"
            "•  scipy 1.17.0\n"
            "•  matplotlib 3.10.8\n"
            "•  joblib 1.5.3",
            ha="left", va="top", fontsize=9, color="#2c3e50")


    ax.text(7.5, 1.5,
            "The implementation is organised into four sequential layers feeding into one another, with shared reproducibility\n"
            "controls and a fixed software stack supporting every stage.",
            ha="center", fontsize=9, style="italic", color="#34495e")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_verdict_workflow(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")


    _box(ax, 4.5, 8.0, 5.0, 1.3,
         "Trained Random Forest classifier\noutputs  triggerbot_probability  p  ∈  [0, 1]",
         PALETTE["model"], fontsize=11)


    _box(ax, 5.0, 5.5, 4.0, 1.5,
         "Threshold routing\n(predict.py · verdict_from_prob)",
         PALETTE["viz"], fontsize=11)

    _arrow(ax, (7.0, 8.0), (7.0, 7.0), lw=2.4)


    outcomes = [
        (0.5, 2.0, "PASS",       "p < 0.30",      "No action.\nPlayer cleared.",                  PALETTE["pass"]),
        (5.0, 2.0, "REVIEW",     "0.30 ≤ p < 0.70", "Routed to human moderator.\nBorderline case — operational review.", PALETTE["review"]),
        (9.5, 2.0, "TRIGGERBOT", "p ≥ 0.70",      "Flagged for action.\nHigh-confidence detection.", PALETTE["trigger"]),
    ]
    for x, y, label, threshold, action, color in outcomes:
        _box(ax, x, y + 1.0, 4.0, 1.3, label, color, fontsize=14)
        ax.text(x + 2.0, y + 0.65, threshold,
                ha="center", va="center", fontsize=10, color="#2c3e50", fontweight="bold")
        ax.text(x + 2.0, y + 0.10, action,
                ha="center", va="center", fontsize=9, color="#2c3e50")


    _arrow(ax, (5.5, 5.5), (2.5, 4.3), style="arc3,rad=-0.2", lw=2.0)
    _arrow(ax, (7.0, 5.5), (7.0, 4.3), lw=2.0)
    _arrow(ax, (8.5, 5.5), (11.5, 4.3), style="arc3,rad=0.2", lw=2.0)


    ax.text(7, 9.6, "Figure 3.  Three-Tier Verdict Workflow",
            ha="center", fontsize=15, fontweight="bold", color="#2c3e50")
    ax.text(7, 0.6,
            "Each kill event is routed by the classifier's probability output to one of three outcomes.\n"
            "The REVIEW tier reserves borderline cases for human moderation rather than automated punishment.",
            ha="center", fontsize=9, style="italic", color="#34495e")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path     = REPORTS_DIR / "pipeline.png"
    architecture_path = REPORTS_DIR / "architecture.png"
    workflow_path     = REPORTS_DIR / "verdict_workflow.png"

    plot_pipeline(pipeline_path)
    print(f"Pipeline diagram         -> {pipeline_path}")
    plot_architecture(architecture_path)
    print(f"Architecture diagram     -> {architecture_path}")
    plot_verdict_workflow(workflow_path)
    print(f"Verdict workflow diagram -> {workflow_path}")


if __name__ == "__main__":
    main()
