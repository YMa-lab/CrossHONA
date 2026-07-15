"""Grouped bar plot of Accuracy and ARI for every method.

X-axis groups: Accuracy, ARI. Within each group there is one bar per method
in METHOD_ORDER, colored with the user's palette in reverse order:
    ["#1d4e89", "#00b2ca", "#7dcfb6", "#fbd1a2", "#f79256"]

Both metrics are computed on the fly from each method's loaded adata so
they stay consistent with the UMAP / confusion plots — including the per-
dataset cell-type relabeling.

STAI-X_code variant: reads the prebuilt `figure2_source.h5ad` via
figure2_source.py instead of the five original per-method h5ads. The published
run resolved every method through the live loaders, so the metrics_basic.json
fallback never fired; it is dropped here because a column missing from a
prebuilt source file is a real error, not something to paper over.
"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score

from metrics_source import (
    LOADERS,
    METHOD_ORDER,
    apply_style,
    figure_paths,
    relabel_cell_types,
)

apply_style()


# user-supplied palette, reversed so the first method gets the darkest hue
PALETTE_REVERSED = ["#e29074", "#f79256", "#fbd1a2", "#7dcfb6", "#00b2ca", "#1d4e89"][::-1]


def subset_adata(adata, subset, species1, species2):
    if subset == "all":
        return adata
    target = species2 if subset == "target" else species1
    mask = adata.obs["species"].astype(str).values == target
    return adata[mask].copy()


def metrics_for(adata):
    y_true = adata.obs["cell_type"].astype(str).values
    y_pred = adata.obs["pred"].astype(str).values
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--species1", required=True)
    ap.add_argument("--species2", required=True)
    ap.add_argument("--cell_col", default="cell_type")
    ap.add_argument("--batch_col", default="species")
    ap.add_argument("--subset", choices=["target", "ref", "all"], default="target",
                    help="Which species' cells to score on. 'target' = species2 (default), "
                         "'ref' = species1, 'all' = pooled.")
    ap.add_argument("--figure", default="Figure2",
                    help="Which figure in this package (Figure2 = MERFISH_Cortex, "
                         "Figure4 = sc_small_intestine). Sets --source/--savedir defaults.")
    ap.add_argument("--source", default=None,
                    help="Path to the figure's *_source.h5ad (default: from --figure).")
    ap.add_argument("--savedir", default=None,
                    help="Literal output dir (default: results/<figure>/reproduced_figures/).")
    args = ap.parse_args()

    src_default, regen_default = figure_paths(args.figure)
    source = args.source or src_default
    out_dir = args.savedir or regen_default
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, f"barplot_acc_ari_{args.subset}")

    # Compute (Accuracy, ARI) per method
    method_scores = {}
    for m in METHOD_ORDER:
        a = LOADERS[m](args.dataset, args.species1, args.species2,
                       args.cell_col, args.batch_col, source=source)
        if a is not None:
            relabel_cell_types(a, args.dataset)
            a = subset_adata(a, args.subset, args.species1, args.species2)
            if a.n_obs > 0:
                s = metrics_for(a)
                method_scores[m] = s
                print(f"[{m}] acc={s['Accuracy']:.3f}  ari={s['ARI']:.3f}", flush=True)
                continue
        print(f"[{m}] missing", flush=True)
        method_scores[m] = {"Accuracy": np.nan, "ARI": np.nan}

    metrics = ["Accuracy", "ARI"]
    n_methods = len(METHOD_ORDER)

    # Figure: square, one row of grouped bars
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_w = 0.8 / n_methods
    x = np.arange(len(metrics))
    for i, m in enumerate(METHOD_ORDER):
        vals = [method_scores[m][k] for k in metrics]
        offsets = x + (i - (n_methods - 1) / 2) * bar_w
        ax.bar(offsets, vals, width=bar_w, color=PALETTE_REVERSED[i], label=m,
               edgecolor="black", linewidth=1.6)
        # value labels on top of each bar
        for xo, v in zip(offsets, vals):
            if not np.isfinite(v):
                continue
            ax.text(xo, v + 0.018, f"{v:.2f}", ha="center", va="bottom", fontsize=14)

    # Metric names sit ABOVE the axes box, not inside it.
    # If any score exceeds 0.9, give the value labels extra headroom (ylim→1.1)
    # but keep tick labels at 1.0 max so the axis still reads as a 0–1 scale.
    all_vals = [v for s in method_scores.values() for v in s.values() if np.isfinite(v)]
    y_top = 1.1 if any(v > 0.9 for v in all_vals) else 1.0
    ax.set_ylim(0, y_top)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks([])
    label_trans = ax.get_xaxis_transform()  # x in data, y in axes fraction
    for xi, name in zip(x, metrics):
        ax.text(xi, 1.04, name, ha="center", va="bottom", transform=label_trans,
                fontsize=22, fontweight="bold", clip_on=False)

    ax.tick_params(axis="y", labelsize=18, length=2.5, width=1.0)
    # Legend across two lines
    legend_cols = (n_methods + 1) // 2  # 5 → 3 cols × 2 rows
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
              ncol=legend_cols, frameon=False, fontsize=18)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
