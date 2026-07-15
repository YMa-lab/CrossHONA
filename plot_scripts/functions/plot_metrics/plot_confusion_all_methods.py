"""Plot per-method cell-type confusion matrices.

Reuses the per-method loaders from plot_umaps_all_methods.py so the dataset
naming, predicted-label decoding, and label rewrites stay consistent.

Default: confusion matrix on the *target* species (the cross-species
annotation-transfer setting). Pass --subset all / ref / target to change.

STAI-X_code variant: reads the prebuilt `figure2_source.h5ad` via
figure2_source.py instead of the five original per-method h5ads.

Note --normalize true is required to match the published figure (the script
default is 'none'); it also gates the shared colorbar.
"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from metrics_source import (
    LOADERS,
    METHOD_ORDER,
    apply_style,
    figure_paths,
    relabel_cell_types,
)

apply_style()


def confusion_for(adata, normalize="true"):
    """Return (matrix, labels) for adata.obs['cell_type'] vs adata.obs['pred']."""
    y_true = adata.obs["cell_type"].astype(str).values
    y_pred = adata.obs["pred"].astype(str).values
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm, labels


def subset_adata(adata, subset, species1, species2):
    if subset == "all":
        return adata
    target = species2 if subset == "target" else species1
    mask = adata.obs["species"].astype(str).values == target
    return adata[mask].copy()


def _short(label, n=10):
    s = str(label)
    return s if len(s) <= n else s[: n - 1] + "…"


def plot_one(ax, cm, labels, title, normalized, show_ylabels=True):
    """normalized=True → values in [0,1], format as 2 decimals; else absolute counts."""
    if normalized:
        vmin, vmax = 0.0, 1.0
        fmt = lambda v: f"{v:.2f}"
        threshold = 0.5
    else:
        vmin, vmax = 0, cm.max() if cm.size and cm.max() > 0 else 1
        fmt = lambda v: f"{int(v)}"
        threshold = vmax / 2 if vmax > 0 else 0.5

    short_labels = [_short(l) for l in labels]
    im = ax.imshow(cm, vmin=vmin, vmax=vmax, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    if show_ylabels:
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_ylabel("Ground Truth", fontsize=14)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xlabel("Prediction", fontsize=14)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if v > threshold else "black"
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9, color=color)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--species1", required=True)
    ap.add_argument("--species2", required=True)
    ap.add_argument("--cell_col", default="cell_type")
    ap.add_argument("--batch_col", default="species")
    ap.add_argument("--subset", choices=["target", "ref", "all"], default="target",
                    help="Which species' cells to score on. 'target' = species2 (cross-species transfer), "
                         "'ref' = species1, 'all' = pooled.")
    ap.add_argument("--normalize", choices=["true", "pred", "all", "none"], default="none",
                    help="Confusion-matrix normalization (sklearn). 'none' = absolute counts (default), "
                         "'true' = per-row, 'pred' = per-col, 'all' = over the whole matrix.")
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
    out_base = os.path.join(out_dir, f"confusion_{args.subset}")

    # Load each method (no UMAP this time)
    method_data = {}
    for m in METHOD_ORDER:
        print(f"[{m}] loading...", flush=True)
        a = LOADERS[m](args.dataset, args.species1, args.species2,
                       args.cell_col, args.batch_col, source=source)
        if a is None:
            print(f"[{m}] missing", flush=True)
            method_data[m] = None
            continue
        relabel_cell_types(a, args.dataset)
        a = subset_adata(a, args.subset, args.species1, args.species2)
        method_data[m] = a
        print(f"[{m}] ready, n_cells={a.n_obs}", flush=True)

    # 1 row × N methods
    n = len(METHOD_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5), squeeze=False)

    norm_arg = None if args.normalize == "none" else args.normalize
    normalized = norm_arg is not None
    last_im = None
    for i, m in enumerate(METHOD_ORDER):
        ax = axes[0, i]
        a = method_data[m]
        show_ylabels = (i == 0)
        if a is None or a.n_obs == 0:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(m, fontsize=20, pad=14)
            continue
        cm, labels = confusion_for(a, normalize=norm_arg)
        last_im = plot_one(ax, cm, labels, m, normalized=normalized, show_ylabels=show_ylabels)

    # In absolute-count mode each method has a different vmax, so a single shared
    # colorbar would be misleading — only attach one when values are normalized.
    if last_im is not None and normalized:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=12)

    fig.suptitle(
        f"{args.dataset}  {args.species1} vs {args.species2}  ({args.subset})",
        fontsize=18,
    )
    for ext in ("png", "pdf"):
        path = f"{out_base}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
