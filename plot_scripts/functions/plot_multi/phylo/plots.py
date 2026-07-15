"""Figure 5 phylo panel renderers — verbatim from phylogenetic/plots.py.

Trimmed to the two functions Figure 5 uses:
    plot_top_genes_per_celltype_grid            -> panel A
    plot_distance_vs_mya_three_metrics_row      -> panel B
plus the two module helpers they call. Dropped: plot_distance_vs_mya,
plot_dendrogram_with_phylogeny, plot_importance_heatmap_phylo_ordered,
plot_dendrograms_per_celltype, plot_cell_type_clock_rates,
plot_homo_fraction_on_tree, and the imports only they needed (os, Path, Tuple,
GridSpec, dendrogram, leaves_list, and the whole trees.py module).

 is rewritten to 
-- upstream reaches out to the parent package; here utils.py sits alongside.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .tensor_builder import ImportanceTensor
from .utils import save_fig


def _is_anchor_view(name: str, anchor: str) -> bool:
    if anchor is None:
        return False
    return name == anchor or name.startswith(f"{anchor}_via_")


def _canonical_anchor(name: str, anchor: str = None) -> str:
    if anchor is None or name == anchor:
        return name
    if name.startswith(f"{anchor}_via_"):
        return anchor
    return name


def plot_distance_vs_mya_three_metrics_row(
    pc_distances_by_metric: Dict[str, Dict[str, pd.DataFrame]],
    overall_rho_by_metric: Dict[str, float],
    species: List[str],
    reference_lookup,
    save_path: str,
    anchor: Optional[str] = None,
    metric_titles: Optional[Dict[str, str]] = None,
    cell_type_colors: Optional[List[str]] = None,
    cell_type_display: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    font_scale: float = 1.6,
):
    """
    One-row, three-panel figure of distance vs MYA for the three metrics
    (typically Spearman, top-K Jaccard, L1).

    Parameters
    ----------
    pc_distances_by_metric : ``{metric_key: {cell_type: pd.DataFrame[S x S]}}``
        Per-cell-type distance frames per metric, in the order they should
        appear left-to-right.
    overall_rho_by_metric : ``{metric_key: float}`` overall Spearman ρ between
        the metric's pooled distance and MYA (printed in each subplot title).
    cell_type_colors : optional list of hex colors, one per cell type in the
        order encountered.  Cycled if shorter than the number of cell types.
    cell_type_display : optional rename map for cell-type legend labels
        (e.g. capitalisation).  Missing entries fall back to ``str.title()``.
    """
    metric_keys = list(pc_distances_by_metric.keys())
    n = len(metric_keys)
    if n == 0:
        return None

    base = 12.0 * font_scale / 1.6                     # base label size
    title_sz  = base * 1.35
    label_sz  = base * 1.15
    tick_sz   = base * 0.95
    legend_sz = base * 1.1
    annot_sz  = base * 0.85

    # Wider per-panel canvas so the large axis labels fit; share-y disabled
    # because the three metrics have very different magnitude scales.
    # Extra bottom margin reserved for rotated species labels under x-axis.
    has_sup = bool(title)
    fig, axes = plt.subplots(1, n, figsize=(8.2 * n, 8.6 if has_sup else 8.2),
                             sharey=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(
        wspace=0.35, left=0.07, right=0.98,
        top=0.86 if has_sup else 0.92,
        bottom=0.32,
    )

    # determine cell-type order from first metric (preserve order)
    first_pcd = pc_distances_by_metric[metric_keys[0]]
    cell_types = list(first_pcd.keys())
    if cell_type_colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(k % 10) for k in range(len(cell_types))]
    else:
        colors = [cell_type_colors[k % len(cell_type_colors)]
                  for k in range(len(cell_types))]

    def _display(ct):
        if cell_type_display and ct in cell_type_display:
            return cell_type_display[ct]
        return ct[:1].upper() + ct[1:]

    ylabel = (f"Importance Distance from {anchor}"
              if anchor is not None else "Importance Distance")

    for ax, mkey in zip(axes, metric_keys):
        pcd = pc_distances_by_metric[mkey]
        all_species_at_points = {}    # x -> set of species labels (for annotation)
        for k, ct in enumerate(cell_types):
            if ct not in pcd:
                continue
            D = pcd[ct]
            xs, ys, others = [], [], []
            species_list = list(D.index)
            for i, a in enumerate(species_list):
                for j, b in enumerate(species_list):
                    if j <= i:
                        continue
                    if anchor is not None:
                        a_is = _is_anchor_view(a, anchor)
                        b_is = _is_anchor_view(b, anchor)
                        if not (a_is or b_is) or (a_is and b_is):
                            continue
                    mya = reference_lookup(_canonical_anchor(a, anchor),
                                            _canonical_anchor(b, anchor))
                    d = float(D.iloc[i, j])
                    if not np.isfinite(mya) or not np.isfinite(d):
                        continue
                    xs.append(mya); ys.append(d)
                    other = (b if _is_anchor_view(a, anchor)
                             else (a if _is_anchor_view(b, anchor)
                                   else f"{a},{b}"))
                    others.append(other)
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]
            others = [others[i] for i in order]
            ax.plot(xs, ys, "-o", color=colors[k], label=_display(ct),
                    lw=4.5, ms=11, alpha=0.92,
                    markeredgecolor="white", markeredgewidth=1.0)
            # accumulate per-x species labels for a single set of annotations
            for x, lab in zip(xs, others):
                all_species_at_points.setdefault(float(x), set()).add(lab)

        # one species label per unique MYA value, placed below the x-axis
        # near the corresponding tick.  Rotated 45° so that closely-spaced
        # primate divergences (Gorilla 8.6, Gibbon 20.2, Macaque 28.8,
        # Marmoset 42.6 MYA) do not overlap.
        if all_species_at_points:
            xs_sorted = sorted(all_species_at_points)
            for x in xs_sorted:
                lab = ", ".join(sorted(all_species_at_points[x]))
                ax.annotate(lab, xy=(x, 0), xytext=(0, -tick_sz * 1.5),
                            xycoords=("data", "axes fraction"),
                            textcoords="offset points",
                            ha="right", va="top",
                            fontsize=annot_sz, color="#333", rotation=45,
                            rotation_mode="anchor", clip_on=False)

        rho = overall_rho_by_metric.get(mkey, float("nan"))
        nice = (metric_titles or {}).get(mkey, mkey)
        rho_str = f"$\\rho$ = {rho:.2f}" if np.isfinite(rho) else "$\\rho$ = n/a"
        ax.set_title(f"{nice}\n(Spearman {rho_str})",
                     fontsize=title_sz, pad=22)
        ax.set_xlabel("Phylogenetic Divergence (MYA)",
                      fontsize=label_sz, labelpad=annot_sz * 4.5)
        ax.set_ylabel(ylabel, fontsize=label_sz)
        ax.tick_params(axis="both", labelsize=tick_sz)
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # single legend at the very bottom, below the xlabel + species annotations
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=min(len(labels), 5), frameon=False, fontsize=legend_sz,
            handlelength=2.4, handletextpad=0.6, columnspacing=2.0,
        )
    if title:
        fig.suptitle(title, fontsize=title_sz * 1.05, y=0.96)
    return save_fig(fig, save_path)


def plot_top_genes_per_celltype_grid(
    it: ImportanceTensor,
    save_path: str,
    top_n: int = 3,
    anchor: Optional[str] = None,
    reference_lookup=None,
    title: Optional[str] = None,
):
    """
    Aggregate panel: for each cell type, take the top-``top_n`` genes (ranked
    by mean |attribution| across species) and concatenate them along the
    x-axis with vertical separators between cell types.  Species (rows) are
    ordered by reference phylogeny from the anchor if ``reference_lookup`` is
    given, otherwise by the input species order.

    Cell-type titles displayed above each block are capitalised.
    """
    species = it.species
    n_ct = len(it.cell_types)
    if n_ct == 0:
        return None

    # ---- row order: by phylogeny from anchor if available
    if reference_lookup is not None and anchor is not None and anchor in species:
        mya = np.array([reference_lookup(anchor, s) for s in species], dtype=float)
        order = np.argsort(np.where(np.isfinite(mya), mya, np.inf))
    else:
        order = np.arange(len(species))
    species_ord = [species[i] for i in order]

    # ---- collect top-N genes per cell type
    blocks = []          # list of (cell_type, gene_names, sub_matrix [S, top_n])
    for c, ct in enumerate(it.cell_types):
        Mc = it.tensor[order, c, :]                      # [S, G]
        abs_imp = np.nan_to_num(np.abs(Mc), nan=0.0)
        score = abs_imp.mean(axis=0)
        K = min(top_n, len(it.genes))
        idx = np.argsort(-score)[:K]
        gene_names = [it.genes[i] for i in idx]
        blocks.append((ct, gene_names, Mc[:, idx]))

    # ---- assemble full matrix with NaN separator columns
    sep = np.full((len(species_ord), 1), np.nan)
    full_cols = []
    gene_labels = []
    block_starts = []
    block_widths = []
    cursor = 0
    for k, (ct, genes, sub) in enumerate(blocks):
        block_starts.append(cursor)
        block_widths.append(sub.shape[1])
        full_cols.append(sub)
        gene_labels.extend(genes)
        cursor += sub.shape[1]
        if k < len(blocks) - 1:
            full_cols.append(sep)
            gene_labels.append("")
            cursor += 1
    full = np.concatenate(full_cols, axis=1)

    plot = np.log1p(np.nan_to_num(full, nan=0.0))
    n_cols = plot.shape[1]
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * n_cols),
                                    max(3.2, 0.5 * len(species_ord) + 1.2)))
    im = ax.imshow(plot, aspect="auto", cmap="viridis")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(species_ord)))
    ax.set_yticklabels(species_ord, fontsize=9)

    # cell-type block headers (capitalised) + separator lines
    for (ct, _, _), start, width in zip(blocks, block_starts, block_widths):
        center = start + (width - 1) / 2.0
        ax.text(center, -0.8, ct[:1].upper() + ct[1:],
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                transform=ax.transData)
    # vertical separators between cell types (drawn on the separator NaN columns)
    cursor = 0
    for k, (_, _, sub) in enumerate(blocks[:-1]):
        cursor += sub.shape[1]
        ax.axvline(cursor - 0.5, color="white", lw=1.5)
        ax.axvline(cursor + 0.5, color="white", lw=1.5)
        cursor += 1

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("log1p(|attribution|)", fontsize=8)
    ax.set_title(title or f"Top-{top_n} contributing genes per cell type",
                 fontsize=10, pad=22)
    fig.subplots_adjust(top=0.85, bottom=0.22)
    return save_fig(fig, save_path)

