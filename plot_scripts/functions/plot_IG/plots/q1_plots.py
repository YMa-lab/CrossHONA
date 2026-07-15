"""Q1 visualization: the IG combined heatmap (Figure 3 panels A and B).

Trimmed from biological_interpretability/plots/q1_plots.py to the single
function Figure 3 uses. `plot_ig_combined_heatmap` is copied verbatim -- byte
for byte -- from the upstream module; only the unused siblings
(plot_class_gene_heatmap, plot_method_agreement, plot_homo_nonhomo_stacked_bar,
plot_homo_nonhomo_paired_stacked_bar, plot_homo_fraction_heatmap, render_q1_all)
and the imports/constants they alone needed (glob, Path, Dict, maybe_basename,
METHODS_DEFAULT, BRANCHES, GENE_SETS) are gone. Verified: the panels render
byte-identical before and after.

Upstream original: scripts_benchmarks_new/biological_interpretability/plots/q1_plots.py
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .utils import save_fig, species_palette


def plot_ig_combined_heatmap(
    ref_homo_csv: str,
    tgt_homo_csv: str,
    ref_nonhomo_csv: Optional[str],
    tgt_nonhomo_csv: Optional[str],
    save_path: str,
    top_k_homo: int = 30,
    top_k_nonhomo: int = 30,
    log: bool = True,
    *,
    branch_labels: tuple = ("ref", "tgt"),
    class_transform=None,
    font_scale: float = 1.0,
    cmap="viridis",
    title_font: Optional[str] = None,
):
    """
    Combined Q1 figure for IG attributions only.

    Layout (single figure, shared cell-type y-axis, shared colorbar):
        [ homologs: ref|tgt twin columns per pair ]
        | [ ref-specific top-K ] | [ tgt-specific top-K ]

    Homolog pairs are matched case-insensitively between ref columns
    (e.g. ``FREM2``) and tgt columns (e.g. ``Frem2``); top-K pairs are
    chosen by combined |attribution| summed over cell types and ranked
    in descending order.  A small species stripe above the homolog block
    indicates ref vs tgt column identity.
    """
    df_rh = pd.read_csv(ref_homo_csv, index_col=0)
    df_th = pd.read_csv(tgt_homo_csv, index_col=0)

    classes = df_rh.index.astype(str).tolist()
    th_classes = df_th.index.astype(str).tolist()
    if th_classes != classes:
        common = [c for c in classes if c in th_classes]
        df_rh = df_rh.loc[common]
        df_th = df_th.loc[common]
        classes = common

    ref_genes = list(df_rh.columns)
    tgt_by_lower = {g.lower(): g for g in df_th.columns}
    pairs = [(rg, tgt_by_lower[rg.lower()])
             for rg in ref_genes if rg.lower() in tgt_by_lower]
    if not pairs:
        raise ValueError(
            "No homolog pairs matched between ref and tgt by case-insensitive name."
        )
    pair_score = np.array([
        float(df_rh[rg].sum() + df_th[tg].sum()) for rg, tg in pairs
    ])
    pair_order = np.argsort(pair_score)[::-1][:top_k_homo]
    pairs_top = [pairs[i] for i in pair_order]

    n_cls = len(classes)
    n_homo = 2 * len(pairs_top)
    homo_mat = np.zeros((n_cls, n_homo), dtype=float)
    species_strip = np.zeros((1, n_homo), dtype=int)
    homo_xtick_pos: List[float] = []
    homo_xtick_lbl: List[str] = []
    # One x-tick per column so both Human (ref) and Mouse (tgt) gene names
    # are visible — left column of each pair gets the ref name, right column
    # gets the tgt name.
    for j, (rg, tg) in enumerate(pairs_top):
        homo_mat[:, 2 * j] = df_rh[rg].values
        homo_mat[:, 2 * j + 1] = df_th[tg].values
        species_strip[0, 2 * j + 1] = 1
        homo_xtick_pos.append(2 * j)
        homo_xtick_pos.append(2 * j + 1)
        homo_xtick_lbl.append(rg)
        homo_xtick_lbl.append(tg)

    def topk_block(path: Optional[str], k: int):
        if not (path and os.path.isfile(path)):
            return None, None
        df = pd.read_csv(path, index_col=0).reindex(classes).fillna(0.0)
        scores = df.values.sum(axis=0)
        order = np.argsort(scores)[::-1][:k]
        return df.values[:, order], list(np.array(df.columns)[order])

    rnh, rnh_lbl = topk_block(ref_nonhomo_csv, top_k_nonhomo)
    tnh, tnh_lbl = topk_block(tgt_nonhomo_csv, top_k_nonhomo)

    blocks = [homo_mat] + [b for b in (rnh, tnh) if b is not None]
    blocks_l = [np.log1p(b) if log else b for b in blocks]
    vmin = float(min(b.min() for b in blocks_l))
    vmax = float(max(b.max() for b in blocks_l))

    n_ref_nh = rnh.shape[1] if rnh is not None else 0
    n_tgt_nh = tnh.shape[1] if tnh is not None else 0

    width_ratios: List[float] = [n_homo]
    if n_ref_nh:
        width_ratios.append(n_ref_nh)
    if n_tgt_nh:
        width_ratios.append(n_tgt_nh)
    cb_w = max(2.0, 0.04 * sum(width_ratios))
    width_ratios.append(cb_w)

    fig_w = max(10.0, 0.22 * (n_homo + n_ref_nh + n_tgt_nh) + 2.5)
    fig_h = max(3.5, n_cls * 0.5 + 1.6)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        2, len(width_ratios),
        height_ratios=[0.10, 1.0],
        width_ratios=width_ratios,
        hspace=0.18, wspace=0.06,
        figure=fig,
    )

    sp_pal = species_palette()
    sp_cmap = mpl.colors.ListedColormap([sp_pal["ref"], sp_pal["tgt"]])
    ref_lbl, tgt_lbl = branch_labels
    fs_small  = 7 * font_scale
    fs_normal = 8 * font_scale
    fs_title  = 9 * font_scale
    fs_sup    = 11 * font_scale
    ytick_classes = [class_transform(c) for c in classes] if class_transform else classes

    ax_strip = fig.add_subplot(gs[0, 0])
    ax_strip.imshow(species_strip, aspect="auto", cmap=sp_cmap, vmin=0, vmax=1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    ax_strip.set_title(
        f"Homologs ({ref_lbl}|{tgt_lbl}; top {len(pairs_top)} pairs)",
        fontsize=fs_title,
    )

    ax_homo = fig.add_subplot(gs[1, 0])
    ax_homo.imshow(
        np.log1p(homo_mat) if log else homo_mat,
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax_homo.set_xticks(homo_xtick_pos)
    ax_homo.set_xticklabels(homo_xtick_lbl, rotation=90, fontsize=fs_small)
    # Colour each tick label by the species that column belongs to so a
    # reader can read off Human vs Mouse gene names at a glance.
    for tick_lbl, idx in zip(ax_homo.get_xticklabels(), homo_xtick_pos):
        tick_lbl.set_color(sp_pal["ref"] if int(idx) % 2 == 0 else sp_pal["tgt"])
    # Keep the visual pair grouping with white separators between pairs.
    for j in range(1, len(pairs_top)):
        ax_homo.axvline(2 * j - 0.5, color="white", lw=0.6, alpha=0.7)
    ax_homo.set_yticks(range(n_cls))
    ax_homo.set_yticklabels(ytick_classes, fontsize=fs_normal)

    next_col = 1
    if rnh is not None:
        ax_rnh = fig.add_subplot(gs[1, next_col])
        ax_rnh.imshow(
            np.log1p(rnh) if log else rnh,
            aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax_rnh.set_xticks(range(len(rnh_lbl)))
        ax_rnh.set_xticklabels(rnh_lbl, rotation=90, fontsize=fs_small)
        ax_rnh.set_yticks([])
        ax_rnh_top = fig.add_subplot(gs[0, next_col])
        ax_rnh_top.axis("off")
        ax_rnh_top.set_title(
            f"{ref_lbl}-specific (top {len(rnh_lbl)})",
            fontsize=fs_title, color=sp_pal["ref"],
        )
        next_col += 1
    if tnh is not None:
        ax_tnh = fig.add_subplot(gs[1, next_col])
        ax_tnh.imshow(
            np.log1p(tnh) if log else tnh,
            aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax_tnh.set_xticks(range(len(tnh_lbl)))
        ax_tnh.set_xticklabels(tnh_lbl, rotation=90, fontsize=fs_small)
        ax_tnh.set_yticks([])
        ax_tnh_top = fig.add_subplot(gs[0, next_col])
        ax_tnh_top.axis("off")
        ax_tnh_top.set_title(
            f"{tgt_lbl}-specific (top {len(tnh_lbl)})",
            fontsize=fs_title, color=sp_pal["tgt"],
        )
        next_col += 1

    cax = fig.add_subplot(gs[:, -1])
    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(
        "log1p(mean |attribution|)" if log else "mean |attribution|",
        fontsize=fs_normal,
    )
    cb.ax.tick_params(labelsize=fs_small)

    fig.legend(
        handles=[
            mpl.patches.Patch(color=sp_pal["ref"], label=ref_lbl),
            mpl.patches.Patch(color=sp_pal["tgt"], label=tgt_lbl),
        ],
        loc="upper right", bbox_to_anchor=(0.99, 0.99),
        ncol=2, frameon=False, fontsize=fs_normal,
    )
    suptitle_kw = {"fontsize": fs_sup}
    if title_font:
        suptitle_kw["fontname"] = title_font
    fig.suptitle("Q1 — IG attribution: combined heatmap", **suptitle_kw)
    return save_fig(fig, save_path)
