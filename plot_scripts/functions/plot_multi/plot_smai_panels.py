"""Figure 5 panels C, D, E — SMAI cell-type-specific analyses (CAMEX_testis).

    panel C : Fig3a_celltype_smai_mean_dots.png          cell-type SMAI-test p-values
    panel D : umap_homo_percelltype_by_species.png       per-cell-type UMAP by species
    panel E : trees_homo_meanL1_UPGMA_percelltype.png    per-cell-type species trees

Extracted from `scripts_benchmarks_new/SMAI/visualize_SMAI.ipynb` cells 16, 17
and 18 (plus the font setup in cell 15 and the BASE/CELLTYPES globals in cell 1).
The notebook is the only generator -- there is no upstream script for these three
panels, and its cells are not independently runnable (16-18 depend on cell 1's
globals and cell 15's font registration). Plot bodies are copied verbatim; the
only changes are `--source`/`--savedir` in place of the hardcoded BASE/OUTDIR,
and `plt.show()` dropped.

The R scripts in SMAI/ are the *upstream* pipeline that produced these CSVs; they
do not draw these panels.

All three read small CSV exports, never the 485 MB of .npy shared spaces:
    panel C : celltype_smai/pvalues_{emb}_{ct}_matrix.csv        15 x ~550 B
    panel D : umap_homo_percelltype_coords.csv                   2.8 MB
    panel E : distmat/cvae_homo_meanL1_{ct}.csv                  5 x ~634 B

CPU only, seconds:
    python plot_smai_panels.py --panel all
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# --- globals from notebook cell 1 --------------------------------------------
EMBS = ["homo", "nonhomo", "combined"]
CELLTYPES = ["sertoli", "somatic", "spermatids", "spermatocytes", "spermatogonia"]

# --- species palette, shared by panels D and E (cells 17/18) ------------------
SPECIES = ["Human", "Gorilla", "Gibbon", "Macaque", "Marmoset", "Mouse"]
SP_COLORS = ["#de5f55", "#ff924c", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]
PALETTE = dict(zip(SPECIES, SP_COLORS))

FONT_PATH = "/oscar/data/yma16/Project/spTransform/2.Downstream/Arial Unicode.ttf"


def _package_root(start=__file__):
    d = os.path.dirname(os.path.abspath(start))
    while True:
        if os.path.basename(d) == "STAI-X_code":
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"STAI-X_code root not found above {start}")
        d = parent


PACKAGE_ROOT = _package_root()
RESULTS_FIG5 = os.path.join(PACKAGE_ROOT, "results", "Figure5")
DEFAULT_SOURCE = os.path.join(RESULTS_FIG5, "source_csv")
REGEN_DIR = os.path.join(RESULTS_FIG5, "reproduced_figures")


def apply_font(font_path=FONT_PATH):
    """Notebook cell 15. Registers Arial Unicode and makes it the global family.

    Cells 16-18 inherit this; without it every panel silently falls back to
    DejaVu Sans and the text metrics (hence bbox_inches="tight" output size)
    change. The notebook guards on os.path.exists, so a missing font is not
    fatal -- just different.
    """
    if not os.path.exists(font_path):
        print(f"[font] {font_path} not found — falling back to matplotlib default",
              flush=True)
        return
    fm.fontManager.addfont(font_path)
    mpl.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()


def _savefig(fig, savedir, stem):
    png = os.path.join(savedir, f"{stem}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(savedir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png}", flush=True)


# ============================== panel C (cell 16) =============================
def panel_C(source, savedir):
    def pairwise_p(emb, ct):
        M = pd.read_csv(f"{source}/celltype_smai/pvalues_{emb}_{ct}_matrix.csv",
                        index_col=0).values
        return M[np.triu_indices(M.shape[0], k=1)]   # 10 off-diagonal pairs

    pmat = {e: {ct: pairwise_p(e, ct) for ct in CELLTYPES} for e in EMBS}
    mean_p = pd.DataFrame({e: [pmat[e][ct].mean() for ct in CELLTYPES] for e in EMBS},
                          index=CELLTYPES)

    colors = ["#456990", "#ef767a", "#49beaa"]
    n = len(EMBS)
    width = 0.8 / n
    x = np.arange(len(CELLTYPES))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for j, e in enumerate(EMBS):
        off = (j - (n - 1) / 2) * width
        ax.bar(x + off, mean_p[e].values, width=width, color=colors[j],
               edgecolor="k", linewidth=1.5, label=e, zorder=2)
        for i, ct in enumerate(CELLTYPES):
            pts = pmat[e][ct]
            ax.scatter(np.full(len(pts), x[i] + off), pts, s=22, color="k", alpha=0.65,
                       edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(0.05, ls="--", c="k", lw=2.5, label="p=0.05")
    ax.set_title("Cell-type-specific SMAI-test", fontsize=16, pad=10)
    ax.set_ylabel("mean p-values", fontsize=15)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CELLTYPES], rotation=25, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=13, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    plt.tight_layout()
    _savefig(fig, savedir, "Fig3a_celltype_smai_mean_dots")


# ============================== panel D (cell 17) =============================
def panel_D(source, savedir):
    co = pd.read_csv(f"{source}/umap_homo_percelltype_coords.csv")
    fig, axes = plt.subplots(1, len(CELLTYPES), figsize=(20, 4.6))
    for ax, ct in zip(axes, reversed(CELLTYPES)):
        sub = co[co.celltype == ct].sample(frac=1, random_state=0)
        for sp in SPECIES:
            s = sub[sub.species == sp]
            ax.scatter(s.UMAP1, s.UMAP2, s=4, alpha=0.6, color=PALETTE[sp], linewidths=0)
        ax.set_title(ct.capitalize(), fontsize=20, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=PALETTE[sp], label=sp,
                          markeredgecolor="k", markeredgewidth=0.5) for sp in SPECIES]
    fig.legend(handles=handles, loc="center right", markerscale=2.2, title="Species",
               fontsize=15, title_fontsize=16, frameon=False)
    fig.suptitle("Per-cell-type alignment (homo) - by species", y=1.03, fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    _savefig(fig, savedir, "umap_homo_percelltype_by_species")


# ============================== panel E (cell 18) =============================
def truth_patristic(sp):
    clades = [{"Human", "Gorilla"}, {"Human", "Gorilla", "Gibbon"},
              {"Human", "Gorilla", "Gibbon", "Macaque"},
              {"Human", "Gorilla", "Gibbon", "Macaque", "Marmoset"}]
    ages = [9, 20, 29, 43]  # MRCA ages (Mya); Mouse split = 87

    def age(a, b):
        s = {a, b}
        for cl, ag in zip(clades, ages):
            if s <= cl:
                return ag
        return 87
    return np.array([[2 * age(a, b) if a != b else 0 for b in sp] for a in sp])


def splits(Z, labels):
    n = len(labels)
    members = {i: {labels[i]} for i in range(n)}
    allset = frozenset(labels)
    ref = labels[0]
    S = set()
    for k, (a, b, *_) in enumerate(Z):
        a, b = int(a), int(b)
        members[n + k] = members[a] | members[b]
        s = members[n + k]
        if 2 <= len(s) <= n - 2:
            S.add(frozenset(s) if ref not in s else frozenset(allset - s))
    return S


def norm_rf(Z1, Z2, labels):
    rf = len(splits(Z1, labels) ^ splits(Z2, labels))
    n = len(labels)
    den = 2 * (n - 3)
    return rf / den if den > 0 else 0.0


def draw_tree(ax, Z, labels, title):
    dd = dendrogram(Z, labels=labels, no_plot=True)
    order = dd["ivl"]                              # leaf order (top -> bottom)
    pos = {lab: 5 + 10 * i for i, lab in enumerate(order)}
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):   # xs=positions, ys=distances
        x1, x2, x3, x4 = xs
        y1, y2, y3, y4 = ys
        ym = y2

        def col(p, d):
            return PALETTE[order[int((p - 5) // 10)]] if d == 0 else "0.45"
        ax.plot([y1, ym], [x1, x1], color=col(x1, y1), lw=2.2, solid_capstyle="round")
        ax.plot([y4, ym], [x4, x4], color=col(x4, y4), lw=2.2, solid_capstyle="round")
        ax.plot([ym, ym], [x1, x4], color="0.45", lw=2.2, solid_capstyle="round")
    ax.invert_xaxis()                              # root left, leaves (dist 0) right
    ax.set_xticks([])
    ax.set_yticks([pos[l] for l in order])
    ax.yaxis.tick_right()
    ax.set_yticklabels(order, fontsize=12)
    for t in ax.get_yticklabels():
        t.set_color(PALETTE[t.get_text()])
    for s in ["top", "left", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=18)


def panel_E(source, savedir):
    sp0 = SPECIES
    Ztrue = linkage(squareform(truth_patristic(sp0), checks=False), method="average")
    order_ct = list(reversed(CELLTYPES))
    fig, axes = plt.subplots(1, 1 + len(order_ct), figsize=(24, 4.6))
    draw_tree(axes[0], Ztrue, sp0, "TimeTree (truth)\nnRF=0.00  TreeCor=1.00")
    for ax, ct in zip(axes[1:], order_ct):
        D = pd.read_csv(f"{source}/distmat/cvae_homo_meanL1_{ct}.csv", index_col=0)
        sp = list(D.index)
        Z = linkage(squareform(D.values, checks=False), method="average")   # UPGMA
        rho = spearmanr(cophenet(Z), squareform(truth_patristic(sp), checks=False))[0]
        nrf = norm_rf(Z, linkage(squareform(truth_patristic(sp), checks=False),
                                 method="average"), sp)
        draw_tree(ax, Z, sp, f"{ct.capitalize()}\nnRF={nrf:.2f}  TreeCor={rho:.2f}")
    fig.suptitle("Per-cell-type species trees (homo, mean_L1, UPGMA, "
                 "cell-type-specific alignment)", y=1.04, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.55)
    _savefig(fig, savedir, "trees_homo_meanL1_UPGMA_percelltype")


PANELS = {"C": panel_C, "D": panel_D, "E": panel_E}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=["C", "D", "E", "all"], default="all")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--savedir", default=REGEN_DIR)
    args = ap.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    apply_font()
    for p in (["C", "D", "E"] if args.panel == "all" else [args.panel]):
        print(f"panel {p}:", flush=True)
        PANELS[p](args.source, args.savedir)


if __name__ == "__main__":
    main()
