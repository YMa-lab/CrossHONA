"""Figure 5 panels A and B — CAMEX_testis importance-phylogeny figures.

    panel A : 4c_top3_per_celltype.png              top-3 genes per cell type
    panel B : 4a_distance_vs_mya_three_metrics.png  distance vs MYA, 3 metrics

Both render from the prebuilt importance tensor in results/Figure5/source_tensor/
(tensor.npy + species/cell_types/genes CSVs + summary.json, 43 KB total). That
tensor is verified bitwise identical to one rebuilt by the upstream
build_importance_tensor from the per-pair Q1 CSVs, so none of the 942 MB
CAMEX_testis tree -- nor the 3.3 MB of homolog TSVs -- is needed here. CPU only,
seconds.

    python plot_phylo_panels.py --panel all

--- Provenance caveats. Read before trusting a pixel diff. -------------------

panel A: upstream `phylogenetic/run.py` renders this via
plot_top_genes_per_celltype_grid. The PUBLISHED PNG IS STALE: it was written at
23:09 but `plots.py` was edited at 23:37, so the committed code no longer
reproduces it (current code gives 1616x772 vs the published 1618x772 -- a 2px
width difference). This script matches what the current upstream code produces,
which is the best available definition of correct.

panel B: THE PUBLISHED PNG HAS NO CALLER ANYWHERE IN THE REPO.
plot_distance_vs_mya_three_metrics_row is defined and re-exported but never
invoked; the figure came from an ad-hoc session whose arguments were never
recorded. The call below reconstructs them from evidence, not from source:
  - metric_titles      : read off the published figure's subplot titles
  - overall_rho_by_metric : summary.json's corr_with_TimeTree__* (0.597/0.471/
                         0.720 -> the published "0.60/0.47/0.72")
  - cell_type_colors   : recovered by sampling the published legend swatches and
                         inverting the alpha=0.92 blend over white
  - font_scale         : the function default (1.6); the true value is unknown.
                         No font_scale reproduces the published 4668x1700 (1.6
                         gives 4641x1669, 2.0 gives 4670x1685, 2.3 gives
                         4674x1701), so some other argument also differed.
So panel B reproduces the published figure's DATA and appearance but not its
exact rasterization. It is a faithful redraw, not a bit-for-bit reproduction.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phylo.distances import per_celltype_distances  # noqa: E402
from phylo.plots import (  # noqa: E402
    plot_distance_vs_mya_three_metrics_row,
    plot_top_genes_per_celltype_grid,
)
from phylo.reference_phylogeny import get_pairwise_mya  # noqa: E402
from phylo.tensor_builder import ImportanceTensor  # noqa: E402


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
DEFAULT_SOURCE = os.path.join(RESULTS_FIG5, "source_tensor")
REGEN_DIR = os.path.join(RESULTS_FIG5, "reproduced_figures")

# --- upstream run parameters (from run_script/results/CAMEX_testis/_run.py) ---
DATASET = "CAMEX_testis"
METHOD = "IG"
ANCHOR = "Human"
TOP_K_JACCARD = 50          # run.py default; the published panel B title says K=50

# --- panel B arguments reconstructed from the published figure (see docstring) -
METRIC_TITLES = {
    "spearman": "Rank-based (1 - Spearman)",
    "topk_jaccard": f"Top-K Jaccard (K={TOP_K_JACCARD})",
    "l1": "Magnitude-based (L1)",
}
CELL_TYPE_COLORS = ["#8CB369", "#F4E285", "#F4A259", "#5B8E7D", "#BC4B51"]


def load_tensor(source):
    """Rebuild the ImportanceTensor from the small prebuilt exports."""
    return ImportanceTensor(
        tensor=np.load(os.path.join(source, "tensor.npy")),
        species=pd.read_csv(os.path.join(source, "species.csv"))["species"].tolist(),
        cell_types=pd.read_csv(os.path.join(source, "cell_types.csv"))["cell_type"].tolist(),
        genes=pd.read_csv(os.path.join(source, "genes.csv"))["gene"].tolist(),
        method=METHOD,
    )


def panel_A(it, source, savedir):
    """4c_top3_per_celltype — mirrors run.py's call exactly."""
    out = os.path.join(savedir, "4c_top3_per_celltype")
    plot_top_genes_per_celltype_grid(
        it,
        save_path=out,
        top_n=3,
        anchor=ANCHOR,
        reference_lookup=get_pairwise_mya,
        title=f"{DATASET} ({METHOD}) — top-3 genes per cell type",
    )
    print(f"  panel A -> {out}.png / .pdf", flush=True)


def panel_B(it, source, savedir, font_scale=1.6):
    """4a_distance_vs_mya_three_metrics — reconstructed call, see docstring."""
    summary = json.load(open(os.path.join(source, "summary.json")))
    pcs = {
        "spearman": per_celltype_distances(it, metric="spearman"),
        "topk_jaccard": per_celltype_distances(it, metric="topk_jaccard", k=TOP_K_JACCARD),
        "l1": per_celltype_distances(it, metric="l1"),
    }
    rho = {m: summary[f"corr_with_TimeTree__{m}"] for m in pcs}
    out = os.path.join(savedir, "4a_distance_vs_mya_three_metrics")
    plot_distance_vs_mya_three_metrics_row(
        pcs, rho, it.species, get_pairwise_mya,
        save_path=out,
        anchor=ANCHOR,
        metric_titles=METRIC_TITLES,
        cell_type_colors=CELL_TYPE_COLORS,
        font_scale=font_scale,
    )
    print(f"  panel B -> {out}.png / .pdf", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=["A", "B", "all"], default="all")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--savedir", default=REGEN_DIR)
    ap.add_argument("--font_scale", type=float, default=1.6,
                    help="panel B only; the published figure's true value is unknown.")
    args = ap.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    it = load_tensor(args.source)
    print(f"tensor {it.tensor.shape}  species={it.species}", flush=True)
    print(f"cell_types={it.cell_types}  genes={len(it.genes)}", flush=True)
    if args.panel in ("A", "all"):
        panel_A(it, args.source, args.savedir)
    if args.panel in ("B", "all"):
        panel_B(it, args.source, args.savedir, font_scale=args.font_scale)


if __name__ == "__main__":
    main()
