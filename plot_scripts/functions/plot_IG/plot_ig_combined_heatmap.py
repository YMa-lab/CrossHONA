"""IG attribution combined heatmaps — Figure 3 panels A/B and Figure 4 panel D.

    Figure3 A : 1f_combined_heatmap_IG_pred.png   MERFISH_cortex, predicted class
    Figure3 B : 1f_combined_heatmap_IG_true.png   MERFISH_cortex, ground-truth class
    Figure4 D : 1f_combined_heatmap_IG_true.png   sc_small_intestine, ground-truth class

STAI-X_code variant of the two upstream `_replot_figures.py` scripts (one per
dataset, under biological_interpretability/run_script/results/<dataset>/).
Differences, all deliberate:

1. `--source` / `--savedir` instead of hardcoded `Path(__file__).parent`. The
   originals write back into the live results tree, overwriting the published
   PNGs in place.
2. Renders only the 1f viridis panels that are actual figure panels. The
   originals also re-render the 1f _diverging variants plus 1e/1g/1h, which need
   CSVs outside this source set.
3. Imports `plots.q1_plots` directly rather than
   `biological_interpretability.plots.q1_plots`, which would trigger
   `biological_interpretability/__init__.py` -> attribution_core -> torch.
   Verified byte-identical output.

The two datasets do NOT share cosmetics -- MERFISH uses font_scale 1.6 and an
uppercase cell-type rename, small intestine uses font_scale 1.8 and a
Capitalized rename. Those live in FIGURE_CONFIG below, copied verbatim from each
dataset's own _replot_figures.py; they move text extents and the figures save
with bbox_inches="tight", so they are load-bearing for pixel fidelity.

Inputs are the aggregated per-class CSVs, NOT the per-cell `attr_*.npy`
attributions. CPU only, seconds to run:

    python plot_ig_combined_heatmap.py --figure Figure3 --variant both
    python plot_ig_combined_heatmap.py --figure Figure4 --variant true
"""
import argparse
import os
import sys

import matplotlib as mpl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plots.q1_plots import plot_ig_combined_heatmap  # noqa: E402


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

# --- per-dataset cosmetics, verbatim from each upstream _replot_figures.py ----

# MERFISH_cortex: results/MERFISH_cortex/_replot_figures.py
_MERFISH_RENAME = {
    "exc": "EXC", "inc": "INC", "lasc": "ASC", "lmgc": "MGC",
    "logc": "OGC", "lopc": "POC", "oendo": "ENDO", "omural": "MURAL",
}


def _merfish_class_transform(c):
    s = str(c)
    return _MERFISH_RENAME.get(s.lower(), s.upper())


# sc_small_intestine: results/sc_small_intestine/_replot_figures.py
def _small_intestine_class_transform(c):
    s = str(c).replace("_", " ")
    return s[:1].upper() + s[1:].lower() if s else s


_DIVERGING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "ig_diverging_green_red", ["#285B4D", "#B93835"],
)

FIGURE_CONFIG = {
    "Figure3": dict(
        dataset="MERFISH_cortex",
        class_transform=_merfish_class_transform,
        font_scale=1.6,
        top_k_homo=30,
        # panel letters for the log line only
        panel_of={"pred": "A", "true": "B"},
    ),
    "Figure4": dict(
        dataset="sc_small_intestine",
        class_transform=_small_intestine_class_transform,
        font_scale=1.8,
        top_k_homo=30,
        panel_of={"true": "D"},
    ),
}

BRANCH_LABELS = ("Human", "Mouse")
TITLE_FONT = "Arial"   # not installed here; matplotlib falls back (upstream too)


def figure_paths(figure):
    d = os.path.join(PACKAGE_ROOT, "results", figure)
    return os.path.join(d, "source_csv"), os.path.join(d, "reproduced_figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=sorted(FIGURE_CONFIG), default="Figure3",
                    help="Figure3 = MERFISH_cortex (panels A/B), "
                         "Figure4 = sc_small_intestine (panel D).")
    ap.add_argument("--variant", choices=["pred", "true", "both"], default="both",
                    help="pred = attributions vs predicted class, true = vs ground truth.")
    ap.add_argument("--source", default=None,
                    help="Dir of cls_importance_*_IG{,_true}.csv (default: from --figure).")
    ap.add_argument("--savedir", default=None)
    ap.add_argument("--cmap", choices=["viridis", "diverging"], default="viridis",
                    help="Published panels use viridis.")
    ap.add_argument("--font_scale", type=float, default=None,
                    help="Override the figure's published font_scale.")
    args = ap.parse_args()

    cfg = FIGURE_CONFIG[args.figure]
    src_default, regen_default = figure_paths(args.figure)
    source = args.source or src_default
    savedir = args.savedir or regen_default
    font_scale = args.font_scale if args.font_scale is not None else cfg["font_scale"]
    os.makedirs(savedir, exist_ok=True)

    cmap_obj = "viridis" if args.cmap == "viridis" else _DIVERGING_CMAP
    cmap_suffix = "" if args.cmap == "viridis" else f"_{args.cmap}"
    variants = ["pred", "true"] if args.variant == "both" else [args.variant]

    for variant in variants:
        if variant not in cfg["panel_of"]:
            print(f"[{args.figure}] variant '{variant}' is not a panel of this figure "
                  f"— skipped", flush=True)
            continue
        suffix = "" if variant == "pred" else "_true"
        csvs = {k: os.path.join(source, f"cls_importance_{k}_IG{suffix}.csv")
                for k in ("ref_homo", "tgt_homo", "ref_nonhomo", "tgt_nonhomo")}
        missing = [v for v in csvs.values() if not os.path.isfile(v)]
        if missing:
            raise FileNotFoundError(f"{args.figure} {variant} missing inputs: {missing}")
        out = os.path.join(savedir, f"1f_combined_heatmap_IG_{variant}{cmap_suffix}")
        plot_ig_combined_heatmap(
            ref_homo_csv=csvs["ref_homo"],
            tgt_homo_csv=csvs["tgt_homo"],
            ref_nonhomo_csv=csvs["ref_nonhomo"],
            tgt_nonhomo_csv=csvs["tgt_nonhomo"],
            save_path=out,
            top_k_homo=cfg["top_k_homo"],
            branch_labels=BRANCH_LABELS,
            class_transform=cfg["class_transform"],
            font_scale=font_scale,
            cmap=cmap_obj,
            title_font=TITLE_FONT,
        )
        print(f"{args.figure} panel {cfg['panel_of'][variant]} ({variant}) "
              f"-> {out}.png / .pdf", flush=True)


if __name__ == "__main__":
    main()
