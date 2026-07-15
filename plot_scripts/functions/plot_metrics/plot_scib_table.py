"""Figure 2 panel b — scIB table for CrossHONA's three spaces + 4 other methods.

STAI-X_code variant of benchmarks_scIB_crosshona_vs_methods.py: the seven
embeddings come straight from the prebuilt `figure2_source.h5ad`, so none of the
original per-method h5ads or the CVAE_proto .npy tree are needed.

The published table was rendered through replot_from_pkl (--plot_only), which
renames CrossHONA_* to two-line "CrossHONA\\n<variant>" labels -- the published
scib_metrics.csv carries those names. run_benchmark alone does NOT rename, so
the rename is applied here explicitly to match.

Must run on a GPU node via sbatch: scib_metrics' numba kernels fail on the login
node. See each figure's sbatch/figure*_panelB_scib_table.sh.

    python plot_scib_table.py                      # benchmark + render
    python plot_scib_table.py --plot_only          # re-render from benchmarker.pkl
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import matplotlib as mpl
import scanpy as sc

# save_scib_table lives in the original benchmark module (pulls torch + scib_metrics).
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmarks_scIB import save_scib_table  # noqa: E402
from metrics_source import figure_paths  # noqa: E402

from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation  # noqa: E402

# Column order as entered in the published run: four baselines, then CrossHONA's spaces.
EMBEDDING_KEYS = ["scVI", "scGen", "NicheFormer", "CAMEX",
                  "CrossHONA_homo", "CrossHONA_nonhomo", "CrossHONA_combined"]
DROP = {"Unintegrated"}


def _rename_crosshona(bm):
    """CrossHONA_homo -> 'CrossHONA\\nhomo', matching replot_from_pkl."""
    rename = {k: "CrossHONA\n" + k[len("CrossHONA_"):]
              for k in list(bm._embedding_obsm_keys) if k.startswith("CrossHONA_")}
    if rename:
        bm._embedding_obsm_keys = [rename.get(k, k) for k in bm._embedding_obsm_keys]
        bm._results = bm._results.rename(columns=rename)
    return bm


def _render(bm, savedir, font_scale, width_scale):
    """Write the min-max-scaled table only.

    Upstream also emitted the un-scaled twins (scib_metrics_raw.csv,
    scib_table_raw.{png,pdf}). Neither is a figure panel -- the published panel
    is the min-max-scaled table -- so they are not produced here.
    """
    bm.get_results(min_max_scale=True).to_csv(f"{savedir}/scib_metrics.csv")
    with mpl.rc_context({"font.size": 10}):
        save_scib_table(bm, savedir, "scib_table.png", min_max_scale=True,
                        drop=DROP, font_scale=font_scale, width_scale=width_scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", default="Figure2",
                    help="Which figure in this package (Figure2 = MERFISH_Cortex, "
                         "Figure4 = sc_small_intestine). Sets --source/--savedir defaults.")
    ap.add_argument("--source", default=None)
    ap.add_argument("--savedir", default=None)
    ap.add_argument("--cell_col", default="cell_type")
    ap.add_argument("--batch_col", default="species")
    ap.add_argument("--font_scale", type=float, default=1.4)
    ap.add_argument("--width_scale", type=float, default=0.75)
    ap.add_argument("--plot_only", action="store_true",
                    help="Skip the benchmark; reload benchmarker.pkl from savedir.")
    args = ap.parse_args()

    start = time.time()
    src_default, regen_default = figure_paths(args.figure)
    source = args.source or src_default
    args.savedir = args.savedir or regen_default
    os.makedirs(args.savedir, exist_ok=True)

    if args.plot_only:
        pkl = os.path.join(args.savedir, "benchmarker.pkl")
        if not os.path.isfile(pkl):
            raise FileNotFoundError(f"--plot_only set but {pkl} doesn't exist.")
        with open(pkl, "rb") as f:
            bm = pickle.load(f)
        # benchmarker.pkl is saved pre-rename (keys still CrossHONA_*), so the
        # rename has to happen here too -- otherwise the re-rendered table reads
        # "CrossHONA_homo" instead of the published two-line label. Idempotent:
        # renamed keys no longer start with "CrossHONA_".
        bm = _rename_crosshona(bm)
        _render(bm, args.savedir, args.font_scale, args.width_scale)
        print("--- %s seconds ---" % (time.time() - start), flush=True)
        return

    adata = sc.read_h5ad(source)
    keys = [k for k in EMBEDDING_KEYS if k in adata.obsm]
    missing = [k for k in EMBEDDING_KEYS if k not in adata.obsm]
    if missing:
        print(f"[warn] absent from source, excluded from the table: {missing}", flush=True)
    if not keys:
        raise SystemExit("no embeddings found in source file")
    print(f"source {adata.shape}, embeddings: {keys}", flush=True)

    bm = Benchmarker(
        adata,
        batch_key=args.batch_col,
        label_key=args.cell_col,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=keys,
        n_jobs=2,
    )
    bm.benchmark()
    with open(f"{args.savedir}/benchmarker.pkl", "wb") as f:
        pickle.dump(bm, f)
    bm = _rename_crosshona(bm)
    _render(bm, args.savedir, args.font_scale, args.width_scale)
    print("--- %s seconds ---" % (time.time() - start), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
