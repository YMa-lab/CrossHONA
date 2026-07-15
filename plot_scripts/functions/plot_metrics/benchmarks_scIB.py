import argparse
import glob
import numpy as np
import os
import torch
import pickle
from plot import *
from metrics import *
import random
import time
import tempfile
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import matplotlib as mpl
from rich import print
import anndata as ad
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

import matplotlib.pyplot as _plt
import matplotlib.font_manager as _fm
import matplotlib as mpl
_FONT_PATH = "/oscar/data/yma16/Project/spTransform/2.Downstream/Arial Unicode.ttf"
if os.path.isfile(_FONT_PATH):
    _fm.fontManager.addfont(_FONT_PATH)
    _plt.rcParams["font.family"] = _fm.FontProperties(fname=_FONT_PATH).get_name()


def patch_kbet_skip_bug():
    """Fix inverted skip condition in scib_metrics.kbet_per_label.

    In scib_metrics<=0.5.4 the skip selector is inverted: clusters with
    >10 cells are skipped (NaN) and only clusters with <=10 cells are
    scored. Result: every kBET value is NaN on normal-sized datasets.
    See site-packages/scib_metrics/metrics/_kbet.py around line 143.

    This rebinds scib_metrics.kbet_per_label (and the module-level binding
    that Benchmarker.benchmark() resolves via getattr) to a fixed copy.
    """
    import inspect, scib_metrics
    from scib_metrics.metrics import _kbet as _kbet_mod

    src = inspect.getsource(_kbet_mod.kbet_per_label)
    if "skipped = clusters[counts > 10]" not in src:
        return  # already fixed upstream
    fixed_src = src.replace(
        "skipped = clusters[counts > 10]\n    clusters = clusters[counts <= 10]",
        "skipped = clusters[counts <= 10]\n    clusters = clusters[counts > 10]",
    )
    ns = dict(_kbet_mod.__dict__)
    exec(fixed_src, ns)
    fixed_fn = ns["kbet_per_label"]
    _kbet_mod.kbet_per_label = fixed_fn
    scib_metrics.kbet_per_label = fixed_fn  # Benchmarker resolves via getattr(scib_metrics, name)
    print("[patch] kbet_per_label skip-condition fix applied", flush=True)


patch_kbet_skip_bug()


# Map a canonical dataset name (e.g. CAMEX_cortex) to candidates used in
# scripts_benchmarks_new/result for scGen / CAMEX (lowercased / aliased).
DATASET_ALIASES = {
    "CAMEX_cortex":        ["CAMEX_cortex", "camex_cortex"],
    "CAMEX_testis":        ["CAMEX_testis", "camex_testis"],
    "MERFISH_Cortex":      ["MERFISH_Cortex", "merfish", "merfish_first"],
    "MERFISH_HMBA":        ["MERFISH_HMBA", "hmba"],
    "scMultispecies_liver":["scMultispecies_liver", "scMultiSpecies"],
    "sc_heart":            ["sc_heart", "heart"],
    # cross_order Pig runs were stored by scGen/CAMEX under a single_cell/pig_ref/
    # pig_ref_<species2> nested layout (see gather_results.ipynb CAMEX_PIG_REMAP).
    "cross_order":         ["cross_order", "single_cell"],
}


def _candidate_pair_dirs(base, method, dataset, species1, species2):
    """Yield possible <base>/<method>/<dataset>/<species_pair> directories.

    Handles two layouts:
      - flat:    <base>/<method>/<dataset>/<species_pair>
      - nested:  <base>/<method>/<dataset>/<species1>_ref/<species1>_ref_<species2>
        (used by scGen/scMultiSpecies — see gather_results.ipynb)
    """
    s1, s2 = species1.lower(), species2.lower()
    pair_variants = [f"{species1}_{species2}", f"{s1}_{s2}"]
    ds_variants = DATASET_ALIASES.get(dataset, [dataset])
    for ds in ds_variants:
        for pair in pair_variants:
            yield os.path.join(base, method, ds, pair)
        # scGen/scMultiSpecies-style nested layout
        yield os.path.join(base, method, ds, f"{s1}_ref", f"{s1}_ref_{s2}")


def find_method_dir(base, method, dataset, species1, species2):
    for p in _candidate_pair_dirs(base, method, dataset, species1, species2):
        if os.path.isdir(p):
            return p
    return None


# ---------------------------------------------------------------------------
# CVAE_proto "ours" layout (current source of truth for our pipeline output).
# Pattern A — flat: <CVAE_proto>/<folder>/<variant_folder>/
#   variant_folder = "<prefix>_NonHomo" / "<prefix>_noNonHomo" when prefix is set,
#                    else "NonHomo" / "noNonHomo".
# Pattern B — paired: <CVAE_proto>/<dataset>/<species1>_<species2>/<variant>/
# ---------------------------------------------------------------------------
OURS_CVAE_ROOT = "/oscar/data/yma16/Project/Cross_species/01_R_Working/CVAE_proto"
OURS_CVAE_FLAT = {
    # canonical dataset -> (folder under CVAE_proto, variant prefix)
    "MERFISH_Cortex":      ("02_results_MERFISH",              "Merfish"),
    "MERFISH_HMBA":        ("02_results_HMBA",                 ""),
    "sc_small_intestine":  ("02_results_small_intestine",      ""),
    "white_adipose":       ("02_results_white_adipose_tissue", ""),
}
OURS_CVAE_PAIRED = {"CAMEX_cortex", "CAMEX_testis"}


def find_ours_cvae_paths(dataset, species1, species2, variant="NonHomo"):
    """Resolve the CVAE_proto 'ours' directories for a (dataset, species_pair, variant).

    Returns (parent_dir, variant_dir, stage_dir) where:
        parent_dir  — holds adata_{ref,target}_{homo,nonhomo}{,_preprocessed}.h5ad
                      and inverse_dict_{ref,target}.pkl
        variant_dir — holds {ref,target}_pred.npy, {ref,target}_gt.npy
        stage_dir   — variant_dir / 'Stage3_BridgedFull', holds {ref,tgt}_homo_mean.npy

    Returns None if the dataset is not registered or variant_dir doesn't exist.
    """
    if dataset in OURS_CVAE_FLAT:
        folder, prefix = OURS_CVAE_FLAT[dataset]
        parent = os.path.join(OURS_CVAE_ROOT, folder)
        vfolder = f"{prefix}_{variant}" if prefix else variant
    elif dataset in OURS_CVAE_PAIRED:
        parent = os.path.join(OURS_CVAE_ROOT, dataset, f"{species1}_{species2}")
        vfolder = variant
    elif dataset == "cross_order":
        # cross_order has one CVAE_proto folder per species pair, e.g.
        # 02_results_Pig_Human / 02_results_Pig_MacaF / 02_results_Pig_MacaqueM,
        # each holding plain NonHomo / noNonHomo variant dirs (no prefix).
        parent = os.path.join(OURS_CVAE_ROOT, f"02_results_{species1}_{species2}")
        vfolder = variant
    else:
        return None
    variant_dir = os.path.join(parent, vfolder)
    if not os.path.isdir(variant_dir):
        return None
    return parent, variant_dir, os.path.join(variant_dir, "Stage3_BridgedFull")


def save_scib_table(bm, savedir, filename, min_max_scale=True, drop=None,
                    font_scale=1.4, width_scale=0.75):
    """Render bm.plot_results_table to disk, optionally dropping embeddings.

    `filename` may be given with or without an extension; the figure is saved
    as both PNG and PDF (extensions are appended/stripped automatically).

    Drops a set of embedding names (e.g. {"SATURN", "Unintegrated"}) by
    temporarily mutating bm._embedding_obsm_keys and bm._results, then
    restoring them — so min-max rescaling is computed over the kept embeddings
    only.

    `font_scale` multiplies every Text artist's fontsize after the figure is
    drawn (scib_metrics' Table hardcodes textprops fontsize=10). `width_scale`
    multiplies the figure width (height is left unchanged) so the table can
    be made narrower without losing aspect on the rows.
    """
    import copy
    import matplotlib.pyplot as plt

    drop = set(drop or [])
    keep_keys = [k for k in bm._embedding_obsm_keys if k not in drop]
    if drop:
        # Stash & swap. _results columns are embeddings + the _METRIC_TYPE column.
        orig_keys = bm._embedding_obsm_keys
        orig_results = bm._results
        bm._embedding_obsm_keys = keep_keys
        bm._results = orig_results.drop(columns=[c for c in drop if c in orig_results.columns])

    try:
        try:
            tab = bm.plot_results_table(min_max_scale=min_max_scale, save_dir=None, show=False)
        except TypeError:
            tab = bm.plot_results_table(min_max_scale=min_max_scale)
        fig = getattr(tab, "figure", None) or getattr(tab, "fig", None)
        if fig is not None:
            # Narrower & taller text
            w, h = fig.get_size_inches()
            fig.set_size_inches(w * width_scale, h)
            for t in fig.findobj(plt.Text):
                t.set_fontsize(t.get_fontsize() * font_scale)
            base, _ = os.path.splitext(filename)
            for ext in ("png", "pdf"):
                p = os.path.join(savedir, f"{base}.{ext}")
                fig.savefig(p, dpi=300, bbox_inches="tight")
                print(f"[plot] saved {os.path.basename(p)}", flush=True)
    finally:
        if drop:
            bm._embedding_obsm_keys = orig_keys
            bm._results = orig_results


def add_embedding(adata, key, src_adata, src_obsm_key=None, use_X=False):
    """Restrict adata to cells shared with src_adata and copy the embedding."""
    src_adata.obs_names_make_unique()
    shared = adata.obs_names.intersection(src_adata.obs_names)
    if len(shared) == 0:
        print(f"[{key}] no overlapping cells — skipped", flush=True)
        return adata
    adata = adata[shared].copy()
    src_adata = src_adata[shared].copy()
    emb = src_adata.X if use_X else src_adata.obsm[src_obsm_key]
    if hasattr(emb, "toarray"):
        emb = emb.toarray()
    adata.obsm[key] = np.asarray(emb)
    print(f"[{key}] embedding loaded, shape={adata.obsm[key].shape}", flush=True)
    return adata

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():
    # laod the argument inputs
    parser = argparse.ArgumentParser(description='benchmarks_scVI')
    parser.add_argument('--savedir', type=str, default='/oscar/data/yma16/Project/Cross_species/results_benchmarks/summary')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--species1', type=str, required=True)
    parser.add_argument('--species2', type=str, required=True)
    parser.add_argument('--cell_col', type=str, default='cell_type')
    parser.add_argument('--batch_col', type=str, default='species')
    parser.add_argument('--plot_only', action='store_true',
                        help='Skip the benchmark; load benchmarker.pkl from --savedir and re-render the plots/CSVs.')
    parser.add_argument('--font_scale', type=float, default=1.4,
                        help='Multiplies every Text artist\'s fontsize in the scib table.')
    parser.add_argument('--width_scale', type=float, default=0.75,
                        help='Multiplies the scib table figure width (height unchanged).')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Seed the run and create saving directory
    set_seed(42)
    
    start_time = time.time()
    args.savedir = os.path.join(args.savedir, args.dataset, f'{args.species1}_{args.species2}')
    os.makedirs(args.savedir, exist_ok=True)
    print(f"The saving directory set to {args.savedir}", flush=True)

    # ------------------------------------------------------------------
    # Plot-only fast path: skip the benchmark, just reload the saved
    # Benchmarker and re-render the four tables (and re-save the CSVs).
    # ------------------------------------------------------------------
    if args.plot_only:
        pkl = os.path.join(args.savedir, "benchmarker.pkl")
        if not os.path.isfile(pkl):
            raise FileNotFoundError(f"--plot_only set but {pkl} doesn't exist; run benchmark first.")
        print(f"[plot_only] loading {pkl}", flush=True)
        with open(pkl, "rb") as f:
            bm = pickle.load(f)
        bm.get_results(min_max_scale=True).to_csv(f"{args.savedir}/scib_metrics.csv")
        bm.get_results(min_max_scale=False).to_csv(f"{args.savedir}/scib_metrics_raw.csv")
        # Always drop Unintegrated from the plotted tables — it's a baseline ref, not a method.
        DROP = {"Unintegrated"}
        with mpl.rc_context({"font.size": 10}):
            save_scib_table(bm, args.savedir, "scib_table.png",     min_max_scale=True,  drop=DROP, font_scale=args.font_scale, width_scale=args.width_scale)
            save_scib_table(bm, args.savedir, "scib_table_raw.png", min_max_scale=False, drop=DROP, font_scale=args.font_scale, width_scale=args.width_scale)
        print("--- %s seconds ---" % (time.time() - start_time), flush=True)
        return

    # baseline skeleton (obs/var only; no Unintegrated PCA embedding plotted)
    adata = sc.read_h5ad(f"/oscar/data/yma16/Project/Cross_species/results_benchmarks/scVI/{args.dataset}/{args.species1}_{args.species2}/adata_all_homo_preprocessed.h5ad")
    adata.obs_names_make_unique()
    print(adata.shape)

    # scVI results
    adata_scVI = sc.read_h5ad(f"/oscar/data/yma16/Project/Cross_species/results_benchmarks/scVI/{args.dataset}/{args.species1}_{args.species2}/final_adata.h5ad")
    adata_scVI.obs_names_make_unique()
    shared = adata.obs_names.intersection(adata_scVI.obs_names)
    adata = adata[shared].copy()
    adata_scVI = adata_scVI[shared].copy()
    adata.obsm["scVI"] = adata_scVI.obsm["X_scVI"]

    embedding_keys = ["scVI"]

    LOCAL_RESULTS = "/oscar/data/yma16/Project/Cross_species/scripts_benchmarks_new/result"
    SHARED_RESULTS = "/oscar/data/yma16/Project/Cross_species/results_benchmarks"

    # --- previous scGen (final_adata + corrected_latent) — kept for recovery ---
    # sg_dir = find_method_dir(LOCAL_RESULTS, "scGen", args.dataset, args.species1, args.species2)
    # sg_path = os.path.join(sg_dir, "final_adata.h5ad") if sg_dir else None
    # if sg_path and os.path.isfile(sg_path):
    #     adata_scGen = sc.read_h5ad(sg_path)
    #     adata_scGen.obs_names = adata_scGen.obs_names.str.replace(
    #         r"-(reference|target)$", "", regex=True
    #     )
    #     adata = add_embedding(adata, "scGen", adata_scGen, src_obsm_key="corrected_latent")
    #     if "scGen" in adata.obsm:
    #         embedding_keys.append("scGen")
    # else:
    #     print(f"[scGen] not found under {sg_dir}", flush=True)
    # ---------------------------------------------------------------------------

    # scGen_leiden results
    # scripts_benchmarks_new/result/scGen_leiden/<dataset>/<species_pair>/final_adata.h5ad
    # → obsm['corrected_latent'], obs['predicted_cell_type']
    # run_scGen*.py concat'd ref+target with `index_unique="-"`, appending
    # "-reference" / "-target" to every obs_name. Strip that to match baseline.
    sg_dir = find_method_dir(LOCAL_RESULTS, "scGen_leiden", args.dataset, args.species1, args.species2)
    sg_path = os.path.join(sg_dir, "final_adata.h5ad") if sg_dir else None
    if sg_path and os.path.isfile(sg_path):
        adata_scGen = sc.read_h5ad(sg_path)
        adata_scGen.obs_names = adata_scGen.obs_names.str.replace(
            r"-(reference|target)$", "", regex=True
        )
        if "corrected_latent" not in adata_scGen.obsm:
            print(f"[scGen_leiden] skipped — obsm['corrected_latent'] not in {sg_path} "
                  f"(available: {list(adata_scGen.obsm.keys())})", flush=True)
        else:
            adata = add_embedding(adata, "scGen", adata_scGen, src_obsm_key="corrected_latent")
            if "scGen" in adata.obsm:
                embedding_keys.append("scGen")
    else:
        print(f"[scGen_leiden] not found (looking for final_adata.h5ad under {sg_dir})", flush=True)

    # SATURN excluded from scib benchmarks — block kept for reference.
    # # SATURN results
    # # results_benchmarks/SATURN/<dataset>/<species_pair>/saturn_results/<run>.h5ad
    # # SATURN stores the integrated embedding directly as adata.X
    # sa_dir = find_method_dir(SHARED_RESULTS, "SATURN_map", args.dataset, args.species1, args.species2)
    # sa_path = None
    # if sa_dir:
    #     cands = [
    #         f for f in glob.glob(os.path.join(sa_dir, "saturn_results", "*.h5ad"))
    #         if "_pretrain" not in os.path.basename(f) and "_ep_" not in os.path.basename(f)
    #     ]
    #     if cands:
    #         sa_path = cands[0]
    # if sa_path and os.path.isfile(sa_path):
    #     adata_SATURN = sc.read_h5ad(sa_path)
    #     adata = add_embedding(adata, "SATURN", adata_SATURN, use_X=True)
    #     if "SATURN" in adata.obsm:
    #         embedding_keys.append("SATURN")
    # else:
    #     print(f"[SATURN] not found under {sa_dir}", flush=True)

    # CAMEX results
    # scripts_benchmarks_new/result/CAMEX/<dataset>/<species_pair>/log/adata_CAMEX.h5ad
    # → obsm["X_CAMEX_Integration"]
    cm_dir = find_method_dir(LOCAL_RESULTS, "CAMEX", args.dataset, args.species1, args.species2)
    cm_path = os.path.join(cm_dir, "log", "adata_CAMEX.h5ad") if cm_dir else None
    if cm_path and os.path.isfile(cm_path):
        adata_CAMEX = sc.read_h5ad(cm_path)
        adata = add_embedding(adata, "CAMEX", adata_CAMEX, src_obsm_key="X_CAMEX_Integration")
        if "CAMEX" in adata.obsm:
            embedding_keys.append("CAMEX")
    else:
        print(f"[CAMEX] not found under {cm_dir}", flush=True)

    # NicheFormer results
    # results_benchmarks/nicheformer/<dataset>/<species_pair>/final_adata.h5ad
    # → obsm["X_nicheformer"]
    nf_dir = find_method_dir(SHARED_RESULTS, "nicheformer", args.dataset, args.species1, args.species2)
    nf_path = os.path.join(nf_dir, "final_adata.h5ad") if nf_dir else None
    if nf_path and os.path.isfile(nf_path):
        adata_NF = sc.read_h5ad(nf_path)
        adata = add_embedding(adata, "NicheFormer", adata_NF, src_obsm_key="X_nicheformer")
        if "NicheFormer" in adata.obsm:
            embedding_keys.append("NicheFormer")
    else:
        print(f"[NicheFormer] not found under {nf_dir}", flush=True)

    # proposed methods results — CVAE_proto, Stage 3 embeddings
    # <variant_dir>/Stage3_BridgedFull/{ref,tgt}_homo_mean.npy
    # The .npy files have no obs_names; use the preprocessed h5ads at <parent>/
    # (adata_{ref,target}_homo_preprocessed.h5ad) to assign obs_names in the same
    # row order. Concatenate ref+tgt vertically for the shared-latent integration.
    ours_paths = find_ours_cvae_paths(args.dataset, args.species1, args.species2)
    if ours_paths:
        parent_dir, _variant_dir, stage_dir = ours_paths
        ref_npy = os.path.join(stage_dir, "ref_homo_mean.npy")
        tgt_npy = os.path.join(stage_dir, "tgt_homo_mean.npy")
        ref_h5  = os.path.join(parent_dir, "adata_ref_homo_preprocessed.h5ad")
        tgt_h5  = os.path.join(parent_dir, "adata_target_homo_preprocessed.h5ad")
        if all(os.path.isfile(p) for p in [ref_npy, tgt_npy, ref_h5, tgt_h5]):
            ref_emb = np.load(ref_npy)
            tgt_emb = np.load(tgt_npy)
            adata_ref_pp = sc.read_h5ad(ref_h5)
            adata_tgt_pp = sc.read_h5ad(tgt_h5)
            adata_ref_pp.obs_names_make_unique()
            adata_tgt_pp.obs_names_make_unique()
            if ref_emb.shape[0] != adata_ref_pp.n_obs or tgt_emb.shape[0] != adata_tgt_pp.n_obs:
                print(
                    f"[Ours] embedding row count != preprocessed n_obs "
                    f"(ref {ref_emb.shape[0]} vs {adata_ref_pp.n_obs}, "
                    f"tgt {tgt_emb.shape[0]} vs {adata_tgt_pp.n_obs}) — skipped",
                    flush=True,
                )
            else:
                # `ad.concat(..., axis=0, join="outer")` reindexes var (genes), which
                # fails when either adata has duplicate var_names. We only need obs —
                # build it directly from the two DataFrames.
                adata_ours = ad.AnnData(
                    X=np.vstack([ref_emb, tgt_emb]),
                    obs=pd.concat([adata_ref_pp.obs, adata_tgt_pp.obs], axis=0, join="outer"),
                )
                adata_ours.obs_names = list(adata_ref_pp.obs_names) + list(adata_tgt_pp.obs_names)
                adata_ours.obsm["CrossHONA"] = adata_ours.X.copy()
                adata = add_embedding(adata, "CrossHONA", adata_ours, src_obsm_key="CrossHONA")
                if "CrossHONA" in adata.obsm:
                    embedding_keys.append("CrossHONA")
        else:
            print(f"[Ours] expected files missing under {stage_dir}", flush=True)
    else:
        print(f"[Ours] CVAE_proto layout not registered for dataset={args.dataset} "
              f"(see OURS_CVAE_FLAT/OURS_CVAE_PAIRED)", flush=True)

    print(f"Final adata: {adata.shape}, obsm keys: {embedding_keys}", flush=True)

    # scIB benchmarks
    bm = Benchmarker(
        adata,
        batch_key=args.batch_col,
        label_key=args.cell_col,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=embedding_keys,
        n_jobs=2,
    )
    bm.benchmark()
    # After bm.benchmark() — save it
    with open(f"{args.savedir}/benchmarker.pkl", "wb") as f:
        pickle.dump(bm, f)

    # Default scib_metrics output is min-max scaled across embeddings — useful
    # for ranking but not the absolute metric values. Save both.
    bm.get_results(min_max_scale=True).to_csv(f"{args.savedir}/scib_metrics.csv")
    bm.get_results(min_max_scale=False).to_csv(f"{args.savedir}/scib_metrics_raw.csv")

    # Plot two table variants:
    #   scib_table.png      — min-max scaled across embeddings
    #   scib_table_raw.png  — raw absolute metric values
    # Unintegrated is excluded from plotting (it isn't a method).
    DROP = {"Unintegrated"}
    # Pin font.size so plottable's bar() annotation in the aggregate columns
    # matches the table's textprops (scib_metrics doesn't forward textprops to
    # plot_fn, so the bar text otherwise picks up rcParams["font.size"]).
    with mpl.rc_context({"font.size": 10}):
        save_scib_table(bm, args.savedir, "scib_table.png",     min_max_scale=True,  drop=DROP, font_scale=args.font_scale, width_scale=args.width_scale)
        save_scib_table(bm, args.savedir, "scib_table_raw.png", min_max_scale=False, drop=DROP, font_scale=args.font_scale, width_scale=args.width_scale)

    print("--- %s seconds ---" % (time.time() - start_time),flush=True)
         
if __name__ == '__main__':
    main()
