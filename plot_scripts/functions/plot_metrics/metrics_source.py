"""Loader shim: serve the metrics panels from a prebuilt source h5ad.

Drop-in replacement for the parts of `plot_umaps_all_methods.py` that the
Figure 2 panel scripts import (`LOADERS`, `METHOD_ORDER`, `relabel_cell_types`),
but reading the single prebuilt source file instead of the five original
per-method h5ads.

Deliberately self-contained: the original chain pulls in `benchmarks_scIB`,
which imports torch and scib_metrics. Panels a and c only need labels, so this
module keeps them to numpy/pandas/anndata and they run anywhere.

`LOADERS[m](...)` returns the same AnnData contract as the original loaders:
    obsm["emb"]      — integration embedding
    obs["species"]   — species names
    obs["cell_type"] — ground truth, lowercased
    obs["pred"]      — prediction, lowercased
Labels are stored pre-relabel, so `relabel_cell_types` still applies on top
exactly as in the published run.

Paths are anchored by walking up to the STAI-X_code root rather than counting
"../.." hops, so moving these modules between plot_scripts/functions/* does not
break them. `--source` / `FIGURE2_SOURCE` override the default.
"""
import os

import anndata as ad
import numpy as np


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


def figure_paths(figure="Figure2"):
    """(source_h5ad, reproduced_figures_dir) for a figure in this package.

    Figure2 -> results/Figure2/figure2_source.h5ad   (MERFISH_Cortex)
    Figure4 -> results/Figure4/figure4_source.h5ad   (sc_small_intestine)
    Both figures run the same panels off the same code; only the dataset and
    the label column differ.
    """
    d = os.path.join(PACKAGE_ROOT, "results", figure)
    return (os.path.join(d, f"{figure.lower()}_source.h5ad"),
            os.path.join(d, "reproduced_figures"))


# Back-compat defaults (Figure 2). Prefer figure_paths(args.figure).
DEFAULT_SOURCE, REGEN_DIR = figure_paths("Figure2")

METHOD_ORDER = ["CrossHONA", "scVI", "scGen", "NicheFormer", "CAMEX"]

# obsm key in figure2_source.h5ad carrying each method's embedding.
# CrossHONA's UMAP/confusion loader used the homo space.
_EMB_KEY = {
    "CrossHONA": "CrossHONA_homo",
    "scVI": "scVI",
    "scGen": "scGen",
    "NicheFormer": "NicheFormer",
    "CAMEX": "CAMEX",
}

_CACHE = {}


def load_source(path=None):
    """Read (and memoize) the figure's source h5ad."""
    p = path or os.environ.get("FIGURE2_SOURCE") or DEFAULT_SOURCE
    if p not in _CACHE:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"source h5ad not found: {p}")
        _CACHE[p] = ad.read_h5ad(p)
    return _CACHE[p]


def _make_loader(method):
    def _loader(dataset, species1, species2, cell_col, batch_col, source=None):
        src = load_source(source)
        need = [f"pred_{method}", f"celltype_{method}", f"species_{method}"]
        if any(c not in src.obs.columns for c in need):
            print(f"[{method}] columns missing from source file — skipped", flush=True)
            return None
        emb_key = _EMB_KEY[method]
        emb = src.obsm[emb_key] if emb_key in src.obsm else np.zeros((src.n_obs, 1), np.float32)
        out = ad.AnnData(X=np.zeros((src.n_obs, 1), dtype=np.float32))
        out.obsm["emb"] = np.asarray(emb)
        out.obs["species"] = src.obs[f"species_{method}"].astype(str).values
        out.obs["cell_type"] = src.obs[f"celltype_{method}"].astype(str).values
        out.obs["pred"] = src.obs[f"pred_{method}"].astype(str).values
        return out
    return _loader


LOADERS = {m: _make_loader(m) for m in METHOD_ORDER}


# --- per-dataset cell-type label rewrites (verbatim from plot_umaps_all_methods) ---
_MERFISH_CORTEX_RENAME = {
    "LASC": "ASC", "LMGC": "MGC", "LOGC": "OGC", "LOPC": "POC",
    "OENDO": "ENDO", "OMURAL": "MURAL",
}


def _merfish_cortex_label(s):
    u = str(s).upper()
    return _MERFISH_CORTEX_RENAME.get(u, u)


def _small_intestine_label(s):
    return str(s).replace("_", " ").capitalize()


DATASET_CELLTYPE_LABEL_MAP = {
    "merfish_cortex": _merfish_cortex_label,
    "sc_small_intestine": _small_intestine_label,
}


def relabel_cell_types(adata, dataset):
    fn = DATASET_CELLTYPE_LABEL_MAP.get(dataset.lower())
    if fn is None:
        return
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str).map(fn)
    adata.obs["pred"] = adata.obs["pred"].astype(str).map(fn)


def apply_style(font_path="/oscar/data/yma16/Project/spTransform/2.Downstream/Arial Unicode.ttf"):
    """Reproduce the published figures' matplotlib styling.

    Two separate effects, both of which the original scripts got by accident:

    1. seaborn's "talk" context (font.size 18, axes.linewidth 1.875, ...; 20
       rcParams in all). The originals never ask for this -- they import
       benchmarks_scIB -> metrics -> `import scib`, and scib/preprocessing.py:18
       runs seaborn.set_context("talk") at import time. Without it the panels
       render at matplotlib defaults and come out a different size. Calling
       set_context directly is verified to give byte-identical output while
       dropping the torch/scib dependency.
    2. The Arial Unicode font. The originals addfont() unguarded and die if it is
       absent; here a missing font only costs exact glyph fidelity.
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns

    sns.set_context("talk")
    if not os.path.isfile(font_path):
        print(f"[font] {font_path} not found — falling back to matplotlib default", flush=True)
        return
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
