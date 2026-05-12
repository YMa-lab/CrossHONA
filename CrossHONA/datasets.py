import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch_geometric.data import Data
from torch.utils.data import Dataset
from utils.preprocess import preprocess


def filter_adata(adata: sc.AnnData) -> sc.AnnData:
    """Basic QC filtering."""
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    return adata


def get_spatial_coords(adata) -> np.ndarray:
    """Return (n_obs, 2) spatial coordinates, falling back to zeros."""
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
    elif "X_spatial" in adata.obsm:
        coords = adata.obsm["X_spatial"]
    elif {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs[["x", "y"]].to_numpy()
    elif {"x_coord", "y_coord"}.issubset(adata.obs.columns):
        coords = adata.obs[["x_coord", "y_coord"]].to_numpy()
    else:
        coords = np.zeros((adata.n_obs, 2), dtype=float)

    if coords.shape[0] != adata.n_obs or coords.shape[1] != 2:
        raise ValueError(f"Unexpected spatial array shape: {coords.shape}")
    return coords.astype(float)


def read_dataset(ref_path: str, target_path: str, species1: str, species2: str,
                 p_homo_df=None, skip_QC=False):
    """Load and QC-filter two AnnData objects. Returns (ad1, ad2, homo_df)."""
    ad1 = sc.read_h5ad(ref_path)
    ad2 = sc.read_h5ad(target_path)

    if not skip_QC:
        ad1 = filter_adata(ad1)
        ad2 = filter_adata(ad2)

    if p_homo_df is None:
        p_homo_df = (
            f"homedir"
            f"homolog_genes/{species1}_{species2}.tsv"
        )
    homo_df = pd.read_csv(p_homo_df, sep="\t")
    return ad1, ad2, homo_df


def extract_and_preprocess_data(args, ad1, ad2, species1, species2, homo_df,
                                homo_gene_id_ref, homo_gene_id_target):
    """
    Subset AnnData to homologous vs non-homologous genes and preprocess.
    Returns: ref_homo, ref_nonhomo, tgt_homo, tgt_nonhomo, valid_pairs.
    """
    genes1  = homo_df[homo_gene_id_ref]
    lowered = homo_gene_id_target[0].lower() + homo_gene_id_target[1:]
    genes2  = homo_df[f"{species2} {lowered}"]

    pairs     = [(g1, g2) for g1, g2 in zip(genes1, genes2)
                 if g1 in ad1.var_names and g2 in ad2.var_names]
    ref_homo  = ad1[:, [p[0] for p in pairs]].copy()
    ref_homo  = ref_homo[:, ~ref_homo.var_names.duplicated()]
    ref_non   = ad1[:, ~ad1.var_names.isin([p[0] for p in pairs])].copy()
    tgt_homo  = ad2[:, [p[1] for p in pairs]].copy()
    tgt_homo  = tgt_homo[:, ~tgt_homo.var_names.duplicated()]
    tgt_non   = ad2[:, ~ad2.var_names.isin([p[1] for p in pairs])].copy()
    for label, adata in [("ref_homo", ref_homo), ("ref_non", ref_non),
                      ("tgt_homo", tgt_homo), ("tgt_non", tgt_non)]:
        dups = adata.var_names[adata.var_names.duplicated()].tolist()
        if dups:
            print(f"{label} has {len(dups)} duplicate var_names: {dups[:10]}")
    return preprocess(args, ref_homo, ref_non, tgt_homo, tgt_non, pairs)


def load_dataset(
    args,
    ad_ref_homo, ad_ref_non,
    ad_tgt_homo, ad_tgt_non,
    ct_ref: str = "cell_type", ct_tgt: str = "cell_type",
):
    """Build node-level Data objects (no edge_index) and return a GraphDataset."""

    def _to_dense(X):
        return X.toarray().astype(float) if hasattr(X, "toarray") else np.array(X, dtype=float)

    Xrh = _to_dense(ad_ref_homo.X)
    Xrn = _to_dense(ad_ref_non.X)
    Xth = _to_dense(ad_tgt_homo.X)
    Xtn = _to_dense(ad_tgt_non.X)

    # Reference labels
    y1_raw = ad_ref_homo.obs[ct_ref].str.lower().values

    # Target labels (may be missing)
    try:
        y2_raw = ad_tgt_homo.obs[ct_tgt].str.lower().values
        tgt_labels_ok = True
    except Exception as e:
        print(f"Target label extraction failed: {e}")
        y2_raw = None
        tgt_labels_ok = False

    # SHARED label encoding across ref and tgt — critical for fair target
    # evaluation. Without this, the same cell type can map to different
    # integer IDs across species and target accuracy is computed against
    # the wrong reference.
    if tgt_labels_ok:
        all_types = sorted(set(y1_raw) | set(y2_raw))
    else:
        all_types = sorted(set(y1_raw))
    d_shared = {t: i for i, t in enumerate(all_types)}
    d1 = d_shared
    y1 = np.array([d_shared[c] for c in y1_raw])
    if tgt_labels_ok:
        d2 = d_shared
        y2 = np.array([d_shared[c] for c in y2_raw])
    else:
        y2 = np.zeros(ad_tgt_homo.n_obs, dtype=int)
        d2 = {}

    pos1 = get_spatial_coords(ad_ref_homo)
    pos2 = get_spatial_coords(ad_tgt_homo)

    ref_data = Data(
        homo_x    = torch.FloatTensor(Xrh),
        nonhomo_x = torch.FloatTensor(Xrn),
        y         = torch.LongTensor(y1),
        pos       = torch.FloatTensor(pos1),
    )

    tgt_data = Data(
        homo_x    = torch.FloatTensor(Xth),
        nonhomo_x = torch.FloatTensor(Xtn),
        y         = torch.LongTensor(y2),
        pos       = torch.FloatTensor(pos2),
    )

    inv_ref = {v: k for k, v in d1.items()}
    inv_tgt = {v: k for k, v in d2.items()}

    return GraphDataset(ref_data, tgt_data), inv_ref, inv_tgt


class GraphDataset(Dataset):
    def __init__(self, ref_data, target_data):
        self.ref_data    = ref_data
        self.target_data = target_data

    @classmethod
    def from_saved(cls, ref_path: str, target_path: str):
        ref = torch.load(ref_path, weights_only=False)
        tgt = torch.load(target_path, weights_only=False)
        return cls(ref, tgt)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.ref_data, self.target_data
