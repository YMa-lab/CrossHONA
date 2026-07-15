"""per_celltype_distances — verbatim from phylogenetic/distances.py.

Trimmed to what Figure 5 panel B needs (the three metrics it plots).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .tensor_builder import ImportanceTensor


def _per_celltype_matrix(it: ImportanceTensor, c: int) -> np.ndarray:
    """Returns shape [n_species, n_genes] for cell type index ``c``."""
    return it.tensor[:, c, :]


def _pairwise(it: ImportanceTensor, fn) -> np.ndarray:
    """Apply ``fn(I_a, I_b, mask)`` across species pairs per cell type and
    average across the cell types that actually contributed for that pair.
    Returns symmetric [S, S] matrix.  Pairs with zero contributing cell types
    are returned as NaN so they can be excluded downstream."""
    S = len(it.species)
    D = np.zeros((S, S))
    counts = np.zeros((S, S), dtype=int)
    for c in range(len(it.cell_types)):
        Mc = _per_celltype_matrix(it, c)
        valid_per_species = ~np.isnan(Mc).all(axis=1)   # species with any data for this CT
        for i in range(S):
            for j in range(i + 1, S):
                if not (valid_per_species[i] and valid_per_species[j]):
                    continue
                ai = Mc[i]; aj = Mc[j]
                mask = ~(np.isnan(ai) | np.isnan(aj))
                if mask.sum() < 5:
                    continue
                d = fn(ai[mask], aj[mask])
                D[i, j] += d
                D[j, i] += d
                counts[i, j] += 1
                counts[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(counts > 0, D / np.maximum(counts, 1), np.nan)
    np.fill_diagonal(D, 0.0)
    return D


def spearman_distance_matrix(it: ImportanceTensor) -> pd.DataFrame:
    """1 - Spearman ρ averaged across cell types."""
    def _f(a, b):
        rho = spearmanr(a, b).correlation
        return 1.0 - (rho if rho is not None and np.isfinite(rho) else 0.0)
    D = _pairwise(it, _f)
    return pd.DataFrame(D, index=it.species, columns=it.species)


def topk_jaccard_distance_matrix(it: ImportanceTensor, k: int = 50) -> pd.DataFrame:
    """1 - top-K Jaccard averaged across cell types.

    Picks top-K genes per species per cell type then computes set-Jaccard.
    """
    def _f(a, b):
        ka = np.argpartition(-np.abs(a), min(k, len(a) - 1))[:k]
        kb = np.argpartition(-np.abs(b), min(k, len(b) - 1))[:k]
        inter = len(set(ka) & set(kb))
        union = len(set(ka) | set(kb))
        return 1.0 - (inter / union if union else 0.0)
    D = _pairwise(it, _f)
    return pd.DataFrame(D, index=it.species, columns=it.species)


def differential_magnitude_distance_matrix(it: ImportanceTensor) -> pd.DataFrame:
    """mean |I_a - I_b| across genes & cell types (L1 distance)."""
    def _f(a, b):
        return float(np.mean(np.abs(a - b)))
    D = _pairwise(it, _f)
    return pd.DataFrame(D, index=it.species, columns=it.species)


def per_celltype_distances(
    it: ImportanceTensor,
    metric: str = "spearman",
    k: int = 50,
) -> dict:
    """
    Returns ``{cell_type: pd.DataFrame[S x S]}`` of pairwise distances,
    one frame per cell type.
    """
    out = {}
    for c, ct in enumerate(it.cell_types):
        Mc = _per_celltype_matrix(it, c)
        S = len(it.species)
        D = np.zeros((S, S))
        for i in range(S):
            for j in range(i + 1, S):
                ai = Mc[i]; aj = Mc[j]
                mask = ~(np.isnan(ai) | np.isnan(aj))
                if mask.sum() < 5:
                    D[i, j] = D[j, i] = float("nan")
                    continue
                a = ai[mask]; b = aj[mask]
                if metric == "spearman":
                    rho = spearmanr(a, b).correlation
                    d = 1.0 - (rho if rho is not None and np.isfinite(rho) else 0.0)
                elif metric == "topk_jaccard":
                    ka = np.argpartition(-np.abs(a), min(k, len(a) - 1))[:k]
                    kb = np.argpartition(-np.abs(b), min(k, len(b) - 1))[:k]
                    inter = len(set(ka) & set(kb))
                    union = len(set(ka) | set(kb))
                    d = 1.0 - (inter / union if union else 0.0)
                elif metric == "l1":
                    d = float(np.mean(np.abs(a - b)))
                else:
                    raise ValueError(f"unknown metric {metric!r}")
                D[i, j] = D[j, i] = d
        out[ct] = pd.DataFrame(D, index=it.species, columns=it.species)
    return out

