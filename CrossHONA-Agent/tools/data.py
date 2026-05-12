"""Dataset inspection tools — read-only, scoped to DATA_ROOT."""

from __future__ import annotations

from pathlib import Path

from . import DATA_ROOT, safe_path


def list_datasets() -> dict:
    """List h5ad, CSV, and TSV files available in the data directory."""
    h5ads = sorted(p.relative_to(DATA_ROOT).as_posix()
                   for p in DATA_ROOT.rglob("*.h5ad"))
    csvs = sorted(p.relative_to(DATA_ROOT).as_posix()
                  for p in DATA_ROOT.rglob("*.csv"))
    tsvs = sorted(p.relative_to(DATA_ROOT).as_posix()
                  for p in DATA_ROOT.rglob("*.tsv"))
    return {"data_root": str(DATA_ROOT), "h5ad": h5ads, "csv": csvs, "tsv": tsvs}


def inspect_dataset(path: str) -> dict:
    """
    Return basic stats about an h5ad: cell count, gene count, obs columns,
    and any cell-type column candidates.
    """
    import anndata as ad

    p = safe_path(path, DATA_ROOT)
    if not p.exists():
        return {"error": f"file not found: {path}"}
    if p.suffix.lower() != ".h5ad":
        return {"error": "only .h5ad supported by this tool"}

    adata = ad.read_h5ad(p, backed="r")
    obs_cols = list(adata.obs.columns)
    candidates = [c for c in obs_cols
                  if "cell" in c.lower() or "type" in c.lower() or "label" in c.lower()]

    out = {
        "path": str(p.relative_to(DATA_ROOT)),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": obs_cols,
        "celltype_column_candidates": candidates,
        "obsm_keys": list(adata.obsm.keys()),
        "has_spatial": "spatial" in adata.obsm or "X_spatial" in adata.obsm,
    }
    if candidates:
        col = candidates[0]
        try:
            counts = adata.obs[col].value_counts().head(15).to_dict()
            out["top_cell_types"] = {str(k): int(v) for k, v in counts.items()}
        except Exception:
            pass
    return out
