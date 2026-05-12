import numpy as np
import pandas as pd
import scanpy as sc
import warnings
import scib

warnings.filterwarnings("ignore")

def build_scib_adata(embeddings: dict, X_emb: np.ndarray, condition: bool = True):
    ref_n = embeddings["ref_y"].shape[0]
    tgt_n = embeddings["tgt_y"].shape[0]
    n = ref_n + tgt_n

    ref_raw_homo = embeddings["ref_raw_homo"]
    tgt_raw_homo = embeddings["tgt_raw_homo"]
    if condition:
        ref_raw_homo = ref_raw_homo[:, :-1]
        tgt_raw_homo = tgt_raw_homo[:, :-1]

    if ref_raw_homo.shape[1] != tgt_raw_homo.shape[1]:
        raise ValueError(
            f"ref_raw_homo dim {ref_raw_homo.shape[1]} != tgt_raw_homo dim {tgt_raw_homo.shape[1]}. "
            "raw X must be shared feature space; check homo gene alignment / condition column."
        )

    X_raw = np.vstack([ref_raw_homo, tgt_raw_homo]).astype(np.float32)

    batch = pd.Categorical(["ref"] * ref_n + ["tgt"] * tgt_n)
    y = np.concatenate([embeddings["ref_y"], embeddings["tgt_y"]]).astype(int)
    label = pd.Categorical([f"type_{i}" for i in y])

    adata = sc.AnnData(X=X_raw)
    adata.obs["batch"] = batch
    adata.obs["label"] = label

    adata_int = sc.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata_int.obs["batch"] = batch
    adata_int.obs["label"] = label
    adata_int.obsm["X_emb"] = X_emb.astype(np.float32)

    return adata, adata_int


def summary_scores(d: dict) -> dict:
    def _f(key):
        v = d.get(key, np.nan)
        try:
            v = float(v)
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    nmi = _f("nmi")
    ari = _f("ari")
    sil = _f("silhouette")
    gc  = _f("graph_conn")

    bio_vals = [v for v in [nmi, ari, sil, gc] if np.isfinite(v)]
    bio_score = float(np.mean(bio_vals)) if len(bio_vals) else float("nan")

    asw_batch = _f("silhouette_batch")
    pcr = _f("pcr")

    batch_vals = []
    if np.isfinite(asw_batch):
        batch_vals.append(asw_batch)
    if np.isfinite(pcr):
        batch_vals.append(1.0 - pcr)

    batch_score = float(np.mean(batch_vals)) if len(batch_vals) else float("nan")

    overall_vals = [v for v in [bio_score, batch_score] if np.isfinite(v)]
    overall_score = float(np.mean(overall_vals)) if len(overall_vals) else float("nan")

    return {"bio_score": bio_score, "batch_score": batch_score, "overall_score": overall_score}


def compute_scib_metrics(
    embeddings: dict,
    *,
    condition: bool = True,
    neighbors_k: int = 15,
    leiden_res: float = 0.5,
):
    def _safe(v):
        try:
            v = float(v)
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    def _run_one_space(X_emb: np.ndarray) -> dict:
        out = {}
        warnings_list = []

        adata, adata_int = build_scib_adata(embeddings, X_emb, condition=condition)

        # neighbors on embedding
        try:
            sc.pp.neighbors(adata_int, use_rep="X_emb", n_neighbors=int(neighbors_k))
        except Exception as e:
            warnings_list.append(f"neighbors_failed: {repr(e)}")
            out["warning"] = "; ".join(warnings_list)
            out.update(summary_scores(out))
            return out

        # leiden
        try:
            sc.tl.leiden(
                adata_int,
                key_added="cluster_scib",
                resolution=float(leiden_res),
                flavor="igraph",
                n_iterations=2,
                directed=False,
            )
        except Exception as e:
            warnings_list.append(f"leiden_failed: {repr(e)}")

        # NMI / ARI
        try:
            if "cluster_scib" in adata_int.obs:
                out["nmi"] = _safe(scib.metrics.nmi(adata_int, label_key="label", cluster_key="cluster_scib"))
                out["ari"] = _safe(scib.metrics.ari(adata_int, label_key="label", cluster_key="cluster_scib"))
            else:
                out["nmi"] = np.nan
                out["ari"] = np.nan
        except Exception as e:
            warnings_list.append(f"nmi_ari_failed: {repr(e)}")
            out["nmi"] = np.nan
            out["ari"] = np.nan

        # silhouette(label)
        try:
            out["silhouette"] = _safe(scib.metrics.silhouette(adata_int, label_key="label", embed="X_emb"))
        except Exception as e:
            warnings_list.append(f"silhouette_failed: {repr(e)}")
            out["silhouette"] = np.nan

        # silhouette(batch)
        try:
            out["silhouette_batch"] = _safe(
                scib.metrics.silhouette_batch(adata_int, batch_key="batch", label_key="label", embed="X_emb")
            )
        except Exception as e:
            warnings_list.append(f"silhouette_batch_failed: {repr(e)}")
            out["silhouette_batch"] = np.nan

        # PCR (lower better)
        try:
            out["pcr"] = _safe(scib.metrics.pcr_comparison(adata, adata_int, covariate="batch", embed="X_emb"))
        except Exception as e:
            warnings_list.append(f"pcr_failed: {repr(e)}")
            out["pcr"] = np.nan

        # graph connectivity
        try:
            out["graph_conn"] = _safe(scib.metrics.graph_connectivity(adata_int, label_key="label"))
        except Exception as e:
            warnings_list.append(f"graph_conn_failed: {repr(e)}")
            out["graph_conn"] = np.nan

        out.update(summary_scores(out))
        if warnings_list:
            out["warning"] = "; ".join(warnings_list)
        return out

    # spaces
    X_homo = np.vstack([embeddings["ref_homo_mean"], embeddings["tgt_homo_mean"]])
    X_full = np.vstack([
        np.hstack([embeddings["ref_homo_mean"], embeddings["ref_nonhomo_mean"]]),
        np.hstack([embeddings["tgt_homo_mean"], embeddings["tgt_nonhomo_mean"]]),
    ])

    return {
        "homo_mean": _run_one_space(X_homo),
        "full_supp": _run_one_space(X_full),
    }