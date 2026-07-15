import numpy as np
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
    balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr


def compute_basic_metrics(embeddings: dict, *, condition: bool = True) -> dict:
    """
    1) reconstruction correlation (per-cell total counts)
    2) classification metrics:
       - accuracy
       - Weighted F1
       - prediction ARI
       - prediction NMI
    3) integration: silhouette by species (lower is better mixing)
       - silhouette on homo_mean / nonhomo_mean
       - silhouette on full concat
    """
    metrics = {}

    # 1) reconstruction correlation (per-cell total counts)
    for species in ["ref", "tgt"]:
        for part in ["homo", "nonhomo"]:
            raw = embeddings[f"{species}_raw_{part}"].sum(1)
            recon_mat = embeddings[f"{species}_recon_{part}"]
            if condition and part == "homo":
                recon_mat = recon_mat[:, :-1]
            recon = recon_mat.sum(1)

            # guard constant vector
            if np.std(raw) < 1e-8 or np.std(recon) < 1e-8:
                corr = np.nan
            else:
                corr, _ = pearsonr(raw, recon)

            metrics[f"recon_corr_{species}_{part}"] = float(corr) if corr is not None else np.nan

    # 2) classification metrics — model's end-to-end cls head
    for species in ["ref", "tgt"]:
        logits = embeddings[f"{species}_logits"]
        y = np.asarray(embeddings[f"{species}_y"]).astype(int)
        preds = np.argmax(logits, axis=1).astype(int)

        # accuracy
        acc = float((preds == y).mean())
        metrics[f"accuracy_{species}"] = acc
        # # balanced accuracy
        # bal = float(balanced_accuracy_score(y, preds))
        # metrics[f"bal_accuracy_{species}"] = bal
        # # weighted f1
        # f1 = float(f1_score(y, preds, average="weighted"))
        # metrics[f"weightedF1_{species}"] = f1
        # ari
        metrics[f"ari_{species}"] = float(adjusted_rand_score(y, preds))
        # # nmi
        # metrics[f"nmi_{species}"] = float(normalized_mutual_info_score(y, preds))

    # # 3) integration: silhouette by species (lower is better mixing)
    # for emb_type in ["homo_mean", "nonhomo_mean"]:
    #     ref_emb = embeddings[f"ref_{emb_type}"]
    #     tgt_emb = embeddings[f"tgt_{emb_type}"]
    #     combined = np.vstack([ref_emb, tgt_emb])
    #     labels = np.array(["ref"] * len(ref_emb) + ["tgt"] * len(tgt_emb))

    #     if len(combined) > 5000:
    #         idx = np.random.choice(len(combined), 5000, replace=False)
    #         combined = combined[idx]
    #         labels = labels[idx]

    #     try:
    #         sil = float(silhouette_score(combined, labels))
    #     except Exception:
    #         sil = np.nan
    #     metrics[f"silhouette_{emb_type}"] = sil

    # # full concat
    # ref_full = np.hstack([embeddings["ref_homo_mean"], embeddings["ref_nonhomo_mean"]])
    # tgt_full = np.hstack([embeddings["tgt_homo_mean"], embeddings["tgt_nonhomo_mean"]])
    # combined_full = np.vstack([ref_full, tgt_full])
    # labels_full = np.array(["ref"] * len(ref_full) + ["tgt"] * len(tgt_full))

    # if len(combined_full) > 5000:
    #     idx = np.random.choice(len(combined_full), 5000, replace=False)
    #     combined_full = combined_full[idx]
    #     labels_full = labels_full[idx]

    # try:
    #     metrics["silhouette_full"] = float(silhouette_score(combined_full, labels_full))
    # except Exception:
    #     metrics["silhouette_full"] = np.nan

    return metrics