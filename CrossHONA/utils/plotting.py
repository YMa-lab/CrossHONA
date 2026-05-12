import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.linear_model import LogisticRegression


def plot_losses(losses: dict, save_dir: str, stage_name: str):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs = range(1, len(losses["loss"]) + 1)

    axes[0, 0].plot(epochs, losses["loss"], lw=2)
    axes[0, 0].set_title("Total Loss"); axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(epochs, losses["recon"], lw=2)
    axes[0, 1].set_title("Reconstruction Loss"); axes[0, 1].set_xlabel("Epoch")

    axes[0, 2].plot(epochs, losses["cls"], lw=2)
    axes[0, 2].set_title("Classification Loss"); axes[0, 2].set_xlabel("Epoch")

    axes[1, 0].plot(epochs, losses["align_intra_ref"], label="Ref")
    axes[1, 0].plot(epochs, losses["align_intra_tgt"], label="Tgt")
    axes[1, 0].set_title("Level 1: Intra-Species"); axes[1, 0].legend(); axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, losses["align_bridge"], lw=2)
    axes[1, 1].set_title("Level 2: Bridged Full (CORAL)"); axes[1, 1].set_xlabel("Epoch")

    if "proto_ref" in losses and "proto_tgt" in losses:
        axes[1, 2].plot(epochs, losses["proto_ref"], label="Ref")
        axes[1, 2].plot(epochs, losses["proto_tgt"], label="Tgt")
        axes[1, 2].set_title("Prototype Contrastive"); axes[1, 2].legend()
    else:
        axes[1, 2].set_visible(False)
    axes[1, 2].set_xlabel("Epoch")

    plt.suptitle(f"{stage_name}: Loss Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{stage_name}_losses.png"), dpi=150)
    plt.close()


def plot_reconstruction(embeddings: dict, save_dir: str, stage_name: str, *, condition: bool):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, species in enumerate(["ref", "tgt"]):
        for j, part in enumerate(["homo", "nonhomo"]):
            ax = axes[i, j]
            raw = embeddings[f"{species}_raw_{part}"].sum(1)
            recon_mat = embeddings[f"{species}_recon_{part}"]
            if condition and part == "homo":
                recon_mat = recon_mat[:, :-1]
            recon = recon_mat.sum(1)

            ax.scatter(raw, recon, alpha=0.3, s=5)
            ax.plot([raw.min(), raw.max()], [raw.min(), raw.max()], "r--", lw=2)

            if np.std(raw) < 1e-8 or np.std(recon) < 1e-8:
                corr = np.nan
            else:
                corr, _ = pearsonr(raw, recon)

            ax.set_title(f"{species.upper()} {part} (r={corr:.3f})")
            ax.set_xlabel("Raw"); ax.set_ylabel("Reconstructed")

    # spatial totals
    for i, species in enumerate(["ref", "tgt"]):
        pos = embeddings[f"{species}_pos"]

        ax = axes[i, 2]
        raw_total = embeddings[f"{species}_raw_homo"].sum(1) + embeddings[f"{species}_raw_nonhomo"].sum(1)
        sca = ax.scatter(pos[:, 0], pos[:, 1], c=raw_total, s=5, cmap="viridis")
        ax.set_title(f"{species.upper()}: Raw Total Counts")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sca, ax=ax)

        ax = axes[i, 3]
        recon_homo = embeddings[f"{species}_recon_homo"]
        if condition:
            recon_homo = recon_homo[:, :-1]
        recon_total = recon_homo.sum(1) + embeddings[f"{species}_recon_nonhomo"].sum(1)
        sca = ax.scatter(pos[:, 0], pos[:, 1], c=recon_total, s=5, cmap="viridis")
        ax.set_title(f"{species.upper()}: Recon Total Counts")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sca, ax=ax)

    plt.suptitle(f"{stage_name}: Reconstruction Quality", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{stage_name}_reconstruction.png"), dpi=150)
    plt.close()


def plot_umap_embeddings(
    embeddings: dict,
    save_dir: str,
    stage_name: str,
    inv_ref: dict = None,
    inv_tgt: dict = None,
):
    """
    3 rows x 3 cols:
    Row 1: species
    Row 2: cluster number (leiden)
    Row 3: celltype (global fixed colormap)
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    emb_types = [
        ("homo_mean", "Homo Only"),
        ("nonhomo_mean", "Nonhomo Only"),
        ("full", "Full (homo+nonhomo)"),
    ]

    species_color = {"Reference": "#DBBED8", "Target": "#D8EAF5"}

    # ---------- build global celltype names FIRST ----------
    ref_y = embeddings["ref_y"]
    tgt_y = embeddings["tgt_y"]

    ref_names = []
    for i in ref_y:
        ii = int(i)
        ref_names.append(inv_ref[ii] if (inv_ref is not None and ii in inv_ref) else f"type_{ii}")

    tgt_names = []
    for i in tgt_y:
        # y2 might be float nan
        if isinstance(i, float) and np.isnan(i):
            tgt_names.append("unknown")
        else:
            ii = int(i)
            tgt_names.append(inv_tgt[ii] if (inv_tgt is not None and ii in inv_tgt) else f"type_{ii}")

    all_names = np.array(ref_names + tgt_names, dtype=str)
    uniq_celltypes = sorted(np.unique(all_names).tolist())

    cmap_ct = plt.cm.get_cmap("tab20", max(len(uniq_celltypes), 1))
    celltype_color_map = {ct: cmap_ct(i) for i, ct in enumerate(uniq_celltypes)}

    for col, (emb_type, title) in enumerate(emb_types):
        if emb_type == "full":
            ref_emb = np.hstack([embeddings["ref_homo_mean"], embeddings["ref_nonhomo_mean"]])
            tgt_emb = np.hstack([embeddings["tgt_homo_mean"], embeddings["tgt_nonhomo_mean"]])
        else:
            ref_emb = embeddings[f"ref_{emb_type}"]
            tgt_emb = embeddings[f"tgt_{emb_type}"]

        combined = np.vstack([ref_emb, tgt_emb])
        species_labels = ["Reference"] * len(ref_emb) + ["Target"] * len(tgt_emb)

        adata = sc.AnnData(combined)
        adata.obs["species"] = pd.Categorical(species_labels)
        adata.obs["celltype"] = pd.Categorical(all_names)  # <-- use names

        sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added="cluster", resolution=0.5)

        # Row 0: species
        ax = axes[0, col]
        for sp in ["Reference", "Target"]:
            mask = (adata.obs["species"] == sp).to_numpy()
            ax.scatter(
                adata.obsm["X_umap"][mask, 0],
                adata.obsm["X_umap"][mask, 1],
                c=species_color[sp],
                s=3,
                alpha=0.8,
                label=sp,
            )
        ax.set_title(f"{title}\ncolored by species")
        ax.legend(markerscale=3, frameon=False)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 1: cluster
        ax = axes[1, col]
        clusters = adata.obs["cluster"].astype(str).to_numpy()
        uniq = np.unique(clusters)

        cmap_cluster = plt.cm.get_cmap("tab20", max(len(uniq), 1))
        cluster_color_map = {
            cid: cmap_cluster(i)
            for i, cid in enumerate(sorted(uniq.tolist(), key=lambda x: int(x) if x.isdigit() else x))
        }

        for cid in cluster_color_map:
            mask = (clusters == cid)
            ax.scatter(
                adata.obsm["X_umap"][mask, 0],
                adata.obsm["X_umap"][mask, 1],
                c=[cluster_color_map[cid]],
                s=3,
                alpha=0.65,
            )
        ax.set_title(f"{title}\ncolored by cluster (leiden)")
        ax.set_xticks([]); ax.set_yticks([])

        # Row 2: celltype (NOW matches names)
        ax = axes[2, col]
        celltypes = adata.obs["celltype"].astype(str).to_numpy()

        for ct in uniq_celltypes:
            mask = (celltypes == ct)
            if mask.sum() == 0:
                continue
            ax.scatter(
                adata.obsm["X_umap"][mask, 0],
                adata.obsm["X_umap"][mask, 1],
                c=[celltype_color_map[ct]],
                s=3,
                alpha=0.65,
                label=ct if col == 2 else None,  # only legend once
            )

        ax.set_title(f"{title}\ncolored by celltype")
        ax.set_xticks([]); ax.set_yticks([])

        if col == 2:
            ax.legend(
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                frameon=False,
                fontsize=7,
                markerscale=4,
                ncol=1,
            )

    plt.suptitle(
        f"{stage_name}: UMAP Embeddings\n(Row1: species | Row2: cluster | Row3: celltype)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{stage_name}_umap.png"), dpi=150)
    plt.close()


def _plot_cm(y_true, y_pred, *, title, save_path, label_ids, id2name,
             normalize=True):
    """Single row-normalised confusion matrix with per-cell annotation."""
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_show = np.nan_to_num(cm_show)
        vmax = 1.0
    else:
        cm_show = cm.astype(float)
        vmax = cm_show.max() if cm_show.max() > 0 else 1.0

    n = len(label_ids)
    names = [id2name.get(i, str(i)) for i in label_ids]

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * n + 2),
                                    max(5, 0.45 * n + 2)))
    im = ax.imshow(cm_show, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    thresh = vmax * 0.5
    for i in range(n):
        for j in range(n):
            v = cm_show[i, j]
            if normalize:
                txt = f"{v:.2f}" if v > 0.005 else ""
            else:
                txt = f"{int(v)}" if v > 0 else ""
            if txt:
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(embeddings: dict, save_dir: str, stage_name: str,
                            inv_ref: dict = None, inv_tgt: dict = None):
    """
    Two confusion-matrix variants written to <save_dir>/<stage>_*.png:
      - cls head (model's end-to-end predictions, from ref_pred / tgt_pred)
      - LogReg fit on ref_homo_mean (matches scGen-style benchmarking)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Pull labels
    y_ref = np.asarray(embeddings["ref_y"]).astype(int)
    y_tgt = np.asarray(embeddings["tgt_y"]).astype(int)

    # Build id -> name lookup spanning both species
    inv_ref = inv_ref or {}
    inv_tgt = inv_tgt or {}
    all_ids = sorted(set(np.concatenate([np.unique(y_ref), np.unique(y_tgt)])))
    id2name = {}
    for i in all_ids:
        if i in inv_ref:
            id2name[i] = inv_ref[i]
        elif i in inv_tgt:
            id2name[i] = inv_tgt[i]
        else:
            id2name[i] = f"cls_{i}"
    label_ids = sorted(id2name.keys())

    # --- 1) cls head ---
    p_ref_cls = np.argmax(embeddings["ref_logits"], axis=1).astype(int)
    p_tgt_cls = np.argmax(embeddings["tgt_logits"], axis=1).astype(int)
    acc_t = accuracy_score(y_tgt, p_tgt_cls)
    bal_t = balanced_accuracy_score(y_tgt, p_tgt_cls)
    _plot_cm(y_tgt, p_tgt_cls,
             title=f"{stage_name}: cls-head tgt  acc={acc_t:.3f}  bal={bal_t:.3f}",
             save_path=os.path.join(save_dir, f"{stage_name}_cm_cls_tgt.png"),
             label_ids=label_ids, id2name=id2name, normalize=True)

    # --- 2) LogReg on homo_mean ---
    try:
        X_ref = np.asarray(embeddings["ref_homo_mean"])
        X_tgt = np.asarray(embeddings["tgt_homo_mean"])
        clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_ref, y_ref)
        p_tgt_lr = clf.predict(X_tgt)
        acc_t = accuracy_score(y_tgt, p_tgt_lr)
        bal_t = balanced_accuracy_score(y_tgt, p_tgt_lr)
        _plot_cm(y_tgt, p_tgt_lr,
                 title=f"{stage_name}: LogReg tgt  acc={acc_t:.3f}  bal={bal_t:.3f}",
                 save_path=os.path.join(save_dir, f"{stage_name}_cm_logreg_tgt.png"),
                 label_ids=label_ids, id2name=id2name, normalize=True)
    except Exception as e:
        print(f"[plot_confusion_matrices] LogReg variant skipped: {e}")


def plot_stage_results(
    stage_name: str,
    embeddings: dict,
    losses: dict,
    metrics: dict,
    scib_metrics: dict,
    save_dir: str,
    condition: bool,
    inv_ref: dict = None,
    inv_tgt: dict = None,
):
    """
    Convenience wrapper: save all plots.
    """
    plot_losses(losses, save_dir, stage_name)
    plot_reconstruction(embeddings, save_dir, stage_name, condition=condition)
    plot_umap_embeddings(embeddings, save_dir, stage_name, inv_ref=inv_ref, inv_tgt=inv_tgt)
    plot_confusion_matrices(embeddings, save_dir, stage_name,
                            inv_ref=inv_ref, inv_tgt=inv_tgt)