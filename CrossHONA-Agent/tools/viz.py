"""Visualization tools — UMAP and loss curves from saved arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import RESULTS_ROOT, safe_path


def plot_umap(project_name: str, run_name: str,
              embedding: str = "ref_latent") -> dict:
    """
    Compute a 2-D UMAP from a saved latent .npy and write it under the run.
    `embedding` selects which file to use: ref_latent | target_latent |
    ref_latent_homo | tgt_latent_homo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = safe_path(project_name, RESULTS_ROOT)
    run = proj / run_name
    npy = run / f"{embedding}.npy"
    if not npy.exists():
        return {"error": f"{npy.relative_to(RESULTS_ROOT)} not found"}

    x = np.load(npy)
    try:
        import umap
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(x)
    except ImportError:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(x)

    side = "ref" if embedding.startswith("ref") else "target"
    gt_path = run / f"{side}_gt.npy"
    labels = np.load(gt_path) if gt_path.exists() else np.zeros(len(coords))

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{project_name}/{run_name} — {embedding}")
    plt.colorbar(sc, ax=ax, label="class")

    out_path = run / f"umap_{embedding}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"image_path": str(out_path), "n_points": int(len(coords))}


def plot_loss_curves(project_name: str, run_name: str) -> dict:
    """Plot per-stage loss curves from a run's saved metrics JSON."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = safe_path(project_name, RESULTS_ROOT)
    run = proj / run_name

    losses_files = sorted(run.rglob("*_losses.json"))
    if not losses_files:
        return {"error": "no *_losses.json found in any stage subdir"}

    fig, axes = plt.subplots(1, len(losses_files),
                             figsize=(4 * len(losses_files), 3.5),
                             sharey=True)
    if len(losses_files) == 1:
        axes = [axes]
    for ax, lf in zip(axes, sorted(losses_files)):
        data = json.loads(lf.read_text())
        for k in ("loss", "recon", "cls"):
            if k in data:
                ax.plot(data[k], label=k, linewidth=1.2)
        ax.set_title(lf.parent.name, fontsize=9)
        ax.set_xlabel("epoch"); ax.legend(fontsize=7)

    out_path = run / "loss_curves.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {"image_path": str(out_path)}
