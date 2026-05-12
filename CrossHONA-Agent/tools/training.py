"""Training tools — preprocess, kick off async training, monitor, stop."""

from __future__ import annotations

import time
from pathlib import Path

from api import CrossSpeciesModel

from . import DATA_ROOT, RESULTS_ROOT, safe_path


_PREPROC_DEFAULTS = dict(
    skip_QC=False,
    hvg=5000,
    nonhomo_hvg=1000,
    distance_thres_ref=20.0,
    distance_thres_target=20.0,
    k_ref=11,
    k_target=11,
    celltype_name_ref="cell_type",
    celltype_name_target="cell_type",
    homo_gene_id_ref="Gene name",
    homo_gene_id_target="Gene name",
    identity_graph=False,
)


def _coerce(value, default):
    """Treat None / empty string / 0 (for thresholds) as 'not specified'."""
    if value is None or value == "" or (isinstance(default, (int, float))
                                        and not isinstance(default, bool)
                                        and value == 0
                                        and default != 0):
        return default
    return value


def preprocess(
    ref_h5ad: str,
    target_h5ad: str,
    species1_name: str,
    species2_name: str,
    homo_table: str,
    project_name: str,
    skip_QC: bool = False,
    hvg: int = 5000,
    nonhomo_hvg: int = 1000,
    distance_thres_ref: float = 20.0,
    distance_thres_target: float = 20.0,
    k_ref: int = 11,
    k_target: int = 11,
    celltype_name_ref: str = "cell_type",
    celltype_name_target: str = "cell_type",
    homo_gene_id_ref: str = "Gene name",
    homo_gene_id_target: str = "Gene name",
    identity_graph: bool = False,
) -> dict:
    """Preprocess two h5ads + a homologous-gene table (CSV/TSV) into graph datasets."""
    # Defend against agents that pass None / "" / 0 for "I don't know".
    raw = dict(
        skip_QC=skip_QC, hvg=hvg, nonhomo_hvg=nonhomo_hvg,
        distance_thres_ref=distance_thres_ref,
        distance_thres_target=distance_thres_target,
        k_ref=k_ref, k_target=k_target,
        celltype_name_ref=celltype_name_ref,
        celltype_name_target=celltype_name_target,
        homo_gene_id_ref=homo_gene_id_ref,
        homo_gene_id_target=homo_gene_id_target,
        identity_graph=identity_graph,
    )
    cleaned = {k: _coerce(v, _PREPROC_DEFAULTS[k]) for k, v in raw.items()}

    ref = safe_path(ref_h5ad, DATA_ROOT)
    tgt = safe_path(target_h5ad, DATA_ROOT)
    homo = safe_path(homo_table, DATA_ROOT)
    proj = safe_path(project_name, RESULTS_ROOT)
    proj.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    m = CrossSpeciesModel(savedir=proj, name="_preproc", **cleaned)
    m.preprocess(str(ref), str(tgt), species1_name, species2_name, str(homo))

    # Verify the four required artifacts actually landed on disk.
    required = ["ref_data.pt", "target_data.pt",
                "inverse_dict_ref.pkl", "inverse_dict_target.pkl"]
    files = {p.name for p in proj.iterdir() if p.is_file()}
    missing = [f for f in required if f not in files]
    if missing:
        return {
            "error": f"preprocess finished but did not produce {missing}. "
                     "Check that ref_h5ad / target_h5ad / homo_table paths exist, "
                     "celltype column names match the obs columns in your h5ads, "
                     "and homo_gene_id_* match the column names in your homo table.",
            "savedir": str(proj),
            "files_produced": sorted(files),
        }

    return {
        "project": project_name,
        "savedir": str(proj),
        "elapsed_sec": round(time.time() - t0, 1),
        "files": sorted(files),
    }


def train(
    project_name: str,
    run_name: str,
    epochs_per_stage: int = 30,
    lr: float = 1e-4,
    latent_dim: int = 128,
    hidden_dim: int = 512,
    batch_size: int = 10,
    use_nonhomo: bool = True,
    seed: int = 42,
    alpha: float = 1.0,
    beta_cls: float = 1.0,
    beta_kl: float = 0.01,
    beta_intra: float = 1.0,
    beta_homo: float = 1.0,
    beta_bridge: float = 0.5,
    cls_on: str = "mix",
    VAE: bool = True,
    denoise: bool = True,
    condition: bool = True,
    identity_graph: bool = True,
) -> dict:
    """
    Launch staged training as a detached subprocess. Returns immediately.
    Use check_status(project_name, run_name) to monitor.
    """
    proj = safe_path(project_name, RESULTS_ROOT)
    if not (proj / "ref_data.pt").exists():
        return {"error": "no preprocessed data found. Run preprocess first."}

    train_defaults = dict(
        epochs_per_stage=30, lr=1e-4, latent_dim=128, hidden_dim=512,
        batch_size=10, use_nonhomo=True, seed=42,
        alpha=1.0, beta_cls=1.0, beta_kl=0.01,
        beta_intra=1.0, beta_homo=1.0, beta_bridge=0.5,
        cls_on="mix",
        VAE=True, denoise=True, condition=True, identity_graph=True,
    )
    raw = dict(
        epochs_per_stage=epochs_per_stage, lr=lr,
        latent_dim=latent_dim, hidden_dim=hidden_dim,
        batch_size=batch_size, use_nonhomo=use_nonhomo, seed=seed,
        alpha=alpha, beta_cls=beta_cls, beta_kl=beta_kl,
        beta_intra=beta_intra, beta_homo=beta_homo, beta_bridge=beta_bridge,
        cls_on=cls_on,
        VAE=VAE, denoise=denoise, condition=condition,
        identity_graph=identity_graph,
    )
    cleaned = {k: _coerce(v, train_defaults[k]) for k, v in raw.items()}

    m = CrossSpeciesModel(
        savedir=proj, name=run_name,
        staged_training=True,
        **cleaned,
    )
    handle = m.train_async()
    return {
        "run_id": handle.run_id,
        "pid": handle.pid,
        "savedir": str(handle.savedir),
        "log": str(handle.log_path),
        "total_epochs": epochs_per_stage * 4,
        "message": "Training started. Use check_status to monitor.",
    }


def check_status(project_name: str, run_name: str) -> dict:
    """Report whether a training run is alive and how far it has progressed."""
    proj = safe_path(project_name, RESULTS_ROOT)
    savedir = proj / run_name
    if not savedir.exists():
        return {"error": f"run {project_name}/{run_name} not found"}
    return CrossSpeciesModel.status(savedir)


def stop_run(project_name: str, run_name: str) -> dict:
    """Terminate a running training subprocess (SIGTERM, then SIGKILL)."""
    proj = safe_path(project_name, RESULTS_ROOT)
    savedir = proj / run_name
    return CrossSpeciesModel.stop(savedir)


def list_runs(project_name: str | None = None) -> dict:
    """List projects and their run subdirectories under the results root."""
    if project_name:
        proj = safe_path(project_name, RESULTS_ROOT)
        if not proj.exists():
            return {"error": f"project {project_name} not found"}
        runs = [d.name for d in proj.iterdir() if d.is_dir()]
        return {"project": project_name, "runs": sorted(runs)}

    out: dict[str, list[str]] = {}
    for proj in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        runs = [d.name for d in proj.iterdir() if d.is_dir()]
        out[proj.name] = sorted(runs)
    return {"projects": out}
