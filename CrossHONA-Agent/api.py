"""
CrossSpeciesModel — user-friendly wrapper around the staged hierarchical aligner.

Synchronous calls (`preprocess`, `predict`) run in-process.
Long-running training is launched as a detached subprocess so the web app
stays responsive; progress is read back from a structured log file.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

# Make the existing scripts importable. SCRIPTS_DIR can be overridden via env var
# so the same code runs both locally and inside Docker. Default assumes the
# sibling CrossHONA/ training package at the repo root.
_default_scripts_dir = Path(__file__).resolve().parent.parent / "CrossHONA"
SCRIPTS_DIR = Path(
    os.environ.get("CROSSSPECIES_SCRIPTS_DIR", str(_default_scripts_dir))
).resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


DEFAULTS: dict[str, Any] = dict(
    seed=42,
    epochs_per_stage=30,
    lr=1e-4,
    wd=5e-2,
    batch_size=10,
    GAT_head=1,
    latent_dim=128,
    hidden_dim=512,
    alpha=1.0,
    beta_cls=1.0,
    beta_kl=0.01,
    beta_intra=1.0,
    beta_homo=1.0,
    beta_bridge=0.5,
    cls_on="mix",
    spot_size=5,
    hvg=5000,
    nonhomo_hvg=1000,
    distance_thres_ref=20.0,
    distance_thres_target=20.0,
    k_ref=11,
    k_target=11,
    homo_gene_id_ref="Gene name",
    homo_gene_id_target="gene name",
    celltype_name_ref="cell_type",
    celltype_name_target="cell_type",
    loc_name_ref="spatial",
    loc_name_target="spatial",
    region_name_ref=None,
    region_name_target=None,
    skip_QC=False,
    identity_graph=False,
    VAE=False,
    denoise=False,
    condition=False,
    use_nonhomo=True,
    staged_training=True,
)


@dataclass
class RunHandle:
    """Lightweight handle to an in-flight or finished training run."""
    run_id: str
    savedir: Path
    pid: int | None = None
    log_path: Path = field(init=False)

    def __post_init__(self):
        self.log_path = self.savedir / "train.log"


class CrossSpeciesModel:
    """
    High-level facade. Three modes:
      1. preprocess() — build graph data from h5ad inputs (sync, minutes).
      2. train() / train_async() — fit the model. Async returns immediately.
      3. predict() — extract embeddings from a trained checkpoint (sync).
    """

    def __init__(self, savedir: str | Path, name: str = "run", **overrides):
        savedir = Path(savedir).resolve()
        savedir.mkdir(parents=True, exist_ok=True)

        cfg = {**DEFAULTS, **overrides}
        cfg["savedirbase"] = str(savedir)
        cfg["savedir"] = str(savedir / name)
        cfg["name"] = name
        cfg["cuda"] = torch.cuda.is_available()
        cfg["device"] = torch.device("cuda" if cfg["cuda"] else "cpu")

        self.args = SimpleNamespace(**cfg)
        Path(self.args.savedir).mkdir(parents=True, exist_ok=True)

        self.dataset = None
        self.trainer = None

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    def preprocess(
        self,
        ref_path: str,
        target_path: str,
        species1_name: str,
        species2_name: str,
        p_homo_df: str,
        **overrides,
    ) -> "CrossSpeciesModel":
        """
        Build graph datasets from two h5ad files plus a homologous-gene CSV.
        Saves ref_data.pt / target_data.pt under savedirbase for reuse.
        """
        from datasets import extract_and_preprocess_data, load_dataset, read_dataset

        for k, v in overrides.items():
            setattr(self.args, k, v)
        self.args.ref_path = ref_path
        self.args.target_path = target_path
        self.args.species1_name = species1_name
        self.args.species2_name = species2_name
        self.args.p_homo_df = p_homo_df

        adata_ref, adata_tgt, homo_df = read_dataset(
            ref_path, target_path,
            species1_name, species2_name,
            p_homo_df, skip_QC=self.args.skip_QC,
        )

        ref_h, ref_nh, tgt_h, tgt_nh = extract_and_preprocess_data(
            self.args, adata_ref, adata_tgt,
            species1_name, species2_name, homo_df,
            self.args.homo_gene_id_ref, self.args.homo_gene_id_target,
        )

        dataset, inv_ref, inv_tgt = load_dataset(
            self.args, ref_h, ref_nh, tgt_h, tgt_nh,
            self.args.k_ref, self.args.k_target,
            self.args.celltype_name_ref, self.args.celltype_name_target,
            self.args.loc_name_ref, self.args.loc_name_target,
            self.args.region_name_ref, self.args.region_name_target,
            self.args.identity_graph,
        )

        base = Path(self.args.savedirbase)
        import pickle
        (base / "inverse_dict_ref.pkl").write_bytes(pickle.dumps(inv_ref))
        (base / "inverse_dict_target.pkl").write_bytes(pickle.dumps(inv_tgt))
        torch.save(dataset.ref_data, base / "ref_data.pt")
        torch.save(dataset.target_data, base / "target_data.pt")

        self.dataset = dataset
        return self

    def load_preprocessed(self) -> "CrossSpeciesModel":
        from datasets import GraphDataset
        base = Path(self.args.savedirbase)
        self.dataset = GraphDataset.from_saved(
            str(base / "ref_data.pt"), str(base / "target_data.pt")
        )
        return self

    # ------------------------------------------------------------------
    # Training (sync — only useful for short tests / notebooks)
    # ------------------------------------------------------------------
    def train(self) -> dict:
        if self.dataset is None:
            self.load_preprocessed()
        from staged_trainer import StagedTrainer
        self.trainer = StagedTrainer(self.args, self.dataset)
        self.trainer.train()
        return {"status": "done"}

    # ------------------------------------------------------------------
    # Training (async — used by the agent / web app)
    # ------------------------------------------------------------------
    def train_async(self) -> RunHandle:
        """
        Launch main_run_staged.py in a detached subprocess. Returns a handle
        immediately. The web app polls `status()` for progress.
        """
        savedir = Path(self.args.savedir)
        savedir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "main_run_staged.py"),
            "--savedir", self.args.savedirbase,
            "--name", self.args.name,
            "--seed", str(self.args.seed),
            "--epochs_per_stage", str(self.args.epochs_per_stage),
            "--lr", str(self.args.lr),
            "--wd", str(self.args.wd),
            "-b", str(self.args.batch_size),
            "--GAT_head", str(self.args.GAT_head),
            "--latent_dim", str(self.args.latent_dim),
            "--hidden_dim", str(self.args.hidden_dim),
            "--alpha", str(self.args.alpha),
            "--beta_cls", str(self.args.beta_cls),
            "--beta_kl", str(self.args.beta_kl),
            "--beta_intra", str(self.args.beta_intra),
            "--beta_homo", str(self.args.beta_homo),
            "--beta_bridge", str(self.args.beta_bridge),
            "--cls_on", str(self.args.cls_on),
            "--spot_size", str(self.args.spot_size),
        ]
        for flag in ("staged_training", "use_nonhomo", "VAE", "denoise",
                     "condition", "identity_graph"):
            if getattr(self.args, flag, False):
                cmd.append(f"--{flag}")

        log_path = savedir / "train.log"
        proc = subprocess.Popen(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPTS_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        (savedir / "pid").write_text(str(proc.pid))

        return RunHandle(run_id=self.args.name, savedir=savedir, pid=proc.pid)

    # ------------------------------------------------------------------
    # Status / control
    # ------------------------------------------------------------------
    @staticmethod
    def status(savedir: str | Path) -> dict:
        """Parse train.log for the latest PROGRESS line and process state."""
        savedir = Path(savedir)
        log_path = savedir / "train.log"
        pid_path = savedir / "pid"
        out: dict[str, Any] = {"savedir": str(savedir)}

        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            out["pid"] = pid
            out["alive"] = _pid_alive(pid)
        else:
            out["alive"] = False

        if not log_path.exists():
            out["status"] = "not_started"
            return out

        tail = _tail(log_path, 200)
        progress = _parse_progress(tail)
        out.update(progress)

        if "TRAINING COMPLETE" in tail:
            out["status"] = "done"
            # Process is gone; the PID slot may have been reused by the OS,
            # so override the (potentially misleading) liveness check.
            out["alive"] = False
        elif "Traceback" in tail and not out.get("alive"):
            out["status"] = "crashed"
        elif out.get("alive"):
            out["status"] = "running"
        else:
            out["status"] = "stopped"
        return out

    @staticmethod
    def stop(savedir: str | Path) -> dict:
        savedir = Path(savedir)
        pid_path = savedir / "pid"
        if not pid_path.exists():
            return {"stopped": False, "reason": "no pid file"}
        pid = int(pid_path.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(1)
            if _pid_alive(pid):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            return {"stopped": True, "pid": pid}
        except ProcessLookupError:
            return {"stopped": False, "reason": "process already gone"}

    # ------------------------------------------------------------------
    # Predict (sync)
    # ------------------------------------------------------------------
    def predict(self) -> dict:
        from torch_geometric.loader import ClusterData, ClusterLoader
        if self.trainer is None:
            raise RuntimeError("No trainer in memory. Use the artifacts saved to "
                               "savedir from a completed run instead.")
        ref, tgt = self.dataset.ref_data, self.dataset.target_data
        bs = self.args.batch_size
        rl = ClusterLoader(
            ClusterData(ref, num_parts=bs, recursive=False),
            batch_size=1, shuffle=False,
        )
        tl = ClusterLoader(
            ClusterData(tgt, num_parts=bs, recursive=False),
            batch_size=1, shuffle=False,
        )
        return self.trainer.pred(rl, tl)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PROGRESS_RE = re.compile(r"PROGRESS\s+(\{.*\})")
_STAGE_RE = re.compile(r"Training:\s+(Stage\d+_\w+)")


def _tail(path: Path, n_lines: int) -> str:
    with path.open("rb") as f:
        try:
            f.seek(-8192 * 4, os.SEEK_END)
        except OSError:
            f.seek(0)
        data = f.read().decode("utf-8", errors="ignore")
    return "\n".join(data.splitlines()[-n_lines:])


def _parse_progress(tail: str) -> dict:
    out: dict[str, Any] = {}
    matches = _PROGRESS_RE.findall(tail)
    if matches:
        try:
            out["progress"] = json.loads(matches[-1].replace("'", '"'))
        except Exception:
            pass
    stages = _STAGE_RE.findall(tail)
    if stages:
        out["current_stage"] = stages[-1]
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
