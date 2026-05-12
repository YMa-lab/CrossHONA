"""
Tool registry for the agent. Every tool is a plain Python function with
typed args; the agent layer turns them into OpenAI function-calling schemas.

All path-taking tools resolve inputs against DATA_ROOT or RESULTS_ROOT and
reject paths that escape those directories.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CROSSSPECIES_DATA_DIR", "/app/data")).resolve()
RESULTS_ROOT = Path(os.environ.get("CROSSSPECIES_RESULTS_DIR", "/app/results")).resolve()
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def safe_path(user_path: str, root: Path) -> Path:
    """
    Resolve user_path under root; raise if the path itself escapes root.
    Symlinks inside root pointing outside are allowed (used for read-only
    references to large shared datasets) — we only check that the
    user-supplied path doesn't traverse out via "..".
    """
    if Path(user_path).is_absolute():
        p = Path(user_path)
    else:
        p = root / user_path
    # collapse .. segments without following symlinks
    p_norm = Path(os.path.normpath(p))
    try:
        p_norm.relative_to(root)
    except ValueError:
        raise ValueError(f"path {p_norm} is outside the allowed root {root}")
    return p_norm


from .data import inspect_dataset, list_datasets  # noqa: E402
from .training import check_status, list_runs, preprocess, stop_run, train  # noqa: E402
from .viz import plot_loss_curves, plot_umap  # noqa: E402

# (name -> callable). Order is the order shown to the LLM.
TOOLS = {
    "list_datasets": list_datasets,
    "inspect_dataset": inspect_dataset,
    "preprocess": preprocess,
    "train": train,
    "check_status": check_status,
    "stop_run": stop_run,
    "list_runs": list_runs,
    "plot_umap": plot_umap,
    "plot_loss_curves": plot_loss_curves,
}
