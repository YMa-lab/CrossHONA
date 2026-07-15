"""Shared plot helpers.

Trimmed from biological_interpretability/plots/utils.py to what the Figure 3 IG
heatmaps use: save_fig (called by plot_ig_combined_heatmap) and species_palette
(the ref/tgt species stripe above the homolog block). The unused siblings
divergent_palette and maybe_basename are dropped, along with the `os` / Optional
imports they alone needed. Both functions below are verbatim from upstream.

Upstream original: scripts_benchmarks_new/biological_interpretability/plots/utils.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt


def save_fig(
    fig,
    save_path: Union[str, Path],
    dpi: int = 200,
    also_pdf: bool = True,
) -> str:
    """Save a figure to PNG (+ PDF), close it, return PNG path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    png = save_path.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(png)


_REF_COLOR = "#432371"   # deep purple
_TGT_COLOR = "#faae7b"   # warm peach


def species_palette() -> dict:
    """Default colours; ref = deep purple, tgt = warm peach."""
    return {"ref": _REF_COLOR, "tgt": _TGT_COLOR}
