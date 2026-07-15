"""save_fig — verbatim from biological_interpretability/plots/utils.py.

Trimmed to the one helper the Figure 5 phylo panels use.
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

