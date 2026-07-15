"""Plot helpers for the Figure 3 IG heatmaps.

Trimmed from the upstream biological_interpretability/plots/__init__.py, which
eagerly re-exported q1/q2/q3 symbols. Only q1_plots.plot_ig_combined_heatmap and
its .utils helpers are used here, and the eager q2/q3 imports pulled in modules
that have nothing to do with Figure 3. Verified to leave the panels
byte-identical.
"""
from .utils import save_fig, species_palette  # noqa: F401
