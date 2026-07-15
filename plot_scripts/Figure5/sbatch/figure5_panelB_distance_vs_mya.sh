#!/bin/bash
# =============================================================================
# Figure 5 — PANEL B : 4a_distance_vs_mya_three_metrics.png  (CAMEX_testis)
#   Importance distance from Human vs phylogenetic divergence (MYA), one panel
#   per metric (Spearman / top-50 Jaccard / L1), one line per cell type.
#   Reads results/Figure5/source_tensor/ (43 KB prebuilt importance tensor).
#   CPU only, seconds.
#   CAVEAT: the published PNG HAS NO CALLER in the repo — it came from an ad-hoc
#   session with unrecorded arguments. The call here reconstructs them from
#   evidence (titles read off the figure, rho from summary.json, palette
#   recovered from the legend pixels); font_scale is unknown and defaults to
#   1.6. Faithful redraw of the same data, NOT a bit-for-bit reproduction.
#   See results/Figure5/README.md.
# =============================================================================
#SBATCH --job-name=fig5_panelB_dist_vs_mya
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelB_dist_vs_mya_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelB_dist_vs_mya_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_multi/plot_phylo_panels.py --panel B
