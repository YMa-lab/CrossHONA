#!/bin/bash
# =============================================================================
# Figure 3 — PANELS A + B in one job (convenience; equivalent to the two
# per-panel scripts above).
#   A : 1f_combined_heatmap_IG_pred.png   (predicted class)
#   B : 1f_combined_heatmap_IG_true.png   (ground-truth class)
# =============================================================================
#SBATCH --job-name=fig3_panelsAB_IG
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch/fig3_panelsAB_IG_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch/fig3_panelsAB_IG_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_IG/plot_ig_combined_heatmap.py --panel both --cmap viridis
