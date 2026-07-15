#!/bin/bash
# =============================================================================
# Figure 3 — PANEL B : 1f_combined_heatmap_IG_true.png
#   Same heatmap as panel A, but attributions aggregated by GROUND-TRUTH class
#   instead of the predicted one.
#   Reads results/Figure3/source_csv/cls_importance_*_IG_true.csv.
#   CPU only, seconds. Verified pixel-identical to the published panel.
# =============================================================================
#SBATCH --job-name=fig3_panelB_IG_true
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch/fig3_panelB_IG_true_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch/fig3_panelB_IG_true_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_IG/plot_ig_combined_heatmap.py --panel B --cmap viridis
