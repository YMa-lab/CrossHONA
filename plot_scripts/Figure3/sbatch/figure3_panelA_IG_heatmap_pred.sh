#!/bin/bash
# =============================================================================
# Figure 3 — PANEL A : 1f_combined_heatmap_IG_pred.png
#   IG attribution combined heatmap, attributions taken against the MODEL'S
#   PREDICTED class. Three blocks: homolog pairs (Human|Mouse, top 30),
#   Human-specific (top 30), Mouse-specific (top 30). Values log1p(mean |attr|).
#   Reads results/Figure3/source_csv/cls_importance_*_IG.csv  (no _true suffix).
#   CPU only, seconds. Verified pixel-identical to the published panel.
# =============================================================================
#SBATCH --job-name=fig3_panelA_IG_pred
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch/fig3_panelA_IG_pred_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure3/sbatch/fig3_panelA_IG_pred_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_IG/plot_ig_combined_heatmap.py --panel A --cmap viridis
