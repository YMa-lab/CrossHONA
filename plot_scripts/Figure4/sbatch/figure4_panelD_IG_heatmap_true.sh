#!/bin/bash
# =============================================================================
# Figure 4 — PANEL D : 1f_combined_heatmap_IG_true.png  (sc_small_intestine)
#   IG attribution combined heatmap, attributions aggregated by GROUND-TRUTH
#   class. Reads results/Figure4/source_csv/cls_importance_*_IG_true.csv.
#   Uses font_scale 1.8 (sc_small_intestine's own setting; MERFISH uses 1.6).
#   CPU only, seconds.
#   CAVEAT: reproduces the published panel's data and layout exactly, but the
#   published PNG is offset ~1px horizontally in text rendering (see the
#   Figure4 README). Not reproducible from the committed code — the original
#   renderer produces the same output as this script.
# =============================================================================
#SBATCH --job-name=fig4_panelD_IG_true
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelD_IG_true_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelD_IG_true_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_IG/plot_ig_combined_heatmap.py --figure Figure4 --variant true --cmap viridis
