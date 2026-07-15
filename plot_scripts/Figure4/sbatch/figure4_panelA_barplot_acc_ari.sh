#!/bin/bash
# =============================================================================
# Figure 4 — PANEL A : barplot_acc_ari_target.png   (sc_small_intestine)
#   Grouped Accuracy / ARI bars, one per method, scored on the target species
#   (Mouse). Reads results/Figure4/figure4_source.h5ad.
#   NOTE --cell_col is 'celltype_coarse' for this dataset (not 'cell_type').
#   CPU only. Verified pixel-identical to the published panel.
# =============================================================================
#SBATCH --job-name=fig4_panelA_barplot
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelA_barplot_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelA_barplot_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_metrics/plot_barplot_acc_ari.py \
    --figure Figure4 \
    --dataset sc_small_intestine \
    --species1 Human --species2 Mouse \
    --cell_col "celltype_coarse" \
    --subset target
