#!/bin/bash
# =============================================================================
# Figure 2 — PANEL A : barplot_acc_ari_target.png
#   Grouped Accuracy / ARI bars, one bar per method, scored on the target
#   species (Mouse). Reads results/Figure2/figure2_source.h5ad.
#   CPU only — no GPU needed (labels-only metrics).
# =============================================================================
#SBATCH --job-name=fig2_panelA_barplot
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch/fig2_panelA_barplot_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch/fig2_panelA_barplot_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_metrics/plot_barplot_acc_ari.py \
    --dataset MERFISH_Cortex \
    --species1 Human --species2 Mouse \
    --cell_col "cell_type" \
    --subset target
