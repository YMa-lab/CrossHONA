#!/bin/bash
# =============================================================================
# Figure 2 — PANEL C : confusion_target.png
#   Per-method cell-type confusion matrices on the target species (Mouse).
#   Reads results/Figure2/figure2_source.h5ad.
#   --normalize true is REQUIRED to match the published panel (script default
#   is 'none'); it also gates the shared colorbar.
#   CPU only — no GPU needed.
# =============================================================================
#SBATCH --job-name=fig2_panelC_confusion
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch/fig2_panelC_confusion_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch/fig2_panelC_confusion_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_metrics/plot_confusion_all_methods.py \
    --dataset MERFISH_Cortex \
    --species1 Human --species2 Mouse \
    --cell_col "cell_type" \
    --subset target \
    --normalize true
