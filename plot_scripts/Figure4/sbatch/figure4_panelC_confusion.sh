#!/bin/bash
# =============================================================================
# Figure 4 — PANEL C : confusion_target.png   (sc_small_intestine)
#   Per-method cell-type confusion matrices on the target species (Mouse).
#   Reads results/Figure4/figure4_source.h5ad.
#   --normalize true is REQUIRED to match the published panel.
#   NOTE --cell_col is 'celltype_coarse' for this dataset.
#   CPU only. Verified pixel-identical to the published panel.
# =============================================================================
#SBATCH --job-name=fig4_panelC_confusion
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelC_confusion_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelC_confusion_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_metrics/plot_confusion_all_methods.py \
    --figure Figure4 \
    --dataset sc_small_intestine \
    --species1 Human --species2 Mouse \
    --cell_col "celltype_coarse" \
    --subset target \
    --normalize true
