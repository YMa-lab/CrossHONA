#!/bin/bash
# =============================================================================
# Figure 5 — PANEL C : Fig3a_celltype_smai_mean_dots.png     (SMAI, CAMEX_testis)
#   Cell-type-specific SMAI-test: mean p-value bars per embedding
#   (homo/nonhomo/combined) with all 10 pairwise p-values as dots, p=0.05 line.
#   Reads results/Figure5/source_csv/celltype_smai/ (15 CSVs, ~550 B each).
#   CPU only, seconds. Verified pixel-identical to the published panel.
# =============================================================================
#SBATCH --job-name=fig5_panelC_smai_pvals
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelC_smai_pvals_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelC_smai_pvals_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_multi/plot_smai_panels.py --panel C
