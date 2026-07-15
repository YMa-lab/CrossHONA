#!/bin/bash
# =============================================================================
# Figure 5 — PANEL E : trees_homo_meanL1_UPGMA_percelltype.png (SMAI, CAMEX_testis)
#   Per-cell-type species trees (homo, mean L1, UPGMA) vs the TimeTree truth,
#   titles annotated with nRF and TreeCor.
#   Reads results/Figure5/source_csv/distmat/ (5 CSVs, ~634 B each).
#   The TimeTree truth is hardcoded in the script (MRCA ages 9/20/29/43 Mya,
#   Mouse split 87), mirroring celltype_specific_trees.R.
#   CPU only, seconds. Verified pixel-identical to the published panel.
# =============================================================================
#SBATCH --job-name=fig5_panelE_trees
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelE_trees_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelE_trees_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_multi/plot_smai_panels.py --panel E
