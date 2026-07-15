#!/bin/bash
# =============================================================================
# Figure 5 — PANEL A : 4c_top3_per_celltype.png            (CAMEX_testis)
#   Top-3 contributing genes per cell type, species x gene heatmap of
#   log1p(|attribution|), rows ordered by phylogenetic distance from Human.
#   Reads results/Figure5/source_tensor/ (43 KB prebuilt importance tensor).
#   CPU only, seconds.
#   CAVEAT: the published PNG is STALE — it predates a plots.py edit, so the
#   committed code produces 1616x772 vs the published 1618x772. This script is
#   bitwise identical to what the current upstream pipeline produces.
# =============================================================================
#SBATCH --job-name=fig5_panelA_top3_genes
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelA_top3_genes_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelA_top3_genes_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_multi/plot_phylo_panels.py --panel A
