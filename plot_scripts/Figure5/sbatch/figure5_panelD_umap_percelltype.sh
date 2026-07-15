#!/bin/bash
# =============================================================================
# Figure 5 — PANEL D : umap_homo_percelltype_by_species.png  (SMAI, CAMEX_testis)
#   Per-cell-type UMAP of the homo cell-type-specific alignment, one panel per
#   cell type, colored by species.
#   Reads results/Figure5/source_csv/umap_homo_percelltype_coords.csv (2.8 MB).
#   CPU only, seconds. Verified pixel-identical to the published panel.
#   NOTE that coords CSV has NO producer in version control (see README) — it is
#   shipped here precisely because it cannot be regenerated.
# =============================================================================
#SBATCH --job-name=fig5_panelD_umap
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelD_umap_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure5/sbatch/fig5_panelD_umap_e.txt

source ~/stellar_py/bin/activate

python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_multi/plot_smai_panels.py --panel D
