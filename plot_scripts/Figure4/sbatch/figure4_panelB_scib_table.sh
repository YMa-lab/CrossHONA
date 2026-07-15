#!/bin/bash
# =============================================================================
# Figure 4 — PANEL B : crosshona_vs_methods/scib_table.png  (sc_small_intestine)
#   scIB benchmark table: CrossHONA homo/nonhomo/combined + scVI, scGen,
#   NicheFormer, CAMEX. Reads results/Figure4/figure4_source.h5ad.
#   GPU REQUIRED — scib_metrics' numba kernels fail on the login node.
#
#   TWO PASSES, both needed to match the published panel:
#     1. benchmark  — computes metrics, writes benchmarker.pkl
#     2. --plot_only --font_scale 0.75 --width_scale 0.65
#                   — re-renders at the published sizing (~seconds)
#   Pass 1's default sizing (1.4/0.75) overflows the text.
# =============================================================================
#SBATCH --job-name=fig4_panelB_scib_table
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelB_scib_table_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure4/sbatch/fig4_panelB_scib_table_e.txt

source ~/stellar_py/bin/activate

# Pass 1 — benchmark (GPU)
python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_metrics/plot_scib_table.py \
    --figure Figure4 \
    --cell_col "celltype_coarse" \
    --batch_col "species"

# Pass 2 — re-render at the published sizing
python /oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_metrics/plot_scib_table.py \
    --figure Figure4 \
    --plot_only \
    --font_scale 0.75 --width_scale 0.65
