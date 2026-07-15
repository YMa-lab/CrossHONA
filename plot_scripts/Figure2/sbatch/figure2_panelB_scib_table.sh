#!/bin/bash
# =============================================================================
# Figure 2 — PANEL B : crosshona_vs_methods/scib_table.png
#   scIB benchmark table: CrossHONA homo/nonhomo/combined + scVI, scGen,
#   NicheFormer, CAMEX. Reads results/Figure2/figure2_source.h5ad.
#   GPU REQUIRED — scib_metrics' numba kernels fail on the login node.
#
#   TWO PASSES, both needed to match the published panel:
#     1. benchmark  — computes the metrics, writes benchmarker.pkl (~25 min GPU)
#     2. --plot_only --font_scale 0.75 --width_scale 0.65
#                   — re-renders the table at the published sizing (~3 s)
#   Pass 1's default sizing (1.4/0.75) overflows the text and yields a 2899px
#   wide table; pass 2 gives the published 2515px. Verified pixel-identical.
#   Re-tuning sizing later needs only pass 2 — the pkl is already on disk.
# =============================================================================
#SBATCH --job-name=fig2_panelB_scib_table
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --chdir=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch
#SBATCH --output=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch/fig2_panelB_scib_table_o.txt
#SBATCH --error=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/Figure2/sbatch/fig2_panelB_scib_table_e.txt

source ~/stellar_py/bin/activate

FN=/oscar/data/yma16/Project/Cross_species/STAI-X_code/plot_scripts/functions/plot_metrics

# Pass 1 — run the benchmark, write benchmarker.pkl (GPU, ~25 min)
python ${FN}/plot_scib_table.py \
    --cell_col "cell_type" \
    --batch_col "species"

# Pass 2 — re-render at the published sizing (seconds, reuses the pkl above)
python ${FN}/plot_scib_table.py \
    --plot_only \
    --font_scale 0.75 --width_scale 0.65
