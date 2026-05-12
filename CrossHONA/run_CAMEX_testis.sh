#!/usr/bin/env bash
# CAMEX testis cross-species benchmark: Human vs (Gibbon, Gorilla, Marmoset, Mouse).
# Configuration: proto-only baseline (BETA_PROTO=1),
# cls_on=homo, intra=0, denoise + size_factor + FiLM (always on now).

cd codedir

# ==================== Common Paths ====================
HOMO_DF_BASE="datadir"
DATA_BASE="datadir/testis"
RESULT_BASE="resultdir/CAMEX_testis"

REF_PATH="$DATA_BASE/raw-testis-human-Murat.h5ad"
HOMO_GENE_ID='Gene name'
SPECIES1="Human"

# ==================== Training Parameters ====================
EPOCHS_PER_STAGE=80
LR="3e-4"

# ==================== Loss Weights ====================
ALPHA=1
BETA_CLS=10
BETA_KL=0.01
BETA_INTRA=1.0       # disabled (no benefit per ablation)
BETA_BRIDGE=1.0
BETA_PROTO=1.0       # proto contrastive on

NONHOMO_FLAG="--use_nonhomo"
PREPROCESS_FLAG="--preprocess"

# Usage: run_one <name> <target_path> <species2> <homo_df_path> <preprocess_flag>
run_one () {
    local NAME="$1"
    local TGT_PATH="$2"
    local SPECIES2="$3"
    local HOMO_DF="$4"
    local PRE_FLAG="$5"

    python main_run_staged.py \
      --ref_path        "$REF_PATH" \
      --target_path     "$TGT_PATH" \
      --name            NonHomo \
      --savedir         "$RESULT_BASE/$NAME" \
      --species1_name   "$SPECIES1" \
      --species2_name   "$SPECIES2" \
      --p_homo_df       "$HOMO_DF" \
      --homo_gene_id_ref "$HOMO_GENE_ID" \
      --homo_gene_id_target "$HOMO_GENE_ID" \
      --celltype_name_ref cell_ontology_class \
      --celltype_name_target cell_ontology_class \
      --alpha           "$ALPHA" \
      --beta_cls        "$BETA_CLS" \
      --beta_kl         "$BETA_KL" \
      --beta_intra      "$BETA_INTRA" \
      --beta_bridge     "$BETA_BRIDGE" \
      --beta_proto      "$BETA_PROTO" \
      --epochs_per_stage "$EPOCHS_PER_STAGE" \
      --lr              "$LR" \
      --nonhomo_hvg     1000 \
      --batch_size      2048 \
      --skip_QC \
      $PRE_FLAG \
      $NONHOMO_FLAG
}

echo "================= CAMEX testis Human vs Gibbon ================="
run_one "Human_Gibbon" \
        "$DATA_BASE/raw-testis-gibbon-Murat.h5ad" \
        "Gibbon" \
        "$DATA_BASE/homolog_genes/Human_Gibbon.tsv" \
        "$PREPROCESS_FLAG"

echo "================= CAMEX testis Human vs Gorilla ================="
run_one "Human_Gorilla" \
        "$DATA_BASE/raw-testis-gorilla-Murat.h5ad" \
        "Gorilla" \
        "$DATA_BASE/homolog_genes/Human_Gorilla.tsv" \
        "$PREPROCESS_FLAG"

echo "================= CAMEX testis Human vs Macaque ================="
run_one "Human_Macaque" \
        "$DATA_BASE/raw-testis-macaque-Murat.h5ad" \
        "Macaque" \
        "$DATA_BASE/homolog_genes/Human_Macaque.tsv" \
        "$PREPROCESS_FLAG"

echo "================= CAMEX testis Human vs Marmoset ================="
run_one "Human_Marmoset" \
        "$DATA_BASE/raw-testis-marmoset-Murat.h5ad" \
        "Marmoset" \
        "$DATA_BASE/homolog_genes/Human_Marmoset.tsv" \
        "$PREPROCESS_FLAG"

echo "================= CAMEX testis Human vs Mouse ================="
run_one "Human_Mouse" \
        "$DATA_BASE/raw-testis-mouse-Murat.h5ad" \
        "Mouse" \
        "$HOMO_DF_BASE/homolog_genes/Human_Mouse.tsv" \
        "$PREPROCESS_FLAG"
