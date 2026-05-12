#!/usr/bin/env bash
cd codedir

# ==================== Data Paths ====================
REF_PATH="datadir/Merfish_brain/human_H19_STG_4000.h5ad"
TARGET_PATH="datadir/Merfish_brain/mouse_mouse1_242.h5ad"
HOMO_DF_PATH="datadir/homolog_genes/Human_Mouse.tsv"
SAVEDIR="resultdir/02_results_MERFISH"

HOMO_GENE_ID='Gene name'
SPECIES1="Human"
SPECIES2="Mouse"

# ==================== Training Parameters ====================
EPOCHS_PER_STAGE=50      # Epochs for each stage
LR="3e-4"

# ==================== Loss Weights ====================
ALPHA=1              # Reconstruction weight
BETA_CLS=10          # Classification weight
BETA_KL=0.01         # KL divergence weight

# ==================== Hierarchical Alignment Weights ====================
BETA_INTRA=1.0       # Level 1: Intra-species alignment
BETA_BRIDGE=1.0      # Level 2: Bridged full alignment (CORAL)
BETA_PROTO=1.0       # Cell-type-aware proto contrastive (Stage2/3)

# ==================== Flags ====================
PREPROCESS_FLAG="--preprocess"
NONHOMO_FLAG="--use_nonhomo"
NONHOMO_NAME="Merfish_NonHomo"
HOMO_NAME="Merfish_noNonHomo"

python main_run_staged.py \
  --ref_path        "$REF_PATH" \
  --target_path     "$TARGET_PATH" \
  --name            "$NONHOMO_NAME" \
  --savedir         "$SAVEDIR" \
  --species1_name   "$SPECIES1" \
  --species2_name   "$SPECIES2" \
  --p_homo_df       "$HOMO_DF_PATH" \
  --homo_gene_id_ref "$HOMO_GENE_ID" \
  --homo_gene_id_target "$HOMO_GENE_ID" \
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
  $PREPROCESS_FLAG \
  $NONHOMO_FLAG

python main_run_staged.py \
  --ref_path        "$REF_PATH" \
  --target_path     "$TARGET_PATH" \
  --name            "$HOMO_NAME" \
  --savedir         "$SAVEDIR" \
  --species1_name   "$SPECIES1" \
  --species2_name   "$SPECIES2" \
  --p_homo_df       "$HOMO_DF_PATH" \
  --homo_gene_id_ref "$HOMO_GENE_ID" \
  --homo_gene_id_target "$HOMO_GENE_ID" \
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
  $PREPROCESS_FLAG
