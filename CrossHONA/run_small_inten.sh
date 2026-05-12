#!/usr/bin/env bash
cd codedir

# ==================== Data Paths ====================
REF_PATH="datadir/sc_small_intestine/human_SI.h5ad"
TARGET_PATH="datadir/sc_small_intestine/mouse_SI.h5ad"
HOMO_DF_PATH="datadir/homolog_genes/Human_Mouse.tsv"
SAVEDIR="resultdir/02_results_small_intestine"

HOMO_GENE_ID='Gene name'
SPECIES1="Human"
SPECIES2="Mouse"

# ==================== Training Parameters ====================
EPOCHS_PER_STAGE=80      # Epochs for each stage
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
NONHOMO_NAME="NonHomo"
HOMO_NAME="noNonHomo"

python main_run_staged.py \
  --ref_path        "$REF_PATH" \
  --target_path     "$TARGET_PATH" \
  --name            "$NONHOMO_NAME" \
  --savedir         "$SAVEDIR" \
  --species1_name   "$SPECIES1" \
  --species2_name   "$SPECIES2" \
  --celltype_name_ref celltype_coarse \
  --celltype_name_target celltype_coarse \
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
  --celltype_name_ref celltype_coarse \
  --celltype_name_target celltype_coarse \
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
  --skip_QC
