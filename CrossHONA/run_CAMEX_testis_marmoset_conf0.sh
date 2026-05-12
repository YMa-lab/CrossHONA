#!/usr/bin/env bash
# Marmoset-only re-run with proto_conf_ratio=0 to test whether the spermatids
# collapse on Marmoset is caused by conf_ratio=0.5 filtering out low-confidence
# spermatids samples from the proto contrastive loss.
# Other 4 species (Gibbon/Gorilla/Macaque/Mouse) work fine with conf_ratio=0.5
# at acc=0.93-0.95, so this is a Marmoset-specific knob test.
# Result subdir overrides: NonHomo_conf0/  (so existing NonHomo/ stays intact)

cd codedir

DATA_BASE="datadir/testis"
RESULT_BASE="resultdir/CAMEX_testis"

REF_PATH="$DATA_BASE/raw-testis-human-Murat.h5ad"
HOMO_GENE_ID='Gene name'
SPECIES1="Human"

EPOCHS_PER_STAGE=80
LR="3e-4"

ALPHA=1
BETA_CLS=10
BETA_KL=0.01
BETA_INTRA=1.0
BETA_BRIDGE=1.0
BETA_PROTO=1.0

NONHOMO_FLAG="--use_nonhomo"
PREPROCESS_FLAG="--preprocess"

echo "================= CAMEX testis Human vs Marmoset (conf_ratio=0) ================="
python main_run_staged.py \
  --ref_path        "$REF_PATH" \
  --target_path     "$DATA_BASE/raw-testis-marmoset-Murat.h5ad" \
  --name            NonHomo_conf0 \
  --savedir         "$RESULT_BASE/Human_Marmoset" \
  --species1_name   "$SPECIES1" \
  --species2_name   "Marmoset" \
  --p_homo_df       "$DATA_BASE/homolog_genes/Human_Marmoset.tsv" \
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
  --cls_on          homo \
  --skip_QC \
  $PREPROCESS_FLAG \
  $NONHOMO_FLAG
