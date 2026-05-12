#!/usr/bin/env bash
# Mouse-only re-run with proto_conf_ratio=0, mirroring the Marmoset conf0 fix
# (Marmoset went from acc=0.43 to 0.94). Result subdir: NonHomo_conf0/.

cd codedir

HOMO_DF_BASE="datadir"
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
PROTO_CONF_RATIO=0   # <-- changed from 0.5

NONHOMO_FLAG="--use_nonhomo"
PREPROCESS_FLAG="--preprocess"

echo "================= CAMEX testis Human vs Mouse (conf_ratio=0) ================="
python main_run_staged.py \
  --ref_path        "$REF_PATH" \
  --target_path     "$DATA_BASE/raw-testis-mouse-Murat.h5ad" \
  --name            NonHomo_conf0 \
  --savedir         "$RESULT_BASE/Human_Mouse" \
  --species1_name   "$SPECIES1" \
  --species2_name   "Mouse" \
  --p_homo_df       "$HOMO_DF_BASE/homolog_genes/Human_Mouse.tsv" \
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
  --proto_conf_ratio "$PROTO_CONF_RATIO" \
  --epochs_per_stage "$EPOCHS_PER_STAGE" \
  --lr              "$LR" \
  --nonhomo_hvg     1000 \
  --batch_size      1024 \
  --cls_on          homo \
  --skip_QC \
  $PREPROCESS_FLAG \
  $NONHOMO_FLAG
