"""
Reproduce the MERFISH smoke test using the CrossSpeciesModel API directly.

Equivalent to scripts_hierarchical/run_staged_training_MERFISH.sh but as
a Python script instead of a shell wrapper. Uses async training so the
script returns immediately after launching; poll status from another shell
or extend this with a wait loop.

Usage:
    conda activate crossspecies
    export PYTHONNOUSERSITE=1
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    cd $REPO_ROOT/CrossHONA-Agent
    python run_merfish_demo.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from api import CrossSpeciesModel

# Match CrossHONA/run_staged_training_MERFISH.sh
DATA = Path("data/merfish")
PROJECT = Path("results/merfish_demo")
RUN_NAME = "Merfish_NonHomo"


def main():
    # 1. preprocess (sync, ~10 s) ------------------------------------------
    m = CrossSpeciesModel(
        savedir=PROJECT, name="_preproc",
        skip_QC=True,
        nonhomo_hvg=1000,
        identity_graph=True,
        homo_gene_id_ref="Gene name",
        homo_gene_id_target="Gene name",
    )
    m.preprocess(
        ref_path=str(DATA / "human_H19_STG_4000.h5ad"),
        target_path=str(DATA / "mouse_mouse1_242.h5ad"),
        species1_name="Human",
        species2_name="Mouse",
        p_homo_df=str(DATA / "Human_Mouse.tsv"),
    )
    print(f"Preprocess done. Files in {PROJECT}:")
    for p in sorted(PROJECT.iterdir()):
        if p.is_file():
            print(f"  {p.name}")

    # 2. train (async, returns immediately) --------------------------------
    runner = CrossSpeciesModel(
        savedir=PROJECT, name=RUN_NAME,
        epochs_per_stage=30, lr=3e-4, batch_size=20,
        alpha=1.0, beta_cls=1.0, beta_kl=0.01,
        beta_intra=1.0, beta_homo=1.0, beta_bridge=1.0,
        cls_on="mix",
        VAE=True, denoise=True, condition=True, identity_graph=True,
        use_nonhomo=True, staged_training=True,
    )
    handle = runner.train_async()
    print(f"\nTraining started: pid={handle.pid}, log={handle.log_path}")

    # 3. poll until done (optional — comment out to exit immediately) ------
    print("\nPolling status every 30 s. Ctrl+C to detach.\n")
    while True:
        time.sleep(30)
        status = CrossSpeciesModel.status(handle.savedir)
        print(json.dumps(status, indent=2))
        if status.get("status") == "done":
            print("\n✓ Training complete.")
            break
        if not status.get("alive") and status.get("status") != "done":
            print("\n✗ Training process died early. See train.log.")
            break


if __name__ == "__main__":
    main()
