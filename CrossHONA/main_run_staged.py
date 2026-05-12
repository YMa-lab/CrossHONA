import argparse
import numpy as np
import os
import sys
import shlex
import torch
import pickle 
from datasets import *
from utils.preprocess import *
import random
import time
from staged_trainer import StagedTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():
    parser = argparse.ArgumentParser(description='Staged Training with Visualization')
    
    # ==================== Path and Data ====================
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--name', type=str, default='run')
    parser.add_argument('--ref_path', type=str, default=None)
    parser.add_argument('--target_path', type=str, default=None)
    parser.add_argument('--species1_name', type=str, default=None)
    parser.add_argument('--species2_name', type=str, default=None)
    parser.add_argument('--p_homo_df', type=str, default=None)
    
    # ==================== Preprocessing ====================
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--celltype_name_ref', type=str, default='cell_type')
    parser.add_argument('--celltype_name_target', type=str, default='cell_type')
    parser.add_argument('--skip_QC', action='store_true')
    parser.add_argument('--hvg', type=int, default=5000)
    parser.add_argument('--nonhomo_hvg', type=int, default=1000)
    parser.add_argument('--homo_gene_id_ref', type=str, default='Gene name')
    parser.add_argument('--homo_gene_id_target', type=str, default='gene name')

    # ==================== Training ====================
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs_per_stage', type=int, default=30,
                        help='Number of epochs for each training stage')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', default=10, type=int)

    # ==================== Model Architecture ====================
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)

    # ==================== Model Variants ====================
    parser.add_argument('--use_nonhomo', action='store_true', help='use non-homologous branch')
    parser.add_argument('--film_emb_dim', type=int, default=16,
                        help='species embedding dim used by FiLM (default 16)')
    parser.add_argument('--beta_proto', type=float, default=0.0,
                        help='weight for prototype-based cell-type-aware alignment '
                             '(InfoNCE on ref labels + tgt pseudo-labels). 0 = off.')
    parser.add_argument('--proto_temperature', type=float, default=0.1,
                        help='temperature for the prototype InfoNCE')
    parser.add_argument('--proto_conf_ratio', type=float, default=0.5,
                        help='confidence filter on tgt pseudo-labels: only keep tgt '
                             'cells whose nearest-prototype cosine distance is '
                             '< conf_ratio * second-nearest distance. 0 = no filter. '
                             'Default 0.5 works for IR>10 datasets with rare classes; '
                             'use 0 for balanced datasets with biologically adjacent '
                             'classes (e.g. CAMEX testis).')


    # ==================== Loss Weights ====================
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta_cls', type=float, default=1.0)
    parser.add_argument('--beta_kl', type=float, default=0.01)
    
    # ==================== Hierarchical Alignment ====================
    parser.add_argument('--beta_intra', type=float, default=1.0)
    parser.add_argument('--beta_bridge', type=float, default=0.5)
    parser.add_argument('--cls_on', type=str, default='mix',
                        choices=['homo', 'mix'],
                        help='which latent feeds the classifier. "mix" (default) '
                             'uses homo+nonhomo via attention fusion; "homo" uses '
                             'only the cross-species-aligned homo space.')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed(args.seed)
    start_time = time.time()

    # ==================== Data Loading ====================
    if args.preprocess:
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        print(f"The saving directory set to {args.savedir}", flush=True)
        
        adata_ref, adata_target, homo_genes_df = read_dataset(
            args.ref_path,
            args.target_path,
            args.species1_name,
            args.species2_name,
            args.p_homo_df,
            skip_QC=args.skip_QC
        )

        adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo = \
            extract_and_preprocess_data(
                args,
                adata_ref,
                adata_target,
                args.species1_name,
                args.species2_name,
                homo_genes_df,
                args.homo_gene_id_ref,
                args.homo_gene_id_target
            )
    
        dataset, inverse_dict_ref, inverse_dict_target = load_dataset(
            args,
            adata_ref_homo,
            adata_ref_nonhomo,
            adata_target_homo,
            adata_target_nonhomo,
            args.celltype_name_ref,
            args.celltype_name_target,
        )
        
        os.makedirs(args.savedir, exist_ok=True)
        with open(f'{args.savedir}/inverse_dict_ref.pkl', 'wb') as f:
            pickle.dump(inverse_dict_ref, f)
        with open(f'{args.savedir}/inverse_dict_target.pkl', 'wb') as f:
            pickle.dump(inverse_dict_target, f)
        
        torch.save(dataset.ref_data, os.path.join(args.savedir, 'ref_data.pt'))
        torch.save(dataset.target_data, os.path.join(args.savedir, 'target_data.pt'))
        print(f"Saved graph data to {args.savedir}")
    else: 
        dataset = GraphDataset.from_saved(
            f'{args.savedir}/ref_data.pt', 
            f'{args.savedir}/target_data.pt'
        )

    # ==================== Setup Save Directory ====================
    args.savedirbase = args.savedir
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    print(f"Results will be saved to: {args.savedir}", flush=True)

    # Save command
    try:
        with open(os.path.join(args.savedir, "run_command.sh"), "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("python " + " ".join(shlex.quote(a) for a in sys.argv) + "\n")
    except:
        pass
    
    # ==================== Staged Training ====================
    print("\n" + "="*70)
    print("STAGED TRAINING WITH VISUALIZATION")
    print("="*70)
    print(f"\nThis will train in 4 stages:")
    print(f"  Stage 0: Reconstruction only (baseline)")
    print(f"  Stage 1: + Intra-species alignment (homo ↔ nonhomo)")
    print(f"  Stage 2: + Cross-species homo alignment (ref ↔ tgt)")
    print(f"  Stage 3: + Bridged full alignment")
    print(f"\nEpochs per stage: {args.epochs_per_stage}")
    print(f"Total epochs: {args.epochs_per_stage * 4}")
    print("="*70 + "\n")
    
    trainer = StagedTrainer(args, dataset)
    results = trainer.train()
    
    # ==================== Final Outputs ====================
    outputs = trainer.pred()
    
    # Save final outputs
    save_map = {
        "preds_ref": "ref_pred.npy",
        "y_ref": "ref_gt.npy",
        "latent_ref": "ref_latent.npy",
        "preds_tgt": "target_pred.npy",
        "y_tgt": "target_gt.npy",
        "latent_tgt": "target_latent.npy",
        "latent_ref_homo": "ref_latent_homo.npy",
        "latent_tgt_homo": "tgt_latent_homo.npy",
        "ref_homo_mean": "ref_homo_mean.npy",
        "ref_nonhomo_mean": "ref_nonhomo_mean.npy",
        "tgt_homo_mean": "tgt_homo_mean.npy",
        "tgt_nonhomo_mean": "tgt_nonhomo_mean.npy",
        "recon_homo_ref": "recon_homo_ref.npy",
        "recon_homo_tgt": "recon_homo_tgt.npy",
        "recon_nonhomo_ref": "recon_nonhomo_ref.npy",
        "recon_nonhomo_tgt": "recon_nonhomo_tgt.npy",
        "pos_ref": "ref_pos.npy",
        "pos_tgt": "tgt_pos.npy",
        "raw_homo_ref": "raw_homo_ref.npy",
        "raw_nonhomo_ref": "raw_nonhomo_ref.npy",
        "raw_homo_tgt": "raw_homo_tgt.npy",
        "raw_nonhomo_tgt": "raw_nonhomo_tgt.npy",
    }

    for key, filename in save_map.items():
        if key in outputs:
            np.save(os.path.join(args.savedir, filename), outputs[key])

    # ==================== Summary ====================
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nResults saved to: {args.savedir}")
    print(f"\nStage directories:")
    for stage in trainer.STAGES:
        stage_dir = os.path.join(args.savedir, stage['name'])
        print(f"  - {stage['name']}/")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
