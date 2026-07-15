"""
Staged Training with Visualization at Each Level

Training stages:
1. Stage 0: Reconstruction only (baseline)
2. Stage 1: + Intra-species alignment (homo <-> nonhomo within species)
3. Stage 2: + Cross-species homo alignment (ref_homo <-> tgt_homo)
4. Stage 3: + Bridged full alignment (propagate through homo)

After each stage, we save:
- Loss curves
- Reconstruction quality plots
- UMAP embeddings (homo only, nonhomo only, full)
- Integration metrics
"""

import os
import json
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
import pickle

from torch.utils.data import DataLoader, TensorDataset

import models

from utils.plotting import plot_stage_results, plot_confusion_matrices
from utils.basic_metrics import compute_basic_metrics

warnings.filterwarnings("ignore")


def _make_tensor_dataset(data_obj):
    """
    Build a TensorDataset from a graph Data object, yielding per-node rows.
    Fields included: homo_x, nonhomo_x, y, pos.
    edge_index is dropped — the CVAE model no longer uses it.
    """
    return TensorDataset(
        data_obj.homo_x,
        data_obj.nonhomo_x,
        data_obj.y,
        data_obj.pos,
    )


class _Batch:
    """Lightweight struct that mimics the old PyG Data batch interface."""
    __slots__ = ("homo_x", "nonhomo_x", "y", "pos", "edge_index")

    def __init__(self, homo_x, nonhomo_x, y, pos, device):
        self.homo_x    = homo_x.to(device)
        self.nonhomo_x = nonhomo_x.to(device)
        self.y         = y.to(device)
        self.pos       = pos.to(device)
        self.edge_index = None   # kept so any stray access doesn't crash


class StagedTrainer:
    """
    Staged training with visualization after each alignment level.
    """

    # Proto loss (cell-type-aware cross-species alignment) is enabled in
    # Stage2/3 only — needs latent to be partially aligned before prototypes
    # are trustworthy enough to drive target pseudo-labels.
    STAGES = [
        {"name": "Stage0_Reconstruction",   "intra": 0.0, "bridge": 0.0, "proto": 0.0},
        {"name": "Stage1_IntraSpecies",     "intra": 1.0, "bridge": 0.0, "proto": 0.0},
        {"name": "Stage2_Proto",            "intra": 1.0, "bridge": 0.0, "proto": 1.0},
        {"name": "Stage3_BridgedFull",      "intra": 1.0, "bridge": 1.0, "proto": 1.0},
    ]

    def __init__(self, args, dataset):
        self.args    = args
        self.dataset = dataset

        self.scib_leiden_res  = float(getattr(args, "scib_leiden_res",  0.5))
        self.scib_neighbors_k = int(getattr(args,   "scib_neighbors_k", 15))

        # Input dimensions
        args.shared_input_dim  = dataset.ref_data.homo_x.shape[-1]
        args.ref_input_dim     = dataset.ref_data.nonhomo_x.shape[-1]
        args.target_input_dim  = dataset.target_data.nonhomo_x.shape[-1]
        args.num_classes       = int(torch.unique(dataset.ref_data.y).numel())

        # Per-class weighting for the cls loss — sqrt-inverse-frequency,
        ref_y_arr = dataset.ref_data.y.numpy()
        counts = np.bincount(ref_y_arr, minlength=args.num_classes).astype(float)
        counts[counts == 0] = 1.0
        class_w = 1.0 / np.sqrt(counts)
        class_w = class_w / class_w.mean()
        args.class_weights_t = torch.tensor(class_w, dtype=torch.float32, device=args.device)
        print(f"[cls weighting] sqrt_inverse weights={class_w.round(3).tolist()}")

        # Model
        self.model = models.cross_GAE_VAE_Hierarchical(
            shared_x_dim=args.shared_input_dim,
            ref_x_dim=args.ref_input_dim,
            target_x_dim=args.target_input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            denoise=getattr(args, "denoise",    True),
            beta_intra=args.beta_intra,
            beta_bridge=args.beta_bridge,
            cls_on=getattr(args, "cls_on",        "mix"),
            use_nonhomo=getattr(args, "use_nonhomo", True),
            beta_proto=getattr(args, "beta_proto", 0.0),
            proto_temperature=getattr(args, "proto_temperature", 0.1),
            proto_conf_ratio=getattr(args, "proto_conf_ratio", 0.0),
            film_emb_dim=getattr(args, "film_emb_dim", 16),
            class_weights=args.class_weights_t,
        ).to(args.device)

        # Plain node-level DataLoaders (no graph partitioning needed)
        ref_ds = _make_tensor_dataset(dataset.ref_data)
        tgt_ds = _make_tensor_dataset(dataset.target_data)

        self.ref_loader_train = DataLoader(
            ref_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        self.tgt_loader_train = DataLoader(
            tgt_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True
        )
        self.ref_loader_eval = DataLoader(
            ref_ds, batch_size=args.batch_size, shuffle=False
        )
        self.tgt_loader_eval = DataLoader(
            tgt_ds, batch_size=args.batch_size, shuffle=False
        )

        self.current_weights = {"cls": 0.0, "intra": 0.0, "bridge": 0.0, "proto": 0.0}
        self.all_metrics     = {}

        ref_path = os.path.join(args.savedirbase, "inverse_dict_ref.pkl")
        tgt_path = os.path.join(args.savedirbase, "inverse_dict_target.pkl")

        if os.path.exists(ref_path):
            with open(ref_path, "rb") as f:
                self.inv_ref = pickle.load(f)

        if os.path.exists(tgt_path):
            with open(tgt_path, "rb") as f:
                self.inv_tgt = pickle.load(f)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, optimizer, epoch: int, pretrain: bool = False) -> Dict[str, float]:
        self.model.train()

        sums = {k: 0.0 for k in [
            "loss", "recon", "cls", "kl",
            "align_intra_ref", "align_intra_tgt", "align_bridge",
            "proto_ref", "proto_tgt", "proto_frac_kept",
        ]}

        # cycle() on a DataLoader iterator doesn't restart it reliably —
        # instead we keep a reference to the loader and reinitialize the
        # iterator manually whenever it runs dry.
        def _infinite(loader):
            while True:
                yield from loader
        tgt_iter = _infinite(self.tgt_loader_train)

        for idx, ref_tensors in enumerate(tqdm(self.ref_loader_train, desc=f"Epoch {epoch}", leave=False)):
            ref_batch = _Batch(*ref_tensors, device=self.args.device)
            tgt_batch = _Batch(*next(tgt_iter), device=self.args.device)

            optimizer.zero_grad()

            outputs = self.model(ref_data=ref_batch, target_data=tgt_batch)

            beta_cls = 0.0 if pretrain else self.args.beta_cls

            loss_dict = self.model.loss_function(
                ref_homo_x=ref_batch.homo_x,
                ref_nonhomo_x=ref_batch.nonhomo_x,
                target_homo_x=tgt_batch.homo_x,
                target_nonhomo_x=tgt_batch.nonhomo_x,
                latent_dict=outputs["latent"],
                ref_logits=outputs["ref_logits"],
                ref_y=ref_batch.y,
                alpha_recon=self.args.alpha,
                beta_cls=beta_cls,
                beta_kl=self.args.beta_kl,
                tgt_logits=outputs.get("target_logits"),
                align_weights=self.current_weights,
            )

            loss_dict["loss"].backward()
            optimizer.step()

            sums["loss"]            += loss_dict["loss"].item()
            sums["recon"]           += loss_dict["Reconstruction_Loss"].item()
            sums["cls"]             += loss_dict["Classification_Loss"].item()
            sums["kl"]              += loss_dict["KL_Loss"].item()
            sums["align_intra_ref"] += float(loss_dict["Align_Intra_Ref"].item())
            sums["align_intra_tgt"] += float(loss_dict["Align_Intra_Tgt"].item())
            sums["align_bridge"]    += float(loss_dict["Align_Bridge"].item())
            sums["proto_ref"]       += float(loss_dict.get("Proto_Ref", torch.tensor(0.0)).item())
            sums["proto_tgt"]       += float(loss_dict.get("Proto_Tgt", torch.tensor(0.0)).item())
            sums["proto_frac_kept"] += float(loss_dict.get("Proto_FracTgtKept", torch.tensor(0.0)).item())

        n = idx + 1
        return {k: v / n for k, v in sums.items()}

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embeddings(self) -> Dict[str, np.ndarray]:
        """Extract all embeddings for visualization + metrics."""
        self.model.eval()

        collectors = {
            "ref": {k: [] for k in [
                "homo_mean", "nonhomo_mean", "homo_z", "nonhomo_z",
                "recon_homo", "recon_nonhomo", "raw_homo", "raw_nonhomo",
                "logits", "y", "pos"
            ]},
            "tgt": {k: [] for k in [
                "homo_mean", "nonhomo_mean", "homo_z", "nonhomo_z",
                "recon_homo", "recon_nonhomo", "raw_homo", "raw_nonhomo",
                "logits", "y", "pos"
            ]},
        }

        for tensors in self.ref_loader_eval:
            batch = _Batch(*tensors, device=self.args.device)
            out   = self.model(ref_data=batch)
            lat   = out["latent"]

            collectors["ref"]["homo_mean"].append(lat["ref_homo_mean"].cpu())
            collectors["ref"]["nonhomo_mean"].append(lat["ref_nonhomo_mean"].cpu())
            collectors["ref"]["homo_z"].append(lat["ref_homo_latent"].cpu())
            collectors["ref"]["nonhomo_z"].append(lat["ref_nonhomo_latent"].cpu())
            collectors["ref"]["recon_homo"].append(lat["ref_homo_mu"].cpu())
            collectors["ref"]["recon_nonhomo"].append(lat["ref_nonhomo_mu"].cpu())
            collectors["ref"]["raw_homo"].append(batch.homo_x.cpu())
            collectors["ref"]["raw_nonhomo"].append(batch.nonhomo_x.cpu())
            collectors["ref"]["logits"].append(out["ref_logits"].cpu())
            collectors["ref"]["y"].append(batch.y.cpu())
            collectors["ref"]["pos"].append(batch.pos.cpu())

        for tensors in self.tgt_loader_eval:
            batch = _Batch(*tensors, device=self.args.device)
            out   = self.model(target_data=batch)
            lat   = out["latent"]

            collectors["tgt"]["homo_mean"].append(lat["target_homo_mean"].cpu())
            collectors["tgt"]["nonhomo_mean"].append(lat["target_nonhomo_mean"].cpu())
            collectors["tgt"]["homo_z"].append(lat["target_homo_latent"].cpu())
            collectors["tgt"]["nonhomo_z"].append(lat["target_nonhomo_latent"].cpu())
            collectors["tgt"]["recon_homo"].append(lat["target_homo_mu"].cpu())
            collectors["tgt"]["recon_nonhomo"].append(lat["target_nonhomo_mu"].cpu())
            collectors["tgt"]["raw_homo"].append(batch.homo_x.cpu())
            collectors["tgt"]["raw_nonhomo"].append(batch.nonhomo_x.cpu())
            collectors["tgt"]["logits"].append(out["target_logits"].cpu())
            collectors["tgt"]["y"].append(batch.y.cpu())
            collectors["tgt"]["pos"].append(batch.pos.cpu())

        result = {}
        for species in ["ref", "tgt"]:
            for key, values in collectors[species].items():
                result[f"{species}_{key}"] = torch.cat(values).numpy()

        return result

    # ------------------------------------------------------------------
    # Stage control
    # ------------------------------------------------------------------

    def train_stage(self, stage_config: Dict, epochs: int, optimizer) -> Tuple[Dict, Dict, Dict, Dict]:
        stage_name = stage_config["name"]
        pretrain   = (stage_name == "Stage0_Reconstruction")

        beta_proto = getattr(self.args, "beta_proto", 0.0)
        proto_w    = stage_config.get("proto", 0.0) * beta_proto

        if not getattr(self.args, "use_nonhomo", True):
            self.current_weights = {
                "intra":  0.0,
                "bridge": 0.0,
                "proto":  proto_w,
            }
        else:
            self.current_weights = {
                "intra":  stage_config["intra"]  * self.args.beta_intra,
                "bridge": stage_config["bridge"] * self.args.beta_bridge,
                "proto":  proto_w,
            }

        print(f"\n{'=' * 60}")
        print(f"Training: {stage_name}")
        print(f"  Applied weights: {self.current_weights}")
        print(f"{'=' * 60}")

        losses: Dict[str, List[float]] = {k: [] for k in [
            "loss", "recon", "cls", "kl",
            "align_intra_ref", "align_intra_tgt", "align_bridge",
            "proto_ref", "proto_tgt", "proto_frac_kept",
        ]}

        for ep in range(epochs):
            epoch_losses = self.train_epoch(optimizer, ep, pretrain=pretrain)
            for k, v in epoch_losses.items():
                losses[k].append(v)

            if (ep + 1) % 10 == 0:
                print(f"  Epoch {ep+1}: loss={epoch_losses['loss']:.4f}, "
                      f"recon={epoch_losses['recon']:.4f}, cls={epoch_losses['cls']:.4f}")

        print("  Extracting embeddings and computing metrics...")
        embeddings = self.extract_embeddings()

        metrics = compute_basic_metrics(
            embeddings,
            condition=True,
        )

        print(f"  Ref Acc: {metrics['accuracy_ref']:.4f}, Tgt Acc: {metrics['accuracy_tgt']:.4f}")
        # print(f"  Ref Weighted F1: {metrics['weightedF1_ref']:.4f}, Tgt Weighted F1: {metrics['weightedF1_tgt']:.4f}")
        print(f"  Ref ARI: {metrics['ari_ref']:.4f}, Tgt ARI: {metrics['ari_tgt']:.4f}")
        # print(f"  Ref NMI: {metrics['nmi_ref']:.4f}, Tgt NMI: {metrics['nmi_tgt']:.4f}")
        # print(f"  Silhouette (homo): {metrics['silhouette_homo_mean']:.4f}")
        # print(f"  Silhouette (full): {metrics['silhouette_full']:.4f}")

        return losses, embeddings, metrics

    def train(self):
        optimizer        = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        epochs_per_stage = getattr(self.args, "epochs_per_stage", 30)

        all_results = {}

        for stage_config in self.STAGES:
            stage_name = stage_config["name"]
            stage_dir  = os.path.join(self.args.savedir, stage_name)
            os.makedirs(stage_dir, exist_ok=True)

            losses, embeddings, metrics = self.train_stage(
                stage_config, epochs_per_stage, optimizer
            )

            with open(os.path.join(stage_dir, f"{stage_name}_metrics.json"), "w") as f:
                json.dump(
                    {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()},
                    f, indent=2,
                )

            for key, value in embeddings.items():
                np.save(os.path.join(stage_dir, f"{key}.npy"), value)

            with open(os.path.join(stage_dir, f"{stage_name}_losses.json"), "w") as f:
                json.dump(losses, f)

            all_results[stage_name] = {"losses": losses, "metrics": metrics}

            # plot_confusion_matrices(embeddings, stage_dir, stage_name,
            #                         inv_ref=self.inv_ref, inv_tgt=self.inv_tgt)
            if stage_name == 'Stage3_BridgedFull':
                plot_stage_results(stage_name, embeddings, losses, stage_dir, True, self.inv_ref, self.inv_tgt)
            torch.save(self.model.state_dict(), os.path.join(stage_dir, "model_checkpoint.pt"))

        with open(os.path.join(self.args.savedir, "all_stage_metrics.json"), "w") as f:
            json.dump(
                {
                    stage: {k: float(v) if np.isfinite(v) else None for k, v in res["metrics"].items()}
                    for stage, res in all_results.items()
                },
                f, indent=2,
            )

        return all_results

    def pred(self, ref_loader=None, tgt_loader=None) -> Dict:
        embeddings = self.extract_embeddings()
        ref_logits = embeddings["ref_logits"]
        tgt_logits = embeddings["tgt_logits"]
        preds_ref  = np.argmax(ref_logits, axis=1)
        preds_tgt  = np.argmax(tgt_logits, axis=1)

        return {
            "preds_ref":         preds_ref,
            "y_ref":             embeddings["ref_y"],
            "latent_ref":        np.hstack([embeddings["ref_homo_z"], embeddings["ref_nonhomo_z"]]),
            "preds_tgt":         preds_tgt,
            "y_tgt":             embeddings["tgt_y"],
            "latent_tgt":        np.hstack([embeddings["tgt_homo_z"], embeddings["tgt_nonhomo_z"]]),
            "latent_ref_homo":   embeddings["ref_homo_z"],
            "latent_tgt_homo":   embeddings["tgt_homo_z"],
            "ref_homo_mean":     embeddings["ref_homo_mean"],
            "ref_nonhomo_mean":  embeddings["ref_nonhomo_mean"],
            "tgt_homo_mean":     embeddings["tgt_homo_mean"],
            "tgt_nonhomo_mean":  embeddings["tgt_nonhomo_mean"],
            "recon_homo_ref":    embeddings["ref_recon_homo"],
            "recon_nonhomo_ref": embeddings["ref_recon_nonhomo"],
            "recon_homo_tgt":    embeddings["tgt_recon_homo"],
            "recon_nonhomo_tgt": embeddings["tgt_recon_nonhomo"],
            "pos_ref":           embeddings["ref_pos"],
            "pos_tgt":           embeddings["tgt_pos"],
            "raw_homo_ref":      embeddings["ref_raw_homo"],
            "raw_nonhomo_ref":   embeddings["ref_raw_nonhomo"],
            "raw_homo_tgt":      embeddings["tgt_raw_homo"],
            "raw_nonhomo_tgt":   embeddings["tgt_raw_nonhomo"],
        }
