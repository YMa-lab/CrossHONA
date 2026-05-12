import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from .loss_hierarchical import (
    log_nb_positive,
    HierarchicalAlignmentLoss,
    intra_species_alignment_loss,
    bridged_full_alignment,
    proto_contrastive_loss,
)


class MLP(nn.Module):
    """Classifier head. Two modes:
    - cls_on='homo': pure linear on homo_mean. attention/cls_mix not built,
      so cls_homo gets the full CE gradient and there's no parameter waste.
    - cls_on='mix': attention-weighted fusion of cls_homo(homo) and
      cls_mix(homo+nonhomo). nonhomo_mean must be passed in.
    """
    def __init__(self, latent_dim: int, output_dim: int, cls_on: str = "mix"):
        super().__init__()
        self.cls_on = cls_on
        self.cls_homo = nn.Linear(latent_dim, output_dim)
        if cls_on == "mix":
            self.attention = nn.Sequential(
                nn.Linear(latent_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=1)
            )
            self.cls_mix = nn.Linear(latent_dim, output_dim)

    def forward(self, homo_mean: torch.Tensor, nonhomo_mean: torch.Tensor) -> torch.Tensor:
        if self.cls_on == "homo":
            return self.cls_homo(homo_mean)
        mix_mean = homo_mean + nonhomo_mean
        concat = torch.cat([homo_mean, nonhomo_mean], dim=1)
        weights = self.attention(concat)
        logits_homo = self.cls_homo(homo_mean)
        logits_mix = self.cls_mix(mix_mean)
        logits = weights[:, 0:1] * logits_homo + weights[:, 1:2] * logits_mix
        return logits_mix


class MLPEncoder(nn.Module):
    """MLP encoder. Optional FiLM modulation by a (gamma, beta) pair injected
    after the linear layer, before activation."""
    def __init__(self, in_dim: int, hidden_dim: int, heads: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index=None,
                film_gamma: Optional[torch.Tensor] = None,
                film_beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.linear(x)
        if film_gamma is not None:
            h = film_gamma * h + film_beta
        return F.elu(h)


class MLPLatentHead(nn.Module):
    """MLP mu/logvar head: replaces SharedLatentHead. edge_index accepted but ignored."""
    def __init__(self, hidden_dim: int, latent_dim: int, heads: int = 1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.constant_(self.logvar.bias, -5)

    def forward(self, h: torch.Tensor, edge_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.elu(self.fc(h))
        mean = self.mean(h)
        logvar = torch.clamp(self.logvar(h), -10.0, 10.0)
        return mean, logvar


class MLPDecoder(nn.Module):
    """MLP decoder. Outputs NB(mu, theta) parameters.

    If `library_size` is provided, decoder predicts the gene-fraction simplex
    (softmax) and multiplies by the per-cell library size to get mu — this is
    the scVI parameterisation. The encoder is then free of the library-size
    nuisance signal.

    If `library_size` is None, falls back to the legacy softplus(logits) form
    where the decoder predicts absolute counts directly.

    `n_theta_groups` controls how many independent dispersion vectors are
    stored. n=1 (default) is the legacy single-group; n=2 lets ref/tgt have
    distinct per-gene dispersions (selected at forward time via `theta_idx`).
    Only relevant for the homo decoder, which is shared across species; the
    nonhomo decoders are already species-specific and can leave n=1.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 1,
        n_theta_groups: int = 1,
    ):
        super().__init__()
        self.dec1_linear = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, out_dim)
        self.n_theta_groups = n_theta_groups
        self.log_theta = nn.Parameter(torch.zeros(n_theta_groups, out_dim))

    def forward(
        self,
        z: torch.Tensor,
        edge_index=None,
        library_size: Optional[torch.Tensor] = None,
        theta_idx: int = 0,
        film_gamma: Optional[torch.Tensor] = None,
        film_beta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.dec1_linear(z)
        if film_gamma is not None:
            h = film_gamma * h + film_beta
        h = F.elu(h)
        logits = self.dec2(h)
        if library_size is not None:
            rho = F.softmax(logits, dim=1)
            mu = rho * library_size.view(-1, 1)
        else:
            mu = F.softplus(logits)
        theta = torch.exp(self.log_theta[theta_idx])
        return mu, theta


class cross_GAE_VAE_Hierarchical(nn.Module):
    """
    Cross-species CVAE with Hierarchical Alignment.

    Graph attention layers have been replaced with plain MLPs so that no
    adjacency / edge_index is required during the encode/decode steps.
    edge_index that arrives in the data batches is accepted but silently
    ignored, keeping the rest of the training pipeline unchanged.

    The condition indicator (species one-hot appended to homo_x) is preserved,
    so this remains a proper CVAE where the condition is baked into the input.

    Three-level alignment (unchanged):
    - Level 1 (Intra-Species): homo <-> nonhomo within each species
    - Level 2 (Cross-Species Homo): ref_homo <-> tgt_homo
    - Level 3 (Bridged Full): propagate alignment through homo anchor
    """

    def __init__(
        self,
        shared_x_dim: int,
        ref_x_dim: int,
        target_x_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_classes: int = None,
        denoise: bool = True,
        noise_std: float = 0.1,
        beta_intra: float = 1.0,
        beta_bridge: float = 0.5,
        cls_on: str = "mix",
        use_nonhomo: bool = True,
        beta_proto: float = 0.0,
        proto_temperature: float = 0.1,
        proto_conf_ratio: float = 0.0,
        film_emb_dim: int = 16,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Training-time input perturbation: Gaussian noise on the input.
        # Acts as a regulariser that "lifts" the zero entries (>80% of the
        # input in sparse count data), forcing the encoder away from
        # memorising ref-specific sparse patterns. Crucial for cross-species
        # generalisation since target zero-patterns differ from ref.
        self.denoise = denoise
        self.noise_std = noise_std if denoise else 0.0
        self.cls_on = cls_on
        self.use_nonhomo = use_nonhomo

        # The model is always: scVI-style size-factor decoder + FiLM species
        # conditioning. The 1-dim concat-indicator pathway has been removed.
        self.homo_x_dim = shared_x_dim
        self.shared_x_dim = shared_x_dim
        self.ref_x_dim = ref_x_dim
        self.tgt_x_dim = target_x_dim
        self.latent_dim = latent_dim

        # ====== Encoders / Decoders ======
        # Homo decoder is shared across species but holds two dispersion
        # vectors (theta_idx=0 -> ref, theta_idx=1 -> tgt) so per-species
        # NB dispersion can be honest about each platform's noise level.
        # Always uses scVI-style size-factor decoder.
        self.homo_enc = MLPEncoder(self.homo_x_dim, hidden_dim)
        self.homo_decoder = MLPDecoder(latent_dim, hidden_dim, shared_x_dim, n_theta_groups=2)

        if self.use_nonhomo:
            self.ref_nonhomo_enc = MLPEncoder(ref_x_dim, hidden_dim)
            self.tgt_nonhomo_enc = MLPEncoder(target_x_dim, hidden_dim)
            self.ref_nonhomo_decoder = MLPDecoder(latent_dim, hidden_dim, ref_x_dim)
            self.tgt_nonhomo_decoder = MLPDecoder(latent_dim, hidden_dim, target_x_dim)
        else:
            self.ref_nonhomo_enc = None
            self.tgt_nonhomo_enc = None
            self.ref_nonhomo_decoder = None
            self.tgt_nonhomo_decoder = None

        # ====== Species conditioning via FiLM ======
        # 2 species, embedded into film_emb_dim. Two separate (gamma, beta)
        # generators: one for the encoder pathway, one for the decoder.
        # gamma is parameterised as 1 + g(emb) so that at init g≈0 -> γ≈1,
        # making FiLM an identity transform until training picks up signal.
        self.species_emb = nn.Embedding(2, film_emb_dim)
        self.film_enc_gamma = nn.Linear(film_emb_dim, hidden_dim)
        self.film_enc_beta  = nn.Linear(film_emb_dim, hidden_dim)
        self.film_dec_gamma = nn.Linear(film_emb_dim, hidden_dim)
        self.film_dec_beta  = nn.Linear(film_emb_dim, hidden_dim)
        for layer in (self.film_enc_gamma, self.film_enc_beta,
                      self.film_dec_gamma, self.film_dec_beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # ====== Latent Heads ======
        # homo_latent is SHARED across species — both ref and target homo
        # features pass through the same weights, which anchors them in the
        # same latent geometry and supports cross-species alignment.
        self.homo_latent = MLPLatentHead(hidden_dim, latent_dim)

        # nonhomo heads are SPECIES-SPECIFIC — ref and target nonhomo genes
        # are entirely different gene sets with different distributions, so
        # forcing them through shared weights is unnecessarily constraining.
        if self.use_nonhomo:
            self.ref_nonhomo_latent = MLPLatentHead(hidden_dim, latent_dim)
            self.tgt_nonhomo_latent = MLPLatentHead(hidden_dim, latent_dim)

        # ====== Classifiers ======
        # When use_nonhomo=False, force cls_on='homo' since there's no nonhomo
        # branch to fuse with.
        effective_cls_on = "homo" if not self.use_nonhomo else cls_on
        self.classifier_homo = MLP(latent_dim, num_classes, cls_on=effective_cls_on)

        # ====== Hierarchical Alignment ======
        self.hierarchical_align = HierarchicalAlignmentLoss(
            beta_intra=beta_intra,
            beta_bridge=beta_bridge,
        )

        self.beta_intra = beta_intra
        self.beta_bridge = beta_bridge
        self.beta_proto = beta_proto
        self.proto_temperature = proto_temperature
        self.proto_conf_ratio = proto_conf_ratio
        self.num_classes = num_classes

        # Per-class loss weights for the cls head. None -> uniform (default).
        # Registered as a buffer so .to(device) carries it along.
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        self.register_buffer("class_weights", class_weights.float())

    def _film_params(self, species_idx: int, n_cells: int, device, target: str):
        """Return (gamma, beta) tensors of shape (n_cells, hidden_dim) for the
        given species and target ('enc' or 'dec')."""
        idx = torch.full((n_cells,), species_idx, dtype=torch.long, device=device)
        e = self.species_emb(idx)
        if target == "enc":
            gamma = 1.0 + self.film_enc_gamma(e)
            beta  = self.film_enc_beta(e)
        else:
            gamma = 1.0 + self.film_dec_gamma(e)
            beta  = self.film_dec_beta(e)
        return gamma, beta

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _perturb_input(self, x: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian noise on the input. noise_std=0 -> identity."""
        if self.noise_std <= 0.0:
            return x
        return x + torch.randn_like(x) * self.noise_std

    def encode_ref(self, ref_data) -> Dict[str, torch.Tensor]:
        """Encode reference data."""
        rh = ref_data.homo_x
        rnh = ref_data.nonhomo_x
        B, device = rh.size(0), rh.device

        # Library sizes are computed from the *clean* counts (before noise).
        ls_h  = rh.sum(dim=1)
        ls_nh = rnh.sum(dim=1) if self.use_nonhomo else None

        rh_input = rh
        if self.training:
            rh_input = self._perturb_input(rh_input)
            rnh = self._perturb_input(rnh)

        # FiLM modulation for the homo encoder/decoder (species_idx=0 -> ref)
        f_enc_g, f_enc_b = self._film_params(0, B, device, "enc")
        f_dec_g, f_dec_b = self._film_params(0, B, device, "dec")

        h_h = self.homo_enc(rh_input, film_gamma=f_enc_g, film_beta=f_enc_b)
        rh_mean, rh_logvar = self.homo_latent(h_h)
        rh_z = self.reparameterize(rh_mean, rh_logvar)

        if self.use_nonhomo:
            h_n = self.ref_nonhomo_enc(rnh)
            rnh_mean, rnh_logvar = self.ref_nonhomo_latent(h_n)
            rnh_z = self.reparameterize(rnh_mean, rnh_logvar)
            rnh_mu, rnh_theta = self.ref_nonhomo_decoder(rnh_z, library_size=ls_nh)
        else:
            rnh_mean = torch.zeros(B, self.latent_dim, device=device)
            rnh_logvar = torch.zeros(B, self.latent_dim, device=device)
            rnh_z = torch.zeros(B, self.latent_dim, device=device)
            rnh_mu = torch.zeros(B, 0, device=device)
            rnh_theta = torch.zeros(0, device=device)

        rh_mu, rh_theta = self.homo_decoder(
            rh_z, library_size=ls_h, theta_idx=0,
            film_gamma=f_dec_g, film_beta=f_dec_b,
        )

        return {
            "homo_mean": rh_mean, "homo_logvar": rh_logvar, "homo_z": rh_z,
            "homo_mu": rh_mu, "homo_theta": rh_theta,
            "nonhomo_mean": rnh_mean, "nonhomo_logvar": rnh_logvar, "nonhomo_z": rnh_z,
            "nonhomo_mu": rnh_mu, "nonhomo_theta": rnh_theta,
            "mix_mean": rh_mean + rnh_mean,
        }

    def encode_target(self, target_data) -> Dict[str, torch.Tensor]:
        """Encode target data."""
        th = target_data.homo_x
        tnh = target_data.nonhomo_x
        B, device = th.size(0), th.device

        ls_h  = th.sum(dim=1)
        ls_nh = tnh.sum(dim=1) if self.use_nonhomo else None

        th_input = th
        if self.training:
            th_input = self._perturb_input(th_input)
            tnh = self._perturb_input(tnh)

        # FiLM modulation for the homo encoder/decoder (species_idx=1 -> tgt)
        f_enc_g, f_enc_b = self._film_params(1, B, device, "enc")
        f_dec_g, f_dec_b = self._film_params(1, B, device, "dec")

        h_h = self.homo_enc(th_input, film_gamma=f_enc_g, film_beta=f_enc_b)
        th_mean, th_logvar = self.homo_latent(h_h)
        th_z = self.reparameterize(th_mean, th_logvar)

        if self.use_nonhomo:
            h_n = self.tgt_nonhomo_enc(tnh)
            tnh_mean, tnh_logvar = self.tgt_nonhomo_latent(h_n)
            tnh_z = self.reparameterize(tnh_mean, tnh_logvar)
            tnh_mu, tnh_theta = self.tgt_nonhomo_decoder(tnh_z, library_size=ls_nh)
        else:
            tnh_mean = torch.zeros(B, self.latent_dim, device=device)
            tnh_logvar = torch.zeros(B, self.latent_dim, device=device)
            tnh_z = torch.zeros(B, self.latent_dim, device=device)
            tnh_mu = torch.zeros(B, 0, device=device)
            tnh_theta = torch.zeros(0, device=device)

        th_mu, th_theta = self.homo_decoder(
            th_z, library_size=ls_h, theta_idx=1,
            film_gamma=f_dec_g, film_beta=f_dec_b,
        )

        return {
            "homo_mean": th_mean, "homo_logvar": th_logvar, "homo_z": th_z,
            "homo_mu": th_mu, "homo_theta": th_theta,
            "nonhomo_mean": tnh_mean, "nonhomo_logvar": tnh_logvar, "nonhomo_z": tnh_z,
            "nonhomo_mu": tnh_mu, "nonhomo_theta": tnh_theta,
            "mix_mean": th_mean + tnh_mean,
        }

    def classify(self, homo_mean: torch.Tensor, nonhomo_mean: torch.Tensor) -> torch.Tensor:
        # MLP itself dispatches on its own cls_on. When use_nonhomo=False or
        # cls_on='homo' the MLP ignores the second arg, so passing nonhomo_mean
        # (or any tensor) is safe.
        return self.classifier_homo(homo_mean, nonhomo_mean)

    def forward(self, ref_data=None, target_data=None) -> Dict:
        outputs = {"latent": {}, "recons": {}}

        if ref_data is not None:
            ref_out = self.encode_ref(ref_data)
            outputs["latent"]["ref_homo_mean"] = ref_out["homo_mean"]
            outputs["latent"]["ref_homo_log_var"] = ref_out["homo_logvar"]
            outputs["latent"]["ref_homo_latent"] = ref_out["homo_z"]
            outputs["latent"]["ref_homo_mu"] = ref_out["homo_mu"]
            outputs["latent"]["ref_homo_theta"] = ref_out["homo_theta"]
            outputs["latent"]["ref_nonhomo_mean"] = ref_out["nonhomo_mean"]
            outputs["latent"]["ref_nonhomo_log_var"] = ref_out["nonhomo_logvar"]
            outputs["latent"]["ref_nonhomo_latent"] = ref_out["nonhomo_z"]
            outputs["latent"]["ref_nonhomo_mu"] = ref_out["nonhomo_mu"]
            outputs["latent"]["ref_nonhomo_theta"] = ref_out["nonhomo_theta"]
            outputs["latent"]["ref_mix_mean"] = ref_out["mix_mean"]
            outputs["ref_logits"] = self.classify(ref_out["homo_mean"], ref_out["nonhomo_mean"])

        if target_data is not None:
            tgt_out = self.encode_target(target_data)
            outputs["latent"]["target_homo_mean"] = tgt_out["homo_mean"]
            outputs["latent"]["target_homo_log_var"] = tgt_out["homo_logvar"]
            outputs["latent"]["target_homo_latent"] = tgt_out["homo_z"]
            outputs["latent"]["target_homo_mu"] = tgt_out["homo_mu"]
            outputs["latent"]["target_homo_theta"] = tgt_out["homo_theta"]
            outputs["latent"]["target_nonhomo_mean"] = tgt_out["nonhomo_mean"]
            outputs["latent"]["target_nonhomo_log_var"] = tgt_out["nonhomo_logvar"]
            outputs["latent"]["target_nonhomo_latent"] = tgt_out["nonhomo_z"]
            outputs["latent"]["target_nonhomo_mu"] = tgt_out["nonhomo_mu"]
            outputs["latent"]["target_nonhomo_theta"] = tgt_out["nonhomo_theta"]
            outputs["latent"]["target_mix_mean"] = tgt_out["mix_mean"]
            outputs["target_logits"] = self.classify(tgt_out["homo_mean"], tgt_out["nonhomo_mean"])

        return outputs

    def loss_function(
        self,
        ref_homo_x: torch.Tensor,
        ref_nonhomo_x: torch.Tensor,
        target_homo_x: torch.Tensor,
        target_nonhomo_x: torch.Tensor,
        latent_dict: Dict[str, torch.Tensor],
        ref_logits: torch.Tensor,
        ref_y: torch.Tensor,
        alpha_recon: float = 1.0,
        beta_cls: float = 1.0,
        beta_kl: float = 0.01,
        tgt_logits: torch.Tensor = None,
        align_weights: Dict[str, float] = None,
    ) -> Dict[str, torch.Tensor]:
        device = latent_dict["ref_homo_mean"].device

        # 1. Reconstruction Loss (NB)
        def _nb_nll(x, mu, theta):
            return -log_nb_positive(x, mu, theta).mean()

        ref_h_mu = latent_dict["ref_homo_mu"]
        tgt_h_mu = latent_dict["target_homo_mu"]
        ref_h_theta = latent_dict["ref_homo_theta"]
        tgt_h_theta = latent_dict["target_homo_theta"]

        loss_recon = _nb_nll(ref_homo_x, ref_h_mu, ref_h_theta) + \
                     _nb_nll(target_homo_x, tgt_h_mu, tgt_h_theta)
        if self.use_nonhomo:
            loss_recon = loss_recon + \
                _nb_nll(ref_nonhomo_x, latent_dict["ref_nonhomo_mu"], latent_dict["ref_nonhomo_theta"]) + \
                _nb_nll(target_nonhomo_x, latent_dict["target_nonhomo_mu"], latent_dict["target_nonhomo_theta"])

        # 2. KL Divergence
        def kl(m, lv):
            return -0.5 * torch.mean(1 + lv - m.pow(2) - lv.exp())

        loss_kl = kl(latent_dict["ref_homo_mean"], latent_dict["ref_homo_log_var"]) + \
                  kl(latent_dict["target_homo_mean"], latent_dict["target_homo_log_var"])
        if self.use_nonhomo:
            loss_kl = loss_kl + \
                kl(latent_dict["ref_nonhomo_mean"], latent_dict["ref_nonhomo_log_var"]) + \
                kl(latent_dict["target_nonhomo_mean"], latent_dict["target_nonhomo_log_var"])

        # 3. Classification Loss
        loss_cls = F.cross_entropy(ref_logits, ref_y, weight=self.class_weights)

        # 4. Hierarchical Alignment Loss
        # use_nonhomo=False: skip intra/bridge entirely (no nonhomo branch to
        # align with). Cross-species alignment then comes solely from the
        # prototype contrastive loss below.
        if self.use_nonhomo:
            align_result = self.hierarchical_align(
                ref_homo=latent_dict["ref_homo_mean"],
                ref_nonhomo=latent_dict["ref_nonhomo_mean"],
                tgt_homo=latent_dict["target_homo_mean"],
                tgt_nonhomo=latent_dict["target_nonhomo_mean"],
                return_components=True,
            )
            if align_weights is not None:
                loss_align = (
                    align_weights.get("intra", self.beta_intra)
                        * (align_result["intra_ref"] + align_result["intra_tgt"])
                    + align_weights.get("bridge", self.beta_bridge) * align_result["bridge"]
                )
            else:
                loss_align = align_result["total"]
        else:
            zero = torch.tensor(0.0, device=device)
            align_result = {
                "intra_ref": zero, "intra_tgt": zero, "bridge": zero, "total": zero,
            }
            loss_align = zero

        # 5. Prototype-based cell-type-aware alignment
        if align_weights is not None:
            w_proto = align_weights.get("proto", self.beta_proto)
        else:
            w_proto = self.beta_proto

        if w_proto > 0.0:
            proto_result = proto_contrastive_loss(
                ref_z=latent_dict["ref_homo_mean"],
                ref_y=ref_y,
                tgt_z=latent_dict["target_homo_mean"],
                num_classes=self.num_classes,
                temperature=self.proto_temperature,
                conf_ratio=self.proto_conf_ratio,
            )
            loss_proto = w_proto * proto_result["total"]
        else:
            zero = torch.tensor(0.0, device=device)
            proto_result = {"ref_loss": zero, "tgt_loss": zero, "total": zero,
                            "n_protos": 0, "frac_tgt_kept": 0.0}
            loss_proto = zero

        # 6. Total Loss
        total = (
            alpha_recon * loss_recon
            + beta_kl * loss_kl
            + beta_cls * loss_cls
            + loss_align
            + loss_proto
        )

        return {
            "loss": total,
            "Reconstruction_Loss": loss_recon.detach(),
            "KL_Loss": loss_kl.detach(),
            "Classification_Loss": loss_cls.detach(),
            "Alignment_Total": loss_align.detach() if torch.is_tensor(loss_align) else torch.tensor(loss_align),
            "Align_Intra_Ref": align_result["intra_ref"].detach(),
            "Align_Intra_Tgt": align_result["intra_tgt"].detach(),
            "Align_Bridge": align_result["bridge"].detach(),
            "Proto_Ref": proto_result["ref_loss"].detach() if torch.is_tensor(proto_result["ref_loss"]) else torch.tensor(0.0),
            "Proto_Tgt": proto_result["tgt_loss"].detach() if torch.is_tensor(proto_result["tgt_loss"]) else torch.tensor(0.0),
            "Proto_Total": (proto_result["total"].detach() if torch.is_tensor(proto_result["total"]) else torch.tensor(0.0)),
            "Proto_FracTgtKept": torch.tensor(proto_result["frac_tgt_kept"]),
        }
