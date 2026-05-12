import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple

def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Log probability of Negative Binomial distribution.

    Args:
        x: observed counts
        mu: mean parameter
        theta: dispersion parameter

    Returns:
        Log probability (same shape as x)
    """
    log_theta_mu = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1)
    )
    return res

"""
Two-Level Alignment Strategy:
===============================
Level 1 (Intra-Species): homo ↔ nonhomo alignment within each species
Level 2 (Bridged Full):  CORAL on the concatenated [homo; nonhomo]

Cross-species cell-type-aware alignment is provided separately by the
prototype contrastive loss below.
"""

# =============================================================================
# Level 1: Intra-Species Alignment (homo ↔ nonhomo)
# =============================================================================

def intra_species_alignment_loss(
    homo_mean: torch.Tensor,
    nonhomo_mean: torch.Tensor,
    detach_homo: bool = True,
) -> torch.Tensor:
    """
    Within-species MSE pulling nonhomo toward (detached) homo. Strong
    constraint that collapses nonhomo into the homo subspace.
    """
    if detach_homo:
        homo_mean = homo_mean.detach()
    return F.mse_loss(homo_mean, nonhomo_mean)


# =============================================================================
# Level 2: Bridged Full Alignment (via aligned homo)
# =============================================================================

def bridged_full_alignment(
    ref_homo: torch.Tensor,
    ref_nonhomo: torch.Tensor,
    tgt_homo: torch.Tensor,
    tgt_nonhomo: torch.Tensor
) -> torch.Tensor:
    ref_full = torch.cat([ref_homo, ref_nonhomo], dim=1)
    tgt_full = torch.cat([tgt_homo, tgt_nonhomo], dim=1)
    return CORAL(ref_full, tgt_full, ref_homo.device)

def CORAL(source: torch.Tensor, target: torch.Tensor, device) -> torch.Tensor:
    eps = 1e-5
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # Source covariance
    tmp_s = torch.ones((1, ns), device=device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1 + eps)

    # Target covariance
    tmp_t = torch.ones((1, nt), device=device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1 + eps)

    loss = (cs - ct).pow(2).sum() / (4 * d * d)
    return loss


# =============================================================================
# Cell-type-aware cross-species alignment via ref prototypes
# =============================================================================

def proto_contrastive_loss(
    ref_z: torch.Tensor,
    ref_y: torch.Tensor,
    tgt_z: torch.Tensor,
    num_classes: int,
    temperature: float = 0.1,
    conf_ratio: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Cell-type-aware cross-species alignment using ref prototypes.

    Steps:
      1. Compute per-class prototypes from ref_z (detached — anchor only).
      2. Ref InfoNCE: each ref_z[i] should be closest to proto[ref_y[i]] and
         far from other prototypes. This tightens ref clusters.
      3. Target InfoNCE: pseudo-label each tgt_z[j] by nearest prototype, then
         InfoNCE with that pseudo-label. Optional confidence filter: only keep
         tgt cells whose nearest-vs-second-nearest cosine-distance ratio is
         < conf_ratio (i.e. the nearest is clearly closest). conf_ratio=0
         disables filtering and keeps all tgt cells.

    Returns dict with `ref_loss`, `tgt_loss`, `total`, plus diagnostics.
    """
    device = ref_z.device

    if ref_z.numel() == 0 or tgt_z.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return {"ref_loss": zero, "tgt_loss": zero, "total": zero,
                "n_protos": 0, "frac_tgt_kept": 0.0}

    # 1. Prototypes (per-class mean over ref in this batch).
    # Some classes may be absent from a given batch — track which exist.
    D = ref_z.size(1)
    protos = torch.zeros(num_classes, D, device=device)
    counts = torch.zeros(num_classes, device=device)
    protos.index_add_(0, ref_y, ref_z)
    counts.index_add_(0, ref_y, torch.ones_like(ref_y, dtype=torch.float))
    valid_class = counts > 0
    protos[valid_class] = protos[valid_class] / counts[valid_class].unsqueeze(1)
    protos = protos.detach()  # anchor — no gradient through proto definition

    if valid_class.sum() < 2:
        zero = torch.tensor(0.0, device=device)
        return {"ref_loss": zero, "tgt_loss": zero, "total": zero,
                "n_protos": int(valid_class.sum()), "frac_tgt_kept": 0.0}

    # Restrict to valid classes for the contrast computation.
    valid_idx = torch.nonzero(valid_class, as_tuple=False).squeeze(1)  # (V,)
    P = protos[valid_idx]                                              # (V, D)
    # Map original class id -> position within P (for ref labels)
    pos_of = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    pos_of[valid_idx] = torch.arange(valid_idx.numel(), device=device)
    ref_pos = pos_of[ref_y]  # (N_ref,) ; -1 only if class absent (shouldn't happen)

    # L2-normalise everything for cosine similarity.
    rz = F.normalize(ref_z, dim=1)
    tz = F.normalize(tgt_z, dim=1)
    Pn = F.normalize(P, dim=1)

    # 2. Ref InfoNCE
    sim_ref = rz @ Pn.t() / temperature                # (N_ref, V)
    # Drop ref cells whose label was filtered out (extremely rare).
    mask_ref = ref_pos >= 0
    if mask_ref.sum() > 0:
        ref_loss = F.cross_entropy(sim_ref[mask_ref], ref_pos[mask_ref])
    else:
        ref_loss = torch.tensor(0.0, device=device)

    # 3. Target InfoNCE with pseudo-labels
    sim_tgt = tz @ Pn.t() / temperature                # (N_tgt, V)
    # Pseudo-labels = argmax similarity = argmin cosine distance.
    pseudo = sim_tgt.argmax(dim=1)                     # (N_tgt,)

    if conf_ratio > 0.0 and sim_tgt.size(1) >= 2:
        # Confidence: ratio of cosine distance to nearest vs second-nearest.
        # Cosine distance = 1 - sim*temperature  (rescale unnecessary for ratio
        # since both share temperature). Use sim directly.
        top2 = sim_tgt.topk(2, dim=1).values            # (N_tgt, 2)
        # Distance: smaller better. dist = 1 - sim_unscaled = 1 - sim*temp
        # But ratio of (1 - top1) / (1 - top2) is what we want.
        sim1 = top2[:, 0] * temperature                 # back to cosine [-1,1]
        sim2 = top2[:, 1] * temperature
        d1 = 1.0 - sim1
        d2 = 1.0 - sim2
        # Confident if d1/d2 small (i.e. nearest is clearly closer)
        keep = d1 < conf_ratio * d2
    else:
        keep = torch.ones(tz.size(0), dtype=torch.bool, device=device)

    if keep.sum() > 0:
        tgt_loss = F.cross_entropy(sim_tgt[keep], pseudo[keep])
    else:
        tgt_loss = torch.tensor(0.0, device=device)

    total = ref_loss + tgt_loss
    return {
        "ref_loss": ref_loss,
        "tgt_loss": tgt_loss,
        "total": total,
        "n_protos": int(valid_class.sum()),
        "frac_tgt_kept": float(keep.float().mean()),
    }


# =============================================================================
# Combined Hierarchical Loss
# =============================================================================

class HierarchicalAlignmentLoss(nn.Module):
    def __init__(
        self,
        beta_intra: float = 1.0,
        beta_bridge: float = 0.5,
    ):
        super().__init__()
        self.beta_intra = beta_intra
        self.beta_bridge = beta_bridge

    def forward(
        self,
        ref_homo: torch.Tensor,
        ref_nonhomo: torch.Tensor,
        tgt_homo: torch.Tensor,
        tgt_nonhomo: torch.Tensor,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Level 1: Intra-species alignment
        loss_intra_ref = intra_species_alignment_loss(ref_homo, ref_nonhomo)
        loss_intra_tgt = intra_species_alignment_loss(tgt_homo, tgt_nonhomo)
        loss_intra = loss_intra_ref + loss_intra_tgt

        # Level 2: Bridged full alignment
        loss_bridge = bridged_full_alignment(
            ref_homo, ref_nonhomo, tgt_homo, tgt_nonhomo
        )

        total = self.beta_intra * loss_intra + self.beta_bridge * loss_bridge

        if return_components:
            return {
                "total": total,
                "intra_ref": loss_intra_ref,
                "intra_tgt": loss_intra_tgt,
                "bridge": loss_bridge,
            }

        return {"total": total}
