"""ImportanceTensor — verbatim from biological_interpretability/phylogenetic/tensor_builder.py.

Trimmed to the dataclass. The Figure 5 panels load a prebuilt tensor from
results/Figure5/source_tensor/ (tensor.npy + species/cell_types/genes CSVs,
43 KB total, verified bitwise identical to a fresh build), so
build_importance_tensor and its readers are not needed -- and neither are the
per-pair Q1 CSVs or the 3.3 MB of homolog TSVs it would parse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ImportanceTensor:
    """
    Output of ``build_importance_tensor``.

    ``tensor`` is a numpy array shape ``[n_species, n_cell_types, n_genes]``
    indexed by (species, cell_type, ref-gene-name).  Genes the species
    doesn't have an ortholog for are filled with ``np.nan``.
    """
    tensor: np.ndarray
    species: List[str]
    cell_types: List[str]
    genes: List[str]                   # canonical (ref-species) gene names
    species_to_pairs: Dict[str, List[str]] = field(default_factory=dict)
    method: str = "IG"
    notes: str = ""

