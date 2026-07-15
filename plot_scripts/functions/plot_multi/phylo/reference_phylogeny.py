"""
Reference divergence times (in millions of years) used as the ground-truth
phylogeny baseline.  Values are TimeTree.org medians (rounded) for
species pairs we touch in this codebase.  Edit / extend as new species
appear.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Symmetric divergence times in MYA.  Keys are sorted tuples of canonical
# species names.
REFERENCE_DIVERGENCE_MYA: Dict[Tuple[str, str], float] = {
    # primates anchored at human
    tuple(sorted(["Human", "Chimpanzee"])): 6.4,
    tuple(sorted(["Human", "Bonobo"])): 6.4,
    tuple(sorted(["Human", "Gorilla"])): 8.6,
    tuple(sorted(["Human", "Gibbon"])): 20.2,
    tuple(sorted(["Human", "Orangutan"])): 15.2,
    tuple(sorted(["Human", "Macaque"])): 28.8,
    tuple(sorted(["Human", "Marmoset"])): 42.6,
    tuple(sorted(["Human", "Mouse"])): 87.0,
    tuple(sorted(["Human", "Rat"])): 87.0,
    tuple(sorted(["Human", "Pig"])): 95.0,
    tuple(sorted(["Human", "Cattle"])): 95.0,
    tuple(sorted(["Human", "Chicken"])): 318.0,
    tuple(sorted(["Human", "Lizard"])): 318.0,
    tuple(sorted(["Human", "Green_anole"])): 318.0,
    tuple(sorted(["Human", "Turtle"])): 318.0,
    tuple(sorted(["Human", "Opossum"])): 159.0,
    tuple(sorted(["Human", "Platypus"])): 180.0,
    tuple(sorted(["Human", "Zebrafish"])): 432.0,
    # mouse anchors
    tuple(sorted(["Mouse", "Rat"])): 14.0,   # TimeTree median ~12-14 MYA
    tuple(sorted(["Mouse", "Pig"])): 95.0,
    tuple(sorted(["Mouse", "Chicken"])): 318.0,
    tuple(sorted(["Mouse", "Macaque"])): 87.0,
    tuple(sorted(["Mouse", "Monkey"])): 87.0,   # generic 'Monkey' label sometimes used
    tuple(sorted(["Mouse", "Hamster"])): 21.0,
    tuple(sorted(["Mouse", "Golden_hamster"])): 21.0,
    # monkey-monkey
    tuple(sorted(["Macaque", "Marmoset"])): 42.6,
    tuple(sorted(["Macaque", "Gibbon"])): 28.8,
    tuple(sorted(["Macaque", "Gorilla"])): 28.8,
    # great-ape internal
    tuple(sorted(["Gorilla", "Gibbon"])): 20.2,
    tuple(sorted(["Gorilla", "Marmoset"])): 42.6,
    tuple(sorted(["Gibbon", "Marmoset"])): 42.6,
}


def get_pairwise_mya(a: str, b: str) -> float:
    """Return MYA between species ``a`` and ``b``.

    Looks up by sorted-tuple key.  Returns ``float('nan')`` if not in the
    table — caller should decide whether to skip or impute.
    """
    if a == b:
        return 0.0
    key = tuple(sorted([a, b]))
    return float(REFERENCE_DIVERGENCE_MYA.get(key, float("nan")))
