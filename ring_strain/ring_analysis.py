"""
Ring detection and validation for monocyclic saturated carbocycles.

Only monocyclic all-carbon saturated rings (cycloalkanes) are supported.
Polycyclic, bridged, spiro, heterocyclic, aromatic, and unsaturated ring
systems are rejected.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import Mol


@dataclass
class RingInfo:
    """Information about a single monocyclic saturated carbocycle."""
    atom_indices: Tuple[int, ...]
    bond_indices: Tuple[int, ...]
    size: int


class RingAnalyzer:
    """Validate that a molecule is a monocyclic saturated carbocycle.

    Only molecules with exactly one ring where all ring atoms are carbon
    and all ring bonds are single bonds are supported. Everything else
    (polycyclic, heterocyclic, aromatic, unsaturated) is rejected.
    """

    def __init__(self, mol: Mol):
        self.mol = mol

    def validate(self) -> Tuple[bool, str, Optional[RingInfo]]:
        """Check if molecule is a supported monocyclic saturated carbocycle.

        Returns:
            (is_valid, message, ring_info_or_None)
        """
        if not self.has_rings():
            return False, "Not Supported: molecule has no rings", None

        sssr = list(Chem.GetSymmSSSR(self.mol))
        if len(sssr) != 1:
            return False, (
                f"Not Supported: polycyclic system "
                f"({len(sssr)} rings detected; only monocyclic supported)"
            ), None

        # Geometry consumers require cyclic topological order. Atom indices
        # reflect SMILES construction order, so sorting them can place
        # non-bonded atoms next to one another for branched SMILES.
        ring_atoms = _order_ring_atoms(tuple(sssr[0]), self.mol)
        if ring_atoms is None:
            return False, "Not Supported: invalid monocyclic ring topology", None
        size = len(ring_atoms)

        # All ring atoms must be carbon
        for idx in ring_atoms:
            atom = self.mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() != 6:
                symbol = atom.GetSymbol()
                return False, (
                    f"Not Supported: heteroatom ({symbol}) in ring"
                ), None

        # All ring bonds must be single bonds (no double/triple/aromatic)
        ring_bonds = _get_ring_bonds(ring_atoms, self.mol)
        if len(ring_bonds) != size:
            return False, "Not Supported: invalid monocyclic ring topology", None
        for bidx in ring_bonds:
            bond = self.mol.GetBondWithIdx(bidx)
            if bond.GetBondType() != rdchem.BondType.SINGLE:
                return False, (
                    "Not Supported: unsaturated ring "
                    "(double/triple/aromatic bonds detected)"
                ), None

        ring_info = RingInfo(
            atom_indices=ring_atoms,
            bond_indices=tuple(ring_bonds),
            size=size,
        )
        return True, f"monocyclic saturated carbocycle ({size}-membered)", ring_info

    def has_rings(self) -> bool:
        return Chem.GetSymmSSSR(self.mol).__len__() > 0


def _order_ring_atoms(
    ring_atoms: Tuple[int, ...], mol: Mol
) -> Optional[Tuple[int, ...]]:
    """Return a deterministic bond-by-bond traversal of a simple ring."""
    ring_set = set(ring_atoms)
    if len(ring_set) < 3:
        return None

    ring_neighbors = {
        idx: sorted(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
            if neighbor.GetIdx() in ring_set
        )
        for idx in ring_set
    }
    if any(len(neighbors) != 2 for neighbors in ring_neighbors.values()):
        return None

    start = min(ring_set)
    ordered = [start]
    previous = None
    current = start
    while len(ordered) < len(ring_set):
        candidates = [
            idx for idx in ring_neighbors[current]
            if idx != previous and idx not in ordered
        ]
        if not candidates:
            return None
        next_idx = candidates[0]
        ordered.append(next_idx)
        previous, current = current, next_idx

    if start not in ring_neighbors[current]:
        return None
    return tuple(ordered)


def _get_ring_bonds(ring_atoms: Tuple[int, ...], mol: Mol) -> List[int]:
    """Get bond indices for bonds within a ring."""
    bonds = []
    n = len(ring_atoms)
    for i in range(n):
        a1 = ring_atoms[i]
        a2 = ring_atoms[(i + 1) % n]
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is not None:
            bonds.append(bond.GetIdx())
    return bonds
