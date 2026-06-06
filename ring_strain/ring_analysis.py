"""
Ring detection and classification for molecular ring systems.

Identifies all rings using SSSR (Smallest Set of Smallest Rings),
classifies ring types (monocyclic, fused, spiro, bridged, cage),
and provides per-ring metadata for downstream strain analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import rdMolDescriptors

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class FusionType(Enum):
    ISOLATED = "isolated"       # Monocyclic, standalone
    FUSED = "fused"             # Shares 2+ atoms
    SPIRO = "spiro"             # Shares exactly 1 atom
    BRIDGED = "bridged"         # Bridged bicyclic
    CAGE = "cage"               # Highly constrained (cubane, adamantane)


class RingSizeCategory(Enum):
    SMALL = "small"             # 3-4 membered
    NORMAL = "normal"           # 5-6 membered
    MEDIUM = "medium"           # 7-9 membered
    LARGE = "large"             # 10+ membered


@dataclass
class RingInfo:
    """Detailed information about a single ring in a molecule."""
    ring_index: int
    atom_indices: Tuple[int, ...]
    bond_indices: Tuple[int, ...]
    size: int
    is_aromatic: bool
    is_heterocyclic: bool
    heteroatom_symbols: List[str]
    atom_symbols: List[str]
    fusion_type: str
    shared_atoms_with_other_rings: int
    size_category: str

    @property
    def is_carbocyclic(self) -> bool:
        return not self.is_heterocyclic


@dataclass
class RingSystemInfo:
    """Overall ring system analysis for a molecule."""
    molecule: Mol
    num_rings: int
    ring_systems: List[List[int]]    # Groups of ring indices sharing atoms
    rings: List[RingInfo]
    has_polycyclic: bool
    ring_system_types: List[str]

    @property
    def ring_sizes(self) -> List[int]:
        return [r.size for r in self.rings]

    @property
    def smallest_ring_size(self) -> int:
        return min(r.size for r in self.rings) if self.rings else 0

    @property
    def largest_ring_size(self) -> int:
        return max(r.size for r in self.rings) if self.rings else 0


# ---------------------------------------------------------------------------
# Ring analyzer
# ---------------------------------------------------------------------------

class RingAnalyzer:
    """Detect and classify ring systems in a molecule.

    Uses RDKit's ring-finding utilities combined with custom classification
    logic for polycyclic, spiro, bridged, and cage systems.

    Parameters
    ----------
    mol : Mol
        RDKit molecule (hydrogens optional — ring detection works either way).
    """

    def __init__(self, mol: Mol):
        self.mol = mol
        Chem.SanitizeMol(self.mol)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify_all_rings(self) -> RingSystemInfo:
        """Get all rings via SSSR and classify them."""
        sssr = list(Chem.GetSymmSSSR(self.mol))
        if not sssr:
            return RingSystemInfo(
                molecule=self.mol,
                num_rings=0,
                ring_systems=[],
                rings=[],
                has_polycyclic=False,
                ring_system_types=[],
            )

        atom_rings = [tuple(sorted(r)) for r in sssr]
        all_rings = []
        for idx, ring_atoms in enumerate(atom_rings):
            ring_info = self._classify_ring(idx, ring_atoms, atom_rings)
            all_rings.append(ring_info)

        ring_groups = self._group_ring_systems(atom_rings)
        system_types = self._classify_ring_systems(all_rings, ring_groups)
        has_poly = any(len(g) >= 2 for g in ring_groups) if ring_groups else False

        return RingSystemInfo(
            molecule=self.mol,
            num_rings=len(all_rings),
            ring_systems=ring_groups,
            rings=all_rings,
            has_polycyclic=has_poly,
            ring_system_types=system_types,
        )

    def has_rings(self) -> bool:
        return rdMolDescriptors.CalcNumRings(self.mol) > 0

    def num_rings(self) -> int:
        return rdMolDescriptors.CalcNumRings(self.mol)

    def get_shared_atoms_between_rings(
        self, ring_i_atoms: Tuple[int, ...], ring_j_atoms: Tuple[int, ...]
    ) -> List[int]:
        """Return atoms shared between two rings."""
        return list(set(ring_i_atoms) & set(ring_j_atoms))

    # ------------------------------------------------------------------
    # Internal ring classification
    # ------------------------------------------------------------------

    def _classify_ring(
        self,
        ring_index: int,
        ring_atoms: Tuple[int, ...],
        all_rings: List[Tuple[int, ...]],
    ) -> RingInfo:
        """Classify a single ring by size, type, and fusion mode."""
        size = len(ring_atoms)

        # Atoms and bonds
        symbols = [self.mol.GetAtomWithIdx(i).GetSymbol() for i in ring_atoms]
        het_syms = [s for s in symbols if s != "C"]
        is_het = len(het_syms) > 0

        bonds = self._get_ring_bonds(ring_atoms)

        # Aromaticity
        is_aro = any(self.mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
                     for a in ring_atoms
                     for b in ring_atoms if self.mol.GetBondBetweenAtoms(a, b) is not None)

        # Shared atoms with other rings
        shared_count = 0
        for j, other in enumerate(all_rings):
            if j == ring_index:
                continue
            shared_count += len(set(ring_atoms) & set(other))

        fusion = self._detect_fusion_type(ring_atoms, all_rings, ring_index)

        # Size category
        if size <= 4:
            cat = RingSizeCategory.SMALL.value
        elif size <= 6:
            cat = RingSizeCategory.NORMAL.value
        elif size <= 9:
            cat = RingSizeCategory.MEDIUM.value
        else:
            cat = RingSizeCategory.LARGE.value

        return RingInfo(
            ring_index=ring_index,
            atom_indices=ring_atoms,
            bond_indices=tuple(bonds),
            size=size,
            is_aromatic=is_aro,
            is_heterocyclic=is_het,
            heteroatom_symbols=het_syms,
            atom_symbols=symbols,
            fusion_type=fusion,
            shared_atoms_with_other_rings=shared_count,
            size_category=cat,
        )

    def _detect_fusion_type(
        self,
        ring_atoms: Tuple[int, ...],
        all_rings: List[Tuple[int, ...]],
        ring_index: int,
    ) -> str:
        """Determine if the ring is isolated, fused, spiro, bridged, or cage."""
        shared_info = []
        for j, other in enumerate(all_rings):
            if j == ring_index:
                continue
            shared = set(ring_atoms) & set(other)
            if shared:
                shared_info.append((j, len(shared), shared))

        if not shared_info:
            return FusionType.ISOLATED.value

        # Count rings in the same system
        same_system_rings = self._rings_in_same_system(ring_index, all_rings)

        if len(same_system_rings) >= 4:
            return FusionType.CAGE.value

        max_shared = max(s[1] for s in shared_info)
        if max_shared == 1:
            return FusionType.SPIRO.value
        if max_shared >= 3:
            return FusionType.BRIDGED.value
        return FusionType.FUSED.value

    def _rings_in_same_system(
        self, ring_index: int, all_rings: List[Tuple[int, ...]]
    ) -> Set[int]:
        """Find all rings that are connected to this one (transitive closure)."""
        visited = set()
        stack = [ring_index]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for j, other in enumerate(all_rings):
                if j in visited:
                    continue
                if set(all_rings[cur]) & set(other):
                    stack.append(j)
        return visited

    # ------------------------------------------------------------------
    # Ring grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_ring_systems(
        atom_rings: List[Tuple[int, ...]],
    ) -> List[List[int]]:
        """Group ring indices into connected ring systems.

        Two rings belong to the same system if they share at least one atom.
        """
        n = len(atom_rings)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if set(atom_rings[i]) & set(atom_rings[j]):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        return list(groups.values())

    def _classify_ring_systems(
        self,
        rings: List[RingInfo],
        ring_groups: List[List[int]],
    ) -> List[str]:
        """Assign a system-level classification to each group of rings."""
        types = []
        for group in ring_groups:
            if len(group) == 1:
                types.append(FusionType.ISOLATED.value)
            elif len(group) >= 4:
                types.append(FusionType.CAGE.value)
            else:
                # Check for bridged in the group
                any_bridged = any(
                    rings[i].fusion_type == FusionType.BRIDGED.value
                    for i in group
                )
                any_spiro = any(
                    rings[i].fusion_type == FusionType.SPIRO.value
                    for i in group
                )
                if any_bridged:
                    types.append(FusionType.BRIDGED.value)
                elif any_spiro:
                    types.append(FusionType.SPIRO.value)
                else:
                    types.append(FusionType.FUSED.value)
        return types

    # ------------------------------------------------------------------
    # Bond utilities
    # ------------------------------------------------------------------

    def _get_ring_bonds(self, ring_atoms: Tuple[int, ...]) -> List[int]:
        """Get bond indices for bonds within this ring."""
        bonds = []
        n = len(ring_atoms)
        for i in range(n):
            a1 = ring_atoms[i]
            a2 = ring_atoms[(i + 1) % n]
            bond = self.mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:
                bonds.append(bond.GetIdx())
        return bonds
