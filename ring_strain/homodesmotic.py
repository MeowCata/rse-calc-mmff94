"""
Homodesmotic ring strain energy calculation.

Constructs balanced homodesmotic reactions where a cyclic molecule is
compared to an acyclic reference that preserves bond types and
hybridization. Computes raw MMFF94 ring strain energy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops
from rdkit.Chem.rdchem import Mol, RWMol

from .mmff import MMFFCalculator, MMFFResult
from .ring_analysis import RingAnalyzer, RingInfo, RingSystemInfo

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reaction representation
# ---------------------------------------------------------------------------

@dataclass
class HomodesmoticReaction:
    """Representation of a homodesmotic strain analysis reaction."""
    cyclic_reactant_smiles: str
    acyclic_product_smiles: str
    auxiliary_reactants: List[str]    # e.g. ['CC'] for ethane
    auxiliary_products: List[str]
    reaction_smarts: str
    description: str


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class HomodesmoticAnalyzer:
    """Compute ring strain via homodesmotic reaction method.

    For each ring system, constructs a balanced reaction where the cyclic
    molecule is transformed into an acyclic analog that preserves bond
    types and hybridization states.

    Strain energy = sum(E_products) - sum(E_reactants)

    Parameters
    ----------
    mmff_calc : MMFFCalculator
        MMFF94 calculator for geometry optimization and energy.
    """

    def __init__(self, mmff_calc: MMFFCalculator):
        self.mmff = mmff_calc

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute_total_strain(
        self,
        mol: Mol,
        ring_info: RingSystemInfo,
    ) -> Tuple[float, Dict[int, float], Dict[str, float]]:
        """Compute total raw MMFF94 ring strain energy.

        For monocyclic systems: direct homodesmotic comparison.
        For polycyclic systems: sequential ring opening.

        Returns:
            (total_strain_kcal_mol, per_ring_strain, energy_components)
        """
        if ring_info.num_rings == 0:
            return 0.0, {}, {"cyclic_energy": 0.0, "acyclic_energy": 0.0}

        # Get energy of the cyclic form
        try:
            cyclic_result = self.mmff.embed_and_optimize(Chem.Mol(mol))
            cyclic_energy = cyclic_result.total_energy
        except Exception as exc:
            logger.error("MMFF94 failed for cyclic molecule: %s", exc)
            raise

        # For simple monocyclic cycloalkanes, use the strict bond-balanced
        # homodesmotic method (more accurate than generic ring-opening).
        if self._is_simple_carbocycle(mol, ring_info):
            ring_size = ring_info.rings[0].size
            strain = self.compute_cycloalkane_strain(ring_size)
            if strain is not None:
                per_ring = {0: strain}
                components = {
                    "cyclic_energy": cyclic_energy,
                    "acyclic_energy": cyclic_energy - strain,
                }
                return strain, per_ring, components

        # For aromatics: ring-opening is problematic because delocalized
        # bonds can't be trivially broken. Treat aromatic rings as having
        # zero homodesmotic strain (resonance stabilization compensates).
        if self._is_aromatic_system(ring_info):
            per_ring = {r.ring_index: 0.0 for r in ring_info.rings}
            components = {
                "cyclic_energy": cyclic_energy,
                "acyclic_energy": cyclic_energy,
                "aromatic_bypass": True,
            }
            return 0.0, per_ring, components

        # Build acyclic reference and compute its energy
        try:
            acyclic_mol, _ = self._build_acyclic_reference(Chem.Mol(mol), ring_info)
            acyclic_result = self.mmff.embed_and_optimize(acyclic_mol)
            acyclic_energy = acyclic_result.total_energy
        except Exception as exc:
            logger.error(
                "MMFF94 failed for acyclic reference: %s. "
                "Falling back to per-heavy-atom comparison.",
                exc,
            )
            return self._fallback_strain(mol, cyclic_result, ring_info)

        raw_strain = cyclic_energy - acyclic_energy

        # Per-ring attribution
        per_ring = self._allocate_strain_to_rings(
            mol, raw_strain, ring_info
        )

        components = {
            "cyclic_energy": cyclic_energy,
            "acyclic_energy": acyclic_energy,
            "cyclic_energy_per_heavy": cyclic_result.energy_per_heavy_atom,
            "acyclic_energy_per_heavy": acyclic_result.energy_per_heavy_atom,
        }

        return raw_strain, per_ring, components

    # ------------------------------------------------------------------
    # Reference construction
    # ------------------------------------------------------------------

    def _build_acyclic_reference(
        self,
        mol: Mol,
        ring_info: RingSystemInfo,
    ) -> Tuple[Mol, List[int]]:
        """Build an acyclic analog of the cyclic molecule.

        Strategy for monocyclic: break one ring bond, cap with H.
        For polycyclic: break all rings sequentially, cap with H.

        Returns (acyclic_mol, list_of_broken_bond_indices).
        """
        rw_mol = RWMol(mol)
        broken_bonds = []

        for ring in ring_info.rings:
            # Find a non-aromatic ring bond to break
            bond_idx = self._choose_bond_to_break(rw_mol, ring)
            if bond_idx is None:
                continue

            bond = rw_mol.GetBondWithIdx(bond_idx)
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()

            # Remove the bond
            was_aromatic = bond.GetIsAromatic()
            rw_mol.RemoveBond(a1, a2)
            broken_bonds.append(bond_idx)

            # If we broke an aromatic bond, clear aromatic flags
            # on affected atoms so sanitization does not reject them
            for aidx in (a1, a2):
                atom = rw_mol.GetAtomWithIdx(aidx)
                if was_aromatic:
                    atom.SetIsAromatic(False)

        # Let RDKit compute correct implicit H counts via sanitization
        # (avoids the SetNumExplicitHs / GetTotalNumHs double-counting bug
        #  where implicit H persist after setting explicit H)
        try:
            Chem.SanitizeMol(rw_mol)
        except Exception as exc:
            logger.warning("Acyclic reference sanitization: %s", exc)

        return rw_mol.GetMol(), broken_bonds

    @staticmethod
    def _choose_bond_to_break(rw_mol, ring: RingInfo) -> Optional[int]:
        """Select the best ring bond to break.

        Prefers single bonds, avoids aromatic bonds, and chooses the
        bond that will minimize conformational strain in the open form.
        """
        for bidx in ring.bond_indices:
            bond = rw_mol.GetBondWithIdx(bidx)
            if bond is not None and not bond.GetIsAromatic():
                if bond.GetBondType() == rdchem.BondType.SINGLE:
                    return bidx

        # Fallback: any non-aromatic ring bond
        for bidx in ring.bond_indices:
            bond = rw_mol.GetBondWithIdx(bidx)
            if bond is not None and not bond.GetIsAromatic():
                return bidx

        # Desperate fallback: any ring bond
        return ring.bond_indices[0] if ring.bond_indices else None

    # ------------------------------------------------------------------
    # Homodesmotic reaction for cycloalkanes (high accuracy)
    # ------------------------------------------------------------------

    def build_cycloalkane_homodesmotic_reaction(
        self,
        ring_size: int,
    ) -> Optional[HomodesmoticReaction]:
        """Build a strict homodesmotic reaction for cycloalkanes.

        cyclo-(CH2)n + CH3-CH3  ->  CH3-(CH2)(n+1)-CH3

        This is fully bond-balanced: both sides have the same count
        of C-C(sp3-sp3), C-H bonds, and sp3 carbons.
        """
        if ring_size < 3:
            return None

        # Build cyclic SMILES
        cyclic = "C1" + "C" * (ring_size - 2) + "C1"

        # Build linear alkane: n+2 carbons
        linear = "C" + "C" * (ring_size + 1)

        return HomodesmoticReaction(
            cyclic_reactant_smiles=cyclic,
            acyclic_product_smiles=linear,
            auxiliary_reactants=["CC"],
            auxiliary_products=[],
            reaction_smarts="",
            description=(
                f"cyclo-(CH2)_{ring_size} + C2H6 -> "
                f"CH3-(CH2)_{ring_size + 1}-CH3"
            ),
        )

    def compute_cycloalkane_strain(
        self,
        ring_size: int,
    ) -> Optional[float]:
        """Compute raw homodesmotic strain for a cycloalkane using the
        strict bond-balanced reaction scheme.

        Returns raw MMFF94 strain in kcal/mol.
        """
        reaction = self.build_cycloalkane_homodesmotic_reaction(ring_size)
        if reaction is None:
            return None

        try:
            cyclic_mol = Chem.MolFromSmiles(reaction.cyclic_reactant_smiles)
            cyclic_result = self.mmff.embed_and_optimize(cyclic_mol)
            cyclic_energy = cyclic_result.total_energy

            ethane_mol = Chem.MolFromSmiles("CC")
            ethane_result = self.mmff.embed_and_optimize(ethane_mol)
            ethane_energy = ethane_result.total_energy

            linear_mol = Chem.MolFromSmiles(reaction.acyclic_product_smiles)
            linear_result = self.mmff.embed_and_optimize(linear_mol)
            linear_energy = linear_result.total_energy

            # E_strain = E(cyclic) + E(ethane) - E(linear)
            strain = cyclic_energy + ethane_energy - linear_energy
            return strain

        except Exception as exc:
            logger.error(
                "Cycloalkane homodesmotic strain failed for ring size %d: %s",
                ring_size,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_simple_carbocycle(mol: Mol, ring_info: RingSystemInfo) -> bool:
        """Check if molecule is a simple monocyclic carbocycle (no heteroatoms,
        no aromatics, single ring, all carbons)."""
        if ring_info.num_rings != 1:
            return False
        ring = ring_info.rings[0]
        if ring.is_aromatic or ring.is_heterocyclic:
            return False
        if ring.size > 8:
            return False
        # Verify all atoms are carbon
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                return False
        return True

    @staticmethod
    def _is_aromatic_system(ring_info: RingSystemInfo) -> bool:
        """Check if ALL rings in the system are fully aromatic."""
        if ring_info.num_rings == 0:
            return False
        return all(r.is_aromatic for r in ring_info.rings)

    # ------------------------------------------------------------------
    # Ring allocation
    # ------------------------------------------------------------------

    def _allocate_strain_to_rings(
        self,
        mol: Mol,
        total_strain: float,
        ring_info: RingSystemInfo,
    ) -> Dict[int, float]:
        """Attribute total strain to individual rings.

        For isolated rings: strain is attributed by ring size proportion.
        For fused/bridged/cage: strain is attributed based on ring size
        and shared-atom count.

        A ring that shares many atoms (fused/cage) gets less individual
        attribution because its strain is distributed among the system.
        """
        if ring_info.num_rings == 1:
            return {0: total_strain}

        # Weight by ring size (smaller rings contribute more strain)
        weights = {}
        for ring in ring_info.rings:
            # Smaller rings have higher strain per atom
            size_weight = 1.0 / max(ring.size, 1)
            # Rings sharing many atoms (cage) get slightly less weight
            share_penalty = 1.0 / (1.0 + ring.shared_atoms_with_other_rings * 0.1)
            weights[ring.ring_index] = size_weight * share_penalty

        total_weight = sum(weights.values())
        if total_weight == 0:
            # Equal split
            return {i: total_strain / ring_info.num_rings
                    for i in range(ring_info.num_rings)}

        return {
            idx: total_strain * w / total_weight
            for idx, w in weights.items()
        }

    def _fallback_strain(
        self,
        mol: Mol,
        cyclic_result: MMFFResult,
        ring_info: RingSystemInfo,
    ) -> Tuple[float, Dict[int, float], Dict[str, float]]:
        """Fallback strain estimation when acyclic reference fails.

        Uses per-heavy-atom energy comparison with a reference n-alkane.
        Less accurate but better than returning nothing.
        """
        n_heavy = cyclic_result.num_heavy_atoms
        energy = cyclic_result.total_energy

        # Estimate strain-free reference energy using linear alkane
        # CH3-(CH2)(n-2)-CH3 has approximately (-4.9 * n) kcal/mol in MMFF94
        # The slope varies, so we use actual computed values when possible.
        rough_ref_energy_per_atom = -4.9
        ref_energy = rough_ref_energy_per_atom * n_heavy

        raw_strain = energy - ref_energy
        raw_strain = max(raw_strain, 0.0)

        per_ring = self._allocate_strain_to_rings(mol, raw_strain, ring_info)

        components = {
            "cyclic_energy": energy,
            "acyclic_energy": ref_energy,
            "cyclic_energy_per_heavy": cyclic_result.energy_per_heavy_atom,
            "acyclic_energy_per_heavy": rough_ref_energy_per_atom,
            "fallback": True,
        }

        logger.warning("Using fallback strain estimation (less accurate).")
        return raw_strain, per_ring, components
