"""
Homodesmotic ring strain energy calculation.

For unsubstituted cycloalkanes, uses the strict bond-balanced reaction:
    cyclo-(CH2)n + CH3-CH3  ->  CH3-(CH2)(n+1)-CH3

For substituted cycloalkanes, uses ring-opening with H-capping:
    strain = E(cyclic) - E(acyclic_H_capped)
Every single ring bond is opened and deduplicated by product SMILES; the
lowest-energy closed-shell acyclic reference is retained.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import Mol, RWMol

from .mmff import MMFFCalculator, ProgressCallback
from .ring_analysis import RingInfo, _get_ring_bonds

import logging

logger = logging.getLogger(__name__)

# Module-level caches for immutable MMFF94 constants
_ETHANE_ENERGY: Optional[float] = None
_CYCLOALKANE_STRAIN_CACHE: Dict[int, float] = {}


def _subtask_progress(
    callback: Optional[ProgressCallback],
    index: int,
    total: int,
    label: str,
) -> Optional[ProgressCallback]:
    """Map a collaborator's local counter into one item of a larger loop."""
    if callback is None:
        return None

    units_per_item = 100

    def report(task: str, current: int, task_total: int) -> None:
        fraction = current / task_total if task_total > 0 else 0.0
        overall_current = (index - 1) * units_per_item + round(
            fraction * units_per_item
        )
        callback(
            f"{label} {index}/{total}: {task}",
            overall_current,
            total * units_per_item,
        )

    return report


def _range_progress(
    callback: Optional[ProgressCallback],
    start: float,
    end: float,
) -> Optional[ProgressCallback]:
    """Map a collaborator's counter into a fractional range of one task."""
    if callback is None:
        return None

    def report(task: str, current: int, total: int) -> None:
        fraction = current / total if total > 0 else 0.0
        mapped = start + min(max(fraction, 0.0), 1.0) * (end - start)
        callback(task, round(mapped * 1000), 1000)

    return report


@dataclass
class HomodesmoticReaction:
    """Representation of a homodesmotic strain analysis reaction."""
    cyclic_reactant_smiles: str
    acyclic_product_smiles: str
    auxiliary_reactants: List[str]
    auxiliary_products: List[str]
    reaction_smarts: str
    description: str


class HomodesmoticAnalyzer:
    """Compute ring strain via homodesmotic reaction method.

    - Unsubstituted cycloalkanes: strict bond-balanced reaction
        cyclo-(CH2)n + CH3-CH3  ->  CH3-(CH2)(n+1)-CH3
      Precomputed from cycloalkane SMILES, cached by ring size.

    - Substituted cycloalkanes: H-capped ring-opening.  Strain = E(cyclic) -
      E(acyclic_H_capped).  Every ring bond is tried, so the reference remains
      general across substitution patterns instead of relying on a hand-picked
      opening.
    """

    def __init__(self, mmff_calc: MMFFCalculator):
        self.mmff = mmff_calc

    def compute_strain(
        self,
        mol: Mol,
        ring_info: RingInfo,
        cyclic_mol: Optional[Mol] = None,
        cyclic_energy_override: Optional[float] = None,
        acyclic_energy_override: Optional[float] = None,
        use_boltzmann: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute raw MMFF94 ring strain energy.

        Args:
            mol: Original molecule (implicit hydrogens, no conformers needed).
            ring_info: Validated ring information.
            cyclic_mol: Pre-optimized cyclic molecule with explicit hydrogens
                and conformers. If provided, its best-conformer energy is
                used directly without re-optimization.
            cyclic_energy_override: If provided, used directly as the cyclic
                energy. For Boltzmann-weighted thermal energy.
            acyclic_energy_override: If provided, used directly as the acyclic
                energy. When both overrides are given, the method uses them
                directly for the strain calculation.
            use_boltzmann: When True (default) and the molecule is substituted,
                the acyclic reference also gets Boltzmann thermal averaging,
                matching the cyclic side treatment.
            progress_callback: Optional local-task progress callback.

        Returns:
            (strain_kcal_mol, energy_components)
        """
        # Get energy of the cyclic form
        if cyclic_energy_override is not None:
            cyclic_energy = cyclic_energy_override
        elif cyclic_mol is not None:
            cyclic_energy = self.mmff.compute_single_point_energy(cyclic_mol)
        else:
            cyclic_result = self.mmff.embed_and_optimize(
                Chem.Mol(mol), progress_callback=progress_callback
            )
            cyclic_energy = cyclic_result.total_energy

        # For unsubstituted cycloalkanes: strict bond-balanced method
        if _is_unsubstituted_cycloalkane(mol, ring_info):
            strain = self.compute_cycloalkane_strain(
                ring_info.size, progress_callback=progress_callback
            )
            if strain is not None:
                components = {
                    "cyclic_energy": cyclic_energy,
                    "acyclic_energy": cyclic_energy - strain,
                    "method": "strict homodesmotic",
                }
                return strain, components

        # For substituted cycloalkanes: ring-opening with H-capping.
        acyclic_mol_for_decomp: Optional[Mol] = None
        if acyclic_energy_override is not None:
            acyclic_energy = acyclic_energy_override
            components = {
                "cyclic_energy": cyclic_energy,
                "acyclic_energy": acyclic_energy,
                "method": "ring-opening + Boltzmann (substituted)",
            }
        else:
            acyclic_mol_for_decomp, acyclic_energy = (
                self._compute_acyclic_energy_with_mol(
                    mol,
                    ring_info,
                    use_boltzmann=use_boltzmann,
                    progress_callback=progress_callback,
                )
            )
            method = (
                "ring-opening + Boltzmann (substituted)"
                if use_boltzmann
                else "ring-opening (substituted)"
            )
            components = {
                "cyclic_energy": cyclic_energy,
                "acyclic_energy": acyclic_energy,
                "method": method,
            }

        raw_strain = cyclic_energy - acyclic_energy

        # Steric confinement is the cyclic-minus-acyclic MMFF94 vdW term at
        # representative geometries. It remains a diagnostic snapshot and
        # need not track the Boltzmann-averaged total strain exactly.
        if cyclic_mol is not None and acyclic_mol_for_decomp is not None:
            try:
                cyclic_vdw = self.mmff.compute_vdw_energy(cyclic_mol)
                acyclic_vdw = self.mmff.compute_vdw_energy(acyclic_mol_for_decomp)
                if cyclic_vdw == cyclic_vdw and acyclic_vdw == acyclic_vdw:
                    components["steric_confinement_kcal_mol"] = (
                        cyclic_vdw - acyclic_vdw
                    )
            except Exception as exc:
                logger.warning("Steric-confinement calculation failed: %s", exc)

        return raw_strain, components

    def _compute_acyclic_energy_with_mol(
        self,
        mol: Mol,
        ring_info: RingInfo,
        use_boltzmann: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[Mol, float]:
        """Compute the acyclic reference energy, with optional Boltzmann averaging.

        Tries opening the ring at **every** single ring bond, optimises each
        resulting acyclic reference, and retains the one with the lowest
        energy (most stable open form).  This removes any bias from picking
        a single "best" bond and guarantees the strain is measured against
        the most favourable ring-opening pathway.

        Returns both the best acyclic molecule (with its conformers) and
        its energy.
        """
        # Build and evaluate all possible ring openings
        candidates = self._build_all_acyclic_references(Chem.Mol(mol), ring_info)
        best_mol = None
        best_energy = float("inf")

        n_candidates = len(candidates)
        for candidate_index, (acyclic_raw, _broken_bond) in enumerate(
            candidates, start=1
        ):
            candidate_progress = _subtask_progress(
                progress_callback,
                candidate_index,
                n_candidates,
                "Acyclic reference",
            )
            try:
                acyclic_result = self.mmff.embed_and_optimize(
                    acyclic_raw,
                    progress_callback=_range_progress(
                        candidate_progress, 0.00, 0.20
                    ),
                )
                acyclic_with_H = acyclic_result.molecule
                acyclic_energy = acyclic_result.total_energy

                # Symmetric bulk-aware scaling on the acyclic side, capped
                # at bulk=15 (see mmff.py for rationale).
                bulk = min(15, self.mmff.compute_bulk_score(acyclic_with_H))

                # Acyclic torsion-diversity seeds: random-coords ETKDG hits
                # both extended (anti) and gauche rotamer families. Skip for
                # low-bulk substrates — those chains are well-covered by
                # default ETKDG and the seeding overhead (1-2 s) is wasted.
                if bulk >= 4:
                    try:
                        self.mmff.seed_random_coords_conformers(
                            acyclic_with_H,
                            n_seeds=min(20, 10 + bulk),
                            seed_offset=3000,
                            progress_callback=_range_progress(
                                candidate_progress, 0.20, 0.28
                            ),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Acyclic random-coords seeding failed: %s", exc,
                        )

                pt_steps_eff = 150 + 12 * bulk
                mc_steps_eff = 200 + 18 * bulk
                # 4 replicas always — see core.py.
                pt_temps_eff = (300.0, 500.0, 1000.0, 2000.0)

                # Monte Carlo / PT torsion search
                try:
                    mc_energy = acyclic_energy
                    pt_id, pt_energy = self.mmff.parallel_tempering_search(
                        acyclic_with_H,
                        n_steps=pt_steps_eff,
                        temperatures=pt_temps_eff,
                        progress_callback=_range_progress(
                            candidate_progress, 0.28, 0.70
                        ),
                    )
                    if pt_id >= 0 and pt_energy < mc_energy:
                        mc_energy = pt_energy
                    else:
                        mc_id, mc_e = self.mmff.monte_carlo_search(
                            acyclic_with_H,
                            n_steps=mc_steps_eff,
                            progress_callback=_range_progress(
                                candidate_progress, 0.70, 0.85
                            ),
                        )
                        if mc_id >= 0 and mc_e < mc_energy:
                            mc_energy = mc_e
                    acyclic_energy = mc_energy
                except Exception as exc:
                    logger.warning("Acyclic MC/PT search failed: %s", exc)

                if use_boltzmann:
                    try:
                        if candidate_progress is not None:
                            candidate_progress("Clustering conformers", 85, 100)
                        kept = self.mmff.cluster_conformers(
                            acyclic_with_H,
                            rmsd_threshold=0.5,
                            energy_window_kcal=8.0,
                        )
                        if len(kept) >= 2:
                            keep_confs = [
                                Chem.Conformer(acyclic_with_H.GetConformer(c))
                                for c in kept
                            ]
                            acyclic_with_H.RemoveAllConformers()
                            for c in keep_confs:
                                acyclic_with_H.AddConformer(c, assignId=True)
                        if candidate_progress is not None:
                            candidate_progress("Averaging conformer energies", 95, 100)
                        boltz_e = self.mmff.compute_boltzmann_energy(
                            acyclic_with_H, temperature=298.15,
                        )
                        if not (boltz_e != boltz_e):  # nan check
                            acyclic_energy = boltz_e
                    except Exception as exc:
                        logger.warning(
                            "Acyclic Boltzmann averaging failed: %s", exc,
                        )

                if acyclic_energy < best_energy:
                    best_energy = acyclic_energy
                    best_mol = acyclic_with_H
            except Exception as exc:
                logger.warning(
                    "Acyclic reference for bond %s failed: %s",
                    _broken_bond, exc,
                )
            finally:
                if candidate_progress is not None:
                    candidate_progress("Complete", 1, 1)

        if best_mol is None:
            raise RuntimeError(
                "Failed to build any valid acyclic reference for ring size %d"
                % ring_info.size,
            )

        return best_mol, best_energy

    def _build_all_acyclic_references(
        self,
        mol: Mol,
        ring_info: RingInfo,
    ) -> List[Tuple[Mol, int]]:
        """Build H-capped acyclic references for every single ring bond.

        Returns a list of ``(acyclic_mol, broken_bond_idx)`` tuples, one per
        breakable single bond in the ring.  Callers can then optimise each
        and pick the lowest-energy representative.
        """
        results: List[Tuple[Mol, int]] = []
        ring_atom_set = set(ring_info.atom_indices)
        # Dedupe by canonical SMILES: ring symmetry makes many openings
        # produce the identical molecule (e.g. methylcyclohexane gives only
        # 4 distinct chains from 6 ring bonds; 1,1-dimethylcyclohexane just
        # 2). Each unique chain still gets the full MC/PT + Boltzmann pass;
        # we just avoid paying for it 2-3x over.
        seen_canonical = set()

        for bidx in ring_info.bond_indices:
            rw_mol = RWMol(mol)
            bond = rw_mol.GetBondWithIdx(bidx)
            if bond is None or bond.GetBondType() != rdchem.BondType.SINGLE:
                continue
            # Verify both endpoints are in the ring (safety check)
            if (bond.GetBeginAtomIdx() not in ring_atom_set
                    or bond.GetEndAtomIdx() not in ring_atom_set):
                continue

            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            rw_mol.RemoveBond(begin_idx, end_idx)
            for atom_idx in (begin_idx, end_idx):
                atom = rw_mol.GetAtomWithIdx(atom_idx)
                atom.SetNoImplicit(False)
                atom.SetNumRadicalElectrons(0)
            rw_mol.UpdatePropertyCache(strict=False)
            try:
                Chem.SanitizeMol(rw_mol)
            except Exception as exc:
                logger.warning(
                    "Sanitization failed for acyclic ref (bond %d): %s",
                    bidx, exc,
                )
                continue
            acyclic_mol = rw_mol.GetMol()
            Chem.AssignStereochemistry(acyclic_mol, force=True, cleanIt=True)
            if any(atom.GetNumRadicalElectrons() for atom in acyclic_mol.GetAtoms()):
                logger.warning(
                    "Skipping radical acyclic ref after opening ring bond %d.",
                    bidx,
                )
                continue
            try:
                can = Chem.MolToSmiles(acyclic_mol, canonical=True)
            except Exception:
                can = None
            if can is not None:
                if can in seen_canonical:
                    continue
                seen_canonical.add(can)
            results.append((acyclic_mol, bidx))

        if not results:
            raise RuntimeError(
                "No breakable single ring bond found for ring size %d"
                % ring_info.size,
            )
        return results

    def _get_ethane_energy(self) -> float:
        """Return cached MMFF94 energy for the ethane auxiliary reactant."""
        global _ETHANE_ENERGY
        if _ETHANE_ENERGY is None:
            ethane_mol = Chem.MolFromSmiles("CC")
            ethane_result = self.mmff.embed_and_optimize(ethane_mol)
            _ETHANE_ENERGY = ethane_result.total_energy
        return _ETHANE_ENERGY

    @staticmethod
    def _choose_bond_to_break(rw_mol, ring_info: RingInfo) -> Optional[int]:
        """Select the ring bond to break for ring opening.

        Preference order:
        1. Single bonds only.
        2. Among single bonds, the one whose two endpoint ring atoms
           carry the most off-ring heavy-atom substituents combined.
           This places bulky substituents at opposite ends of the
           opened chain.
        """
        ring_atom_set = set(ring_info.atom_indices)

        def substituent_count(atom_idx: int) -> int:
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            return sum(
                1 for nb in atom.GetNeighbors()
                if nb.GetIdx() not in ring_atom_set
            )

        best_idx = None
        best_score = -1
        for bidx in ring_info.bond_indices:
            bond = rw_mol.GetBondWithIdx(bidx)
            if bond is None or bond.GetBondType() != rdchem.BondType.SINGLE:
                continue
            score = (
                substituent_count(bond.GetBeginAtomIdx())
                + substituent_count(bond.GetEndAtomIdx())
            )
            if score > best_score:
                best_score = score
                best_idx = bidx

        if best_idx is not None:
            return best_idx
        return ring_info.bond_indices[0] if ring_info.bond_indices else None

    # ------------------------------------------------------------------
    # Strict homodesmotic reaction for unsubstituted cycloalkanes
    # ------------------------------------------------------------------

    def build_cycloalkane_homodesmotic_reaction(
        self,
        ring_size: int,
    ) -> Optional[HomodesmoticReaction]:
        """Build the bond-balanced homodesmotic reaction.

        cyclo-(CH2)n + CH3-CH3  ->  CH3-(CH2)(n+1)-CH3
        """
        if ring_size < 3:
            return None

        cyclic = "C1" + "C" * (ring_size - 2) + "C1"
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
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[float]:
        """Compute raw homodesmotic strain for an unsubstituted cycloalkane.

        Uses the strict bond-balanced reaction scheme.
        Results are cached at module level.
        """
        global _ETHANE_ENERGY, _CYCLOALKANE_STRAIN_CACHE

        if ring_size in _CYCLOALKANE_STRAIN_CACHE:
            if progress_callback is not None:
                progress_callback("Using cached cycloalkane reference", 1, 1)
            return _CYCLOALKANE_STRAIN_CACHE[ring_size]

        reaction = self.build_cycloalkane_homodesmotic_reaction(ring_size)
        if reaction is None:
            return None

        try:
            cyclic_mol = Chem.MolFromSmiles(reaction.cyclic_reactant_smiles)
            cyclic_result = self.mmff.embed_and_optimize(
                cyclic_mol,
                progress_callback=_subtask_progress(
                    progress_callback, 1, 3, "Cycloalkane reaction"
                ),
            )
            cyclic_energy = cyclic_result.total_energy

            if _ETHANE_ENERGY is None:
                ethane_mol = Chem.MolFromSmiles("CC")
                ethane_result = self.mmff.embed_and_optimize(
                    ethane_mol,
                    progress_callback=_subtask_progress(
                        progress_callback, 2, 3, "Cycloalkane reaction"
                    ),
                )
                _ETHANE_ENERGY = ethane_result.total_energy
            elif progress_callback is not None:
                progress_callback("Cycloalkane reaction 2/3: cached ethane", 2, 3)

            linear_mol = Chem.MolFromSmiles(reaction.acyclic_product_smiles)
            linear_result = self.mmff.embed_and_optimize(
                linear_mol,
                progress_callback=_subtask_progress(
                    progress_callback, 3, 3, "Cycloalkane reaction"
                ),
            )
            linear_energy = linear_result.total_energy

            strain = cyclic_energy + _ETHANE_ENERGY - linear_energy
            _CYCLOALKANE_STRAIN_CACHE[ring_size] = strain
            return strain

        except Exception as exc:
            logger.error(
                "Cycloalkane homodesmotic strain failed for ring size %d: %s",
                ring_size, exc,
            )
            return None


def _is_unsubstituted_cycloalkane(mol: Mol, ring_info: RingInfo) -> bool:
    """Check if molecule is a pure cycloalkane (no substituents, no heteroatoms).

    All atoms must be carbon, all bonds must be single bonds, the ring
    size must be <= 8, and the heavy-atom count must exactly equal the
    ring size (no extra atoms beyond the ring).
    """
    if ring_info.size > 8:
        return False
    if mol.GetNumHeavyAtoms() != ring_info.size:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            return False
    for bond in mol.GetBonds():
        if bond.GetBondType() != rdchem.BondType.SINGLE:
            return False
    return True
