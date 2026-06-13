"""
MMFF94 force field wrapper for geometry optimization and energy calculation.

Provides robust conformer embedding, multi-conformer sampling, MMFF94
optimization, and energy computation with error handling for edge cases.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdchem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeProperties,
    MMFFGetMoleculeForceField,
)
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C++ stderr suppression
# ---------------------------------------------------------------------------

import contextlib
import os as _os


@contextlib.contextmanager
def _suppress_cpp_stderr():
    """Temporarily redirect file descriptor 2 (stderr) to /dev/null.

    RDKit's C++ BFGS optimizer writes "Invariant Violation" messages
    directly to stderr, bypassing Python's logging. This context manager
    silences that noise for operations known to trigger it (ETKDG
    conformer embedding and MMFF94 optimization).
    """
    saved_fd = _os.dup(2)
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        _os.dup2(saved_fd, 2)
        _os.close(saved_fd)
        _os.close(devnull_fd)


@dataclass
class MMFFResult:
    """Results of an MMFF94 calculation."""
    molecule: Mol                         # Optimized molecule (with conformer)
    total_energy: float                   # kcal/mol
    conformer_count: int                  # Number of conformers sampled
    best_conf_id: int                     # ID of the lowest-energy conformer
    optimization_converged: bool
    num_heavy_atoms: int
    energy_per_heavy_atom: float
    mmff_param_coverage: float            # Fraction of atoms with MMFF94 parameters
    energy_breakdown: Dict[str, float] = field(default_factory=dict)


class MMFFCalculator:
    """Robust MMFF94 energy and geometry calculator.

    Handles conformer generation, MMFF94 force field optimization,
    and energy computation for organic molecules.

    Parameters
    ----------
    n_conformers : int
        Number of initial conformers to sample.
    max_optimization_iters : int
        Maximum MMFF minimization steps.
    energy_convergence : float
        Convergence threshold for optimization (kcal/mol).
    random_seed : int
        Seed for reproducible conformer generation.
    use_etkdg : bool
        Use ETKDGv3 embedding (better for ring systems).
    """

    def __init__(
        self,
        n_conformers: int = 200,
        max_optimization_iters: int = 2000,
        energy_convergence: float = 1e-7,
        random_seed: int = 42,
        use_etkdg: bool = True,
    ):
        self.n_conformers = n_conformers
        self.max_optimization_iters = max_optimization_iters
        self.energy_convergence = energy_convergence
        self.random_seed = random_seed
        self.use_etkdg = use_etkdg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_and_optimize(self, mol: Mol) -> MMFFResult:
        """Full pipeline: sanitize, add H, generate conformers, MMFF94 optimize.

        Returns the molecule with the lowest-energy conformer and
        associated metadata.
        """
        mol = self.sanitize_for_mmff(mol)
        mol = Chem.AddHs(mol)

        n_heavy = CalcNumHeavyAtoms(mol)
        if n_heavy < 1:
            raise ValueError("Molecule has no heavy atoms.")

        # Adaptive conformer count: rigid molecules need far fewer samples
        # than flexible ones. Measure flexibility by rotatable bonds count.
        effective_n_conf = self._compute_effective_conformers(mol)

        # Suppress C++ stderr noise during the conformer+optimize phase.
        # RDKit's BFGS optimizer can emit "Invariant Violation: bad
        # direction in linearSearch" directly to stderr from C++ when
        # the initial geometry is poor (e.g., cyclopentane ETKDG failure).
        # This is non-fatal — RDKit handles it internally and convergence
        # still succeeds — but the C++ output bypasses Python's logging.
        with _suppress_cpp_stderr():
            conf_ids = self.generate_conformers(mol, n_conformers=effective_n_conf)
            if not conf_ids:
                raise RuntimeError(
                    f"Failed to generate any conformers for molecule with "
                    f"{n_heavy} heavy atoms."
                )

            best_conf_id, best_energy, all_converged = self._optimize_all_conformers(
                mol, conf_ids
            )

        # Keep all conformers on the molecule (do NOT remove/re-add --
        # that invalidates C++ Conformer references in RDKit 2022.09).

        coverage = self.check_mmff_coverage(mol)
        if coverage < 1.0:
            logger.warning(
                "MMFF94 parameter coverage = %.1f%% for this molecule. "
                "Some atoms lack MMFF94 parameters; results may be approximate.",
                coverage * 100,
            )

        breakdown = self.get_energy_breakdown(mol, conf_id=best_conf_id)

        return MMFFResult(
            molecule=mol,
            total_energy=best_energy,
            conformer_count=len(conf_ids),
            best_conf_id=best_conf_id,
            optimization_converged=all_converged,
            num_heavy_atoms=n_heavy,
            energy_per_heavy_atom=best_energy / n_heavy,
            mmff_param_coverage=coverage,
            energy_breakdown=breakdown,
        )

    def compute_single_point_energy(self, mol: Mol, conf_id: int = -1) -> float:
        """Compute MMFF94 single-point energy at the current geometry.

        The molecule must already have hydrogens added and a conformer.
        """
        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return float("nan")

        ff = MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return float("nan")

        return ff.CalcEnergy()

    def compute_boltzmann_energy(
        self,
        mol: Mol,
        temperature: float = 298.15,
        energy_window_kcal: float = 10.0,
    ) -> float:
        """Compute Boltzmann-weighted free energy across all conformers.

            E_thermal = -RT * ln( sum_i exp(-(E_i - E_min) / RT) ) + E_min

        Conformers more than ``energy_window_kcal`` above the minimum are
        skipped (negligible weight at room temperature).

        This is the thermodynamically correct quantity for strain
        calculation when the molecule has multiple low-lying conformers,
        which is exactly the situation for bulky substituted rings.
        """
        import math as _math

        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return float("nan")

        conformer_ids = [conf.GetId() for conf in mol.GetConformers()]
        if not conformer_ids:
            return float("nan")

        energies = []
        for cid in conformer_ids:
            ff = MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                continue
            try:
                energies.append(ff.CalcEnergy())
            except Exception:
                continue

        if not energies:
            return float("nan")

        R = 0.001987  # kcal/(mol K)
        RT = R * temperature

        e_min = min(energies)
        partition = 0.0
        for e in energies:
            de = e - e_min
            if de > energy_window_kcal:
                continue
            partition += _math.exp(-de / RT)

        if partition <= 0:
            return e_min

        return e_min - RT * _math.log(partition)

    def generate_conformers(
        self, mol: Mol, n_conformers: Optional[int] = None
    ) -> List[int]:
        """Generate diverse conformers using ETKDG or standard DG.

        The molecule must already have explicit hydrogens added.

        Parameters
        ----------
        n_conformers : Optional[int]
            Number of conformers to embed. Defaults to ``self.n_conformers``.
            Callers pass the adaptive (flexibility-based) count here so that
            rigid molecules are not oversampled.

        Returns a list of conformer IDs that were successfully embedded.

        Embedding is parallelized across all available cores (numThreads=0).
        With a fixed randomSeed, RDKit assigns deterministic per-conformer
        seeds, so results are reproducible regardless of thread count.
        """
        if n_conformers is None:
            n_conformers = self.n_conformers

        # Strategy 1: ETKDG with torsion preferences and small-ring corrections
        # (best for most organic molecules, especially ring systems)
        conf_ids = []
        try:
            conf_ids = list(
                rdDistGeom.EmbedMultipleConfs(
                    mol,
                    numConfs=n_conformers,
                    randomSeed=self.random_seed,
                    useExpTorsionAnglePrefs=self.use_etkdg,
                    useBasicKnowledge=True,
                    useSmallRingTorsions=True,
                    numThreads=0,
                )
            )
        except Exception:
            pass

        # Strategy 2: ETKDG without torsion preferences (still uses DG + basic
        # knowledge, which gives better geometries than pure random coords)
        if not conf_ids:
            logger.warning(
                "ETKDG embedding produced 0 conformers; retrying without "
                "torsion preferences (standard DG with basic knowledge)."
            )
            try:
                conf_ids = list(
                    rdDistGeom.EmbedMultipleConfs(
                        mol,
                        numConfs=n_conformers,
                        randomSeed=self.random_seed,
                        useExpTorsionAnglePrefs=False,
                        useBasicKnowledge=True,
                        useSmallRingTorsions=True,
                        numThreads=0,
                    )
                )
            except Exception:
                pass

        # Strategy 3: Pure random coordinates (last resort — may produce bad
        # geometries that cause BFGS optimizer noise)
        if not conf_ids:
            logger.warning(
                "DG with basic knowledge also failed; falling back to "
                "random coordinates."
            )
            try:
                conf_ids = list(
                    rdDistGeom.EmbedMultipleConfs(
                        mol,
                        numConfs=min(n_conformers, 100),
                        randomSeed=self.random_seed,
                        useRandomCoords=True,
                        useBasicKnowledge=False,
                        numThreads=0,
                    )
                )
            except Exception as exc:
                raise RuntimeError(f"All conformer embedding methods failed: {exc}")

        if not conf_ids:
            raise RuntimeError("No conformers were generated.")

        return conf_ids

    def optimize_conformer(self, mol: Mol, conf_id: int) -> Tuple[float, bool]:
        """MMFF94-optimize a single conformer.

        Returns (optimized_energy, converged).
        """
        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return float("inf"), False

        ff = MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return float("inf"), False

        try:
            converged = ff.Minimize(
                maxIts=self.max_optimization_iters,
                forceTol=self.energy_convergence,
            )
        except Exception:
            return float("inf"), False

        energy = ff.CalcEnergy()
        return energy, converged == 0

    def seed_random_coords_conformers(
        self,
        mol: Mol,
        n_seeds: int = 20,
        seed_offset: int = 2000,
    ) -> List[int]:
        """Add ``n_seeds`` random-coords ETKDG conformers, MMFF94-relaxed.

        ``useRandomCoords=True`` distance-geometry seeds the conformer
        search from random Cartesian positions instead of from CSD-derived
        torsion preferences. This is the key technique for breaking out of
        whatever basin standard ETKDG happens to land in: for acyclic
        chains it surfaces both extended and gauche rotamer families; for
        rings it samples every puckering basin. Existing conformers are
        preserved (``clearConfs=False``).

        Returns the list of newly added conformer IDs that survived MMFF94
        optimization. ``seed_offset`` is added to ``self.random_seed`` so
        callers can request multiple independent random-coords passes
        without re-using the same draw.
        """
        new_ids: List[int] = []
        try:
            with _suppress_cpp_stderr():
                extra_ids = list(
                    rdDistGeom.EmbedMultipleConfs(
                        mol,
                        numConfs=n_seeds,
                        randomSeed=self.random_seed + seed_offset,
                        useRandomCoords=True,
                        useBasicKnowledge=True,
                        useSmallRingTorsions=True,
                        clearConfs=False,
                        numThreads=0,
                    )
                )
        except Exception as exc:
            logger.warning("Random-coords seed embedding failed: %s", exc)
            return []

        for cid in extra_ids:
            energy, _ = self.optimize_conformer(mol, cid)
            if energy == float("inf"):
                try:
                    mol.RemoveConformer(cid)
                except Exception:
                    pass
                continue
            new_ids.append(cid)
        return new_ids

    def seed_ring_pucker_conformers(
        self,
        mol: Mol,
        ring_atom_indices: List[int],
        n_seeds: int = 24,
    ) -> List[int]:
        """Add ring-puckering-diverse conformer seeds for 4-7 membered rings.

        Delegates to ``seed_random_coords_conformers`` after gating on ring
        size — random-coords DG hits every puckering basin (chair-flip for
        6-rings, envelope vs twist for 5-rings, etc.) that torsion-preference
        ETKDG can lock out. MC/PT then refines substituent rotamers on top
        of each pucker.
        """
        ring_size = len(ring_atom_indices)
        if ring_size < 4 or ring_size > 7:
            return []
        return self.seed_random_coords_conformers(
            mol, n_seeds=n_seeds, seed_offset=1000,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sanitize_for_mmff(mol: Mol) -> Mol:
        """Prepare molecule for MMFF94: add explicit hydrogens, sanitize."""
        mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Try kekulization then re-sanitize
            try:
                Chem.Kekulize(mol)
                Chem.SanitizeMol(mol)
            except Exception:
                pass
        return mol

    @staticmethod
    def check_mmff_coverage(mol: Mol) -> float:
        """Return fraction of atoms with valid MMFF94 parameters (0.0-1.0)."""
        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return 0.0
        covered = sum(
            1 for a in mol.GetAtoms() if props.GetMMFFAtomType(a.GetIdx()) is not None
        )
        n_atoms = mol.GetNumAtoms()
        return covered / n_atoms if n_atoms > 0 else 0.0

    @staticmethod
    def get_energy_breakdown(mol: Mol, conf_id: int = -1) -> Dict[str, float]:
        """Return MMFF94 per-term energy decomposition.

        Builds one full force field for the total energy, then seven trimmed
        force fields each with one term toggled off, and reports each term as
        ``E_full - E_without_that_term``. Uses the per-term toggles exposed
        on ``MMFFMolProperties`` (SetMMFF{Bond,Angle,StretchBend,Oop,Torsion,
        VdW,Ele}Term) — the only way to isolate per-term contributions
        through the Python API in this RDKit build.
        """
        return MMFFCalculator.decompose_energy(mol, conf_id=conf_id)

    @staticmethod
    def decompose_energy(mol: Mol, conf_id: int = -1) -> Dict[str, float]:
        """MMFF94 per-term decomposition.

        Returns a dict with keys ``total``, ``bond``, ``angle``,
        ``stretch_bend``, ``oop``, ``torsion``, ``vdw``, ``electrostatic``.
        Each component is the energy lost when its term is disabled —
        i.e. its contribution to the total. The ``vdw`` term is the
        physically meaningful "steric" component of MMFF94.
        """
        # Total energy
        full_props = MMFFGetMoleculeProperties(mol)
        if full_props is None:
            return {"total": float("nan")}
        full_ff = MMFFGetMoleculeForceField(mol, full_props, confId=conf_id)
        if full_ff is None:
            return {"total": float("nan")}
        try:
            total = float(full_ff.CalcEnergy())
        except Exception:
            return {"total": float("nan")}

        # Per-term contributions via toggle-off
        term_setters = {
            "bond":          "SetMMFFBondTerm",
            "angle":         "SetMMFFAngleTerm",
            "stretch_bend":  "SetMMFFStretchBendTerm",
            "oop":           "SetMMFFOopTerm",
            "torsion":       "SetMMFFTorsionTerm",
            "vdw":           "SetMMFFVdWTerm",
            "electrostatic": "SetMMFFEleTerm",
        }
        breakdown: Dict[str, float] = {"total": total}
        for key, setter in term_setters.items():
            props = MMFFGetMoleculeProperties(mol)
            if props is None or not hasattr(props, setter):
                breakdown[key] = float("nan")
                continue
            getattr(props, setter)(False)
            ff = MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            if ff is None:
                breakdown[key] = float("nan")
                continue
            try:
                e_without = float(ff.CalcEnergy())
                breakdown[key] = total - e_without
            except Exception:
                breakdown[key] = float("nan")
        return breakdown

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_effective_conformers(self, mol: Mol) -> int:
        """Return an adaptive conformer count based on molecular flexibility.

        For bulky substituted rings the conformer space grows fast and the
        budget must scale with both rotational freedom AND branching: a
        tert-butyl (one quaternary carbon, three methyls) creates many more
        local minima than a propyl of the same heavy-atom count, because each
        branch interacts independently with its environment.

        Formula: 50 + 40 * rot_bonds + 25 * bulk_score, clamped to
        [20, max(self.n_conformers, 5000)]. ``bulk_score`` weights each
        off-ring heavy atom by ``max(1, heavy_degree - 1)**2`` so that
        quaternary centers count 9, branched centers 4, and linear chain
        atoms 1. See ``compute_bulk_score``.
        """
        n_heavy = CalcNumHeavyAtoms(mol)
        if n_heavy <= 1:
            return 1

        n_rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        # Cap bulk at 15 for sampling-budget purposes: beyond that the
        # conformer space is dominated by long-range vdW couplings that
        # are already covered by PT replicas, so adding linearly more
        # conformers gives diminishing returns.
        bulk = min(15, self.compute_bulk_score(mol))

        # Reduced from 50+40r+25b (cap 5000) — that produced 510-conformer
        # pools for tert-butyl-substituted rings, taking ~30 s just in
        # initial MMFF optimization. The smaller pool still seeds every
        # major basin once combined with random-coords ring puckering seeds.
        effective = 30 + 20 * n_rot_bonds + 15 * bulk
        upper = max(self.n_conformers, 1500)
        effective = max(20, min(upper, effective))

        logger.debug(
            "Adaptive conformers: heavy=%d rot_bonds=%d bulk=%d → %d",
            n_heavy, n_rot_bonds, bulk, effective,
        )
        return effective

    @staticmethod
    def compute_bulk_score(mol: Mol) -> int:
        """Branching-weighted measure of substituent steric complexity.

        For each non-ring heavy atom, contributes ``max(1, heavy_degree - 1)**2``
        to the score — so a linear CH2 counts 1, a CH (branched) counts 4,
        and a quaternary C counts 9. Captures the conformer-space combinatorics
        of bulky substituents (a tert-butyl group scores 12, an n-butyl scores
        4).
        """
        ring_atoms = set()
        for ring in Chem.GetSymmSSSR(mol):
            ring_atoms.update(ring)

        score = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            if atom.GetIdx() in ring_atoms:
                continue
            heavy_deg = sum(
                1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1
            )
            score += max(1, heavy_deg - 1) ** 2
        return score

    def _optimize_all_conformers(
        self, mol: Mol, conf_ids: List[int]
    ) -> Tuple[int, float, bool]:
        """Optimize all conformers, return (best_conf_id, best_energy, all_converged).

        Uses RDKit's native batch optimizer (MMFFOptimizeMoleculeConfs) with
        numThreads=0, which minimizes every conformer in parallel C++ threads
        instead of a serial Python loop. Falls back to the serial loop if the
        batch call is unavailable or raises.
        """
        try:
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol,
                numThreads=0,
                maxIters=self.max_optimization_iters,
            )
        except Exception:
            results = None

        if results:
            best_energy = float("inf")
            best_conf_id = -1
            all_converged = True
            for conf_id, (not_converged, energy) in zip(conf_ids, results):
                if not_converged != 0:
                    all_converged = False
                if energy < best_energy:
                    best_energy = energy
                    best_conf_id = conf_id
            if best_conf_id != -1:
                return best_conf_id, best_energy, all_converged

        # Fallback: serial per-conformer minimization
        best_energy = float("inf")
        best_conf_id = -1
        all_converged = True
        for conf_id in conf_ids:
            energy, converged = self.optimize_conformer(mol, conf_id)
            if not converged:
                all_converged = False
            if energy < best_energy:
                best_energy = energy
                best_conf_id = conf_id

        return best_conf_id, best_energy, all_converged

    # ------------------------------------------------------------------
    # Monte Carlo conformer search
    # ------------------------------------------------------------------

    def monte_carlo_search(
        self,
        mol: Mol,
        n_steps: int = 500,
        temperature: float = 300.0,
        perturb_fraction: float = 0.3,
    ) -> Tuple[int, float]:
        """Monte Carlo torsion search for global minimum conformer.

        Randomly perturbs rotatable bond torsions, optimizes each trial
        with MMFF94, and accepts/rejects via the Metropolis criterion.
        Tracks the lowest-energy conformer found across all steps.

        Critical for substituted cycloalkanes where steric interactions
        between substituents create a complex conformational landscape.

        Parameters
        ----------
        mol : Mol
            Molecule with conformers (takes the best existing conformer
            as the starting point).
        n_steps : int
            Number of Monte Carlo steps.
        temperature : float
            Temperature for Metropolis acceptance (K). Lower = more
            selective for downhill moves.
        perturb_fraction : float
            Fraction of rotatable bonds to perturb per step.

        Returns:
            (best_conf_id, best_energy_kcal_mol)
        """
        import random as _random
        import numpy as _np

        rng = _np.random.RandomState(self.random_seed)
        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return -1, float("inf")

        # Identify rotatable bonds (exclude bonds inside rings)
        rotatable_bonds = []
        ssr = Chem.GetSymmSSSR(mol)
        ring_bond_set = set()
        for ring in ssr:
            ring_atoms = list(ring)
            for i in range(len(ring_atoms)):
                a1 = ring_atoms[i]
                a2 = ring_atoms[(i + 1) % len(ring_atoms)]
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is not None:
                    ring_bond_set.add(bond.GetIdx())

        for bond in mol.GetBonds():
            if bond.GetIdx() not in ring_bond_set:
                if bond.GetBondType() == rdchem.BondType.SINGLE:
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    if (a1.GetAtomicNum() > 1 and a2.GetAtomicNum() > 1):
                        rotatable_bonds.append(bond.GetIdx())

        # Iterate over actual conformer IDs, not range(N). IDs become
        # non-contiguous once any conformer is removed (e.g. after a prior
        # PT search), and `range(N)` would request IDs that don't exist,
        # triggering RDKit's "Bad Conformer Id" error.
        existing_ids = [c.GetId() for c in mol.GetConformers()]

        if not rotatable_bonds:
            best_conf_id = -1
            best_energy = float("inf")
            for cid in existing_ids:
                ff = MMFFGetMoleculeForceField(mol, props, confId=cid)
                if ff is None:
                    continue
                energy = ff.CalcEnergy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf_id = cid
            return best_conf_id, best_energy

        # Find starting conformer (lowest energy among existing)
        start_conf_id = -1
        start_energy = float("inf")
        for cid in existing_ids:
            ff = MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                continue
            energy = ff.CalcEnergy()
            if energy < start_energy:
                start_energy = energy
                start_conf_id = cid

        if start_conf_id < 0:
            return -1, float("inf")

        current_conf = mol.GetConformer(start_conf_id)
        current_energy = start_energy
        best_conf_id = start_conf_id
        best_energy = start_energy

        R = 0.001987  # Gas constant in kcal/(mol*K)
        n_perturb = max(1, int(len(rotatable_bonds) * perturb_fraction))

        # Pre-compute heavy-atom indices and bonded pairs once so the
        # vdW-clash pre-screen runs in O(n^2) without re-scanning topology
        # every step.
        heavy_idxs, bonded_pairs = _collect_heavy_atoms(mol)

        for step in range(n_steps):
            new_conf = Chem.Conformer(current_conf)
            new_conf_id = mol.AddConformer(new_conf, assignId=True)

            bonds_to_perturb = rng.choice(
                rotatable_bonds,
                size=min(n_perturb, len(rotatable_bonds)),
                replace=False,
            )
            for bidx in bonds_to_perturb:
                bond = mol.GetBondWithIdx(int(bidx))
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()

                a0 = _find_neighbor(mol, a1, exclude=a2)
                a3 = _find_neighbor(mol, a2, exclude=a1)
                if a0 is None or a3 is None:
                    continue

                current_dihedral = _get_dihedral(new_conf, a0, a1, a2, a3)
                # Mixed perturbation: 80% small Gaussian (local refinement,
                # keeps Metropolis acceptance > 30% for bulky molecules),
                # 20% large uniform jump (escapes local basins).
                if rng.random() < 0.8:
                    perturbation = float(rng.normal(0.0, 20.0))
                else:
                    perturbation = float(rng.uniform(-120.0, 120.0))
                new_dihedral = current_dihedral + perturbation
                _set_dihedral(new_conf, a0, a1, a2, a3, new_dihedral)

            # vdW-clash pre-screen: a large dihedral jump on a bulky group
            # can drive non-bonded heavy atoms inside a covalent-bond
            # distance, which MMFF then minimizes into a frustrated state.
            # Reject such moves before paying for the optimization.
            if _has_heavy_clash(new_conf, heavy_idxs, bonded_pairs=bonded_pairs):
                mol.RemoveConformer(new_conf_id)
                continue

            ff = MMFFGetMoleculeForceField(mol, props, confId=new_conf_id)
            if ff is None:
                mol.RemoveConformer(new_conf_id)
                continue

            try:
                ff.Minimize(
                    maxIts=self.max_optimization_iters,
                    forceTol=self.energy_convergence,
                )
            except Exception:
                mol.RemoveConformer(new_conf_id)
                continue

            new_energy = ff.CalcEnergy()

            delta = new_energy - current_energy
            if delta < 0:
                accept = True
            else:
                prob = _np.exp(-delta / (R * temperature))
                accept = rng.random() < prob

            if accept:
                current_conf = mol.GetConformer(new_conf_id)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_conf_id = new_conf_id
            else:
                mol.RemoveConformer(new_conf_id)

        return best_conf_id, best_energy

    # ------------------------------------------------------------------
    # Parallel tempering MC (replica exchange)
    # ------------------------------------------------------------------

    def parallel_tempering_search(
        self,
        mol: Mol,
        n_steps: int = 300,
        temperatures: Tuple[float, ...] = (300.0, 500.0, 1000.0, 2000.0),
        swap_interval: int = 20,
        perturb_fraction: float = 0.3,
    ) -> Tuple[int, float]:
        """Replica-exchange MC for crossing energy barriers in bulky rings.

        Runs ``len(temperatures)`` independent MC chains in parallel
        (logically — sequential here for simplicity).  Every
        ``swap_interval`` steps, attempts to swap conformations between
        adjacent temperature replicas with Metropolis criterion
            min(1, exp((1/RT_i - 1/RT_j)(E_i - E_j)))
        High-T replicas escape basins, low-T replicas refine minima.
        Returns the lowest-energy conformer seen across all replicas.
        """
        import numpy as _np

        rng = _np.random.RandomState(self.random_seed)
        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return -1, float("inf")

        # Identify rotatable bonds (same logic as monte_carlo_search)
        ssr = Chem.GetSymmSSSR(mol)
        ring_bond_set = set()
        for ring in ssr:
            ring_atoms = list(ring)
            for i in range(len(ring_atoms)):
                a1 = ring_atoms[i]
                a2 = ring_atoms[(i + 1) % len(ring_atoms)]
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is not None:
                    ring_bond_set.add(bond.GetIdx())

        rotatable_bonds = []
        for bond in mol.GetBonds():
            if bond.GetIdx() in ring_bond_set:
                continue
            if bond.GetBondType() != rdchem.BondType.SINGLE:
                continue
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if a1.GetAtomicNum() > 1 and a2.GetAtomicNum() > 1:
                rotatable_bonds.append(bond.GetIdx())

        # Find starting conformer (lowest energy among existing). Iterate
        # actual conformer IDs — they are not guaranteed to be 0..N-1 once
        # any conformer has been removed.
        existing_ids = [c.GetId() for c in mol.GetConformers()]
        start_conf_id = -1
        start_energy = float("inf")
        for cid in existing_ids:
            ff = MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                continue
            e = ff.CalcEnergy()
            if e < start_energy:
                start_energy = e
                start_conf_id = cid

        if start_conf_id < 0:
            return -1, float("inf")

        if not rotatable_bonds:
            return start_conf_id, start_energy

        R = 0.001987
        n_perturb = max(1, int(len(rotatable_bonds) * perturb_fraction))

        # Initialize one replica per temperature, each starts from the best
        # existing conformer.
        base_conf = mol.GetConformer(start_conf_id)
        replicas = []
        for _ in temperatures:
            cid = mol.AddConformer(Chem.Conformer(base_conf), assignId=True)
            replicas.append({"conf_id": cid, "energy": start_energy})

        best_energy = start_energy
        # Snapshot the coordinates of the best conformer rather than tracking
        # its conformer ID. PT removes old replica conformers when a trial is
        # accepted, which would otherwise leave best_conf_id stale.
        best_conf_snapshot = Chem.Conformer(base_conf)

        # Pre-compute heavy-atom indices and bonded pairs for the clash
        # pre-screen — see monte_carlo_search for the rationale.
        heavy_idxs, bonded_pairs = _collect_heavy_atoms(mol)

        for step in range(n_steps):
            # One MC move per replica at its own temperature
            for ridx, T in enumerate(temperatures):
                rep = replicas[ridx]
                current_conf = mol.GetConformer(rep["conf_id"])

                trial_conf = Chem.Conformer(current_conf)
                trial_cid = mol.AddConformer(trial_conf, assignId=True)

                bonds_to_perturb = rng.choice(
                    rotatable_bonds,
                    size=min(n_perturb, len(rotatable_bonds)),
                    replace=False,
                )
                for bidx in bonds_to_perturb:
                    bond = mol.GetBondWithIdx(int(bidx))
                    a1 = bond.GetBeginAtomIdx()
                    a2 = bond.GetEndAtomIdx()
                    a0 = _find_neighbor(mol, a1, exclude=a2)
                    a3 = _find_neighbor(mol, a2, exclude=a1)
                    if a0 is None or a3 is None:
                        continue
                    cur_dih = _get_dihedral(trial_conf, a0, a1, a2, a3)
                    # High-T replicas use larger steps
                    if T > 800:
                        perturb = float(rng.uniform(-150.0, 150.0))
                    else:
                        if rng.random() < 0.8:
                            perturb = float(rng.normal(0.0, 25.0))
                        else:
                            perturb = float(rng.uniform(-120.0, 120.0))
                    _set_dihedral(trial_conf, a0, a1, a2, a3, cur_dih + perturb)

                # vdW clash pre-screen — high-T replicas use ±150° jumps that
                # often slam bulky groups into each other; rejecting them here
                # saves the MMFF minimization cost on dead-end moves.
                if _has_heavy_clash(
                    trial_conf, heavy_idxs, bonded_pairs=bonded_pairs,
                ):
                    mol.RemoveConformer(trial_cid)
                    continue

                ff = MMFFGetMoleculeForceField(mol, props, confId=trial_cid)
                if ff is None:
                    mol.RemoveConformer(trial_cid)
                    continue
                try:
                    ff.Minimize(
                        maxIts=self.max_optimization_iters,
                        forceTol=self.energy_convergence,
                    )
                except Exception:
                    mol.RemoveConformer(trial_cid)
                    continue
                trial_energy = ff.CalcEnergy()

                delta = trial_energy - rep["energy"]
                if delta < 0:
                    accept = True
                else:
                    prob = float(_np.exp(-delta / (R * T)))
                    accept = float(rng.random()) < prob

                if accept:
                    if trial_energy < best_energy:
                        best_energy = trial_energy
                        best_conf_snapshot = Chem.Conformer(
                            mol.GetConformer(trial_cid)
                        )
                    mol.RemoveConformer(rep["conf_id"])
                    rep["conf_id"] = trial_cid
                    rep["energy"] = trial_energy
                else:
                    mol.RemoveConformer(trial_cid)

            # Replica exchange: try swapping adjacent (T_i, T_j) pairs
            if (step + 1) % swap_interval == 0:
                for i in range(len(temperatures) - 1):
                    rep_i = replicas[i]
                    rep_j = replicas[i + 1]
                    T_i = temperatures[i]
                    T_j = temperatures[i + 1]
                    delta = (1.0 / (R * T_i) - 1.0 / (R * T_j)) * (
                        rep_i["energy"] - rep_j["energy"]
                    )
                    if delta >= 0:
                        accept_swap = True
                    else:
                        accept_swap = float(rng.random()) < float(_np.exp(delta))
                    if accept_swap:
                        replicas[i], replicas[i + 1] = rep_j, rep_i

        # Re-attach the best snapshot as a fresh conformer with a valid ID
        best_conf_id = mol.AddConformer(best_conf_snapshot, assignId=True)
        return best_conf_id, best_energy

    # ------------------------------------------------------------------
    # Conformer clustering
    # ------------------------------------------------------------------

    def cluster_conformers(
        self,
        mol: Mol,
        rmsd_threshold: float = 0.5,
        energy_window_kcal: float = 10.0,
    ) -> List[int]:
        """Cluster conformers by heavy-atom RMSD; return cluster representatives.

        Within the same energy window, keeps the lowest-energy
        representative of each cluster.  Drops conformers beyond
        ``energy_window_kcal`` above the minimum (negligible thermal
        weight). Reduces redundant near-identical geometries that
        otherwise inflate the Boltzmann partition.
        """
        from rdkit.Chem import AllChem as _AllChem

        conformer_ids = [conf.GetId() for conf in mol.GetConformers()]
        if len(conformer_ids) <= 1:
            return conformer_ids

        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return conformer_ids

        energies = []
        for cid in conformer_ids:
            ff = MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                energies.append(float("inf"))
            else:
                try:
                    energies.append(ff.CalcEnergy())
                except Exception:
                    energies.append(float("inf"))

        energy_by_id = dict(zip(conformer_ids, energies))
        e_min = min(energies)
        order = [cid for _, cid in sorted(zip(energies, conformer_ids))]

        keep: List[int] = []
        for cid in order:
            energy = energy_by_id[cid]
            if energy - e_min > energy_window_kcal:
                continue
            duplicate = False
            for kept in keep:
                try:
                    rmsd = _AllChem.GetConformerRMS(
                        mol, cid, kept, prealigned=False
                    )
                except Exception:
                    rmsd = 0.0
                if rmsd < rmsd_threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(cid)

        return keep


# ---------------------------------------------------------------------------
# Monte Carlo helper functions (module-level)
# ---------------------------------------------------------------------------

# Minimum allowed heavy-atom separation for a trial MC move. Set just below
# a single C-C bond (1.54 A) so any pair closer than this is a non-bonded
# clash that MMFF would minimize into a high-energy frustrated state.
# Choosing 1.3 A avoids rejecting any chemically reachable configuration
# (bonded pairs sit around 1.4-1.5 A, non-bonded contacts are >= ~2.5 A).
_VDW_CLASH_THRESHOLD_A = 1.3


def _has_heavy_clash(
    conf,
    heavy_idxs,
    threshold: float = _VDW_CLASH_THRESHOLD_A,
    bonded_pairs=None,
) -> bool:
    """Fast O(n^2) check for any non-bonded heavy-atom pair closer than ``threshold``.

    Returns True if a clash is found. ``bonded_pairs`` is an optional set of
    frozensets of bonded heavy-atom index pairs to exclude from the check.
    Used as a pre-screen before MMFF minimization: a perturbed conformer
    whose closest non-bonded heavy pair is below 1.3 A almost always
    relaxes to a frustrated high-energy minimum, so we reject it outright
    and skip the expensive optimization.
    """
    import numpy as _np

    if len(heavy_idxs) < 2:
        return False

    coords = _np.array(
        [list(conf.GetAtomPosition(i)) for i in heavy_idxs]
    )
    # Squared distances upper triangle
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    n = len(heavy_idxs)
    thr2 = threshold * threshold

    for i in range(n):
        for j in range(i + 1, n):
            if d2[i, j] >= thr2:
                continue
            if bonded_pairs is not None:
                pair = frozenset((heavy_idxs[i], heavy_idxs[j]))
                if pair in bonded_pairs:
                    continue
            return True
    return False


def _collect_heavy_atoms(mol):
    """Return (heavy_idxs, bonded_pairs) tuple for clash pre-screening."""
    heavy_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    bonded_pairs = set()
    for bond in mol.GetBonds():
        b = bond.GetBeginAtom()
        e = bond.GetEndAtom()
        if b.GetAtomicNum() > 1 and e.GetAtomicNum() > 1:
            bonded_pairs.add(frozenset((b.GetIdx(), e.GetIdx())))
    return heavy_idxs, bonded_pairs


def _find_neighbor(mol, atom_idx, exclude):
    """Find a heavy-atom neighbor of atom_idx, excluding the given atom index."""
    atom = mol.GetAtomWithIdx(atom_idx)
    for neighbor in atom.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx != exclude and neighbor.GetAtomicNum() > 1:
            return nidx
    for neighbor in atom.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx != exclude:
            return nidx
    return None


def _get_dihedral(conf, i, j, k, l):
    """Compute dihedral angle in degrees for atoms i-j-k-l."""
    import numpy as _np

    p_i = _np.array(conf.GetAtomPosition(i))
    p_j = _np.array(conf.GetAtomPosition(j))
    p_k = _np.array(conf.GetAtomPosition(k))
    p_l = _np.array(conf.GetAtomPosition(l))

    b1 = p_i - p_j
    b2 = p_k - p_j
    b3 = p_l - p_k

    n1 = _np.cross(b1, b2)
    n2 = _np.cross(b2, b3)

    n1_norm = _np.linalg.norm(n1)
    n2_norm = _np.linalg.norm(n2)

    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0

    cos_phi = _np.dot(n1, n2) / (n1_norm * n2_norm)
    cos_phi = max(-1.0, min(1.0, cos_phi))

    sign = _np.sign(_np.dot(_np.cross(n1, n2), b2))
    if abs(sign) < 1e-10:
        sign = 1.0

    return float(_np.degrees(_np.arccos(cos_phi)) * sign)


def _get_atoms_on_side(mol, bond_begin, bond_end, conf, p_j, p_k):
    """Identify atom indices on the bond_end side of the j-k bond.

    Uses BFS from bond_end, stopping at bond_begin, to collect all
    atoms that should rotate when the j-k dihedral is adjusted.
    """
    from collections import deque as _deque

    atoms_to_move = set()
    visited = {bond_begin}
    queue = _deque([bond_end])

    while queue:
        aidx = queue.popleft()
        if aidx in visited:
            continue
        visited.add(aidx)
        atoms_to_move.add(aidx)

        atom = mol.GetAtomWithIdx(aidx)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx not in visited:
                queue.append(nidx)

    return atoms_to_move


def _set_dihedral(conf, i, j, k, l, target_deg):
    """Rotate atoms on the l-side around bond j-k to set dihedral i-j-k-l."""
    import numpy as _np
    import math as _math

    current = _get_dihedral(conf, i, j, k, l)
    delta_deg = target_deg - current

    while delta_deg > 180.0:
        delta_deg -= 360.0
    while delta_deg < -180.0:
        delta_deg += 360.0

    if abs(delta_deg) < 0.01:
        return

    delta_rad = _math.radians(delta_deg)

    p_j = _np.array(conf.GetAtomPosition(j))
    p_k = _np.array(conf.GetAtomPosition(k))
    axis = p_k - p_j
    axis_norm = _np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return
    axis = axis / axis_norm

    atoms_to_move = _get_atoms_on_side(
        conf.GetOwningMol(), j, k, conf, p_j, p_k
    )

    cos_t = _math.cos(delta_rad)
    sin_t = _math.sin(delta_rad)

    for aidx in atoms_to_move:
        p = _np.array(conf.GetAtomPosition(aidx))
        v = p - p_j
        v_parallel = _np.dot(v, axis) * axis
        v_perp = v - v_parallel
        v_perp_rotated = cos_t * v_perp + sin_t * _np.cross(axis, v_perp)
        new_p = p_j + v_parallel + v_perp_rotated
        conf.SetAtomPosition(
            aidx,
            (float(new_p[0]), float(new_p[1]), float(new_p[2])),
        )
