"""
MMFF94 force field wrapper for geometry optimization and energy calculation.

Provides robust conformer embedding, multi-conformer sampling, MMFF94
optimization, and energy computation with error handling for edge cases.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
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
        max_optimization_iters: int = 500,
        energy_convergence: float = 1e-6,
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
        """Return approximate MMFF94 energy component breakdown.

        RDKit's Python API does not directly expose per-term MMFF94 energies,
        so this provides the total as the authoritative value with a note
        that component decomposition is not available through this interface.
        """
        props = MMFFGetMoleculeProperties(mol)
        if props is None:
            return {"total": float("nan")}
        ff = MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if ff is None:
            return {"total": float("nan")}
        return {"total": ff.CalcEnergy()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_effective_conformers(self, mol: Mol) -> int:
        """Return an adaptive conformer count based on molecular flexibility.

        Rigid ring systems (no rotatable bonds) need few conformers —
        sampling 200 is wasteful for cyclopropane or cubane. Flexible
        molecules with many rotatable bonds get the full requested count.

        Formula: base 10 + 10 per rotatable bond + 5 per ring, clamped to
        [10, self.n_conformers].
        """
        n_heavy = CalcNumHeavyAtoms(mol)
        if n_heavy <= 1:
            return 1

        n_rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_rings = Chem.rdMolDescriptors.CalcNumRings(mol)

        effective = 10 + 10 * n_rot_bonds + 5 * n_rings
        effective = max(5, min(self.n_conformers, effective))

        logger.debug(
            "Adaptive conformers: heavy=%d rot_bonds=%d rings=%d → %d",
            n_heavy, n_rot_bonds, n_rings, effective,
        )
        return effective

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
