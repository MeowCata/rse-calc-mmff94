"""
Calibration of raw MMFF94 homodesmotic strain to experimental scale.

MMFF94 is parameterized for conformational energetics, not heats of
formation. Its raw homodesmotic strain values deviate systematically
from experimental ring strain energies, particularly for small rings.
This module computes per-ring-size correction factors and applies them.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .reference import ReferenceCompound, ReferenceDatabase
from .mmff import MMFFCalculator
from .homodesmotic import HomodesmoticAnalyzer

import logging

logger = logging.getLogger(__name__)


class StrainCalibrator:
    """Calibrate raw MMFF94 homodesmotic strain to experimental scale.

    Computes per-ring-size correction factors from reference compounds,
    applies them to arbitrary ring systems, and handles polycyclic
    decomposition.

    Calibration factors are cached at class level — the reference data
    and MMFF94 force field are deterministic, so calibration needs to
    run only once per process.

    Parameters
    ----------
    reference_db : ReferenceDatabase
        Database of compounds with known experimental strain values.
    mmff_calc : MMFFCalculator
        For computing raw MMFF94 strain of reference compounds.
    """

    # Class-level cache: calibration is deterministic, run once globally
    _factors_cache: Optional[Dict[int, float]] = None
    _interpolator_cache: Optional[interp1d] = None

    def __init__(
        self,
        reference_db: ReferenceDatabase,
        mmff_calc: MMFFCalculator,
    ):
        self.ref_db = reference_db
        self.mmff = mmff_calc
        self.homo_analyzer = HomodesmoticAnalyzer(mmff_calc)

        # Calibration factors: ring_size -> correction factor
        self._factors: Dict[int, float] = {}
        self._interpolator: Optional[interp1d] = None
        self._calibrated = False

    # ------------------------------------------------------------------
    # Build calibration
    # ------------------------------------------------------------------

    def build_calibration(self) -> Dict[int, float]:
        """Compute per-ring-size calibration factors from reference data.

        For each ring size in the reference database, computes the ratio
        experimental_strain / raw_mmff_strain and stores the mean.

        Results are cached at class level — the reference data and MMFF94
        force field are deterministic, so calibration runs only once per
        process regardless of how many StrainAnalyzer instances are created.

        Returns:
            {ring_size: correction_factor}
        """
        if StrainCalibrator._factors_cache is not None:
            self._factors = dict(StrainCalibrator._factors_cache)
            self._interpolator = StrainCalibrator._interpolator_cache
            self._calibrated = True
            return dict(self._factors)

        pairs = self.ref_db.get_calibration_pairs()
        if not pairs:
            raise ValueError("No calibration data available in reference database.")

        # Group by ring size
        size_values: Dict[int, List[Tuple[float, float]]] = {}
        for ref, smiles in pairs:
            try:
                raw = self._compute_raw_strain_for_ref(ref, smiles)
                if raw is None:
                    continue
                for size in set(ref.ring_sizes):
                    size_values.setdefault(size, []).append((raw, ref.strain_energy_kcal_mol))
            except Exception as exc:
                logger.warning(
                    "Failed to compute raw strain for %s: %s", ref.name, exc
                )

        # Compute correction factor per ring size
        for size, pairs_list in size_values.items():
            if size == 6:
                # Cyclohexane is the reference zero — we know its raw MMFF
                # homodesmotic strain should calibrate to ~0
                raw_values = [p[0] for p in pairs_list]
                exp_values = [p[1] for p in pairs_list]
                # Use linear regression forced through ~0 intercept for size 6
                if raw_values and exp_values:
                    mean_raw = np.mean(raw_values)
                    mean_exp = np.mean(exp_values)
                    if mean_raw > 0.01:
                        self._factors[size] = mean_exp / mean_raw
                    else:
                        self._factors[size] = 1.0
                else:
                    self._factors[size] = 1.0
            else:
                raw_values = [p[0] for p in pairs_list]
                exp_values = [p[1] for p in pairs_list]
                if raw_values:
                    ratios = [e / max(r, 0.01) for e, r in zip(exp_values, raw_values)]
                    self._factors[size] = float(np.median(ratios))
                else:
                    self._factors[size] = 1.0

        # Build interpolator for non-standard sizes
        if len(self._factors) >= 2:
            sizes = sorted(self._factors.keys())
            values = [self._factors[s] for s in sizes]
            self._interpolator = interp1d(
                sizes, values,
                kind="linear",
                fill_value=(values[0], values[-1]),
                bounds_error=False,
            )

        self._calibrated = True

        # Persist to class-level cache so future instances skip computation
        StrainCalibrator._factors_cache = dict(self._factors)
        StrainCalibrator._interpolator_cache = self._interpolator

        return dict(self._factors)

    # ------------------------------------------------------------------
    # Apply calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        raw_strain_mmff: float,
        ring_sizes: List[int],
    ) -> float:
        """Convert raw MMFF94 strain to calibrated experimental scale.

        For mono-sized systems: apply ring-size-specific factor.
        For mixed-size systems: apply weighted average factor.

        Args:
            raw_strain_mmff: Raw homodesmotic strain from MMFF94 (kcal/mol).
            ring_sizes: List of ring sizes present in the system.

        Returns:
            Calibrated strain energy in kcal/mol (experimental scale).
        """
        if not self._calibrated:
            self.build_calibration()

        if not ring_sizes:
            return max(raw_strain_mmff, 0.0)

        # Get correction factor (weighted average by 1/ring_size,
        # since smaller rings drive most of the strain)
        weights = [1.0 / max(s, 1) for s in ring_sizes]
        total_w = sum(weights)

        factor = 0.0
        for size, w in zip(ring_sizes, weights):
            f = self.get_correction_factor(size)
            factor += f * w

        factor /= total_w

        calibrated = raw_strain_mmff * factor
        return max(calibrated, 0.0)

    def calibrate_per_ring(
        self,
        raw_per_ring: Dict[int, float],
        ring_sizes: Dict[int, int],
    ) -> Dict[int, float]:
        """Apply ring-size-specific calibration to individual rings.

        Args:
            raw_per_ring: ring_index -> raw MMFF strain.
            ring_sizes: ring_index -> ring size.

        Returns:
            ring_index -> calibrated strain (kcal/mol).
        """
        if not self._calibrated:
            self.build_calibration()

        calibrated = {}
        for idx, raw_strain in raw_per_ring.items():
            size = ring_sizes.get(idx, 6)
            factor = self.get_correction_factor(size)
            calibrated[idx] = max(raw_strain * factor, 0.0)
        return calibrated

    def get_correction_factor(self, ring_size: int) -> float:
        """Get the calibration factor for a specific ring size.

        Uses interpolation for sizes not in the reference set.
        """
        if not self._calibrated:
            self.build_calibration()

        if ring_size in self._factors:
            return self._factors[ring_size]

        if self._interpolator is not None:
            return float(self._interpolator(ring_size))

        # Fallback: use size 6 factor as default
        return self._factors.get(6, 1.0)

    def estimate_uncertainty(self, ring_sizes: List[int]) -> float:
        """Estimate calibration uncertainty based on ring composition.

        Returns estimated uncertainty in kcal/mol.
        """
        if not self._calibrated:
            self.build_calibration()

        # More uncertainty for exotic ring sizes and mixed systems
        uncertainty = 1.0  # Base uncertainty

        for size in ring_sizes:
            if size not in self._factors:
                uncertainty += 2.0  # No reference data for this size
            elif size <= 4:
                uncertainty += 1.5  # MMFF94 is less accurate for small rings
            elif size >= 9:
                uncertainty += 1.0  # Medium/large rings have less reference data

        uncertainty = min(uncertainty, 5.0)
        return uncertainty

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_raw_strain_for_ref(
        self,
        ref: ReferenceCompound,
        smiles: str,
    ) -> Optional[float]:
        """Compute raw MMFF94 homodesmotic strain for a reference compound."""
        from rdkit import Chem
        from .ring_analysis import RingAnalyzer

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # For simple cycloalkanes, use the strict homodesmotic method
        if (
            ref.ring_count == 1
            and not any(c not in "C" for c in smiles if c.isalpha())
            and len(ref.ring_sizes) == 1
            and ref.ring_sizes[0] <= 8
        ):
            return self.homo_analyzer.compute_cycloalkane_strain(ref.ring_sizes[0])

        # For other compounds, use the generic ring-opening method
        analyzer = RingAnalyzer(mol)
        ring_info = analyzer.identify_all_rings()
        raw_strain, _, _ = self.homo_analyzer.compute_total_strain(mol, ring_info)
        return raw_strain

    @property
    def factors(self) -> Dict[int, float]:
        if not self._calibrated:
            self.build_calibration()
        return dict(self._factors)
