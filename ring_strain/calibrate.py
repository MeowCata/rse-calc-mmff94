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

    Computes per-ring-size correction factors from reference compounds
    and applies them to monocyclic saturated carbocycles.

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
        self._factors: Dict[int, float] = {}
        self._interpolator: Optional[interp1d] = None
        self._calibrated = False

    def build_calibration(self) -> Dict[int, float]:
        """Compute per-ring-size calibration factors from reference data.

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

        # Persist to class-level cache
        StrainCalibrator._factors_cache = dict(self._factors)
        StrainCalibrator._interpolator_cache = self._interpolator

        return dict(self._factors)

    def calibrate(self, raw_strain_mmff: float, ring_size: int) -> float:
        """Convert raw MMFF94 strain to calibrated experimental scale.

        Args:
            raw_strain_mmff: Raw homodesmotic strain from MMFF94 (kcal/mol).
            ring_size: Ring size for correction factor lookup.

        Returns:
            Calibrated strain energy in kcal/mol (experimental scale).
        """
        if not self._calibrated:
            self.build_calibration()

        factor = self.get_correction_factor(ring_size)
        calibrated = raw_strain_mmff * factor
        return max(calibrated, 0.0)

    def get_correction_factor(self, ring_size: int) -> float:
        """Get the calibration factor for a specific ring size."""
        if not self._calibrated:
            self.build_calibration()

        if ring_size in self._factors:
            return self._factors[ring_size]

        if self._interpolator is not None:
            return float(self._interpolator(ring_size))

        return self._factors.get(6, 1.0)

    def estimate_uncertainty(self, ring_size: int) -> float:
        """Estimate calibration uncertainty for a given ring size (kcal/mol)."""
        if not self._calibrated:
            self.build_calibration()

        uncertainty = 1.0
        if ring_size not in self._factors:
            uncertainty += 2.0
        elif ring_size <= 4:
            uncertainty += 1.5
        elif ring_size >= 9:
            uncertainty += 1.0

        return min(uncertainty, 5.0)

    def _compute_raw_strain_for_ref(
        self,
        ref: ReferenceCompound,
        smiles: str,
    ) -> Optional[float]:
        """Compute raw MMFF94 homodesmotic strain for a reference compound.

        Returns ``None`` for substituted references (heavy-atom count >
        sum of ring sizes), which must NOT contribute to the per-ring-size
        calibration regression: the strict-bond-balanced cycloalkane raw
        strain is the same for all 6-rings, but experimental strain
        differs for substituted variants — mixing them would pin the
        factor at ``exp_substituted / 0`` (catastrophic).

        Substituted references still appear in ``ref_db.get_by_smiles``
        for user-facing comparison; they just don't feed the regression.
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Substitution test: substituted compounds have heavy atoms beyond
        # the ring. Single-ring assumption is valid here (calibration set
        # is monocyclic).
        if ref.ring_sizes:
            ring_total = sum(ref.ring_sizes)
            if mol.GetNumHeavyAtoms() > ring_total:
                return None

        if ref.ring_sizes and ref.ring_sizes[0] <= 8:
            return self.homo_analyzer.compute_cycloalkane_strain(ref.ring_sizes[0])

        return None

    @property
    def factors(self) -> Dict[int, float]:
        if not self._calibrated:
            self.build_calibration()
        return dict(self._factors)
