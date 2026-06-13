"""
Calibration of raw MMFF94 homodesmotic strain to experimental scale.

MMFF94 is parameterized for conformational energetics, not heats of
formation. Its raw homodesmotic strain values deviate systematically
from experimental ring strain energies, particularly for small rings
and substituted rings.

Calibration model: ``calibrated = a[ring_size] * raw + b[ring_size]``.
Per-size coefficients are derived by linear regression on the reference
set (both unsubstituted and substituted). When a ring size has only
one reference (so regression is under-determined), we fall back to the
multiplicative form ``raw * a`` with ``b = 0``.

Coefficients are persisted to ``calibration_coefficients.json`` next to
this file so user-facing init is instant; rerun
``scripts/derive_calibration.py`` after touching the reference set to
regenerate them.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .reference import ReferenceCompound, ReferenceDatabase
from .mmff import MMFFCalculator
from .homodesmotic import HomodesmoticAnalyzer

import logging

logger = logging.getLogger(__name__)

# Path to persisted calibration coefficients, relative to this module.
_COEFFS_FILE = Path(__file__).parent / "calibration_coefficients.json"


def _fit_size_coefficients(
    pairs: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Fit ``(a, b)`` such that ``a * raw + b ≈ exp`` for the given pairs.

    Strategies by number of points:
      0: ``(1.0, 0.0)`` — pass-through.
      1: multiplicative ``(exp / max(raw, 0.01), 0.0)`` — preserves the
         historical behaviour when only one reference is available.
      >=2 with an anchor (``exp ≈ 0`` for the unsubstituted ring): constrained
         linear fit forced through ``(raw_anchor, 0)``. This is critical for
         ring sizes whose unsubstituted reference defines the "strain-free"
         endpoint (most prominently cyclohexane = 0) — an unconstrained
         lstsq fit would otherwise predict non-zero strain for the
         unsubstituted ring itself.
      >=2 without an anchor: unconstrained ``numpy.linalg.lstsq``.
    """
    if not pairs:
        return 1.0, 0.0
    if len(pairs) == 1:
        raw, exp = pairs[0]
        return exp / max(raw, 0.01), 0.0

    anchor_raws = [r for r, e in pairs if abs(e) < 0.01]
    if anchor_raws:
        # Constrained fit y = a * (x - anchor): one-parameter regression
        # through the origin in shifted coordinates. Forces predicted
        # strain to be exactly 0 at the anchor's raw value.
        x_anchor = float(np.mean(anchor_raws))
        non_anchor = [(r, e) for r, e in pairs if abs(e) >= 0.01]
        if not non_anchor:
            return 0.0, 0.0
        xs = np.array([r - x_anchor for r, _ in non_anchor], dtype=float)
        ys = np.array([e for _, e in non_anchor], dtype=float)
        denom = float(np.sum(xs * xs))
        if denom < 1e-12:
            return 0.0, 0.0
        a = float(np.sum(xs * ys) / denom)
        b = -a * x_anchor
        return a, b

    raws = np.array([p[0] for p in pairs], dtype=float)
    exps = np.array([p[1] for p in pairs], dtype=float)
    A = np.column_stack([raws, np.ones_like(raws)])
    coeffs, *_ = np.linalg.lstsq(A, exps, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


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

    # Class-level caches survive across analyzer instances in the same
    # process. ``_coeffs_cache`` is the new ``{size: (a, b)}`` map;
    # ``_factors_cache`` is kept for backwards compatibility (callers that
    # read it get the slope ``a``).
    _coeffs_cache: Optional[Dict[int, Tuple[float, float]]] = None
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
        # ``_coeffs[size] = (a, b)`` is the authoritative storage; ``_factors``
        # mirrors ``a`` for backwards compatibility with callers that read it
        # via ``get_correction_factor``.
        self._coeffs: Dict[int, Tuple[float, float]] = {}
        self._factors: Dict[int, float] = {}
        self._interpolator: Optional[interp1d] = None
        self._calibrated = False

    def build_calibration(self) -> Dict[int, float]:
        """Build per-size calibration ``(a, b)`` coefficients from the reference set.

        Order of preference:
          1. Class-level cache from a previous call in this process.
          2. Persisted JSON at ``calibration_coefficients.json`` (instant load).
          3. Live derivation from the unsubstituted-cycloalkane cached raw
             values only — this matches the historical behaviour and is
             fast (~1 s) but produces only a multiplicative factor per
             size. Substituted references aren't run by this path because
             the full pipeline costs ~10-15 min; run
             ``scripts/derive_calibration.py`` once to populate the JSON.

        Returns ``{size: a}`` for backwards compatibility with older callers.
        """
        if StrainCalibrator._coeffs_cache is not None:
            self._coeffs = dict(StrainCalibrator._coeffs_cache)
            self._factors = dict(StrainCalibrator._factors_cache or {})
            self._interpolator = StrainCalibrator._interpolator_cache
            self._calibrated = True
            return dict(self._factors)

        # Tier 2: load from JSON if available.
        if _COEFFS_FILE.exists():
            try:
                self._load_coeffs_from_json(_COEFFS_FILE)
                self._calibrated = True
                StrainCalibrator._coeffs_cache = dict(self._coeffs)
                StrainCalibrator._factors_cache = dict(self._factors)
                StrainCalibrator._interpolator_cache = self._interpolator
                logger.debug(
                    "Loaded calibration from %s (%d sizes).",
                    _COEFFS_FILE, len(self._coeffs),
                )
                return dict(self._factors)
            except Exception as exc:
                logger.warning(
                    "Failed to load %s: %s. Falling back to live derivation.",
                    _COEFFS_FILE, exc,
                )

        # Tier 3: live derivation from unsubstituted refs only.
        pairs = self.ref_db.get_calibration_pairs()
        if not pairs:
            raise ValueError("No calibration data available in reference database.")

        size_values: Dict[int, List[Tuple[float, float]]] = {}
        for ref, smiles in pairs:
            try:
                raw = self._compute_raw_strain_for_ref(ref, smiles)
                if raw is None:
                    continue
                for size in set(ref.ring_sizes):
                    size_values.setdefault(size, []).append(
                        (raw, ref.strain_energy_kcal_mol),
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to compute raw strain for %s: %s", ref.name, exc,
                )

        for size, pts in size_values.items():
            self._coeffs[size] = _fit_size_coefficients(pts)
            self._factors[size] = self._coeffs[size][0]

        self._build_interpolator()
        self._calibrated = True

        StrainCalibrator._coeffs_cache = dict(self._coeffs)
        StrainCalibrator._factors_cache = dict(self._factors)
        StrainCalibrator._interpolator_cache = self._interpolator

        return dict(self._factors)

    # ------------------------------------------------------------------
    # Coefficient persistence
    # ------------------------------------------------------------------

    def _load_coeffs_from_json(self, path: Path) -> None:
        """Populate ``_coeffs``, ``_factors``, and the interpolator from JSON."""
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        size_coeffs = data.get("size_coeffs", {})
        for size_str, entry in size_coeffs.items():
            size = int(size_str)
            a = float(entry["a"])
            b = float(entry.get("b", 0.0))
            self._coeffs[size] = (a, b)
            self._factors[size] = a
        self._build_interpolator()

    def _build_interpolator(self) -> None:
        if len(self._factors) >= 2:
            sizes = sorted(self._factors.keys())
            values = [self._factors[s] for s in sizes]
            self._interpolator = interp1d(
                sizes, values,
                kind="linear",
                fill_value=(values[0], values[-1]),
                bounds_error=False,
            )

    def calibrate(self, raw_strain_mmff: float, ring_size: int) -> float:
        """Convert raw MMFF94 strain to calibrated experimental scale.

        Applies ``a[size] * raw + b[size]`` and clamps to ``>= 0``. Falls
        back to interpolated slope (intercept = 0) for ring sizes outside
        the reference set.
        """
        if not self._calibrated:
            self.build_calibration()

        a, b = self._get_coeffs(ring_size)
        return max(a * raw_strain_mmff + b, 0.0)

    def _get_coeffs(self, ring_size: int) -> Tuple[float, float]:
        """Return ``(a, b)`` for ``ring_size``, interpolating if needed."""
        if ring_size in self._coeffs:
            return self._coeffs[ring_size]
        if self._interpolator is not None:
            return float(self._interpolator(ring_size)), 0.0
        return self._factors.get(6, 1.0), 0.0

    def get_correction_factor(self, ring_size: int) -> float:
        """Get the calibration slope ``a`` for a specific ring size.

        Kept for backwards compatibility — older callers read this as the
        multiplicative factor. The full calibration also adds an intercept;
        use ``calibrate`` to get the corrected value.
        """
        if not self._calibrated:
            self.build_calibration()
        return self._get_coeffs(ring_size)[0]

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
