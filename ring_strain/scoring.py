"""
Strain scoring — converts calibrated ring strain energy into an
intuitive 0-100 score where 100 = strain-free and 0 = extreme strain.

The scoring function uses exponential decay so that low-strain rings
cluster near 100 while highly-strained rings score near 0.

    score = 100 * exp(-strain / threshold)

With threshold = 8.0 kcal/mol:
  - cyclohexane (~0)  → 100   (strain-free)
  - cyclopentane (6.5)→  44   (moderate strain)
  - cyclobutane (26.3)→   3.7 (significant strain)
  - cyclopropane(27.5)→   3.2 (highly strained)
  - cubane (166 total) →  ~0  (extreme strain)
"""

from enum import Enum
from typing import Dict, Tuple

import numpy as np


class StabilityCategory(Enum):
    HIGHLY_STRAINED = "highly strained"              # 0 - 10
    SIGNIFICANT_STRAIN = "significant strain"        # 10 - 30
    MODERATE_STRAIN = "moderate strain"              # 30 - 60
    LOW_STRAIN = "low strain"                        # 60 - 90
    STRAIN_FREE = "strain-free"                      # 90 - 100


# ---------------------------------------------------------------------------
# Default threshold: maps cyclohexane (~0) to ~100 and cyclobutane (~26) to ~3.7
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLD: float = 8.0


class StabilityScorer:
    """Compute stability scores (0-100) and qualitative categories.

    Parameters
    ----------
    threshold_kcal_mol : float
        Characteristic strain energy scale. Lower values make the scoring
        more discriminating for moderately strained compounds.
        Default is 8.0 kcal/mol.
    """

    def __init__(self, threshold_kcal_mol: float = _DEFAULT_THRESHOLD):
        if threshold_kcal_mol <= 0:
            raise ValueError("Threshold must be positive.")
        self.threshold = threshold_kcal_mol

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def compute_score(self, strain_kcal_mol: float) -> float:
        """Compute stability score from calibrated strain energy.

        Args:
            strain_kcal_mol: Calibrated strain (kcal/mol, experimental scale).

        Returns:
            Stability score (0-100), where 100 = strain-free, 0 = extreme strain.
        """
        if strain_kcal_mol < 0:
            strain_kcal_mol = 0.0
        score = 100.0 * np.exp(-strain_kcal_mol / self.threshold)
        # Clamp to [0, 100]
        return float(np.clip(score, 0.0, 100.0))

    def compute_per_ring_scores(
        self,
        per_ring_strain: Dict[int, float],
    ) -> Dict[int, float]:
        """Compute individual stability scores for each ring.

        Args:
            per_ring_strain: ring_index -> calibrated strain (kcal/mol).

        Returns:
            ring_index -> stability score (0-100).
        """
        return {
            idx: self.compute_score(strain)
            for idx, strain in per_ring_strain.items()
        }

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    def categorize(self, score: float) -> str:
        """Map numerical score to qualitative strain category."""
        if score < 10:
            return StabilityCategory.HIGHLY_STRAINED.value
        if score < 30:
            return StabilityCategory.SIGNIFICANT_STRAIN.value
        if score < 60:
            return StabilityCategory.MODERATE_STRAIN.value
        if score < 90:
            return StabilityCategory.LOW_STRAIN.value
        return StabilityCategory.STRAIN_FREE.value

    def get_interpretation(
        self,
        strain_kcal_mol: float,
        score: float,
        ring_sizes: Tuple[int, ...] = (),
    ) -> str:
        """Generate human-readable interpretation of the result."""
        cat = self.categorize(score)
        size_str = ", ".join(str(s) for s in ring_sizes) if ring_sizes else "ring"

        lines = [
            f"Ring size(s): {size_str}",
            f"Ring strain energy: {strain_kcal_mol:.2f} kcal/mol",
            f"Strain score: {score:.1f} / 100  ({cat})",
        ]

        if score >= 90:
            lines.append(
                "This ring system is essentially strain-free. The bond angles "
                "and torsions are near their ideal values."
            )
        elif score >= 60:
            lines.append(
                "Low ring strain. The ring geometry is slightly distorted from "
                "ideal bond angles and torsional preferences."
            )
        elif score >= 30:
            lines.append(
                "Moderate ring strain. The ring has noticeable geometric "
                "distortion from ideal values."
            )
        elif score >= 10:
            lines.append(
                "Significant ring strain. The ring is substantially distorted "
                "by geometric constraints. This strain is a meaningful "
                "contributor to the molecule's potential energy."
            )
        else:
            lines.append(
                "High ring strain. The ring geometry is severely distorted "
                "from ideal bond angles and torsions. The strain energy is "
                "a dominant factor in the molecule's energetics."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_threshold(self, threshold_kcal_mol: float) -> None:
        """Adjust the sensitivity of the scoring function."""
        if threshold_kcal_mol <= 0:
            raise ValueError("Threshold must be positive.")
        self.threshold = threshold_kcal_mol
