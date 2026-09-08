"""
Ring strain energy quantification using MMFF94 force field.

Main API
--------
StrainAnalyzer  - Primary entry point. Call analyze(smiles) to get a StrainReport.
StrainReport    - Dataclass with all strain analysis results.

Modules
-------
core           - Orchestrator, reports, and structured progress events
mmff           - MMFF94 wrapper (embedding, optimization, energy)
ring_analysis  - Ring detection and classification
homodesmotic   - Homodesmotic reaction construction
calibrate      - MMFF94 to experimental scale calibration
scoring        - 0-100 stability score
reference      - Known reference compound database
geometry       - Geometric strain decomposition
"""

from .core import ProgressUpdate, StrainAnalyzer, StrainReport
from .reference import ReferenceDatabase, ReferenceCompound
from .mmff import MMFFCalculator, MMFFResult
from .scoring import StabilityScorer, StabilityCategory

__all__ = [
    "StrainAnalyzer",
    "StrainReport",
    "ProgressUpdate",
    "ReferenceDatabase",
    "ReferenceCompound",
    "MMFFCalculator",
    "MMFFResult",
    "StabilityScorer",
    "StabilityCategory",
]
