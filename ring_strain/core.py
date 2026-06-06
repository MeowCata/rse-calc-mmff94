"""
Main orchestrator for ring strain energy quantification.

Provides the primary user-facing API (StrainAnalyzer.analyze()) and
the StrainReport dataclass that holds all computation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger

from .mmff import MMFFCalculator
from .ring_analysis import RingAnalyzer, RingSystemInfo
from .reference import ReferenceCompound, ReferenceDatabase
from .homodesmotic import HomodesmoticAnalyzer
from .calibrate import StrainCalibrator
from .scoring import StabilityScorer

import logging

logger = logging.getLogger(__name__)

# Suppress RDKit warnings during normal operation
RDLogger.logger().setLevel(RDLogger.ERROR)


# ---------------------------------------------------------------------------
# Strain report
# ---------------------------------------------------------------------------

@dataclass
class StrainReport:
    """Complete ring strain analysis result for a molecule."""

    # Input
    smiles: str
    canonical_smiles: str
    formula: str
    molecular_weight: float
    num_heavy_atoms: int

    # Ring system overview
    num_rings: int
    ring_system_type: str
    ring_sizes: List[int]

    # Strain energies (kcal/mol)
    total_strain_mmff_kcal_mol: float
    total_strain_calibrated_kcal_mol: float
    strain_per_heavy_atom_kcal_mol: float
    calibration_uncertainty: float

    # Stability score
    stability_score: float
    stability_category: str
    per_ring_scores: Dict[int, float]

    # MMFF94 diagnostics (required, no defaults)
    mmff_coverage: float
    optimization_converged: bool
    conformers_sampled: int

    # Per-ring details (defaulted)
    per_ring_details: List[Dict] = field(default_factory=list)

    # Comparison with known reference (if available, defaulted)
    reference_match: Optional[Dict] = None

    # Methodology info (defaulted)
    method: str = "MMFF94 homodesmotic + calibration"
    threshold_kcal_mol: float = 8.0

    # Errors/warnings (defaulted)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        import json as _json
        return _json.dumps(self.to_dict(), indent=indent, default=str)

    def print_summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Ring Strain Analysis: {self.smiles}",
            "=" * 60,
            f"  Formula:           {self.formula}",
            f"  Molecular weight:  {self.molecular_weight:.2f} g/mol",
            f"  Heavy atoms:       {self.num_heavy_atoms}",
            f"  Number of rings:   {self.num_rings}",
            f"  Ring system type:  {self.ring_system_type}",
            f"  Ring sizes:        {self.ring_sizes}",
            "",
            f"  Raw MMFF94 strain:       {self.total_strain_mmff_kcal_mol:+.2f} kcal/mol",
            f"  Calibrated strain:       {self.total_strain_calibrated_kcal_mol:+.2f} kcal/mol",
            f"  Strain per heavy atom:   {self.strain_per_heavy_atom_kcal_mol:+.3f} kcal/mol",
            f"  Uncertainty:             +/- {self.calibration_uncertainty:.1f} kcal/mol",
            "",
            f"  Strain score:      {self.stability_score:.1f} / 100  ({self.stability_category})",
        ]

        if self.per_ring_details:
            lines.append("")
            lines.append("  Per-ring breakdown:")
            lines.append("  " + "-" * 40)
            for ring in self.per_ring_details:
                lines.append(
                    f"    Ring {ring['ring_index']} ({ring['size']}-membered {ring['type']}): "
                    f"{ring['strain_calibrated']:+.2f} kcal/mol, "
                    f"score={ring['score']:.1f}"
                )

        if self.reference_match:
            lines.append("")
            lines.append(f"  Reference match: {self.reference_match.get('name', 'N/A')}")
            lines.append(
                f"    Computed:  {self.total_strain_calibrated_kcal_mol:.2f} kcal/mol"
            )
            lines.append(
                f"    Reference: {self.reference_match.get('exp_strain', 'N/A')} kcal/mol"
            )

        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.print_summary()


# ---------------------------------------------------------------------------
# Strain Analyzer (main entry point)
# ---------------------------------------------------------------------------

class StrainAnalyzer:
    """Main entry point for ring strain energy quantification.

    Usage::

        analyzer = StrainAnalyzer()
        report = analyzer.analyze("C1CC1")  # cyclopropane
        print(report)

    Parameters
    ----------
    n_conformers : int
        Number of conformers for MMFF94 global minimum search. Default 200.
        Increase for flexible molecules, decrease for rigid rings.
    random_seed : int
        Seed for reproducible conformer generation.
    stability_threshold : float
        Threshold for stability score calculation (kcal/mol). Default 8.0.
        Lower = more discriminating for strained compounds.
    """

    def __init__(
        self,
        n_conformers: int = 200,
        random_seed: int = 42,
        stability_threshold: float = 8.0,
    ):
        self.mmff_calc = MMFFCalculator(
            n_conformers=n_conformers,
            random_seed=random_seed,
        )
        self.ref_db = ReferenceDatabase()
        self.calibrator = StrainCalibrator(self.ref_db, self.mmff_calc)
        self.scorer = StabilityScorer(threshold_kcal_mol=stability_threshold)
        self.homo_analyzer = HomodesmoticAnalyzer(self.mmff_calc)

        # Pre-build calibration
        self.calibrator.build_calibration()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def analyze(self, smiles: str) -> StrainReport:
        """Analyze ring strain for a molecule given its SMILES.

        Full pipeline:
        1. Parse SMILES and validate
        2. Detect and classify all rings
        3. Compute raw MMFF94 homodesmotic strain
        4. Calibrate to experimental scale
        5. Compute stability scores
        6. Compare with reference data if available

        Args:
            smiles: Input SMILES string.

        Returns:
            StrainReport with full analysis results.

        Raises:
            ValueError: Invalid SMILES or no rings detected.
            RuntimeError: MMFF94 computation failed.
        """
        # --- Step 1: Parse ---
        if not smiles or not smiles.strip():
            raise ValueError(f"Invalid SMILES string: {smiles!r}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles!r}")

        warnings: List[str] = []

        # --- Step 2: Ring analysis ---
        ring_analyzer = RingAnalyzer(mol)
        if not ring_analyzer.has_rings():
            return self._no_ring_report(smiles, mol)

        ring_info = ring_analyzer.identify_all_rings()

        # --- Step 3: Raw MMFF94 strain ---
        try:
            raw_strain, raw_per_ring, components = self.homo_analyzer.compute_total_strain(
                mol, ring_info
            )
        except Exception as exc:
            raise RuntimeError(
                f"MMFF94 computation failed for {smiles!r}: {exc}"
            ) from exc

        # --- Step 4: Calibrate ---
        ring_sizes = [r.size for r in ring_info.rings]

        # Polycyclic systems (fused, bridged, spiro, cage) should NOT use
        # per-ring-size calibration factors derived from monocyclic compounds.
        # The ring-opening method already produces values closer to experimental
        # scale for these systems (e.g., norbornane raw=14.3 vs exp=15.0).
        if ring_info.has_polycyclic:
            calibrated = max(raw_strain, 0.0)
            calibrated_per_ring = {
                idx: max(v, 0.0) for idx, v in raw_per_ring.items()
            }
            uncertainty = self.calibrator.estimate_uncertainty(ring_sizes) + 1.0
        else:
            calibrated = self.calibrator.calibrate(raw_strain, ring_sizes)
            uncertainty = self.calibrator.estimate_uncertainty(ring_sizes)
            # Per-ring calibration
            ring_size_map = {r.ring_index: r.size for r in ring_info.rings}
            calibrated_per_ring = self.calibrator.calibrate_per_ring(
                raw_per_ring, ring_size_map
            )

        # --- Step 5: Stability score ---
        score = self.scorer.compute_score(calibrated)
        per_ring_scores = self.scorer.compute_per_ring_scores(calibrated_per_ring)
        category = self.scorer.categorize(score)

        # --- Step 6: Reference comparison ---
        ref_match = self._find_reference_match(mol, calibrated, ring_info)

        # --- Diagnostics ---
        mmff_coverage = MMFFCalculator.check_mmff_coverage(mol)
        if mmff_coverage < 0.9:
            warnings.append(
                f"Low MMFF94 parameter coverage ({mmff_coverage:.0%}). "
                f"Results may be less accurate for atoms without MMFF94 params."
            )

        # Build per-ring details
        per_ring_details = []
        for ring in ring_info.rings:
            per_ring_details.append({
                "ring_index": ring.ring_index,
                "size": ring.size,
                "atoms": list(ring.atom_indices),
                "type": ring.fusion_type,
                "is_aromatic": ring.is_aromatic,
                "is_heterocyclic": ring.is_heterocyclic,
                "heteroatoms": ring.heteroatom_symbols,
                "strain_raw_mmff": round(raw_per_ring.get(ring.ring_index, 0.0), 3),
                "strain_calibrated": round(
                    calibrated_per_ring.get(ring.ring_index, 0.0), 3
                ),
                "score": round(per_ring_scores.get(ring.ring_index, 100.0), 1),
                "category": self.scorer.categorize(
                    per_ring_scores.get(ring.ring_index, 100.0)
                ),
            })

        # System type
        system_type = _describe_ring_system(ring_info)

        # Formula and MW
        formula = CalcMolFormula(mol)
        mw = MolWt(mol)

        return StrainReport(
            smiles=smiles,
            canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
            formula=formula,
            molecular_weight=mw,
            num_heavy_atoms=mol.GetNumHeavyAtoms(),
            num_rings=ring_info.num_rings,
            ring_system_type=system_type,
            ring_sizes=ring_sizes,
            total_strain_mmff_kcal_mol=round(raw_strain, 3),
            total_strain_calibrated_kcal_mol=round(calibrated, 3),
            strain_per_heavy_atom_kcal_mol=round(
                calibrated / max(mol.GetNumHeavyAtoms(), 1), 3
            ),
            calibration_uncertainty=round(uncertainty, 1),
            stability_score=round(score, 1),
            stability_category=category,
            per_ring_scores={
                k: round(v, 1) for k, v in per_ring_scores.items()
            },
            per_ring_details=per_ring_details,
            reference_match=ref_match,
            mmff_coverage=mmff_coverage,
            optimization_converged=True,
            conformers_sampled=self.mmff_calc.n_conformers,
            threshold_kcal_mol=self.scorer.threshold,
            warnings=warnings,
        )

    def analyze_batch(
        self,
        smiles_list: List[str],
        show_progress: bool = True,
    ) -> List[StrainReport]:
        """Batch analysis of multiple SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to analyze.
        show_progress : bool
            Print progress during batch processing.

        Returns:
            List of StrainReport, one per SMILES.
        """
        results = []
        for i, smi in enumerate(smiles_list):
            if show_progress:
                print(f"[{i+1}/{len(smiles_list)}] {smi}")
            try:
                report = self.analyze(smi)
            except Exception as exc:
                report = StrainReport(
                    smiles=smi,
                    canonical_smiles=smi,
                    formula="N/A",
                    molecular_weight=0.0,
                    num_heavy_atoms=0,
                    num_rings=0,
                    ring_system_type="error",
                    ring_sizes=[],
                    total_strain_mmff_kcal_mol=float("nan"),
                    total_strain_calibrated_kcal_mol=float("nan"),
                    strain_per_heavy_atom_kcal_mol=float("nan"),
                    calibration_uncertainty=0.0,
                    stability_score=0.0,
                    stability_category="error",
                    per_ring_scores={},
                    mmff_coverage=0.0,
                    optimization_converged=False,
                    conformers_sampled=0,
                    warnings=[str(exc)],
                )
            results.append(report)
        return results

    def compare(self, smiles_a: str, smiles_b: str) -> Tuple[StrainReport, StrainReport, str]:
        """Compare ring strain between two molecules."""
        report_a = self.analyze(smiles_a)
        report_b = self.analyze(smiles_b)

        diff = report_b.total_strain_calibrated_kcal_mol - report_a.total_strain_calibrated_kcal_mol
        if abs(diff) < 0.1:
            verdict = f"{smiles_a} and {smiles_b} have similar ring strain."
        elif diff > 0:
            verdict = f"{smiles_b} is {diff:.1f} kcal/mol MORE strained than {smiles_a}."
        else:
            verdict = f"{smiles_b} is {abs(diff):.1f} kcal/mol LESS strained than {smiles_a}."

        return report_a, report_b, verdict

    def get_reference_value(self, smiles: str) -> Optional[float]:
        """Look up known experimental strain value for a compound."""
        ref = self.ref_db.get_by_smiles(smiles)
        return ref.strain_energy_kcal_mol if ref else None

    def list_references(self) -> List[Dict]:
        """List all reference compounds in the database."""
        return [
            {
                "name": r.name,
                "smiles": r.smiles,
                "strain_kcal_mol": r.strain_energy_kcal_mol,
                "ring_sizes": r.ring_sizes,
                "ring_count": r.ring_count,
            }
            for r in self.ref_db.get_all()
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _no_ring_report(self, smiles: str, mol: Mol) -> StrainReport:
        """Generate a report for molecules without rings."""
        formula = CalcMolFormula(mol)
        mw = MolWt(mol)
        return StrainReport(
            smiles=smiles,
            canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
            formula=formula,
            molecular_weight=mw,
            num_heavy_atoms=mol.GetNumHeavyAtoms(),
            num_rings=0,
            ring_system_type="acyclic",
            ring_sizes=[],
            total_strain_mmff_kcal_mol=0.0,
            total_strain_calibrated_kcal_mol=0.0,
            strain_per_heavy_atom_kcal_mol=0.0,
            calibration_uncertainty=0.0,
            stability_score=100.0,
            stability_category="strain-free (acyclic)",
            per_ring_scores={},
            mmff_coverage=1.0,
            optimization_converged=True,
            conformers_sampled=0,
            warnings=["Molecule has no rings — strain is zero by definition."],
        )

    def _find_reference_match(
        self,
        mol: Mol,
        calibrated_strain: float,
        ring_info: RingSystemInfo,
    ) -> Optional[Dict]:
        """Try to match the molecule against known reference compounds."""
        can_smiles = Chem.MolToSmiles(mol, canonical=True)
        ref = self.ref_db.get_by_smiles(can_smiles)

        if ref is not None:
            error = abs(calibrated_strain - ref.strain_energy_kcal_mol)
            return {
                "name": ref.name,
                "smiles": ref.smiles,
                "exp_strain": ref.strain_energy_kcal_mol,
                "computed_strain": calibrated_strain,
                "absolute_error": round(error, 2),
                "source": ref.source,
            }

        # Try matching by ring sizes
        sizes = tuple(sorted(r.size for r in ring_info.rings))
        for candidate in self.ref_db.get_all():
            c_sizes = tuple(sorted(candidate.ring_sizes))
            if sizes == c_sizes:
                return {
                    "name": candidate.name,
                    "smiles": candidate.smiles,
                    "exp_strain": candidate.strain_energy_kcal_mol,
                    "computed_strain": calibrated_strain,
                    "absolute_error": round(
                        abs(calibrated_strain - candidate.strain_energy_kcal_mol), 2
                    ),
                    "source": f"matched by ring sizes {sizes}; exact SMILES may differ",
                    "approximate_match": True,
                }

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _describe_ring_system(ring_info: RingSystemInfo) -> str:
    """Generate a human-readable description of the ring system."""
    if ring_info.num_rings == 0:
        return "acyclic"
    if ring_info.num_rings == 1:
        size = ring_info.rings[0].size
        aro = "aromatic " if ring_info.rings[0].is_aromatic else ""
        het = "heterocyclic " if ring_info.rings[0].is_heterocyclic else ""
        return f"{aro}{het}monocyclic ({size}-membered)"
    if ring_info.has_polycyclic:
        types = set(ring_info.ring_system_types)
        if "cage" in types:
            return f"polycyclic cage ({ring_info.num_rings} rings)"
        if "bridged" in types:
            return f"bridged polycyclic ({ring_info.num_rings} rings)"
        if "spiro" in types:
            return f"spiro polycyclic ({ring_info.num_rings} rings)"
        return f"fused polycyclic ({ring_info.num_rings} rings)"
    return f"polycyclic ({ring_info.num_rings} rings)"
