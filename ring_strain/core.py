"""
Main orchestrator for ring strain energy quantification.

Provides the primary user-facing API (StrainAnalyzer.analyze()) and
the StrainReport dataclass that holds all computation results.

Only monocyclic saturated carbocycles are supported. Polycyclic, bridged,
spiro, heterocyclic, aromatic, and unsaturated systems are rejected with
a "Not Supported" message.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple  # noqa: F401

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDLogger

from .mmff import MMFFCalculator
from .ring_analysis import RingAnalyzer, RingInfo
from .reference import ReferenceCompound, ReferenceDatabase
from .homodesmotic import HomodesmoticAnalyzer
from .calibrate import StrainCalibrator
from .scoring import StabilityScorer
from .geometry import GeometryAnalyzer

import logging

logger = logging.getLogger(__name__)

# Suppress RDKit warnings during normal operation
RDLogger.logger().setLevel(RDLogger.ERROR)


def _round_or_none(value: Optional[float], digits: int = 3) -> Optional[float]:
    """Round a float for report output, propagating None and NaN as None."""
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Strain report
# ---------------------------------------------------------------------------

@dataclass
class StrainReport:
    """Complete ring strain analysis result for a monocyclic saturated carbocycle."""

    # Input
    smiles: str
    canonical_smiles: str
    formula: str
    molecular_weight: float
    num_heavy_atoms: int

    # Ring info
    ring_size: int
    is_substituted: bool

    # Strain energies (kcal/mol)
    total_strain_mmff_kcal_mol: float
    total_strain_calibrated_kcal_mol: float
    strain_per_heavy_atom_kcal_mol: float
    calibration_uncertainty: float

    # Stability score
    stability_score: float
    stability_category: str

    # MMFF94 diagnostics
    mmff_coverage: float
    optimization_converged: bool
    conformers_sampled: int
    monte_carlo_used: bool

    # Comparison with known reference (if available)
    reference_match: Optional[Dict] = None

    # Methodology
    method: str = "MMFF94 homodesmotic + calibration"
    threshold_kcal_mol: float = 8.0

    # Validation
    is_supported: bool = True
    validation_message: str = ""

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Geometry breakdown (Baeyer, Pitzer, transannular)
    # ------------------------------------------------------------------
    geometry_breakdown: Optional[Dict] = None
    baeyer_strain_rms_deg: Optional[float] = None
    pitzer_n_eclipsed: Optional[int] = None
    transannular_contacts: Optional[List[Dict]] = None
    steric_confinement_kcal_mol: Optional[float] = None

    # ------------------------------------------------------------------
    # MMFF94 per-term energy decomposition (cyclic - acyclic).
    # vdW is the physically meaningful "true steric" strain; torsion is
    # Pitzer-like; angle is Baeyer-like; bond is stretch. Diagnostic
    # snapshots — not guaranteed to sum to total_strain_mmff_kcal_mol when
    # Boltzmann averaging is in effect.
    # ------------------------------------------------------------------
    vdw_strain_kcal_mol: Optional[float] = None
    torsion_strain_kcal_mol: Optional[float] = None
    angle_strain_kcal_mol: Optional[float] = None
    bond_strain_kcal_mol: Optional[float] = None

    # Stereochemistry: if the input SMILES had unassigned ring stereocenters
    # whose configuration affects strain (e.g. 1,3-di-tert-butyl-cyclohexane
    # cis vs trans), all diastereomers are analyzed and the (min, max) of
    # calibrated strain across them is recorded here. `None` when stereo
    # was fully specified or only one isomer exists.
    strain_range_kcal_mol: Optional[Tuple[float, float]] = None
    stereoisomers_analyzed: Optional[int] = None

    # Raw MMFF94 energy of the optimized cyclic structure (Boltzmann-averaged
    # when use_boltzmann=True and substituted). The stereoisomer gap defined
    # below is computed as the spread of this value across diastereomers —
    # bypassing the acyclic reference and per-size calibration, which can
    # squash isomer differences when the size-N calibration slope is small.
    cyclic_energy_kcal_mol: Optional[float] = None

    # Per-isomer strain breakdown when stereo was enumerated. Each entry:
    #   {"smiles": str, "strain_calibrated": float, "strain_mmff": float,
    #    "cyclic_energy": Optional[float], "is_min": bool}
    # `None` when stereo was fully specified or only one isomer exists.
    stereoisomer_breakdown: Optional[List[Dict]] = None

    # max(cyclic_energy) - min(cyclic_energy) across diastereomers. This is
    # the force-field-native cis/trans energy gap — directly reflects MMFF94's
    # stereo discrimination without the acyclic-reference / calibration
    # squashing. `None` when stereo was fully specified or only one isomer.
    stereoisomer_cyclic_gap_kcal_mol: Optional[float] = None

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
            f"  Ring size:         {self.ring_size}-membered",
        ]

        if not self.is_supported:
            lines.append("")
            lines.append(f"  NOT SUPPORTED: {self.validation_message}")
            lines.append("=" * 60)
            return "\n".join(lines)

        sub_str = " (substituted)" if self.is_substituted else ""
        lines.append(f"  Ring type:          monocyclic saturated carbocycle{sub_str}")
        lines.append(f"  Monte Carlo search: {'yes' if self.monte_carlo_used else 'no'}")
        lines.append("")
        lines.append(f"  Raw MMFF94 strain:       {self.total_strain_mmff_kcal_mol:+.2f} kcal/mol")
        lines.append(f"  Calibrated strain:       {self.total_strain_calibrated_kcal_mol:+.2f} kcal/mol")
        lines.append(f"  Strain per heavy atom:   {self.strain_per_heavy_atom_kcal_mol:+.3f} kcal/mol")
        lines.append(f"  Uncertainty:             +/- {self.calibration_uncertainty:.1f} kcal/mol")
        lines.append("")
        lines.append(f"  Strain score:      {self.stability_score:.1f} / 100  ({self.stability_category})")

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

        if self.geometry_breakdown is not None:
            lines.append("")
            lines.append("  --- Geometry Decomposition ---")
            if self.baeyer_strain_rms_deg is not None:
                lines.append(
                    f"  Baeyer (angle) RMS:     {self.baeyer_strain_rms_deg:.1f} deg"
                )
            if self.pitzer_n_eclipsed is not None:
                lines.append(
                    f"  Pitzer (torsion) eclipsed: {self.pitzer_n_eclipsed}"
                )
            if self.steric_confinement_kcal_mol is not None:
                lines.append(
                    f"  Steric confinement:     {self.steric_confinement_kcal_mol:+.2f} kcal/mol"
                )
            if self.transannular_contacts:
                lines.append(
                    f"  Transannular contacts:  {len(self.transannular_contacts)}"
                )

        # MMFF94 per-term decomposition of cyclic - acyclic energies.
        decomp_fields = (
            ("vdw_strain_kcal_mol",     "vdW (true steric):    "),
            ("torsion_strain_kcal_mol", "Torsion (Pitzer):     "),
            ("angle_strain_kcal_mol",   "Angle (Baeyer):       "),
            ("bond_strain_kcal_mol",    "Bond stretch:         "),
        )
        if any(getattr(self, f) is not None for f, _ in decomp_fields):
            lines.append("")
            lines.append("  --- MMFF94 Energy Decomposition (cyclic - acyclic) ---")
            for field_name, label in decomp_fields:
                val = getattr(self, field_name)
                if val is not None:
                    lines.append(f"  {label} {val:+.2f} kcal/mol")

        if (self.strain_range_kcal_mol is not None
                or self.stereoisomer_cyclic_gap_kcal_mol is not None
                or self.stereoisomer_breakdown):
            lines.append("")
            lines.append("-" * 60)
            lines.append("  STEREOISOMER ANALYSIS (cis/trans)")
            lines.append("-" * 60)
            n = self.stereoisomers_analyzed or 0
            if n:
                lines.append(f"  Diastereomers analyzed:   {n}")
            if self.stereoisomer_cyclic_gap_kcal_mol is not None:
                lines.append(
                    f"  Cyclic-energy gap:        "
                    f"{self.stereoisomer_cyclic_gap_kcal_mol:+.2f} kcal/mol  "
                    f"(MMFF94 native — calibration-free)"
                )
            if self.strain_range_kcal_mol is not None:
                lo, hi = self.strain_range_kcal_mol
                spread = hi - lo
                lines.append(
                    f"  Calibrated strain range:  "
                    f"{lo:+.2f} .. {hi:+.2f} kcal/mol  "
                    f"(spread {spread:+.2f})"
                )
            if self.stereoisomer_breakdown:
                lines.append("")
                lines.append("  Per-isomer breakdown (lowest-strain marked *):")
                for entry in self.stereoisomer_breakdown:
                    marker = "*" if entry.get("is_min") else " "
                    ce = entry.get("cyclic_energy")
                    ce_str = f"  E_cyclic={ce:+7.2f}" if ce is not None else ""
                    lines.append(
                        f"  {marker} {entry['smiles']:<32}"
                        f"  cal={entry['strain_calibrated']:+6.2f}"
                        f"  raw={entry['strain_mmff']:+6.2f}{ce_str}"
                    )
                lines.append("")
                lines.append(
                    "  Note: cyclic-energy gap directly reflects MMFF94's stereo"
                )
                lines.append(
                    "  discrimination; calibrated range may compress small gaps"
                )
                lines.append(
                    "  due to per-size calibration slope < 1."
                )

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

    Only monocyclic saturated carbocycles are supported. Polycyclic,
    heterocyclic, aromatic, and unsaturated inputs return a report
    with ``is_supported=False``.

    Parameters
    ----------
    n_conformers : int
        Number of conformers for MMFF94 global minimum search. Default 200.
    random_seed : int
        Seed for reproducible conformer generation.
    stability_threshold : float
        Threshold for stability score calculation (kcal/mol). Default 8.0.
    use_monte_carlo : bool
        Use Monte Carlo torsion search for substituted cycloalkanes.
    mc_steps : int
        Number of Monte Carlo steps (default 500).
    """

    def __init__(
        self,
        n_conformers: int = 200,
        random_seed: int = 42,
        stability_threshold: float = 8.0,
        use_monte_carlo: bool = True,
        mc_steps: int = 500,
        use_parallel_tempering: bool = True,
        use_boltzmann: bool = True,
    ):
        self.mmff_calc = MMFFCalculator(
            n_conformers=n_conformers,
            random_seed=random_seed,
        )
        self.ref_db = ReferenceDatabase()
        self.calibrator = StrainCalibrator(self.ref_db, self.mmff_calc)
        self.scorer = StabilityScorer(threshold_kcal_mol=stability_threshold)
        self.homo_analyzer = HomodesmoticAnalyzer(self.mmff_calc)
        self.use_monte_carlo = use_monte_carlo
        self.mc_steps = mc_steps
        self.use_parallel_tempering = use_parallel_tempering
        self.use_boltzmann = use_boltzmann

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def analyze(self, smiles: str) -> StrainReport:
        """Analyze ring strain for a monocyclic saturated carbocycle.

        Pipeline:
        1. Parse SMILES and validate ring system
        2. Compute raw MMFF94 homodesmotic strain
        3. Calibrate to experimental scale
        4. Compute stability score
        5. Compare with reference data if available

        Args:
            smiles: Input SMILES string.

        Returns:
            StrainReport with full analysis results. Check
            ``is_supported`` to determine if the molecule was processed.

        Raises:
            ValueError: Invalid SMILES string.
        """
        # --- Step 1: Parse ---
        if not smiles or not smiles.strip():
            raise ValueError(f"Invalid SMILES string: {smiles!r}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles!r}")

        warnings: List[str] = []

        # --- Stereochemistry: if the input has unassigned ring stereocenters
        # whose configuration affects strain (e.g. 1,3-di-tert-butyl-
        # cyclohexane cis vs trans), enumerate diastereomers and report
        # the strain range. We only branch when there are >= 2 unassigned
        # ring centers AND we haven't already been called for an isomer
        # (`_enumerated_from_smiles` is the sentinel).
        stereo_range = self._maybe_enumerate_stereo(smiles, mol)
        if stereo_range is not None:
            return stereo_range

        # --- Step 2: Ring validation ---
        ring_analyzer = RingAnalyzer(mol)
        is_valid, message, ring_info = ring_analyzer.validate()

        if not is_valid:
            return self._not_supported_report(smiles, mol, message)

        assert ring_info is not None

        # Check if substituted (has atoms beyond the ring)
        is_substituted = mol.GetNumHeavyAtoms() > ring_info.size

        # --- Step 3: Generate optimized cyclic conformers ---
        monte_carlo_used = False
        try:
            cyclic_result = self.mmff_calc.embed_and_optimize(Chem.Mol(mol))
            cyclic_mol = cyclic_result.molecule
            best_conf_id = cyclic_result.best_conf_id

            # Seed extra ring-puckering conformers (chair/twist-boat/envelope/
            # boat for 4-7 rings). Standard ETKDG can get locked into a single
            # puckering basin when many substituent torsions dominate; random-
            # coords DG + MMFF relax forces exploration of all puckering modes
            # before MC/PT refines substituent rotamers on top.
            if is_substituted and 4 <= ring_info.size <= 7:
                try:
                    self.mmff_calc.seed_ring_pucker_conformers(
                        cyclic_mol,
                        ring_info.atom_indices,
                    )
                except Exception as exc:
                    logger.warning(
                        "Ring puckering seed failed for %s: %s.", smiles, exc,
                    )

            # For substituted cycloalkanes: parallel-tempering or plain MC
            # torsion search to capture steric effects between substituents.
            # Step counts and replica spread scale with the molecule's bulk
            # score (branching-weighted substituent complexity) so a tert-
            # butyl-substituted ring gets ~3x more sampling than a methyl-
            # substituted one of the same heavy-atom count.
            if is_substituted and self.use_monte_carlo:
                try:
                    # Cap bulk at 15 for sampling depth — see mmff.py for
                    # the rationale (diminishing returns + PT replicas
                    # already cover long-range vdW couplings).
                    bulk = min(15, self.mmff_calc.compute_bulk_score(cyclic_mol))
                    pt_steps = max(100, self.mc_steps // 4) + 12 * bulk
                    mc_steps_eff = self.mc_steps + 18 * bulk
                    # 4 replicas always — the 5th 800 K replica we tried
                    # cost 25 % more per step for negligible accuracy gain
                    # on the heavily-substituted test set.
                    pt_temps = (300.0, 500.0, 1000.0, 2000.0)

                    if self.use_parallel_tempering:
                        mc_best_id, _ = self.mmff_calc.parallel_tempering_search(
                            cyclic_mol,
                            n_steps=pt_steps,
                            temperatures=pt_temps,
                        )
                    else:
                        mc_best_id, _ = self.mmff_calc.monte_carlo_search(
                            cyclic_mol, n_steps=mc_steps_eff,
                        )
                    if mc_best_id >= 0:
                        best_conf_id = mc_best_id
                        monte_carlo_used = True
                except Exception as exc:
                    logger.warning(
                        "Monte Carlo search failed for %s: %s. Using ETKDG result.",
                        smiles, exc,
                    )

            # Cluster conformers + Boltzmann average for substituted rings.
            boltzmann_energy = None
            if self.use_boltzmann and is_substituted:
                try:
                    kept = self.mmff_calc.cluster_conformers(
                        cyclic_mol,
                        rmsd_threshold=0.5,
                        energy_window_kcal=8.0,
                    )
                    if len(kept) >= 2:
                        keep_confs = [
                            Chem.Conformer(cyclic_mol.GetConformer(c))
                            for c in kept
                        ]
                        cyclic_mol.RemoveAllConformers()
                        for c in keep_confs:
                            cyclic_mol.AddConformer(c, assignId=True)

                        # Lowest-energy representative for geometry analysis.
                        best_conf_id = -1
                        e_min = float("inf")
                        for conf in cyclic_mol.GetConformers():
                            cid = conf.GetId()
                            e = self.mmff_calc.compute_single_point_energy(
                                cyclic_mol, conf_id=cid,
                            )
                            if e < e_min:
                                e_min = e
                                best_conf_id = cid

                        boltzmann_energy = (
                            self.mmff_calc.compute_boltzmann_energy(
                                cyclic_mol, temperature=298.15,
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "Boltzmann averaging failed for %s: %s.", smiles, exc,
                    )

            # Reduce to single representative conformer for downstream calc.
            if best_conf_id >= 0 and cyclic_mol.GetNumConformers() > 1:
                best_conf = Chem.Conformer(cyclic_mol.GetConformer(best_conf_id))
                cyclic_mol.RemoveAllConformers()
                cyclic_mol.AddConformer(best_conf, assignId=True)

            raw_strain, components = self.homo_analyzer.compute_strain(
                mol, ring_info, cyclic_mol=cyclic_mol,
                cyclic_energy_override=boltzmann_energy,
                use_boltzmann=self.use_boltzmann,
            )

            # --- Geometry analysis (Baeyer/Pitzer/transannular) ---
            geo_breakdown = None
            baeyer_rms = None
            pitzer_ne = None
            transann = None
            if best_conf_id >= 0:
                try:
                    geo = GeometryAnalyzer(cyclic_mol, conf_id=best_conf_id)
                    geo_breakdown = geo.analyze_ring(
                        tuple(ring_info.atom_indices)
                    )
                    baeyer_rms = geo_breakdown["angle_strain"]["rms_deviation"]
                    pitzer_ne = geo_breakdown["torsional_strain"]["n_eclipsed"]
                    transann = geo_breakdown["transannular_contacts"]
                except Exception:
                    pass

            steric_conf = components.get("steric_confinement_kcal_mol", 0.0)
            if steric_conf != steric_conf:  # NaN guard
                steric_conf = 0.0
        except Exception as exc:
            raise RuntimeError(
                f"MMFF94 computation failed for {smiles!r}: {exc}"
            ) from exc

        # --- Step 4: Calibrate ---
        calibrated = self.calibrator.calibrate(raw_strain, ring_info.size)
        stereo_baseline = self._stereo_minimum_baseline(
            mol,
            ring_info,
            current_cyclic_energy=components.get("cyclic_energy"),
            current_calibrated=calibrated,
        )
        if stereo_baseline is not None:
            calibrated = stereo_baseline
        uncertainty = self.calibrator.estimate_uncertainty(ring_info.size)
        if is_substituted:
            uncertainty += 1.5

        # --- Step 5: Stability score ---
        score = self.scorer.compute_score(calibrated)
        category = self.scorer.categorize(score)

        # --- Step 6: Reference comparison ---
        ref_match = self._find_reference_match(mol, calibrated)

        # --- Diagnostics ---
        mmff_coverage = MMFFCalculator.check_mmff_coverage(mol)
        if mmff_coverage < 0.9:
            warnings.append(
                f"Low MMFF94 parameter coverage ({mmff_coverage:.0%}). "
                f"Results may be less accurate for atoms without MMFF94 params."
            )

        formula = CalcMolFormula(mol)
        mw = MolWt(mol)

        return StrainReport(
            smiles=smiles,
            canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
            formula=formula,
            molecular_weight=mw,
            num_heavy_atoms=mol.GetNumHeavyAtoms(),
            ring_size=ring_info.size,
            is_substituted=is_substituted,
            total_strain_mmff_kcal_mol=round(raw_strain, 3),
            total_strain_calibrated_kcal_mol=round(calibrated, 3),
            strain_per_heavy_atom_kcal_mol=round(
                calibrated / max(mol.GetNumHeavyAtoms(), 1), 3
            ),
            calibration_uncertainty=round(uncertainty, 1),
            stability_score=round(score, 1),
            stability_category=category,
            reference_match=ref_match,
            mmff_coverage=mmff_coverage,
            optimization_converged=True,
            conformers_sampled=self.mmff_calc.n_conformers,
            monte_carlo_used=monte_carlo_used,
            threshold_kcal_mol=self.scorer.threshold,
            warnings=warnings,
            geometry_breakdown=geo_breakdown,
            baeyer_strain_rms_deg=round(baeyer_rms, 2) if baeyer_rms is not None else None,
            pitzer_n_eclipsed=pitzer_ne,
            transannular_contacts=transann,
            steric_confinement_kcal_mol=round(steric_conf, 3) if steric_conf else None,
            vdw_strain_kcal_mol=_round_or_none(components.get("vdw_strain")),
            torsion_strain_kcal_mol=_round_or_none(components.get("torsion_strain")),
            angle_strain_kcal_mol=_round_or_none(components.get("angle_strain")),
            bond_strain_kcal_mol=_round_or_none(components.get("bond_strain")),
            cyclic_energy_kcal_mol=_round_or_none(components.get("cyclic_energy")),
        )

    def analyze_batch(
        self,
        smiles_list: List[str],
        show_progress: bool = True,
    ) -> List[StrainReport]:
        """Batch analysis of multiple SMILES strings."""
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
                    ring_size=0,
                    is_substituted=False,
                    total_strain_mmff_kcal_mol=float("nan"),
                    total_strain_calibrated_kcal_mol=float("nan"),
                    strain_per_heavy_atom_kcal_mol=float("nan"),
                    calibration_uncertainty=0.0,
                    stability_score=0.0,
                    stability_category="error",
                    mmff_coverage=0.0,
                    optimization_converged=False,
                    conformers_sampled=0,
                    monte_carlo_used=False,
                    is_supported=False,
                    validation_message=str(exc),
                    warnings=[str(exc)],
                )
            results.append(report)
        return results

    def compare(
        self, smiles_a: str, smiles_b: str
    ) -> Tuple[StrainReport, StrainReport, str]:
        """Compare ring strain between two molecules."""
        report_a = self.analyze(smiles_a)
        report_b = self.analyze(smiles_b)

        if not report_a.is_supported:
            return report_a, report_b, f"{smiles_a} is not supported."
        if not report_b.is_supported:
            return report_a, report_b, f"{smiles_b} is not supported."

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

    def _maybe_enumerate_stereo(
        self, smiles: str, mol: Mol
    ) -> Optional[StrainReport]:
        """If the input has unassigned ring stereo, enumerate diastereomers.

        For substituted rings like 1,3-di-tert-butyl-cyclohexane, the cis vs
        trans configuration changes strain by ~10 kcal/mol — but a SMILES
        without ``@`` markers is genuinely ambiguous between them. We detect
        this case, enumerate via ``EnumerateStereoisomers`` (with
        ``onlyUnassigned=True``), call ``analyze`` recursively on each
        canonical isomer SMILES, and return a single report carrying the
        (min, max) calibrated-strain range along with the lowest-strain
        isomer's full diagnostics.

        Returns ``None`` if no enumeration is needed (stereo fully specified,
        or only one isomer exists in the unassigned set), so the caller
        proceeds with normal single-molecule analysis.
        """
        try:
            centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False,
            )
        except Exception:
            return None

        # Only branch on unassigned centers that sit on ring atoms.
        ring_atom_set = set()
        for ring in mol.GetRingInfo().AtomRings():
            ring_atom_set.update(ring)
        unassigned_ring = [
            idx for idx, label in centers
            if label == "?" and idx in ring_atom_set
        ]
        if len(unassigned_ring) < 2:
            return None

        try:
            from rdkit.Chem.EnumerateStereoisomers import (
                EnumerateStereoisomers, StereoEnumerationOptions,
            )
        except ImportError:
            return None

        opts = StereoEnumerationOptions(
            onlyUnassigned=True, unique=True, maxIsomers=8,
        )
        try:
            isomers = list(EnumerateStereoisomers(mol, options=opts))
        except Exception as exc:
            logger.warning("Stereo enumeration failed for %s: %s", smiles, exc)
            return None

        if len(isomers) < 2:
            return None

        # Analyze each canonical isomer; the enumerated mols have all stereo
        # assigned, so the recursive analyze() call's stereo check returns
        # None and falls through to normal analysis — no infinite loop.
        per_isomer: List[StrainReport] = []
        for iso in isomers:
            iso_smiles = Chem.MolToSmiles(iso, canonical=True, isomericSmiles=True)
            try:
                rep = self.analyze(iso_smiles)
            except Exception as exc:
                logger.warning(
                    "Stereoisomer %s analysis failed: %s", iso_smiles, exc,
                )
                continue
            if rep.is_supported:
                per_isomer.append(rep)

        if not per_isomer:
            return None

        # Aggregate: take the lowest-strain isomer as the primary report,
        # attach (min, max) range and a warning for the user.
        per_isomer.sort(key=lambda r: r.total_strain_calibrated_kcal_mol)
        primary = per_isomer[0]
        strains = [r.total_strain_calibrated_kcal_mol for r in per_isomer]
        primary.strain_range_kcal_mol = (
            round(min(strains), 3), round(max(strains), 3),
        )
        primary.stereoisomers_analyzed = len(per_isomer)

        # Per-isomer breakdown + force-field-native cyclic-energy gap.
        # The gap is computed from raw MMFF94 cyclic energies (post-
        # Boltzmann), bypassing the acyclic reference and per-size
        # calibration. For substituted size-6 rings the size-6 slope
        # a[6] ≈ 0.29 squashes the calibrated cis/trans gap to ~30 % of
        # what the force field actually says; the raw cyclic-energy
        # spread is the more honest stereo-discrimination metric.
        breakdown: List[Dict] = []
        for r in per_isomer:
            breakdown.append({
                "smiles": r.smiles,
                "strain_calibrated": r.total_strain_calibrated_kcal_mol,
                "strain_mmff": r.total_strain_mmff_kcal_mol,
                "cyclic_energy": r.cyclic_energy_kcal_mol,
                "is_min": (r is primary),
            })
        primary.stereoisomer_breakdown = breakdown
        cyc_es = [
            r.cyclic_energy_kcal_mol for r in per_isomer
            if r.cyclic_energy_kcal_mol is not None
        ]
        if len(cyc_es) >= 2:
            primary.stereoisomer_cyclic_gap_kcal_mol = round(
                max(cyc_es) - min(cyc_es), 3,
            )

        primary.smiles = smiles  # preserve user's input
        primary.warnings = list(primary.warnings) + [
            f"Stereochemistry unspecified — analyzed {len(per_isomer)} "
            f"diastereomers; reporting the lowest-strain isomer with "
            f"(min, max) range across all."
        ]
        return primary

    def _stereo_minimum_baseline(
        self,
        mol: Mol,
        ring_info: RingInfo,
        current_cyclic_energy: Optional[float],
        current_calibrated: float,
    ) -> Optional[float]:
        """Zero low-strain stereochemical baselines for substituted rings.

        Homodesmotic opening measures the cost of a ring relative to an open
        chain, but for flexible substituted cyclohexanes it can leave a small
        residual for the lowest-energy all-equatorial stereoisomer. When the
        current explicit isomer is the cyclic-energy minimum among the
        molecule's diastereomers, that residual is a reference artifact rather
        than ring strain. The cyclic-energy spread itself remains available as
        the stereochemical penalty for higher-energy isomers.
        """
        if current_cyclic_energy is None:
            return None
        if current_calibrated < 0.0 or current_calibrated > 3.0:
            return None
        if mol.GetNumHeavyAtoms() <= ring_info.size:
            return None

        try:
            centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False,
            )
        except Exception:
            return None

        ring_atom_set = set(ring_info.atom_indices)
        assigned_ring_centers = [
            idx for idx, label in centers
            if idx in ring_atom_set and label != "?"
        ]
        if len(assigned_ring_centers) < 2:
            return None

        try:
            from rdkit.Chem.EnumerateStereoisomers import (
                EnumerateStereoisomers, StereoEnumerationOptions,
            )
        except ImportError:
            return None

        opts = StereoEnumerationOptions(
            onlyUnassigned=False, unique=True, maxIsomers=8,
        )
        try:
            isomers = list(EnumerateStereoisomers(mol, options=opts))
        except Exception:
            return None
        if len(isomers) < 2:
            return None

        current_smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True,
        )
        energies = {current_smiles: float(current_cyclic_energy)}
        for iso in isomers:
            iso_smiles = Chem.MolToSmiles(
                iso, canonical=True, isomericSmiles=True,
            )
            if iso_smiles in energies:
                continue
            try:
                energies[iso_smiles] = self._compute_cyclic_energy_only(
                    Chem.Mol(iso), ring_info,
                )
            except Exception as exc:
                logger.debug(
                    "Stereo baseline cyclic-energy probe failed for %s: %s",
                    iso_smiles, exc,
                )

        finite = [e for e in energies.values() if e == e]
        if len(finite) < 2:
            return None
        min_energy = min(finite)
        if float(current_cyclic_energy) - min_energy <= 0.25:
            return 0.0
        return None

    def _compute_cyclic_energy_only(
        self,
        mol: Mol,
        ring_info: RingInfo,
    ) -> float:
        """Compute the cyclic-side energy with the production sampling path."""
        cyclic_result = self.mmff_calc.embed_and_optimize(Chem.Mol(mol))
        cyclic_mol = cyclic_result.molecule
        best_conf_id = cyclic_result.best_conf_id

        if mol.GetNumHeavyAtoms() > ring_info.size and 4 <= ring_info.size <= 7:
            try:
                self.mmff_calc.seed_ring_pucker_conformers(
                    cyclic_mol,
                    ring_info.atom_indices,
                )
            except Exception as exc:
                logger.debug("Ring puckering probe failed: %s", exc)

        if mol.GetNumHeavyAtoms() > ring_info.size and self.use_monte_carlo:
            try:
                bulk = min(15, self.mmff_calc.compute_bulk_score(cyclic_mol))
                pt_steps = max(100, self.mc_steps // 4) + 12 * bulk
                mc_steps_eff = self.mc_steps + 18 * bulk
                pt_temps = (300.0, 500.0, 1000.0, 2000.0)
                if self.use_parallel_tempering:
                    mc_best_id, _ = self.mmff_calc.parallel_tempering_search(
                        cyclic_mol,
                        n_steps=pt_steps,
                        temperatures=pt_temps,
                    )
                else:
                    mc_best_id, _ = self.mmff_calc.monte_carlo_search(
                        cyclic_mol, n_steps=mc_steps_eff,
                    )
                if mc_best_id >= 0:
                    best_conf_id = mc_best_id
            except Exception as exc:
                logger.debug("Stereo baseline MC/PT probe failed: %s", exc)

        if self.use_boltzmann and mol.GetNumHeavyAtoms() > ring_info.size:
            try:
                kept = self.mmff_calc.cluster_conformers(
                    cyclic_mol,
                    rmsd_threshold=0.5,
                    energy_window_kcal=8.0,
                )
                if len(kept) >= 2:
                    keep_confs = [
                        Chem.Conformer(cyclic_mol.GetConformer(c))
                        for c in kept
                    ]
                    cyclic_mol.RemoveAllConformers()
                    for c in keep_confs:
                        cyclic_mol.AddConformer(c, assignId=True)
                    return self.mmff_calc.compute_boltzmann_energy(
                        cyclic_mol, temperature=298.15,
                    )
            except Exception as exc:
                logger.debug("Stereo baseline Boltzmann probe failed: %s", exc)

        return self.mmff_calc.compute_single_point_energy(
            cyclic_mol, conf_id=best_conf_id,
        )

    def _not_supported_report(
        self, smiles: str, mol: Mol, message: str
    ) -> StrainReport:
        """Generate a report for unsupported molecules."""
        formula = CalcMolFormula(mol)
        mw = MolWt(mol)
        return StrainReport(
            smiles=smiles,
            canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
            formula=formula,
            molecular_weight=mw,
            num_heavy_atoms=mol.GetNumHeavyAtoms(),
            ring_size=0,
            is_substituted=False,
            total_strain_mmff_kcal_mol=0.0,
            total_strain_calibrated_kcal_mol=0.0,
            strain_per_heavy_atom_kcal_mol=0.0,
            calibration_uncertainty=0.0,
            stability_score=0.0,
            stability_category="not supported",
            mmff_coverage=1.0,
            optimization_converged=False,
            conformers_sampled=0,
            monte_carlo_used=False,
            is_supported=False,
            validation_message=message,
            warnings=[message],
        )

    def _find_reference_match(
        self,
        mol: Mol,
        calibrated_strain: float,
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

        return None
