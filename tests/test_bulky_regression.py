"""Regression tests for the bulky-substituent accuracy refactor.

These tests focus on the new infrastructure introduced in the
``frolicking-coalescing-crescent`` plan: vdW confinement diagnostics,
ring-puckering seeds, bulk-aware sampling budget, clash pre-screen, and
stereochemistry enumeration. They run quickly (no full strain pipeline
on every reference compound) so they can run as part of the standard
test suite.

Heavier accuracy benchmarks on the 8 new reference compounds belong in
a separate slow-test file run on demand, not in the default suite —
each bulky reference takes 30-90 seconds under the new bulk-aware
sampling depth.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ring_strain.mmff import MMFFCalculator
from ring_strain.core import StrainAnalyzer
from ring_strain.homodesmotic import HomodesmoticAnalyzer
from ring_strain.ring_analysis import RingAnalyzer
from ring_strain.geometry import GeometryAnalyzer


# ---------------------------------------------------------------------------
# MMFF94 vdW confinement term
# ---------------------------------------------------------------------------

class TestMMFFVdwEnergy:
    """Verify the retained vdW-only diagnostic."""

    def _embed_and_optimize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def test_cyclohexane_vdw_nontrivial(self):
        """Cyclohexane chair has measurable vdW contribution."""
        mol = self._embed_and_optimize("C1CCCCC1")
        # Chair has gauche-staggered carbons; vdW > 0 from 1,4 repulsions.
        assert MMFFCalculator.compute_vdw_energy(mol) > 1.0

    def test_boltzmann_energy_is_not_conformer_free_energy(self):
        """Conformer averaging must not add a degeneracy entropy bonus."""
        calc = MMFFCalculator(n_conformers=20)
        result = calc.embed_and_optimize(Chem.MolFromSmiles("CCCC"))
        mol = result.molecule
        energies = [
            calc.compute_single_point_energy(mol, conf_id=conf.GetId())
            for conf in mol.GetConformers()
        ]
        ensemble_energy = calc.compute_boltzmann_energy(mol)

        assert min(energies) <= ensemble_energy <= max(energies)


# ---------------------------------------------------------------------------
# Ring topology and Pitzer geometry
# ---------------------------------------------------------------------------

class TestPitzerGeometry:
    """Pitzer counts use real ring bonds and valid projected neighbours."""

    @staticmethod
    def _optimized(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def test_branched_smiles_ring_atoms_are_in_bond_order(self):
        mol = Chem.MolFromSmiles("C(C1)(CCC1)")
        valid, _, ring = RingAnalyzer(mol).validate()
        assert valid
        assert len(ring.bond_indices) == ring.size == 5
        for idx, atom_idx in enumerate(ring.atom_indices):
            next_idx = ring.atom_indices[(idx + 1) % ring.size]
            assert mol.GetBondBetweenAtoms(atom_idx, next_idx) is not None

    def test_cyclopropane_counts_three_real_eclipsed_bonds(self):
        mol = self._optimized("C1CC1")
        ring = RingAnalyzer(mol).validate()[2]
        result = GeometryAnalyzer(mol).compute_torsion_strain_in_ring(
            ring.atom_indices
        )

        assert result["n_eclipsed"] == 3
        assert len(result["torsions"]) == ring.size
        for torsion in result["torsions"]:
            assert len(set(torsion["atoms"])) == 4
            j, k = torsion["bond"]
            assert mol.GetBondBetweenAtoms(j, k) is not None


# ---------------------------------------------------------------------------
# Bulk score
# ---------------------------------------------------------------------------

class TestBulkScore:
    """Branching-weighted bulk score for substituent steric complexity."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CC1CCCCC1", 1),               # methyl: 1 linear
        ("CCC1CCCCC1", 2),              # ethyl: 2 linear
        ("CC(C)C1CCCCC1", 6),           # isopropyl: 1 branched (4) + 2 linear (1+1)
        ("CC(C)(C)C1CCCCC1", 12),       # tert-butyl: 1 quaternary (9) + 3 linear (3)
        ("C1CCCCC1", 0),                # cyclohexane: no off-ring atoms
    ])
    def test_bulk_score(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert MMFFCalculator.compute_bulk_score(mol) == expected


# ---------------------------------------------------------------------------
# Homodesmotic acyclic references
# ---------------------------------------------------------------------------

class TestAcyclicReferenceConstruction:
    """Ring opening must produce saturated closed-shell alkane references."""

    def test_opened_substituted_ring_is_h_capped_and_closed_shell(self):
        smiles = "CC(C)[C@H]1CC[C@H](C(C)(C)C)CC1"
        mol = Chem.MolFromSmiles(smiles)
        ring_info = RingAnalyzer(mol).validate()[2]
        homo = HomodesmoticAnalyzer(MMFFCalculator(n_conformers=10))

        candidates = homo._build_all_acyclic_references(Chem.Mol(mol), ring_info)

        assert candidates
        assert CalcMolFormula(mol) == "C13H26"
        for acyclic, _ in candidates:
            assert CalcMolFormula(acyclic) == "C13H28"
            assert all(
                atom.GetNumRadicalElectrons() == 0
                for atom in acyclic.GetAtoms()
            )


# ---------------------------------------------------------------------------
# Stereo enumeration
# ---------------------------------------------------------------------------

class TestStereoEnumeration:
    """Unassigned ring stereocenters trigger diastereomer enumeration."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        # Fewer conformers for speed; the test is about wiring, not accuracy.
        return StrainAnalyzer(n_conformers=30, mc_steps=80)

    def test_no_stereo_unspecified_no_range(self, analyzer):
        """Cyclohexane has no stereocenters: no range, no enumeration."""
        rep = analyzer.analyze("C1CCCCC1")
        assert rep.strain_range_kcal_mol is None
        assert rep.stereoisomers_analyzed is None

    def test_fully_specified_stereo_no_range(self, analyzer):
        """Explicit stereo on 1,3-DTB-cyclohexane: no enumeration triggered."""
        rep = analyzer.analyze("CC(C)(C)[C@H]1CCC[C@H](C(C)(C)C)C1")
        assert rep.strain_range_kcal_mol is None
        assert rep.stereoisomers_analyzed is None

    def test_lowest_cyclic_stereoisomer_sets_zero_baseline(self, analyzer):
        """Low-strain all-equatorial cyclohexane stereoisomers define baseline."""
        rep = analyzer.analyze("CC(C)[C@H]1CC[C@H](C(C)(C)C)CC1")
        assert rep.total_strain_calibrated_kcal_mol == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Retained strain diagnostics in StrainReport
# ---------------------------------------------------------------------------

class TestStrainReportDiagnostics:
    """Reports retain vdW confinement but omit per-term decomposition."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        return StrainAnalyzer(n_conformers=30, mc_steps=80)

    def test_substituted_ring_has_requested_diagnostics(self, analyzer):
        rep = analyzer.analyze("CC1CC1")
        assert rep.steric_confinement_kcal_mol is not None
        assert rep.baeyer_strain_rms_deg is not None
        assert rep.pitzer_n_eclipsed is not None
        for removed in (
            "vdw_strain_kcal_mol",
            "torsion_strain_kcal_mol",
            "angle_strain_kcal_mol",
            "bond_strain_kcal_mol",
        ):
            assert not hasattr(rep, removed)
            assert removed not in rep.to_dict()
        assert "MMFF94 Energy Decomposition" not in rep.print_summary()


# ---------------------------------------------------------------------------
# Stereoisomer breakdown + cyclic-energy gap (the cis/trans highlight)
# ---------------------------------------------------------------------------

class TestStereoIsomerBreakdown:
    """When stereo is unspecified, breakdown + gap fields are populated."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        # Fast wiring test — n_conformers/mc_steps reduced for speed.
        return StrainAnalyzer(n_conformers=30, mc_steps=80)

    def test_bare_dimethylcyclohexane_has_breakdown(self, analyzer):
        """1,4-dimethylcyclohexane (bare) populates breakdown + gap."""
        rep = analyzer.analyze("CC1CCC(C)CC1")
        assert rep.stereoisomers_analyzed is not None
        assert rep.stereoisomers_analyzed >= 2
        assert rep.stereoisomer_breakdown is not None
        assert len(rep.stereoisomer_breakdown) == rep.stereoisomers_analyzed
        # The cyclic-energy gap exists and is positive.
        assert rep.stereoisomer_cyclic_gap_kcal_mol is not None
        assert rep.stereoisomer_cyclic_gap_kcal_mol > 0
        # Exactly one entry flagged is_min.
        n_min = sum(1 for e in rep.stereoisomer_breakdown if e.get("is_min"))
        assert n_min == 1
        # Each entry has the expected keys.
        for entry in rep.stereoisomer_breakdown:
            assert "smiles" in entry
            assert "strain_calibrated" in entry
            assert "strain_mmff" in entry
            assert "cyclic_energy" in entry
            assert "is_min" in entry

    def test_unsubstituted_has_no_breakdown(self, analyzer):
        """Cyclohexane (no stereo) has no breakdown / gap."""
        rep = analyzer.analyze("C1CCCCC1")
        assert rep.stereoisomer_breakdown is None
        assert rep.stereoisomer_cyclic_gap_kcal_mol is None

    def test_substituted_has_cyclic_energy(self, analyzer):
        """Cyclic_energy is populated on every substituted report."""
        rep = analyzer.analyze("CC1CCCCC1")  # methylcyclohexane, no stereo
        assert rep.cyclic_energy_kcal_mol is not None
