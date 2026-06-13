"""Regression tests for the bulky-substituent accuracy refactor.

These tests focus on the new infrastructure introduced in the
``frolicking-coalescing-crescent`` plan: MMFF94 per-term decomposition,
ring-puckering seeds, bulk-aware sampling budget, vdW pre-screen, and
stereochemistry enumeration. They run quickly (no full strain pipeline
on every reference compound) so they can run as part of the standard
test suite.

Heavier accuracy benchmarks on the 8 new reference compounds belong in
a separate slow-test file run on demand, not in the default suite —
each bulky reference takes 30-90 seconds under the new bulk-aware
sampling depth.
"""

import math
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ring_strain.mmff import MMFFCalculator
from ring_strain.core import StrainAnalyzer


# ---------------------------------------------------------------------------
# MMFF94 term decomposition
# ---------------------------------------------------------------------------

class TestMMFFDecomposition:
    """Verify decompose_energy isolates each MMFF94 term correctly."""

    def _embed_and_optimize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def test_decomposition_returns_all_terms(self):
        mol = self._embed_and_optimize("CC")
        decomp = MMFFCalculator.decompose_energy(mol)
        expected_keys = {
            "total", "bond", "angle", "stretch_bend",
            "oop", "torsion", "vdw", "electrostatic",
        }
        assert expected_keys <= set(decomp.keys())

    def test_terms_sum_to_total(self):
        mol = self._embed_and_optimize("C1CCCCC1")
        d = MMFFCalculator.decompose_energy(mol)
        # All non-total terms should sum to total within numerical tolerance.
        # MMFF94 terms are additive in the force field.
        components_sum = sum(
            v for k, v in d.items()
            if k != "total" and v == v  # skip NaN
        )
        assert math.isclose(components_sum, d["total"], abs_tol=0.05)

    def test_cyclohexane_vdw_nontrivial(self):
        """Cyclohexane chair has measurable vdW contribution."""
        mol = self._embed_and_optimize("C1CCCCC1")
        d = MMFFCalculator.decompose_energy(mol)
        # Chair has gauche-staggered carbons; vdW > 0 from 1,4 repulsions.
        assert d["vdw"] > 1.0


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


# ---------------------------------------------------------------------------
# Per-term strain plumbing in StrainReport
# ---------------------------------------------------------------------------

class TestStrainReportDecomposition:
    """The four new *_strain_kcal_mol fields are populated for substituted rings."""

    @pytest.fixture(scope="class")
    def analyzer(self):
        return StrainAnalyzer(n_conformers=30, mc_steps=80)

    def test_substituted_ring_has_decomposition(self, analyzer):
        """methylcyclopropane (substituted) gets per-term strain in report."""
        rep = analyzer.analyze("CC1CC1")
        assert rep.vdw_strain_kcal_mol is not None
        assert rep.torsion_strain_kcal_mol is not None
        # Cyclopropane derivatives: torsion (eclipsed) dominates strain.
        assert abs(rep.torsion_strain_kcal_mol) > 1.0
