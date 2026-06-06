"""
Regression tests for ring strain analysis.

Validates computed strain values against known experimental values
for standard cycloalkanes (C3-C8) and representative polycyclic compounds.
"""

import pytest
from mmff94.ring_strain.core import StrainAnalyzer

# Tolerance: +/- 5 kcal/mol for calibrated values
# (MMFF94 systematic errors + calibration interpolation)
_CALIBRATED_TOLERANCE = 5.0

# Known experimental values for core cycloalkanes
_CYCLOALKANE_REFERENCE = {
    "C1CC1": 27.5,      # cyclopropane
    "C1CCC1": 26.3,     # cyclobutane
    "C1CCCC1": 6.5,     # cyclopentane
    "C1CCCCC1": 0.0,    # cyclohexane
    "C1CCCCCC1": 6.3,   # cycloheptane
    "C1CCCCCCC1": 9.7,  # cyclooctane
}

# Known values for polycyclic / other compounds
_OTHER_REFERENCE = {
    "c1ccccc1": 0.0,     # benzene (aromatic, strain-free)
    "C1CC2CCC1C2": 15.0, # norbornane
}


@pytest.fixture(scope="module")
def analyzer():
    """Module-level analyzer fixture (expensive to initialize)."""
    return StrainAnalyzer(n_conformers=100, random_seed=42)


class TestCycloalkaneStrain:
    """Test that cycloalkane strain values match literature."""

    @pytest.mark.parametrize("smiles,expected", _CYCLOALKANE_REFERENCE.items())
    def test_cycloalkane_strain(self, analyzer, smiles, expected):
        report = analyzer.analyze(smiles)
        error = abs(report.total_strain_calibrated_kcal_mol - expected)
        assert error < _CALIBRATED_TOLERANCE, (
            f"{smiles}: computed {report.total_strain_calibrated_kcal_mol:.1f} "
            f"vs expected {expected:.1f} kcal/mol (error={error:.1f})"
        )


class TestRingSizes:
    """Test that ring size detection is correct."""

    @pytest.mark.parametrize("smiles,expected_size", [
        ("C1CC1", 3),
        ("C1CCC1", 4),
        ("C1CCCC1", 5),
        ("C1CCCCC1", 6),
        ("C1CCCCCC1", 7),
        ("C1CCCCCCC1", 8),
    ])
    def test_ring_size_detection(self, analyzer, smiles, expected_size):
        report = analyzer.analyze(smiles)
        assert report.ring_sizes == [expected_size]


class TestAcyclicMolecules:
    """Test that molecules without rings return zero strain."""

    @pytest.mark.parametrize("smiles", [
        "CC",           # ethane
        "CCCC",         # n-butane
        "CCCCCC",       # n-hexane
        "CCO",          # ethanol (acyclic)
        "CC(=O)O",      # acetic acid
    ])
    def test_acyclic_zero_strain(self, analyzer, smiles):
        report = analyzer.analyze(smiles)
        assert report.num_rings == 0
        assert report.stability_score >= 99.0


class TestAromaticRings:
    """Test that aromatic rings score as stable."""

    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",     # benzene
        "c1ccncc1",     # pyridine
    ])
    def test_aromatic_stable(self, analyzer, smiles):
        report = analyzer.analyze(smiles)
        assert report.stability_score > 80.0, (
            f"{smiles}: score={report.stability_score:.1f}, "
            f"expected > 80 (aromatic stabilization)"
        )


class TestStabilityScore:
    """Test that stability scores make qualitative sense."""

    def test_small_rings_low_score(self, analyzer):
        """Cyclopropane and cyclobutane should score low (< 20)."""
        for smi in ["C1CC1", "C1CCC1"]:
            report = analyzer.analyze(smi)
            assert report.stability_score < 20.0, (
                f"{smi}: score={report.stability_score:.1f}"
            )

    def test_cyclohexane_high_score(self, analyzer):
        """Cyclohexane should score near 100."""
        report = analyzer.analyze("C1CCCCC1")
        assert report.stability_score > 80.0, (
            f"cyclohexane: score={report.stability_score:.1f}"
        )

    def test_strain_ordering(self, analyzer):
        """Strain should follow: C3 > C4 > C5 > C6."""
        strains = {}
        for smi in ["C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1"]:
            report = analyzer.analyze(smi)
            strains[smi] = report.total_strain_calibrated_kcal_mol

        assert strains["C1CC1"] > strains["C1CCC1"]
        assert strains["C1CCC1"] > strains["C1CCCC1"]
        assert strains["C1CCCC1"] > strains["C1CCCCC1"]


class TestInvalidInputs:
    """Test graceful handling of invalid inputs."""

    def test_invalid_smiles(self, analyzer):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            analyzer.analyze("not_a_smiles!!")

    def test_empty_string(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze("")


class TestReportFormat:
    """Test that StrainReport produces correct output formats."""

    def test_to_dict(self, analyzer):
        report = analyzer.analyze("C1CC1")
        d = report.to_dict()
        assert "smiles" in d
        assert "total_strain_calibrated_kcal_mol" in d
        assert "stability_score" in d

    def test_to_json(self, analyzer):
        report = analyzer.analyze("C1CC1")
        j = report.to_json()
        assert '"smiles"' in j

    def test_print_summary(self, analyzer):
        report = analyzer.analyze("C1CC1")
        s = report.print_summary()
        assert "cyclopropane" in s.lower() or "C1CC1" in s or "Ring Strain" in s
