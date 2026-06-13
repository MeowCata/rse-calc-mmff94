"""Slow accuracy benchmark on the 8 new bulky reference compounds.

Each test runs the full StrainAnalyzer pipeline on one substituted ring
and asserts the calibrated strain falls within ±2.5 kcal/mol of the
literature value. This file is skipped from the default test suite
because it takes ~10 minutes total; enable it with::

    RUN_BENCHMARKS=1 python -m pytest tests/test_bulky_benchmark.py -v --tb=short

The assertion-failure messages print the absolute error so the output
tells you how far each compound is from experimental — useful for
deciding whether to add a bulk-class calibration term in
``ring_strain/calibrate.py``.
"""

import os
import pytest

from mmff94.ring_strain.core import StrainAnalyzer


pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_BENCHMARKS"),
    reason="bulky benchmark — run with RUN_BENCHMARKS=1",
)


# (name, SMILES, experimental strain kcal/mol)
BULKY_REFERENCES = [
    ("1,1-dimethylcyclopropane",         "CC1(C)CC1",                                      31.0),
    ("1,1,2,2-tetramethylcyclopropane",  "CC1(C)C(C)(C)C1",                                35.5),
    ("tert-butylcyclohexane",            "CC(C)(C)C1CCCCC1",                                5.4),
    ("cis-1,3-di-tert-butyl-cyclohexane",   "CC(C)(C)[C@H]1CCC[C@@H](C(C)(C)C)C1",        11.2),
    ("trans-1,3-di-tert-butyl-cyclohexane", "CC(C)(C)[C@H]1CCC[C@H](C(C)(C)C)C1",          0.5),
    ("1,1-dimethylcyclobutane",          "CC1(C)CCC1",                                     27.5),
    ("1,1-dimethylcyclopentane",         "CC1(C)CCCC1",                                     7.5),
    ("1,1-dimethylcyclohexane",          "CC1(C)CCCCC1",                                    1.4),
]

# Tolerance for the assertion. A wider band would always pass and hide
# accuracy gaps; ±2.5 kcal/mol is tight enough to surface them.
TOLERANCE = 2.5


@pytest.fixture(scope="module")
def analyzer():
    # Production defaults — this is an accuracy benchmark, not a speed test.
    return StrainAnalyzer()


@pytest.mark.parametrize("name,smiles,exp", BULKY_REFERENCES, ids=[r[0] for r in BULKY_REFERENCES])
def test_bulky_reference_within_tolerance(analyzer, name, smiles, exp):
    rep = analyzer.analyze(smiles)
    assert rep.is_supported, f"{name}: pipeline rejected the input ({rep.validation_message})"
    computed = rep.total_strain_calibrated_kcal_mol
    error = computed - exp
    abs_err = abs(error)
    # Embed the diagnostic numbers in the assertion message so even on
    # failure the output tells you the gap, not just that it's outside ±X.
    assert abs_err <= TOLERANCE, (
        f"{name}: computed={computed:+.2f} kcal/mol, exp={exp:+.2f}, "
        f"signed_err={error:+.2f} (|err|={abs_err:.2f} > {TOLERANCE})"
    )
