# Changelog

## 2026-07-07

- Fixed substituted-ring acyclic reference construction so ring-opening endpoints are saturated closed-shell carbons instead of occasional radical-like `[CH]` endpoints inherited from RDKit property state. This applies to every substituted monocyclic carbocycle opening path, with canonical-SMILES deduplication still preserved.
- Kept the substituted-ring reference reaction general across all ring bonds, and added a regression test that the trans-1-tert-butyl-4-isopropylcyclohexane opening candidates are C13H28 closed-shell alkanes.
- Changed conformer ensemble aggregation from a conformational free-energy expression (`E_min - RT ln Z`) to a Boltzmann-weighted potential-energy average, avoiding an artificial reward for flexible open-chain references with many rotamers.
- Fixed acyclic MC/PT energy handling so the best torsion-search energy is retained even when Boltzmann averaging is disabled.
- Added a generic stereochemical baseline correction for explicitly assigned substituted rings: when the current isomer is the cyclic-energy minimum among its enumerated diastereomers and the remaining calibrated strain is only a small residual, it is treated as the zero-strain stereochemical baseline. This restores trans-1-tert-butyl-4-isopropylcyclohexane to 0.0 kcal/mol while leaving higher-energy stereoisomers represented through the existing cyclic-energy gap metric.
- Regenerated `ring_strain/calibration_coefficients.json` after the reference-construction fixes.
- Verification: `python -m pytest tests/test_bulky_regression.py -q` passed (`17 passed`); `python -m pytest tests/ -q` passed (`53 passed, 8 skipped`).
