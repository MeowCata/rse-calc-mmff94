# Changelog

## 2026-09-09

- Expanded the Matplotlib renderer into an interactive conformer viewer with
  a conformer slider, hydrogen-visibility toggle, reset-view button, and the
  standard Matplotlib 3D mouse rotation and zoom controls.
- Retained energy-sorted low-energy representatives from the production
  ETKDG/MMFF94 and substituted-ring PT/MC sampling pool. Added symmetry-aware
  heavy-atom RMSD clustering so explicit-hydrogen rotations and graph-equivalent
  coordinates are not presented as distinct conformers.
- Displayed both the MMFF94 conformer energy and its energy relative to the
  lowest-energy representative. Conformer changes preserve the current camera
  angle, while reset restores both the default orientation and molecular scale.
- Changed `scripts/render_conformation.py` to open the interactive viewer when
  `--output` is omitted; static PNG/SVG/PDF export remains available, and
  `--show-hydrogens` now also initializes the interactive viewer correctly.
- Exported `ConformerViewer` and `create_conformer_viewer` from the package API,
  added ETKDG fallback behavior when optional ring-pucker or PT/MC exploration
  fails, and closed Matplotlib figures after static export.
- Verification: `python -m pytest tests/test_visualization.py -q` passed
  (`5 passed`); Python compilation, TkAgg backend detection, interactive-widget
  event handling, camera preservation, and CLI static export were also checked.

## 2026-09-08

- Added an optional Matplotlib 3D renderer for the model's optimized cyclic
  conformers. `scripts/render_conformation.py` runs the production cyclic
  sampling sequence and renders the selected lowest-energy geometry to PNG,
  SVG, PDF, or an interactive Matplotlib window.

## 2026-09-08

- Removed the public MMFF94 per-term energy decomposition from reports, JSON,
  and the CLI while retaining total/raw/calibrated strain, per-atom strain,
  score, Baeyer RMS, Pitzer eclipsed-ring-bond counts, and vdW steric
  confinement.
- Added structured progress callbacks with monotonic overall progress,
  per-analysis elapsed wall time, nested stereoisomer/candidate scopes, and a
  CLI renderer that keeps progress on stderr (`--no-progress` disables it).
- Corrected stereochemical baseline-search indentation and expanded acyclic
  reference progress reporting to cover seeding, MC/PT, clustering, and
  Boltzmann averaging.
- Optimized MC/PT heavy-atom clash screening with vectorized coordinate access
  without changing the clash threshold or bonded-pair exclusions.
- Documented deferred accuracy and uncertainty improvements in
  `improve-accu.md`.

## 2026-07-07

- Fixed substituted-ring acyclic reference construction so ring-opening endpoints are saturated closed-shell carbons instead of occasional radical-like `[CH]` endpoints inherited from RDKit property state. This applies to every substituted monocyclic carbocycle opening path, with canonical-SMILES deduplication still preserved.
- Kept the substituted-ring reference reaction general across all ring bonds, and added a regression test that the trans-1-tert-butyl-4-isopropylcyclohexane opening candidates are C13H28 closed-shell alkanes.
- Changed conformer ensemble aggregation from a conformational free-energy expression (`E_min - RT ln Z`) to a Boltzmann-weighted potential-energy average, avoiding an artificial reward for flexible open-chain references with many rotamers.
- Fixed acyclic MC/PT energy handling so the best torsion-search energy is retained even when Boltzmann averaging is disabled.
- Added a generic stereochemical baseline correction for explicitly assigned substituted rings: when the current isomer is the cyclic-energy minimum among its enumerated diastereomers and the remaining calibrated strain is only a small residual, it is treated as the zero-strain stereochemical baseline. This restores trans-1-tert-butyl-4-isopropylcyclohexane to 0.0 kcal/mol while leaving higher-energy stereoisomers represented through the existing cyclic-energy gap metric.
- Regenerated `ring_strain/calibration_coefficients.json` after the reference-construction fixes.
- Verification: `python -m pytest tests/test_bulky_regression.py -q` passed (`17 passed`); `python -m pytest tests/ -q` passed (`53 passed, 8 skipped`).
