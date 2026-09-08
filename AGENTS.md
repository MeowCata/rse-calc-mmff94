# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project scope

MMFF94-based ring-strain energy quantification for **monocyclic saturated carbocycles only**. Inputs that are polycyclic (bridged / fused / spiro), heterocyclic, aromatic, or unsaturated are rejected with `is_supported=False` and a "Not Supported" message — do not extend the analysis pipeline to other ring systems.

The intended use case is **synthesizable real-world rings**: predict strain (including steric / vdW contributions) and discriminate stereoisomers (cis/trans) for compounds that may or may not be in the reference database. Generalization is achieved by per-ring-size linear calibration over a curated set of literature anchors plus a force-field-native stereoisomer gap that bypasses calibration entirely.

## Known limitations: extreme crowding

The homodesmotic ring-opening reference becomes unphysical for **all-substituted small rings** where every ring carbon carries multiple non-H substituents. Examples observed during gradient-pressure testing:

- 1,1,2,2-tetraethylcyclopropane: model +28.16 kcal/mol — calibrated strain order vs the methyl analog inverted.
- hexamethylcyclopropane: model +27.41 kcal/mol — qualitatively below tetra-tert-butyl despite stronger crowding.
- hexaethylcyclopropane: **raw MMFF strain −10.62 kcal/mol**, all four energy decomposition terms (vdW, torsion, angle, bond) negative. Calibration force-pushes this back to +12.65 but the number has no physical meaning.

Root cause: in the open-chain reference, every methyl/ethyl pair contributes a vdW repulsion that is *also* present in the ring; the difference `cyclic − acyclic` then cancels the very strain it should expose. The per-size linear calibrator cannot recover meaningful strain when the raw energy decomposition itself flips sign.

These compounds are at best fleeting laboratory curiosities (hexaethylcyclopropane has not been synthesized as of literature consulted), so this failure zone is documented rather than fixed. **Do not attempt to expand the analysis to multi-substituent small rings without changing the reference reaction to a strict bond-balanced scheme** (which is path 2 of `plans/humming-wobbling-porcupine.md` and was deliberately not taken).

## Running

All commands run from this directory (`C:\Users\miaoc\Desktop\mmff94`). Imports inside the codebase are top-level (`from ring_strain.core import ...`), so the project root must be on `sys.path` — which it automatically is when invoked via `python -m ...` from here.

```bash
# Single SMILES (cyclopropane)
python -m cli.main "C1CC1"

# JSON output, with reference comparison
python -m cli.main "C1CC1" --json --compare-ref

# Batch from file (one SMILES per line, `#` comments allowed)
python -m cli.main --batch compounds.txt

# Reference DB / benchmark vs experimental values
python -m cli.main --list-refs
python -m cli.main --benchmark

# Regenerate per-ring-size calibration coefficients
# (writes ring_strain/calibration_coefficients.json; ~10-15 min)
python scripts/derive_calibration.py
```

CLI knobs that materially change results: `--n-conformers` (default 200), `--mc-steps` (default 500), `--seed` (default 42), `--threshold` (default 8.0), `--no-monte-carlo` (disables MC/PT for substituted rings — see "Conformer sampling contract" below). Progress is shown on stderr by default; `--no-progress` suppresses it without changing results or JSON stdout.

## Tests

```bash
python -m pytest tests/                                  # full suite (~5 min)
python -m pytest tests/test_regression.py                # core cycloalkane regression
python -m pytest tests/test_bulky_regression.py          # tert-butyl / gem-dimethyl wiring tests (fast)
RUN_BENCHMARKS=1 python -m pytest tests/test_bulky_benchmark.py   # 8-compound bulky accuracy benchmark (~7 min)
python -m pytest tests/test_regression.py::TestCycloalkaneStrain::test_cycloalkane_strain
```

`tests/test_bulky_benchmark.py` is skipped by default; set `RUN_BENCHMARKS=1` to run it. The core cycloalkane regression uses a ±5 kcal/mol tolerance vs literature; the bulky benchmark uses ±2.5 kcal/mol.

## Scripts

```bash
# Derivation: regenerate calibration coefficients from reference set (~10-15 min)
python scripts/derive_calibration.py

# Validation: held-out generalization — 8 real-world compounds NOT in reference DB
python scripts/test_heldout_generalization.py

# Validation: cis/trans cyclic-energy gap on 1,2/1,3/1,4-dimethylcyclohexane
python scripts/verify_stereo_gap.py

# Validation: real-world monosubstituted strain accuracy + stereo gap
python scripts/validate_real_world.py

# Diagnostic: trace homodesmotic ring-opening for cis/trans pairs
python scripts/diag_homodesmotic.py
```

## Examples

`examples/basic_usage.py` demonstrates the API: single-molecule analysis, cycloalkane series, steric effects, unsupported inputs, and pairwise comparison. Run with `python examples/basic_usage.py`.

## Architecture

`StrainAnalyzer.analyze(smiles)` in `ring_strain/core.py` is the single entry point; everything else is a collaborator it instantiates. The pipeline is:

1. **Parse + stereo enumeration** (`core._maybe_enumerate_stereo`) — if the SMILES has ≥2 unassigned ring stereocenters (e.g. 1,3-di-tert-butyl-cyclohexane without `@` markers), `EnumerateStereoisomers` is invoked, `analyze` recurses on each canonical isomer, and the lowest-strain report is returned with `strain_range_kcal_mol = (min, max)` and `stereoisomers_analyzed` attached. The recursion terminates because enumerated isomers have stereo fully assigned. The report additionally exposes `stereoisomer_breakdown` (per-isomer SMILES + calibrated + raw + cyclic energy) and `stereoisomer_cyclic_gap_kcal_mol` (max−min of the raw MMFF cyclic energy across diastereomers). **The cyclic-energy gap is the force-field-native cis/trans discrimination metric** — it bypasses the acyclic reference and per-size calibration, both of which can compress small isomer differences when the size-N calibration slope is well below 1. Use this field when reporting stereo gaps; use `total_strain_calibrated_kcal_mol` for absolute strain comparisons.
2. **Ring validation** (`ring_analysis.RingAnalyzer.validate`) — rejects everything outside monocyclic saturated carbocycles.
3. **Cyclic conformer search** (`mmff.MMFFCalculator`) — ETKDG embed + MMFF94 optimize all conformers in parallel, with an adaptive conformer count that scales on rotatable bonds and `compute_bulk_score(mol)` (see sampling contract below). For substituted 4–7 rings, `seed_ring_pucker_conformers` then adds extra random-coords ETKDG seeds (`useRandomCoords=True`, `clearConfs=False`) so chair / twist-boat / envelope / half-chair basins are all sampled. PT + MC torsion search follows, with step counts also scaling on bulk. Cluster + Boltzmann-average the surviving conformers when `use_boltzmann=True`.
4. **Strain calculation** (`homodesmotic.HomodesmoticAnalyzer.compute_strain`) — two regimes:
   - **Unsubstituted cycloalkanes**: strict bond-balanced reaction `cyclo-(CH₂)ₙ + CH₃CH₃ → CH₃(CH₂)ₙ₊₁CH₃`. Per-ring-size strain is cached at `_CYCLOALKANE_STRAIN_CACHE`.
   - **Substituted**: ring opening at every single ring bond, with canonical-SMILES deduplication so symmetric rings don't pay for equivalent openings. Each unique candidate runs through the full embed + MMFF + (bulk-scaled) PT/MC + Boltzmann pipeline; the lowest-energy acyclic reference is kept.
5. **Focused diagnostics** — `GeometryAnalyzer` reports force-field-independent Baeyer angle RMS and counts ring bonds with eclipsed neighbour projections for the Pitzer diagnostic. `MMFFCalculator.compute_vdw_energy` isolates only the vdW contribution for the cyclic and acyclic representative geometries; their difference is exposed as `steric_confinement_kcal_mol`. The former public MMFF94 bond/angle/torsion/vdW decomposition fields are intentionally not reported.
6. **Calibration** (`calibrate.StrainCalibrator`) — per-ring-size linear correction `calibrated = a[size] * raw + b[size]`, loaded from `ring_strain/calibration_coefficients.json` for instant init. For ring sizes whose unsubstituted reference defines a strain-free anchor (notably cyclohexane = 0), the fit is constrained to pass exactly through `(raw_anchor, 0)`. Sizes with a single reference fall back to the historical multiplicative form. To regenerate after editing the reference set, run `python scripts/derive_calibration.py`.
7. **Scoring / reference match** — `scoring.StabilityScorer` maps calibrated strain to a 0–100 score (default threshold 8 kcal/mol); `reference.ReferenceDatabase` is queried by canonical SMILES.

Geometry diagnostics (`geometry.GeometryAnalyzer`) run alongside the energy path and contribute Baeyer (angle RMS), Pitzer (eclipsed-ring-bond count), and transannular-contact fields on the report. They are geometric snapshots from the single best conformer. `steric_confinement_kcal_mol` is likewise a representative-geometry vdW difference and is not expected to equal the Boltzmann-averaged total strain.

Long-running analyses emit structured `ProgressUpdate` events through the optional `StrainAnalyzer(..., progress_callback=...)` default or the per-call `analyze(..., progress_callback=...)` override. Events carry the current task, monotonic overall progress, local counters, SMILES, and elapsed wall time. Nested stereoisomer analyses share the root timer. Every returned `StrainReport` includes `elapsed_time_seconds`, and the CLI keeps progress on stderr so `--json` stdout remains machine-readable.

## Conformer sampling contract (critical)

For **substituted** rings, the Monte Carlo + parallel-tempering torsion search in `mmff.monte_carlo_search` / `parallel_tempering_search` must remain on the critical path. Single-conformer ETKDG produces chemically wrong results (e.g. tert-butyl-cyclopropane reports the same strain as cyclopropane because substituent rotamers are never explored). When fixing a bug in that path:

- Only repair the failing line; do not "simplify" by deleting the MC/PT call.
- The flow `parallel_tempering_search → cluster_conformers → compute_boltzmann_energy` is the validated chain inside `homodesmotic._compute_acyclic_energy_with_mol`. New requirements should add a bypass switch, not replace the default.
- Conformer IDs are not guaranteed to be sequential — iterate `[c.GetId() for c in mol.GetConformers()]` rather than `range(mol.GetNumConformers())`, otherwise RDKit raises "Bad Conformer Id" once a prior PT pass has removed and re-added conformers.
- If a substituted ring reports `~0 kcal/mol` strain, suspect the sampling before the force field.

Step counts and PT temperature ladders scale with `compute_bulk_score(mol) = Σ max(1, heavy_degree − 1)²` over non-ring heavy atoms, capped at 15. The cyclic side uses `pt_steps = max(100, mc_steps // 4) + 12 * bulk`, `mc_steps = mc_steps_arg + 18 * bulk`; the acyclic side uses `pt_steps = 150 + 12 * bulk`, `mc_steps = 200 + 18 * bulk` — close enough that neither pool gets a systematic sampling advantage that would bias `cyclic_E − acyclic_E`. PT runs with four replicas at (300, 500, 1000, 2000) K. Initial ETKDG conformer count is also adaptive: `30 + 20 * rotatable_bonds + 15 * bulk`, clamped to `[20, max(n_conformers, 1500)]`. Heavy-atom pre-screening uses `_VDW_CLASH_THRESHOLD_A = 1.3` Å before each MMFF call inside MC/PT steps, rejecting moves that drove non-bonded atoms into covalent-bond range.

## Calibration coefficients

`ring_strain/calibration_coefficients.json` is generated by `scripts/derive_calibration.py` and committed to the repo so user-facing init is sub-millisecond. The JSON carries both the fitted `(a, b)` per ring size and the raw `(raw, exp)` data points used to fit them. Regenerate it whenever `ring_strain/reference.py` changes. The script takes ~10–15 minutes because it runs the full production pipeline on every reference (including the 14 substituted compounds, each of which costs 30–90 s).

The `StrainCalibrator` load order:
1. Class-level cache from a previous call in this process.
2. JSON file (the production path).
3. Live derivation from the **unsubstituted** references only — fast (~1 s) but produces only a multiplicative factor per size. This is the fallback when the JSON is missing or corrupt; it intentionally skips substituted references because they take minutes each.

## Conventions specific to this repo

- `StrainReport` carries both raw MMFF94 strain (`total_strain_mmff_kcal_mol`) and the post-calibration value (`total_strain_calibrated_kcal_mol`) — when comparing to literature, always use the calibrated value.
- `MMFFGetMoleculeForceField` calls are wrapped in `_suppress_cpp_stderr()` in `mmff.py` because RDKit's C++ BFGS optimizer writes "Invariant Violation" messages directly to fd 2, bypassing Python logging.
- Energies are kcal/mol throughout; angles are degrees; distances are Ångström.
- `ring_strain/calibration_coefficients.json` is generated, not hand-edited — regenerate via `scripts/derive_calibration.py` whenever the reference set changes.
