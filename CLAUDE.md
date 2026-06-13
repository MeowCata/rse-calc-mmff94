# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope

MMFF94-based ring-strain energy quantification for **monocyclic saturated carbocycles only**. Inputs that are polycyclic (bridged / fused / spiro), heterocyclic, aromatic, or unsaturated are rejected with `is_supported=False` and a "Not Supported" message — do not extend the analysis pipeline to other ring systems without first consulting `instructions.md`, which fixes this scope.

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

CLI knobs that materially change results: `--n-conformers` (default 200), `--mc-steps` (default 500), `--no-monte-carlo` (disables MC/PT for substituted rings — see "Conformer sampling contract" below).

## Tests

```bash
python -m pytest tests/                                  # full suite (~5 min)
python -m pytest tests/test_regression.py                # core cycloalkane regression
python -m pytest tests/test_bulky_regression.py          # tert-butyl / gem-dimethyl wiring tests (fast)
RUN_BENCHMARKS=1 python -m pytest tests/test_bulky_benchmark.py   # 8-compound bulky accuracy benchmark (~7 min)
python -m pytest tests/test_regression.py::TestCycloalkaneStrain::test_cycloalkane_strain
```

`tests/test_bulky_benchmark.py` is skipped by default; set `RUN_BENCHMARKS=1` to run it. The core cycloalkane regression uses a ±5 kcal/mol tolerance vs literature; the bulky benchmark uses ±2.5 kcal/mol and currently fails on the 1,3-di-tert-butyl-cyclohexane cis/trans pair — that residual is a known MMFF94 limitation (the force field inverts the experimental ranking on that pair).

## Architecture

`StrainAnalyzer.analyze(smiles)` in `ring_strain/core.py` is the single entry point; everything else is a collaborator it instantiates. The pipeline is:

1. **Parse + stereo enumeration** (`core._maybe_enumerate_stereo`) — if the SMILES has ≥2 unassigned ring stereocenters (e.g. 1,3-di-tert-butyl-cyclohexane without `@` markers), `EnumerateStereoisomers` is invoked, `analyze` recurses on each canonical isomer, and the lowest-strain report is returned with `strain_range_kcal_mol = (min, max)` and `stereoisomers_analyzed` attached. The recursion terminates because enumerated isomers have stereo fully assigned.
2. **Ring validation** (`ring_analysis.RingAnalyzer.validate`) — rejects everything outside monocyclic saturated carbocycles.
3. **Cyclic conformer search** (`mmff.MMFFCalculator`) — ETKDG embed + MMFF94 optimize all conformers in parallel. For substituted 4–7 rings, `seed_ring_pucker_conformers` then adds extra random-coords ETKDG seeds (`useRandomCoords=True`, `clearConfs=False`) so chair / twist-boat / envelope / half-chair basins are all sampled. PT + MC torsion search follows, with step counts and conformer count scaling on `compute_bulk_score(mol)`. Cluster + Boltzmann-average the surviving conformers when `use_boltzmann=True`.
4. **Strain calculation** (`homodesmotic.HomodesmoticAnalyzer.compute_strain`) — two regimes:
   - **Unsubstituted cycloalkanes**: strict bond-balanced reaction `cyclo-(CH₂)ₙ + CH₃CH₃ → CH₃(CH₂)ₙ₊₁CH₃`. Per-ring-size strain is cached at `_CYCLOALKANE_STRAIN_CACHE`.
   - **Substituted**: ring opening at every single ring bond, with canonical-SMILES deduplication so symmetric rings don't pay for equivalent openings. Each unique candidate runs through the full embed + MMFF + (bulk-scaled) PT/MC + Boltzmann pipeline; the lowest-energy acyclic reference is kept.
5. **Per-term decomposition** — `MMFFCalculator.decompose_energy` toggles `SetMMFFXxxTerm(False)` to extract `{bond, angle, stretch_bend, oop, torsion, vdw, electrostatic}` for both cyclic and acyclic structures; the cyclic-minus-acyclic deltas populate `vdw_strain_kcal_mol / torsion_strain_kcal_mol / angle_strain_kcal_mol / bond_strain_kcal_mol` on `StrainReport`. **The vdW delta is the physically meaningful "true steric" strain** and is what `steric_confinement_kcal_mol` now reports.
6. **Calibration** (`calibrate.StrainCalibrator`) — per-ring-size linear correction `calibrated = a[size] * raw + b[size]`, loaded from `ring_strain/calibration_coefficients.json` for instant init. For ring sizes whose unsubstituted reference defines a strain-free anchor (notably cyclohexane = 0), the fit is constrained to pass exactly through `(raw_anchor, 0)`. Sizes with a single reference fall back to the historical multiplicative form. To regenerate after editing the reference set, run `python scripts/derive_calibration.py`.
7. **Scoring / reference match** — `scoring.StabilityScorer` maps calibrated strain to a 0–100 score (default threshold 8 kcal/mol); `reference.ReferenceDatabase` is queried by canonical SMILES.

Geometry diagnostics (`geometry.GeometryAnalyzer`) run alongside the energy path and contribute Baeyer (angle RMS), Pitzer (eclipsed-torsion count), and transannular-contact fields on the report — they are diagnostic snapshots from the single best conformer and **are not guaranteed to sum to `total_strain_mmff_kcal_mol`** when Boltzmann averaging is active. The same caveat applies to the per-term decomposition.

## Conformer sampling contract (critical)

For **substituted** rings, the Monte Carlo + parallel-tempering torsion search in `mmff.monte_carlo_search` / `parallel_tempering_search` must remain on the critical path. Single-conformer ETKDG produces chemically wrong results (e.g. tert-butyl-cyclopropane reports the same strain as cyclopropane because substituent rotamers are never explored). When fixing a bug in that path:

- Only repair the failing line; do not "simplify" by deleting the MC/PT call.
- The flow `parallel_tempering_search → cluster_conformers → compute_boltzmann_energy` is the validated chain inside `homodesmotic._compute_acyclic_energy_with_mol`. New requirements should add a bypass switch, not replace the default.
- Conformer IDs are not guaranteed to be sequential — iterate `[c.GetId() for c in mol.GetConformers()]` rather than `range(mol.GetNumConformers())`, otherwise RDKit raises "Bad Conformer Id" once a prior PT pass has removed and re-added conformers.
- If a substituted ring reports `~0 kcal/mol` strain, suspect the sampling before the force field.

Step counts and PT temperature ladders scale with `compute_bulk_score(mol) = Σ max(1, heavy_degree − 1)²` over non-ring heavy atoms, capped at 15. The cyclic side uses `pt_steps = 100 + 12 * bulk`, `mc_steps = mc_steps_arg + 18 * bulk`; the acyclic side mirrors this so neither pool gets a sampling advantage that would bias `cyclic_E − acyclic_E`. PT runs with four replicas at (300, 500, 1000, 2000) K. Heavy-atom pre-screening uses `_VDW_CLASH_THRESHOLD_A = 1.3` Å before each MMFF call inside MC/PT steps, rejecting moves that drove non-bonded atoms into covalent-bond range.

## Calibration coefficients

`ring_strain/calibration_coefficients.json` is generated by `scripts/derive_calibration.py` and committed to the repo so user-facing init is sub-millisecond. The JSON carries both the fitted `(a, b)` per ring size and the raw `(raw, exp)` data points used to fit them. Regenerate it whenever `ring_strain/reference.py` changes. The script takes ~10–15 minutes because it runs the full production pipeline on every reference (including the 8 substituted compounds, each of which costs 30–90 s).

The `StrainCalibrator` load order:
1. Class-level cache from a previous call in this process.
2. JSON file (the production path).
3. Live derivation from the **unsubstituted** references only — fast (~1 s) but produces only a multiplicative factor per size. This is the fallback when the JSON is missing or corrupt; it intentionally skips substituted references because they take minutes each.

## Conventions specific to this repo

- `StrainReport` carries both raw MMFF94 strain (`total_strain_mmff_kcal_mol`) and the post-calibration value (`total_strain_calibrated_kcal_mol`) — when comparing to literature, always use the calibrated value.
- `MMFFGetMoleculeForceField` calls are wrapped in `_suppress_cpp_stderr()` in `mmff.py` because RDKit's C++ BFGS optimizer writes "Invariant Violation" messages directly to fd 2, bypassing Python logging.
- Energies are kcal/mol throughout; angles are degrees; distances are Ångström.
- `ring_strain/calibration_coefficients.json` is generated, not hand-edited — regenerate via `scripts/derive_calibration.py` whenever the reference set changes.
