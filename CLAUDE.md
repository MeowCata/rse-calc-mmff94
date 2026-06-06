# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MMFF94 ring strain energy quantification system (`mmff94` package). Computes ring strain energy for organic molecules from SMILES strings using RDKit's MMFF94 force field with homodesmotic reactions, calibration to experimental scale, and 0-100 stability scoring.

## Commands

```bash
# Run tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_regression.py::TestCycloalkaneStrain -v

# Analyze a single molecule via CLI
python -m cli.main "C1CC1"

# CLI with JSON output
python -m main "C1CC1" --json

# CLI benchmark against reference compounds
python -m cli.main --benchmark

# List reference compounds
python -m cli.main --list-refs

# Run examples
python -m examples.basic_usage
```

Python version: 3.10+. Key dependencies: `rdkit`, `numpy`, `scipy`, `pytest`.

## Architecture

### Main API entry point

`StrainAnalyzer.analyze(smiles) -> StrainReport` in `ring_strain/core.py`. This is the only public API users call.

### Analysis pipeline (in order)

1. **Ring detection** (`ring_analysis.py`) — RDKit SSSR + custom classification. Detects isolated, fused, spiro, bridged, and cage ring systems. Returns `RingSystemInfo` with per-ring `RingInfo`.
2. **MMFF94 energy** (`mmff.py`) — Conformer generation (ETKDG, multi-conformer sampling), MMFF94 force field optimization, energy calculation. 200 conformers by default. Suppresses C++ stderr noise from BFGS optimizer.
3. **Homodesmotic strain** (`homodesmotic.py`) — Computes raw MMFF94 strain via balanced reactions. For simple cycloalkanes (C3-C8 carbocycles): strict bond-balanced `cyclo-(CH2)n + C2H6 -> CH3-(CH2)(n+1)-CH3`. For other systems: ring-opening + H-capping. Aromatic systems: bypassed (treated as zero homodesmotic strain).
4. **Calibration** (`calibrate.py`) — MMFF94 raw values are systematically biased vs. experimental scale. Per-ring-size correction factors computed from reference compounds. Polycyclic systems skip per-ring-size calibration because the ring-opening method already produces values closer to experimental scale.
5. **Scoring** (`scoring.py`) — `score = 100 * exp(-strain / threshold)` with default threshold 8.0 kcal/mol. Categories: highly strained (0-10), significant strain (10-30), moderate strain (30-60), low strain (60-90), strain-free (90-100).
6. **Reference comparison** (`reference.py`) — 25 built-in reference compounds with literature strain values (cycloalkanes, polycyclics, heterocycles, aromatics). Matched by canonical SMILES first, then by ring sizes.

### Key design decisions

- **Polycyclic calibration**: Polycyclic systems (fused, bridged, spiro, cage) do NOT use per-ring-size calibration factors derived from monocyclic compounds. The ring-opening homodesmotic method already produces values closer to experimental scale for these (e.g., norbornane raw=14.3 vs exp=15.0).
- **SSSR ring detection**: Uses `Chem.GetSymmSSR` (Smallest Set of Smallest Rings), not all possible cycles. This means some rings in cage compounds may not be individually enumerated — cubane returns 6 rings (the 4-membered faces), not all possible cycles.
- **Aromatic bypass**: Fully aromatic ring systems skip the ring-opening homodesmotic step (cannot break aromatic bonds meaningfully) and are assigned zero homodesmotic strain.
- **Conformer strategy**: Three-tier fallback for conformer generation — ETKDG with torsion preferences → ETKDG without torsion preferences → random coordinates. MMFF94 optimization picks the lowest-energy conformer.
- **C++ stderr suppression**: RDKit's BFGS optimizer can emit "Invariant Violation" directly to stderr from C++. `_suppress_cpp_stderr()` in `mmff.py` redirects fd 2 during optimization. This is non-fatal.

### Module responsibilities

| Module | Responsibility |
|--------|---------------|
| `core.py` | Orchestrator, `StrainAnalyzer`, `StrainReport` dataclass |
| `mmff.py` | MMFF94 conformer generation, optimization, energy calculation |
| `ring_analysis.py` | Ring detection (SSSR), fusion classification, ring system grouping |
| `homodesmotic.py` | Homodesmotic reaction construction, acyclic reference building |
| `calibrate.py` | Per-ring-size calibration factors, interpolation for non-standard sizes |
| `scoring.py` | Exponential 0-100 stability score and qualitative categories |
| `reference.py` | 25 reference compounds with literature strain values |
| `geometry.py` | Baeyer (angle), Pitzer (torsional), and transannular strain decomposition |
| `cli/main.py` | CLI with single, batch, benchmark, and list-refs modes |
