"""Real-world ring strain validation: cis/trans pairs + monosubstituted absolutes.

Runs the full StrainAnalyzer pipeline on a curated set of monocyclic
saturated carbocycles with well-established literature strain values.
Goal: decide whether the model is accurate enough on *synthesizable*
rings to pivot scope away from extreme-crowding theoretical-only
systems (see issue.txt + plans/humming-wobbling-porcupine.md).

Two sections:
  A) cis/trans diastereomer pairs — compare |delta_cis-trans| against
     literature. Input is the bare SMILES (no @ markers) so the
     existing stereo enumeration path in core._maybe_enumerate_stereo
     is exercised and (lo, hi) calibrated-strain range is harvested.
  B) Monosubstituted real-world rings — compare absolute calibrated
     strain against literature.

Run:
    python scripts/validate_real_world.py
"""

from __future__ import annotations

import time
from typing import List, Tuple

from ring_strain.core import StrainAnalyzer


# --------------------------------------------------------------------
# Phase A: cis/trans pairs (bare SMILES; analyzer enumerates both)
# Literature |delta| = |E_cis - E_trans| in kcal/mol.
# --------------------------------------------------------------------
# Sources:
#   Eliel & Wilen, "Stereochemistry of Organic Compounds" (1994):
#     1,2/1,3/1,4-dimethylcyclohexane delta = 1.87 kcal/mol
#       (one methyl axial vs both equatorial; A-value of CH3)
#   Schleyer et al., JACS 92, 2377 (1970):
#     1,2-dimethylcyclopropane delta = 1.05 kcal/mol (cis higher)
CIS_TRANS_PAIRS: List[Tuple[str, str, float, str]] = [
    # (name, bare_smiles, |delta| kcal/mol, source)
    ("1,2-dimethylcyclohexane",  "CC1CCCCC1C",   1.87, "Eliel 1994 (A-value methyl)"),
    ("1,3-dimethylcyclohexane",  "CC1CCCC(C)C1", 1.87, "Eliel 1994"),
    ("1,4-dimethylcyclohexane",  "CC1CCC(C)CC1", 1.87, "Eliel 1994"),
    ("1,2-dimethylcyclopropane", "CC1CC1C",      1.05, "Schleyer JACS 92, 2377 (1970)"),
]

# --------------------------------------------------------------------
# Phase B: monosubstituted real-world rings — compare absolute strain.
# Sources: NIST WebBook + Wiberg group additivity.
# --------------------------------------------------------------------
MONO_REFERENCES: List[Tuple[str, str, float, str]] = [
    # (name, smiles, lit strain kcal/mol, source)
    ("methylcyclopropane",   "CC1CC1",        26.5, "NIST + Wiberg group additivity"),
    ("methylcyclobutane",    "CC1CCC1",       25.8, "NIST"),
    ("methylcyclopentane",   "CC1CCCC1",       6.4, "NIST"),
    ("methylcyclohexane",    "CC1CCCCC1",      1.4, "NIST"),
    ("ethylcyclohexane",     "CCC1CCCCC1",     1.4, "NIST"),
    ("isopropylcyclohexane", "CC(C)C1CCCCC1",  2.1, "NIST"),
]


TOL_DELTA = 1.0       # cis/trans delta tolerance (kcal/mol)
TOL_ABSOLUTE = 2.5    # monosubstituted absolute tolerance (kcal/mol)


def run() -> None:
    analyzer = StrainAnalyzer()
    print("\n" + "=" * 80)
    print("PHASE A: cis/trans delta-strain validation")
    print("(bare SMILES; analyzer auto-enumerates both diastereomers)")
    print("=" * 80)
    a_results: List[Tuple[str, float, float, float, bool]] = []
    for name, smiles, lit_delta, source in CIS_TRANS_PAIRS:
        t0 = time.perf_counter()
        rep = analyzer.analyze(smiles)
        dt = time.perf_counter() - t0
        if rep.strain_range_kcal_mol is None:
            print(f"  [SKIP] {name}: enumeration did NOT trigger")
            continue
        lo, hi = rep.strain_range_kcal_mol
        model_delta = hi - lo
        err = model_delta - lit_delta
        pass_ = abs(err) <= TOL_DELTA
        a_results.append((name, model_delta, lit_delta, err, pass_))
        flag = "OK " if pass_ else "FAIL"
        print(
            f"  [{flag}] {name:<26} model |d|={model_delta:5.2f}  "
            f"lit |d|={lit_delta:5.2f}  err={err:+.2f}  ({dt:.0f}s)"
        )
        print(f"          source: {source}")
        print(f"          range:  ({lo:+.2f}, {hi:+.2f}) kcal/mol")

    print("\n" + "=" * 80)
    print("PHASE B: monosubstituted absolute strain validation")
    print("=" * 80)
    b_results: List[Tuple[str, float, float, float, bool]] = []
    for name, smiles, lit, source in MONO_REFERENCES:
        t0 = time.perf_counter()
        rep = analyzer.analyze(smiles)
        dt = time.perf_counter() - t0
        if not rep.is_supported:
            print(f"  [SKIP] {name}: pipeline rejected ({rep.validation_message})")
            continue
        model = rep.total_strain_calibrated_kcal_mol
        err = model - lit
        pass_ = abs(err) <= TOL_ABSOLUTE
        b_results.append((name, model, lit, err, pass_))
        flag = "OK " if pass_ else "FAIL"
        print(
            f"  [{flag}] {name:<22} model={model:+6.2f}  lit={lit:+6.2f}  "
            f"err={err:+.2f}  ({dt:.0f}s)"
        )
        print(f"          source: {source}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    a_pass = sum(1 for *_, p in a_results if p)
    b_pass = sum(1 for *_, p in b_results if p)
    a_max = max((abs(e) for *_, e, _ in a_results), default=0.0)
    b_max = max((abs(e) for *_, e, _ in b_results), default=0.0)
    print(f"  Phase A (delta, tol={TOL_DELTA} kcal/mol): {a_pass}/{len(a_results)} pass, worst |err|={a_max:.2f}")
    print(f"  Phase B (abs,   tol={TOL_ABSOLUTE} kcal/mol): {b_pass}/{len(b_results)} pass, worst |err|={b_max:.2f}")
    all_a = a_pass == len(a_results) and len(a_results) > 0
    all_b = b_pass == len(b_results) and len(b_results) > 0
    decision = (
        "PIVOT — model reliable on real-world synthesizable rings"
        if (all_a and all_b)
        else "INVESTIGATE — at least one compound outside tolerance"
    )
    print(f"\n  Decision: {decision}\n")


if __name__ == "__main__":
    run()
