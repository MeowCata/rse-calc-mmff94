"""Held-out generalization test: rings NOT in the reference database.

The goal of the project is to predict strain for synthesizable monocyclic
saturated carbocycles that we have not put into the reference DB. This
script tests that ability on a curated held-out set with literature
strain values, after the 5 new mono-substituted anchors were added
(methylcyclopropane / methylcyclobutane / methylcyclopentane /
methylcyclohexane / trans-1,4-dimethylcyclohexane).

Run after `python scripts/derive_calibration.py` has been re-executed.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from ring_strain.core import StrainAnalyzer


# (name, smiles, lit_strain_kcal_mol, source, tolerance)
HELD_OUT: List[Tuple[str, str, float, str, float]] = [
    # Mono-substituted ethyl / propyl / isopropyl rings — test generalization
    # from the methyl-anchored calibration to other alkyl groups.
    ("ethylcyclopropane",        "CCC1CC1",       28.1, "NIST",  2.5),
    ("ethylcyclopentane",        "CCC1CCCC1",      6.0, "NIST",  2.5),
    ("ethylcyclohexane",         "CCC1CCCCC1",     1.4, "NIST",  2.5),
    ("isopropylcyclohexane",     "CC(C)C1CCCCC1",  2.1, "NIST",  2.5),
    ("n-propylcyclohexane",      "CCCC1CCCCC1",    1.4, "NIST",  2.5),
    # Stereo test: cis-1,4-dimethylcyclohexane has one axial methyl,
    # +1.87 kcal/mol vs trans (A-value). Tests stereo discrimination on
    # a different position than the cyclohexane gap previously verified.
    ("cis-1,4-dimethylcyclohexane",
     "C[C@H]1CC[C@@H](C)CC1",                      3.8, "NIST + Eliel A-value", 2.5),
    # Stereo on a different ring size: 1,2-dimethylcyclopentane cis vs trans.
    ("trans-1,2-dimethylcyclopentane",
     "C[C@H]1CCC[C@H]1C",                          6.5, "NIST",  2.5),
    ("cis-1,2-dimethylcyclopentane",
     "C[C@H]1CCC[C@@H]1C",                         7.1, "NIST + Eliel",  2.5),
]


# Stereo pair: |Δ_cyclic_gap| should match literature axial-methyl A-value
# (~1.87 kcal/mol).
STEREO_PAIR = ("1,4-dimethylcyclohexane bare", "CC1CCC(C)CC1", 1.87, 1.0)


def main() -> None:
    analyzer = StrainAnalyzer()
    print("=" * 80)
    print("HELD-OUT GENERALIZATION TEST")
    print("Rings NOT in reference DB — tests prediction of unknown carbocycles")
    print("=" * 80)

    n_pass = 0
    n_total = 0
    max_err = 0.0
    for name, smi, lit, src, tol in HELD_OUT:
        t0 = time.perf_counter()
        try:
            rep = analyzer.analyze(smi)
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
            continue
        dt = time.perf_counter() - t0
        if not rep.is_supported:
            print(f"  [SKIP] {name}: pipeline rejected ({rep.validation_message})")
            continue
        n_total += 1
        model = rep.total_strain_calibrated_kcal_mol
        err = model - lit
        ok = abs(err) <= tol
        n_pass += ok
        max_err = max(max_err, abs(err))
        flag = "OK " if ok else "FAIL"
        print(
            f"  [{flag}] {name:<32} model={model:+6.2f}  lit={lit:+6.2f}  "
            f"err={err:+.2f}  tol={tol:.1f}  ({dt:.0f}s)"
        )
        print(f"         source: {src}")

    print("\n" + "=" * 80)
    print("STEREO GAP TEST (independent of calibration)")
    print("=" * 80)
    name, smi, lit, tol = STEREO_PAIR
    try:
        rep = analyzer.analyze(smi)
        gap = rep.stereoisomer_cyclic_gap_kcal_mol
        if gap is None:
            print(f"  [SKIP] {name}: enumeration did not trigger")
        else:
            err = gap - lit
            ok = abs(err) <= tol
            flag = "OK " if ok else "FAIL"
            print(
                f"  [{flag}] {name:<32} gap={gap:.2f}  lit={lit:.2f}  "
                f"err={err:+.2f}  tol={tol:.1f}"
            )
    except Exception as exc:
        print(f"  [ERROR] {name}: {exc}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Held-out absolute strain: {n_pass}/{n_total} pass  worst |err|={max_err:.2f}")


if __name__ == "__main__":
    main()
