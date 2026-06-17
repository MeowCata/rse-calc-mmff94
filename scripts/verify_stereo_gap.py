"""Verify the new cis/trans cyclic-energy gap metric.

Runs analyze() on bare-SMILES dimethylcyclohexanes (which trigger
stereo enumeration). For each, reads the new
stereoisomer_cyclic_gap_kcal_mol field and compares it against the
literature |Delta| value.

Expected post-fix:
  1,2-dimethylCy gap ~0.9 kcal/mol (lit ~0.9)
  1,3-dimethylCy gap ~1.2 kcal/mol (lit 1.87; MMFF94 underestimates)
  1,4-dimethylCy gap ~1.3 kcal/mol (lit 1.87; MMFF94 underestimates)

The gap should be much larger than the calibrated strain_range_kcal_mol
spread, which is squashed by a[6]=0.292.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from ring_strain.core import StrainAnalyzer


CASES: List[Tuple[str, str, float]] = [
    # (label, bare_smiles_no_stereo, lit_delta_kcal_mol)
    ("1,2-dimethylcyclohexane", "CC1CCCCC1C",   0.9),
    ("1,3-dimethylcyclohexane", "CC1CCCC(C)C1", 1.87),
    ("1,4-dimethylcyclohexane", "CC1CCC(C)CC1", 1.87),
]


def main() -> None:
    analyzer = StrainAnalyzer()
    print("=" * 80)
    print("STEREO GAP VERIFICATION: cyclic-energy delta as cis/trans highlight")
    print("=" * 80)
    for label, smi, lit in CASES:
        t0 = time.perf_counter()
        rep = analyzer.analyze(smi)
        dt = time.perf_counter() - t0
        cal_range = rep.strain_range_kcal_mol
        cal_spread = (cal_range[1] - cal_range[0]) if cal_range else None
        gap = rep.stereoisomer_cyclic_gap_kcal_mol
        bd = rep.stereoisomer_breakdown or []

        print(f"\n--- {label} (lit |Delta| = {lit:.2f} kcal/mol)  [{dt:.0f}s] ---")
        print(f"  Calibrated strain range:    {cal_range}")
        print(f"  Calibrated cis/trans spread: "
              f"{cal_spread:.3f}" if cal_spread is not None else "N/A")
        print(f"  Cyclic-energy gap (NEW):     {gap}")
        if gap is not None:
            err = gap - lit
            flag = "OK " if abs(err) <= 1.0 else "FAIL"
            print(f"  [{flag}] gap_err vs lit: {err:+.2f} kcal/mol  (tol 1.0)")
        print(f"  Per-isomer breakdown ({len(bd)} isomers):")
        for entry in bd:
            mark = "*" if entry.get("is_min") else " "
            ce = entry.get("cyclic_energy")
            ce_str = f"  cyclic_E={ce:+7.3f}" if ce is not None else ""
            print(f"    {mark} {entry['smiles']:<42} "
                  f"cal={entry['strain_calibrated']:+.3f}"
                  f"  raw={entry['strain_mmff']:+.3f}{ce_str}")


if __name__ == "__main__":
    main()
