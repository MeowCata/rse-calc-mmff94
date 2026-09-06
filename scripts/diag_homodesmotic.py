"""Diagnostic: trace homodesmotic ring-opening for cis/trans dimethylcyclohexane.

For each explicit-stereo cis and trans diastereomer, prints:
  - raw cyclic energy (MMFF94 optimized + MC/PT + Boltzmann)
  - chosen acyclic reference: canonical SMILES + energy
  - raw strain (cyclic - acyclic)
  - calibrated strain
  - per-isomer Delta (cis - trans) on each metric

This is to test the hypothesis that cis and trans share the same
acyclic reference (n-octane / 2-methylheptane etc.), so any
cis/trans gap in the strain should equal the cyclic energy gap.
"""

from __future__ import annotations

from typing import List, Tuple

from rdkit import Chem

from ring_strain.core import StrainAnalyzer
from ring_strain.homodesmotic import HomodesmoticAnalyzer
from ring_strain.mmff import MMFFCalculator
from ring_strain.ring_analysis import RingAnalyzer


# Explicit-stereo SMILES so analyze() does NOT trigger enumeration.
# We supply both members of each pair separately and compute Delta manually.
PAIRS: List[Tuple[str, str, str, float]] = [
    # (label, cis_smiles, trans_smiles, lit_delta_kcal_mol)
    ("1,2-dimethylCy", "C[C@H]1CCCC[C@@H]1C", "C[C@H]1CCCC[C@H]1C", 1.87),
    ("1,3-dimethylCy", "C[C@H]1CCC[C@@H](C)C1", "C[C@H]1CCC[C@H](C)C1", 1.87),
    ("1,4-dimethylCy", "C[C@H]1CC[C@@H](C)CC1", "C[C@H]1CC[C@H](C)CC1", 1.87),
]


def trace(analyzer: StrainAnalyzer, label: str, smiles: str) -> dict:
    """Run analyze() and harvest energy components from the homodesmotic step."""
    rep = analyzer.analyze(smiles)
    if not rep.is_supported:
        return {"label": label, "smiles": smiles, "supported": False}

    # Re-derive the acyclic reference directly so we can inspect its SMILES.
    mol = Chem.MolFromSmiles(smiles)
    ring_info = RingAnalyzer(mol).validate()[2]
    homo = HomodesmoticAnalyzer(analyzer.mmff_calc)
    # Build all ring-opening candidates and report the one chosen as best.
    candidates = homo._build_all_acyclic_references(Chem.Mol(mol), ring_info)
    candidate_smiles = [Chem.MolToSmiles(m, canonical=True) for m, _ in candidates]

    return {
        "label": label,
        "smiles": smiles,
        "supported": True,
        "raw_strain": rep.total_strain_mmff_kcal_mol,
        "cal_strain": rep.total_strain_calibrated_kcal_mol,
        "candidate_acyclic_smiles": candidate_smiles,
    }


def main():
    analyzer = StrainAnalyzer()
    print("\n" + "=" * 80)
    print("HOMODESMOTIC DIAGNOSTIC: cis vs trans 1,2-/1,3-/1,4-dimethylcyclohexane")
    print("=" * 80)

    for label, cis_smi, trans_smi, lit_delta in PAIRS:
        cis = trace(analyzer, f"cis-{label}", cis_smi)
        trans = trace(analyzer, f"trans-{label}", trans_smi)
        print(f"\n--- {label} (lit |Delta| = {lit_delta:.2f} kcal/mol) ---")
        if not (cis.get("supported") and trans.get("supported")):
            print(f"  Pipeline rejection: cis={cis}, trans={trans}")
            continue
        for tag, data in (("cis", cis), ("trans", trans)):
            print(f"  {tag:<5} input={data['smiles']}")
            print(f"        raw_strain={data['raw_strain']:+.3f}  cal_strain={data['cal_strain']:+.3f}")
            print(f"        acyclic candidates (dedup'd): {len(data['candidate_acyclic_smiles'])}")
            for s in data["candidate_acyclic_smiles"]:
                print(f"          - {s}")
        d_raw = cis["raw_strain"] - trans["raw_strain"]
        d_cal = cis["cal_strain"] - trans["cal_strain"]
        # Calibration: cal = a*raw + b  =>  Delta_cal = a * Delta_raw
        # Implied a from observed deltas (sanity check vs JSON a[6]=0.292):
        implied_a = d_cal / d_raw if abs(d_raw) > 1e-3 else float("nan")
        print(f"  Delta_raw (cis - trans) = {d_raw:+.3f} kcal/mol")
        print(f"  Delta_cal (cis - trans) = {d_cal:+.3f} kcal/mol")
        print(f"  Implied slope a from deltas: {implied_a:.3f}  (JSON a[6]=0.292)")
        print(f"  Lit |Delta|: {lit_delta:.2f} kcal/mol")
        print(f"  Raw-vs-lit residual:  {abs(d_raw) - lit_delta:+.3f} kcal/mol")
        print(f"  Cal-vs-lit residual:  {abs(d_cal) - lit_delta:+.3f} kcal/mol")


if __name__ == "__main__":
    main()
