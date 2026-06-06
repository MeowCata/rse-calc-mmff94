"""
Example usage of the ring strain analysis system.

Demonstrates:
- Basic single-molecule analysis
- Batch processing
- Comparing molecules
- Listing reference compounds
"""

from mmff94.ring_strain.core import StrainAnalyzer


def main():
    # Initialize analyzer (uses default settings)
    analyzer = StrainAnalyzer()

    # Example 1: Single molecule analysis
    print("=" * 60)
    print("Example 1: Cyclopropane")
    print("=" * 60)
    report = analyzer.analyze("C1CC1")
    print(report)
    print()

    # Example 2: Compare cycloalkanes
    print("=" * 60)
    print("Example 2: Cycloalkane Series")
    print("=" * 60)
    cycloalkanes = [
        "C1CC1",       # 3
        "C1CCC1",      # 4
        "C1CCCC1",     # 5
        "C1CCCCC1",    # 6
        "C1CCCCCC1",   # 7
        "C1CCCCCCC1",  # 8
    ]
    for smi in cycloalkanes:
        r = analyzer.analyze(smi)
        ref = analyzer.get_reference_value(smi)
        ref_str = f"(lit: {ref:.1f})" if ref else ""
        print(
            f"  {smi:<18}  "
            f"strain = {r.total_strain_calibrated_kcal_mol:>5.1f} kcal/mol, "
            f"score = {r.stability_score:>5.1f}  {ref_str}"
        )
    print()

    # Example 3: Aromatic and heterocyclic
    print("=" * 60)
    print("Example 3: Aromatic & Heterocyclic Rings")
    print("=" * 60)
    aromatics = [
        "c1ccccc1",     # benzene
        "c1ccncc1",     # pyridine
        "c1ccc2ccccc2c1",  # naphthalene
        "C1CCOC1",      # tetrahydrofuran
    ]
    for smi in aromatics:
        r = analyzer.analyze(smi)
        print(
            f"  {r.canonical_smiles:<30}  "
            f"rings={r.num_rings}, "
            f"strain={r.total_strain_calibrated_kcal_mol:+.2f}, "
            f"score={r.stability_score:.1f} ({r.stability_category})"
        )
    print()

    # Example 4: Polycyclic compounds
    print("=" * 60)
    print("Example 4: Polycyclic Compounds")
    print("=" * 60)
    polycyclics = [
        ("C1CC2CCC1C2", "norbornane"),
        ("C1C2CC3CC1CC(C2)C3", "adamantane"),
        ("C1CCC2CCCCC2C1", "trans-decalin"),
    ]
    for smi, name in polycyclics:
        try:
            r = analyzer.analyze(smi)
            print(
                f"  {name:<18}  "
                f"{r.ring_system_type:<30}  "
                f"strain={r.total_strain_calibrated_kcal_mol:.1f}, "
                f"score={r.stability_score:.1f}"
            )
        except Exception as exc:
            print(f"  {name:<18}  FAILED: {exc}")
    print()

    # Example 5: Compare two molecules
    print("=" * 60)
    print("Example 5: Compare Cyclopropane vs Cyclohexane")
    print("=" * 60)
    report_a, report_b, verdict = analyzer.compare("C1CC1", "C1CCCCC1")
    print(verdict)
    print()

    # Example 6: Acyclic molecule
    print("=" * 60)
    print("Example 6: Acyclic molecule (n-hexane)")
    print("=" * 60)
    r = analyzer.analyze("CCCCCC")
    print(f"  Category: {r.stability_category}")
    print(f"  Strain:   {r.total_strain_calibrated_kcal_mol} kcal/mol")
    print()


if __name__ == "__main__":
    main()
