"""
Example usage of the ring strain analysis system.

Demonstrates:
- Basic single-molecule analysis
- Cycloalkane series
- Substituted cycloalkanes (steric effects)
- Comparing molecules
- Handling "Not Supported" inputs
"""

from ring_strain.core import StrainAnalyzer


def main():
    analyzer = StrainAnalyzer()

    # Example 1: Single molecule analysis
    print("=" * 60)
    print("Example 1: Cyclopropane")
    print("=" * 60)
    report = analyzer.analyze("C1CC1")
    print(report)
    print()

    # Example 2: Cycloalkane series
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

    # Example 3: Substituted cycloalkanes (steric effects)
    print("=" * 60)
    print("Example 3: Steric Hindrance Effects")
    print("=" * 60)
    substituted = [
        ("C1CC1", "cyclopropane (unsubstituted)"),
        ("CC1CC1C", "1,2-dimethylcyclopropane"),
        ("CC(C)(C)C1CC1C(C)(C)C", "1,2-di-tert-butylcyclopropane"),
    ]
    for smi, name in substituted:
        try:
            r = analyzer.analyze(smi)
            sub_str = " (substituted)" if r.is_substituted else ""
            mc_str = " [MC]" if r.monte_carlo_used else ""
            print(
                f"  {name:<40} "
                f"strain={r.total_strain_calibrated_kcal_mol:>5.1f}, "
                f"score={r.stability_score:>5.1f}"
                f"{sub_str}{mc_str}"
            )
        except Exception as exc:
            print(f"  {name:<40} FAILED: {exc}")
    print()

    # Example 4: "Not Supported" inputs
    print("=" * 60)
    print("Example 4: Not Supported Inputs")
    print("=" * 60)
    unsupported = [
        ("c1ccccc1", "benzene (aromatic)"),
        ("C1CC2CCC1C2", "norbornane (polycyclic)"),
        ("C1=CCCCC1", "cyclohexene (unsaturated)"),
        ("C1CCOC1", "THF (heterocyclic)"),
    ]
    for smi, name in unsupported:
        r = analyzer.analyze(smi)
        print(
            f"  {name:<35}  "
            f"supported={r.is_supported}, "
            f"message='{r.validation_message}'"
        )
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
    print(f"  Supported: {r.is_supported}")
    print(f"  Message:   {r.validation_message}")
    print()


if __name__ == "__main__":
    main()
