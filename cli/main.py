"""
Command-line interface for ring strain analysis.

Usage:
    python -m cli.main "C1CC1"                     # cyclopropane
    python -m cli.main "C1CC1" --verbose           # detailed output
    python -m cli.main "C1CC1" --json              # JSON output
    python -m cli.main "C1CC1" --compare-ref       # compare with ref
    python -m cli.main --batch compounds.csv       # batch mode
    python -m cli.main --list-refs                 # list references
    python -m cli.main --benchmark                  # benchmark vs refs
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path so we can run as python mmff94/cli/main.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mmff94.ring_strain.core import StrainAnalyzer, StrainReport


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmff94-strain",
        description="Quantify ring strain energy using MMFF94 force field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C1CC1"              Analyze cyclopropane
  %(prog)s "c1ccccc1"           Analyze benzene
  %(prog)s "C1CC1" --json       Output as JSON
  %(prog)s "C1CC1" --compare-ref Compare with experimental data
  %(prog)s --list-refs           List all reference compounds
  %(prog)s --benchmark           Run benchmark on reference compounds
        """,
    )
    parser.add_argument(
        "smiles",
        nargs="?",
        help="SMILES string to analyze",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-ring breakdown.",
    )
    parser.add_argument(
        "--compare-ref", "-c",
        action="store_true",
        help="Compare result against known reference compound.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=8.0,
        help="Stability score threshold in kcal/mol (default: 8.0).",
    )
    parser.add_argument(
        "--n-conformers",
        type=int,
        default=200,
        help="Number of conformers to sample (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for conformer generation (default: 42).",
    )
    parser.add_argument(
        "--batch",
        metavar="FILE",
        help="Batch process SMILES from a file (one per line).",
    )
    parser.add_argument(
        "--list-refs",
        action="store_true",
        help="List all reference compounds in the database.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on reference compounds.",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = StrainAnalyzer(
        n_conformers=args.n_conformers,
        random_seed=args.seed,
        stability_threshold=args.threshold,
    )

    # --- List references ---
    if args.list_refs:
        _cmd_list_refs(analyzer)
        return

    # --- Benchmark ---
    if args.benchmark:
        _cmd_benchmark(analyzer)
        return

    # --- Batch mode ---
    if args.batch:
        _cmd_batch(analyzer, args.batch, args.json)
        return

    # --- Single molecule ---
    if not args.smiles:
        parser.print_help()
        sys.exit(1)

    report = analyzer.analyze(args.smiles)

    if args.json:
        print(report.to_json())
    else:
        print(report.print_summary())

        if args.compare_ref:
            ref = analyzer.ref_db.get_by_smiles(args.smiles)
            if ref:
                print(f"\n--- Known Reference ---")
                print(f"  Name:        {ref.name}")
                print(f"  Exp. strain: {ref.strain_energy_kcal_mol:.2f} kcal/mol")
                print(f"  Computed:    {report.total_strain_calibrated_kcal_mol:.2f} kcal/mol")
                error = abs(report.total_strain_calibrated_kcal_mol - ref.strain_energy_kcal_mol)
                print(f"  Error:       {error:.2f} kcal/mol")
                print(f"  Source:      {ref.source}")
            else:
                print(f"\n  (No experimental reference found for this SMILES)")

        if args.verbose and report.per_ring_details:
            print(f"\n--- Geometric Details ---")
            for ring in report.per_ring_details:
                print(f"  Ring {ring['ring_index']} ({ring['size']}-membered, {ring['type']}):")
                print(f"    Atoms: {ring['atoms']}")
                if ring['heteroatoms']:
                    print(f"    Heteroatoms: {ring['heteroatoms']}")
                print(f"    Aromatic: {ring['is_aromatic']}")


# -----------------------------------------------------------------------
# Command implementations
# -----------------------------------------------------------------------

def _cmd_list_refs(analyzer: StrainAnalyzer):
    refs = analyzer.list_references()
    print(f"\nReference Compounds ({len(refs)} total)")
    print("=" * 70)
    print(f"{'Name':<28} {'SMILES':<20} {'Strain':>8} {'Ring Sizes':>12}")
    print("-" * 70)
    for r in refs:
        print(
            f"{r['name']:<28} {r['smiles']:<20} "
            f"{r['strain_kcal_mol']:>7.1f}  {str(r['ring_sizes']):>12}"
        )


def _cmd_benchmark(analyzer: StrainAnalyzer):
    refs = analyzer.ref_db.get_all()
    results = []

    print(f"\nBenchmark: {len(refs)} reference compounds")
    print("=" * 75)
    print(f"{'Name':<28} {'Exp':>7} {'MMFF':>7} {'Calib':>7} {'Error':>6}")

    errors = []
    for ref in refs:
        try:
            report = analyzer.analyze(ref.smiles)
            error = abs(report.total_strain_calibrated_kcal_mol - ref.strain_energy_kcal_mol)
            errors.append(error)
            print(
                f"{ref.name:<28} "
                f"{ref.strain_energy_kcal_mol:>6.1f} "
                f"{report.total_strain_mmff_kcal_mol:>6.1f} "
                f"{report.total_strain_calibrated_kcal_mol:>6.1f} "
                f"{error:>5.2f}"
            )
            results.append({
                "name": ref.name,
                "smiles": ref.smiles,
                "exp": ref.strain_energy_kcal_mol,
                "mmff_raw": report.total_strain_mmff_kcal_mol,
                "calibrated": report.total_strain_calibrated_kcal_mol,
                "error": error,
            })
        except Exception as exc:
            print(f"{ref.name:<28} {'FAILED':>7} ({exc})")

    if errors:
        print("-" * 75)
        print(f"  Mean absolute error: {sum(errors)/len(errors):.2f} kcal/mol")
        print(f"  Max  absolute error: {max(errors):.2f} kcal/mol")
        print(f"  N compounds:         {len(errors)}")


def _cmd_batch(analyzer: StrainAnalyzer, filepath: str, json_output: bool):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    with open(path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if json_output:
        results = []
        for smi in smiles_list:
            try:
                report = analyzer.analyze(smi)
                results.append(report.to_dict())
            except Exception as exc:
                results.append({"smiles": smi, "error": str(exc)})
        print(json.dumps(results, indent=2))
    else:
        for smi in smiles_list:
            try:
                report = analyzer.analyze(smi)
                print(report.print_summary())
                print()
            except Exception as exc:
                print(f"ERROR [{smi}]: {exc}\n")


if __name__ == "__main__":
    main()
