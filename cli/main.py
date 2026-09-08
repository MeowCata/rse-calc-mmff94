"""
Command-line interface for ring strain analysis.

Usage:
    python -m cli.main "C1CC1"                     # cyclopropane
    python -m cli.main "C1CC1" --json              # JSON output
    python -m cli.main "C1CC1" --compare-ref       # compare with ref
    python -m cli.main --batch compounds.csv       # batch mode
    python -m cli.main --list-refs                 # list references
    python -m cli.main --benchmark                  # benchmark vs refs
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from ring_strain.core import ProgressUpdate, StrainAnalyzer


class _CliProgress:
    """Render progress on stderr without contaminating result output."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._line_width = 0
        self._last_signature = None

    def __call__(self, update: ProgressUpdate) -> None:
        counter = ""
        if update.current is not None and update.total is not None:
            counter = f" ({update.current}/{update.total})"
        line = (
            f"[{update.percent:6.1f}%] {update.task}{counter} "
            f"| elapsed {update.elapsed_seconds:.1f} s"
        )
        terminal = update.stage in {"complete", "error"}

        if self.is_tty:
            padded = line.ljust(self._line_width)
            self._line_width = max(self._line_width, len(line))
            print(padded, end="\n" if terminal else "\r", file=self.stream, flush=True)
            if terminal:
                self._line_width = 0
            return

        # Redirected logs receive stage/task transitions and 10% boundaries,
        # avoiding one line per MC step while retaining meaningful feedback.
        signature = (update.stage, update.task, int(update.percent // 10))
        if terminal or signature != self._last_signature:
            print(line, file=self.stream, flush=True)
            self._last_signature = signature


def _item_progress_callback(callback, index, total, label):
    """Map one analysis into its item range for batch/benchmark progress."""
    if callback is None:
        return None

    def report(update: ProgressUpdate) -> None:
        callback(replace(
            update,
            task=f"{label} {index}/{total}: {update.task}",
            progress=((index - 1) + update.progress) / total,
        ))

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmff94-strain",
        description="Quantify ring strain energy using MMFF94 force field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "C1CC1"              Analyze cyclopropane
  %(prog)s "C1CCCC1"            Analyze cyclopentane
  %(prog)s "C1CC1" --json       Output as JSON
  %(prog)s "C1CC1" --compare-ref Compare with experimental data
  %(prog)s --list-refs           List all reference compounds
  %(prog)s --benchmark           Run benchmark on reference compounds
        """,
    )
    parser.add_argument(
        "smiles",
        nargs="?",
        help="SMILES string to analyze (monocyclic saturated carbocycles only)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON.",
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
        "--no-monte-carlo",
        action="store_true",
        help="Disable Monte Carlo search for substituted cycloalkanes.",
    )
    parser.add_argument(
        "--mc-steps",
        type=int,
        default=500,
        help="Monte Carlo steps (default: 500).",
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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress calculation progress on stderr.",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Initialize analyzer
    progress_callback = None if args.no_progress else _CliProgress()
    analyzer = StrainAnalyzer(
        n_conformers=args.n_conformers,
        random_seed=args.seed,
        stability_threshold=args.threshold,
        use_monte_carlo=not args.no_monte_carlo,
        mc_steps=args.mc_steps,
        progress_callback=progress_callback,
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

        if args.compare_ref and report.is_supported:
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


# -----------------------------------------------------------------------
# Command implementations
# -----------------------------------------------------------------------

def _cmd_list_refs(analyzer: StrainAnalyzer):
    refs = analyzer.list_references()
    print(f"\nReference Compounds ({len(refs)} total)")
    print("=" * 70)
    print(f"{'Name':<38} {'SMILES':<20} {'Strain':>8} {'Ring Sizes':>12}")
    print("-" * 70)
    for r in refs:
        print(
            f"{r['name']:<38} {r['smiles']:<20} "
            f"{r['strain_kcal_mol']:>7.1f}  {str(r['ring_sizes']):>12}"
        )


def _cmd_benchmark(analyzer: StrainAnalyzer):
    refs = analyzer.ref_db.get_all()
    results = []
    started_at = time.perf_counter()

    print(f"\nBenchmark: {len(refs)} reference compounds")
    print("=" * 80)
    print(f"{'Name':<38} {'Exp':>7} {'MMFF':>7} {'Calib':>7} {'Error':>6} {'Note'}")

    errors = []
    for index, ref in enumerate(refs, start=1):
        try:
            report = analyzer.analyze(
                ref.smiles,
                progress_callback=_item_progress_callback(
                    analyzer.progress_callback, index, len(refs), "Benchmark"
                ),
            )
            error = abs(report.total_strain_calibrated_kcal_mol - ref.strain_energy_kcal_mol)
            if report.is_supported:
                errors.append(error)
            note = "" if report.is_supported else "skipped"
            print(
                f"{ref.name:<38} "
                f"{ref.strain_energy_kcal_mol:>6.1f} "
                f"{report.total_strain_mmff_kcal_mol:>6.1f} "
                f"{report.total_strain_calibrated_kcal_mol:>6.1f} "
                f"{error:>5.2f}  {note}"
            )
            results.append({
                "name": ref.name,
                "smiles": ref.smiles,
                "exp": ref.strain_energy_kcal_mol,
                "mmff_raw": report.total_strain_mmff_kcal_mol,
                "calibrated": report.total_strain_calibrated_kcal_mol,
                "error": error,
                "supported": report.is_supported,
            })
        except Exception as exc:
            print(f"{ref.name:<38} {'FAILED':>7} ({exc})")

    if errors:
        print("-" * 80)
        print(f"  Mean absolute error: {sum(errors)/len(errors):.2f} kcal/mol")
        print(f"  Max  absolute error: {max(errors):.2f} kcal/mol")
        print(f"  N compounds:         {len(errors)}")
    if analyzer.progress_callback is not None:
        print(
            f"Benchmark completed in {time.perf_counter() - started_at:.2f} s",
            file=sys.stderr,
        )


def _cmd_batch(analyzer: StrainAnalyzer, filepath: str, json_output: bool):
    path = Path(filepath)
    if not path.exists():
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        smiles_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    started_at = time.perf_counter()
    total = len(smiles_list)
    if json_output:
        results = []
        for index, smi in enumerate(smiles_list, start=1):
            try:
                report = analyzer.analyze(
                    smi,
                    progress_callback=_item_progress_callback(
                        analyzer.progress_callback, index, total, "Batch"
                    ),
                )
                results.append(report.to_dict())
            except Exception as exc:
                results.append({"smiles": smi, "error": str(exc)})
        print(json.dumps(results, indent=2))
    else:
        for index, smi in enumerate(smiles_list, start=1):
            try:
                report = analyzer.analyze(
                    smi,
                    progress_callback=_item_progress_callback(
                        analyzer.progress_callback, index, total, "Batch"
                    ),
                )
                print(report.print_summary())
                print()
            except Exception as exc:
                print(f"ERROR [{smi}]: {exc}\n")
    if analyzer.progress_callback is not None:
        print(
            f"Batch completed in {time.perf_counter() - started_at:.2f} s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
