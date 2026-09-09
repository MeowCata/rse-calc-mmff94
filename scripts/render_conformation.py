"""Render the production MMFF94 cyclic conformer with Matplotlib.

Examples
--------
python scripts/render_conformation.py "C1CCCCC1" --output cyclohexane.png
python scripts/render_conformation.py "CC1CCCCC1" --show-hydrogens

The first command writes a PNG. Omitting ``--output`` opens an interactive
viewer for the optimized conformer ensemble.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow the documented ``python scripts/render_conformation.py`` invocation to
# resolve the top-level ``ring_strain`` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ring_strain.visualization import (
    create_conformer_viewer,
    optimize_cyclic_conformer,
    save_conformer_plot,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an MMFF94-optimized monocyclic carbocycle in 3D."
    )
    parser.add_argument("smiles", help="Monocyclic saturated-carbocycle SMILES")
    parser.add_argument(
        "--output", "-o", type=Path,
        help=(
            "Image destination (.png, .svg, or .pdf). Opens the interactive "
            "viewer when omitted."
        ),
    )
    parser.add_argument(
        "--show-hydrogens", action="store_true",
        help="Include hydrogen atoms and C-H bonds in the render.",
    )
    parser.add_argument("--n-conformers", type=int, default=200)
    parser.add_argument("--mc-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-monte-carlo", action="store_true",
        help="Disable MC/PT sampling for substituted rings.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.output is None:
        viewer = create_conformer_viewer(
            args.smiles,
            n_conformers=args.n_conformers,
            random_seed=args.seed,
            use_monte_carlo=not args.no_monte_carlo,
            mc_steps=args.mc_steps,
            show_hydrogens=args.show_hydrogens,
        )
        num_conformers = len(viewer.conformer.conformer_ids)
        energy_span = (
            viewer.conformer.conformer_energies_kcal_mol[-1]
            - viewer.conformer.conformer_energies_kcal_mol[0]
        )
        print(f"Interactive viewer: {num_conformers} conformer(s) retained")
        print(f"MMFF94 conformer energy span: {energy_span:.2f} kcal/mol")
        viewer.show()
        return

    conformer = optimize_cyclic_conformer(
        args.smiles,
        n_conformers=args.n_conformers,
        random_seed=args.seed,
        use_monte_carlo=not args.no_monte_carlo,
        mc_steps=args.mc_steps,
    )

    path = save_conformer_plot(
        conformer, args.output, show_hydrogens=args.show_hydrogens
    )
    print(f"Saved optimized 3D conformation: {path.resolve()}")


if __name__ == "__main__":
    main()
