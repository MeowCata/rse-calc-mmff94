"""Matplotlib rendering of MMFF94-optimized cyclic conformers.

The renderer deliberately uses the same cyclic-side conformer-search sequence
as :class:`ring_strain.core.StrainAnalyzer`: ETKDG/MMFF94 optimization,
ring-pucker seeds for substituted 4-7 membered rings, bulk-aware PT/MC search,
and conformer clustering.  It then returns all clustered representatives,
enabling interactive exploration of low-energy conformational basins.

Matplotlib is imported only by rendering functions, so the normal strain
calculation API remains usable when the optional visualization dependency is
not installed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.rdchem import Mol

from .mmff import MMFFCalculator
from .ring_analysis import RingAnalyzer, RingInfo

logger = logging.getLogger(__name__)


@dataclass
class OptimizedConformer:
    """Cyclic conformers prepared with the production sampling settings.

    When clustering produces multiple representatives, all are retained with
    sequential conformer IDs 0, 1, 2, ... and sorted by ascending energy.
    """

    smiles: str
    canonical_smiles: str
    formula: str
    molecular_weight: float
    molecule: Mol
    conformer_id: int
    conformer_ids: Tuple[int, ...]
    conformer_energies_kcal_mol: Tuple[float, ...]
    ring_info: RingInfo
    energy_kcal_mol: float
    conformers_considered: int
    monte_carlo_used: bool


def optimize_cyclic_conformer(
    smiles: str,
    *,
    n_conformers: int = 200,
    random_seed: int = 42,
    use_monte_carlo: bool = True,
    mc_steps: int = 500,
    use_parallel_tempering: bool = True,
    use_boltzmann: bool = True,
) -> OptimizedConformer:
    """Return optimized cyclic conformers for ``smiles``.

    This intentionally runs only the cyclic side of the strain model.  Ring
    opening, calibration, and scoring do not alter molecular coordinates, so
    omitting them makes visualization faster while preserving the geometry
    selected by the production conformer search.

    When clustering is enabled and produces multiple representatives, all are
    retained in the returned molecule, sorted by energy with sequential IDs.

    Raises:
        ValueError: If the SMILES is invalid or outside the supported
            monocyclic saturated-carbocycle scope.
    """
    if not smiles or not smiles.strip():
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    is_valid, message, ring_info = RingAnalyzer(mol).validate()
    if not is_valid or ring_info is None:
        raise ValueError(message)

    calculator = MMFFCalculator(
        n_conformers=n_conformers,
        random_seed=random_seed,
    )
    is_substituted = mol.GetNumHeavyAtoms() > ring_info.size
    cyclic_result = calculator.embed_and_optimize(Chem.Mol(mol))
    cyclic_mol = cyclic_result.molecule
    best_conf_id = cyclic_result.best_conf_id
    monte_carlo_used = False

    # Random-coordinate seeds sample ring-pucker basins before PT/MC refines
    # substituent rotamers. This mirrors the cyclic branch in core.py.
    if is_substituted and 4 <= ring_info.size <= 7:
        try:
            calculator.seed_ring_pucker_conformers(
                cyclic_mol, list(ring_info.atom_indices)
            )
        except Exception as exc:
            logger.warning("Ring-pucker seeding failed for %s: %s", smiles, exc)

    if is_substituted and use_monte_carlo:
        bulk = min(15, calculator.compute_bulk_score(cyclic_mol))
        pt_steps = max(100, mc_steps // 4) + 12 * bulk
        mc_steps_effective = mc_steps + 18 * bulk
        temperatures: Tuple[float, ...] = (300.0, 500.0, 1000.0, 2000.0)

        try:
            if use_parallel_tempering:
                best_id, _ = calculator.parallel_tempering_search(
                    cyclic_mol,
                    n_steps=pt_steps,
                    temperatures=temperatures,
                )
            else:
                best_id, _ = calculator.monte_carlo_search(
                    cyclic_mol,
                    n_steps=mc_steps_effective,
                )
            if best_id >= 0:
                best_conf_id = best_id
                monte_carlo_used = True
        except Exception as exc:
            logger.warning(
                "Cyclic torsion search failed for %s: %s. Using ETKDG results.",
                smiles,
                exc,
            )

    conformers_considered = cyclic_mol.GetNumConformers()

    if use_boltzmann:
        id_energy_pairs = _cluster_for_visualization(calculator, cyclic_mol)
    else:
        energy = calculator.compute_single_point_energy(
            cyclic_mol, conf_id=best_conf_id
        )
        id_energy_pairs = [(best_conf_id, energy)]

    if not id_energy_pairs:
        raise RuntimeError("MMFF94 did not produce an optimized conformer.")

    # Copy selected coordinates without modifying the production sampling pool.
    compact_mol = Chem.Mol(cyclic_mol)
    compact_mol.RemoveAllConformers()

    new_ids = []
    energies = []
    for old_id, energy in id_energy_pairs:
        conf = Chem.Conformer(cyclic_mol.GetConformer(old_id))
        new_id = compact_mol.AddConformer(conf, assignId=True)
        new_ids.append(new_id)
        energies.append(energy)

    return OptimizedConformer(
        smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        formula=CalcMolFormula(mol),
        molecular_weight=MolWt(mol),
        molecule=compact_mol,
        conformer_id=new_ids[0],
        conformer_ids=tuple(new_ids),
        conformer_energies_kcal_mol=tuple(energies),
        ring_info=ring_info,
        energy_kcal_mol=energies[0],
        conformers_considered=conformers_considered,
        monte_carlo_used=monte_carlo_used,
    )


def _cluster_for_visualization(
    calculator: MMFFCalculator,
    molecule: Mol,
    *,
    rmsd_threshold: float = 0.5,
    energy_window_kcal: float = 8.0,
) -> list[tuple[int, float]]:
    """Select symmetry-distinct, low-energy heavy-atom conformers.

    ``MMFFCalculator.cluster_conformers`` currently includes explicit
    hydrogens in its RMSD calculation. That can turn equivalent C-H rotations
    into dozens of apparent molecular conformers. The viewer clusters the
    same optimized pool using the intended heavy-atom metric and molecular
    automorphisms, leaving the production calculator untouched.
    """
    conformer_ids = [conf.GetId() for conf in molecule.GetConformers()]
    id_energy_pairs = []
    for conf_id in conformer_ids:
        energy = calculator.compute_single_point_energy(molecule, conf_id=conf_id)
        if math.isfinite(energy):
            id_energy_pairs.append((conf_id, energy))
    if not id_energy_pairs:
        return []

    id_energy_pairs.sort(key=lambda item: item[1])
    minimum_energy = id_energy_pairs[0][1]
    heavy_atom_ids = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    heavy_molecule = Chem.RemoveHs(Chem.Mol(molecule))
    automorphisms = heavy_molecule.GetSubstructMatches(
        heavy_molecule,
        uniquify=False,
        useChirality=True,
        maxMatches=1000,
    )
    if not automorphisms:
        automorphisms = (tuple(range(len(heavy_atom_ids))),)

    coordinates = {
        conf_id: np.asarray(
            molecule.GetConformer(conf_id).GetPositions(), dtype=float
        )[heavy_atom_ids]
        for conf_id, _ in id_energy_pairs
    }
    selected: list[tuple[int, float]] = []
    for conf_id, energy in id_energy_pairs:
        if energy - minimum_energy > energy_window_kcal:
            continue
        if any(
            _symmetry_aware_rmsd(
                coordinates[conf_id], coordinates[kept_id], automorphisms
            ) < rmsd_threshold
            for kept_id, _ in selected
        ):
            continue
        selected.append((conf_id, energy))
    return selected


def _symmetry_aware_rmsd(
    probe: np.ndarray,
    reference: np.ndarray,
    automorphisms: Sequence[Sequence[int]],
) -> float:
    """Return minimum aligned RMSD across graph-equivalent atom mappings."""
    return min(
        _aligned_rmsd(probe[np.asarray(mapping, dtype=int)], reference)
        for mapping in automorphisms
    )


def _aligned_rmsd(probe: np.ndarray, reference: np.ndarray) -> float:
    """Return Kabsch-aligned RMSD without mutating either coordinate set."""
    probe_centred = probe - probe.mean(axis=0)
    reference_centred = reference - reference.mean(axis=0)
    left, _, right_t = np.linalg.svd(probe_centred.T @ reference_centred)
    if np.linalg.det(left @ right_t) < 0:
        left[:, -1] *= -1
    rotation = left @ right_t
    delta = probe_centred @ rotation - reference_centred
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _draw_scene(
    ax,
    conformer: OptimizedConformer,
    conf_id: int,
    show_hydrogens: bool,
) -> None:
    """Draw atoms and bonds for the specified conformer ID."""
    ax.clear()

    molecule = conformer.molecule
    positions = np.asarray(
        molecule.GetConformer(conf_id).GetPositions(), dtype=float
    )
    ring_atoms = set(conformer.ring_info.atom_indices)
    ring_bonds = set(conformer.ring_info.bond_indices)
    visible_atoms = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if show_hydrogens or atom.GetAtomicNum() != 1
    }

    # Draw bonds before atoms so spheres remain visually distinct
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in visible_atoms or end not in visible_atoms:
            continue
        colour = "#e67e22" if bond.GetIdx() in ring_bonds else "#59636d"
        linewidth = 3.0 if bond.GetIdx() in ring_bonds else 1.8
        ax.plot(
            positions[[begin, end], 0],
            positions[[begin, end], 1],
            positions[[begin, end], 2],
            color=colour,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=1,
        )

    element_colours = {
        1: "#d5dbe0",   # H
        6: "#34495e",   # off-ring C
        7: "#3f6fb5",   # N
        8: "#c84b4b",   # O
        9: "#5baf74",   # F
        15: "#d9903d",  # P
        16: "#d6b63b",  # S
        17: "#63a77a",  # Cl
    }
    atom_sizes = {1: 45, 6: 260, 7: 250, 8: 250, 9: 225, 15: 280, 16: 280, 17: 300}
    for atom in molecule.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx not in visible_atoms:
            continue
        atomic_number = atom.GetAtomicNum()
        colour = "#f39c12" if atom_idx in ring_atoms else element_colours.get(
            atomic_number, "#8e44ad"
        )
        size = 310 if atom_idx in ring_atoms else atom_sizes.get(atomic_number, 240)
        ax.scatter(
            positions[atom_idx, 0],
            positions[atom_idx, 1],
            positions[atom_idx, 2],
            s=size,
            c=colour,
            depthshade=True,
            edgecolors="#ffffff",
            linewidths=0.65,
            zorder=2,
        )

    _set_equal_3d_limits(ax, positions[list(visible_atoms)])
    ax.set_axis_off()
    ax.set_proj_type("ortho")

    # Show current conformer info
    conf_idx = conformer.conformer_ids.index(conf_id)
    energy = conformer.conformer_energies_kcal_mol[conf_idx]
    relative_energy = energy - conformer.conformer_energies_kcal_mol[0]
    title = (
        f"MMFF94 optimized carbocycle\n"
        f"{conformer.canonical_smiles}\n"
        f"Conformer {conf_idx + 1}/{len(conformer.conformer_ids)}    "
        f"E = {energy:+.2f} kcal/mol    dE = {relative_energy:.2f} kcal/mol"
    )
    ax.set_title(title, pad=8, fontsize=10)


class ConformerViewer:
    """Interactive Matplotlib viewer for multiple conformers.

    Provides slider to switch between clustered representatives, checkbox to
    toggle hydrogen visibility, and button to reset camera view.
    """

    def __init__(
        self,
        conformer: OptimizedConformer,
        *,
        show_hydrogens: bool = False,
    ):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider, CheckButtons, Button
        except ImportError as exc:
            raise RuntimeError(
                "Matplotlib is required for 3D rendering. Install it with "
                "`python -m pip install matplotlib`."
            ) from exc

        self.conformer = conformer
        self.current_conf_idx = 0
        self.show_hydrogens = show_hydrogens
        self.default_elev = 22
        self.default_azim = 36

        # Fixed widget rows avoid constrained-layout changes during redraws.
        self.figure = plt.figure(figsize=(9, 8))
        self.ax = self.figure.add_axes(
            (0.04, 0.17, 0.92, 0.67), projection="3d"
        )
        self.ax.view_init(elev=self.default_elev, azim=self.default_azim)

        if len(conformer.conformer_ids) > 1:
            slider_ax = self.figure.add_axes((0.18, 0.105, 0.64, 0.035))
            self.slider = Slider(
                slider_ax,
                "Conformer",
                1,
                len(conformer.conformer_ids),
                valinit=1,
                valstep=1,
                valfmt="%0.0f",
                color="#3498db",
            )
            self.slider.on_changed(self._on_slider_change)
        else:
            self.slider = None

        check_ax = self.figure.add_axes((0.06, 0.025, 0.18, 0.055))
        self.checkbox = CheckButtons(
            check_ax, ["Show H"], [self.show_hydrogens]
        )
        self.checkbox.on_clicked(self._on_checkbox_toggle)

        button_ax = self.figure.add_axes((0.79, 0.035, 0.15, 0.045))
        self.button = Button(
            button_ax,
            "Reset View",
            color="#ecf0f1",
            hovercolor="#bdc3c7",
        )
        self.button.on_clicked(self._on_reset_button)

        self._redraw()

    def _on_slider_change(self, val):
        """Handle conformer slider movement."""
        self.current_conf_idx = int(val) - 1
        self._redraw()

    def _on_checkbox_toggle(self, label):
        """Handle hydrogen visibility checkbox."""
        self.show_hydrogens = bool(self.checkbox.get_status()[0])
        self._redraw()

    def _on_reset_button(self, event):
        """Handle reset view button click."""
        self._redraw(preserve_view=False)

    def _redraw(self, *, preserve_view: bool = True):
        """Redraw the 3D scene with current settings."""
        elevation = self.ax.elev if preserve_view else self.default_elev
        azimuth = self.ax.azim if preserve_view else self.default_azim
        conf_id = self.conformer.conformer_ids[self.current_conf_idx]
        _draw_scene(self.ax, self.conformer, conf_id, self.show_hydrogens)
        self.ax.view_init(elev=elevation, azim=azimuth)
        self.figure.canvas.draw_idle()

    def show(self):
        """Display the interactive viewer window."""
        import matplotlib.pyplot as plt
        plt.show()
        return self.figure, self.ax


def create_conformer_viewer(
    smiles: str,
    *,
    n_conformers: int = 200,
    random_seed: int = 42,
    use_monte_carlo: bool = True,
    mc_steps: int = 500,
    show_hydrogens: bool = False,
) -> ConformerViewer:
    """Create an interactive conformer viewer for ``smiles``.

    Returns a :class:`ConformerViewer` with controls to explore clustered
    conformational representatives. Call ``.show()`` to display.

    Example:
        >>> viewer = create_conformer_viewer("C1CCCCC1")
        >>> viewer.show()
    """
    conformer = optimize_cyclic_conformer(
        smiles,
        n_conformers=n_conformers,
        random_seed=random_seed,
        use_monte_carlo=use_monte_carlo,
        mc_steps=mc_steps,
        use_parallel_tempering=True,
        use_boltzmann=True,
    )
    return ConformerViewer(conformer, show_hydrogens=show_hydrogens)


def plot_conformer(
    conformer: OptimizedConformer,
    *,
    show_hydrogens: bool = False,
    ax=None,
):
    """Draw an optimized conformer in an interactive Matplotlib 3D axis.

    Carbon atoms in the validated ring are highlighted in orange; off-ring
    atoms use standard element colours. The returned ``(figure, axis)`` keeps
    Matplotlib's normal pan, zoom, rotate, and save controls available.

    This function renders only the lowest-energy conformer. For interactive
    exploration of multiple conformers, use :func:`create_conformer_viewer`.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for 3D rendering. Install it with "
            "`python -m pip install matplotlib`."
        ) from exc

    if ax is None:
        figure = plt.figure(figsize=(8, 8), layout="constrained")
        ax = figure.add_subplot(projection="3d")
    else:
        figure = ax.figure

    _draw_scene(ax, conformer, conformer.conformer_id, show_hydrogens)
    ax.view_init(elev=22, azim=36)
    
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="Ring carbon",
                   markerfacecolor="#f39c12", markeredgecolor="#ffffff", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Other atoms",
                   markerfacecolor="#34495e", markeredgecolor="#ffffff", markersize=9),
            Line2D([0], [0], color="#e67e22", linewidth=3, label="Ring bond"),
        ],
        loc="upper left",
        frameon=False,
    )
    return figure, ax


def save_conformer_plot(
    conformer: OptimizedConformer,
    output_path: str | Path,
    *,
    show_hydrogens: bool = False,
    dpi: int = 220,
) -> Path:
    """Render ``conformer`` to a Matplotlib-supported image format."""
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_conformer(conformer, show_hydrogens=show_hydrogens)
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return path


def _set_equal_3d_limits(ax, positions: np.ndarray) -> None:
    """Set equal XYZ limits so molecular bond angles are not distorted."""
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    centre = (lower + upper) / 2.0
    radius = max(float((upper - lower).max()) / 2.0, 1.0) + 0.55
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1))
