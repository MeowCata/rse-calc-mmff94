"""Fast checks for optimized-conformer Matplotlib rendering."""

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

from ring_strain.visualization import (
    ConformerViewer,
    create_conformer_viewer,
    optimize_cyclic_conformer,
    plot_conformer,
)


@pytest.fixture(scope="module")
def methylcyclohexane_conformers():
    return optimize_cyclic_conformer(
        "CC1CCCCC1", n_conformers=30, mc_steps=50
    )


def test_cyclopropane_renderer_uses_optimized_3d_coordinates():
    conformer = optimize_cyclic_conformer(
        "C1CC1", n_conformers=20, use_monte_carlo=False
    )
    positions = conformer.molecule.GetConformer(conformer.conformer_id).GetPositions()

    assert conformer.ring_info.size == 3
    assert conformer.conformers_considered >= 1
    assert positions.shape == (conformer.molecule.GetNumAtoms(), 3)
    assert len({tuple(row) for row in positions[:3]}) == 3


def test_plot_conformer_draws_ring_bonds_and_atoms():
    import matplotlib.pyplot as plt

    conformer = optimize_cyclic_conformer(
        "C1CC1", n_conformers=20, use_monte_carlo=False
    )
    figure, axis = plot_conformer(conformer)

    assert figure is axis.figure
    assert len(axis.lines) == conformer.ring_info.size
    assert len(axis.collections) == conformer.ring_info.size
    plt.close(figure)


def test_symmetry_distinct_conformers_retained_for_substituted_ring(
    methylcyclohexane_conformers,
):
    conformer = methylcyclohexane_conformers

    # The sampled axial/equatorial chairs remain, while symmetry-equivalent
    # coordinates and explicit-hydrogen rotations are removed.
    assert len(conformer.conformer_ids) == 2
    assert len(conformer.conformer_energies_kcal_mol) == len(conformer.conformer_ids)
    assert conformer.conformer_id == conformer.conformer_ids[0]

    energies = list(conformer.conformer_energies_kcal_mol)
    assert energies == sorted(energies)
    assert 1.0 < energies[1] - energies[0] < 2.0


def test_conformer_viewer_controls_preserve_camera(
    methylcyclohexane_conformers,
):
    import matplotlib.pyplot as plt

    conformer = methylcyclohexane_conformers
    viewer = ConformerViewer(conformer)

    assert viewer.conformer is conformer
    assert viewer.slider is not None
    assert viewer.checkbox is not None
    assert viewer.button is not None

    viewer.ax.view_init(elev=14, azim=71)
    viewer.slider.set_val(2)
    assert viewer.current_conf_idx == 1
    assert viewer.ax.elev == pytest.approx(14)
    assert viewer.ax.azim == pytest.approx(71)

    viewer.checkbox.set_active(0)
    assert viewer.show_hydrogens is True
    assert len(viewer.ax.collections) == conformer.molecule.GetNumAtoms()

    viewer._on_reset_button(None)
    assert viewer.ax.elev == pytest.approx(viewer.default_elev)
    assert viewer.ax.azim == pytest.approx(viewer.default_azim)
    plt.close(viewer.figure)


def test_create_conformer_viewer_factory():
    import matplotlib.pyplot as plt

    viewer = create_conformer_viewer(
        "C1CCCCC1",
        n_conformers=20,
        mc_steps=30,
        show_hydrogens=True,
    )

    assert isinstance(viewer, ConformerViewer)
    assert viewer.conformer.canonical_smiles == "C1CCCCC1"
    assert len(viewer.conformer.conformer_ids) >= 1
    assert viewer.show_hydrogens is True
    assert viewer.checkbox.get_status() == [True]
    plt.close(viewer.figure)
