"""Evaluation: Many Peptides table cells, solvation free energies and the water placement input."""

from __future__ import annotations

import numpy as np
import pytest

from pbg.eval.metrics import summarize
from pbg.eval.solvation import cutoff_offset, reference_selection, solvation_estimate


def test_table1_cells_average_seed_means_and_starred_cells_average_seed_medians() -> None:
    values = {1: {"A": 1.0, "B": 2.0, "C": 3.0, "YFPHAGYT": 100.0}, 2: {"A": 3.0, "B": 4.0}}
    results = [
        {"sequence": sequence, "size": "2AA", "seed": seed,
         "metrics": {"end_to_end_ress_drop_0_2pct": value, "end_to_end_ress": value}}
        for seed, systems in values.items() for sequence, value in systems.items()
    ]

    table1, starred = summarize(results)

    # Table 1: systems with a value in every seed (A, B), mean over seeds, then over systems.
    assert table1["label"] == "rESS"
    assert table1["value"] == pytest.approx(2.5)
    assert (table1["seed_min"], table1["seed_max"], table1["n_systems"]) == (1.5, 3.5, 2)
    # Starred: median over systems within each seed (2 and 3.5), then the mean; YFPHAGYT is left out.
    assert starred["label"] == "rESS*"
    assert starred["value"] == pytest.approx(2.75)
    assert starred["n_systems"] == 3


def test_solvation_estimate_subtracts_the_legs_and_adds_errors_in_quadrature() -> None:
    vacuum = dict(forward_kj_mol=1.0, forward_se_kj_mol=0.3, bar_kj_mol=2.0, bar_se_kj_mol=None,
                  bar_overlap=0.2, forward_ress=0.01)
    water = dict(forward_kj_mol=-4.0, forward_se_kj_mol=0.4, bar_kj_mol=-3.0, bar_se_kj_mol=0.1,
                 bar_overlap=0.05, forward_ress=None)

    estimate = solvation_estimate(vacuum, water)

    assert estimate["forward_kj_mol"] == pytest.approx(-5.0)
    assert estimate["forward_se_kj_mol"] == pytest.approx(0.5)
    assert estimate["bar_kj_mol"] == pytest.approx(-5.0)
    assert estimate["bar_se_kj_mol"] is None
    assert estimate["min_bar_overlap"] == 0.05
    assert estimate["min_forward_ress"] is None


@pytest.mark.parametrize("sample_temperature_K", [None, 350.0])
def test_cutoff_offset_of_a_constant_energy_shift_is_that_shift(sample_temperature_K) -> None:
    scored = np.random.default_rng(0).normal(-100.0, 5.0, size=64)

    offset = cutoff_offset(scored, scored + 2.5, sample_temperature_K=sample_temperature_K)

    assert offset == pytest.approx(2.5)


def test_reference_selection_balances_frames_over_trajectories() -> None:
    np.testing.assert_array_equal(reference_selection([10, 5], 6), [2, 5, 9, 11, 12, 14])
    every = reference_selection([10, 5], 100)
    np.testing.assert_array_equal(every, np.r_[np.arange(2, 10), np.arange(11, 15)])
    with pytest.raises(ValueError, match="budget"):
        reference_selection([10, 5], 0)


def test_place_waters_needs_frames_in_the_ff14sb_atom_order() -> None:
    pytest.importorskip("openmm")
    from pbg.core.molecule import atom_numbers_from_smiles
    from pbg.energy import expected_solvated_smiles
    from pbg.eval.microsolvation import place_waters

    atomic_numbers = atom_numbers_from_smiles(expected_solvated_smiles("A", 0)).astype(np.float64)
    solute = np.zeros((1, atomic_numbers.size, 4))
    solute[:, :, 0] = atomic_numbers

    with pytest.raises(ValueError, match="method"):  # the atom order passes, the method does not
        place_waters(solute, "A", 4, "packmol")
    solute[:, :, 0] = atomic_numbers[::-1]
    with pytest.raises(ValueError, match="ff14SB atom order"):
        place_waters(solute, "A", 4, "tleap")
