"""Coordinate binning, the core and joint vocabularies and prompt encoding."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from jax import random
from rdkit import Chem

from pbg.core.augmenter import init_augmenter
from pbg.core.frame import DataFrame
from pbg.core.peptide import sequence_to_smiles
from pbg.core.tokeniser import (
    CoreTokenizer,
    VocabSchema,
    load_tokeniser_default,
    make_tokeniser,
    upgrade_core_mask_token,
)
from pbg.data.bin import apply_discretise, undo_discretise
from pbg.models.pbg.loader import prepare_frame

# ── Coordinate binning ───────────────────────────────────────────────


def test_binning_round_trip_stays_within_one_bin() -> None:
    num_bins, min_val, max_val = 256, -3.0, 3.0
    x = np.random.default_rng(42).standard_normal((100, 3)).astype(np.float32) * 0.8

    indices = apply_discretise(x, num_bins=num_bins, min_val=min_val, max_val=max_val)
    decoded = undo_discretise(indices, num_bins=num_bins, min_val=min_val, max_val=max_val, method="center")

    assert indices.shape == x.shape
    assert indices.dtype in (np.int32, np.int64)
    assert 0 <= int(indices.min()) and int(indices.max()) < num_bins
    assert decoded.shape == x.shape
    assert decoded.dtype == np.float32
    assert np.abs(decoded - x).max() <= (max_val - min_val) / num_bins


def test_binning_clips_to_the_edge_bins_whose_centres_are_the_box_edges() -> None:
    clipped = apply_discretise(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), num_bins=10, min_val=0.0, max_val=1.0)
    assert (int(clipped[0]), int(clipped[-1])) == (0, 9)

    edges = apply_discretise(np.array([-4.0, 0.0, 4.0], dtype=np.float32), num_bins=8, min_val=-4.0, max_val=4.0)
    assert (int(edges[0]), int(edges[2])) == (0, 7)
    centres = undo_discretise(edges, num_bins=8, min_val=-4.0, max_val=4.0, method="center")
    np.testing.assert_allclose(centres[[0, 2]], [-4.0, 4.0], atol=1e-6)


def test_uniform_decoding_is_reproducible_and_stays_in_its_cell() -> None:
    # Five grid points on [-1, 1] (endpoint-inclusive, spacing 0.5): uniform
    # decoding draws from [centre - 0.25, centre + 0.25). The edge cells are
    # not clamped to the box, as in the paper's round-trip experiments.
    indices = np.array([0, 0, 4, 4], dtype=np.int32)
    kwargs = dict(num_bins=5, min_val=-1.0, max_val=1.0, method="uniform", key=random.PRNGKey(17))

    first = undo_discretise(indices, **kwargs)

    np.testing.assert_array_equal(first, undo_discretise(indices, **kwargs))
    assert first.dtype == np.float32
    assert np.all((first[:2] >= -1.25) & (first[:2] <= -0.75))
    assert np.all((first[2:] >= 0.75) & (first[2:] <= 1.25))


def test_decoding_rejects_unsupported_methods() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        undo_discretise(np.array([0]), num_bins=64, min_val=-2.0, max_val=2.0, method="empirical_offsets")


# ── Vocabularies ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("num_bins", "last_coordinate", "named", "mask", "next_id", "padding"),
    [
        (4096, 4197, {"water": 4198, "chloroform": 4221}, 4222, 4224, [[4223, 4224]]),
        (65537, 65638, {"water": 65639, "octanol": 65659, "chloroform": 65662}, 65663, 65664, []),
    ],
)
def test_core_vocab_ids_match_the_paper_checkpoints(num_bins, last_coordinate, named, mask, next_id, padding) -> None:
    vocab = load_tokeniser_default(vocab_name=f"core_bin{num_bins}")
    coordinates = sorted(vocab["coordinate_tokens"].values())

    assert vocab["core_special_tokens"] == {"<BOS>": 0, "<PAD>": 0}
    assert len(vocab["core_tokens"]) == 120
    assert (len(coordinates), coordinates[0], coordinates[-1]) == (num_bins, 102, last_coordinate)
    assert {token: vocab["core_tokens"][token] for token in named} == named
    assert "decane" not in vocab["core_tokens"]
    assert vocab["core_tokens"]["[MASK]"] == VocabSchema.of(vocab).mask_id == mask
    assert vocab["meta"]["next_id"] == next_id
    assert next_id % 128 == 0
    assert vocab["meta"]["padding_ranges"] == padding


def test_original_core_bin65537_solvent_rows_upgrade_to_mask() -> None:
    vocab = load_tokeniser_default(vocab_name="core_bin65537")
    core = vocab["core_tokens"]
    del core["[MASK]"]
    core["decane"] = 65662
    core["chloroform"] = 65663
    vocab["meta"]["version"] = 3

    upgraded = upgrade_core_mask_token(vocab)

    assert "decane" not in upgraded["core_tokens"]
    assert upgraded["core_tokens"]["chloroform"] == 65662
    assert upgraded["core_tokens"]["[MASK]"] == 65663
    assert upgraded["meta"]["version"] == 4


def test_vocab_schema_rejects_non_contiguous_coordinates() -> None:
    vocab = load_tokeniser_default(vocab_name="joint_bin4096")
    vocab["coordinate_tokens"]["coord_bin_0001"] += 1

    with pytest.raises(ValueError, match=r"unique|contiguous"):
        VocabSchema.of(vocab)


def test_decode_bias_allows_only_coordinates_and_the_terminator() -> None:
    vocab = load_tokeniser_default(vocab_name="joint_bin4096")
    schema = VocabSchema.of(vocab)
    size = int(vocab["meta"]["next_id"])

    with_end = schema.decode_bias(size, allow_terminator=True)
    without_end = schema.decode_bias(size, allow_terminator=False)

    assert set(np.flatnonzero(np.isfinite(with_end)).tolist()) == set(schema.coord_ids) | {schema.terminator_id}
    assert set(np.flatnonzero(np.isfinite(without_end)).tolist()) == set(schema.coord_ids)


def test_core_tokeniser_uses_longest_match_and_ascii_fallback() -> None:
    vocab = load_tokeniser_default(vocab_name="core_bin4096")
    core = vocab["core_tokens"]

    encoded = CoreTokenizer(vocab).encode("CHEM1{CCCl}$$$$")

    assert encoded == [core[token] for token in ("CHEM", "1", "{", "CC", "Cl", "}", "$$$$")]


# ── Prompts ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("temperature", "environment", "text"),
    [
        (310.0, "water", "310K, water"),
        (373.4214782714844, "water, ff14sb-obc1-2nm", "373.421K, water, ff14sb-obc1-2nm"),
    ],
)
def test_default_prompt_writes_temperature_and_environment_as_text(temperature, environment, text) -> None:
    vocab = load_tokeniser_default(vocab_name="core_bin4096")
    tokeniser = make_tokeniser(vocab)
    helm = "PEPTIDE1{A.C}$$$$"

    ids = tokeniser.encode_prompt(helm, environment=environment, temperature_K=temperature)

    explicit = make_tokeniser(vocab, prompt_format="legacy")
    np.testing.assert_array_equal(ids, explicit.encode_prompt(helm, environment=environment, temperature_K=temperature))
    env_end = list(ids).index(vocab["dipper_added_tokens"]["<|env_end|>"])
    inverse = {value: key for key, value in vocab["core_tokens"].items()}
    assert "".join(inverse[int(i)] for i in ids[1:env_end]) == text
    assert vocab["core_tokens"]["water"] in ids
    np.testing.assert_array_equal(ids[env_end + 1 :], tokeniser.encode_helm(helm))


@pytest.mark.parametrize(
    ("temperature", "environment", "text"),
    [
        (310.0, "water, ff14sb-obc1-2nm", "310K,water,ff14sb-obc1-2nm"),
        (373.4214782714844, "21 water, ff14sb-tip3p-rf2nm", "373K,21water,ff14sb-tip3p-rf2nm"),
        (464.8119812011719, "vacuum, ff14sb", "465K,vacuum,ff14sb"),
        (300.5, "water, ff14sb-obc1-2nm", "301K,water,ff14sb-obc1-2nm"),
        (None, "vacuum, gaff2-am1bcc", "vacuum,gaff2-am1bcc"),
        (300.0, None, "300K"),
    ],
)
def test_compact_kelvin_prompt_reuses_existing_tokens(temperature, environment, text) -> None:
    vocab = load_tokeniser_default(vocab_name="core_bin4096")
    original = deepcopy(vocab)
    tokeniser = make_tokeniser(vocab, prompt_format="compact_kelvin_v1")
    helm = "PEPTIDE1{A.G}$$$$"

    ids = tokeniser.encode_prompt(helm, environment=environment, temperature_K=temperature)

    env_end = list(ids).index(vocab["dipper_added_tokens"]["<|env_end|>"])
    inverse = {value: key for key, value in vocab["core_tokens"].items()}
    assert "".join(inverse[int(i)] for i in ids[1:env_end]) == text
    np.testing.assert_array_equal(ids[env_end + 1 :], tokeniser.encode_helm(helm))
    assert vocab == original


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -1.0, 0.0])
def test_compact_kelvin_prompt_rejects_invalid_temperatures(temperature) -> None:
    tokeniser = make_tokeniser(load_tokeniser_default(vocab_name="core_bin4096"), prompt_format="compact_kelvin_v1")

    with pytest.raises(ValueError, match="temperature"):
        tokeniser.encode_prompt("PEPTIDE1{A}$$$$", environment="water", temperature_K=temperature)


def test_solvated_peptide_frame_gets_a_peptide_helm_prompt(tiny_config) -> None:
    smiles = f"{sequence_to_smiles('AC')}.[O:999]([H:1000])[H:1001]"
    atoms = [atom.GetAtomicNum() for atom in Chem.MolFromSmiles(smiles, sanitize=False).GetAtoms()]
    conformer = np.zeros((len(atoms), 4), dtype=np.float32)
    conformer[:, 0] = atoms
    frame = DataFrame(conformer=conformer, smiles=smiles, environment="water", temperature=300.0)

    prepared = prepare_frame(
        frame,
        tiny_config(),
        load_tokeniser_default(vocab_name="core_bin4096"),
        init_augmenter(center=False, rotate=False, translate=False),
        rng=np.random.default_rng(0),
        clip_solvent_to_fit=True,
        prompt_notation="PEPTIDE",
    )

    assert prepared is not None
    assert prepared.helm == "PEPTIDE1{A.C}$$$$"
