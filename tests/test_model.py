"""The PBG model: checkpoints, collation, loss, atom conditioning, ROBIN-protocol
switches and the sampling policy."""

from __future__ import annotations

import pickle
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from pbg.core.frame import DataFrame
from pbg.core.peptide import sequence_to_helm
from pbg.core.tokeniser import VocabSchema, load_tokeniser_default, make_tokeniser
from pbg.data.joint import fit_frame
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.model import PBG
from pbg.models.pbg.robin import atom_features
from pbg.models.pbg.system import MolecularSystem, SamplingPolicy, prompt_arrays
from pbg.models.pbg.transformer import apply_qwen, qwen_params
from pbg.models.robin.conditioning import encode_sequences

# ── Initialisation and checkpoints ───────────────────────────────────


@pytest.mark.parametrize("vocab_name", ["core_bin4096", "joint_bin4096"])
def test_init_sizes_the_embedding_to_the_vocabulary(tiny_config, vocab_name) -> None:
    model = PBG.init(tiny_config(vocab_name=vocab_name), random.PRNGKey(0))
    vocab = load_tokeniser_default(vocab_name=vocab_name)

    assert model.vocab == vocab
    assert "dipper_added_tokens" in model.vocab
    assert model.qwen["embed"].shape[0] == int(vocab["meta"]["next_id"])


def test_init_rejects_a_vocabulary_with_another_coordinate_count(tiny_config) -> None:
    with pytest.raises(ValueError, match="4096 coordinate tokens, expected 65537"):
        PBG.init(tiny_config(bin_tokens=65537), random.PRNGKey(0))


def test_save_and_load_round_trip(tiny_config, tmp_path) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    path = tmp_path / "model.ckpt"
    model.save(path)

    loaded = PBG.load(path)

    assert loaded.config == model.config
    assert loaded.vocab == model.vocab
    for left, right in zip(jax.tree.leaves(loaded.qwen), jax.tree.leaves(model.qwen), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_load_ignores_config_fields_that_no_longer_exist(tiny_config, tmp_path) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    path = tmp_path / "model.ckpt"
    model.save(path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload["config"]["weight_mode"] = "none"

    assert PBG.load_payload(payload).config == model.config


def test_load_rejects_an_embedding_that_does_not_match_the_vocabulary(tiny_config, tmp_path) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    path = tmp_path / "model.ckpt"
    model.save(path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload["vocab"]["meta"]["next_id"] += 1

    with pytest.raises(ValueError, match="do not match vocab next_id"):
        PBG.load_payload(payload)


def test_qwen_params_drop_removed_weights(tiny_config) -> None:
    cfg = tiny_config()
    qwen = PBG.init(cfg, random.PRNGKey(0)).qwen
    stored = jax.tree.map(np.asarray, qwen)
    stored["temperature_slope"] = np.zeros((cfg.dim,), dtype=np.float32)

    loaded = qwen_params(stored, cfg)

    assert set(loaded) == set(qwen)
    np.testing.assert_array_equal(loaded["embed"], qwen["embed"])


def test_load_checks_and_drops_retired_config_fields(tiny_config, tmp_path) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    path = tmp_path / "model.ckpt"
    model.save(path)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload["config"].update(coordinate_method="bin", max_position=40960, reorder=False)
    assert PBG.load_payload(payload).config == model.config

    payload["config"]["use_environment_tokens"] = True
    with pytest.raises(ValueError, match="use_environment_tokens"):
        PBG.load_payload(payload)


def test_vocabulary_growth_zero_fills_new_rows_and_never_shrinks(tiny_config) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    rows = int(model.qwen["embed"].shape[0])
    untied = PBG(dict(model.qwen, lm_head=model.qwen["embed"] + 1), model.config, model.vocab)

    grown = untied.resize_embeddings(rows + 2)

    for name in ("embed", "lm_head"):
        assert grown.qwen[name].shape == (rows + 2, model.config.dim)
        np.testing.assert_array_equal(grown.qwen[name][:rows], untied.qwen[name])
        assert float(jnp.abs(grown.qwen[name][rows:]).sum()) == 0.0

    parent = PBG(dict(model.qwen, embed=model.qwen["embed"][: rows - 128]), model.config, model.vocab)
    attached = parent.with_vocab(model.vocab)
    assert attached.qwen["embed"].shape[0] == rows
    assert float(jnp.abs(attached.qwen["embed"][rows - 128 :]).sum()) == 0.0

    with pytest.raises(ValueError, match="cannot shrink"):
        model.resize_embeddings(rows - 1)


# ── Forward pass ─────────────────────────────────────────────────────


def test_untied_lm_head_produces_the_logits(tiny_config) -> None:
    cfg = tiny_config()
    model = PBG.init(cfg, random.PRNGKey(0))
    lm_head = jnp.zeros_like(model.qwen["embed"]).at[3].set(1.0)

    logits = apply_qwen(dict(model.qwen, lm_head=lm_head), cfg, jnp.asarray([[0]], dtype=jnp.int32))

    assert logits.shape == (1, 1, lm_head.shape[0])
    assert float(logits[0, 0, 3]) != 0.0
    assert float(jnp.abs(logits[0, 0, :3]).sum()) == 0.0


def test_completep_residual_and_unembedding_scales(tiny_config) -> None:
    ids = jnp.zeros((1, 4), dtype=jnp.int32)
    model = PBG.init(tiny_config(), random.PRNGKey(0))

    base = apply_qwen(model.qwen, model.config, ids)
    residual = apply_qwen(model.qwen, replace(model.config, residual_scale=0.5), ids)
    unembedding = apply_qwen(model.qwen, replace(model.config, unembedding_scale=0.25), ids)

    assert not jnp.allclose(base, residual)
    np.testing.assert_allclose(unembedding, base * 0.25, rtol=1e-5, atol=1e-5)


def test_packed_frames_do_not_attend_to_each_other(tiny_config, peptide_frame, collate) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    batch = collate(model, [[peptide_frame("AA", seed=0), peptide_frame("GK", seed=1)]])
    segment = np.asarray(batch["segment_id"][0])
    k = int(np.flatnonzero(segment == 1)[-2])  # the last coordinate of the second frame
    tokens = np.asarray(batch["tokens"]).copy()
    tokens[0, k] += 1
    changed = dict(batch, tokens=jnp.asarray(tokens))

    base = np.asarray(model.target_log_probs(batch))
    other = np.asarray(model.target_log_probs(changed))

    np.testing.assert_allclose(base[0, segment == 0], other[0, segment == 0], rtol=0, atol=1e-6)
    assert not np.allclose(base[0, k], other[0, k])


# ── Collation ────────────────────────────────────────────────────────


@pytest.mark.parametrize("vocab_name", ["core_bin4096", "joint_bin4096"])
def test_collate_lays_out_bos_prompt_coordinates_and_terminator(tiny_config, peptide_frame, vocab_name) -> None:
    model = PBG.init(tiny_config(vocab_name=vocab_name), random.PRNGKey(0))
    schema = VocabSchema.of(model.vocab)
    frame = peptide_frame("AA", helm=sequence_to_helm("AA"))
    prompt = make_tokeniser(model.vocab).encode_prompt(frame.helm, environment=None, temperature_K=None)

    batch = model.collate([[frame]], rng=np.random.default_rng(0))

    tokens = batch["tokens"][0][batch["segment_id"][0] >= 0]
    assert tokens.size == model.config.sequence_length_for_frame(frame.n_atoms, prompt.size)
    assert tokens[0] == schema.bos_id
    np.testing.assert_array_equal(tokens[1 : 1 + prompt.size], prompt)
    assert np.all(np.isin(tokens[1 + prompt.size : -1], schema.coord_ids))
    assert tokens[-1] == schema.terminator_id
    trained = batch["targets"][0][batch["loss_mask"][0] > 0]
    assert trained.size == 3 * frame.n_atoms + 1
    assert np.all(np.isin(trained[:-1], schema.coord_ids))
    assert trained[-1] == schema.terminator_id


def test_collate_pads_sample_metadata_to_a_fixed_capacity(tiny_config, peptide_frame) -> None:
    model = PBG.init(tiny_config(max_seq_len=320, max_samples_factor=0.004), random.PRNGKey(0))
    capacity = model.config.max_samples_per_batch(1)
    assert capacity == 2  # ceil(0.004 * 1 row * 320 tokens)
    frame = peptide_frame("AA", environment="vacuum", temperature=300.0, energy=-100.0, logq=-50.0)

    for n_valid in (1, 2):
        batch = model.collate([[frame] * n_valid], rng=np.random.default_rng(n_valid))
        for key in ("sample_energy", "sample_temperature", "sample_logq", "sample_valid"):
            assert batch[key].shape == (capacity,)
        active = batch["sample_valid"] > 0
        assert int(active.sum()) == n_valid
        np.testing.assert_allclose(batch["sample_energy"][active], -100.0)
        np.testing.assert_allclose(batch["sample_logq"][active], -50.0)
        assert np.all(np.isnan(batch["sample_energy"][~active]))

    with pytest.raises(ValueError, match="max_samples_per_batch"):
        model.collate([[frame] * 3], rng=np.random.default_rng(0))


# ── Loss ─────────────────────────────────────────────────────────────


def test_loss_reduction_averages_over_tokens_or_molecules(tiny_config, peptide_frame, collate) -> None:
    groups = [[peptide_frame("AG"), peptide_frame("WYK", seed=1)], [peptide_frame("GG", seed=2)]]
    token = PBG.init(tiny_config(max_seq_len=512), random.PRNGKey(0))
    sample = PBG(token.qwen, replace(token.config, loss_reduction="sample"), token.vocab)
    batch = collate(token, groups)

    mask = np.asarray(batch["loss_mask"]) > 0
    nll = -np.where(mask, np.asarray(token.target_log_probs(batch)), 0.0)
    ids = np.asarray(batch["sample_id"])
    per_molecule = [nll[mask & (ids == i)].mean() for i in range(3)]

    np.testing.assert_allclose(float(token.loss(batch)), nll[mask].mean(), rtol=1e-5)
    np.testing.assert_allclose(float(sample.loss(batch)), np.mean(per_molecule), rtol=1e-5)
    assert abs(float(sample.loss(batch)) - float(token.loss(batch))) > 1e-4

    # The trainer's put_batch keeps only row-major arrays (no sample vectors).
    rows = batch["tokens"].shape[0]
    on_device = {k: v for k, v in batch.items() if v.ndim >= 1 and v.shape[0] == rows}
    assert "sample_valid" not in on_device
    np.testing.assert_allclose(float(sample.loss(on_device)), np.mean(per_molecule), rtol=1e-5)


def test_loss_metrics_apply_the_self_distillation_weight(tiny_config, peptide_frame, collate) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    frame = peptide_frame("AA", environment="vacuum", temperature=300.0, self_distillation_weight=0.5)

    metrics = model.loss_metrics(collate(model, [[frame]]))

    np.testing.assert_allclose(metrics.loss, 0.5 * metrics.nll, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(metrics.self_distillation_weight_mean, 0.5)


def test_mixed_precision_returns_finite_float32_log_probs(tiny_config, peptide_frame, collate) -> None:
    model = PBG.init(tiny_config(mixed_precision=True), random.PRNGKey(0))
    batch = collate(model, [[peptide_frame("AA", environment="vacuum", temperature=300.0)]])

    log_probs = model.target_log_probs(batch)

    assert log_probs.dtype == jnp.float32
    # Prompt and padding targets lie outside the decode support, so only the
    # trained positions carry finite log-probabilities.
    trained = batch["loss_mask"] > 0
    assert jnp.any(trained)
    assert jnp.all(jnp.isfinite(log_probs[trained]))


def test_loss_without_terminator_trains_only_coordinates(tiny_config, peptide_frame, collate) -> None:
    frame = peptide_frame("AK", seed=0)
    with_end = PBG.init(tiny_config(), random.PRNGKey(0))
    without_end = PBG(with_end.qwen, replace(with_end.config, loss_terminator=False), with_end.vocab)
    terminator = VocabSchema.of(with_end.vocab).terminator_id

    kept = collate(with_end, [[frame]])
    dropped = collate(without_end, [[frame]])

    n_coords = 3 * frame.n_atoms
    assert int(np.sum(kept["loss_mask"])) == n_coords + 1
    assert int(np.sum(dropped["loss_mask"])) == n_coords
    trained = np.asarray(dropped["loss_mask"]) > 0
    assert not np.any(np.asarray(dropped["targets"])[trained] == terminator)
    # The density of the coordinates does not change.
    np.testing.assert_array_equal(kept["post_coord_mask"], dropped["post_coord_mask"])


# ── Configuration ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"conditioning": "robin_atom"}, "unknown conditioning"),
        ({"conditioning": "atom", "atom_order": "prompt_canonical_v1"}, "source order"),
        ({"loss_reduction": "molecule"}, "unknown loss_reduction"),
        ({"loss_terminator": 0}, "loss_terminator must be a bool"),
        ({"center": "mass"}, "center must be a bool"),
        ({"center": 1}, "center must be a bool"),
        ({"center_of_mass": "yes"}, "center_of_mass must be a bool"),
    ],
)
def test_config_rejects_invalid_settings(tiny_config, overrides, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        tiny_config(**overrides)


def test_config_box_edge_is_the_bin_width_times_the_bin_gaps() -> None:
    config = PBGConfig(vocab_name="core_bin65537", bin_tokens=65537, bin_width_pm=0.05)

    assert config.box_width_nm == pytest.approx(3.2768)
    assert config.min_val == pytest.approx(-1.6384)


# ── Atom conditioning (ROBIN conditioning on the PBG trunk) ──────────


@pytest.mark.parametrize("sequence", ["AA", "WYF", "HRK", "ACDEFGHIKLMNPQRSTVWY"])
def test_atom_features_follow_the_frame_atom_order(peptide_frame, sequence) -> None:
    frame = peptide_frame(sequence)
    features = atom_features(sequence, frame.conformer[:, 0])
    encodings, _ = encode_sequences([sequence], frame.n_atoms)

    assert features.shape == (frame.n_atoms, 5)
    np.testing.assert_array_equal(features[:, 0], encodings["atom_type"][0])
    np.testing.assert_array_equal(features[:, 1], encodings["aa_type"][0])
    np.testing.assert_array_equal(features[:, 2], encodings["aa_pos"][0])
    np.testing.assert_array_equal(features[:, 3], len(sequence))
    np.testing.assert_array_equal(features[:, 4], np.arange(frame.n_atoms))


def test_atom_features_refuse_a_different_atom_order(peptide_frame) -> None:
    z = peptide_frame("AG").conformer[:, 0]
    with pytest.raises(ValueError, match="template order"):
        atom_features("AG", z[::-1])
    with pytest.raises(ValueError, match="exceeds"):
        atom_features("AG", z[:-1])
    with pytest.raises(ValueError, match="template has"):
        atom_features("AG", np.append(z, 1.0))


def test_atom_collate_drops_the_prompt_and_maps_each_position_to_its_target_atom(tiny_config, peptide_frame) -> None:
    model = PBG.init(tiny_config(conditioning="atom"), random.PRNGKey(0))
    first, second = peptide_frame("AA", seed=0), peptide_frame("GK", seed=1)
    batch = model.collate([[first, second]], rng=np.random.default_rng(0))
    schema = VocabSchema.of(model.vocab)
    n1, n2 = first.n_atoms, second.n_atoms
    second_start = 3 * n1 + 2

    # [BOS | coords | EOS] per peptide: no text prompt.
    assert batch["tokens"][0, 0] == schema.bos_id
    assert batch["tokens"][0, 1] >= schema.coord_base_id
    assert batch["tokens"][0, second_start] == schema.bos_id

    slot, sample = batch["atom_slot"][0], batch["atom_sample"][0]
    np.testing.assert_array_equal(sample[:n1], 0)
    np.testing.assert_array_equal(sample[n1 : n1 + n2], 1)
    assert np.all(sample[n1 + n2 :] == -1)
    np.testing.assert_array_equal(batch["atom_features"][0, :n1], atom_features("AA", first.conformer[:, 0]))
    np.testing.assert_array_equal(
        batch["atom_features"][0, n1 : n1 + n2], atom_features("GK", second.conformer[:, 0])
    )
    # BOS predicts atom 0; the last coordinate predicts EOS and gets no atom.
    assert slot[0] == 0 and slot[3 * n1] == -1
    assert slot[second_start] == n1
    # Every conditioned position predicts a coordinate of exactly that atom.
    positions = np.flatnonzero(slot >= 0)
    assert positions.size == 3 * (n1 + n2)
    first_slot = np.where(positions < second_start, 0, n1)
    np.testing.assert_array_equal(slot[positions] - first_slot, batch["atom_index"][0, positions + 1] - 1)


def test_atom_conditioned_peptides_cannot_see_each_others_atoms(tiny_config, peptide_frame, collate) -> None:
    model = PBG.init(tiny_config(conditioning="atom"), random.PRNGKey(0))
    batch = collate(model, [[peptide_frame("AA", seed=0), peptide_frame("GK", seed=1)]])
    n1 = int(np.sum(np.asarray(batch["atom_sample"][0]) == 0))
    features = np.asarray(batch["atom_features"]).copy()
    features[0, n1:, 1] = 1  # every residue of the second peptide becomes ALA
    changed = dict(batch, atom_features=jnp.asarray(features))

    base = np.asarray(model.target_log_probs(batch))
    other = np.asarray(model.target_log_probs(changed))
    first = np.asarray(batch["sample_id"][0]) == 0
    second = np.asarray(batch["sample_id"][0]) == 1

    np.testing.assert_allclose(base[0, first], other[0, first], rtol=0, atol=1e-6)
    assert not np.allclose(base[0, second], other[0, second])


def test_atom_conditioning_stays_causal(tiny_config, peptide_frame, collate) -> None:
    model = PBG.init(tiny_config(conditioning="atom"), random.PRNGKey(0))
    batch = collate(model, [[peptide_frame("AG", seed=0)]])
    k = 20  # a coordinate token inside the peptide
    tokens = np.asarray(batch["tokens"]).copy()
    targets = np.asarray(batch["targets"]).copy()
    tokens[0, k] = targets[0, k - 1] = tokens[0, k] + 1
    changed = dict(batch, tokens=jnp.asarray(tokens), targets=jnp.asarray(targets))

    base = np.asarray(model.target_log_probs(batch))
    other = np.asarray(model.target_log_probs(changed))

    np.testing.assert_allclose(base[0, : k - 1], other[0, : k - 1], rtol=0, atol=1e-6)
    assert not np.allclose(base[0, k:], other[0, k:])


def test_atom_conditioner_trains_and_round_trips(tiny_config, peptide_frame, collate, tmp_path) -> None:
    model = PBG.init(tiny_config(conditioning="atom", max_seq_len=128), random.PRNGKey(0))
    batch = collate(model, [[peptide_frame("AA", seed=0)], [peptide_frame("GK", seed=1)]])

    value, grads = jax.jit(jax.value_and_grad(lambda params: PBG(params, model.config, model.vocab).loss(batch)))(
        model.qwen
    )

    assert np.isfinite(float(value))
    for name in ("atom_embed", "residue_embed", "mlp_in", "mlp_out", "attn_q", "attn_kv", "attn_out"):
        assert float(jnp.abs(grads["atom_conditioner"][name]).sum()) > 0, name

    path = tmp_path / "atom.ckpt"
    model.save(path)
    loaded = PBG.load(path)
    assert loaded.config.conditioning == "atom"
    np.testing.assert_allclose(np.asarray(loaded.loss(batch)), float(value), rtol=1e-5)

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    del payload["params"]["atom_conditioner"]
    with pytest.raises(ValueError, match="atom-conditioner"):
        PBG.load_payload(payload)


def test_atom_and_prefix_runs_share_the_trunk_initialisation(tiny_config) -> None:
    atom = PBG.init(tiny_config(conditioning="atom"), random.PRNGKey(3))
    prefix = PBG.init(tiny_config(), random.PRNGKey(3))

    assert "atom_conditioner" not in prefix.qwen
    trunk = {key: value for key, value in atom.qwen.items() if key != "atom_conditioner"}
    for left, right in zip(jax.tree.leaves(trunk), jax.tree.leaves(prefix.qwen), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


# ── Sampling policy ──────────────────────────────────────────────────


@pytest.mark.parametrize("family", ["core", "joint"])
def test_sampling_policy_follows_the_checkpoint_vocabulary(family) -> None:
    name = {"core": "core_bin4096", "joint": "joint_bin4096"}[family]
    cfg = PBGConfig(
        vocab_name=name,
        prompt_format="compact_kelvin_v1" if family == "joint" else "legacy",
        atom_order="prompt_canonical_v1" if family == "joint" else "source_v1",
    )
    model = SimpleNamespace(config=cfg, vocab=load_tokeniser_default(vocab_name=name))

    assert SamplingPolicy.from_model(model).generation == family
    assert SamplingPolicy.from_model(model, family).generation == family
    with pytest.raises(ValueError, match="checkpoint uses"):
        SamplingPolicy.from_model(model, "joint" if family == "core" else "core")


@pytest.mark.parametrize(
    ("short", "target", "temperature"),
    [("water", "ff14sb-obc1-2nm", 310.0), ("vacuum", "ff14sb", 300.0)],
)
def test_joint_sampling_prompt_matches_the_training_prompt(short, target, temperature) -> None:
    cfg = PBGConfig(vocab_name="joint_bin4096", prompt_format="compact_kelvin_v1", atom_order="prompt_canonical_v1")
    vocab = load_tokeniser_default(vocab_name=cfg.vocab_name)
    model = SimpleNamespace(config=cfg, vocab=vocab)
    policy = SamplingPolicy.from_model(model)
    system = MolecularSystem.from_peptide("NEVI", environment=short, temperature_K=temperature)
    with pytest.raises(ValueError, match="registered energy protocol"):
        policy.system(system)

    system = policy.system(system, target_id=target)
    prompt, *_, z = prompt_arrays(model, system, 1)
    frame = DataFrame(
        smiles=system.mapped_smiles,
        helm=system.helm,
        sequence="NEVI",
        environment=short,
        temperature=temperature,
        conformer=np.column_stack((z, np.zeros((len(z), 3)))),
        metadata={"n_solute_atoms": len(z), "n_waters": 0, "target_id": target},
    )
    tokeniser = make_tokeniser(vocab, prompt_format=cfg.prompt_format)
    training = fit_frame(frame, cfg, tokeniser, cfg.max_seq_len)
    expected = tokeniser.encode_prompt(training.helm, environment=training.environment, temperature_K=training.temperature)

    np.testing.assert_array_equal(prompt[0, 1:], expected)
    with pytest.raises(ValueError, match="disagree"):
        policy.system(replace(system, environment="water, ff14sb"))


def test_core_sampling_keeps_the_literal_checkpoint_prompt() -> None:
    system = MolecularSystem.from_peptide("AA", environment="310 K, water", temperature_K=310.0)

    assert SamplingPolicy("core").system(system) is system
    with pytest.raises(ValueError, match="checkpoint-specific environment prompt"):
        SamplingPolicy("core").system(system, target_id="ff14sb")
