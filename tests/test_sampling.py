"""Sampling: prompts, KV-cached decoding, stereo repair, explicit-water energies,
the batch-size tuner, Leviathan speculative decoding and ROBIN."""

from __future__ import annotations

import pickle
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Geometry import Point3D

import pbg.models.pbg.rollout as rollout
import pbg.models.pbg.sample as sample
from pbg.core.frame import DataFrame
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_smiles
from pbg.core.stereo import (
    compile_stereo_spec,
    intended_stereo_mask,
    stereo_groups,
    stereo_keys,
)
from pbg.core.tokeniser import (
    load_tokeniser_default,
    make_tokeniser,
    undo_tokeniser_coordinate,
)
from pbg.energy import expected_solvated_smiles
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.model import PBG
from pbg.models.pbg.sample import SampleBatch
from pbg.models.pbg.system import (
    MolecularSystem,
    checkpoint_environment_prompt,
    prompt_arrays,
)
from pbg.models.robin import Robin, RobinConfig, sample_robin_conformers
from pbg.models.robin.cgpt import apply_model
from pbg.models.robin.conditioning import encode_sequences
from pbg.models.speculation.decode import (
    _propose_compact,
    decode_compiled_batched_junior_ar,
)
from pbg.models.speculation.sample import _assert_compatible_vocabs

# ── Prompts and checkpoint metadata ──────────────────────────────────

LONG_PROMPT = "Amber-ff14SB in OBC1 implicit water"
VACUUM_PROMPT = "isolated peptide in vacuum"


@pytest.mark.parametrize(
    ("sources", "label", "expected"),
    [
        ([{"environment_prompt": {"water": "water"}}], "water", "water"),
        ([{"environment_prompt": {"water": LONG_PROMPT}}], "water", LONG_PROMPT),
        ([{"environment_prompt": {"water": LONG_PROMPT, "vacuum": VACUUM_PROMPT}}], "water", LONG_PROMPT),
        ([{"environment_prompt": {"water": LONG_PROMPT, "vacuum": VACUUM_PROMPT}}], "vacuum", VACUUM_PROMPT),
        ([{"environment_prompt": {"vacuum": VACUUM_PROMPT}}, {"environment_prompt": {"water": LONG_PROMPT}}], "water", LONG_PROMPT),
        ([{"environment_prompt": {"implicit": LONG_PROMPT}}], "water", LONG_PROMPT),
    ],
    ids=["short", "long", "labelled", "other-label", "across-sources", "sole-prompt"],
)
def test_checkpoint_environment_prompt_recovers_the_training_wording(tmp_path, sources, label, expected) -> None:
    """The prompt is encoded verbatim, so sampling must reuse the training wording."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(pickle.dumps({"metadata": {"parameters": {"data": {"sources": sources}}}}))

    assert checkpoint_environment_prompt(checkpoint, label=label) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {"parameters": {"data": {"sources": [{"environment_prompt": {"implicit": LONG_PROMPT, "vacuum": VACUUM_PROMPT}}]}}},
        None,
        {"parameters": {"data": {"sources": [{"name": "many_peptides"}]}}},
    ],
    ids=["ambiguous", "no-metadata", "no-mapping"],
)
def test_checkpoint_environment_prompt_refuses_to_guess(tmp_path, metadata) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(pickle.dumps({"format_version": 2} if metadata is None else {"metadata": metadata}))

    with pytest.raises(KeyError):
        checkpoint_environment_prompt(checkpoint)


def test_compact_kelvin_sampling_prompt_keeps_the_physical_temperature() -> None:
    config = PBGConfig(vocab_name="core_bin4096", prompt_format="compact_kelvin_v1")
    vocab = load_tokeniser_default(vocab_name="core_bin4096")
    system = MolecularSystem(
        id="test",
        mapped_smiles="[H:1][H:2]",
        helm="CHEM1{[[H][H]]}$$$$",
        environment="vacuum, gaff2-am1bcc",
        temperature_K=373.421,
    )

    prompt, *_ = prompt_arrays(SimpleNamespace(config=config, vocab=vocab), system, 1)

    expected = make_tokeniser(vocab, prompt_format=config.prompt_format).encode_prompt(
        system.helm, environment=system.environment, temperature_K=system.temperature_K
    )
    np.testing.assert_array_equal(prompt[0, 1:], expected)
    assert system.temperature_K == 373.421


# ── Decoding ─────────────────────────────────────────────────────────


def test_atom_conditioned_sampling_logq_equals_the_training_density(tiny_config, collate, monkeypatch) -> None:
    """KV-cached atom-conditioned decoding replays the training forward pass."""
    model = PBG.init(tiny_config(conditioning="atom", num_layers=2, center=False), random.PRNGKey(0))
    # Strengthen the conditioner so that a decoder which drops it would fail.
    conditioner = dict(model.qwen["atom_conditioner"])
    for name in ("mlp_out", "attn_out"):
        conditioner[name] = conditioner[name] * 20.0
    model = PBG(dict(model.qwen, atom_conditioner=conditioner), model.config, model.vocab)
    system = MolecularSystem.from_peptide("AK", environment="water", temperature_K=310.0)

    batch = sample.sample_molecular_conformers(model, system, random.PRNGKey(1), num_samples=3, batch_size=3)

    # Sampling normalises over coordinate bins only; score with the same support.
    monkeypatch.setattr(PBG, "_decode_clamp", lambda self: sample._coordinate_decode_bias(self))
    frames = [
        DataFrame(conformer=conformer.astype(np.float32), smiles=system.mapped_smiles, sequence="AK")
        for conformer in batch.samples
    ]
    scored = collate(model, [[frame] for frame in frames])
    coords = np.asarray(scored["post_coord_mask"]) > 0
    teacher_forced = np.where(coords, np.asarray(model.target_log_probs(scored)), 0.0).sum(axis=1)
    np.testing.assert_allclose(batch.logq, teacher_forced, rtol=0, atol=2e-4)

    # The same tokens are much less likely without the conditioner's attention.
    blind = dict(conditioner, attn_out=jnp.zeros_like(conditioner["attn_out"]))
    other = PBG(dict(model.qwen, atom_conditioner=blind), model.config, model.vocab)
    blind_logq = np.where(coords, np.asarray(other.target_log_probs(scored)), 0.0).sum(axis=1)
    assert np.max(np.abs(blind_logq - batch.logq)) > 1e-2


def test_batch_tuner_scans_past_a_local_throughput_drop(monkeypatch) -> None:
    monkeypatch.setattr(rollout, "require_gpu_backend", lambda: None)
    monkeypatch.setattr(rollout.jax, "clear_caches", lambda: None)
    monkeypatch.setattr(rollout.gc, "collect", lambda: None)
    rates = {16: 16.0, 32: 20.0, 64: 18.0, 128: 30.0}
    timestamps = iter(t for batch_size in rates for t in (0.0, batch_size / rates[batch_size]))
    monkeypatch.setattr(rollout.time, "perf_counter", lambda: next(timestamps))
    calls: list[int] = []
    monkeypatch.setattr(rollout, "sample_molecular_conformers", lambda *args, **kwargs: calls.append(kwargs["batch_size"]))

    selected, trials = rollout.tune_sampling_batch_size(object(), object(), jax.random.key(0), candidates=tuple(rates))

    assert selected == 128
    assert [trial.batch_size for trial in trials] == list(rates)
    assert calls == [batch_size for batch_size in rates for _ in range(2)]


# ── Stereochemistry ──────────────────────────────────────────────────


def _mapped_mol(smiles: str) -> Chem.Mol:
    """Explicit-H molecule whose atom order follows the SMILES atom maps."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    by_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    return rdmolops.RenumberAtoms(mol, [by_map[i] for i in range(1, mol.GetNumAtoms() + 1)])


@pytest.fixture(scope="module")
def aa_frames() -> np.ndarray:
    """Embedded Zxyz frames of the intended (L,L) AA dipeptide."""
    mol = _mapped_mol(sequence_to_smiles("AA"))
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=8, randomSeed=0)
    assert len(conf_ids) == 8
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.float64)
    return np.stack([np.concatenate([z[:, None], mol.GetConformer(i).GetPositions()], axis=1) for i in conf_ids])


def test_stereo_keys_match_rdkit_cip_labels(aa_frames) -> None:
    smiles = sequence_to_smiles("AA")
    mirrored = aa_frames.copy()
    mirrored[:, :, 1:4] *= -1.0
    frames = np.concatenate([aa_frames, mirrored], axis=0)
    spec = compile_stereo_spec(smiles)

    config, resolved = (np.asarray(value) for value in jax.jit(stereo_keys)(spec, frames[:, :, 1:4]))

    template = _mapped_mol(smiles)
    centers = [int(atom) for atom in np.asarray(spec.atoms)[:, 0]]
    expected_config = np.zeros(frames.shape[0], dtype=np.uint32)
    expected_resolved = np.zeros_like(expected_config)
    for frame_index, frame in enumerate(frames):
        mol = Chem.Mol(template)
        mol.RemoveAllConformers()
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom_index, (x, y, z) in enumerate(frame[:, 1:4]):
            conformer.SetAtomPosition(atom_index, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conformer, assignId=True)
        Chem.AssignStereochemistryFrom3D(mol)
        for center_index, atom_index in enumerate(centers):
            atom = mol.GetAtomWithIdx(atom_index)
            cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
            if cip in {"R", "S"}:
                expected_resolved[frame_index] |= np.uint32(1 << center_index)
                if cip == "S":
                    expected_config[frame_index] |= np.uint32(1 << center_index)
    np.testing.assert_array_equal(config, expected_config)
    np.testing.assert_array_equal(resolved, expected_resolved)


def test_stereo_keys_are_rigid_motion_invariant_and_reflection_sensitive(aa_frames) -> None:
    xyz = aa_frames[:, :, 1:4]
    spec = compile_stereo_spec(sequence_to_smiles("AA"))
    rotation, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    original = tuple(np.asarray(x) for x in stereo_keys(spec, xyz))
    moved = tuple(np.asarray(x) for x in stereo_keys(spec, (xyz @ rotation) * 2.3 + 7.0))
    reflected = tuple(np.asarray(x) for x in stereo_keys(spec, xyz * np.array([-1.0, 1.0, 1.0])))

    assert all(np.array_equal(a, b) for a, b in zip(original, moved, strict=True))
    assert np.array_equal(original[1], reflected[1])
    assert np.array_equal(reflected[0], original[0] ^ original[1])


def test_stereo_repair_folds_only_full_enantiomers(aa_frames) -> None:
    smiles = sequence_to_smiles("AA")
    mirrored = aa_frames[:4].copy()
    mirrored[:, :, 1:4] *= -1.0

    repaired, repair_mask, fold_mask = rollout.repair_all_mirrored_stereo(
        np.concatenate([aa_frames[:4], mirrored], axis=0), smiles
    )

    config, resolved = (np.asarray(value) for value in stereo_keys(compile_stereo_spec(smiles), repaired[:, :, 1:4]))
    assert repair_mask.tolist() == [False] * 4 + [True] * 4
    assert np.all(fold_mask)
    assert np.all(resolved == 3)
    assert np.all(config == 3)


@pytest.mark.parametrize("mirrored", [[False, False], [False, True]])
def test_stereo_repair_reflects_coordinates_and_keeps_the_generated_logq(monkeypatch, tmp_path, mirrored) -> None:
    """Reflection changes coordinates only; it never rescores a conformer."""
    original = np.array([[[6.0, 1.0, 2.0, 3.0]], [[6.0, -1.0, -2.0, -3.0]]])
    repaired = original.copy()
    repaired[np.asarray(mirrored), :, 1:] *= -1
    system = MolecularSystem(
        id="x", mapped_smiles="[C:1]", helm="CHEM1{[C]}$$$$", environment="vacuum", temperature_K=300.0
    )
    monkeypatch.setattr(rollout, "repair_all_mirrored_stereo", lambda *args: (repaired, np.asarray(mirrored), np.ones(2, bool)))
    exported = {}

    def build(system, batch, energies, *, stereo_repair_mask, **kwargs):
        exported.update(batch=batch, mask=stereo_repair_mask)
        return ()

    monkeypatch.setattr(rollout, "build_sampled_ensembles", build)
    batch = SampleBatch(original, np.log([0.2, 0.3]), np.zeros((2, 3)), np.zeros((2, 3), dtype=np.int32))

    # The model is never called: an object without methods stands in for it.
    rollout.export_sample_batch(
        SimpleNamespace(config=None), system, batch, energy_scorer=lambda system, samples: np.zeros(len(samples)),
        output_dir=tmp_path, repair_all_mirrored=True,
    )

    result, repair_mask = exported["batch"], exported["mask"]
    assert repair_mask.tolist() == mirrored
    np.testing.assert_array_equal(result.samples, repaired)
    assert result.logq is batch.logq
    np.testing.assert_array_equal(batch.samples, original)
    if any(mirrored):
        # Token traces describe the pre-repair coordinates and are dropped.
        assert result.behavior_token_log_probs is None
        assert result.behavior_token_ids is None


def test_stereo_groups_count_the_keys() -> None:
    config = np.array([1, 0, 1, 3], dtype=np.uint32)
    resolved = np.array([3, 3, 3, 3], dtype=np.uint32)

    unique_config, unique_resolved, inverse, counts = stereo_groups(config, resolved)

    assert unique_config.tolist() == [0, 1, 3]
    assert unique_resolved.tolist() == [3, 3, 3]
    assert counts.tolist() == [1, 2, 1]
    assert inverse.tolist() == [1, 0, 1, 2]


def test_stereo_spec_compiles_from_unmapped_smiles() -> None:
    assert compile_stereo_spec("CC(O)C(=O)O").atoms.shape == (1, 4)


def test_intended_stereo_mask_rejects_mirrored_unresolved_and_non_finite_frames() -> None:
    smiles = "[F:1][C@:2]([Cl:3])([Br:4])[I:5]"
    xyz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float32)
    mirror = xyz * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    nonfinite = xyz.copy()
    nonfinite[0, 0] = np.nan

    mask = intended_stereo_mask(smiles, np.stack([xyz, mirror, np.zeros_like(xyz), nonfinite]))

    assert mask[:2].sum() == 1
    assert not mask[2]
    assert not mask[3]


# ── Explicit-water energies ──────────────────────────────────────────


def _solvated_frame(zxyz: np.ndarray, *, n_waters: int) -> DataFrame:
    return DataFrame(
        id="AA",
        frame_id=0,
        smiles=expected_solvated_smiles("AA", n_waters),
        sequence="AA",
        helm="",
        environment="water",
        temperature=300.0,
        conformer=np.asarray(zxyz, dtype=np.float64),
    )


def test_energy_frames_rejects_extra_atoms_that_are_not_whole_waters() -> None:
    peptide = atom_numbers_from_smiles(sequence_to_smiles("AA")).astype(np.float64)
    zxyz = np.zeros((peptide.size + 1, 4), dtype=np.float64)
    zxyz[:, 0] = 1.0
    zxyz[: peptide.size, 0] = peptide

    with pytest.raises(ValueError, match="cannot infer n_waters"):
        rollout.energy_frames([_solvated_frame(zxyz, n_waters=0)], sequence="AA", environment="water")


def test_energy_frames_scores_microsolvated_frames_with_amber14_tip3p() -> None:
    pytest.importorskip("openmm")
    peptide = atom_numbers_from_smiles(sequence_to_smiles("AA")).astype(np.float64)
    n = peptide.size
    zxyz = np.zeros((n + 3, 4), dtype=np.float64)
    zxyz[:n, 0] = peptide
    # The geometry need only be finite: the topology and atom count are under test.
    zxyz[:n, 1:4] = np.random.default_rng(0).normal(scale=0.05, size=(n, 3))
    zxyz[n:] = [[8.0, 0.4, 0.0, 0.0], [1.0, 0.4, 0.1, 0.0], [1.0, 0.4, -0.05, 0.08]]

    energies = rollout.energy_frames(
        [_solvated_frame(zxyz, n_waters=1)], sequence="AA", environment="water", platform_name="CPU"
    )

    assert energies.shape == (1,)
    assert np.isfinite(energies[0])


# ── Leviathan speculative decoding ───────────────────────────────────


def test_argmax_proposal_uses_a_one_hot_proposal_law() -> None:
    probabilities = jnp.array([[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]])

    tokens, proposal_law = _propose_compact(random.PRNGKey(2), probabilities, "argmax")

    np.testing.assert_array_equal(tokens, [1, 0])
    np.testing.assert_array_equal(proposal_law, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def _peak_forward(peak: int):
    """A forward pass whose logits always put all mass on one compact token."""

    def forward(tokens, proposal_mask, atom_index, level_index, sequence_index, caches, cache_index):
        batch, width = tokens.shape
        cache = jnp.zeros((batch, 1, 32), dtype=jnp.int32) if caches is None else caches
        indices = cache_index[:, None] + jnp.arange(width)[None]
        cache = cache.at[jnp.arange(batch)[:, None], 0, indices].set(tokens)
        logits = jnp.full((batch, width, 3), -20.0).at[:, :, peak].set(20.0)
        return logits, cache

    return forward


@pytest.mark.parametrize(("target_peak", "token"), [(0, 1), (1, 2)], ids=["agreeing-target", "disagreeing-target"])
def test_batched_junior_decoder_accepts_agreeing_drafts_and_rejects_to_the_target(target_peak, token) -> None:
    draft = _peak_forward(0)
    target = draft if target_peak == 0 else _peak_forward(target_peak)
    common = dict(
        prompt_tokens=jnp.array([0, 0], dtype=jnp.int32),
        prompt_atom_index=jnp.array([0, 0], dtype=jnp.int32),
        prompt_level_index=jnp.array([0, 0], dtype=jnp.int32),
        coordinate_atom_index=jnp.zeros((6,), dtype=jnp.int32),
        coordinate_level_index=jnp.zeros((6,), dtype=jnp.int32),
        legal_token_ids=jnp.array([1, 2, 3], dtype=jnp.int32),
        num_proposals=2,
        compact_logits=True,
        proposal_policy="argmax",
    )

    result = jax.jit(lambda keys: decode_compiled_batched_junior_ar(keys, draft, target, **common))(
        random.split(random.PRNGKey(3), 4)
    )

    np.testing.assert_array_equal(result.tokens, np.full((4, 6), token))
    if target_peak == 0:
        assert jnp.array_equal(result.accepted, result.proposed)
        assert jnp.all(result.forwards < 2 + 6 * 4)
    else:
        assert jnp.all(result.accepted == 0)
        assert jnp.all(result.proposed > 0)


def test_speculative_decoding_needs_the_same_coordinate_vocabulary(monkeypatch) -> None:
    class Model:
        def __init__(self, n_coordinates: int) -> None:
            self.vocab = object()
            self.n_coordinates = n_coordinates
            self.config = type("Config", (), {"bin_tokens": 4096, "coordinate_method": "bin"})()

    draft, target = Model(3), Model(4)

    def schema_of(vocab):
        n = draft.n_coordinates if vocab is draft.vocab else target.n_coordinates
        return type("Schema", (), {"coord_ids": tuple(range(n))})()

    monkeypatch.setattr("pbg.models.speculation.sample.VocabSchema.of", staticmethod(schema_of))

    with pytest.raises(ValueError, match="coordinate vocabulary"):
        _assert_compatible_vocabs(draft, target)  # type: ignore[arg-type]


# ── ROBIN ────────────────────────────────────────────────────────────


@pytest.fixture
def robin():
    """Random ROBIN (CGPT) parameters in the converted checkpoint layout."""
    vocab, dim, layers, hidden, cond_hidden, max_atoms = 16, 8, 2, 16, 16, 32
    keys = iter(jax.random.split(jax.random.key(1), 32))

    def normal(*shape):
        return jax.random.normal(next(keys), shape) * 0.1

    def norm():
        return {"scale": jnp.ones(dim)}

    params = {
        "token_embed": normal(vocab, dim),
        "pos_embed": normal(max_atoms * 3 + 1, dim),
        "coord_dim_embed": normal(3, dim),
        "atom_pos_embed": normal(max_atoms, dim),
        "cond_embed": {
            "atom_embed": normal(55, dim),
            "residue_embed": normal(21, dim),
            "proj": {
                "w1": normal(5 * dim, cond_hidden),
                "b1": jnp.zeros(cond_hidden),
                "w2": normal(cond_hidden, dim),
                "b2": jnp.zeros(dim),
            },
        },
        "global_cond_attn": {
            "q_w": normal(dim, dim),
            "q_b": jnp.zeros(dim),
            "kv_w": normal(dim, 2 * dim),
            "kv_b": jnp.zeros(2 * dim),
            "proj_w": normal(dim, dim),
            "proj_b": jnp.zeros(dim),
        },
        "blocks": [
            {
                "norm1": norm(),
                "attn": {
                    "qkv_w": normal(dim, 3 * dim),
                    "qkv_b": jnp.zeros(3 * dim),
                    "proj_w": normal(dim, dim),
                    "proj_b": jnp.zeros(dim),
                },
                "norm2": norm(),
                "ff": {"w1": normal(dim, hidden), "w2": normal(dim, hidden), "w3": normal(hidden, dim), "b3": jnp.zeros(dim)},
            }
            for _ in range(layers)
        ],
        "final_norm": norm(),
        "output_proj_w": normal(dim, vocab),
    }
    return Robin(params, RobinConfig(vocab, 100.0, dim, 2, layers, 2.0, max_atoms))


def test_robin_cached_logq_matches_teacher_forcing(robin) -> None:
    robin = replace(robin, config=replace(robin.config, max_num_atoms=23))
    batch = jax.tree.map(jnp.asarray, robin.collate([DataFrame.from_sequence("AA")]))

    tokens, logq = jax.jit(lambda key: robin.predict(batch, key, 69))(jax.random.key(9))

    ids = jnp.concatenate((jnp.zeros((1, 1), dtype=jnp.int32), tokens[:, :-1]), axis=1)
    logits = apply_model(robin.params, {"dim": 8, "num_heads": 2}, ids, batch, batch["mask"])
    expected = jnp.take_along_axis(jax.nn.log_softmax(logits), tokens[..., None], axis=-1).sum(axis=(1, 2))
    np.testing.assert_allclose(logq, expected, rtol=2e-5)


def test_robin_checkpoint_loads_and_samples(robin, tmp_path) -> None:
    config = dict(vars(robin.config))
    config.pop("bin_width_pm")
    config.update(min_val=-1.0, max_val=1.0, standardize_std=0.75, binning="linear_scale")
    checkpoint = tmp_path / "converted.ckpt"
    checkpoint.write_bytes(pickle.dumps({"config": config, "params": jax.device_get(robin.params)}))

    loaded = Robin.load(checkpoint)
    system = MolecularSystem.from_peptide("AA", environment="water", temperature_K=310.0)
    batch = sample_robin_conformers(loaded, system, jax.random.key(4), num_samples=3, batch_size=2)

    assert loaded.config.bin_width_pm == 100.0
    assert batch.samples.shape == (3, 23, 4)
    assert batch.logq.shape == (3,)
    assert np.isfinite(batch.samples).all() and np.isfinite(batch.logq).all()
    too_long = MolecularSystem.from_peptide("IDHRQLKW", environment="water", temperature_K=310.0)
    with pytest.raises(ValueError, match="positional embedding"):
        sample_robin_conformers(loaded, too_long, jax.random.key(4), num_samples=1)


def test_robin_decodes_bin_centres_and_rejects_unsupported_inputs(robin) -> None:
    tokens = jnp.arange(69)[None, :] % 16

    result = robin.decollate(tokens, np.array([69]))[0]

    expected = undo_tokeniser_coordinate(
        np.asarray(tokens).reshape(1, 23, 3),
        "bin",
        num_bins=16,
        min_val=robin.config.min_val,
        max_val=robin.config.max_val,
        decode_method="center",
    )
    np.testing.assert_array_equal(result, np.asarray(expected)[0])
    with pytest.raises(ValueError, match="Unsupported ROBIN peptide"):
        encode_sequences(["dAA"], 23)
    with pytest.raises(TypeError):
        RobinConfig.from_checkpoint(vars(robin.config) | {"unknown_architecture_flag": True})
