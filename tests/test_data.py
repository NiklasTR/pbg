"""Data: tar adapters and frame iterators, the data cursor, loader workers,
molecular preparation, augmentation and ensemble files."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import fields, replace
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

import pbg.data.webdataset as webdataset
import pbg.models.pbg.loader as loader
from pbg.core.augmenter import AugmentationBoundsError, apply_augmenter, init_augmenter
from pbg.core.ensemble import Ensemble, encode_ensemble_metadata
from pbg.core.environment import environment_description
from pbg.core.frame import DataFrame
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_helm, sequence_to_smiles
from pbg.core.tokeniser import (
    load_tokeniser_default,
    make_tokeniser,
)
from pbg.data.joint import fit_frame, select_representation
from pbg.data.webdataset import iter_frames_from_tar
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.loader import (
    BATCH_DONE_PASS_KEY,
    BATCH_DONE_SHARD_KEY,
    DataCursorTracker,
    augmenter_from_config,
    catch_up_shard_work,
    iter_assigned_tar_shards,
    make_data_cursor_tracker,
    parse_saved_cursor,
    split_cursor_batch,
    tar_fingerprint,
)

_SMILES = "[H:1][H:2]"
_BIN = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.1, 0.0, 0.0]], dtype=np.float32).tobytes()
_METADATA = {
    "smiles": _SMILES,
    "environment": "implicit solvent",
    "temperature": 300.0,
    "helm": "CHEM1{[H][H]}$$$$",
    "frame_idx": 7,
    "custom": "retained",
}


def _write_tar(path: Path, members) -> Path:
    with tarfile.open(path, "w") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def _n4_frame(tmp_path: Path, sample: dict[str, bytes], **kwargs) -> DataFrame:
    """Decode one n4_smiles sample stored as sibling tar members ``AA_000001{suffix}``."""
    tar = _write_tar(tmp_path / "n4.tar", [(f"AA_000001{suffix}", payload) for suffix, payload in sample.items()])
    (frame,) = iter_frames_from_tar(tar, adapter="n4_smiles", **kwargs)
    return frame


def _peptide_tar(path: Path, names: list[str]) -> Path:
    """Write Many Peptides members ``{sequence}_{frame}.bin`` with zero coordinates."""
    atoms = {name: len(atom_numbers_from_smiles(sequence_to_smiles(name.split("_")[0]))) for name in names}
    return _write_tar(path, [(f"{name}.bin", np.zeros((atoms[name], 3), np.float32).tobytes()) for name in names])


# ── n4_smiles adapter ────────────────────────────────────────────────


def test_n4_smiles_adapter_reads_the_json_record(tmp_path) -> None:
    record = {
        "smiles": _SMILES,
        "environment": "vacuum",
        "temperature": 300.0,
        "energy": -1.5,
        "logq": -0.25,
        "self_distillation_weight": 0.4,
        "generator_checkpoint": "teacher",
        "alpha": 0.8,
    }

    frame = _n4_frame(tmp_path, {".bin": _BIN, ".json": json.dumps(record).encode()})

    assert frame.frame_id == 1
    for key, value in record.items():
        assert getattr(frame, key) == value, key


def test_n4_smiles_adapter_resolves_smiles_and_environment_by_precedence(tmp_path) -> None:
    # Sibling files fill in what the JSON record lacks.
    frame = _n4_frame(
        tmp_path,
        {
            ".bin": _BIN,
            ".smiles": _SMILES.encode(),
            ".environment": b"water",
            ".json": json.dumps({"temperature": 310.0}).encode(),
        },
    )
    assert (frame.smiles, frame.environment, frame.temperature) == (_SMILES, "water", 310.0)

    # The JSON record wins over sibling files, and an explicit argument wins over both.
    sample = {
        ".bin": _BIN,
        ".smiles": b"[H:1][H:2][H:3]",
        ".environment": b"water",
        ".json": json.dumps({"smiles": _SMILES, "environment": "vacuum"}).encode(),
    }
    frame = _n4_frame(tmp_path, sample)
    assert (frame.smiles, frame.environment) == (_SMILES, "vacuum")
    assert _n4_frame(tmp_path, sample, environment="water").environment == "water"


def test_n4_smiles_adapter_requires_smiles(tmp_path) -> None:
    with pytest.raises(ValueError, match="smiles"):
        _n4_frame(tmp_path, {".bin": _BIN, ".json": b"{}"})
    tar = _write_tar(tmp_path / "skip.tar", [("AA_000001.bin", _BIN), ("AA_000001.json", b"{}")])
    assert list(iter_frames_from_tar(tar, adapter="n4_smiles", on_decode_error="skip")) == []


@pytest.mark.parametrize(
    ("key", "sidecars", "environment"),
    [
        ("AA_000001", {".environment": b"vacuum"}, "vacuum"),
        ("06057_014588", {}, None),  # MDQM9nc shards carry only .bin and .smiles
    ],
)
def test_iter_frames_reads_shards_of_sibling_files(tmp_path, key, sidecars, environment) -> None:
    members = [(f"{key}.bin", _BIN), (f"{key}.smiles", _SMILES.encode())]
    members += [(f"{key}{suffix}", payload) for suffix, payload in sidecars.items()]
    tar = _write_tar(tmp_path / "0000.tar", members)

    frames = list(
        iter_frames_from_tar(tar, adapter="n4_smiles", environment=None, temperature=300.0)
    )

    assert len(frames) == 1
    assert frames[0].smiles == _SMILES
    assert frames[0].environment == environment
    assert frames[0].temperature == 300.0


# ── Strict JSON/bin pair shards ──────────────────────────────────────


def test_repeated_smiles_decode_to_identical_frames(tmp_path) -> None:
    # Later samples reuse the topology that the first sample of the shard cached.
    members = []
    for index in range(10):
        members += [(f"key-{index:02d}.json", json.dumps(_METADATA).encode()), (f"key-{index:02d}.bin", _BIN)]
    tar = _write_tar(tmp_path / "pairs.tar", members)

    first, *rest = iter_frames_from_tar(tar, adapter="n4_smiles", sample_layout="n4_json_pair_v1")

    assert len(rest) == 9
    for left, right in ((first, frame) for frame in rest):
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            else:
                assert a == b, field.name


def test_strict_pair_reader_streams_before_a_later_shard_error(tmp_path) -> None:
    tar = _write_tar(
        tmp_path / "pairs.tar",
        [
            ("first.json", json.dumps(_METADATA).encode()),
            ("first.bin", _BIN),
            ("second.json", json.dumps(_METADATA).encode()),
            ("wrong.bin", _BIN),
        ],
    )

    frames = iter_frames_from_tar(tar, adapter="n4_smiles", sample_layout="n4_json_pair_v1")
    first = next(frames)

    assert first.frame_id == 7
    assert first.helm == _METADATA["helm"]
    assert first.environment == "implicit solvent"
    assert first.temperature == 300.0
    assert first.metadata == {"frame_idx": 7, "custom": "retained"}
    with pytest.raises(ValueError, match="must be followed"):
        next(frames)


@pytest.mark.parametrize(
    ("members", "match"),
    [
        ([("a.json", json.dumps(_METADATA).encode()), ("a.smiles", _SMILES.encode())], "must be followed"),
        (
            [("a.json", json.dumps(_METADATA).encode()), ("a.bin", _BIN), ("a.json", json.dumps(_METADATA).encode())],
            "duplicate",
        ),
        ([("a.bin", _BIN), ("a.json", json.dumps(_METADATA).encode())], "expects consecutive"),
        ([("a.json", json.dumps({"temperature": 300}).encode()), ("a.bin", _BIN)], "required fields"),
    ],
    ids=["missing-bin", "duplicate-key", "bin-first", "no-identity"],
)
def test_strict_pair_reader_rejects_malformed_shards(tmp_path, members, match) -> None:
    tar = _write_tar(tmp_path / "malformed.tar", members)

    with pytest.raises(ValueError, match=match):
        list(iter_frames_from_tar(tar, adapter="n4_smiles", sample_layout="n4_json_pair_v1"))


def test_strict_pair_reader_rejects_an_unknown_layout_or_adapter(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown sample layout"):
        list(iter_frames_from_tar(tmp_path / "unused.tar", sample_layout="future"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires adapter"):
        list(iter_frames_from_tar(tmp_path / "unused.tar", adapter="many_peptides", sample_layout="n4_json_pair_v1"))


# ── Many Peptides subsets: frame_id_below and sequence_allowlist ─────


@pytest.fixture
def decoded_keys(monkeypatch) -> list[str]:
    """Record the key of every decoded Many Peptides frame and yield the key in place of the frame."""
    decoded: list[str] = []

    def frame(**kwargs) -> str:
        decoded.append(f"{kwargs['sequence']}_{kwargs['frame_id']:06d}")
        return decoded[-1]

    monkeypatch.setattr(webdataset, "DataFrame", frame)
    return decoded


def test_frame_id_below_keeps_an_early_trajectory_prefix_before_decoding(tmp_path, decoded_keys) -> None:
    names = ["AA_000000", "GW_003120", "AA_003125", "GW_199990"]
    tar = _peptide_tar(tmp_path / "0000.tar", names)
    assert list(iter_frames_from_tar(tar)) == names
    decoded_keys.clear()

    kept = list(iter_frames_from_tar(tar, frame_id_below=3125))

    assert kept == ["AA_000000", "GW_003120"]
    assert decoded_keys == kept


def test_sequence_allowlist_keeps_whole_systems_before_decoding(tmp_path, decoded_keys) -> None:
    names = ["AA_000000", "GW_003120", "AA_003125", "KLLR_000010"]
    tar = _peptide_tar(tmp_path / "0000.tar", names)
    allowlist = tmp_path / "subset.txt"
    allowlist.write_text("AA\nKLLR\n\n")

    kept = list(iter_frames_from_tar(tar, sequence_allowlist=allowlist))

    assert kept == ["AA_000000", "AA_003125", "KLLR_000010"]
    assert decoded_keys == kept
    both = list(
        iter_frames_from_tar(tar, sequence_allowlist=str(allowlist), frame_id_below=3125)
    )
    assert both == ["AA_000000", "KLLR_000010"]


@pytest.mark.parametrize(
    ("name", "kwargs", "match"),
    [
        ("AA.bin", {"frame_id_below": 10}, "frame id"),
        *(("AA_000000.bin", {"frame_id_below": value}, "positive integer") for value in (0, -5, 2.5, True)),
        ("AA_000000.bin", {"frame_id_below": 10, "adapter": "n4_smiles"}, "many_peptides"),
        ("AA_000000.bin", {"sequence_allowlist": "\n"}, "empty"),
        ("AA_000000.bin", {"sequence_allowlist": "AA\n", "adapter": "n4_smiles"}, "many_peptides"),
    ],
)
def test_subset_options_reject_invalid_inputs(tmp_path, decoded_keys, name, kwargs, match) -> None:
    tar = _write_tar(tmp_path / "0000.tar", [(name, _BIN)])
    if "sequence_allowlist" in kwargs:
        allowlist = tmp_path / "subset.txt"
        allowlist.write_text(kwargs["sequence_allowlist"])
        kwargs = dict(kwargs, sequence_allowlist=allowlist)

    with pytest.raises(ValueError, match=match):
        list(iter_frames_from_tar(tar, **kwargs))


# ── Data cursor ──────────────────────────────────────────────────────


def test_parse_saved_cursor_accepts_only_version_one() -> None:
    assert parse_saved_cursor(None) is None
    assert parse_saved_cursor({}) is None
    assert parse_saved_cursor({"version": 2, "sources": {}}) is None
    assert parse_saved_cursor({"version": 1, "sources": "nope"}) is None
    parsed = parse_saved_cursor(
        {
            "version": 1,
            "sources": {"many": {"fingerprint": "abc", "n_shards": 2, "shard_pass": [0, 4]}},
            "mix_rng_by_process": {"0": {"bit_generator": "unused"}},
        }
    )
    assert parsed["sources"]["many"]["shard_pass"] == [0, 4]


def test_cold_start_assignments_stride_over_the_tar_shards() -> None:
    paths = tuple(Path(f"{index}.tar") for index in range(8))

    assigned = islice(iter_assigned_tar_shards(paths, worker_id=1, num_workers=4), 4)

    assert [shard for _pass, shard, _path in assigned] == [1, 5, 1, 5]


def test_catch_up_skips_completed_shards_before_restriping() -> None:
    due = catch_up_shard_work((1, 1, 1, 1, 0, 0, 0, 0))

    assert [shard for _pass, shard in due[:4]] == [4, 5, 6, 7]
    assert [shard for _pass, shard in due[0::2]][:2] == [4, 6]
    assert [shard for _pass, shard in due[0::4]][:1] == [4]


@pytest.mark.parametrize("matching", [True, False])
def test_saved_shard_passes_restore_only_for_the_same_shards(tmp_path, matching) -> None:
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    for name in ("a.tar", "b.tar"):
        (data_dir / name).write_bytes(b"")
    fingerprint = tar_fingerprint(data_dir, tuple(sorted(data_dir.rglob("*.tar")))) if matching else "deadbeef"
    saved = {
        "version": 1,
        "sources": {str(data_dir): {"fingerprint": fingerprint, "n_shards": 2, "shard_pass": [3, 1]}},
        "mix_rng_by_process": {},
    }

    tracker = make_data_cursor_tracker({"sources": [{"dir": str(data_dir)}], "batch_size": 1}, saved=saved)

    assert tracker.shard_pass.tolist() == ([3, 1] if matching else [0, 0])


def test_trained_batches_advance_the_shard_passes() -> None:
    tracker = DataCursorTracker(key="many", fingerprint="x", shard_pass=np.zeros(4, dtype=np.int64), tar_paths=())

    tracker.note_trained_batch(
        {
            BATCH_DONE_PASS_KEY: np.asarray([0, 0], dtype=np.int32),
            BATCH_DONE_SHARD_KEY: np.asarray([1, 3], dtype=np.int32),
        }
    )

    assert tracker.shard_pass.tolist() == [0, 1, 0, 1]


def test_split_cursor_batch_strips_the_cursor_keys() -> None:
    batch = {
        "tokens": np.zeros((2, 4), dtype=np.int32),
        BATCH_DONE_PASS_KEY: np.zeros((0,), dtype=np.int32),
        BATCH_DONE_SHARD_KEY: np.zeros((0,), dtype=np.int32),
    }

    clean, meta = split_cursor_batch(batch)

    assert set(clean) == {"tokens"}
    assert set(meta) == {BATCH_DONE_PASS_KEY, BATCH_DONE_SHARD_KEY}


# ── Loader workers ───────────────────────────────────────────────────


@pytest.fixture
def stub_worker_pipeline(monkeypatch):
    """Replace tar decoding, packing and collation of a loader worker by identities."""

    def install(frames, n_shards: int) -> None:
        monkeypatch.setattr(loader, "list_source_tar_paths", lambda source: tuple(range(n_shards)))
        monkeypatch.setattr(loader, "make_tokeniser", lambda vocab, **kwargs: None)
        monkeypatch.setattr(
            loader, "iter_assigned_tar_shards", lambda *args, **kwargs: iter((0, i, i) for i in range(n_shards))
        )
        monkeypatch.setattr(loader, "_iter_prepared_frames_for_tar", frames)
        monkeypatch.setattr(loader, "pack_rows", lambda config, frames, source: frames)
        monkeypatch.setattr(loader, "collate_tokenised_frames", lambda config, vocab, rows: {"rows": list(rows)})

    return install


def _worker_batches():
    return loader._iter_tagged_collated_batches(
        {},
        worker_id=0,
        num_workers=1,
        global_worker_offset=None,
        global_num_workers=None,
        seed=42,
        batch_size=4,
        config=SimpleNamespace(prompt_format="legacy"),
        vocab={},
        augmenter={},
        shard_pass=None,
    )


@pytest.mark.parametrize("lengths", [(8, 4), (3, 1, 4), (0, 4), (2, 0, 2)])
def test_worker_streams_and_marks_only_consumed_shards(stub_worker_pipeline, lengths) -> None:
    produced = []

    def frames(source, path, *args, **kwargs):
        for index in range(lengths[path]):
            produced.append((path, index))
            yield (path, index)

    stub_worker_pipeline(frames, len(lengths))
    consumed, completed = [], set()
    for batch in _worker_batches():
        consumed.extend(batch["rows"])
        assert len(produced) - len(consumed) <= 1
        for shard in batch[BATCH_DONE_SHARD_KEY]:
            assert all((int(shard), j) in consumed for j in range(lengths[int(shard)]))
            completed.add(int(shard))

    assert consumed == [(shard, i) for shard, length in enumerate(lengths) for i in range(length)]
    assert completed == set(range(len(lengths)))


def test_worker_yields_before_a_later_shard_error(stub_worker_pipeline) -> None:
    def frames(*args, **kwargs):
        yield from range(5)
        raise RuntimeError("late error")

    stub_worker_pipeline(frames, 1)
    batches = _worker_batches()
    batch = next(batches)

    assert batch["rows"] == [0, 1, 2, 3]
    assert batch[BATCH_DONE_SHARD_KEY].size == 0
    with pytest.raises(RuntimeError, match="late error"):
        next(batches)


# ── Molecular preparation and the finite molecular loader ────────────


def test_environment_description_states_the_water_count() -> None:
    assert environment_description("ff14sb-obc1-2nm") == "water, ff14sb-obc1-2nm"
    assert environment_description("gaff2-am1bcc-tip3p-rf2nm", 3) == "3 water, gaff2-am1bcc-tip3p-rf2nm"
    with pytest.raises(ValueError):
        environment_description("ff14sb-tip3p-rf2nm", 0)


def test_representation_choice_is_seeded_and_keeps_the_graph() -> None:
    frame = replace(
        DataFrame.from_smiles(sequence_to_smiles("AGA")),
        helm=sequence_to_helm("AGA"),
        metadata={"pilot_key": "example"},
    )

    a, _ = select_representation(frame, 42)
    b, _ = select_representation(frame, 42)

    assert a.helm == b.helm
    assert a.smiles == frame.smiles
    assert 0 <= a.metadata["helm_rewrite_selected"] <= 2


def test_fit_frame_drops_waters_that_exceed_the_budget_and_states_the_rest() -> None:
    frame = replace(
        DataFrame.from_smiles("[He:1].[O:2]([H:3])[H:4].[O:5]([H:6])[H:7]"),
        helm="CHEM1{[[He]]}$$$$",
        temperature=300.0,
        conformer=np.array(
            [[2, 0, 0, 0], [8, 0.3, 0, 0], [1, 0.3, 0.1, 0], [1, 0.3, 0, 0.1],
             [8, 0.6, 0, 0], [1, 0.6, 0.1, 0], [1, 0.6, 0, 0.1]],
            dtype=np.float32,
        ),
        metadata={"n_solute_atoms": 1, "n_waters": 2, "target_id": "gaff2-am1bcc-tip3p-rf2nm"},
    )
    tokeniser = make_tokeniser(load_tokeniser_default(vocab_name="core_bin4096"))
    one_water = environment_description("gaff2-am1bcc-tip3p-rf2nm", 1)
    budget = 2 + len(tokeniser.encode_prompt(frame.helm, environment=one_water, temperature_K=300.0)) + 12

    fitted = fit_frame(frame, PBGConfig(vocab_name="core_bin4096"), tokeniser, budget)

    assert fitted.metadata["presented_waters"] == 1
    assert fitted.environment == one_water
    np.testing.assert_array_equal(fitted.conformer, frame.conformer[:4])


def _molecular_frame(pilot_key: str) -> DataFrame:
    smiles = sequence_to_smiles("AG")
    z = atom_numbers_from_smiles(smiles)
    return replace(
        DataFrame.from_smiles(smiles),
        helm=sequence_to_helm("AG"),
        temperature=310.0,
        conformer=np.column_stack([z, np.zeros((len(z), 3))]).astype(np.float32),
        metadata={"n_solute_atoms": len(z), "n_waters": 0, "target_id": "ff14sb-obc1-2nm", "pilot_key": pilot_key},
    )


def _molecular_model(**overrides) -> SimpleNamespace:
    config = PBGConfig(vocab_name="core_bin4096", max_seq_len=256, max_samples_factor=0.03, **overrides)
    return SimpleNamespace(config=config, vocab=load_tokeniser_default(vocab_name="core_bin4096"))


@pytest.mark.parametrize("contract", ["many_molecules_pilot_v1", "many_molecules_joint_v1"])
def test_finite_molecular_loader_resumes_with_the_same_batches(tmp_path, monkeypatch, contract) -> None:
    manifest = {"release_contract": contract, "total_samples": 23, "shards": [{"path": "one.tar"}]}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        loader, "iter_frames_from_tar", lambda *args, **kwargs: (_molecular_frame(str(i)) for i in range(23))
    )
    model, source = _molecular_model(), {"dir": str(tmp_path)}

    batches = list(loader.molecular_batches(model, source, batch_size=4, seed=42))
    resumed = list(loader.molecular_batches(model, source, batch_size=4, seed=42, start_step=2))

    assert len(batches) > 2
    assert len(resumed) == len(batches) - 2
    for expected, actual in zip(batches[2:], resumed, strict=True):
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])


def test_molecular_cursor_survives_prefetch_and_dataset_appends(tmp_path, monkeypatch) -> None:
    def frames(path, **kwargs):
        offset = 0 if path.name == "one.tar" else 23
        return (_molecular_frame(str(offset + i)) for i in range(23))

    monkeypatch.setattr(loader, "iter_frames_from_tar", frames)
    first, second = dict(path="one.tar", sha256="first"), dict(path="two.tar", sha256="second")
    manifest = dict(release_contract="many_molecules_joint_v1", total_samples=23, shards=[first])
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    source = dict(name="joint", dir=str(tmp_path), molecular_single_pass=True, molecular_append_only=True)
    data = dict(sources=[source])
    model = _molecular_model()
    tracker = make_data_cursor_tracker(data)

    stream = loader.molecular_batches(model, source, batch_size=4, seed=42, cursor_tracker=tracker)
    _, metadata = split_cursor_batch(next(stream))
    tracker.note_trained_batch(metadata)
    next(stream)  # Producer prefetch must not advance the trained cursor.
    saved = tracker.to_serializable()
    stream.close()

    assert saved["molecular"]["step"] == 1
    assert "atom_order" not in saved["molecular"]["contract"]["model_config"]
    assert "atom_order_implementation" not in saved["molecular"]["contract"]["model_config"]

    manifest.update(total_samples=46, shards=[first, second])
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    restored = make_data_cursor_tracker(data, saved=saved)
    with pytest.raises(ValueError, match="model preparation"):
        next(loader.molecular_batches(
            _molecular_model(atom_order="prompt_canonical_v1"), source, batch_size=4, seed=42,
            start_step=1, cursor_tracker=restored,
        ))
    resumed = list(loader.molecular_batches(
        model, source, batch_size=4, seed=42, start_step=1, cursor_tracker=restored
    ))
    full = list(loader.molecular_batches(model, source, batch_size=4, seed=42))
    assert len(resumed) == len(full) - 1
    for actual, expected in zip(resumed, full[1:], strict=True):
        actual, _ = split_cursor_batch(actual)
        expected, _ = split_cursor_batch(expected)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])

    manifest["shards"][0]["sha256"] = "changed"
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact old shard prefix"):
        next(loader.molecular_batches(
            model, source, batch_size=4, seed=42, start_step=1, cursor_tracker=restored
        ))


# ── Augmentation ─────────────────────────────────────────────────────


def _hydrogen_frame(positions: np.ndarray | None) -> DataFrame:
    conformer = None
    if positions is not None:
        positions = np.asarray(positions, dtype=np.float32)
        conformer = np.concatenate([np.ones((positions.shape[0], 1), dtype=np.float32), positions], axis=1)
    return DataFrame(conformer=conformer, smiles="[H][O]", energy=None, frame_id=7)


def test_augmenter_passes_frames_without_positions() -> None:
    frame = _hydrogen_frame(None)

    out = apply_augmenter(init_augmenter(), frame)

    assert isinstance(out, DataFrame)
    assert out.conformer is None
    assert out.smiles == frame.smiles


def test_augmenter_centres_positions_and_keeps_atomic_numbers() -> None:
    frame = _hydrogen_frame([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    out = apply_augmenter(init_augmenter(center=True, rotate=False), frame)

    np.testing.assert_allclose(out.conformer[:, 1:4], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(out.conformer[:, 0], [1.0, 1.0])


def test_mass_centring_puts_the_centre_of_mass_at_the_origin(peptide_frame) -> None:
    frame = peptide_frame("WK", seed=0)
    frame = replace(frame, conformer=frame.conformer + np.asarray([0.0, 1.0, -2.0, 0.5], dtype=np.float32))
    table = Chem.GetPeriodicTable()
    masses = [table.GetAtomicWeight(int(z)) for z in frame.conformer[:, 0]]

    by_mass = apply_augmenter(init_augmenter(center="mass"), frame).conformer[:, 1:]
    by_atom = apply_augmenter(init_augmenter(center=True), frame).conformer[:, 1:]

    np.testing.assert_allclose(np.average(by_mass, axis=0, weights=masses), 0.0, atol=1e-5)
    np.testing.assert_allclose(by_atom.mean(axis=0), 0.0, atol=1e-5)
    assert np.abs(by_mass.mean(axis=0)).max() > 1e-3
    np.testing.assert_allclose(by_mass - by_mass[0], by_atom - by_atom[0], atol=1e-5)


def test_config_centring_reaches_the_augmenter(tiny_config) -> None:
    assert augmenter_from_config(tiny_config())["center"] is True
    assert augmenter_from_config(tiny_config(center_of_mass=True))["center"] == "mass"
    assert augmenter_from_config(tiny_config(center=False, center_of_mass=True))["center"] is False


def test_rotation_preserves_distances() -> None:
    frame = _hydrogen_frame([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    augmenter = init_augmenter(center=True, rotate=True, min_val=-4.0, max_val=4.0)

    out = apply_augmenter(augmenter, frame, rng=np.random.default_rng(0))

    np.testing.assert_allclose(np.linalg.norm(out.conformer[1, 1:4] - out.conformer[0, 1:4]), 1.0, atol=1e-6)


def test_rotation_retries_until_the_frame_fits_the_box() -> None:
    augmenter = init_augmenter(center=True, rotate=True, translate=True, min_val=-1.0, max_val=1.0)

    out = apply_augmenter(augmenter, _hydrogen_frame([[-1.25, 0.0, 0.0], [1.25, 0.0, 0.0]]), rng=np.random.default_rng(0))

    assert np.all(np.abs(out.conformer[:, 1:4]) <= 1.0)
    with pytest.raises(AugmentationBoundsError, match="after 16 attempts"):
        apply_augmenter(augmenter, _hydrogen_frame([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), rng=np.random.default_rng(0))


def test_translation_moves_the_frame_rigidly_within_the_box() -> None:
    frame = _hydrogen_frame([[-1.0, 0.5, 0.0], [1.0, 1.5, 0.0]])
    augmenter = init_augmenter(center=False, rotate=False, translate=True, min_val=-4.0, max_val=4.0)

    out = apply_augmenter(augmenter, frame, rng=np.random.default_rng(0))

    shifted = out.conformer[:, 1:4] - frame.conformer[:, 1:4]
    np.testing.assert_allclose(shifted[0], shifted[1])
    assert np.all(np.abs(out.conformer[:, 1:4]) <= 4.0)
    with pytest.raises(ValueError, match="min_val and max_val"):
        apply_augmenter(init_augmenter(center=False, rotate=False, translate=True), frame, rng=np.random.default_rng(0))


# ── Ensemble files ───────────────────────────────────────────────────


def test_ensemble_npz_round_trips_the_training_fields(tmp_path) -> None:
    path = tmp_path / "ensemble.npz"
    ensemble = Ensemble(
        samples=np.array([[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]]),
        smiles="[H:1][H:2]",
        id="HH",
        logq=np.array([1.25]),
        energy=np.array([2.5]),
        evaluation_weight=np.array([1.0]),
        self_distillation_weight=np.array([0.75]),
        behavior_token_log_probs=np.array([[-1.0, 0.0]], dtype=np.float32),
        behavior_token_ids=np.array([[7, 8]], dtype=np.int32),
        frame_valid=np.array([True]),
        stereo_valid=np.array([True]),
        sequence="HH",
        helm="CHEM1{[H][H]}$$$$",
        environment="vacuum",
        source="generator",
        generator_checkpoint="step_10",
        alpha=0.5,
        epsilon=0.1,
        temperature=300.0,
    )

    ensemble.to_npz(path)
    loaded = Ensemble.from_npz(path)

    np.testing.assert_allclose(loaded.logq, [1.25])
    np.testing.assert_allclose(loaded.self_distillation_weight, [0.75])
    np.testing.assert_allclose(loaded.behavior_token_log_probs, [[-1.0, 0.0]])
    np.testing.assert_array_equal(loaded.behavior_token_ids, [[7, 8]])
    np.testing.assert_array_equal(loaded.frame_valid, [True])
    np.testing.assert_array_equal(loaded.stereo_valid, [True])
    frame = loaded.frame(0)
    assert frame.sequence == "HH"
    assert frame.helm == "CHEM1{[H][H]}$$$$"
    assert frame.environment == "vacuum"
    assert frame.generator_checkpoint == "step_10"
    assert frame.alpha == 0.5
    assert frame.epsilon == 0.1
    assert frame.stereo_valid is True
    np.testing.assert_allclose(frame.behavior_token_log_probs, [-1.0, 0.0])


def test_ensemble_metadata_encoding_skips_none_values() -> None:
    encoded = encode_ensemble_metadata(
        {"fit_alpha": False, "inclusion_nats": None, "target_ress": None, "minimum_stereo_count": 1}
    )

    assert json.loads(str(np.asarray(encoded).item())) == {"fit_alpha": False, "minimum_stereo_count": 1}
