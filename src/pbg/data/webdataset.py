"""Read WebDataset tar shards as raw :class:`DataFrame` frames.

Two adapters decode the ``.bin`` member of a sample:

- ``many_peptides``: ``{sequence}_{frame}.bin`` holds float32 ``(x, y, z)`` in nm
  per atom, in the atom order of the mapped SMILES of the sequence.
- ``n4_smiles``: ``{key}.bin`` holds float32 ``(Z, x, y, z)`` per atom. The graph
  and the scalar fields come from ``{key}.json``; ``.smiles`` and
  ``.environment`` sidecars are a fallback. With ``sample_layout="n4_json_pair_v1"``
  every sample is a strict ``.json`` then ``.bin`` pair, read as a stream.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from pbg.core.frame import DataFrame
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import (
    infer_linear_peptide_sequence_from_smiles,
    sequence_to_helm,
    sequence_to_smiles,
)

# Scalar fields that an n4 ``.json`` record writes onto the frame.
_N4_FIELDS = {
    "helm": str,
    "energy": float,
    "logq": float,
    "self_distillation_weight": float,
    "frame_valid": bool,
    "generator_checkpoint": str,
    "alpha": float,
}
_N4_RESERVED = {*_N4_FIELDS, "smiles", "environment", "temperature", "stereo_valid"}


def _n4_frame(
    key: str,
    members: dict[str, bytes],
    environment: str | None,
    temperature: float | None,
    topology: dict[str, tuple[str | None, str | None]],
    *,
    strict: bool = False,
) -> DataFrame:
    """Decode one n4 sample from its members ``{suffix: bytes}``.

    ``strict`` is the ``n4_json_pair_v1`` contract: the ``.json`` record must
    state ``smiles``, ``environment`` and ``temperature``, and its ``frame_idx``
    sets the frame id. ``topology`` caches the ``(sequence, helm)`` of each
    SMILES, which is the expensive step and a pure function of the SMILES.
    """
    if ".bin" not in members:
        raise ValueError(f"{key} is missing .bin")
    record: dict[str, object] = {}
    if ".json" in members:
        try:
            record = json.loads(members[".json"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{key}.json is not valid UTF-8 JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{key}.json must be an object")
    if strict:
        missing = [field for field in ("smiles", "environment", "temperature") if field not in record]
        if missing:
            raise ValueError(f"{key}.json is missing required fields: {', '.join(missing)}")
        for field in ("smiles", "environment"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"{key}.json {field} must be a non-empty string")
        try:
            strict_temperature = float(record["temperature"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}.json temperature must be numeric") from exc
        if not np.isfinite(strict_temperature):
            raise ValueError(f"{key}.json temperature must be finite")
        frame_idx = record.get("frame_idx")
        if "frame_idx" in record and (isinstance(frame_idx, bool) or not isinstance(frame_idx, int)):
            raise ValueError(f"{key}.json frame_idx must be an integer")

    def text(field: str) -> str | None:
        """Return a field from the record, else from its sidecar, stripped; empty is None."""
        value = record.get(field)
        value = None if value is None else str(value).strip()
        if not value and f".{field}" in members:
            value = members[f".{field}"].decode("utf-8").strip()
        return value or None

    smiles = text("smiles")
    if smiles is None:
        raise ValueError(f"{key} is missing smiles")
    if environment is None:
        environment = text("environment")
    if temperature is None and record.get("temperature") is not None:
        temperature = float(record["temperature"])

    raw = members[".bin"]
    if len(raw) % 16 or not raw:
        raise ValueError(f"n4_smiles .bin length {len(raw)} is not a positive multiple of 16 (float32 Zxyz)")
    zxyz = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
    z = np.rint(zxyz[:, 0]).astype(np.int32)
    expected_z = atom_numbers_from_smiles(smiles).astype(np.int32, copy=False)
    if not np.array_equal(z, expected_z):
        raise ValueError(f"z(data)={z} != z(SMILES)={expected_z} for {key}: {smiles!r}")

    if smiles not in topology:
        try:
            sequence = infer_linear_peptide_sequence_from_smiles(smiles)
            topology[smiles] = (sequence, sequence_to_helm(sequence))
        except ValueError:
            topology[smiles] = (None, None)
    fields: dict[str, object] = dict(zip(("sequence", "helm"), topology[smiles]))
    # Record fields override the topology, so a stored HELM wins.
    fields.update({name: cast(record[name]) for name, cast in _N4_FIELDS.items() if record.get(name) is not None})
    if "stereo_valid" in record:
        fields["stereo_valid"] = None if record["stereo_valid"] is None else bool(record["stereo_valid"])

    # The key is {sample}_{frame}; a key without a numeric suffix has no frame id.
    _, _, suffix = key.strip().rpartition("_")
    frame_id = int(suffix) if suffix.isdigit() and "_" in key else None
    if strict and "frame_idx" in record:
        frame_id = record["frame_idx"]

    extra = {name: value for name, value in record.items() if name not in _N4_RESERVED}
    return DataFrame(
        conformer=np.concatenate([z.astype(np.float32)[:, None], zxyz[:, 1:4]], axis=1),
        smiles=smiles,
        environment=environment,
        temperature=temperature,
        frame_id=frame_id,
        metadata=extra or None,
        **fields,
    )


def iter_frames_from_tar(
    paths: Path | str | list[Path] | list[str],
    adapter: str = "many_peptides",
    on_decode_error: str = "raise",
    environment: str | None = None,
    temperature: float | None = None,
    sample_layout: str | None = None,
    frame_id_below: int | None = None,
    sequence_allowlist: str | Path | None = None,
) -> Iterator[DataFrame]:
    """Yield the frames of tar shards in member order.

    ``on_decode_error="skip"`` drops samples that fail to decode; the strict
    ``n4_json_pair_v1`` layout always raises. ``environment`` and
    ``temperature`` override the stored values (``many_peptides`` defaults:
    water, 310 K).

    Two ``many_peptides`` filters read the member name only, so skipped frames
    are never decoded: ``frame_id_below`` keeps members ``{sequence}_{frame}``
    with ``frame < frame_id_below``, and ``sequence_allowlist`` is a text file
    with one sequence per line (a relative path resolves against the
    repository root).
    """
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if on_decode_error not in ("raise", "skip"):
        raise ValueError(f"on_decode_error must be 'raise' or 'skip', got {on_decode_error!r}")
    if adapter not in ("many_peptides", "n4_smiles"):
        raise ValueError(f"unknown adapter: {adapter!r}")
    if sample_layout not in (None, "n4_json_pair_v1"):
        raise ValueError(f"unknown sample layout: {sample_layout!r}")
    if sample_layout is not None and adapter != "n4_smiles":
        raise ValueError(f"sample layout {sample_layout!r} requires adapter 'n4_smiles', got {adapter!r}")
    if frame_id_below is not None:
        if isinstance(frame_id_below, bool) or not isinstance(frame_id_below, int) or frame_id_below < 1:
            raise ValueError(f"frame_id_below must be a positive integer, got {frame_id_below!r}")
        if adapter != "many_peptides":
            raise ValueError(f"frame_id_below requires adapter 'many_peptides', got {adapter!r}")
    allowed: frozenset[str] | None = None
    if sequence_allowlist is not None:
        if adapter != "many_peptides":
            raise ValueError(f"sequence_allowlist requires adapter 'many_peptides', got {adapter!r}")
        allowlist_path = Path(sequence_allowlist)
        if not allowlist_path.is_absolute():
            allowlist_path = Path(__file__).resolve().parents[3] / allowlist_path
        allowed = frozenset(line.strip() for line in allowlist_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not allowed:
            raise ValueError(f"sequence_allowlist {allowlist_path} is empty")

    for path in path_list:
        # One topology cache per shard: it is bounded by the distinct SMILES of the shard.
        topology: dict[str, tuple[str | None, str | None]] = {}
        if sample_layout == "n4_json_pair_v1":
            yield from _iter_n4_json_pairs(path, topology)
            continue
        with tarfile.open(path, "r") as tar:
            if adapter == "many_peptides":
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".bin"):
                        continue
                    sequence, separator, frame_token = Path(member.name).stem.partition("_")
                    if allowed is not None and sequence not in allowed:
                        continue
                    if frame_id_below is not None:
                        if not frame_token.isdigit():
                            raise ValueError(f"frame_id_below needs a frame id in member {member.name!r}")
                        if int(frame_token) >= frame_id_below:
                            continue
                    raw = tar.extractfile(member).read()
                    try:
                        if len(raw) % 12 or not raw:
                            raise ValueError(f"many_peptides .bin length {len(raw)} is not a positive multiple of 12 (float32 xyz)")
                        smiles = sequence_to_smiles(sequence)
                        atoms = atom_numbers_from_smiles(smiles).astype(np.float32)[:, None]
                        frame = DataFrame(
                            conformer=np.concatenate([atoms, np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)], axis=1),
                            smiles=smiles,
                            sequence=sequence,
                            helm=sequence_to_helm(sequence),
                            environment="water" if environment is None else environment,
                            temperature=310.0 if temperature is None else temperature,
                            frame_id=int(frame_token) if separator else None,
                        )
                    except Exception:
                        if on_decode_error == "raise":
                            raise
                        continue
                    yield frame
                continue
            # n4_smiles with sidecars: a sample is complete once it has its .bin,
            # a graph (.json or .smiles), and every .json or .environment member
            # the shard holds for its key. Datasets such as MDQM9-nc ship .bin
            # and .smiles only.
            suffixes = (".bin", ".smiles", ".environment", ".json")
            members = [m for m in tar.getmembers() if m.isfile() and Path(m.name).suffix.lower() in suffixes]
            present = {(Path(m.name).stem, Path(m.name).suffix.lower()) for m in members}
            pending: dict[str, dict[str, bytes]] = {}
            for member in members:
                stem, suffix = Path(member.name).stem, Path(member.name).suffix.lower()
                sample = pending.setdefault(stem, {})
                sample[suffix] = tar.extractfile(member).read()
                if (
                    ".bin" not in sample
                    or not {".json", ".smiles"} & sample.keys()
                    or any((stem, s) in present and s not in sample for s in (".json", ".environment"))
                ):
                    continue
                del pending[stem]
                try:
                    frame = _n4_frame(stem, sample, environment, temperature, topology)
                except Exception:
                    if on_decode_error == "raise":
                        raise
                    continue
                yield frame


def _iter_n4_json_pairs(path: Path | str, topology: dict[str, tuple[str | None, str | None]]) -> Iterator[DataFrame]:
    """Stream a strict ``n4_json_pair_v1`` shard without indexing the whole tar."""
    seen: set[str] = set()
    with tarfile.open(path, "r|") as tar:
        while (json_member := tar.next()) is not None:
            try:
                if not json_member.isfile() or not json_member.name.endswith(".json"):
                    raise ValueError(
                        f"n4_json_pair_v1 expects consecutive <key>.json then <key>.bin members, got {json_member.name!r}"
                    )
                key = json_member.name[:-5]
                if not key or "/" in key:
                    raise ValueError(f"invalid n4_json_pair_v1 key: {key!r}")
                if key in seen:
                    raise ValueError(f"duplicate n4_json_pair_v1 key: {key!r}")
                json_bytes = tar.extractfile(json_member).read()
                bin_member = tar.next()
                if bin_member is None or not bin_member.isfile() or bin_member.name != f"{key}.bin":
                    found = None if bin_member is None else bin_member.name
                    raise ValueError(f"n4_json_pair_v1 key {key!r} must be followed by {key}.bin, got {found!r}")
                bin_bytes = tar.extractfile(bin_member).read()
                seen.add(key)
            finally:
                # TarFile.next() keeps every header in ``members``, even in
                # stream mode. Clear that cache; the stream cursor stays valid.
                tar.members.clear()
            yield _n4_frame(key, {".json": json_bytes, ".bin": bin_bytes}, None, None, topology, strict=True)
