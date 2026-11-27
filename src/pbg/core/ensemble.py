"""Conformer ensembles and trajectories, and their NPZ files.

``Ensemble`` holds zxyz coordinates ``samples`` of shape (K, N, 4) with
per-frame and per-ensemble data. ``Trajectory`` adds time and fitted tICA
parameters. ``load_ensemble`` reads the canonical schema and the historical
NPZ variants; ``save_ensemble`` writes the canonical schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float

from pbg.core.molecule import (
    check_smiles_is_indexed,
    check_smiles_z_matches_frame_z,
    check_z_column_fixed_across_frames,
)

# zxyz layout: (K, N, 4) = frames x atoms x (Z + xyz in nm).
ConformerArray = Float[np.ndarray, "frames atoms 4"]
ConformerScalars = Float[np.ndarray, "frames"]
AtomFrame = Float[np.ndarray, "atoms 4"]
FrameForceArray = Float[np.ndarray, "frames atoms xyz"]
MetadataScalar = bool | int | float | str
FrameMetadataValue = MetadataScalar | list[Any] | dict[str, Any]
EnsembleMetadata = dict[str, MetadataScalar]
FrameMetadata = dict[str, FrameMetadataValue]
FrameMetadataList = tuple[FrameMetadata, ...]


@dataclass(frozen=True, slots=True)
class TorsionAngles:
    """Per-frame torsion angles stored alongside ensemble coordinates."""

    angles: np.ndarray
    indices: np.ndarray
    names: np.ndarray
    mode: str = "peptide_bond"


@dataclass(frozen=True, slots=True)
class RadialDistribution:
    """One-dimensional radial-distance distribution stored with an ensemble."""

    values: np.ndarray
    bin_edges: np.ndarray
    atom_numbers: np.ndarray
    mode: str = "solvent_solvent_heavy_atom"
    normalization: str = "probability_density"


def _normalise_metadata_scalar(value: object) -> MetadataScalar:
    if isinstance(value, (list, tuple, dict)):
        raise TypeError("metadata scalar values cannot be lists, tuples, or dicts")
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported metadata scalar value {value!r}")


def _normalise_frame_metadata_value(value: object) -> FrameMetadataValue:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_normalise_frame_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalise_frame_metadata_value(item)
            for key, item in value.items()
        }
    return _normalise_metadata_scalar(value)


def validate_frame_metadata(
    frame_metadata: FrameMetadataList | None,
    *,
    n_frames: int,
    context: object,
) -> None:
    if frame_metadata is None:
        return
    if len(frame_metadata) != n_frames:
        raise ValueError(
            f"{context}: frame_metadata has length {len(frame_metadata)}, expected {n_frames}"
        )
    for frame_idx, values in enumerate(frame_metadata):
        if not isinstance(values, dict):
            raise TypeError(
                f"{context}: frame_metadata[{frame_idx}] must be a dict, got {type(values).__name__}"
            )
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{context}: frame_metadata[{frame_idx}] key {key!r} is not a str"
                )
            try:
                _normalise_frame_metadata_value(value)
            except TypeError as exc:
                raise TypeError(
                    f"{context}: frame_metadata[{frame_idx}][{key!r}] must be JSON-compatible metadata"
                ) from exc


def encode_ensemble_metadata(
    ensemble_metadata: EnsembleMetadata | None,
) -> np.ndarray | None:
    if ensemble_metadata is None:
        return None
    normalised = {
        key: _normalise_metadata_scalar(value)
        for key, value in ensemble_metadata.items()
        if value is not None
    }
    return np.array(json.dumps(normalised, sort_keys=True), dtype=np.str_)


def decode_ensemble_metadata(data: Any) -> EnsembleMetadata | None:
    key = "ensemble_metadata_json" if "ensemble_metadata_json" in data.files else "metadata_json"
    if key not in data.files:
        return None
    decoded = json.loads(str(np.asarray(data[key]).item()))
    if not isinstance(decoded, dict):
        raise ValueError(f"{key} must decode to a dict")
    return {
        str(metadata_key): _normalise_metadata_scalar(value)
        for metadata_key, value in decoded.items()
    }


def encode_frame_metadata(frame_metadata: FrameMetadataList | None) -> np.ndarray | None:
    if frame_metadata is None:
        return None
    normalised = tuple(
        {
            key: _normalise_frame_metadata_value(value)
            for key, value in frame_values.items()
            if value is not None
        }
        for frame_values in frame_metadata
    )
    return np.array(json.dumps(normalised, sort_keys=True), dtype=np.str_)


def decode_frame_metadata(
    data: Any, *, n_frames: int, context: object
) -> FrameMetadataList | None:
    frame_metadata: FrameMetadataList | None = None
    if "frame_metadata_json" in data.files:
        decoded = json.loads(str(np.asarray(data["frame_metadata_json"]).item()))
        if not isinstance(decoded, list):
            raise ValueError(f"{context}: frame_metadata_json must decode to a list")
        frame_metadata = tuple(
            {
                str(key): _normalise_frame_metadata_value(value)
                for key, value in frame_values.items()
            }
            for frame_values in decoded
        )
    elif "decode_token_ids" in data.files:
        token_ids = np.asarray(data["decode_token_ids"], dtype=np.int32)
        vocab_name = str(np.asarray(data["decode_vocab_name"]).item())
        frame_metadata = tuple(
            {
                "decode_token_ids": [int(token) for token in row],
                "decode_vocab_name": vocab_name,
            }
            for row in token_ids
        )
    validate_frame_metadata(frame_metadata, n_frames=n_frames, context=context)
    return frame_metadata


def _validate_torsions(
    torsions: TorsionAngles | None,
    *,
    n_frames: int,
    context: object,
) -> None:
    if torsions is None:
        return
    angles = np.asarray(torsions.angles)
    indices = np.asarray(torsions.indices)
    names = np.asarray(torsions.names)
    if angles.ndim != 2:
        raise ValueError(f"{context}: torsion angles must have shape (frames, torsions)")
    if int(angles.shape[0]) != int(n_frames):
        raise ValueError(
            f"{context}: torsion angles have {angles.shape[0]} frames, expected {n_frames}"
        )
    if indices.ndim != 2 or int(indices.shape[1]) != 4:
        raise ValueError(f"{context}: torsion indices must have shape (torsions, 4)")
    if int(indices.shape[0]) != int(angles.shape[1]):
        raise ValueError(
            f"{context}: torsion indices count {indices.shape[0]} does not match "
            f"angle count {angles.shape[1]}"
        )
    if names.ndim != 1 or int(names.shape[0]) != int(angles.shape[1]):
        raise ValueError(
            f"{context}: torsion names count {names.shape[0]} does not match "
            f"angle count {angles.shape[1]}"
        )


def encode_torsions(
    torsions: TorsionAngles | None,
    *,
    n_frames: int,
    context: object,
) -> dict[str, np.ndarray]:
    _validate_torsions(torsions, n_frames=n_frames, context=context)
    if torsions is None:
        return {}
    return {
        "torsion_angles": np.asarray(torsions.angles, dtype=np.float64),
        "torsion_indices": np.asarray(torsions.indices, dtype=np.int64),
        "torsion_names": np.asarray(torsions.names, dtype=np.str_),
        "torsion_mode": np.array(str(torsions.mode), dtype=np.str_),
    }


def decode_torsions(
    data: Any,
    *,
    n_frames: int,
    context: object,
) -> TorsionAngles | None:
    if "torsion_angles" not in data.files:
        return None
    required = ("torsion_indices", "torsion_names", "torsion_mode")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{context}: torsion_angles present but missing {missing}")
    torsions = TorsionAngles(
        angles=np.asarray(data["torsion_angles"], dtype=np.float64),
        indices=np.asarray(data["torsion_indices"], dtype=np.int64),
        names=np.asarray(data["torsion_names"], dtype=np.str_),
        mode=str(np.asarray(data["torsion_mode"]).item()),
    )
    _validate_torsions(torsions, n_frames=n_frames, context=context)
    return torsions


def _validate_radial_distribution(
    radial_distribution: RadialDistribution | None,
    *,
    context: object,
) -> None:
    if radial_distribution is None:
        return
    values = np.asarray(radial_distribution.values)
    bin_edges = np.asarray(radial_distribution.bin_edges)
    atom_numbers = np.asarray(radial_distribution.atom_numbers)
    if values.ndim != 1:
        raise ValueError(f"{context}: radial_distribution values must be a vector")
    if bin_edges.ndim != 1:
        raise ValueError(f"{context}: radial_distribution bin_edges must be a vector")
    if int(bin_edges.shape[0]) != int(values.shape[0]) + 1:
        raise ValueError(
            f"{context}: radial_distribution bin_edges length {bin_edges.shape[0]} "
            f"must be values length + 1 ({values.shape[0] + 1})"
        )
    if atom_numbers.ndim != 1 or int(atom_numbers.shape[0]) != 2:
        raise ValueError(
            f"{context}: radial_distribution atom_numbers must have shape (2,)"
        )
    if np.any(np.diff(bin_edges) <= 0.0):
        raise ValueError(f"{context}: radial_distribution bin_edges must increase")


def encode_radial_distribution(
    radial_distribution: RadialDistribution | None,
    *,
    context: object,
) -> dict[str, np.ndarray]:
    _validate_radial_distribution(radial_distribution, context=context)
    if radial_distribution is None:
        return {}
    return {
        "radial_distribution_values": np.asarray(
            radial_distribution.values, dtype=np.float64
        ),
        "radial_distribution_bin_edges": np.asarray(
            radial_distribution.bin_edges, dtype=np.float64
        ),
        "radial_distribution_atom_numbers": np.asarray(
            radial_distribution.atom_numbers, dtype=np.int64
        ),
        "radial_distribution_mode": np.array(
            str(radial_distribution.mode), dtype=np.str_
        ),
        "radial_distribution_normalization": np.array(
            str(radial_distribution.normalization), dtype=np.str_
        ),
    }


def decode_radial_distribution(data: Any, *, context: object) -> RadialDistribution | None:
    if "radial_distribution_values" not in data.files:
        return None
    required = (
        "radial_distribution_bin_edges",
        "radial_distribution_atom_numbers",
        "radial_distribution_mode",
        "radial_distribution_normalization",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(
            f"{context}: radial_distribution_values present but missing {missing}"
        )
    radial_distribution = RadialDistribution(
        values=np.asarray(data["radial_distribution_values"], dtype=np.float64),
        bin_edges=np.asarray(data["radial_distribution_bin_edges"], dtype=np.float64),
        atom_numbers=np.asarray(data["radial_distribution_atom_numbers"], dtype=np.int64),
        mode=str(np.asarray(data["radial_distribution_mode"]).item()),
        normalization=str(np.asarray(data["radial_distribution_normalization"]).item()),
    )
    _validate_radial_distribution(radial_distribution, context=context)
    return radial_distribution


@dataclass(frozen=True, slots=True, kw_only=True)
class Ensemble:
    """An unordered collection of conformers: zxyz ``samples`` of shape (K, N, 4)."""

    samples: ConformerArray
    smiles: str
    id: int | str
    sequence: str | None = None
    helm: str | None = None
    environment: str | None = None
    temperature: float | None = None
    energy: ConformerScalars | None = None
    logq: ConformerScalars | None = None
    self_distillation_weight: ConformerScalars | None = None
    behavior_token_log_probs: np.ndarray | None = None
    behavior_token_ids: np.ndarray | None = None
    frame_valid: np.ndarray | None = None
    stereo_valid: np.ndarray | None = None
    evaluation_weight: ConformerScalars | None = None
    source: str | None = None
    generator_checkpoint: str | None = None
    alpha: float | None = None
    epsilon: float | None = None
    ensemble_metadata: EnsembleMetadata | None = None
    frame_metadata: FrameMetadataList | None = None
    torsions: TorsionAngles | None = None
    radial_distribution: RadialDistribution | None = None

    @property
    def length(self) -> int:
        return int(self.samples.shape[0])

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f"samples must have shape (K, N, 4), got {samples.shape}")
        for name in (
            "energy",
            "logq",
            "self_distillation_weight",
            "frame_valid",
            "stereo_valid",
            "evaluation_weight",
        ):
            values = getattr(self, name)
            if values is not None and np.asarray(values).shape != (self.length,):
                raise ValueError(f"{name} must have shape {(self.length,)}, got {np.asarray(values).shape}")
        if self.self_distillation_weight is not None:
            weights = np.asarray(self.self_distillation_weight, dtype=np.float64)
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("self_distillation_weight must be finite and non-negative")
        for name in ("behavior_token_log_probs", "behavior_token_ids"):
            value = getattr(self, name)
            if value is not None and (
                np.asarray(value).ndim != 2 or np.asarray(value).shape[0] != self.length
            ):
                raise ValueError(f"{name} must have shape (frames, trained_tokens)")
        if self.behavior_token_log_probs is not None:
            log_probs = np.asarray(self.behavior_token_log_probs)
            if not np.all(np.isfinite(log_probs)):
                raise ValueError("behavior_token_log_probs must be finite")
            if self.behavior_token_ids is None or np.asarray(
                self.behavior_token_ids
            ).shape != log_probs.shape:
                raise ValueError("behavior token ids and log probabilities must have equal shapes")
        if self.temperature is not None and (not np.isfinite(self.temperature) or self.temperature <= 0.0):
            raise ValueError("temperature must be finite and positive")

    def to_npz(self, path: str | Path) -> None:
        save_ensemble(self, path)

    @classmethod
    def from_npz(cls, path: str | Path):
        loaded = load_ensemble(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"{path} contains {type(loaded).__name__}, expected {cls.__name__}")
        return loaded

    def frame(self, index: int):
        from pbg.core.frame import DataFrame

        def item(values, default=None):
            return default if values is None else np.asarray(values)[index].item()

        return DataFrame(
            conformer=np.asarray(self.samples[index], dtype=np.float32),
            id=self.id,
            frame_id=index,
            smiles=self.smiles,
            sequence=self.sequence,
            helm=self.helm,
            environment=self.environment,
            temperature=self.temperature,
            energy=item(self.energy),
            logq=item(self.logq),
            self_distillation_weight=float(item(self.self_distillation_weight, 1.0)),
            behavior_token_log_probs=None
            if self.behavior_token_log_probs is None
            else np.asarray(self.behavior_token_log_probs[index]),
            behavior_token_ids=None
            if self.behavior_token_ids is None
            else np.asarray(self.behavior_token_ids[index]),
            frame_valid=bool(item(self.frame_valid, True)),
            stereo_valid=None if self.stereo_valid is None else bool(item(self.stereo_valid)),
            source=self.source,
            generator_checkpoint=self.generator_checkpoint,
            alpha=self.alpha,
            epsilon=self.epsilon,
            time=item(self.time) if isinstance(self, Trajectory) else None,
        )

    def iter_frames(self):
        return (self.frame(index) for index in range(self.length))


Featurization = Literal["heavy_pairs", "cns_dihedrals"]
# Files without ``tica_featurization`` (the Many Peptides reference MD) use
# cns_dihedrals. ``pbg.estimator.features.tica_projection`` computes the
# features and rejects a tICA whose size does not match them.
DEFAULT_FEATURIZATION: Featurization = "cns_dihedrals"


@dataclass(frozen=True, slots=True)
class TicaParameters:
    mean: np.ndarray
    projection: np.ndarray
    dim: np.ndarray
    featurization: Featurization = DEFAULT_FEATURIZATION


@dataclass(frozen=True, slots=True, kw_only=True)
class Trajectory(Ensemble):
    """A time-ordered ensemble with optional fitted tICA parameters."""

    time: ConformerScalars | None = None
    tica: TicaParameters | None = None


def _scalar(data: Any, name: str, default=None):
    return np.asarray(data[name]).item() if name in data.files else default


def _vector(data: Any, names: tuple[str, ...], dtype, n_frames: int):
    for name in names:
        if name in data.files:
            values = np.asarray(data[name], dtype=dtype)
            if values.shape != (n_frames,):
                raise ValueError(f"{name} has shape {values.shape}, expected {(n_frames,)}")
            return values
    return None


def _matrix(data: Any, name: str, dtype, n_frames: int):
    if name not in data.files:
        return None
    values = np.asarray(data[name], dtype=dtype)
    if values.ndim != 2 or values.shape[0] != n_frames:
        raise ValueError(f"{name} must have shape (frames, trained_tokens), got {values.shape}")
    return values


def load_ensemble(path: str | Path) -> Ensemble:
    """Load the canonical schema and the historical ensemble NPZ variants."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        samples = np.asarray(data["positions"], dtype=np.float64)
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f"{path}: positions must have shape (K, N, 4), got {samples.shape}")
        n_frames = int(samples.shape[0])
        smiles = str(_scalar(data, "smiles", _scalar(data, "mapped_smiles", "")))
        raw_id = _scalar(data, "id", _scalar(data, "molecule_id", path.stem))
        ensemble_id = int(raw_id) if isinstance(raw_id, (int, np.integer)) else str(raw_id)
        metadata = decode_ensemble_metadata(data) or {}
        sequence = _scalar(data, "sequence", metadata.get("sequence"))
        helm = _scalar(data, "helm", metadata.get("helm"))
        environment = _scalar(data, "environment", metadata.get("environment"))
        temperature = _scalar(data, "temperature", metadata.get("temperature"))
        logq = _vector(data, ("generator_log_likelihood", "logq", "log_likelihood"), np.float64, n_frames)
        self_distillation_weight = _vector(
            data,
            ("self_distillation_weight", "loss_weight", "distillation_normalised_weight"),
            np.float64,
            n_frames,
        )
        evaluation_weight = _vector(data, ("evaluation_weight", "normalised_weight", "raw_weight"), np.float64, n_frames)
        valid = _vector(data, ("frame_valid", "qc", "correct_stereo"), np.bool_, n_frames)
        stereo_valid = _vector(data, ("stereo_valid",), np.bool_, n_frames)
        common = dict(
            samples=samples,
            smiles=smiles,
            id=ensemble_id,
            sequence=None if sequence is None else str(sequence),
            helm=None if helm is None else str(helm),
            environment=None if environment is None else str(environment),
            temperature=None if temperature is None else float(temperature),
            energy=_vector(data, ("energy", "target_energy"), np.float64, n_frames),
            logq=logq,
            self_distillation_weight=self_distillation_weight,
            behavior_token_log_probs=_matrix(data, "behavior_token_log_probs", np.float32, n_frames),
            behavior_token_ids=_matrix(data, "behavior_token_ids", np.int32, n_frames),
            frame_valid=valid,
            stereo_valid=stereo_valid,
            evaluation_weight=evaluation_weight,
            source=_scalar(data, "source", metadata.get("source")),
            generator_checkpoint=_scalar(data, "generator_checkpoint", metadata.get("checkpoint_version")),
            alpha=_scalar(data, "alpha", metadata.get("alpha")),
            epsilon=_scalar(data, "epsilon", metadata.get("epsilon_relative_to_max")),
            ensemble_metadata=metadata or None,
            frame_metadata=decode_frame_metadata(data, n_frames=n_frames, context=path),
            torsions=decode_torsions(data, n_frames=n_frames, context=path),
            radial_distribution=decode_radial_distribution(data, context=path),
        )
        check_z_column_fixed_across_frames(samples, context=path)
        check_smiles_is_indexed(smiles, context=path)
        check_smiles_z_matches_frame_z(smiles, samples, context=path)
        kind = str(_scalar(data, "ensemble_kind", "trajectory" if "tica_mean" in data.files or "time" in data.files else "ensemble"))
        if kind == "trajectory":
            tica = None
            if "tica_mean" in data.files:
                tica = TicaParameters(
                    mean=np.asarray(data["tica_mean"], dtype=np.float64),
                    projection=np.asarray(data["tica_projection"], dtype=np.float64),
                    dim=np.asarray(data["tica_dim"]),
                    featurization=str(_scalar(data, "tica_featurization", DEFAULT_FEATURIZATION)),
                )
            return Trajectory(**common, time=_vector(data, ("time",), np.float64, n_frames), tica=tica)
        if kind != "ensemble":
            raise ValueError(f"{path}: unknown ensemble_kind {kind!r}")
        return Ensemble(**common)


def save_ensemble(ensemble: Ensemble, path: str | Path) -> None:
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "ensemble_schema_version": np.array(
            3 if ensemble.stereo_valid is not None else 2, dtype=np.int64
        ),
        "ensemble_kind": np.array("trajectory" if isinstance(ensemble, Trajectory) else "ensemble", dtype=np.str_),
        "positions": np.asarray(ensemble.samples, dtype=np.float64),
        "smiles": np.array(ensemble.smiles, dtype=np.str_),
        "id": np.asarray(ensemble.id),
    }
    scalars = {
        "sequence": ensemble.sequence,
        "helm": ensemble.helm,
        "environment": ensemble.environment,
        "temperature": ensemble.temperature,
        "source": ensemble.source,
        "generator_checkpoint": ensemble.generator_checkpoint,
        "alpha": ensemble.alpha,
        "epsilon": ensemble.epsilon,
    }
    for name, value in scalars.items():
        if value is not None:
            payload[name] = np.asarray(value)
    vectors = {
        "energy": ensemble.energy,
        "generator_log_likelihood": ensemble.logq,
        "self_distillation_weight": ensemble.self_distillation_weight,
        "frame_valid": ensemble.frame_valid,
        "stereo_valid": ensemble.stereo_valid,
        "evaluation_weight": ensemble.evaluation_weight,
    }
    for name, value in vectors.items():
        if value is not None:
            payload[name] = np.asarray(
                value, dtype=np.bool_ if name in {"frame_valid", "stereo_valid"} else np.float64
            )
    if ensemble.behavior_token_log_probs is not None:
        payload["behavior_token_log_probs"] = np.asarray(
            ensemble.behavior_token_log_probs, dtype=np.float32
        )
        payload["behavior_token_ids"] = np.asarray(ensemble.behavior_token_ids, dtype=np.int32)
    encoded = encode_ensemble_metadata(ensemble.ensemble_metadata)
    if encoded is not None:
        payload["ensemble_metadata_json"] = encoded
    encoded_frames = encode_frame_metadata(ensemble.frame_metadata)
    if encoded_frames is not None:
        payload["frame_metadata_json"] = encoded_frames
    payload.update(encode_torsions(ensemble.torsions, n_frames=ensemble.length, context=path))
    payload.update(encode_radial_distribution(ensemble.radial_distribution, context=path))
    if isinstance(ensemble, Trajectory):
        if ensemble.time is not None:
            payload["time"] = np.asarray(ensemble.time, dtype=np.float64)
        if ensemble.tica is not None:
            payload.update(
                tica_mean=np.asarray(ensemble.tica.mean, dtype=np.float64),
                tica_projection=np.asarray(ensemble.tica.projection, dtype=np.float64),
                tica_dim=np.asarray(ensemble.tica.dim),
                tica_featurization=np.array(ensemble.tica.featurization, dtype=np.str_),
            )
    np.savez_compressed(path, **payload)
