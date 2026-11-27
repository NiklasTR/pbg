"""Train a PBG (PBG) model from a JSON configuration."""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import queue
import signal
import threading
import time
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Generic, NamedTuple, TypeVar

# Slurm may narrow CUDA_VISIBLE_DEVICES for each task after sbatch setup. Allow
# the job config to pass a stable override before JAX initializes its backend.
if override := os.environ.get("PBG_CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = override

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from pbg.checkpoint import load_pickle, retire
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.distributed import (
    bootstrap_distributed,
    build_mesh,
    constrain_tree_replicated,
    put_batch,
    put_train_state,
)
from pbg.models.pbg.loader import (
    iter_training_batches,
    make_data_cursor_tracker,
    parse_saved_cursor,
    split_cursor_batch,
)
from pbg.models.pbg.model import PBG
from pbg.models.pbg.optimizer import (
    AdaGCState,
    OptimizerConfig,
    create_optimizer,
    current_learning_rates,
)
from pbg.models.pbg.robin import init_atom_conditioner
from pbg.models.pbg.transformer import qwen_vocab_size

log = logging.getLogger(__name__)
CONFIG_ROOT = Path(__file__).resolve().parents[4] / "configs"


SOURCE_KEYS = {
    "adapter",
    "derive_chem_helm_from_smiles",
    "prompt_notation",
    "dir",
    "environment_prompt",
    "frame_id_below",
    "on_decode_error",
    "packing",
    "sequence_allowlist",
    "temperature",
}


PARAMETERS: dict[str, Any] = {
    "run_name": "pbg-train",
    "run_notes": None,
    "seed": 42,
    # Root of the relative data.sources[].dir paths; PBG_DATA_BASE overrides it.
    "data_base": os.environ.get("PBG_DATA_BASE", "data"),
    "checkpoint_base": os.environ.get("PBG_CHECKPOINT_BASE", "."),
    "model": {},
    "data": {
        "batch_size": 16,
        "prefetch": 16,
        "data_loader_workers": 4,
        "batch_mode": "mixed",
        "mix": {"mode": "row_weighted_fixed"},
        "sources": [
            {
                "name": "many_peptides",
                "dir": "many-peptides-md/webdatasets/single_frames",
                "source_format": "webdataset_tar",
                "adapter": "many_peptides",
                "environment_prompt": {"water": "Amber-ff14SB in OBC1 implicit water"},
                "temperature": 310.0,
                "packing": "greedy",
            },
        ],
    },
    # The recipe of the paper's core runs; every run config sets these keys.
    "optimizer": {
        "lr": 0.001,
        "schedule": "wsd",
        "total_steps": 250_000,
        "warmup_steps": 10000,
        "cooldown_steps": 50_000,
        "final_lr_frac": 0.0,
        "div_factor": 500.0,
        "grouping": "matrix_except_embed",
        "muon_momentum": 0.95,
        "ns_iters": 5,
        "ns_dtype": "bfloat16",
        "fan_in_scale": False,
        "muon_scaling_patches": True,
        "weight_decay": 0.1,
        "adamw_lr_ratio": 1.0,
        "adam_b1": 0.9,
        "adam_b2": 0.95,
        "adam_eps": 1e-8,
        "grad_clip": 10.0,
        "grad_clip_rel": 1.5,
        "grad_clip_rel_beta": 0.99,
        "grad_clip_rel_warmup": 100,
    },
    "training": {
        "num_steps": 250_000,
        "log_every": 20,
        "final_log_steps": 0,
        "final_log_every": 1,
        "save_every": 1000,
        "wandb": {
            "enabled": False,
            "project": "pbg",
            "run_id": None,  # set to resume a specific W&B run; None = one run per Slurm job
        },
    },
    "checkpoint": {
        "dir": "checkpoints",
        "start_checkpoint": None,
        "resume_optimizer": False,
        "recent": 3,
        "resume_latest": False,
        "prefix": None,
    },
}


class PBGTrainState(NamedTuple):
    params: Any
    opt_state: optax.OptState
    step: Array


_LOSS_TOKEN_CALIBRATION_BATCHES = 16


def _select_dataclass_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Merge a JSON run config onto :data:`PARAMETERS`."""
    with open(Path(path), encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return _merge_dicts(PARAMETERS, config)


def _resolve_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Join relative data and checkpoint paths with their bases; resolve ``resume_latest``."""
    parameters = deepcopy(parameters)
    data = parameters.get("data")
    data_base = parameters.get("data_base")
    if isinstance(data, dict):
        for item in data.get("sources") or []:
            path = Path(item["dir"])
            if path.is_absolute():
                continue
            if not data_base:
                raise ValueError("data_base required for relative source dir")
            item["dir"] = str(Path(data_base) / path)

    checkpoint = parameters.get("checkpoint")
    checkpoint_base = parameters.get("checkpoint_base")
    if isinstance(checkpoint, dict) and checkpoint_base:
        checkpoint_dir = checkpoint.get("dir")
        if checkpoint_dir is not None:
            path = Path(checkpoint_dir)
            if not path.is_absolute():
                checkpoint["dir"] = str(Path(checkpoint_base) / path)
        start_checkpoint = checkpoint.get("start_checkpoint")
        if start_checkpoint is not None:
            path = Path(start_checkpoint)
            if not path.is_absolute():
                checkpoint["start_checkpoint"] = str(Path(checkpoint_base) / path)
    checkpoint = parameters.get("checkpoint", {})
    if bool(checkpoint.get("resume_latest", False)):
        _resolve_latest_checkpoint(parameters, _checkpoint_prefix(parameters))
    return parameters


def _validate_parameters(parameters: dict[str, Any]) -> None:
    data = parameters["data"]
    if int(data["batch_size"]) <= 0:
        raise ValueError("data.batch_size must be positive")
    if int(data.get("prefetch", 2)) <= 0:
        raise ValueError("data.prefetch must be positive")
    if int(data.get("data_loader_workers", 1)) <= 0:
        raise ValueError("data.data_loader_workers must be positive")
    if int(data.get("prefetch_per_process", 1)) <= 0:
        raise ValueError("data.prefetch_per_process must be positive")
    if int(data.get("data_loader_workers_per_process", 1)) <= 0:
        raise ValueError("data.data_loader_workers_per_process must be positive")
    if len(data["sources"]) != 1:
        raise ValueError("data.sources must hold exactly one source")
    ambiguous = sorted(SOURCE_KEYS.intersection(data))
    if ambiguous:
        raise ValueError(
            "data.sources is enabled, so source-specific keys must live only on "
            f"each source, not data: {', '.join(ambiguous)}"
        )
    source = data["sources"][0]
    if "environment" in source:
        raise ValueError(
            "data.sources[0].environment was replaced by environment_prompt"
        )
    missing = sorted({"adapter", "dir", "environment_prompt"} - set(source))
    if missing:
        raise ValueError(
            f"data.sources[0] is missing required keys: {', '.join(missing)}"
        )
    prompts = source["environment_prompt"]
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError(
            "data.sources[0].environment_prompt must be a non-empty mapping"
        )
    if not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in prompts.items()
    ):
        raise ValueError(
            "data.sources[0].environment_prompt keys and values must be strings"
        )
    if "frame_id_below" in source:
        frame_id_below = source["frame_id_below"]
        if isinstance(frame_id_below, bool) or not isinstance(frame_id_below, int) or frame_id_below < 1:
            raise ValueError("data.sources[0].frame_id_below must be a positive integer")
        if source["adapter"] != "many_peptides":
            raise ValueError("data.sources[0].frame_id_below requires a many_peptides source")
    if "sequence_allowlist" in source:
        allowlist = Path(str(source["sequence_allowlist"]))
        if not allowlist.is_absolute():
            allowlist = CONFIG_ROOT.parent / allowlist
        if not allowlist.is_file():
            raise ValueError(f"data.sources[0].sequence_allowlist {allowlist} does not exist")
        if source["adapter"] != "many_peptides":
            raise ValueError("data.sources[0].sequence_allowlist requires a many_peptides source")
    for section in ("model", "optimizer", "data", "distributed", "checkpoint"):
        retire(section, parameters.get(section, {}))
    retire("source", source)
    start_checkpoint = parameters.get("checkpoint", {}).get("start_checkpoint")
    if start_checkpoint and not Path(start_checkpoint).is_file():
        ckpt_dir = Path(parameters.get("checkpoint", {}).get("dir", ""))
        hint = ""
        if ckpt_dir.is_dir():
            names = sorted(p.name for p in ckpt_dir.glob("*.ckpt"))
            if names:
                hint = f"; available in {ckpt_dir}: {', '.join(names[-5:])}"
        raise FileNotFoundError(f"checkpoint not found: {start_checkpoint}{hint}")


def _init_wandb(parameters: dict[str, Any]) -> tuple[Any | None, bool]:
    wandb_cfg = parameters.get("training", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None, False
    try:
        import wandb

        init_kwargs: dict[str, Any] = {
            "project": str(wandb_cfg.get("project", "pbg")),
            "name": parameters.get("run_name"),
            "notes": parameters.get("run_notes"),
            "config": parameters,
        }
        if wandb_cfg.get("group"):
            init_kwargs["group"] = str(wandb_cfg["group"])
        # A requeued Slurm job keeps its job ID, so it resumes the same W&B run.
        slurm_ids = (os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID"))
        run_id = wandb_cfg.get("run_id") or "_".join(part for part in slurm_ids if part)
        if run_id:
            init_kwargs["id"] = str(run_id)
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)
        run.define_metric("train/tokens_seen")
        run.define_metric("train/loss_tokens_seen")
        log.info("wandb run: project=%s id=%s", init_kwargs["project"], run.id)
        return run, True
    except Exception as exc:
        log.warning("wandb init failed: %s", exc)
        return None, False


_T = TypeVar("_T")


class _PrefetchIterator(Generic[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        buf_size: int,
        *,
        name: str = "prefetch",
        wait_log_s: float = 30.0,
    ) -> None:
        self._q: queue.Queue[object] = queue.Queue(maxsize=max(1, int(buf_size)))
        self._sentinel = object()
        self._stop = threading.Event()
        self._name = name
        self._wait_log_s = wait_log_s

        def fill() -> None:
            try:
                for item in iterator:
                    while not self._stop.is_set():
                        try:
                            self._q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if self._stop.is_set():
                        return
            except Exception as exc:
                log.exception("prefetch thread failed")
                while not self._stop.is_set():
                    try:
                        self._q.put(exc, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            while not self._stop.is_set():
                try:
                    self._q.put(self._sentinel, timeout=0.2)
                    break
                except queue.Full:
                    continue

        threading.Thread(target=fill, daemon=True, name=name).start()
        self._wait_start = time.monotonic()

    @property
    def buffered_batches(self) -> int:
        return self._q.qsize()

    @property
    def buffer_capacity(self) -> int:
        return self._q.maxsize

    def __iter__(self) -> "_PrefetchIterator[_T]":
        return self

    def __next__(self) -> _T:
        while True:
            try:
                item = self._q.get(timeout=max(float(self._wait_log_s), 1.0))
            except queue.Empty:
                log.info(
                    "%s waiting %.1fs queue=%d/%d",
                    self._name,
                    time.monotonic() - self._wait_start,
                    self._q.qsize(),
                    self._q.maxsize,
                )
                continue
            self._wait_start = time.monotonic()
            if item is self._sentinel:
                raise StopIteration
            if isinstance(item, Exception):
                raise item
            return item

    def close(self) -> None:
        self._stop.set()


def init_train_state(
    model: PBG,
    optimizer: optax.GradientTransformation,
    *,
    step: int = 0,
) -> PBGTrainState:
    arrays = model.qwen
    return PBGTrainState(
        params=arrays,
        opt_state=optimizer.init(arrays),
        step=jnp.asarray(int(step), dtype=jnp.int32),
    )


def _grad_norm(grads) -> Array:
    return jnp.sqrt(sum(jnp.sum(jnp.square(grad)) for grad in jax.tree_util.tree_leaves(grads)))


def _grouped_rms(tree: Any, num_layers: int) -> dict[str, Array]:
    """Geometric-mean RMS of the leaves of each depth third and parameter family."""
    groups: dict[str, list[Array]] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        keys = [str(getattr(key, "key", getattr(key, "idx", key))) for key in path]
        if "blocks" not in keys:
            group = keys[-1]
        else:
            layer = int(keys[keys.index("blocks") + 1])
            depth = ("bottom", "centre", "top")[min(2, 3 * layer // num_layers)]
            family = (
                "muon_attention"
                if "attn" in keys and leaf.ndim >= 2
                else ("muon_mlp" if "mlp" in keys and leaf.ndim >= 2 else "adam_norms")
            )
            group = f"{depth}/{family}"
        groups.setdefault(group, []).append(leaf)
    return {
        group: jnp.exp(
            jnp.mean(
                jnp.stack(
                    [
                        jnp.log(
                            jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32))))
                            + 1e-30
                        )
                        for x in leaves
                    ]
                )
            )
        )
        for group, leaves in groups.items()
    }


def _should_log_train_step(
    step: int,
    *,
    num_steps: int,
    log_every: int,
    final_log_steps: int = 0,
    final_log_every: int = 1,
) -> bool:
    if step < 1:
        return False
    if step == 1 or step == num_steps:
        return True
    if log_every > 0 and step % log_every == 0:
        return True
    if final_log_steps <= 0 or step <= num_steps - final_log_steps:
        return False
    return step % max(int(final_log_every), 1) == 0


def _make_jit_train_step(
    optimizer: optax.GradientTransformation,
    model: PBG,
    *,
    param_labels: Any = None,
    clip_threshold: float = 0.0,
    clip_warmup: int = 0,
    mesh: Any = None,
):
    @jax.jit
    def train_step(state: PBGTrainState, batch: dict[str, Array]):
        arrays = state.params
        if mesh is not None:
            arrays = constrain_tree_replicated(arrays, mesh)

        def objective(model_arrays):
            fresh = PBG(model_arrays, model.config, model.vocab)
            metrics = fresh.loss_metrics(batch)
            return metrics.loss, metrics

        with jax.profiler.TraceAnnotation("forward_backward"):
            (loss, loss_metrics), grads = jax.value_and_grad(objective, has_aux=True)(
                arrays
            )
        grad_norm = _grad_norm(grads)

        opt_state_in = state.opt_state
        with jax.profiler.TraceAnnotation("optimizer_update"):
            updates, opt_state = optimizer.update(grads, opt_state_in, arrays)
        arrays = optax.apply_updates(arrays, updates)
        new_state = PBGTrainState(
            params=arrays,
            opt_state=opt_state,
            step=state.step + 1,
        )
        adagc_out = _adagc_state(opt_state)
        metrics = {
            "loss": loss,
            "nll": loss_metrics.nll,
            "loss_tokens": loss_metrics.loss_tokens,
            "grad_norm": grad_norm,
            "grad_clip_tensors": (
                jnp.zeros((), dtype=jnp.int32) if adagc_out is None else adagc_out.num_clipped
            ),
            "self_distillation_weight_mean": loss_metrics.self_distillation_weight_mean,
        }
        metrics.update(
            {
                f"grad_rms/{k}": v
                for k, v in _grouped_rms(grads, model.config.num_layers).items()
            }
        )
        adagc = _adagc_state(opt_state_in)
        if adagc is not None:
            pressure = jax.tree.map(
                lambda grad, ref: jnp.where(
                    adagc.count >= clip_warmup,
                    _grad_norm((grad,)) / jnp.maximum(ref, 1e-12),
                    jnp.nan,
                ),
                grads,
                adagc.gamma,
            )
            metrics.update(
                {
                    f"clip_pressure/{k}": v
                    for k, v in _grouped_rms(pressure, model.config.num_layers).items()
                }
            )
            if param_labels is not None:
                pressure_leaves = jax.tree_util.tree_leaves(pressure)
                label_leaves = jax.tree_util.tree_leaves(param_labels)
                for label in ("muon", "adam"):
                    selected = [
                        p
                        for p, branch in zip(pressure_leaves, label_leaves)
                        if branch == label
                    ]
                    metrics[f"clip_fraction/{label}"] = (
                        (adagc.count >= clip_warmup)
                        * sum(p > clip_threshold for p in selected)
                        / len(selected)
                    )
        if mesh is not None:
            metrics = constrain_tree_replicated(metrics, mesh)
        return new_state, metrics

    return train_step


def _assert_finite_train_metrics(step: int, metrics: dict[str, float]) -> None:
    bad = [
        f"{key}={metrics[key]}"
        for key in ("loss", "grad_norm")
        if not math.isfinite(float(metrics[key]))
    ]
    if bad:
        raise RuntimeError(
            f"non-finite train metrics at step {step}: {', '.join(bad)}"
        )


def _adagc_state(opt_state: optax.OptState) -> AdaGCState | None:
    states = [
        x
        for x in jax.tree_util.tree_leaves(
            opt_state, is_leaf=lambda x: isinstance(x, AdaGCState)
        )
        if isinstance(x, AdaGCState)
    ]
    return states[0] if states else None


def _checkpoint_step(path: Path) -> int | None:
    try:
        return int(path.stem.rsplit("_step_", 1)[1])
    except (IndexError, ValueError):
        return None


def _checkpoint_prefix(parameters: dict[str, Any]) -> str:
    checkpoint = parameters.get("checkpoint", {})
    if checkpoint.get("prefix"):
        return str(checkpoint["prefix"])
    return str(parameters.get("run_name", "pbg"))


def _latest_checkpoint_for_prefix(
    checkpoint_dir: str | Path, prefix: str
) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in Path(checkpoint_dir).glob(f"{prefix}_step_*.ckpt"):
        step = _checkpoint_step(path)
        if step is not None:
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _resolve_latest_checkpoint(parameters: dict[str, Any], prefix: str) -> Path | None:
    checkpoint = parameters.get("checkpoint", {})
    if not bool(checkpoint.get("resume_latest", False)):
        return None
    latest = _latest_checkpoint_for_prefix(checkpoint["dir"], prefix)
    if latest is None:
        if bool(checkpoint.get("resume_latest_if_available", False)):
            log.info(
                "no checkpoint matching %s_step_*.ckpt; using configured start checkpoint",
                prefix,
            )
            return None
        raise FileNotFoundError(
            f"checkpoint.resume_latest=true but no checkpoint matching "
            f"{prefix}_step_*.ckpt exists in {checkpoint['dir']}"
        )
    checkpoint["start_checkpoint"] = str(latest)
    return latest


def _prune_checkpoints(
    checkpoint_dir: Path,
    prefix: str,
    recent: int,
    permanent: set[int],
) -> None:
    checkpoints = {
        step: path
        for path in checkpoint_dir.glob(f"{prefix}_step_*.ckpt")
        if (step := _checkpoint_step(path)) is not None
    }
    newest = set(sorted(checkpoints)[-recent:]) if recent > 0 else set()
    for step, path in checkpoints.items():
        if step not in permanent and step not in newest:
            path.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    state: PBGTrainState,
    *,
    model: PBG,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "params": jax.tree.map(_checkpoint_array, state.params),
        "config": asdict(model.config),
        "vocab": deepcopy(model.vocab),
        "opt_state": jax.tree.map(_checkpoint_array, state.opt_state),
        "step": int(jax.device_get(state.step)),
        "metadata": {} if metadata is None else dict(metadata),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)
    log.info("saved checkpoint %s", path)


def _checkpoint_array(value: Any) -> Any:
    if isinstance(value, (jax.Array, np.ndarray)):
        return np.asarray(jax.device_get(value)).copy()
    return value


def load_checkpoint(
    path: str | Path,
    optimizer: optax.GradientTransformation | None = None,
    *,
    model_override: PBG | None = None,
    optimizer_config: OptimizerConfig | None = None,
    metadata_out: dict[str, Any] | None = None,
) -> PBGTrainState | PBG:
    payload = load_pickle(path)
    if metadata_out is not None:
        metadata_out.update(payload.get("metadata", {}))
    model = PBG.load_payload(payload)
    if optimizer is None:
        return model
    state_model = model if model_override is None else model_override
    arrays = state_model.qwen
    if (
        "opt_state" in payload
        and "atom_conditioner" in arrays
        and "atom_conditioner" not in model.qwen
    ):
        raise ValueError(
            f"{path} holds a prefix-conditioning optimizer state; an atom-conditioning "
            "run can start from its weights only (checkpoint.resume_optimizer=false)"
        )
    if "opt_state" in payload and optimizer_config is not None:
        saved_optimizer = (
            payload.get("metadata", {}).get("parameters", {}).get("optimizer", {})
        )
        # Runs before decoupled weight decay saved no ``*_weight_decay_impl``.
        saved_muon_wd = saved_optimizer.get("muon_weight_decay_impl", "legacy")
        saved_adam_wd = saved_optimizer.get("adam_weight_decay_impl", "legacy_l2")
        saved_scaling_patches = saved_optimizer.get("muon_scaling_patches", False)
        # Turning AdaGC on or off changes the optimizer state tree (it adds a
        # per-tensor gamma), so it belongs in the resume compatibility check.
        saved_adagc = float(saved_optimizer.get("grad_clip_rel", 0.0)) > 0
        expected_policy = (
            "decoupled",
            "decoupled",
            optimizer_config.muon_scaling_patches,
            optimizer_config.grad_clip_rel > 0,
        )
        saved_policy = (
            saved_muon_wd,
            saved_adam_wd,
            saved_scaling_patches,
            saved_adagc,
        )
        if saved_policy != expected_policy:
            raise ValueError(
                "Optimizer policy differs from checkpoint state: "
                f"checkpoint={saved_policy}, configured={expected_policy} "
                "(muon_wd, adam_wd, scaling_patches, adagc). Set "
                "checkpoint.resume_optimizer=false or select the checkpoint's policy."
            )
    opt_state = (
        payload["opt_state"] if "opt_state" in payload else optimizer.init(arrays)
    )
    return PBGTrainState(
        params=arrays,
        opt_state=opt_state,
        step=jnp.asarray(int(payload.get("step", 0)), dtype=jnp.int32),
    )


_POSITIONAL_ENCODING_KEYS = ("rope_theta", "level_pe_theta")


def _validate_positional_encoding(
    config: PBGConfig, requested: dict[str, Any], checkpoint: str
) -> None:
    """Fail loudly when a config requests a different positional encoding.

    ``rope_theta`` and ``level_pe_theta`` cannot be overridden when starting
    from an existing checkpoint: the weights encode positions relative to the
    theta they were trained with. Previously these keys were silently dropped,
    so configs asking for a new positional encoding trained with the parent's
    instead and the intended experiment never ran.
    """
    mismatches = {
        key: (getattr(config, key), requested[key])
        for key in _POSITIONAL_ENCODING_KEYS
        if key in requested and float(requested[key]) != float(getattr(config, key))
    }
    if not mismatches:
        return
    detail = ", ".join(
        f"{key}: checkpoint={found!r} config={want!r}"
        for key, (found, want) in sorted(mismatches.items())
    )
    raise ValueError(
        f"Positional encoding differs from start_checkpoint {checkpoint}: {detail}. "
        "These are baked into the trained weights and cannot be overridden on "
        "resume. Either drop the key from the config to keep the checkpoint's "
        "value, or train from scratch (checkpoint.start_checkpoint=null) to use "
        "the requested value."
    )


def _build_model(
    parameters: dict[str, Any], *, checkpoint_path: str | Path | None = None
) -> PBG:
    checkpoint = (
        parameters["checkpoint"].get("start_checkpoint")
        if checkpoint_path is None
        else checkpoint_path
    )
    if checkpoint:
        log.info("loading model from %s", checkpoint)
        model = load_checkpoint(checkpoint)
        assert isinstance(model, PBG)
        requested_order = parameters["model"].get("atom_order", model.config.atom_order)
        if parameters["checkpoint"].get("resume_optimizer", False) and requested_order != model.config.atom_order:
            raise ValueError("Cannot change atom traversal when resuming a trained optimizer/cursor")
        requested_conditioning = parameters["model"].get("conditioning", model.config.conditioning)
        if requested_conditioning != model.config.conditioning:
            # load_checkpoint refuses to resume a prefix optimizer state here.
            if (model.config.conditioning, requested_conditioning) != ("prefix", "atom"):
                raise ValueError(
                    f"start_checkpoint {checkpoint} uses conditioning="
                    f"{model.config.conditioning!r}, config requests {requested_conditioning!r}"
                )
            # An atom-conditioning run may start from a prefix parent: the trunk
            # is the parent's and the conditioner is new, as in PBG.init.
            config = replace(model.config, conditioning="atom")
            key = jax.random.fold_in(jax.random.PRNGKey(int(parameters.get("seed", 0))), 1)
            model = PBG(
                qwen=dict(model.qwen, atom_conditioner=init_atom_conditioner(key, config)),
                config=config,
                vocab=model.vocab,
            )
        checkpoint_overrides = {
            key: parameters["model"][key]
            for key in (
                "augment",
                "translate",
                "bin_tokens",
                "max_seq_len",
                "max_samples_factor",
                "prompt_format",
                "atom_order",
                "bin_width_pm",
                "vocab_name",
                "mixed_precision",
                "remat_blocks",
                "residual_scale",
                "unembedding_scale",
                "center_of_mass",
                "loss_reduction",
                "loss_terminator",
            )
            if key in parameters["model"]
        }
        # Positional-encoding geometry is baked into the trained weights: the
        # attention bands and level embeddings are learned against a specific
        # theta. Silently overriding it on resume would reinterpret every
        # learned position, so these keys are deliberately excluded from
        # ``checkpoint_overrides`` above and validated instead.
        _validate_positional_encoding(model.config, parameters["model"], checkpoint)
        if checkpoint_overrides:
            model = PBG(
                qwen=model.qwen,
                config=replace(model.config, **checkpoint_overrides),
                vocab=model.vocab,
            )
            PBG._validate_vocab_embed(
                qwen_vocab_size(model.qwen), model.vocab, model.config
            )
        return model
    config_overrides = _select_dataclass_fields(PBGConfig, parameters["model"])
    config = PBGConfig(**config_overrides)
    return PBG.init(config, jax.random.PRNGKey(int(parameters.get("seed", 0))))


_PREEMPT_SAVE_REQUESTED = False


def _request_preempt_save(signum: int, _frame: object) -> None:
    global _PREEMPT_SAVE_REQUESTED
    _PREEMPT_SAVE_REQUESTED = True
    log.warning(
        "received signal %s; will save checkpoint and exit after this step",
        signum,
    )


def _run_training(parameters: dict[str, Any]) -> None:
    training_cfg = parameters.get("training", {})
    ckpt_prefix = _checkpoint_prefix(parameters)
    latest_checkpoint = _resolve_latest_checkpoint(parameters, ckpt_prefix)
    if latest_checkpoint is not None and (
        _checkpoint_step(latest_checkpoint) or 0
    ) >= int(training_cfg["num_steps"]):
        return
    ctx = bootstrap_distributed()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if training_cfg.get("log_file") is not None:
        log_path = Path(training_cfg["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [rank {ctx.process_index}/{ctx.process_count}] %(message)s",
        handlers=handlers,
        force=True,
    )
    # Slurm ``--signal=USR1@N`` (or SIGTERM) asks for a final checkpoint before the kill.
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_preempt_save)
    signal.signal(signal.SIGTERM, _request_preempt_save)

    devices = jax.devices()
    platforms = sorted({device.platform for device in devices})
    log.info(
        "jax devices: local=%d global=%d processes=%d initialized=%s devices=%s",
        ctx.local_device_count,
        ctx.global_device_count,
        ctx.process_count,
        ctx.initialized,
        ",".join(str(device) for device in devices),
    )
    if platforms == ["cpu"] and not (
        bool(training_cfg.get("allow_cpu_training", False))
        or os.environ.get("PBG_ALLOW_CPU_TRAINING") == "1"
    ):
        raise RuntimeError(
            "JAX sees only CPU devices. Refusing to compile the PBG train_step on CPU; "
            "the 0.6B model compile/step can look like a hang on login nodes. "
            "Run on a GPU node, or set PBG_ALLOW_CPU_TRAINING=1 / "
            "training.allow_cpu_training=true to override."
        )

    run_name = str(parameters.get("run_name", "pbg"))
    run, wandb_enabled = _init_wandb(parameters) if ctx.is_leader else (None, False)
    if latest_checkpoint is not None:
        log.info(
            "resuming latest checkpoint for prefix %s: %s",
            ckpt_prefix,
            latest_checkpoint,
        )

    checkpoint = parameters["checkpoint"].get("start_checkpoint")
    t_setup = time.monotonic()
    log.info("building model")
    model = _build_model(parameters, checkpoint_path=checkpoint)
    log.info(
        "model ready in %.1fs vocab=%s max_seq_len=%d qwen_vocab=%d",
        time.monotonic() - t_setup,
        model.config.vocab_name or "legacy",
        model.config.max_seq_len,
        model.qwen["embed"].shape[0],
    )
    arrays = model.qwen
    log.info("building optimizer")
    opt_config = OptimizerConfig(
        **_select_dataclass_fields(OptimizerConfig, parameters["optimizer"])
    )
    if opt_config.schedule not in {"wsd", "cosine"}:
        raise ValueError("pbg.models.pbg.train expects optimizer.schedule 'wsd' or 'cosine'")
    if opt_config.schedule == "cosine" and (
        opt_config.cooldown_steps or opt_config.total_steps <= 0
    ):
        # The cosine decay spans total_steps; a WSD cooldown has no meaning.
        raise ValueError("schedule='cosine' needs total_steps > 0 and cooldown_steps=0")
    optimizer, labels = create_optimizer(
        arrays, config=opt_config
    )
    checkpoint_metadata: dict[str, Any] = {}
    if checkpoint and bool(parameters["checkpoint"].get("resume_optimizer", False)):
        t_state = time.monotonic()
        log.info("loading optimizer state from %s", checkpoint)
        state = load_checkpoint(
            checkpoint,
            optimizer,
            model_override=model,
            optimizer_config=opt_config,
            metadata_out=checkpoint_metadata,
        )
        assert isinstance(state, PBGTrainState)
        log.info(
            "optimizer state loaded in %.1fs step=%d",
            time.monotonic() - t_state,
            int(jax.device_get(state.step)),
        )
    else:
        t_state = time.monotonic()
        log.info("initializing optimizer state")
        state = init_train_state(model, optimizer)
        log.info("optimizer state initialized in %.1fs", time.monotonic() - t_state)
    per_replica_rows = int(parameters["data"]["batch_size"])
    process_rows = per_replica_rows * ctx.local_device_count
    train_rows = per_replica_rows * ctx.global_device_count
    mesh = build_mesh(ctx)
    t_mesh = time.monotonic()
    state = put_train_state(state, mesh)
    log.info("train state placed on mesh=%s in %.1fs", mesh, time.monotonic() - t_mesh)

    train_step = _make_jit_train_step(
        optimizer,
        model,
        param_labels=labels,
        clip_threshold=opt_config.grad_clip_rel,
        clip_warmup=opt_config.grad_clip_rel_warmup,
        mesh=mesh,
    )
    num_steps = int(training_cfg["num_steps"])
    log_every = int(training_cfg.get("log_every", 20))
    final_log_steps = int(training_cfg.get("final_log_steps", 0))
    final_log_every = int(training_cfg.get("final_log_every", 1))
    save_every = int(training_cfg.get("save_every", 0))
    ckpt_dir = Path(parameters["checkpoint"]["dir"])
    permanent_checkpoints = {
        step
        for step in (
            *(int(step) for step in parameters["checkpoint"].get("permanent_steps", [])),
            int(opt_config.warmup_steps),
            int(opt_config.total_steps) - int(opt_config.cooldown_steps),
            int(opt_config.total_steps),
            num_steps,
        )
        if 0 < step <= num_steps
    }
    data_cfg = parameters["data"]
    finite_molecular = bool(data_cfg["sources"][0].get("molecular_single_pass"))
    if finite_molecular:
        if ctx.process_count != 1:
            raise ValueError("finite molecular pilot requires one host")
        plan = parameters["planning"]
        append_only_molecular = bool(data_cfg['sources'][0].get('molecular_append_only'))
        if not append_only_molecular and int(plan["finite_batches"]) != num_steps:
            raise ValueError("training endpoint must equal the measured finite pilot pass")
        if checkpoint_metadata:
            previous = checkpoint_metadata.get("parameters", {}).get("planning", {})
            if not append_only_molecular and previous.get("pilot_manifest_sha256") != plan["pilot_manifest_sha256"]:
                raise ValueError("cannot resume finite pass with a different pilot manifest")
            previous_model = checkpoint_metadata.get("parameters", {}).get("model", {})
            if previous_model.get("prompt_format", "legacy") != model.config.prompt_format:
                raise ValueError("cannot resume finite pass with a different prompt format")
    prefetch_batches = int(
        data_cfg.get(
            "prefetch_per_process",
            int(data_cfg.get("prefetch", 2)) * ctx.local_device_count,
        )
    )
    data_loader_workers = int(
        data_cfg.get(
            "data_loader_workers_per_process",
            int(data_cfg.get("data_loader_workers", 1)) * ctx.local_device_count,
        )
    )
    batch_parameters = deepcopy(parameters)
    batch_parameters["data"] = dict(
        batch_parameters["data"],
        batch_size=process_rows,
        prefetch=prefetch_batches,
        data_loader_workers=data_loader_workers,
    )
    resume_optimizer = bool(
        checkpoint and parameters["checkpoint"].get("resume_optimizer", False)
    )
    saved_cursor = checkpoint_metadata.get("data_cursor") if resume_optimizer else None
    if resume_optimizer and parse_saved_cursor(saved_cursor) is None:
        log.warning(
            "checkpoint has no data_cursor; data streams restart at shard 0"
        )
    cursor_tracker = make_data_cursor_tracker(batch_parameters["data"], saved=saved_cursor)
    batches = _PrefetchIterator(
        iter_training_batches(
            model,
            batch_parameters["data"],
            seed=int(parameters.get("seed", 0)) + 100_003 * ctx.process_index,
            start_step=int(jax.device_get(state.step)),
            process_index=ctx.process_index,
            process_count=ctx.process_count,
            data_cursor=cursor_tracker,
        ),
        prefetch_batches,
        name="train-batch-prefetch",
    )

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(arrays) if hasattr(x, "size")
    )
    log.info(
        "training %s: params=%s start_step=%d target_step=%d schedule=%s warmup=%d "
        "cooldown=%d global_devices=%d local_devices=%d "
        "per_replica_rows=%d process_rows=%d global_rows=%d prefetch=%d data_loader_workers=%d",
        run_name,
        f"{n_params:,}",
        int(jax.device_get(state.step)),
        num_steps,
        opt_config.schedule,
        opt_config.warmup_steps,
        opt_config.cooldown_steps,
        ctx.global_device_count,
        ctx.local_device_count,
        per_replica_rows,
        process_rows,
        train_rows,
        prefetch_batches,
        data_loader_workers,
    )
    wallclock_start = time.monotonic()
    start_step = int(jax.device_get(state.step))
    loss_tokens_per_step = checkpoint_metadata.get("loss_tokens_per_step")
    exact_loss_tokens = int(checkpoint_metadata.get("exact_loss_tokens_seen", 0))
    calibration_loss_tokens = 0.0
    calibration_batches = 0
    t0 = time.monotonic()
    last_log_step = start_step
    t_fetch_sum = 0.0
    t_step_sum = 0.0
    n_timed = 0
    for _ in range(int(jax.device_get(state.step)), num_steps):
        step_before = int(jax.device_get(state.step))
        t_fetch = time.monotonic()
        if step_before == start_step:
            log.info("fetching first training batch")
        batch_np = next(batches)
        batch_np, cursor_meta = split_cursor_batch(batch_np)
        fetch_seconds = time.monotonic() - t_fetch
        t_fetch_sum += fetch_seconds
        batch_loss_tokens = float(np.asarray(batch_np.get("loss_mask", 0.0)).sum())
        if step_before == start_step:
            token_shape = batch_np.get("tokens", np.asarray(())).shape
            total_tokens = (
                int(np.prod(token_shape, dtype=np.int64)) if token_shape else 0
            )
            log.info(
                "first batch ready in %.1fs tokens_shape=%s total_tokens=%d loss_tokens=%.0f",
                time.monotonic() - t_fetch,
                token_shape,
                total_tokens,
                batch_loss_tokens,
            )
        batch = put_batch(batch_np, mesh, ctx)
        t_step = time.monotonic()
        if step_before == start_step:
            log.info("starting first train_step (JIT compile + execute)")
        with jax.profiler.StepTraceAnnotation("train", step_num=step_before):
            state, step_metrics = train_step(state, batch)
            jax.block_until_ready((state, step_metrics))
        t_step_sum += time.monotonic() - t_step
        jax.experimental.multihost_utils.sync_global_devices(
            f"pbg-after-gpu-{step_before}"
        )
        if step_before == start_step:
            log.info("first train_step finished in %.1fs", time.monotonic() - t_step)
        cursor_tracker.note_trained_batch(cursor_meta)
        exact_loss_tokens += int(batch_loss_tokens)
        n_timed += 1

        step = int(jax.device_get(state.step))
        step_floats = {key: float(np.asarray(value)) for key, value in jax.device_get(step_metrics).items()}
        _assert_finite_train_metrics(step, step_floats)
        loss_metrics = step_floats if ctx.is_leader else None
        if ctx.is_leader:
            assert loss_metrics is not None
            if (
                calibration_batches < _LOSS_TOKEN_CALIBRATION_BATCHES
                and not checkpoint_metadata.get("loss_tokens_per_step")
            ):
                calibration_loss_tokens += loss_metrics["loss_tokens"]
                calibration_batches += 1
                if calibration_batches == _LOSS_TOKEN_CALIBRATION_BATCHES:
                    loss_tokens_per_step = round(
                        calibration_loss_tokens / calibration_batches
                    )
                    log.info(
                        "token accounting calibrated over %d batches: allocated_per_step=%d "
                        "loss_per_step=%d loss_fraction=%.4f",
                        calibration_batches,
                        train_rows * model.config.max_seq_len,
                        loss_tokens_per_step,
                        loss_tokens_per_step / (train_rows * model.config.max_seq_len),
                    )
        if ctx.is_leader and _should_log_train_step(
            step,
            num_steps=num_steps,
            log_every=log_every,
            final_log_steps=final_log_steps,
            final_log_every=final_log_every,
        ):
            assert loss_metrics is not None
            lr_muon, lr_adam = current_learning_rates(opt_config, step_before)
            elapsed = max(time.monotonic() - t0, 1e-9)
            wallclock = time.monotonic() - wallclock_start
            steps_since_log = max(1, step - last_log_step)
            steps_per_second = steps_since_log / elapsed
            tokens_per_second = (
                steps_per_second * train_rows * model.config.max_seq_len
            )
            log.info(
                "step %d loss %.5f grad %.4g clips %.0f lr_mu %.5g lr_ad %.5g %.3f step/s "
                "%.0f tok/s "
                "wallclock=%.1fs buffered_batches=%d/%d t_step=%.3fs",
                step,
                loss_metrics["loss"],
                loss_metrics["grad_norm"],
                loss_metrics["grad_clip_tensors"],
                lr_muon,
                lr_adam,
                steps_per_second,
                tokens_per_second,
                wallclock,
                batches.buffered_batches,
                batches.buffer_capacity,
                t_step_sum / max(n_timed, 1),
            )
            if wandb_enabled and run is not None:
                wandb_metrics = {
                    "train/loss": loss_metrics["loss"],
                    "train/nll": loss_metrics["nll"],
                    "train/grad_norm": loss_metrics["grad_norm"],
                    "train/grad_clip_tensors": loss_metrics["grad_clip_tensors"],
                    "train/lr_muon": lr_muon,
                    "train/lr_adam": lr_adam,
                    "train/steps_per_second": steps_per_second,
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens_seen": step * train_rows * model.config.max_seq_len,
                    "train/self_distillation_weight_mean": loss_metrics[
                        "self_distillation_weight_mean"
                    ],
                }
                if finite_molecular:
                    wandb_metrics["train/exact_loss_tokens_seen"] = exact_loss_tokens
                if loss_tokens_per_step is not None:
                    wandb_metrics["train/loss_tokens_seen"] = round(step * float(loss_tokens_per_step))
                wandb_metrics.update(
                    {
                        f"train/{key}": value
                        for key, value in loss_metrics.items()
                        if "/" in key
                    }
                )
                run.log(wandb_metrics, step=step)
            t0 = time.monotonic()
            last_log_step = step
            t_fetch_sum = 0.0
            t_step_sum = 0.0
            n_timed = 0

        preempt_save = bool(_PREEMPT_SAVE_REQUESTED)
        checkpoint_due = (
            preempt_save
            or (save_every > 0 and step % save_every == 0)
            or (step in permanent_checkpoints)
        )
        if checkpoint_due:
            jax.experimental.multihost_utils.sync_global_devices(
                f"pbg-checkpoint-{step}-before"
            )
            # Each shard's highest pass over all hosts.
            data_cursor_payload = cursor_tracker.to_serializable()
            if ctx.process_count > 1 and cursor_tracker.shard_pass.size:
                gathered = jax.experimental.multihost_utils.process_allgather(
                    jnp.asarray(cursor_tracker.shard_pass, dtype=jnp.int32)
                )
                data_cursor_payload["sources"][cursor_tracker.key]["shard_pass"] = (
                    np.asarray(gathered).max(axis=0).astype(np.int64).tolist()
                )
            if ctx.is_leader:
                save_checkpoint(
                    ckpt_dir / f"{ckpt_prefix}_step_{step}.ckpt",
                    state,
                    model=model,
                    metadata={
                        "parameters": parameters,
                        "loss_tokens_per_step": loss_tokens_per_step,
                        "exact_loss_tokens_seen": exact_loss_tokens,
                        "data_cursor": data_cursor_payload,
                    },
                )
                if preempt_save:
                    log.info("preemption checkpoint saved at step %d", step)
                _prune_checkpoints(
                    ckpt_dir,
                    ckpt_prefix,
                    int(parameters["checkpoint"].get("recent", 3)),
                    permanent_checkpoints,
                )
            jax.experimental.multihost_utils.sync_global_devices(
                f"pbg-checkpoint-{step}-after"
            )
            if preempt_save:
                log.info("exiting after preemption save at step %d", step)
                break
        jax.experimental.multihost_utils.sync_global_devices(
            f"pbg-after-host-{step}"
        )
    if wandb_enabled and run is not None:
        run.finish()


def run(parameters: dict[str, Any]) -> None:
    """Train from merged parameters (see :func:`load_config`).

    With ``checkpoint.auto_resume`` the run continues from the newest
    checkpoint of its prefix, with optimizer state and data cursor, if one
    exists; an interrupted run restarts with the same command.
    """
    parameters = deepcopy(parameters)
    checkpoint = parameters.setdefault("checkpoint", {})
    if checkpoint.get("auto_resume"):
        checkpoint.update(resume_latest=True, resume_latest_if_available=True, resume_optimizer=True)
    parameters = _resolve_parameters(parameters)
    _validate_parameters(parameters)
    _run_training(parameters)
