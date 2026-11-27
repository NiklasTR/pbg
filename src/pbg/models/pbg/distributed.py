"""Data-parallel JAX helpers for PBG training.

Every device holds a full replica of the parameters and the optimizer state;
each batch is split along its rows over a 1D ``data`` mesh of all devices.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedContext:
    process_index: int
    process_count: int
    local_device_count: int
    global_device_count: int
    initialized: bool

    @property
    def is_leader(self) -> bool:
        return self.process_index == 0


def bootstrap_distributed() -> DistributedContext:
    """Initialize JAX distributed from Slurm and return the device counts.

    JAX auto-detects Slurm when ``jax.distributed.initialize`` is called with
    no explicit coordinator/process arguments. This must run before any JAX
    device computation in the training process.
    """

    def env_int(name: str) -> int:
        try:
            return int(os.environ.get(name) or 1)
        except ValueError:
            return 1

    initialized = (
        os.environ.get("PBG_FORCE_DISTRIBUTED") == "1"
        or env_int("SLURM_NTASKS") > 1
        or env_int("SLURM_NNODES") > 1
    )
    if initialized:
        jax.distributed.initialize()
    global_device_count = len(jax.devices())
    if global_device_count <= 0:
        raise RuntimeError("JAX did not report any devices")
    return DistributedContext(
        process_index=int(jax.process_index()),
        process_count=int(jax.process_count()),
        local_device_count=len(jax.local_devices()),
        global_device_count=global_device_count,
        initialized=initialized,
    )


def build_mesh(ctx: DistributedContext) -> Mesh:
    """Return a 1D ``data`` mesh over all devices."""
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(devices.reshape((ctx.global_device_count,)), axis_names=("data",))


def constrain_tree_replicated(tree: Any, mesh: Mesh) -> Any:
    """Pin every array leaf in a pytree to a fully replicated layout."""
    replicated = NamedSharding(mesh, PS())
    return jax.tree_util.tree_map(
        lambda value: (
            jax.lax.with_sharding_constraint(value, replicated)
            if isinstance(value, (jax.Array, np.ndarray))
            else value
        ),
        tree,
    )


def put_train_state(state: Any, mesh: Mesh) -> Any:
    """Place a full replica of the train state on every device."""
    replicated = NamedSharding(mesh, PS())
    shardings = jax.tree_util.tree_map(
        lambda value: replicated if isinstance(value, (jax.Array, np.ndarray)) else None,
        state,
    )
    return jax.device_put(state, shardings)


def put_batch(batch: dict[str, np.ndarray], mesh: Mesh, ctx: DistributedContext) -> dict[str, Array]:
    """Create global JAX arrays from each process' local batch shard."""

    out: dict[str, Array] = {}
    local_batch_rows = int(np.asarray(batch["tokens"]).shape[0]) if "tokens" in batch else None
    for key, value in batch.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] > 0 and array.shape[0] == local_batch_rows:
            # Shard the leading (row) axis only. Column shardings interact
            # badly with elementwise ops between row-major and per-token
            # arrays (e.g. a (rows, cols) weight times a (rows,) mask), so
            # every non-leading axis stays replicated by construction.
            sharding = NamedSharding(mesh, PS("data", *([None] * (array.ndim - 1))))
            if ctx.process_count > 1:
                global_shape = (
                    array.shape[0] * ctx.process_count,
                    *array.shape[1:],
                )
                out[key] = jax.make_array_from_process_local_data(
                    sharding,
                    array,
                    global_shape=global_shape,
                )
            else:
                out[key] = jax.device_put(array, sharding)
        elif array.ndim == 0:
            out[key] = jax.device_put(array, NamedSharding(mesh, PS()))
        else:
            log.debug("skipping non-row-major batch array %s shape=%s", key, array.shape)
    return out
