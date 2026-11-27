"""Load checkpoint pickles written by this package or by the research code base.

All paper checkpoints store ``params``, ``config`` and ``vocab`` as plain
Python and NumPy values (``format_version`` 2). Their optimizer state pickles
one class by module path, the AdaGC state, which the research code base kept
in ``thermax.models.optimiser``; the unpickler maps that name to this package.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

# Earlier module paths of ``AdaGCState``: the research code base, then this package.
_OPTIMIZER_MODULES = (
    "thermax.models.optimiser",
    "pbg.models.optimiser",
    "pbg.generator.optimiser",
    "pbg.models.pbg.optimiser",
)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module in _OPTIMIZER_MODULES:
            module = "pbg.models.pbg.optimizer"
        return super().find_class(module, name)


def load_pickle(path: str | Path) -> Any:
    """Unpickle a trusted checkpoint file."""
    with open(path, "rb") as handle:
        return _Unpickler(handle).load()


# Any value is accepted and ignored.
ANY = object()

# Options that the research code read and this release fixes to one value.
# Run configs, checkpoint configs and data cursors written earlier still carry
# them, so ``retire`` accepts them at that value and drops them.
RETIRED_OPTIONS: dict[str, dict[str, Any]] = {
    "model": {
        "coordinate_method": "bin",
        "undiscretize_method": "center",
        "octree_tokens": ANY,
        "octree_order": ANY,
        "tokeniser_raise_on_unk": ANY,
        "max_position": ANY,
        "use_environment_tokens": False,
        "use_temperature_tokens": False,
        "reorder": False,
        "permute_intra": False,
        "permute_inter": False,
        "reflect": False,
        "self_distillation_condition": "naive",
    },
    "optimizer": {
        "orthogonalization": "newton_schulz",
        "ns_order": 5,
        "ns_rescale": False,
        "ns_after_momentum": True,
        "normuon": False,
        "normuon_beta2": ANY,
        "cautious_wd": False,
        "nesterov": True,
        "transpose_for_ns": True,
        "grad_clip_position": "before_optimizer",
        "muon_weight_decay_impl": "decoupled",
        "adam_weight_decay_impl": "decoupled",
        "hyperball": False,
    },
    "data": {"batch_mode": "mixed", "mix": {"mode": "row_weighted_fixed"}},
    "source": {"source_format": "webdataset_tar", "compute_energy": False, "weight": ANY},
    "distributed": {"zero_stage": 0, "parameter_sharding": "ddp"},
    "checkpoint": {
        "allow_missing_start_checkpoint": ANY,
        "scaling_base": 0,
        "load_rocm_compatible_checkpoint": False,
    },
}


def retire(section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Return ``values`` without the retired options of ``section``.

    Raises ``ValueError`` when a retired option holds another value than the
    one this release implements.
    """
    retired = RETIRED_OPTIONS[section]
    for key, required in retired.items():
        if key in values and required is not ANY and values[key] != required:
            raise ValueError(
                f"{section}.{key}={values[key]!r} is not supported; this release fixes it to {required!r}"
            )
    return {key: value for key, value in values.items() if key not in retired}
