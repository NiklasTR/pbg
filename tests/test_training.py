"""Training: configuration, path resolution, checkpoints, model construction, the
train step, data mixing on resume, distributed sharding and loader workers."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import random

import pbg.models.pbg.loader as loader
from pbg.models.pbg.model import PBG
from pbg.models.pbg.optimizer import OptimizerConfig, create_optimizer
from pbg.models.pbg.train import (
    PARAMETERS,
    PBGTrainState,
    _assert_finite_train_metrics,
    _build_model,
    _grouped_rms,
    _init_wandb,
    _latest_checkpoint_for_prefix,
    _make_jit_train_step,
    _prune_checkpoints,
    _resolve_latest_checkpoint,
    _resolve_parameters,
    _run_training,
    _should_log_train_step,
    _validate_parameters,
    init_train_state,
    load_checkpoint,
    run,
    save_checkpoint,
)

# ── Configuration validation ─────────────────────────────────────────


@pytest.mark.parametrize(
    "key", ["prefetch", "data_loader_workers", "prefetch_per_process", "data_loader_workers_per_process"]
)
def test_validate_parameters_requires_positive_loader_sizes(key) -> None:
    parameters = deepcopy(PARAMETERS)
    parameters["data"][key] = 4
    _validate_parameters(parameters)

    parameters["data"][key] = 0
    with pytest.raises(ValueError, match=key):
        _validate_parameters(parameters)


def test_validate_parameters_requires_an_environment_prompt_mapping() -> None:
    parameters = deepcopy(PARAMETERS)
    source = {
        "name": "mixed",
        "dir": "/tmp/mixed",
        "adapter": "n4_smiles",
        "environment_prompt": {
            "vacuum": "GAFF2/AM1-BCC in vacuum",
            "implicit": "GAFF2/AM1-BCC in OBC1 implicit water",
        },
    }
    parameters["data"]["sources"] = [source]
    _validate_parameters(parameters)

    source["environment"] = "vacuum"
    with pytest.raises(ValueError, match="replaced by environment_prompt"):
        _validate_parameters(parameters)

    del source["environment"], source["environment_prompt"]
    with pytest.raises(ValueError, match="missing required keys: environment_prompt"):
        _validate_parameters(parameters)


def test_validate_parameters_checks_frame_id_below() -> None:
    parameters = deepcopy(PARAMETERS)
    source = {
        "name": "many_peptides",
        "dir": "/tmp/single_frames",
        "adapter": "many_peptides",
        "environment_prompt": {"water": "Amber-ff14SB in OBC1 implicit water"},
        "frame_id_below": 3125,
    }
    parameters["data"]["sources"] = [source]
    _validate_parameters(parameters)

    source["frame_id_below"] = 0
    with pytest.raises(ValueError, match="frame_id_below"):
        _validate_parameters(parameters)

    source.update(frame_id_below=3125, adapter="n4_smiles")
    with pytest.raises(ValueError, match="frame_id_below"):
        _validate_parameters(parameters)

    parameters["data"]["frame_id_below"] = 3125
    with pytest.raises(ValueError, match="source-specific"):
        _validate_parameters(parameters)


def test_validate_parameters_checks_the_sequence_allowlist(tmp_path) -> None:
    parameters = deepcopy(PARAMETERS)
    allowlist = tmp_path / "subset.txt"
    allowlist.write_text("AA\n")
    source = {
        "name": "many_peptides",
        "dir": "/tmp/single_frames",
        "adapter": "many_peptides",
        "environment_prompt": {"water": "Amber-ff14SB in OBC1 implicit water"},
        "sequence_allowlist": str(allowlist),
    }
    parameters["data"]["sources"] = [source]
    _validate_parameters(parameters)

    source["sequence_allowlist"] = str(tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="sequence_allowlist"):
        _validate_parameters(parameters)

    source.update(sequence_allowlist=str(allowlist), adapter="n4_smiles")
    with pytest.raises(ValueError, match="sequence_allowlist"):
        _validate_parameters(parameters)


def test_validate_parameters_resolves_relative_allowlists_from_the_repository_root() -> None:
    """The system-subset configs name their allowlists relative to the repository."""
    parameters = deepcopy(PARAMETERS)
    parameters["data"]["sources"] = [
        {
            "name": "many_peptides",
            "dir": "/tmp/single_frames",
                "adapter": "many_peptides",
            "environment_prompt": {"water": "Amber-ff14SB in OBC1 implicit water"},
            "sequence_allowlist": "configs/system_subsets/seed43_1of8.txt",
        }
    ]

    _validate_parameters(parameters)


def test_missing_start_checkpoint_names_the_available_checkpoints(tmp_path) -> None:
    (tmp_path / "run_step_1000.ckpt").write_bytes(b"x")
    parameters = deepcopy(PARAMETERS)
    parameters["checkpoint"].update(dir=str(tmp_path), start_checkpoint=str(tmp_path / "missing.ckpt"))
    with pytest.raises(FileNotFoundError, match="available in .*run_step_1000.ckpt"):
        _validate_parameters(parameters)


def test_validate_parameters_accepts_one_data_parallel_source() -> None:
    parameters = deepcopy(PARAMETERS)
    parameters["data"]["sources"] *= 2
    with pytest.raises(ValueError, match="exactly one source"):
        _validate_parameters(parameters)

    parameters = deepcopy(PARAMETERS)
    parameters["distributed"] = {"zero_stage": 2}
    with pytest.raises(ValueError, match="zero_stage"):
        _validate_parameters(parameters)


# ── Path resolution ──────────────────────────────────────────────────


def test_resolve_parameters_joins_relative_paths_with_their_bases() -> None:
    parameters = deepcopy(PARAMETERS)
    parameters.update(data_base="/data/base", checkpoint_base="/checkpoint/base")
    parameters["checkpoint"].update(dir="checkpoints/pbg", start_checkpoint="checkpoints/pbg/run_step_1000.ckpt")

    resolved = _resolve_parameters(parameters)

    assert resolved["data"]["sources"][0]["dir"] == f"/data/base/{PARAMETERS['data']['sources'][0]['dir']}"
    parameters["data"]["sources"][0]["dir"] = "/absolute/shards"
    assert _resolve_parameters(parameters)["data"]["sources"][0]["dir"] == "/absolute/shards"
    assert resolved["checkpoint"]["dir"] == "/checkpoint/base/checkpoints/pbg"
    assert resolved["checkpoint"]["start_checkpoint"] == "/checkpoint/base/checkpoints/pbg/run_step_1000.ckpt"
    parameters["checkpoint"]["start_checkpoint"] = None
    assert _resolve_parameters(parameters)["checkpoint"]["start_checkpoint"] is None


def test_resume_latest_picks_the_highest_step_of_the_run(tmp_path) -> None:
    for name in ("run_step_10", "run_step_30", "run_step_20", "other_step_100", "run_step_bad", "bootstrap"):
        (tmp_path / f"{name}.ckpt").write_bytes(b"x")

    resolved = _resolve_parameters(
        {
            "run_name": "run",
            "checkpoint": {
                "dir": str(tmp_path),
                "start_checkpoint": str(tmp_path / "bootstrap.ckpt"),
                "resume_latest": True,
            },
        }
    )

    assert _latest_checkpoint_for_prefix(tmp_path, "run") == tmp_path / "run_step_30.ckpt"
    assert resolved["checkpoint"]["start_checkpoint"] == str(tmp_path / "run_step_30.ckpt")


def test_resume_latest_without_a_run_checkpoint(tmp_path) -> None:
    checkpoint = {"dir": str(tmp_path), "start_checkpoint": "parent.ckpt", "resume_latest": True}
    with pytest.raises(FileNotFoundError, match="resume_latest=true"):
        _resolve_parameters({"run_name": "run", "checkpoint": dict(checkpoint)})

    # resume_latest_if_available keeps the configured parent.
    parameters = {"checkpoint": dict(checkpoint, resume_latest_if_available=True)}
    assert _resolve_latest_checkpoint(parameters, "run") is None
    assert parameters["checkpoint"]["start_checkpoint"] == "parent.ckpt"


# ── Checkpoints ──────────────────────────────────────────────────────


def test_checkpoint_round_trip_restores_state_and_metadata(tiny_config, tmp_path) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    optimizer, _ = create_optimizer(model.qwen, config=OptimizerConfig(lr=1e-3, total_steps=0))
    state = init_train_state(model, optimizer, step=12)
    cursor = {
        "version": 1,
        "sources": {"many": {"fingerprint": "abc", "n_shards": 2, "shard_pass": [1, 0]}},
    }
    path = tmp_path / "run_step_12.ckpt"
    save_checkpoint(path, state, model=model, metadata={"loss_tokens_per_step": 73_800, "data_cursor": cursor})

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    stored = jax.tree.leaves(payload["params"])[0]
    assert isinstance(stored, np.ndarray)
    assert isinstance(jax.tree.leaves(payload["opt_state"])[0], np.ndarray)
    stored.fill(0.0)  # stored arrays are copies, not views of the live buffers
    assert float(np.abs(np.asarray(jax.tree.leaves(state.params)[0])).max()) != 0.0

    # A collate-only override keeps the stored optimizer state, which is not re-initialised.
    override = PBG(model.qwen, replace(model.config, max_samples_factor=0.02), model.vocab)
    spy = mock.Mock(wraps=optimizer)
    metadata: dict[str, object] = {}
    restored = load_checkpoint(path, spy, model_override=override, metadata_out=metadata)

    spy.init.assert_not_called()
    assert int(restored.step) == 12
    assert restored.params is override.qwen
    for left, right in zip(jax.tree.leaves(restored.opt_state), jax.tree.leaves(state.opt_state), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert metadata["loss_tokens_per_step"] == 73_800
    assert metadata["data_cursor"]["sources"]["many"]["shard_pass"] == [1, 0]


@pytest.mark.parametrize(
    ("saved", "configured"),
    [
        ({"muon_weight_decay_impl": "legacy", "adam_weight_decay_impl": "legacy_l2"}, {}),
        (
            {"muon_weight_decay_impl": "decoupled", "adam_weight_decay_impl": "decoupled", "muon_scaling_patches": False},
            {"muon_scaling_patches": True},
        ),
        (
            {"muon_weight_decay_impl": "decoupled", "adam_weight_decay_impl": "decoupled", "grad_clip_rel": 0.0},
            {"grad_clip_rel": 1.04},
        ),
    ],
    ids=["weight-decay", "scaling-patches", "adagc"],
)
def test_load_checkpoint_refuses_another_optimizer_policy(tiny_config, tmp_path, saved, configured) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    path = tmp_path / "resume.ckpt"
    state = PBGTrainState(params=model.qwen, opt_state={"m": jnp.zeros(())}, step=jnp.asarray(5))
    save_checkpoint(path, state, model=model, metadata={"parameters": {"optimizer": saved}})

    with pytest.raises(ValueError, match="policy differs"):
        load_checkpoint(path, mock.Mock(), optimizer_config=OptimizerConfig(**configured))


def test_checkpoint_retention_keeps_the_permanent_and_the_most_recent(tmp_path) -> None:
    for step in (5000, 10000, 15000, 20000, 25000, 30000, 40000, 95000, 100000):
        (tmp_path / f"run_step_{step}.ckpt").write_bytes(b"x")

    _prune_checkpoints(tmp_path, "run", 3, {5000, 10000, 20000, 30000, 95000, 100000})

    retained = sorted(int(path.stem.rsplit("_", 1)[1]) for path in tmp_path.glob("run_step_*.ckpt"))
    assert retained == [5000, 10000, 20000, 30000, 40000, 95000, 100000]


# ── Model construction from a start checkpoint ───────────────────────


def test_build_model_applies_collate_and_precision_overrides(tiny_config, tmp_path) -> None:
    model = PBG.init(tiny_config(), random.PRNGKey(0))
    checkpoint = tmp_path / "parent.ckpt"
    model.save(checkpoint)

    built = _build_model(
        {
            "seed": 0,
            "checkpoint": {"start_checkpoint": str(checkpoint)},
            "model": {
                "augment": True,
                "translate": True,
                "max_samples_factor": 0.02,
                "bin_tokens": 4096,
                "mixed_precision": True,
            },
        }
    )

    assert built.config == replace(
        model.config, augment=True, translate=True, max_samples_factor=0.02, mixed_precision=True
    )
    for left, right in zip(jax.tree.leaves(built.qwen), jax.tree.leaves(model.qwen), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


@pytest.mark.parametrize(
    ("key", "trained", "requested"),
    [("rope_theta", 1_000_000.0, 1000.0), ("level_pe_theta", 10_000.0, 10.0)],
)
def test_build_model_keeps_the_trained_positional_encoding(tiny_config, tmp_path, key, trained, requested) -> None:
    """Positional encoding is baked into the weights: restating it is a no-op, changing it fails."""
    checkpoint = tmp_path / "parent.ckpt"
    PBG.init(tiny_config(**{key: trained}), random.PRNGKey(0)).save(checkpoint)
    parameters = {"seed": 0, "checkpoint": {"start_checkpoint": str(checkpoint)}, "model": {key: trained}}

    assert getattr(_build_model(parameters).config, key) == trained

    parameters["model"][key] = requested
    with pytest.raises(ValueError, match="Positional encoding differs"):
        _build_model(parameters)


def test_build_model_rejects_a_coordinate_count_mismatch(tiny_config, tmp_path) -> None:
    checkpoint = tmp_path / "parent.ckpt"
    PBG.init(tiny_config(), random.PRNGKey(0)).save(checkpoint)

    with pytest.raises(ValueError, match="coordinate tokens, expected 65537"):
        _build_model({"seed": 0, "checkpoint": {"start_checkpoint": str(checkpoint)}, "model": {"bin_tokens": 65537}})


def test_resuming_an_optimizer_cannot_change_the_atom_traversal(tiny_config, tmp_path) -> None:
    checkpoint = tmp_path / "trained.ckpt"
    PBG.init(tiny_config(), random.PRNGKey(0)).save(checkpoint)

    with pytest.raises(ValueError, match="Cannot change atom traversal"):
        _build_model(
            {
                "checkpoint": {"start_checkpoint": str(checkpoint), "resume_optimizer": True},
                "model": {"atom_order": "prompt_canonical_v1"},
            }
        )


def test_atom_run_starts_from_a_prefix_parent_with_a_new_conditioner(tiny_config, tmp_path) -> None:
    parent = PBG.init(tiny_config(), random.PRNGKey(0))
    path = tmp_path / "parent.ckpt"
    parent.save(path)

    def build(conditioning: str, *, resume_optimizer: bool = False, start=path) -> PBG:
        return _build_model(
            {
                "seed": 7,
                "checkpoint": {"start_checkpoint": str(start), "resume_optimizer": resume_optimizer},
                "model": {"conditioning": conditioning},
            }
        )

    built = build("atom")
    assert built.config.conditioning == "atom"
    trunk = {key: value for key, value in built.qwen.items() if key != "atom_conditioner"}
    for left, right in zip(jax.tree.leaves(parent.qwen), jax.tree.leaves(trunk), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    expected = PBG.init(tiny_config(conditioning="atom"), random.PRNGKey(7)).qwen["atom_conditioner"]
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(built.qwen["atom_conditioner"]), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    # Submitted jobs set resume_optimizer even on their first start from a
    # parent; a parent has no optimizer state, so the start must still work.
    assert build("atom", resume_optimizer=True).config.conditioning == "atom"
    trained = tmp_path / "trained_prefix.ckpt"
    save_checkpoint(
        trained,
        PBGTrainState(params=parent.qwen, opt_state={"m": jnp.zeros(())}, step=jnp.asarray(5)),
        model=parent,
    )
    with pytest.raises(ValueError, match="prefix-conditioning optimizer state"):
        load_checkpoint(trained, optax.identity(), model_override=built)
    atom_path = tmp_path / "atom.ckpt"
    built.save(atom_path)
    with pytest.raises(ValueError, match="conditioning"):
        build("prefix", start=atom_path)


def test_robin_switches_override_a_parent_checkpoint(tiny_config, tmp_path) -> None:
    path = tmp_path / "parent.ckpt"
    PBG.init(tiny_config(), random.PRNGKey(0)).save(path)

    built = _build_model(
        {
            "seed": 43,
            "checkpoint": {"start_checkpoint": str(path)},
            "model": {
                "conditioning": "atom",
                "loss_reduction": "sample",
                "loss_terminator": False,
                "center_of_mass": True,
                # Ignored on a checkpoint start, as before: configs that set it
                # must keep their behaviour.
                "center": False,
            },
        }
    )

    assert built.config.conditioning == "atom"
    assert built.config.loss_reduction == "sample"
    assert built.config.loss_terminator is False
    assert built.config.center_of_mass is True
    assert built.config.center is True


# ── Train loop ───────────────────────────────────────────────────────


def test_fused_train_step_updates_the_state(tiny_config, peptide_frame, collate) -> None:
    model = PBG.init(tiny_config(max_seq_len=128), random.PRNGKey(0))
    optimizer, _ = create_optimizer(model.qwen, config=OptimizerConfig(lr=1e-3, total_steps=0))
    state = init_train_state(model, optimizer)
    train_step = _make_jit_train_step(optimizer, model)
    batch = collate(model, [[peptide_frame("AA", environment="vacuum", temperature=300.0)]])

    new_state, metrics = train_step(state, batch)

    assert int(jax.device_get(new_state.step)) == 1
    assert float(metrics["loss"]) > 0.0
    assert float(metrics["grad_norm"]) > 0.0
    assert float(metrics["loss_tokens"]) == float(np.sum(batch["loss_mask"]))
    assert not np.array_equal(np.asarray(new_state.params["final_norm"]), np.asarray(state.params["final_norm"]))


def test_optimizer_health_metrics_split_parameters_into_depth_thirds(tiny_config) -> None:
    model = PBG.init(tiny_config(num_layers=7), random.PRNGKey(0))

    grouped = _grouped_rms(model.qwen, model.config.num_layers)

    assert set(grouped) == {
        f"{depth}/{family}"
        for depth in ("bottom", "centre", "top")
        for family in ("muon_attention", "muon_mlp", "adam_norms")
    } | {"embed", "final_norm"}
    assert all(np.isfinite(float(value)) for value in grouped.values())


def test_train_logging_covers_the_cadence_the_last_step_and_the_final_window() -> None:
    kwargs = {"num_steps": 11309, "log_every": 120, "final_log_steps": 128, "final_log_every": 1}

    assert _should_log_train_step(1, **kwargs)
    assert _should_log_train_step(120, **kwargs)
    assert not _should_log_train_step(121, **kwargs)
    assert not _should_log_train_step(11180, **kwargs)
    assert _should_log_train_step(11182, **kwargs)
    assert _should_log_train_step(11308, **kwargs)
    assert _should_log_train_step(11309, **kwargs)
    sparse = {**kwargs, "final_log_steps": 0}
    assert not _should_log_train_step(11308, **sparse)
    assert _should_log_train_step(11309, **sparse)


def test_non_finite_loss_or_gradient_stops_training() -> None:
    _assert_finite_train_metrics(1, {"loss": 1.2, "grad_norm": 0.4})
    with pytest.raises(RuntimeError, match="non-finite train metrics at step 31060"):
        _assert_finite_train_metrics(31060, {"loss": float("nan"), "grad_norm": float("nan")})
    with pytest.raises(RuntimeError, match="grad_norm"):
        _assert_finite_train_metrics(3, {"loss": 1.0, "grad_norm": float("inf")})


def test_completed_run_exits_before_distributed_setup(tmp_path) -> None:
    (tmp_path / "pbg-run_step_12000.ckpt").touch()
    parameters = {
        "checkpoint": {"dir": str(tmp_path), "prefix": "pbg-run", "resume_latest": True},
        "distributed": {},
        "training": {"num_steps": 12000},
    }

    with mock.patch("pbg.models.pbg.train.bootstrap_distributed") as bootstrap_distributed:
        _run_training(parameters)

    bootstrap_distributed.assert_not_called()


# ── Weights & Biases ─────────────────────────────────────────────────


def test_explicit_wandb_id_initialises_or_resumes_the_run() -> None:
    run = mock.Mock(id="qwen3-ws-scaling")
    wandb = mock.Mock()
    wandb.init.return_value = run

    with mock.patch.dict("sys.modules", {"wandb": wandb}):
        initialized_run, enabled = _init_wandb(
            {
                "run_name": "qwen3-ws-scaling",
                "training": {"wandb": {"enabled": True, "project": "pbg", "run_id": "qwen3-ws-scaling"}},
            },
        )

    assert initialized_run is run
    assert enabled is True
    assert wandb.init.call_args.kwargs["id"] == "qwen3-ws-scaling"
    assert wandb.init.call_args.kwargs["resume"] == "allow"
    assert run.define_metric.call_args_list == [mock.call("train/tokens_seen"), mock.call("train/loss_tokens_seen")]


def test_auto_resume_continues_from_the_latest_checkpoint_if_available(tmp_path) -> None:
    parameters = deepcopy(PARAMETERS)
    parameters["checkpoint"].update(dir=str(tmp_path), prefix="run", auto_resume=True)
    with mock.patch("pbg.models.pbg.train._run_training") as run_training:
        run(parameters)
    resolved = run_training.call_args.args[0]["checkpoint"]
    assert resolved["resume_latest"] is True
    assert resolved["resume_latest_if_available"] is True
    assert resolved["resume_optimizer"] is True
    assert resolved["start_checkpoint"] is None

    (tmp_path / "run_step_20.ckpt").write_bytes(b"x")
    with mock.patch("pbg.models.pbg.train._run_training") as run_training:
        run(parameters)
    assert run_training.call_args.args[0]["checkpoint"]["start_checkpoint"] == str(tmp_path / "run_step_20.ckpt")


# ── Data-loader worker processes ─────────────────────────────────────


def test_loader_module_does_not_import_jax() -> None:
    """Spawned loader workers import this module before they hide the GPUs from JAX."""
    code = "import sys, pbg.models.pbg.loader; assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)"
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))


def test_loader_worker_hides_accelerators_from_jax(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "rocm")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "rocm")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    loader.hide_accelerator_devices()

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"
    assert os.environ["ROCR_VISIBLE_DEVICES"] == ""
    assert os.environ["HIP_VISIBLE_DEVICES"] == ""
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
