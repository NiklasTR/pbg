#!/usr/bin/env python3
"""Create the untrained parent checkpoints that every PBG training run starts from.

``core``: one parent per model size and coordinate vocabulary. All vocabulary
variants of a size share the transformer initialization and the embeddings of
their common text tokens, so bin count is the only difference between them.
CompleteP sizes (47M-640M) set the depth- and width-scaled residual and
read-out factors of ``completep_width_rules``.

``joint``: convert a ``core_bin4096`` parent into the ``joint_bin4096``
parent of PBG-Joint. Seven unused text-token rows receive force-field words
and fresh embeddings; every other weight is copied bitwise.

Run on a CPU Slurm node, e.g.
``srun -p cpu -t 00:30:00 -c 4 --mem 32G python scripts/create_parent_checkpoints.py core --models 85m``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pbg.checkpoint import load_pickle
from pbg.core.tokeniser import JOINT_TOKEN_REPLACEMENTS, joint_vocab, make_tokeniser
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.model import PBG

ARCHITECTURES = {
    "8m": dict(dim=384, num_layers=4, num_heads=6, num_kv_heads=3, head_dim=128, intermediate_size=1024),
    "15m": dict(dim=512, num_layers=4, num_heads=8, num_kv_heads=4, head_dim=128, intermediate_size=1408),
    "30m": dict(dim=512, num_layers=8, num_heads=8, num_kv_heads=4, head_dim=128, intermediate_size=1408),
    "49m": dict(dim=640, num_layers=8, num_heads=10, num_kv_heads=5, head_dim=128, intermediate_size=1920),
    "85m": dict(dim=768, num_layers=10, num_heads=12, num_kv_heads=6, head_dim=128, intermediate_size=2176),
    "47m": dict(dim=512, num_layers=16, num_heads=4, num_kv_heads=2, head_dim=128, intermediate_size=1408),
    "160m": dict(dim=768, num_layers=24, num_heads=6, num_kv_heads=3, head_dim=128, intermediate_size=2112),
    "320m": dict(dim=1024, num_layers=28, num_heads=8, num_kv_heads=4, head_dim=128, intermediate_size=2816),
    "640m": dict(dim=1536, num_layers=24, num_heads=12, num_kv_heads=6, head_dim=128, intermediate_size=4224),
}
COMPLETEP_MODEL_NAMES = ("47m", "160m", "320m", "640m")
COMPLETEP_RESIDUAL_BASE_LAYERS = 16
COMPLETEP_BASE_DIM = 512
COMPLETEP_BASE_LR = 0.001
COMPLETEP_BASE_WEIGHT_DECAY = 0.1
# Bin count and bin width (pm) of each coordinate vocabulary; the box edge is
# (bins - 1) * width, 3.2768 nm for the default 4096 bins.
VOCAB_VARIANTS = {
    "core_bin256": (256, 12.8),
    "core_bin512": (512, 6.4),
    "core_bin704": (704, 3276.8 / 704),
    "core_bin768": (768, 3276.8 / 768),
    "core_bin832": (832, 3276.8 / 832),
    "core_bin896": (896, 3276.8 / 896),
    "core_bin960": (960, 3276.8 / 960),
    "core_bin1024": (1024, 3.2),
    "core_bin2048": (2048, 1.6),
    "core_bin4096": (4096, 0.8),
    "core_bin8192": (8192, 0.4),
    "core_bin12288": (12288, 3276.8 / 12288),
    "core_bin16384": (16384, 0.2),
    "core_bin32768": (32768, 0.1),
    "core_bin65537": (65537, 0.05),
    "core_bin131074": (131074, 3276.8 / 131074),
}


def completep_width_rules(dim: int, num_layers: int) -> dict[str, float]:
    """Kimi-Muon CompleteP width and depth factors relative to the 47M proxy."""
    m_d = dim / COMPLETEP_BASE_DIM
    sqrt_m_d = math.sqrt(m_d)
    return {
        "m_d": m_d,
        "m_L": num_layers / COMPLETEP_RESIDUAL_BASE_LAYERS,
        "residual_scale": COMPLETEP_RESIDUAL_BASE_LAYERS / num_layers,
        "unembedding_scale": COMPLETEP_BASE_DIM / dim,
        "lr": COMPLETEP_BASE_LR / sqrt_m_d,
        "adamw_lr_ratio": sqrt_m_d,
        "weight_decay": COMPLETEP_BASE_WEIGHT_DECAY * sqrt_m_d,
        "adam_weight_decay": COMPLETEP_BASE_WEIGHT_DECAY,
    }


def vocab_sha256(vocab: dict[str, Any]) -> str:
    """SHA-256 of the vocabulary serialized as the original JSON asset."""
    return hashlib.sha256((json.dumps(vocab, indent=2) + "\n").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def non_embedding_params(architecture: dict[str, int]) -> int:
    dim, layers = architecture["dim"], architecture["num_layers"]
    heads, kv_heads = architecture["num_heads"], architecture["num_kv_heads"]
    head_dim, intermediate = architecture["head_dim"], architecture["intermediate_size"]
    attention = (
        dim * heads * head_dim + 2 * dim * kv_heads * head_dim + heads * head_dim * dim + 2 * head_dim
    )
    return layers * (2 * dim + attention + 3 * dim * intermediate) + dim


def write_checkpoint(path: Path, model: PBG, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "params": jax.device_get(model.qwen),
        "config": asdict(model.config),
        "vocab": copy.deepcopy(model.vocab),
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def create_core_parents(
    output_dir: Path, *, model_names: list[str], seed: int, vocab_names: list[str]
) -> list[Path]:
    unknown = sorted(set(vocab_names) - set(VOCAB_VARIANTS))
    if unknown:
        raise ValueError(f"unknown vocab names: {unknown}")
    # The 4096-bin parent is the canonical source of the shared text embeddings.
    init_variants = {name: VOCAB_VARIANTS[name] for name in vocab_names}
    init_variants.setdefault("core_bin4096", VOCAB_VARIANTS["core_bin4096"])
    outputs: list[Path] = []
    for model_name in model_names:
        architecture = ARCHITECTURES[model_name]
        key = jax.random.PRNGKey(seed)
        completep = model_name in COMPLETEP_MODEL_NAMES
        rules = completep_width_rules(architecture["dim"], architecture["num_layers"])
        models: dict[str, PBG] = {}
        for vocab_name, (bin_tokens, bin_width_pm) in init_variants.items():
            config = PBGConfig(
                **architecture,
                vocab_name=vocab_name,
                bin_tokens=bin_tokens,
                bin_width_pm=bin_width_pm,
                max_seq_len=1028,
                residual_scale=rules["residual_scale"] if completep else 1.0,
                unembedding_scale=rules["unembedding_scale"] if completep else 1.0,
                rope_theta=1000.0,
                level_pe_theta=10.0,
                mixed_precision=True,
                augment=False,
                translate=False,
            )
            model = PBG.init(config, key)
            embed = model.qwen["embed"]
            for start, end in model.vocab["meta"].get("padding_ranges", []):
                embed = embed.at[int(start) : int(end)].set(
                    jnp.zeros((int(end) - int(start), config.dim), dtype=embed.dtype)
                )
            models[vocab_name] = PBG(qwen=dict(model.qwen, embed=embed), config=config, vocab=model.vocab)

        # Text-token IDs after the coordinate block move with the bin count:
        # copy every shared token row by name from the canonical parent.
        canonical = models["core_bin4096"]
        for vocab_name, model in models.items():
            embed = model.qwen["embed"]
            for section in ("core_special_tokens", "core_tokens", "dipper_added_tokens"):
                for token, target_id in model.vocab[section].items():
                    source_id = canonical.vocab[section][token]
                    embed = embed.at[int(target_id)].set(canonical.qwen["embed"][int(source_id)])
            models[vocab_name] = PBG(qwen=dict(model.qwen, embed=embed), config=model.config, vocab=model.vocab)

        for vocab_name in vocab_names:
            model = models[vocab_name]
            output = output_dir / f"dipper_{model_name}_{vocab_name}_parent_seed{seed}.ckpt"
            write_checkpoint(output, model, {
                "initialization": "jax_random_normal_dipper_init",
                "seed": seed,
                "model_name": model_name,
                "architecture": architecture,
                "non_embedding_params": non_embedding_params(architecture),
                "shared_transformer_initialization_across_vocabularies": True,
                "shared_semantic_embedding_initialization_across_vocabularies": True,
                "shared_coordinate_prefix_initialization_across_vocabularies": True,
                "padding_initialization": "zeros",
                "residual_scale": float(model.config.residual_scale),
                "unembedding_scale": float(model.config.unembedding_scale),
                "residual_base_layers": COMPLETEP_RESIDUAL_BASE_LAYERS if completep else None,
                "base_dim": COMPLETEP_BASE_DIM if completep else None,
                "vocab_name": vocab_name,
                "vocab_sha256": vocab_sha256(model.vocab),
                "tie_token_embeddings": True,
                "optimizer_state": "omitted",
            })
            outputs.append(output)
            print(f"wrote {output}: embedding={tuple(model.qwen['embed'].shape)}", flush=True)
    return outputs


def create_joint_parent(source_path: Path, output_path: Path, *, seed: int = 42) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output_path}")
    payload = load_pickle(source_path)
    if payload.get("step") not in (None, 0) or "opt_state" in payload:
        raise ValueError("source must be an untrained parent without optimizer state")
    source = PBG.load_payload(payload)
    if source.config.vocab_name != "core_bin4096" or "lm_head" in source.qwen:
        raise ValueError("expected a core_bin4096 parent with tied embeddings")
    vocab = joint_vocab()
    if vocab["coordinate_tokens"] != source.vocab["coordinate_tokens"]:
        raise ValueError("coordinate token IDs changed")
    old_params = jax.device_get(source.qwen)
    embed = np.array(old_params["embed"], copy=True)
    changed = []
    for token_id, _old, _new in JOINT_TOKEN_REPLACEMENTS:
        # Same distribution as the embedding initialization; independent rows
        # drop the meaning of the displaced solvent tokens.
        key = jax.random.fold_in(jax.random.PRNGKey(seed), token_id)
        embed[token_id] = np.asarray(jax.random.normal(key, (embed.shape[1],)), dtype=embed.dtype) * 0.02
        changed.append(token_id)
    params = dict(old_params, embed=embed)
    config = replace(source.config, vocab_name="joint_bin4096", prompt_format="compact_kelvin_v1")
    metadata = {
        "initialization": "same_shape_parent_text_vocab_remap",
        "seed": seed,
        "model_name": payload.get("metadata", {}).get("model_name", "85m"),
        "source_checkpoint": str(source_path),
        "source_sha256": file_sha256(source_path),
        "vocab_name": "joint_bin4096",
        "vocab_version": 5,
        "vocab_sha256": vocab_sha256(vocab),
        "changed_embedding_rows": changed,
        "new_row_initialization": "jax.random.normal(fold_in(PRNGKey(seed), token_id), (dim,)) * 0.02",
        "unchanged_rows": "all shared text, delimiter, BOS/PAD, MASK, and coordinate rows",
        "transformer_parameters": "bitwise preserved from source parent",
        "tie_token_embeddings": True,
        "optimizer_state": "omitted",
        "parent_step": 0,
        "source_metadata": payload.get("metadata", {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".ckpt.partial")
    with temporary.open("wb") as handle:
        pickle.dump(
            dict(format_version=2, params=params, config=asdict(config), vocab=vocab, metadata=metadata),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    # Check the written file through the production loader before publishing it.
    loaded = PBG.load(temporary)
    loaded_params = jax.device_get(loaded.qwen)
    unchanged = np.ones(embed.shape[0], dtype=bool)
    unchanged[changed] = False
    np.testing.assert_array_equal(loaded_params["embed"][unchanged], old_params["embed"][unchanged])
    for old_array, new_array in zip(
        jax.tree.leaves({k: v for k, v in old_params.items() if k != "embed"}),
        jax.tree.leaves({k: v for k, v in loaded_params.items() if k != "embed"}),
        strict=True,
    ):
        np.testing.assert_array_equal(old_array, new_array)
    tokenizer = make_tokeniser(loaded.vocab, prompt_format=loaded.config.prompt_format)
    prompt = tokenizer.encode_prompt(
        "PEPTIDE1{A.G}$$$$", environment="64 water, ff14sb-tip3p-rf2nm", temperature_K=373.421
    )
    if any(vocab["core_tokens"][token] not in prompt for token in ("ff14sb", "tip3p", "rf", "nm")):
        raise ValueError("new tokens absent from encoded prompt")
    temporary.replace(output_path)
    return {
        "checkpoint": str(output_path),
        "checkpoint_sha256": file_sha256(output_path),
        "changed_embedding_rows": changed,
        "preserved_embedding_rows": int(unchanged.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    core = sub.add_parser("core", help="core_bin* parents for the listed sizes")
    core.add_argument("--output-dir", type=Path, required=True)
    core.add_argument("--seed", type=int, default=42)
    core.add_argument("--models", nargs="+", choices=tuple(ARCHITECTURES), required=True)
    core.add_argument("--vocabs", nargs="+", choices=tuple(VOCAB_VARIANTS), default=["core_bin4096"])
    joint = sub.add_parser("joint", help="joint_bin4096 parent from a core_bin4096 parent")
    joint.add_argument("--source", type=Path, required=True)
    joint.add_argument("--output", type=Path, required=True)
    joint.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "core":
        create_core_parents(args.output_dir, model_names=args.models, seed=args.seed, vocab_names=args.vocabs)
    else:
        print(json.dumps(create_joint_parent(args.source, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
