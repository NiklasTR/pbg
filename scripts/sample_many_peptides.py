#!/usr/bin/env python3
"""Sample the Many Peptides test systems with a PBG or ROBIN checkpoint (Table 1).

For every test system the script draws ``--num-samples`` proposals at logit
temperature 1, reconstructs bin centres, reflects fully mirrored stereoisomers
(PBG only; the stored log-likelihood stays that of the generated conformer),
scores Amber ff14SB/OBC1 energies with OpenMM and writes one rollout NPZ per
system: ``{SEQ}__{environment}__{T}K__intended.npz``.

The paper used 10^4 samples at 310 K and the sampling seeds 2026111101,
2026111102 and 2026111103. The JAX key of a system hashes
``f"{seed_path}:{sequence}:{seed}"``, where ``seed_path`` defaults to the
resolved checkpoint path, and samples also depend on the batch size. To
reproduce a stored rollout exactly, pass the original checkpoint path as
``--seed-path`` and the batch size recorded in its batch-tuning JSON.

Run on a GPU node, e.g. ``srun -p gpu --gres=gpu:1 -t 04:00:00 -c 8 --mem 64G``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from pbg.core.ensemble import Ensemble, save_ensemble
from pbg.core.frame import DataFrame
from pbg.eval.many_peptides import (
    DEFAULT_REFERENCE_ROOT,
    discover_many_peptides_test_systems,
)
from pbg.models.pbg.model import PBG
from pbg.models.pbg.rollout import (
    energy_frames,
    require_gpu_backend,
    sample_ensembles,
    tune_sampling_batch_size,
)
from pbg.models.pbg.system import (
    MolecularSystem,
    SamplingPolicy,
    checkpoint_environment_prompt,
)

PAPER_SEEDS = (2026111101, 2026111102, 2026111103)


def key_seed(seed_path: str, sequence: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed_path}:{sequence}:{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def frames_for(system: MolecularSystem, samples: np.ndarray) -> tuple[DataFrame, ...]:
    return tuple(
        DataFrame(
            id=system.id,
            frame_id=index,
            smiles=system.mapped_smiles,
            sequence=system.sequence,
            helm=system.helm,
            environment=system.environment,
            temperature=system.temperature_K,
            conformer=sample,
        )
        for index, sample in enumerate(samples)
    )


def sample_pbg(
    model: PBG,
    checkpoint: Path,
    output_dir: Path,
    sequence: str,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Sample one system with a PBG checkpoint and write its rollout(s)."""
    policy = SamplingPolicy.from_model(model)
    environment_prompt = args.environment_prompt
    if environment_prompt is None and policy.generation != "joint":
        environment_prompt = checkpoint_environment_prompt(checkpoint)
    system = MolecularSystem.from_peptide(
        sequence, environment=environment_prompt or "water", temperature_K=args.temperature_k
    )
    # PBG-Joint is prompted with the implicit-water protocol of Many Peptides.
    system = policy.system(
        system,
        target_id="ff14sb-obc1-2nm" if policy.generation == "joint" and environment_prompt is None else None,
    )
    seed_path = args.seed_path or str(checkpoint)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size, trials = tune_sampling_batch_size(
            model,
            system,
            jax.random.key(key_seed(seed_path, sequence, args.seed) ^ 0xA5A5A5A5),
            candidates=tuple(args.batch_size_candidates),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{sequence}__water__{args.temperature_k:g}K__batch-tuning.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "sequence": sequence,
                    "environment_prompt": environment_prompt,
                    "temperature_K": args.temperature_k,
                    "selected_batch_size": batch_size,
                    "trials": [
                        {"batch_size": t.batch_size, "seconds": t.seconds, "samples_per_second": t.samples_per_second}
                        for t in trials
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def score_energy(system: MolecularSystem, samples: np.ndarray) -> np.ndarray:
        return energy_frames(
            frames_for(system, samples),
            sequence=sequence,
            environment="water",
            platform_name=args.energy_platform,
        )

    paths = sample_ensembles(
        model,
        system,
        jax.random.key(key_seed(seed_path, sequence, args.seed)),
        energy_scorer=score_energy,
        num_samples=args.num_samples,
        output_dir=output_dir,
        batch_size=batch_size,
        generator_checkpoint=str(checkpoint),
        repair_all_mirrored=True,
    )
    return tuple(str(path) for path in paths)


def sample_robin(robin, checkpoint: Path, output_dir: Path, sequence: str, args: argparse.Namespace) -> str:
    """Sample one system with the converted ROBIN checkpoint (no reflection)."""
    from pbg.models.robin import sample_robin_conformers

    system = MolecularSystem.from_peptide(sequence, environment="water", temperature_K=args.temperature_k)
    template = DataFrame.from_sequence(sequence)
    seed = key_seed(args.seed_path or str(checkpoint), sequence, args.seed)
    batch = sample_robin_conformers(
        robin, system, jax.random.key(seed),
        num_samples=args.num_samples, batch_size=args.batch_size or 512, temperature=1.0,
    )
    frames = tuple(
        DataFrame(id=sequence, frame_id=index, smiles=template.smiles, sequence=sequence,
                  environment="water", temperature=args.temperature_k, conformer=conformer)
        for index, conformer in enumerate(batch.samples)
    )
    energy = energy_frames(frames, sequence=sequence, environment="water", platform_name=args.energy_platform)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sequence}__water__{args.temperature_k:g}K__intended.npz"
    save_ensemble(
        Ensemble(samples=batch.samples, smiles=template.smiles, id=sequence, sequence=sequence,
                 environment="water", temperature=args.temperature_k, energy=energy, logq=batch.logq,
                 source="robin_sample", generator_checkpoint=str(checkpoint)),
        path,
    )
    path.with_suffix(".json").write_text(json.dumps({
        "sampler": "pbg.models.robin.sample_robin_conformers", "sampling_seed": args.seed,
        "key_seed": seed, "temperature": 1.0, "batch_size": args.batch_size or 512, "reflection": "none",
    }, indent=2) + "\n")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robin", action="store_true", help="checkpoint is a converted ROBIN JAX pickle")
    parser.add_argument("--sequence", nargs="*", help="default: all test systems of --sizes")
    parser.add_argument("--sizes", nargs="+", default=["2AA", "4AA", "8AA"])
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--temperature-k", type=float, default=310.0)
    parser.add_argument("--seed", type=int, default=PAPER_SEEDS[0])
    parser.add_argument("--seed-path", help="string hashed into the JAX key instead of the checkpoint path")
    parser.add_argument("--batch-size", type=int, help="default: tuned by timing (PBG) or 512 (ROBIN)")
    parser.add_argument("--batch-size-candidates", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--environment-prompt", help="default: the checkpoint's training prompt")
    parser.add_argument("--energy-platform", default="CUDA")
    args = parser.parse_args()

    require_gpu_backend()
    checkpoint = args.checkpoint.resolve()
    sequences = args.sequence or [
        system.sequence
        for system in discover_many_peptides_test_systems(args.reference_root, sizes=args.sizes)
    ]
    if args.robin:
        from pbg.models.robin import Robin

        robin = Robin.load(checkpoint)
    else:
        model = PBG.load(checkpoint)
    for sequence in sequences:
        if next(args.output_dir.glob(f"{sequence}__*__intended.npz"), None) is not None:
            print(f"skip {sequence}: rollout present", flush=True)
            continue
        if args.robin:
            paths = (sample_robin(robin, checkpoint, args.output_dir, sequence, args),)
        else:
            paths = sample_pbg(model, checkpoint, args.output_dir, sequence, args)
        print(f"{sequence}: {', '.join(paths)}", flush=True)


if __name__ == "__main__":
    main()
