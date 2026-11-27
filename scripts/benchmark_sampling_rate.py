#!/usr/bin/env python3
"""Time autoregressive (AR) and speculative-decoding (SD) sampling (Section 4.2, Appendix).

For each peptide the script times the production AR sampler (``K=0``) and,
with ``--draft-checkpoint``, junior speculative decoding with a small PBG
drafter at each proposal stride ``K``. Protocol of the paper: batch 512, one
warm-up call of 512 samples, then five timed calls of 2048 samples each, JAX
key 20260924. SD timings exclude the reported compile time; both include host
transfer, coordinate reconstruction and log-likelihoods. One JSON per peptide
records the per-repeat rates and the SD acceptance counts used by the rate model.

Run on one GPU (the paper used an H100 NVL), e.g.
``srun -p gpu --gres=gpu:1 -t 04:00:00 -c 8 --mem 64G python scripts/benchmark_sampling_rate.py ...``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import numpy as np

from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.tokeniser import VocabSchema
from pbg.models.pbg.model import PBG
from pbg.models.pbg.sample import sample_molecular_conformers
from pbg.models.pbg.system import MolecularSystem, checkpoint_environment_prompt
from pbg.models.speculation.sample import sample_junior_conformers

BATCH_SIZE = 512
WARMUP_SAMPLES = 512
TIMED_SAMPLES = 2048
REPEATS = 5
SEED = 20260924


def time_stride(target: PBG, draft: PBG | None, system: MolecularSystem, n_atoms: int, proposals: int) -> dict:
    """Time AR (``proposals == 0``) or SD with stride ``proposals``."""
    key = jax.random.key(SEED)
    timings, accepted, proposed = [], 0, 0
    for repeat in range(REPEATS + 1):
        count = WARMUP_SAMPLES if repeat == 0 else TIMED_SAMPLES
        key, subkey = jax.random.split(key)
        jax.block_until_ready(subkey)
        start = time.perf_counter()
        if proposals == 0:
            batch = sample_molecular_conformers(
                target, system, subkey, num_samples=count, batch_size=BATCH_SIZE, proposal_temperature=1.0
            )
            elapsed = time.perf_counter() - start
        else:
            batch, metrics = sample_junior_conformers(
                draft, target, system, subkey, num_samples=count, batch_size=BATCH_SIZE,
                num_proposals=proposals, temperature=1.0,
            )
            elapsed = time.perf_counter() - start - metrics.compile_seconds
            if repeat:
                accepted += metrics.accepted
                proposed += metrics.proposals
        if (
            batch.samples.shape != (count, n_atoms, 4)
            or not np.isfinite(batch.samples).all()
            or not np.isfinite(batch.logq).all()
        ):
            raise ValueError(f"invalid generated conformers: {system.id} K={proposals}")
        if repeat:
            timings.append(elapsed)
            print(f"{system.id} K={proposals} repeat={repeat}: {count / elapsed:.3f} samples/s", flush=True)
    rates = [TIMED_SAMPLES / seconds for seconds in timings]
    return {
        "proposals": proposals,
        "method": "ar" if proposals == 0 else "speculative",
        "seconds": timings,
        "samples_per_second": rates,
        "median_samples_per_second": statistics.median(rates),
        "accepted": accepted,
        "proposed": proposed,
        "acceptance": accepted / proposed if proposed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="target PBG checkpoint")
    parser.add_argument("--draft-checkpoint", type=Path, help="drafter (the paper used PBG 15M)")
    parser.add_argument("--proposals", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--sequence", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    target = PBG.load(args.checkpoint)
    environment = checkpoint_environment_prompt(args.checkpoint)
    draft = None
    strides = [0]
    if args.draft_checkpoint is not None:
        draft = PBG.load(args.draft_checkpoint)
        if json.dumps(draft.vocab, sort_keys=True) != json.dumps(target.vocab, sort_keys=True):
            raise ValueError("draft and target vocabularies differ")
        if tuple(VocabSchema.of(draft.vocab).coord_ids) != tuple(VocabSchema.of(target.vocab).coord_ids):
            raise ValueError("coordinate token IDs differ")
        if checkpoint_environment_prompt(args.draft_checkpoint) != environment:
            raise ValueError("draft and target use different conditioning prompts")
        strides += sorted(set(args.proposals))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0].device_kind
    for sequence in args.sequence:
        system = MolecularSystem.from_peptide(sequence, environment=environment, temperature_K=310.0)
        n_atoms = int(atom_numbers_from_smiles(system.mapped_smiles).size)
        results = [time_stride(target, draft, system, n_atoms, k) for k in strides]
        payload = {
            "sequence": sequence,
            "atom_count": n_atoms,
            "target_checkpoint": str(args.checkpoint),
            "draft_checkpoint": None if draft is None else str(args.draft_checkpoint),
            "device": device,
            "protocol": "b512_warm512_repeat5_n2048",
            "results": results,
        }
        (args.output_dir / f"{sequence}.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
