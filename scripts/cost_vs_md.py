#!/usr/bin/env python3
"""Sampling cost of PBG against molecular dynamics (Appendix "Sampling Cost against Molecular Dynamics").

Data of Figure "Sampling Efficiency against Molecular Dynamics" (fig:cost-vs-md-main) and of the appendix
figures fig:cost-vs-md-tica, fig:cost-vs-md-length-evaluations and fig:cost-vs-md-length-h100: E-W2, T-W2
and TICA-W2 of MD and of PBG 640M samples against energy evaluations and H100-hours, as means over the 89
Many Peptides test systems of Table 1 (YFPHAGYT excluded).

A PBG proposal costs one target-energy evaluation for its importance weight and an MD step one force
evaluation. A budget of n energy evaluations therefore scores
  PBG  the first n proposals of one rollout in generation order (10, 20, 50, ... up to the rollout size; the
       paper drew 10^6 proposals per system with the 15M drafter, as seeded chunks that are concatenated
       in sorted path order);
  MD   the frames saved during the first n steps: up to 2e7 steps from a dense-output run that starts at
       reference frame 0 (``short-md``; a frame every 1, 10, 100, 1,000 or 2,000 steps by decade), from
       1e8 steps every saved frame (one per 10 ps) of the 5 us test trajectory. The reference frames are a
       subsample of that trajectory, so long MD prefixes are optimistic.
Each budget gets the Table 1 estimators: rESS with the top 0.2% of the weights dropped; E-W2 and TICA-W2 on
1,024 frames resampled from those weights with the seeds 42, 43 and 44 (below 1,024 positive weights every
positive weight is resampled and E-W2 draws one energy per reference frame); and, for seed 42 only, the
exact T-W2 between 10^4 resampled frames and the 10^4 reference frames. PBG frames keep the log-likelihood of
the generated conformer; fully mirrored conformers are reflected (coordinates only). MD frames get uniform
weights.

Modes (heavy ones on a cluster node):
  throughput  plain OpenMM MD steps per second, one simulation per GPU, with the dataset settings
              (amber14-all + implicit/obc1, CutoffNonPeriodic 2 nm, no constraints, LangevinMiddle 310 K,
              0.3/ps, 1 fs), reading positions and velocities every 10,000 steps as when saving frames.
  short-md    dense-output MD of one system from reference frame 0 up to 2e7 steps (GPU).
  md-curve    MD curve of one system from a trajectory NPZ (test trajectory ``positions`` in nm, or a
              short-md output); long trajectories get OpenMM CPU energies for every scored frame.
  pbg-curve   PBG curve of one system from its rollout NPZs (scripts/sample_many_peptides.py output).
  summary     figure cells: per-system means over the resampling seeds, then means over systems for each
              length and for All. H100-hours per system are n / rate / 3600 with the PBG rate of the
              timing JSONs of scripts/benchmark_sampling_rate.py (median repeat; SD: fastest stride K=1-4,
              AR: K=0) and the MD steps per second of ``throughput``. The paper timed both on H100 SXM.

Example (one 8-core CPU node per system):
  python scripts/cost_vs_md.py pbg-curve --sequence AA --rollouts pbg_1e6/AA/chunk_*/AA__*__intended.npz \\
      --topology topology_dict.pkl --output curves/pbg_AA.json
  python scripts/cost_vs_md.py md-curve --sequence AA --trajectory trajectories/test/2AA/AA.npz \\
      --topology topology_dict.pkl --output curves/md_long_AA.json
  python scripts/cost_vs_md.py summary --curves curves/*.json --pbg-timing timing/*.json \\
      --md-throughput md_throughput.json --output summary
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K
from pbg.eval.many_peptides import (
    DEFAULT_REFERENCE_ROOT,
    discover_many_peptides_test_systems,
    load_many_peptides_test_reference,
)

BUDGETS = tuple(int(m * 10**e) for e in range(1, 8) for m in (1, 2, 5) if m * 10**e <= 10**7)
MD_SHORT_BUDGETS = BUDGETS + (20_000_000,)
MD_SHORT_PHASES = ((10_000, 1), (100_000, 10), (1_000_000, 100), (10_000_000, 1_000), (20_000_000, 2_000))
MD_LONG_BUDGETS = (20_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000,
                   1_000_000_000, 2_000_000_000, 5_000_000_000)
STEPS_PER_FRAME = 10_000  # 10 ps per saved frame at 1 fs
TEMPERATURE_K = 310.0
SEEDS = (42, 43, 44)
METRIC_COUNT = 1_024
TORUS_SEED = 42
TORUS_FRAMES = 10_000
RESS = "end_to_end_ress_drop_0_2pct"
ENERGY = "uniform_empirical_energy_w2_kbt_drop_0_2pct"
TORUS = "uniform_empirical_backbone_torus_w2_rad_drop_0_2pct_10k"
TICA = "uniform_empirical_tica2_w2_drop_0_2pct"


def openmm_system(sequence: str):
    """OpenMM system of the Many Peptides MD (amber14-all, OBC1, 2 nm cutoff, no constraints)."""
    import openmm.app as app
    import openmm.unit as unit

    from pbg.core.peptide import sequence_to_pdb_block

    pdb = app.PDBFile(io.StringIO(sequence_to_pdb_block(sequence)))
    return app.ForceField("amber14-all.xml", "implicit/obc1.xml").createSystem(
        pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=2.0 * unit.nanometer, constraints=None)


def score(samples, energy, smiles, weights, pool, p, reference, angles, topology, kbt, min_support) -> list[dict]:
    """E-W2 and TICA-W2 of the legacy Table 1 estimator for each seed and the exact T-W2 for the first one.

    ``weights`` (one per frame of ``samples``) drive E-W2 and TICA-W2; T-W2 draws 10^4 frames from ``pool``
    with probabilities ``p`` (None: uniform). This is the legacy estimator of pbg.eval.metrics.table1_metrics with
    the low-support rule of the curves: with ``min_support`` set, fewer than 1,024 positive weights are still scored.
    """
    from pbg.estimator import backbone_torsions, tica_projection, w2, w2_1d

    weights = np.asarray(weights, dtype=np.float64)
    positive = np.flatnonzero(np.isfinite(weights) & (weights > 0.0))
    n_reference = reference.length
    count = n_reference if positive.size < METRIC_COUNT else min(n_reference, positive.size)
    points = []
    for replicate, seed in enumerate(SEEDS):
        point = {"replicate": replicate, "seed": seed, "legacy_positive_weights": int(positive.size),
                 "legacy_status": "ok", "legacy_energy_count": count}
        if positive.size < (METRIC_COUNT if min_support is None else min_support) or n_reference < METRIC_COUNT:
            points.append({**point, "legacy_status": "omitted_low_support", "legacy_energy_count": 0})
            continue
        probabilities = weights[positive] / float(weights[positive].sum())
        rng = np.random.default_rng(seed)
        energy_draw = rng.choice(positive, size=count, replace=True, p=probabilities)
        geometry = rng.choice(positive, size=METRIC_COUNT, replace=True, p=probabilities)
        reference_draw = rng.choice(n_reference, size=METRIC_COUNT, replace=False)
        try:
            point[TICA] = w2(tica_projection(np.asarray(reference.samples, dtype=np.float64)[reference_draw], reference),
                             tica_projection(np.asarray(samples, dtype=np.float64)[geometry], reference))
        except ValueError as error:  # no valid TICA frame in a tiny prefix
            points.append({**point, "legacy_status": f"error: {error}", "legacy_energy_count": 0})
            continue
        point[ENERGY] = w2_1d(np.asarray(reference.energy, dtype=np.float64)[:count] / kbt, energy[energy_draw] / kbt)
        if replicate == 0:
            left = backbone_torsions(samples[np.random.default_rng(TORUS_SEED).choice(
                pool, size=TORUS_FRAMES, replace=True, p=p)], topology)
            point[TORUS] = w2(left, angles, period=2.0 * np.pi)
        points.append(point)
    return points


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("throughput", "short-md", "md-curve", "pbg-curve", "summary"))
    parser.add_argument("--output", type=Path, required=True, help="JSON file (summary: directory)")
    parser.add_argument("--sequence", nargs="*", help="system(s); one for short-md and the curves")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT,
                        help="trajectories_subsampled/ (default: $PBG_MANY_PEPTIDES_REFERENCE)")
    parser.add_argument("--topology", type=Path, default=os.environ.get("PBG_MANY_PEPTIDES_TOPOLOGY"),
                        help="pickled dict of MDTraj topologies by sequence (default: $PBG_MANY_PEPTIDES_TOPOLOGY)")
    parser.add_argument("--trajectory", type=Path, help="md-curve: trajectory or short-md NPZ")
    parser.add_argument("--rollouts", type=Path, nargs="+", help="pbg-curve: rollout NPZs of one system")
    parser.add_argument("--curves", type=Path, nargs="+", default=[], help="summary: curve JSONs")
    parser.add_argument("--pbg-timing", type=Path, nargs="+", default=[], help="summary: PBG timing JSONs")
    parser.add_argument("--md-throughput", type=Path, help="summary: throughput JSON")
    parser.add_argument("--pbg-label", default="PBG 640M")
    parser.add_argument("--sizes", nargs="+", default=["2AA", "4AA", "8AA"])
    parser.add_argument("--platform", default="CUDA", help="throughput and short-md")
    parser.add_argument("--precision", default="single", choices=("single", "mixed", "double"))
    parser.add_argument("--seconds", type=float, default=20.0, help="throughput: minimum timed seconds")
    parser.add_argument("--warmup", type=int, default=20_000, help="throughput: warm-up steps")
    parser.add_argument("--seed", type=int, help="short-md: Langevin seed (default: paper seed of the system)")
    parser.add_argument("--threads", default=os.environ.get("SLURM_CPUS_PER_TASK", "1"), help="OpenMM CPU threads")
    args = parser.parse_args()
    if args.mode not in ("throughput", "summary") and (not args.sequence or len(args.sequence) != 1):
        parser.error(f"{args.mode} needs exactly one --sequence")
    started = time.perf_counter()
    kbt = GAS_CONSTANT_KJ_MOL_K * TEMPERATURE_K
    systems = [s for s in discover_many_peptides_test_systems(args.reference_root, sizes=args.sizes)
               if s.sequence != "YFPHAGYT"] if args.mode in ("throughput", "short-md") else []

    if args.mode in ("throughput", "short-md"):
        import openmm
        import openmm.unit as unit

        platform = openmm.Platform.getPlatformByName(args.platform)
        properties = ({"Precision": args.precision} if args.platform in ("CUDA", "OpenCL", "HIP")
                      else {"Threads": str(args.threads)} if args.platform == "CPU" else {})
        rows = []
        for system_info in systems:
            if args.sequence and system_info.sequence not in args.sequence:
                continue
            with np.load(system_info.reference_path, allow_pickle=False) as data:
                positions = np.asarray(data["positions"][0], dtype=np.float64)
                velocities = np.asarray(data["velocities"][0], dtype=np.float64) if "velocities" in data.files else None
                stored_energy = float(np.asarray(data["energy"])[0])
            system = openmm_system(system_info.sequence)
            integrator = openmm.LangevinMiddleIntegrator(TEMPERATURE_K * unit.kelvin, 0.3 / unit.picosecond,
                                                         1.0 * unit.femtosecond)
            if args.mode == "short-md":
                # Paper seed: 20261110 + index of the system among the test systems of its length.
                same_length = [s.sequence for s in systems if s.size == system_info.size]
                integrator.setRandomNumberSeed(args.seed if args.seed is not None
                                               else 20261110 + same_length.index(system_info.sequence))
            context = openmm.Context(system, integrator, platform, properties)
            context.setPositions(positions * unit.nanometer)
            if velocities is None:
                context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin, 20261110)
            else:
                context.setVelocities(velocities * unit.nanometer / unit.picosecond)
            energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if args.mode == "short-md":
                steps, saved, energies, done = [], [], [], 0
                for end, every in MD_SHORT_PHASES:
                    while done < end:
                        state = context.getState(getPositions=True, getEnergy=True)
                        steps.append(done)
                        saved.append(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).astype(np.float32))
                        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                        integrator.step(every)
                        done += every
                args.output.parent.mkdir(parents=True, exist_ok=True)
                np.savez(args.output, sequence=np.asarray(system_info.sequence), step=np.asarray(steps, dtype=np.int64),
                         positions=np.asarray(saved), energy_kj_mol=np.asarray(energies, dtype=np.float64),
                         openmm_version=np.asarray(openmm.__version__), precision=np.asarray(args.precision),
                         wall_seconds=np.asarray(time.perf_counter() - started))
                return
            integrator.step(args.warmup)
            context.getState(getPositions=True, getVelocities=True)
            frames, timer = 0, time.perf_counter()
            while frames < 5 or time.perf_counter() - timer < args.seconds:
                integrator.step(STEPS_PER_FRAME)
                context.getState(getPositions=True, getVelocities=True)
                frames += 1
            elapsed = time.perf_counter() - timer
            rows.append(dict(sequence=system_info.sequence, size_label=system_info.size,
                             n_atoms=int(system.getNumParticles()), frame0_energy_kj_mol=energy,
                             energy_abs_diff_kj_mol=abs(energy - stored_energy), timed_steps=frames * STEPS_PER_FRAME,
                             timed_seconds=elapsed, steps_per_second=frames * STEPS_PER_FRAME / elapsed))
            print(json.dumps(rows[-1]), flush=True)
            del context, integrator
        write(args.output, dict(openmm_version=openmm.__version__, platform=args.platform,
                                precision=properties.get("Precision"), warmup_steps=args.warmup,
                                min_timed_seconds=args.seconds, systems=rows))
        return

    if args.mode == "summary":
        values = defaultdict(lambda: defaultdict(list))  # (model, sequence, budget) -> metric -> values
        for path in args.curves:
            curve = json.loads(path.read_text())
            model = "MD" if curve["kind"].startswith("md") else "PBG"
            for point in curve["points"]:
                n = int(point["energy_evaluations"])
                # MD: short runs up to 2e7 steps, the long trajectory from 1e8 steps.
                if (curve["kind"] == "md_short" and n > 20_000_000) or (curve["kind"] == "md_long" and n < 100_000_000):
                    continue
                cell = values[(model, curve["sequence"], n)]
                for metric in (RESS, ENERGY, TORUS, TICA):
                    if point.get(metric) is not None and np.isfinite(point[metric]):
                        cell[metric].append(float(point[metric]))
        rates = {}  # (decoding, sequence) -> samples or steps per second
        for path in args.pbg_timing:
            payload = json.loads(path.read_text())
            medians = {int(r["proposals"]): statistics.median(r["samples_per_second"]) for r in payload["results"]}
            sequence = payload.get("sequence") or payload["results"][0]["sequence"]
            if 0 in medians:
                rates[("AR", sequence)] = medians[0]
            if len([k for k in medians if k > 0]) == 4:
                rates[("SD", sequence)] = max(rate for k, rate in medians.items() if k > 0)
        if args.md_throughput:
            for row in json.loads(args.md_throughput.read_text())["systems"]:
                rates[("MD", row["sequence"])] = float(row["steps_per_second"])
        names = {RESS: "ress", ENERGY: "energy_w2_kbt", TORUS: "backbone_torus_w2_rad", TICA: "tica_w2"}
        system_rows = [dict(model=model, sequence=sequence, size_label=f"{len(sequence)}AA", energy_evaluations=n,
                            **{name: float(np.mean(cell[metric])) if cell[metric] else None for metric, name in names.items()})
                       for (model, sequence, n), cell in sorted(values.items())]
        cells = []
        for model, decodings in (("MD", ("MD",)), ("PBG", ("SD", "AR"))):
            for decoding in decodings:
                for cohort in [*args.sizes, "All"]:
                    for n in sorted({row["energy_evaluations"] for row in system_rows if row["model"] == model}):
                        group = [row for row in system_rows if row["model"] == model and row["energy_evaluations"] == n
                                 and cohort in (row["size_label"], "All")]
                        if not group:
                            continue
                        hours = [n / rates[(decoding, row["sequence"])] / 3600.0 for row in group
                                 if (decoding, row["sequence"]) in rates]
                        cell = dict(model="MD" if model == "MD" else f"{args.pbg_label}, {decoding}", decoding=decoding,
                                    size_label=cohort, energy_evaluations=n, evaluated_systems=len(group))
                        for name in names.values():
                            present = [row[name] for row in group if row[name] is not None]
                            cell[f"{name}_systems"] = len(present)
                            cell[name] = sum(present) / len(present) if present else None
                        cells.append(dict(cell, timed_systems=len(hours), h100_hours_per_system=sum(hours) / len(hours)
                                          if hours and len(hours) == len(group) else None))
        args.output.mkdir(parents=True, exist_ok=True)
        for name, rows in (("systems", system_rows), ("cells", cells)):
            with (args.output / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write(args.output / "summary.json", dict(rates={f"{k[0]}:{k[1]}": v for k, v in sorted(rates.items())},
                                                 systems=system_rows, cells=cells))
        print(f"wrote {args.output} ({len(system_rows)} system rows, {len(cells)} cells)")
        return

    # Curves of one system.
    if args.topology is None:
        parser.error("curves need --topology or $PBG_MANY_PEPTIDES_TOPOLOGY")
    from pbg.estimator import backbone_torsions

    sequence = args.sequence[0]
    size = f"{len(sequence)}AA"
    reference = load_many_peptides_test_reference(
        args.reference_root / "test" / size / f"{sequence}_subsampled.npz", sequence=sequence)
    with args.topology.open("rb") as handle:
        topology = pickle.load(handle)[sequence]
    angles = backbone_torsions(reference.samples, topology)
    points, extra = [], {}

    if args.mode == "md-curve":
        import openmm
        import openmm.unit as unit

        with np.load(args.trajectory, allow_pickle=False) as data:
            positions = np.asarray(data["positions"])
            steps = np.asarray(data["step"]) if "step" in data.files else None  # short-md output
            saved_energy = np.asarray(data["energy_kj_mol"], dtype=np.float64) if steps is not None else None
        kind = "md_long" if steps is None else "md_short"
        context = openmm.Context(openmm_system(sequence), openmm.VerletIntegrator(1.0),
                                 openmm.Platform.getPlatformByName("CPU"), {"Threads": str(args.threads)})
        energies: dict[int, float] = {}
        atomic_numbers = np.asarray(reference.samples, dtype=np.float64)[0, :, 0]
        for budget in MD_LONG_BUDGETS if steps is None else MD_SHORT_BUDGETS:
            if steps is None:
                picked = np.arange(budget // STEPS_PER_FRAME, dtype=np.int64)
                if picked.size > positions.shape[0]:
                    break
            else:
                spacing = next(every for end, every in MD_SHORT_PHASES if budget <= end)
                picked = np.flatnonzero((steps < budget) & (steps % spacing == 0))
                energies.update({int(frame): float(saved_energy[frame]) for frame in picked})
            for frame in picked:
                if int(frame) not in energies:
                    context.setPositions(positions[frame].astype(np.float64) * unit.nanometer)
                    energies[int(frame)] = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                        unit.kilojoule_per_mole)
            samples = np.concatenate((np.broadcast_to(atomic_numbers, (picked.size, atomic_numbers.size))[:, :, None],
                                      positions[picked].astype(np.float64)), axis=-1)
            energy = np.asarray([energies[int(frame)] for frame in picked])
            for point in score(samples, energy, reference.smiles, np.full(picked.size, 1.0 / picked.size), picked.size,
                               None, reference, angles, topology, kbt, None if steps is None else 1):
                points.append(dict(energy_evaluations=budget, md_frames_used=int(picked.size), **point))
            print(f"MD {sequence} steps={budget} frames={picked.size} {time.perf_counter() - started:.0f}s", flush=True)
        extra = dict(trajectory_frames=int(positions.shape[0]))
    else:
        from pbg.core.stereo import intended_stereo_mask
        from pbg.estimator import importance_weights, relative_ess
        from pbg.eval.metrics import Rollout

        kind = "pbg"
        chunks = [Rollout.load(path, reflect=True) for path in sorted(args.rollouts)]
        positions = np.concatenate([chunk.positions for chunk in chunks])
        energy = np.concatenate([chunk.energy for chunk in chunks])
        log_weight = -energy / (GAS_CONSTANT_KJ_MOL_K * chunks[0].temperature) - np.concatenate([c.logq for c in chunks])
        torus_valid = intended_stereo_mask(chunks[0].smiles, positions[:, :, 1:4]) & np.isfinite(log_weight)
        valid = torus_valid.copy()
        if all(chunk.frame_valid is not None for chunk in chunks):  # sampler mask, unless reflected here
            valid &= np.concatenate([chunk.frame_valid for chunk in chunks])
        in_prefix = np.zeros(len(energy), dtype=bool)
        for budget in (b for b in BUDGETS if b <= len(energy)):
            in_prefix[:budget] = True
            legacy_weights = importance_weights(np.where(valid & in_prefix, log_weight, np.nan), drop_top_percent=0.2)
            ress = relative_ess(legacy_weights, n_proposals=budget)
            weights = importance_weights(np.where(torus_valid & in_prefix, log_weight, np.nan), drop_top_percent=0.2)
            support = np.flatnonzero(weights > 0.0)
            for point in score(positions, energy, chunks[0].smiles, legacy_weights, support,
                               weights[support] / float(weights[support].sum()), reference, angles, topology,
                               GAS_CONSTANT_KJ_MOL_K * chunks[0].temperature, 1):
                points.append(dict(energy_evaluations=budget, **{RESS: ress}, n_valid=int(np.sum(valid & in_prefix)),
                                   **point))
            print(f"PBG {sequence} n={budget} {time.perf_counter() - started:.0f}s", flush=True)
        extra = dict(generated_proposals=int(len(energy)), n_reflected=sum(chunk.n_reflected for chunk in chunks))
    write(args.output, dict(sequence=sequence, size_label=size, kind=kind,
                            inputs=[str(path) for path in ([args.trajectory] if kind != "pbg" else sorted(args.rollouts))],
                            reference_frames=int(reference.length), seconds=time.perf_counter() - started, **extra,
                            points=points))


if __name__ == "__main__":
    main()
