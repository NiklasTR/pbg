#!/usr/bin/env python3
"""Score Many Peptides rollouts with the paper's metrics (Table 1, starred table, D-proline and box-volume appendices).

Pass one rollout directory per sampling seed, as written by scripts/sample_many_peptides.py
(``{SEQ}__*__intended.npz``). The script writes ``OUTPUT/seed<k>/<SEQ>.json`` for each system and
seed (k counts the directories from 1) and ``OUTPUT/summary.csv`` with the paper cells (see
``pbg.eval.metrics.summarize``). A rerun keeps the metric groups that a result file already holds,
so a Slurm array can split the systems with ``--shard`` and ``--num-shards``. A run without rollout
directories only rewrites the summary.

Metric groups (``--metrics``):
  table1      rESS, E-W2, TICA-W2 (top 0.2% of weights dropped) and the exact 10^4-frame T-W2.
              T-W2 solves a 10^4 x 10^4 transport problem (about 8 GB of memory).
  starred     rESS*, E-W2* and the sliced T-W2* with every weight kept.
  box_volume  the starred metrics with box-volume corrected weights (appendix).
``--floor`` scores the perfect-model floor of the box-volume bias on the reference MD instead
(``OUTPUT/floor/<SEQ>.json``; no rollouts needed).

PBG rollouts get the total-reflection correction (coordinates only); ``--robin`` turns it off, as in
the paper's ROBIN row. YFPHAGYT is left out unless named with ``--sequence``. ``--reference-stereo``
scores against the stereochemistry of the reference MD frames (D-proline appendix, with
``--sequence YFPHAGYT``); give it its own ``--output``.

All three groups take about half a minute per rollout. Run it on a CPU node, for example:
  srun -p cpu -t 04:00:00 -c 2 --mem 16G python scripts/evaluate_many_peptides.py \\
      rollouts/seed1 rollouts/seed2 rollouts/seed3 --output results/pbg --topology topology_dict.pkl
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
from pathlib import Path

from pbg.eval.many_peptides import (
    DEFAULT_REFERENCE_ROOT,
    discover_many_peptides_test_systems,
    load_many_peptides_test_reference,
)
from pbg.eval.metrics import (
    EXCLUDED_SYSTEMS,
    Rollout,
    box_volume_floor,
    box_volume_metrics,
    box_volume_rotations,
    reference_stereo_smiles,
    starred_metrics,
    summarize,
    table1_metrics,
)

GROUPS = ("table1", "starred", "box_volume")
SIZES = ("2AA", "4AA", "8AA")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path, nargs="*", help="one rollout directory per sampling seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT,
                        help="default: $PBG_MANY_PEPTIDES_REFERENCE")
    parser.add_argument("--topology", type=Path, default=os.environ.get("PBG_MANY_PEPTIDES_TOPOLOGY"),
                        help="pickled dict of MDTraj topologies by sequence, for table1 "
                             "(default: $PBG_MANY_PEPTIDES_TOPOLOGY)")
    parser.add_argument("--metrics", nargs="+", choices=GROUPS, default=["table1", "starred"])
    parser.add_argument("--robin", action="store_true", help="ROBIN rollouts: no reflection correction")
    parser.add_argument("--reference-stereo", action="store_true",
                        help="intended stereochemistry from the reference MD frames (D-proline appendix)")
    parser.add_argument("--floor", action="store_true", help="box-volume floor on the reference MD")
    parser.add_argument("--sequence", nargs="*", help="default: every test system of --sizes but YFPHAGYT")
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.rollouts and "table1" in args.metrics and args.topology is None:
        parser.error("table1 needs --topology or $PBG_MANY_PEPTIDES_TOPOLOGY")

    failures = []
    if args.rollouts or args.floor:
        systems = discover_many_peptides_test_systems(args.reference_root, sizes=SIZES)
        # The floor seeds each system by its index in the test list without YFPHAGYT, as in the paper.
        floor_seed = {system.sequence: index for index, system in enumerate(
            system for system in systems if system.sequence not in EXCLUDED_SYSTEMS)}
        chosen = [
            system for system in systems
            if system.size in args.sizes
            and (system.sequence in args.sequence if args.sequence else system.sequence not in EXCLUDED_SYSTEMS)
        ][args.shard::args.num_shards]
        topologies = None
        if args.rollouts and "table1" in args.metrics:
            with args.topology.open("rb") as handle:
                topologies = pickle.load(handle)
        rotations = box_volume_rotations() if args.floor or "box_volume" in args.metrics else None
        for system in chosen:
            reference = load_many_peptides_test_reference(system.reference_path, sequence=system.sequence)
            floor = args.output / "floor" / f"{system.sequence}.json"
            if args.floor and not floor.exists():
                seed = floor_seed[system.sequence]
                payload = {"sequence": system.sequence, "size": system.size, "seed": seed,
                           "metrics": box_volume_floor(reference, seed=seed, rotations=rotations)}
                write_atomic(floor, json.dumps(payload, indent=1) + "\n")
                print("floor", system.sequence, json.dumps(payload["metrics"]), flush=True)
            smiles = reference_stereo_smiles(reference) if args.reference_stereo and args.rollouts else None
            for seed, directory in enumerate(args.rollouts, start=1):
                path = next(directory.glob(f"{system.sequence}__*__intended.npz"), None)
                if path is None:
                    print(f"MISSING seed{seed} {system.sequence}: no rollout in {directory}", flush=True)
                    failures.append(f"seed{seed}/{system.sequence}")
                    continue
                output = args.output / f"seed{seed}" / f"{system.sequence}.json"
                settings = {"rollout": str(path.resolve()), "reflect": not args.robin, "smiles": smiles}
                result = json.loads(output.read_text()) if output.exists() else {
                    "sequence": system.sequence, "size": system.size, "seed": seed, **settings,
                    "groups": [], "metrics": {},
                }
                if any(result[key] != value for key, value in settings.items()):
                    raise SystemExit(f"{output} holds results for other settings; use another --output")
                todo = [group for group in args.metrics if group not in result["groups"]]
                if not todo:
                    continue
                rollout = Rollout.load(path, reflect=not args.robin, smiles=smiles)
                result["n_reflected"] = rollout.n_reflected
                for group in todo:
                    try:
                        if group == "table1":
                            metrics = table1_metrics(rollout, reference, topologies[system.sequence])
                        elif group == "starred":
                            metrics = starred_metrics(rollout, reference)
                        else:
                            metrics = box_volume_metrics(rollout, reference, rotations)
                    except ValueError as error:
                        print(f"FAILED {group} seed{seed} {system.sequence}: {error}", flush=True)
                        failures.append(f"seed{seed}/{system.sequence}/{group}")
                        continue
                    result["metrics"].update(metrics)
                    result["groups"].append(group)
                write_atomic(output, json.dumps(result, indent=1) + "\n")
                print(f"seed{seed} {system.sequence}", json.dumps(result["metrics"]), flush=True)

    rows = summarize(json.loads(path.read_text()) for path in sorted(args.output.glob("seed*/*.json")))
    if rows:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        write_atomic(args.output / "summary.csv", buffer.getvalue())
        print(f"wrote {args.output / 'summary.csv'} ({len(rows)} cells)", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
