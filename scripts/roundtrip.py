#!/usr/bin/env python3
"""Quantization roundtrips of reference MD frames: paper Appendix "Quantization and Reconstruction
Effects", Figures "Force Scaling and Roundtrip Quantization Error", "Midpoint versus Uniform
Reconstruction" and "Quantization Error across Resolutions" (the MD roundtrip series).

Each Many Peptides test frame $x$ (10,000 per system, centred on its centroid) is quantized on a
$b$-bin grid over $[-1.638, 1.638]$ nm per axis, $\\Delta = 3.276 / (b - 1)$ nm, and reconstructed at
the cell centre (midpoint) or uniformly inside the cell (uniform). Per system and bin count:

  E-W2*   exact W2 between the reduced energies $\\beta U$ of the reconstructed and original frames (kBT)
  rESS*   Kish ESS over the frame count, with paired log weights $-\\beta (U(y) - U(x))$
  MSF     total mean squared force $E[\\sum_i |F_i|^2]$ of the original frames (kJ^2 mol^-2 nm^-2)
  predicted E-W2*  $c \\beta^2 \\Delta^2 \\mathrm{MSF} / 24$, $c = 1$ (midpoint) or $2$ (uniform)

Energies and forces use Amber14 with OBC1 implicit solvent and a 2 nm non-periodic cutoff at 310 K.
The paper computed them with a float64 JAX port of this potential; here the OpenMM Reference platform
(double precision) evaluates the same potential. Uniform noise is seeded with
``uint32[20260720 + i, b]``, where ``i`` is the system's index in
``discover_many_peptides_test_systems`` (92 systems).

Subcommands:
  run        one system (``--sequence`` or ``--index``, default SLURM_ARRAY_TASK_ID):
             OUTPUT/systems/<size>/<SEQ>.json and <SEQ>.npz (per-frame energies, squared forces)
  summarize  OUTPUT/summary.json: per-system rows and medians across systems per bin count

Example (CPU node, one Slurm array task per system):
  sbatch -p cpu -t 04:00:00 -c 2 --mem 16G --array=0-91 --wrap \\
      "python scripts/roundtrip.py run --reference-root data/many-peptides-md/trajectories_subsampled"
  python scripts/roundtrip.py summarize
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K
from pbg.data.bin import apply_discretise, undo_discretise
from pbg.energy import ProtocolEvaluator
from pbg.estimator import importance_weights, relative_ess, w2_1d
from pbg.eval.many_peptides import (
    DEFAULT_REFERENCE_ROOT,
    discover_many_peptides_test_systems,
)

BINS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
HALF_BOX_NM = 1.638
TEMPERATURE_K = 310.0
UNIFORM_SEED = 20260720
RECONSTRUCTIONS = {"midpoint": ("center", 1.0), "uniform": ("uniform", 2.0)}  # kernel, prediction factor


def run(args: argparse.Namespace) -> None:
    """Roundtrip every frame of one system at every bin count with both reconstruction kernels."""
    systems = discover_many_peptides_test_systems(args.reference_root)
    if args.sequence is not None:
        index = next(i for i, system in enumerate(systems) if system.sequence == args.sequence)
    else:
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    system = systems[index]
    with np.load(system.reference_path, allow_pickle=False) as data:
        xyz = np.asarray(data["positions"], dtype=np.float32)
    if args.max_frames is not None:
        xyz = xyz[: args.max_frames]
    xyz -= xyz.mean(axis=1, keepdims=True)
    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * TEMPERATURE_K)
    evaluator = ProtocolEvaluator.peptide(system.sequence, "ff14sb-obc1-2nm", platform_name=args.platform)
    energy, force = evaluator(xyz.astype(np.float64), forces=True)
    force_squared = np.sum(force * force, axis=(1, 2))
    msf = float(np.mean(force_squared))
    arrays = {"reference_energy_kj_mol": energy, "force_squared_kj2_mol2_nm2": force_squared}
    rows = []
    for bins in args.bins:
        spacing = 2.0 * HALF_BOX_NM / (bins - 1)
        indices = apply_discretise(xyz, num_bins=bins, min_val=-HALF_BOX_NM, max_val=HALF_BOX_NM)
        for name in args.reconstructions:
            kernel, factor = RECONSTRUCTIONS[name]
            reconstructed = undo_discretise(
                indices, num_bins=bins, min_val=-HALF_BOX_NM, max_val=HALF_BOX_NM, method=kernel,
                key=np.asarray([UNIFORM_SEED + index, bins], dtype=np.uint32),
            )
            reconstructed_energy = evaluator(reconstructed.astype(np.float64))
            log_weight = -beta * (reconstructed_energy - energy)
            arrays[f"bins_{bins}_{name}_energy_kj_mol"] = reconstructed_energy
            rows.append({
                "bins": bins,
                "reconstruction": name,
                "spacing_nm": spacing,
                "energy_w2_kbt": w2_1d(beta * reconstructed_energy, beta * energy),
                "ress": relative_ess(importance_weights(log_weight), n_proposals=log_weight.size),
                "mean_energy_change_kbt": float(np.mean(-log_weight)),
                "predicted_energy_w2_kbt": factor * beta**2 * spacing**2 * msf / 24.0,
                "coordinate_rmse_nm": float(np.sqrt(np.mean((reconstructed - xyz) ** 2))),
            })
    output = args.output / "systems" / system.size / system.sequence
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    payload = {
        "status": "ok", "sequence": system.sequence, "size": system.size, "system_index": index,
        "atom_count": xyz.shape[1], "n_frames": int(xyz.shape[0]), "temperature_k": TEMPERATURE_K,
        "mean_squared_force_kj2_mol2_nm2": msf, "mean_squared_force_per_atom_kj2_mol2_nm2": msf / xyz.shape[1],
        "energy_backend": f"OpenMM {args.platform}: Amber14 + OBC1, CutoffNonPeriodic 2.0 nm",
        "rows": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"system": system.sequence, "output": str(output.with_suffix(".json"))}))


def summarize(args: argparse.Namespace) -> None:
    """Collect per-system rows and take medians across systems per reconstruction and bin count."""
    systems = [
        json.loads(path.read_text(encoding="utf-8"))
        for size_dir in sorted((args.output / "systems").iterdir())
        for path in sorted(size_dir.glob("*.json"))
    ]
    rows = [{"sequence": s["sequence"], "size": s["size"], "atom_count": s["atom_count"],
             "mean_squared_force_kj2_mol2_nm2": s["mean_squared_force_kj2_mol2_nm2"], **row}
            for s in systems for row in s["rows"]]
    medians = []
    for key in sorted({(row["reconstruction"], row["bins"]) for row in rows}):
        group = [row for row in rows if (row["reconstruction"], row["bins"]) == key]
        medians.append({"reconstruction": key[0], "bins": key[1], "n_systems": len(group),
                        **{f"median_{metric}": float(np.median([row[metric] for row in group]))
                           for metric in ("energy_w2_kbt", "ress", "predicted_energy_w2_kbt")}})
    output = args.output / "summary.json"
    output.write_text(json.dumps({"systems": rows, "medians": medians}, indent=2) + "\n", encoding="utf-8")
    for row in medians:
        print(json.dumps(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="roundtrip one system")
    p.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    p.add_argument("--sequence")
    p.add_argument("--index", type=int)
    p.add_argument("--bins", type=int, nargs="+", default=list(BINS))
    p.add_argument("--reconstructions", nargs="+", choices=sorted(RECONSTRUCTIONS), default=list(RECONSTRUCTIONS))
    p.add_argument("--max-frames", type=int, help="first N frames only (smoke tests)")
    p.add_argument("--platform", default="Reference", help="OpenMM platform")
    p.add_argument("--output", type=Path, default=Path("results/roundtrip"))
    p.set_defaults(handler=run)
    p = sub.add_parser("summarize", help="medians across systems")
    p.add_argument("--output", type=Path, default=Path("results/roundtrip"))
    p.set_defaults(handler=summarize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
