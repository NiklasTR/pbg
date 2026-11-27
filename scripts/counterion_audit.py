#!/usr/bin/env python3
"""Counterions in the explicit-water MD of the charged test peptides (Appendix, Microsolvated
Peptides dataset paragraph and Table "Counterions in the Explicit-Water MD of Charged Test
Peptides", tab:counterion-audit).

The periodic TIP3P box of a charged peptide holds one Na+ or Cl- per unit of net charge. For each
test peptide the script reads the 300 K replica (state 0) of ``--runs-per-peptide`` evenly spaced
REMD runs and measures, with minimum-image distances in the rectangular box:

* contact: the counterion is within 0.4 nm of a peptide heavy atom; the same fraction for a
  uniformly placed point gives the no-attraction baseline;
* inside cluster: the counterion is closer to the peptide than the 128th nearest water (water
  centre of mass to the nearest peptide atom), so a 128-water crop would contain it; also for the
  Table 3 water count (80, 64, 32 for 2, 4, 8 residues).

The uniform points come from one random stream (seed 20261115) over the peptides in sorted order,
so ``--sequence`` subsets reproduce every value of a full run except ``uniform_contact_fraction``.
The summary repeats the paper table: charged peptides only, grouped by counterion (Na+ for net
charge < 0, Cl- for net charge > 0), medians over peptides with the range; a peptide is "mixed"
when its counterion is inside the 128-water cluster in 10-90% of its frames.

Input layout (``--test-root`` or $PBG_MICROSOLVATED_TEST_ROOT): ``<SEQ>/<frame>.npz`` holds the
box atom numbers (``atoms``, peptide first, then waters, then ions) and every REMD run
``<SEQ>/<frame>_sim_*/`` with a ``DONE`` marker holds ``trajectories.npz`` (``positions`` and
``box_vectors`` in nm, replica first). Writes ``OUTPUT/peptides.json`` (one row per peptide) and
``OUTPUT/summary.json``. It reads a few GB; run it on a CPU node, for example:
  srun -p cpu -t 01:00:00 -c 2 --mem 16G python scripts/counterion_audit.py --output results/counterion
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from rdkit import Chem

from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_smiles

AVOGADRO = 6.02214076e23
CONTACT_NM = 0.4
CLUSTER_WATERS = 128
TABLE3_WATERS = {2: 80, 4: 64, 8: 32}
MIXED = (0.1, 0.9)
SEED = 20261115


def audit(test_root: Path, runs_per_peptide: int, sequences: set[str]) -> list[dict]:
    """One row of box and counterion statistics per test peptide."""
    rng = np.random.default_rng(SEED)
    rows = []
    for sequence_dir in sorted(p for p in test_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        sequence = sequence_dir.name
        if sequences and sequence not in sequences:
            continue
        smiles = sequence_to_smiles(sequence)
        charge = int(Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)))
        solute_z = atom_numbers_from_smiles(smiles)
        n_solute, heavy = int(solute_z.size), solute_z > 1
        runs = sorted(p for p in sequence_dir.iterdir()
                      if p.is_dir() and (p / "DONE").is_file() and (p / "trajectories.npz").is_file())
        picks = sorted(set(np.linspace(0, len(runs) - 1, min(runs_per_peptide, len(runs))).round().astype(int).tolist()))
        k_table = TABLE3_WATERS[len(sequence)]
        stats: dict[str, list[float]] = {key: [] for key in (
            "edge", "r128", "rk", "ion_min", "uniform_min", "in_128", "in_k")}
        for run in (runs[i] for i in picks):
            with np.load(sequence_dir / f"{run.name.split('_sim_')[0]}.npz", allow_pickle=False) as data:
                atoms = np.asarray(data["atoms"], dtype=np.int64)
            ions = np.flatnonzero(np.isin(atoms, (11, 17)))
            n_waters = int((atoms.size - n_solute - ions.size) // 3)
            with np.load(run / "trajectories.npz", allow_pickle=False) as data:
                xyz = np.asarray(data["positions"][0], dtype=np.float64)  # 300 K replica
                box = np.asarray(data["box_vectors"][0], dtype=np.float64)
            if xyz.shape[1] != atoms.size:
                raise ValueError(f"{run}: {xyz.shape[1]} atoms, template {atoms.size}")
            edges = np.stack([np.diag(b) for b in box])
            if not np.allclose(box - np.stack([np.diag(e) for e in edges]), 0.0):
                raise ValueError(f"{run}: box is not rectangular")
            for frame, edge in enumerate(edges):
                solute = xyz[frame, :n_solute]
                water = xyz[frame, n_solute:n_solute + 3 * n_waters].reshape(n_waters, 3, 3).mean(axis=1)
                delta = water[:, None, :] - solute[None, :, :]
                delta -= edge * np.round(delta / edge)
                order = np.sort(np.linalg.norm(delta, axis=-1).min(axis=1), kind="stable")
                r128, rk = order[min(CLUSTER_WATERS, n_waters) - 1], order[k_table - 1]
                stats["edge"].append(float(edge.mean()))
                stats["r128"].append(float(r128))
                stats["rk"].append(float(rk))
                if ions.size:
                    d = xyz[frame, ions][:, None, :] - solute[None, heavy, :]
                    d -= edge * np.round(d / edge)
                    ion_min = float(np.linalg.norm(d, axis=-1).min())
                    stats["ion_min"].append(ion_min)
                    stats["in_128"].append(float(ion_min < r128))
                    stats["in_k"].append(float(ion_min < rk))
                    u = (rng.random(3) * edge)[None, :] - solute[heavy]
                    u -= edge * np.round(u / edge)
                    stats["uniform_min"].append(float(np.linalg.norm(u, axis=-1).min()))
        if ions.size != abs(charge):
            raise ValueError(f"{sequence}: {ions.size} counterions for net charge {charge}")
        edge_mean = float(np.mean(stats["edge"]))
        row = {
            "sequence": sequence, "length": len(sequence), "net_charge": charge,
            "n_runs_read": len(picks), "n_runs_available": len(runs), "n_frames": len(stats["edge"]),
            "n_waters_box": n_waters, "n_ions": int(ions.size), "box_edge_nm": edge_mean,
            "ion_concentration_M": int(ions.size) / (AVOGADRO * edge_mean**3 * 1e-24),
            "r128_median_nm": float(np.median(stats["r128"])),
            "k_table3": k_table, "rk_median_nm": float(np.median(stats["rk"])),
        }
        if ions.size:
            ion_min = np.asarray(stats["ion_min"])
            row.update({
                "ion_min_median_nm": float(np.median(ion_min)),
                "ion_contact_fraction": float(np.mean(ion_min < CONTACT_NM)),
                "uniform_contact_fraction": float(np.mean(np.asarray(stats["uniform_min"]) < CONTACT_NM)),
                "ion_in_128_shell_fraction": float(np.mean(stats["in_128"])),
                "ion_in_k_shell_fraction": float(np.mean(stats["in_k"])),
            })
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    """The paper table and caption numbers: charged peptides by counterion."""
    charged = [row for row in rows if row["n_ions"] > 0]
    if not charged:
        raise ValueError("no charged peptide among the audited ones")

    def percent(value: float) -> int:  # half up, as Math.round in the paper export
        return math.floor(100 * value + 0.5)

    groups = {}
    for name, members in (("Na+", [r for r in charged if r["net_charge"] < 0]),
                          ("Cl-", [r for r in charged if r["net_charge"] > 0]), ("all charged", charged)):
        if not members:
            continue
        cell = {"peptides": len(members), "net_charge_range": [min(r["net_charge"] for r in members),
                                                                max(r["net_charge"] for r in members)]}
        for label, key in (("contact", "ion_contact_fraction"), ("inside_cluster", "ion_in_128_shell_fraction")):
            values = [r[key] for r in members]
            cell[label] = {"median": float(np.median(values)), "min": min(values), "max": max(values),
                           "table": f"{percent(np.median(values))}% ({percent(min(values))}-{percent(max(values))}%)"}
        cell["mixed"] = sum(MIXED[0] <= r["ion_in_128_shell_fraction"] <= MIXED[1] for r in members)
        groups[name] = cell
    return {
        "peptides": len(rows), "charged_peptides": len(charged),
        "mixed_peptides": groups["all charged"]["mixed"], "mixed_bounds": list(MIXED),
        "random_contact_median": float(np.median([r["uniform_contact_fraction"] for r in charged])),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-root", type=Path, default=os.environ.get("PBG_MICROSOLVATED_TEST_ROOT"),
                        help="test split of the explicit-water REMD data (default: $PBG_MICROSOLVATED_TEST_ROOT)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-per-peptide", type=int, default=8, help="evenly spaced REMD runs read (paper: 8)")
    parser.add_argument("--sequence", nargs="*", default=[], help="default: every test peptide")
    args = parser.parse_args()
    if args.test_root is None:
        parser.error("needs --test-root or $PBG_MICROSOLVATED_TEST_ROOT")
    rows = audit(args.test_root, args.runs_per_peptide, set(args.sequence))
    summary = {"test_root": str(args.test_root), "runs_per_peptide": args.runs_per_peptide,
               "full_stream": not args.sequence, **summarize(rows)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "peptides.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
