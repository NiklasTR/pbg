#!/usr/bin/env python3
"""Stereochemistry of generated samples (Appendix "Stereochemistry of Generated Samples").

Every generated conformer is valid as generated, fully mirrored and reflected (every stereocenter
inverted; the reflection changes coordinates only and the conformer keeps its generated
log-likelihood), or excluded (another stereoisomer, an unresolved stereocenter, or a non-finite
energy or log-likelihood). Per rollout NPZ the script counts

  n_kept               intended stereochemistry after the reflection, finite energy and log-q
                       (the frames that the evaluation keeps)
  n_reflected          fully mirrored frames that were reflected
  n_nonfinite          frames with a non-finite energy or log-q (always excluded)
  n_excluded_stereo    n_samples - n_kept - n_nonfinite
  n_valid_as_generated n_kept - n_reflected; null unless the split is exact, i.e. no reflection or
                       no non-finite frame with the intended stereochemistry (n_nonfinite_intended)
  n_reflected_kept_min lower bound of the reflected frames among the kept ones

Rollouts of scripts/sample_many_peptides.py store the frames already reflected: their counts come
from the sampler's ``all_mirrored_stereo_repair_count`` and ``stereo_valid`` mask. Other rollouts
are classified from the positions with ``pbg.core.stereo`` and reflected here. Models named in
``--robin`` reflect nothing, as in the paper's ROBIN row, so their mirror images count as excluded.

Give each model as a label and one rollout directory per sampling seed
(``{SEQ}__*__intended.npz``, as written by scripts/sample_many_peptides.py). Outputs in OUTPUT:

  rollouts.csv          one row per model, seed and system
  by_length.csv         Table "Stereochemistry of Generated Samples" (tab:stereo-validity): per model
                        and cohort (2AA, 4AA, 8AA without YFPHAGYT, YFPHAGYT) the mean over systems of
                        the per-system seed mean; a system needs every seed. worst_excluded_fraction
                        is the largest per-system excluded share (ties: first sequence).
  by_stereocenters.csv  Table "Stereochemistry by Number of Stereocenters" (tab:stereo-centers): PBG
                        models, per system pooled over models and seeds, then the mean over the
                        systems with the same number of stereocenters (YFPHAGYT left out).
  by_model.csv          ladder and scaling-cohort tables (tab:stereo-ladder, tab:stereo-scaling): per
                        model the mean over systems (YFPHAGYT left out), the mean per-system mirrored
                        share of the kept samples, its state (half: >= 40%, none: <= 5%, else mixed)
                        and the systems in each state. reflected_of_kept is the ratio of the means.
  stereo_statistics.json all four tables.

Reading 10^4-frame rollouts is I/O bound; for the full Table 1 set use a CPU node, e.g.
  srun -p cpu -t 01:00:00 -c 4 --mem 16G python scripts/stereo_statistics.py \\
      --model "PBG 640M" pbg640/seed1 pbg640/seed2 pbg640/seed3 \\
      --model "ROBIN (T=1.0)" robin/seed1 robin/seed2 robin/seed3 --robin "ROBIN (T=1.0)" \\
      --output results/stereo
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pbg.core.stereo import (
    compile_stereo_spec,
    configured_stereo_key,
    intended_stereo_mask,
    stereo_keys,
)
from pbg.eval.metrics import EXCLUDED_SYSTEMS

REFLECTED_ON_DISK = "repaired_total_reflection_v1"  # ensemble_metadata.enantiomer_correction of the PBG sampler
COHORTS = ("2AA", "4AA", "8AA", *EXCLUDED_SYSTEMS)
HALF, NONE = 0.4, 0.05


def count(path: Path, *, reflect: bool) -> dict:
    """Stereo split of one rollout NPZ."""
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["positions"][:, :, 1:4], dtype=np.float64)
        finite = (np.isfinite(np.asarray(data["energy"], dtype=np.float64).reshape(-1))
                  & np.isfinite(np.asarray(data["generator_log_likelihood"], dtype=np.float64).reshape(-1)))
        smiles = str(np.asarray(data["smiles"]).item())
        metadata = json.loads(str(data["ensemble_metadata_json"])) if "ensemble_metadata_json" in data.files else {}
        stored_kept = np.asarray(data["stereo_valid"], dtype=bool).reshape(-1) if "stereo_valid" in data.files else None
    spec = compile_stereo_spec(smiles)
    n_centers = int(spec.atoms.shape[0])
    if metadata.get("enantiomer_correction") == REFLECTED_ON_DISK:
        # The sampler reflected the fully mirrored frames before it saved them and counted them.
        if not reflect:
            raise ValueError(f"{path}: the sampler reflected mirror images on disk; it is not a ROBIN rollout")
        if metadata.get("proposal_density") != "generated" or stored_kept is None:
            raise ValueError(f"{path}: needs proposal_density 'generated' and a stereo_valid mask")
        n_reflected, kept, source = int(metadata["all_mirrored_stereo_repair_count"]), stored_kept, "sampler"
        if kept.size != finite.size or np.any(kept & ~finite):
            raise ValueError(f"{path}: stereo_valid differs in length or finiteness from energy and log-q")
        if metadata.get("valid_output_frame_count") not in (None, int(kept.sum())):
            raise ValueError(f"{path}: valid_output_frame_count differs from stereo_valid")
    else:
        intended_key = configured_stereo_key(smiles, spec)
        mirrored = np.zeros(finite.size, dtype=bool)
        if reflect and n_centers and intended_key is not None:
            config, resolved = (np.asarray(value) for value in stereo_keys(spec, xyz))
            full = (1 << n_centers) - 1
            mirrored = (resolved == full) & (config == (intended_key ^ full))
            xyz[mirrored] *= -1.0
        n_reflected, kept, source = int(mirrored.sum()), None, "positions"
    intended = intended_stereo_mask(smiles, xyz)
    kept = intended & finite if kept is None else kept
    n_samples, n_kept = int(finite.size), int(kept.sum())
    n_nonfinite, n_nonfinite_intended = int((~finite).sum()), int((intended & ~finite).sum())
    exact = n_reflected == 0 or n_nonfinite_intended == 0
    return {
        "n_samples": n_samples, "n_kept": n_kept, "n_reflected": n_reflected,
        "n_valid_as_generated": n_kept - n_reflected if exact else None,
        "n_reflected_kept_min": n_reflected - min(n_reflected, n_nonfinite_intended),
        "n_nonfinite": n_nonfinite, "n_nonfinite_intended": n_nonfinite_intended,
        "n_excluded_stereo": n_samples - n_kept - n_nonfinite, "n_stereocenters": n_centers,
        "exact": exact, "source": source,
        "valid_as_generated": (n_kept - n_reflected) / n_samples if exact else None,
        "reflected": n_reflected / n_samples, "kept": n_kept / n_samples, "excluded": 1.0 - n_kept / n_samples,
        "excluded_nonfinite": n_nonfinite / n_samples,
        "excluded_stereo": (n_samples - n_kept - n_nonfinite) / n_samples,
    }


def mean(values) -> float | None:
    """Mean that is null when any value is null (a share that needs every input)."""
    values = list(values)
    return None if not values or any(value is None for value in values) else float(np.mean(values))


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", action="append", default=[], metavar=("LABEL", "DIR"),
                        help="model label, then one rollout directory per sampling seed (repeat per model)")
    parser.add_argument("--robin", nargs="+", default=[], metavar="LABEL",
                        help="models whose evaluation reflects nothing (ROBIN)")
    parser.add_argument("--sequence", nargs="*", help="only these systems (default: every rollout)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = [(spec[0], [Path(d) for d in spec[1:]], spec[0] in args.robin) for spec in args.model]
    if not models or any(not directories for _, directories, _ in models):
        parser.error("give each --model a label and at least one rollout directory")
    unknown = set(args.robin) - {label for label, _, _ in models}
    if unknown:
        parser.error(f"--robin names unknown models: {sorted(unknown)}")

    rows = []
    for label, directories, robin in models:
        for seed, directory in enumerate(directories, start=1):
            for path in sorted(directory.glob("*__intended.npz")):
                sequence = path.name.split("__")[0]
                if args.sequence and sequence not in args.sequence:
                    continue
                size = f"{len(sequence)}AA"
                rows.append({"model": label, "robin": robin, "seed": seed, "sequence": sequence, "size_label": size,
                             "cohort": sequence if sequence in EXCLUDED_SYSTEMS else size, "rollout": str(path),
                             **count(path, reflect=not robin)})
                print(label, f"seed{seed}", sequence, json.dumps({k: rows[-1][k] for k in ("n_kept", "n_reflected")}),
                      flush=True)

    systems = defaultdict(list)  # (model, cohort, sequence) -> one row per seed
    for row in rows:
        systems[row["model"], row["cohort"], row["sequence"]].append(row)
    seeds = {label: len(directories) for label, directories, _ in models}
    cells = defaultdict(list)
    for (label, cohort, sequence), group in systems.items():
        cells[label, cohort].append({
            "sequence": sequence, "seeds": len(group), "n_reflected": sum(r["n_reflected"] for r in group),
            "n_kept": sum(r["n_kept"] for r in group),
            **{share: mean(r[share] for r in group) for share in
               ("valid_as_generated", "reflected", "kept", "excluded", "excluded_nonfinite")}})

    by_length = []
    for label, _, robin in models:
        for cohort in COHORTS:
            group = [cell for cell in cells.get((label, cohort), []) if cell["seeds"] == seeds[label]]
            if not group:
                continue
            worst = min(group, key=lambda cell: (-cell["excluded"], cell["sequence"]))
            by_length.append({
                "model": label, "robin": robin, "cohort": cohort, "evaluated_systems": len(group),
                "seeds": seeds[label],
                **{f"{share}_fraction": mean(cell[share] for cell in group) for share in
                   ("valid_as_generated", "reflected", "kept", "excluded")},
                "worst_excluded_fraction": worst["excluded"], "worst_sequence": worst["sequence"],
                "excluded_nonfinite_fraction": mean(cell["excluded_nonfinite"] for cell in group)})

    by_model = []
    for label, _, robin in models:
        group = [cell for (model, cohort), own in cells.items() if model == label and cohort not in EXCLUDED_SYSTEMS
                 for cell in own]
        if not group:
            continue
        of_kept = [cell["n_reflected"] / cell["n_kept"] for cell in group if cell["n_kept"]]
        mirrored_of_kept = float(np.mean(of_kept)) if of_kept else None
        means = {f"{share}_fraction": mean(cell[share] for cell in group) for share in
                 ("reflected", "valid_as_generated", "kept", "excluded")}
        by_model.append({
            "model": label, "robin": robin, "evaluated_systems": len(group),
            "seeds_min": min(cell["seeds"] for cell in group), **means,
            "reflected_of_kept": means["reflected_fraction"] / means["kept_fraction"] if means["kept_fraction"] else None,
            "mirrored_of_kept_fraction": mirrored_of_kept,
            "state": None if mirrored_of_kept is None else
            "half" if mirrored_of_kept >= HALF else "none" if mirrored_of_kept <= NONE else "mixed",
            "half_mirrored_systems": sum(value >= HALF for value in of_kept),
            "unmirrored_systems": sum(value <= NONE for value in of_kept)})

    pooled = defaultdict(list)  # PBG systems pooled over models and seeds
    for row in rows:
        if not row["robin"] and row["cohort"] not in EXCLUDED_SYSTEMS:
            pooled[row["n_stereocenters"], row["size_label"], row["sequence"]].append(row)
    centers = defaultdict(list)
    for (n_centers, size, sequence), group in pooled.items():
        centers[n_centers].append({
            "size_label": size, "runs": len(group), "samples": sum(r["n_samples"] for r in group),
            "nonfinite_frames": sum(r["n_nonfinite"] for r in group),
            **{share: mean(r[share] for r in group) for share in
               ("reflected", "valid_as_generated", "excluded", "excluded_nonfinite", "excluded_stereo")}})
    by_stereocenters = [{
        "n_stereocenters": n_centers, "systems": len(group),
        "size_labels": ",".join(sorted({cell["size_label"] for cell in group})),
        "runs_min": min(cell["runs"] for cell in group),
        "mirrored_fraction": mean(cell["reflected"] for cell in group),
        **{f"{share}_fraction": mean(cell[share] for cell in group) for share in
           ("valid_as_generated", "excluded", "excluded_nonfinite", "excluded_stereo")},
        "worst_excluded_fraction": max(cell["excluded"] for cell in group),
        "samples": sum(cell["samples"] for cell in group),
        "nonfinite_frames": sum(cell["nonfinite_frames"] for cell in group),
    } for n_centers, group in sorted(centers.items())]

    args.output.mkdir(parents=True, exist_ok=True)
    tables = {"rollouts": rows, "by_length": by_length, "by_stereocenters": by_stereocenters, "by_model": by_model}
    for name, table in tables.items():
        write_csv(args.output / f"{name}.csv", table)
    (args.output / "stereo_statistics.json").write_text(json.dumps(tables, indent=1) + "\n")
    print(json.dumps({name: len(table) for name, table in tables.items()}), flush=True)


if __name__ == "__main__":
    main()
