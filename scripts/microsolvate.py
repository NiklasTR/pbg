#!/usr/bin/env python3
"""Microsolvation task: paper Table "Microsolvation Task" and Figures "Autoregressive microsolvation
of an octapeptide" (right panel) and "Energy Change per Added Water".

For each net-neutral test peptide, 512 reference 300 K solute conformers from explicit-solvent REMD
are held fixed and k waters are placed around each of them (2AA: 80, 4AA: 64, 8AA: 32). Methods:

  pbg    PBG microsolvation model (paper: dipper_85m_core_bin4096_solvated_maxd_from160n_step_148756),
         waters sampled after the fixed solute tokens (sample_waters_given_solute)
  orca   ORCA 6.1.1 SOLVATOR, stochastic cluster mode with GFN-xTB and ALPB water, nsolv = k
  tleap  AmberTools tleap solvateBox (TIP3P, 10 Angstrom padding), then the k nearest waters
  md     Reference MD: the k nearest waters of the source REMD frame, by minimum-image distance

Every frame is scored with the non-periodic Amber14/TIP3P reaction-field potential (2 nm cutoff).

Subcommands:
  prepare    select the 512 solute conformers of one peptide and crop its reference shell (case file)
  run        place waters with one method on one case and score the frames (result .npz and .json)
  ladder     score U(k) for k = 0..K waters of one result, adding waters in the method's order
  summarize  table cells and the per-water energy ladder over a results root

Layout under ``--output-root`` (the research layout, so the published outputs summarize directly):
  cases/<size>/<SEQ>__w<k>__f512__case.npz
  <method>/<size>/<SEQ>__w<k>__f512__<method>.{npz,json}       method: reference_md, ambertools_tleap,
  pbg/<model-tag>/<size>/<SEQ>__w<k>__f512__pbg.{npz,json}          orca_solvator
  ladder/<method dir>/<size>/<stem>__ladder.npz

The table averages per-system minimum and median energies over systems. With ``--seed-roots`` (the
paper uses seeds 1-3) the PBG, tleap and ORCA cells are seed means and the printed spread is the
largest deviation of one seed mean, in units of the last printed digit. ``run --seed-index k`` makes
seed k: PBG changes its sampling seed; tleap and ORCA are deterministic, so their solute frames are
rotated by a random rotation about the centroid before placement and the result is rotated back.
Speeds come from ``--root`` only. Ladders follow the figure: per-system frame median of U(k) - U(0),
cohort median over systems, adjacent difference, kcal/mol. MD and tleap waters are ordered by the
distance from the water geometric centre to the nearest solute atom; ORCA and PBG waters keep their
addition order.

External programs: ``--orca`` or OPI_ORCA, ``--xtb`` or XTBEXE, ``--amberhome`` or AMBERHOME. Set
``--workdir`` (PBG_SOLVATE_TMP) to node-local disk when many workers run tleap or ORCA.

Example (CPU node; pbg needs a GPU node):
  python scripts/microsolvate.py prepare --test-root data/solvated_peptides/test --sequence MRDPVLFA \\
      --output-root results/microsolvation
  python scripts/microsolvate.py run tleap results/microsolvation/cases/8AA/MRDPVLFA__w032__f512__case.npz \\
      --output-root results/microsolvation
  python scripts/microsolvate.py ladder results/microsolvation/ambertools_tleap/8AA/*__f512__*.npz \\
      --output-root results/microsolvation
  python scripts/microsolvate.py summarize --root results/microsolvation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from pbg.core.constants import KJ_PER_KCAL
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_helm, sequence_to_smiles
from pbg.data.subsample import subsample_solvent
from pbg.energy import ProtocolEvaluator, expected_solvated_smiles

PROTOCOL = "fixed_solute_solvator_baselines_v1"
WATERS_BY_SIZE = {"2AA": 80, "4AA": 64, "8AA": 32}
N_FRAMES = 512
PILOT_FRAMES = 100  # the published cases extend a 100-frame pilot selection to 512 frames
CASE_SEED = 20260823
SEED_BASE = 2026111100  # seed replicate k uses SEED_BASE + k; k = 0 is the published run
TEMPERATURE_K = 300.0
SOLVENT_SELECTION = "min_image_distance_to_any_solute_atom"
METHODS = {"md": "reference_md", "tleap": "ambertools_tleap", "orca": "orca_solvator", "pbg": "pbg"}
LABELS = {"reference_md": "Reference MD", "ambertools_tleap": "AMBER tleap", "orca_solvator": "ORCA SOLVATOR"}
NEAREST_ORDER = {"reference_md", "ambertools_tleap"}


def method_dir(method: str, model_tag: str) -> str:
    return f"pbg/{model_tag}" if method == "pbg" else method


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def item(value: np.ndarray) -> object:
    return np.asarray(value).item()


def prepare(args: argparse.Namespace) -> None:
    """Select the solute conformers of one peptide and crop the Reference MD shell.

    Frames come from the 300 K replica of every finished REMD run (``<SEQ>/*/trajectories.npz`` with
    a DONE marker) that holds at least k waters. The first 100 frames are drawn with ``CASE_SEED``
    and 412 more from the rest, as in the published cases. The shell is the k waters whose geometric
    centre is nearest to any solute atom after moving each water to its periodic image nearest the
    peptide; all coordinates are centred on the solute centroid. ``case_id`` hashes the unimaged
    crop of the published cases, so PBG sampling seeds and seed rotations match the published run.
    """
    sequences = sorted(
        path.name for path in args.test_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    from rdkit import Chem

    neutral = [s for s in sequences if Chem.GetFormalCharge(Chem.MolFromSmiles(sequence_to_smiles(s))) == 0]
    if args.sequence is None and args.index is None and "SLURM_ARRAY_TASK_ID" not in os.environ:
        rows = [{"index": i, "sequence": s, "size_label": f"{len(s)}AA"} for i, s in enumerate(neutral)]
        print(json.dumps(rows, indent=2))
        return
    index = args.index if args.index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    sequence = args.sequence or neutral[int(index)]
    size = f"{len(sequence)}AA"
    k = WATERS_BY_SIZE[size]
    solute_z = atom_numbers_from_smiles(expected_solvated_smiles(sequence, 0)).astype(np.float64)
    n_solute = solute_z.size
    blocks = []
    for run_dir in sorted(p for p in (args.test_root / sequence).iterdir() if p.is_dir()):
        if not (run_dir / "DONE").is_file() or not (run_dir / "trajectories.npz").is_file():
            continue
        with np.load(run_dir / "trajectories.npz", allow_pickle=False) as data:
            xyz, box = data["positions"][0].copy(), data["box_vectors"][0].copy()  # 300 K replica only
        n_solvent = xyz.shape[1] - n_solute
        if n_solvent >= 0 and n_solvent % 3 == 0 and n_solvent // 3 >= k:
            blocks.append((xyz, box))
    lengths = np.asarray([xyz.shape[0] for xyz, _ in blocks], dtype=np.int64)
    total = int(lengths.sum())
    n_pilot = min(args.n_frames, PILOT_FRAMES)
    selected = np.sort(np.random.default_rng(args.seed).choice(total, size=n_pilot, replace=False))
    if args.n_frames > n_pilot:
        rest = np.setdiff1d(np.arange(total, dtype=np.int64), selected, assume_unique=True)
        rng = np.random.default_rng(args.seed ^ args.n_frames ^ 0x512)
        selected = np.concatenate([selected, np.sort(rng.choice(rest, size=args.n_frames - n_pilot, replace=False))])
    ends = np.cumsum(lengths)
    block_index = np.searchsorted(ends, selected, side="right")
    local_index = selected - (ends - lengths)[block_index]
    crops = {imaged: np.empty((args.n_frames, n_solute + 3 * k, 4)) for imaged in (False, True)}
    for b, (xyz_block, box_block) in enumerate(blocks):
        rows = np.flatnonzero(block_index == b)
        if rows.size == 0:
            continue
        for imaged in (False, True):
            xyz = np.asarray(xyz_block[local_index[rows]], dtype=np.float64)
            if imaged:
                edges = np.diagonal(np.asarray(box_block[local_index[rows]], dtype=np.float64), axis1=1, axis2=2)
                waters = xyz[:, n_solute:].reshape(rows.size, -1, 3, 3)
                for f in range(rows.size):
                    delta = waters[f].mean(axis=1)[:, None, :] - xyz[f, None, :n_solute]
                    shift = -edges[f] * np.round(delta / edges[f])
                    nearest = np.linalg.norm(delta + shift, axis=-1).argmin(axis=1)
                    waters[f] += shift[np.arange(waters.shape[1]), nearest][:, None, :]
                xyz[:, n_solute:] = waters.reshape(rows.size, -1, 3)
            zxyz = np.empty((*xyz.shape[:2], 4))
            zxyz[:, :, 1:4] = xyz
            zxyz[:, :n_solute, 0] = solute_z
            zxyz[:, n_solute:, 0] = np.tile([8.0, 1.0, 1.0], (xyz.shape[1] - n_solute) // 3)
            crops[imaged][rows] = subsample_solvent(zxyz, n_solute=n_solute, k=k)
    for crop in crops.values():
        crop[:, :, 1:4] -= crop[:, :n_solute, 1:4].mean(axis=1, keepdims=True)
    if not np.array_equal(crops[False][:, :n_solute], crops[True][:, :n_solute]):
        raise RuntimeError(f"{sequence}: imaging changed the solute frames")
    header = {"protocol": PROTOCOL, "sequence": sequence, "k_waters": k, "n_frames": args.n_frames, "seed": args.seed}
    digest = hashlib.sha256(json.dumps(header, separators=(",", ":"), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(selected).tobytes())
    digest.update(np.ascontiguousarray(crops[False]).tobytes())
    path = args.output_root / "cases" / size / f"{sequence}__w{k:03d}__f{args.n_frames}__case.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, protocol=PROTOCOL, case_id=digest.hexdigest(), sequence=sequence, size_label=size, k_waters=k,
        n_frames=args.n_frames, n_solute_atoms=n_solute, temperature=TEMPERATURE_K, seed=args.seed,
        solute_positions=crops[True][:, :n_solute], reference_positions=crops[True],
        source_global_indices=selected, source_block_lengths=lengths, solvent_selection=SOLVENT_SELECTION,
    )
    print(json.dumps({"case": str(path), "case_id": digest.hexdigest(), "source_frames": total}))


def run(args: argparse.Namespace) -> None:
    """Place k waters around every solute frame of one case with one method and score the frames.

    tleap and ORCA write rounded solute coordinates. Their frames are shifted by the mean solute
    displacement, the exact solute frames are written back, and the frames are scored again.
    """
    case = load_npz(args.case)
    method = METHODS[args.method]
    sequence, size, k = str(item(case["sequence"])), str(item(case["size_label"])), int(item(case["k_waters"]))
    case_id, n_frames = str(item(case["case_id"])), int(item(case["n_frames"]))
    solute = case["solute_positions"]
    n_solute = solute.shape[1]
    payload: dict[str, object] = {"seed_index": args.seed_index}
    arrays: dict[str, object] = {}
    started = time.perf_counter()
    if method == "reference_md":
        if str(item(case.get("solvent_selection", ""))) != SOLVENT_SELECTION:
            raise ValueError(f"{args.case} has no minimum-image reference shell; rebuild it with prepare")
        positions = case["reference_positions"]
    elif method == "pbg":
        import jax

        from pbg.models.pbg.model import PBG
        from pbg.models.pbg.sample import sample_waters_given_solute
        from pbg.models.pbg.system import MolecularSystem, checkpoint_environment_prompt

        model_id = args.model_id or args.checkpoint.stem
        key = f"{PROTOCOL}:{model_id}:{case_id}" + (f":{SEED_BASE + args.seed_index}" if args.seed_index else "")
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        model = PBG.load(args.checkpoint)
        system = MolecularSystem(
            id=f"{sequence}_w{k}", mapped_smiles=expected_solvated_smiles(sequence, k),
            helm=sequence_to_helm(sequence), environment=checkpoint_environment_prompt(args.checkpoint),
            temperature_K=TEMPERATURE_K, sequence=sequence,
        )
        started = time.perf_counter()
        batch = sample_waters_given_solute(
            model, system, jax.random.key(seed), solute_frames=solute, n_solute_atoms=n_solute,
            solute_smiles=expected_solvated_smiles(sequence, 0), batch_size=args.batch_size,
        )
        positions = np.asarray(batch.samples, dtype=np.float64)
        arrays.update(logq=batch.logq, model_id=model_id, model_tag=args.model_tag, sampling_seed=seed)
        payload.update(model_id=model_id, model_tag=args.model_tag, checkpoint=str(args.checkpoint),
                       sampling_seed=seed, batch_size=args.batch_size, helm=system.helm,
                       environment_prompt=system.environment)
    else:
        for name, value in (("OPI_ORCA", args.orca), ("XTBEXE", args.xtb), ("AMBERHOME", args.amberhome),
                            ("PBG_SOLVATE_TMP", args.workdir)):
            if value is not None:
                os.environ[name] = str(value)
        from pbg.eval.microsolvation import place_waters

        placed = solute
        rotation = None
        if args.seed_index:
            digest = hashlib.sha256(f"{case_id}:{SEED_BASE + args.seed_index}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:4], "little"))
            # Haar-uniform rotations: QR with the sign fix, improper ones flipped in one column.
            q, r = np.linalg.qr(rng.normal(size=(n_frames, 3, 3)))
            rotation = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
            rotation[np.linalg.det(rotation) < 0, :, 0] *= -1
            centroid = solute[:, :, 1:4].mean(axis=1, keepdims=True)
            placed = solute.copy()
            placed[:, :, 1:4] = np.einsum("fai,fji->faj", solute[:, :, 1:4] - centroid, rotation) + centroid
        positions = place_waters(placed, sequence, k, args.method, padding_angstrom=args.padding_angstrom,
                                 pal_workers=args.pal_workers, timeout_s=args.timeout_s)
        translation = (positions[:, :n_solute, 1:4] - placed[:, :, 1:4]).mean(axis=1, keepdims=True)
        positions[:, :, 1:4] -= translation
        if rotation is not None:
            positions[:, :, 1:4] = np.einsum("fai,fij->faj", positions[:, :, 1:4] - centroid, rotation) + centroid
        payload.update(
            rigid_translation_max_abs_nm=float(np.max(np.abs(translation))),
            solute_max_abs_error_nm=float(np.max(np.abs(positions[:, :n_solute, 1:4] - solute[:, :, 1:4]))),
            **({"padding_angstrom": args.padding_angstrom} if args.method == "tleap" else {"orca_nsolv": k}),
        )
        positions[:, :n_solute] = solute
    placement_s = time.perf_counter() - started
    energy = ProtocolEvaluator.peptide(sequence, "ff14sb-tip3p-rf2nm", n_waters=k, platform_name=args.platform)(positions)
    wallclock_s = time.perf_counter() - started
    if not np.array_equal(positions[:, :n_solute], solute):
        raise RuntimeError("the solute frames changed during water placement")
    directory = args.output_root / method_dir(method, args.model_tag) / size
    stem = f"{sequence}__w{k:03d}__f{n_frames}__{method}"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{stem}.npz", protocol=PROTOCOL, method=method, case_id=case_id, sequence=sequence,
        size_label=size, k_waters=k, n_frames=n_frames, n_solute_atoms=n_solute, temperature=TEMPERATURE_K,
        positions=positions, energy=energy, seed_index=args.seed_index, **arrays,
    )
    payload.update(
        status="ok", protocol=PROTOCOL, method=method, case_id=case_id, case_path=str(args.case),
        artifact_path=str(directory / f"{stem}.npz"), sequence=sequence, size_label=size, k_waters=k,
        n_frames=n_frames, n_solute_atoms=n_solute, temperature_k=TEMPERATURE_K, energy_unit="kJ/mol",
        energy_backend="amber14_tip3p", placement_wallclock_s=placement_s, wallclock_s=wallclock_s,
    )
    finite = energy[np.isfinite(energy)]
    payload["energy_distribution"] = {"count": int(finite.size), "nan_count": int(energy.size - finite.size)}
    if finite.size:
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        payload["energy_distribution"].update(
            mean=float(np.mean(finite)), std=float(np.std(finite)), min=float(np.min(finite)), max=float(np.max(finite)),
            percentiles={str(p): float(v) for p, v in zip(percentiles, np.percentile(finite, percentiles), strict=True)},
        )
    if method != "reference_md":
        payload["samples_per_second"] = n_frames / wallclock_s
    temporary = directory / f"{stem}.json.tmp"  # written in full, then renamed
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(directory / f"{stem}.json")
    print(json.dumps({key: payload[key] for key in ("artifact_path", "wallclock_s")}))


def ladder(args: argparse.Namespace) -> None:
    """Score U(k) for k = 0..K on every frame of each result; column K repeats the full-system energy."""
    for path in args.results:
        result = load_npz(path)
        positions, method = result["positions"], str(item(result["method"]))
        sequence, size = str(item(result["sequence"])), str(item(result["size_label"]))
        n_solute, k_waters = int(item(result["n_solute_atoms"])), int(item(result["k_waters"]))
        n_frames = positions.shape[0]
        waters = positions[:, n_solute:].reshape(n_frames, k_waters, 3, 4)
        if method in NEAREST_ORDER:
            centre = waters[..., 1:4].mean(axis=2)
            distance = np.linalg.norm(centre[:, :, None] - positions[:, None, :n_solute, 1:4], axis=-1).min(axis=-1)
            order = np.argsort(distance, axis=1, kind="stable")
        else:
            order = np.broadcast_to(np.arange(k_waters), (n_frames, k_waters))
        energy = np.empty((n_frames, k_waters + 1))
        for k in range(k_waters + 1):
            added = np.take_along_axis(waters, order[:, :k, None, None], axis=1).reshape(n_frames, 3 * k, 4)
            energy[:, k] = ProtocolEvaluator.peptide(sequence, "ff14sb-tip3p-rf2nm", n_waters=k,
                                                     platform_name=args.platform)(
                np.concatenate([positions[:, :n_solute], added], axis=1))
        model_tag = str(item(result["model_tag"])) if "model_tag" in result else ""
        output = args.output_root / "ladder" / method_dir(method, model_tag) / size / f"{path.stem}__ladder.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        endpoint = float(np.max(np.abs(energy[:, -1] - result["energy"])))
        np.savez_compressed(output, energy=energy, water_order=order, sequence=sequence, method=method,
                            size_label=size, endpoint_max_abs_diff_kj_mol=endpoint)
        print(json.dumps({"ladder": str(output), "endpoint_max_abs_diff_kj_mol": endpoint}))


def summarize(args: argparse.Namespace) -> None:
    """Write the table cells (kcal/mol) and the per-water energy ladders as JSON and print the table."""
    methods = ("reference_md", "ambertools_tleap", "orca_solvator", "pbg")
    labels = LABELS | {"pbg": f"PBG {args.model_tag.upper()}"}

    def results(root: Path, method: str, size: str) -> list[Path]:
        k = WATERS_BY_SIZE[size]
        directory = root / method_dir(method, args.model_tag) / size
        return sorted(directory.glob(f"*__w{k:03d}__f{args.n_frames}__{method}.npz")) if directory.is_dir() else []

    def cohort_means(root: Path, method: str, size: str) -> tuple[float, float, int]:
        energies = []
        for path in results(root, method, size):
            with np.load(path, allow_pickle=False) as data:
                energies.append(data["energy"])
        if not energies:
            return float("nan"), float("nan"), 0
        return float(np.mean([e.min() for e in energies])), float(np.mean([np.median(e) for e in energies])), len(energies)

    table, ladders, cohort = [], [], []
    for method in methods:
        for size in WATERS_BY_SIZE:
            low, median, n_systems = cohort_means(args.root, method, size)
            row = {"method": method, "label": labels[method], "size_label": size, "n_systems": n_systems,
                   "min_energy_kj_mol": low, "median_energy_kj_mol": median, "seeds": [0]}
            if args.seed_roots and method != "reference_md":
                seeds = [cohort_means(root, method, size) for root in args.seed_roots]
                for index, name in ((0, "min"), (1, "median")):
                    values = np.asarray([seed[index] for seed in seeds])
                    row[f"{name}_energy_kj_mol"] = float(values.mean())
                    row[f"{name}_energy_seed_range_kj_mol"] = [float(values.min()), float(values.max())]
                row["seeds"] = list(range(1, len(seeds) + 1))
                row["n_systems"] = min(seed[2] for seed in seeds)
            speeds = [json.loads(path.with_suffix(".json").read_text()).get("samples_per_second")
                      for path in results(args.root, method, size)]
            speeds = [s for s in speeds if s is not None]
            row["samples_per_second"] = float(np.mean(speeds)) if speeds else None
            table.append(row)

            deltas = []
            for path in results(args.root, method, size):
                ladder_path = (args.ladder_root or args.root) / "ladder" / method_dir(method, args.model_tag) / size
                ladder_path = ladder_path / f"{path.stem}__ladder.npz"
                if not ladder_path.is_file():
                    continue
                with np.load(ladder_path, allow_pickle=False) as data:
                    energy = data["energy"]
                delta = energy[:, 1:] - energy[:, :1]
                deltas.append(np.median(delta, axis=0))
                ladders.append({"method": method, "size_label": size, "sequence": path.name.split("__")[0],
                                "median_delta_u_kj_mol": deltas[-1].tolist(),
                                "mean_delta_u_kj_mol": delta.mean(axis=0).tolist()})
            if deltas:
                median = np.median(np.stack(deltas), axis=0) / KJ_PER_KCAL
                cohort.append({"method": method, "size_label": size, "n_systems": len(deltas),
                               "cohort_median_delta_u_kcal_mol": median.tolist(),
                               "energy_change_per_water_kcal_mol": np.diff(median, prepend=0.0).tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"table": table, "ladder_cohort": cohort, "ladder_systems": ladders},
                                      indent=2, allow_nan=True) + "\n")

    def cell(row: dict, name: str) -> str:
        value = row[f"{name}_energy_kj_mol"] / KJ_PER_KCAL
        if not np.isfinite(value):
            return "--"
        text = f"{value:#.4g}".rstrip(".")  # four significant digits, as printed in the paper
        spread = row.get(f"{name}_energy_seed_range_kj_mol")
        if spread:
            unit = 10.0 ** (np.floor(np.log10(abs(value))) - 3)
            deviation = max(spread[1] - row[f"{name}_energy_kj_mol"], row[f"{name}_energy_kj_mol"] - spread[0])
            units = int(np.floor(deviation / KJ_PER_KCAL / unit + 0.5))
            text += " <1" if units < 1 else f" ±{units}"
        return text

    print(f"{'kcal/mol':16s}" + "".join(f"{size + ' min':>18s}{size + ' median':>18s}{'samples/s':>11s}" for size in WATERS_BY_SIZE))
    for method in methods:
        rows = [row for row in table if row["method"] == method]
        line = "".join(
            f"{cell(row, 'min'):>18s}{cell(row, 'median'):>18s}"
            f"{'--' if row['samples_per_second'] is None else format(row['samples_per_second'], '#.4g').rstrip('.'):>11s}"
            for row in rows
        )
        print(f"{labels[method]:16s}{line}   (systems: {', '.join(str(row['n_systems']) for row in rows)})")
    print(json.dumps({"output": str(args.output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build the case file of one peptide (or list the neutral peptides)")
    p.add_argument("--test-root", type=Path, required=True, help="solvated peptide test REMD runs, <root>/<SEQ>/")
    p.add_argument("--output-root", type=Path, default=Path("results/microsolvation"))
    p.add_argument("--sequence")
    p.add_argument("--index", type=int, help="index into the neutral peptides (default: SLURM_ARRAY_TASK_ID)")
    p.add_argument("--n-frames", type=int, default=N_FRAMES)
    p.add_argument("--seed", type=int, default=CASE_SEED)
    p.set_defaults(handler=prepare)

    p = sub.add_parser("run", help="place waters with one method on one case")
    p.add_argument("method", choices=sorted(METHODS))
    p.add_argument("case", type=Path)
    p.add_argument("--output-root", type=Path, default=Path("results/microsolvation"))
    p.add_argument("--seed-index", type=int, default=0, help="0 is the published run; k > 0 is seed replicate k")
    p.add_argument("--platform", default="CPU", help="OpenMM platform for the energies")
    p.add_argument("--checkpoint", type=Path, help="pbg: model checkpoint")
    p.add_argument("--model-id", help="pbg: model id in the sampling seed (default: checkpoint stem)")
    p.add_argument("--model-tag", default="85m", help="pbg: output directory name")
    p.add_argument("--batch-size", type=int, default=512, help="pbg: frames per sampling batch")
    p.add_argument("--orca", type=Path, help="orca: ORCA executable (default: OPI_ORCA or PATH)")
    p.add_argument("--xtb", type=Path, help="orca: xtb executable (default: XTBEXE or PATH)")
    p.add_argument("--pal-workers", type=int, default=1, help="orca: %%pal nprocs")
    p.add_argument("--timeout-s", type=int, default=3600, help="orca: time limit per frame")
    p.add_argument("--amberhome", type=Path, help="tleap: AmberTools prefix (default: AMBERHOME or PATH)")
    p.add_argument("--padding-angstrom", type=float, default=10.0, help="tleap: solvateBox padding")
    p.add_argument("--workdir", type=Path, help="tleap/orca: scratch folder for per-frame runs")
    p.set_defaults(handler=run)

    p = sub.add_parser("ladder", help="score U(k) for k = 0..K waters of result files")
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--output-root", type=Path, default=Path("results/microsolvation"))
    p.add_argument("--platform", default="CPU")
    p.set_defaults(handler=ladder)

    p = sub.add_parser("summarize", help="table cells and per-water energy ladders")
    p.add_argument("--root", type=Path, default=Path("results/microsolvation"))
    p.add_argument("--seed-roots", type=Path, nargs="*", default=[], help="roots of seed replicates 1, 2, 3")
    p.add_argument("--ladder-root", type=Path, help="root that holds ladder/ (default: --root)")
    p.add_argument("--model-tag", default="85m")
    p.add_argument("--n-frames", type=int, default=N_FRAMES)
    p.add_argument("--output", type=Path, default=Path("results/microsolvation/summary.json"))
    p.set_defaults(handler=summarize)
    args = parser.parse_args()
    if args.command == "run" and args.method == "pbg" and args.checkpoint is None:
        parser.error("run pbg needs --checkpoint")
    args.handler(args)


if __name__ == "__main__":
    main()
