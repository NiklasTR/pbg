#!/usr/bin/env python3
"""Audit the D-proline artifact of Many Peptides (Appendix "D-Proline Artifact in Many Peptides").

Some published starting structures of the ManyPeptidesMD dataset (Hugging Face
transferable-samplers/many-peptides-md) put a side-chain atom of the residue before a proline
inside the proline ring. The structures are all L, but the default OpenMM minimization before the
MD can push the proline C-alpha through the plane of its substituents, and the MD keeps the D form.
The script reproduces the statistics of the appendix (Table tab:dpro-prev, the split counts, the
minimization figure fig:dpro-minimization and the YFPHAGYT sample shares):

1. Handedness from PDB atom names only: sign((N - CA) x (C - CA) . (CB - CA)) > 0 for L. A residue
   whose N, C or CB bond to CA is not 1.3-1.7 A long counts as failed (guards the atom order).
2. Clash: a side-chain heavy atom of the preceding residue lies within 1 A of the proline ring
   (N, CA, CB, CG, CD) in the published starting structure.
3. Artifact: the proline is D in a dataset frame: every training frame of the given webdataset
   shards (4 per peptide and shard; the paper read shards 0000 and 0001) and every 100th frame of the
   validation and test references.
4. Minimization: OpenMM ``simulation.minimizeEnergy()`` with default settings, as in md-runner
   (amber14-all.xml with implicit/obc1.xml, CutoffNonPeriodic at 2 nm, no constraints, CPU), of every
   peptide with a clash and ``--controls`` evenly spaced training peptides with a proline but no
   clash. The full path of ``--example`` (paper: FPL) is kept for the figure.
5. ``--yfphagyt LABEL NPZ ...`` (repeatable): stereo pattern of every YFPHAGYT sample relative to its
   SMILES (L-Pro3): ``reference`` = only the Pro3 C-alpha inverted (D-Pro3, as in the reference MD),
   ``smiles`` = the SMILES form, ``*_mirror`` = their mirror images. The paper passes the three
   Table 1 sampling seeds of every PBG model as one label and those of ROBIN as another.

Writes ``OUTPUT/report.json``: ``splits`` (peptides and prolines by split and length), ``by_prev``
(training prolines by the residue before them, with the minimized ones: Table tab:dpro-prev),
``rule`` (precision and recall of "Phe before Pro"), ``minimization`` (agreement of minimized and
dataset handedness, and the example path), ``yfphagyt`` (per rollout and per label) and
``prolines`` (one record per proline). Run on a CPU node, for example:
  srun -p cpu -t 01:00:00 -c 8 --mem 32G python scripts/dpro_artifact.py --pdbs pdb_tarfiles \\
      --shards webdatasets/single_frames/0000.tar webdatasets/single_frames/0001.tar \\
      --subsampled trajectories_subsampled --processes 8 --output results/dpro
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PRO_RING = ("N", "CA", "CB", "CG", "CD")
BACKBONE = {"N", "CA", "C", "O", "OXT"}
SPLITS = ("train", "val", "test")


def handedness(frames: np.ndarray, index: dict, residue: int) -> np.ndarray:
    """+1 for L, -1 for D and 0 for a failed bond check; frames (f, n, 3) in nm or angstrom."""
    ca, n, c, cb = (frames[:, index[(residue, name)]] for name in ("CA", "N", "C", "CB"))
    lengths = np.stack([np.linalg.norm(v - ca, axis=1) for v in (n, c, cb)])
    lengths *= 10.0 if np.median(lengths) < 0.3 else 1.0
    sign = np.sign(np.einsum("ij,ij->i", np.cross(n - ca, c - ca), cb - ca))
    return np.where(np.all((lengths > 1.3) & (lengths < 1.7), axis=0), sign, 0)


def minimize(job: tuple[str, str, str, bool]) -> dict:
    """Default OpenMM minimization of one published PDB: proline signed volumes before and after."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    split, sequence, text, keep_path = job
    pdb = app.PDBFile(io.StringIO(text))
    system = app.ForceField("amber14-all.xml", "implicit/obc1.xml").createSystem(
        pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=2.0 * unit.nanometer, constraints=None)
    index = {(atom.residue.index, atom.name): atom.index for atom in pdb.topology.atoms()}
    prolines = [residue.index for residue in pdb.topology.residues() if residue.name == "PRO"]

    def volumes(x: np.ndarray) -> list[float]:
        return [float(np.dot(np.cross(x[index[(r, "N")]] - x[index[(r, "CA")]], x[index[(r, "C")]] - x[index[(r, "CA")]]),
                             x[index[(r, "CB")]] - x[index[(r, "CA")]])) for r in prolines]

    start = np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    simulation = app.Simulation(
        pdb.topology, system,
        openmm.LangevinMiddleIntegrator(310 * unit.kelvin, 0.3 / unit.picosecond, 1.0 * unit.femtosecond),
        openmm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    simulation.context.setPositions(pdb.positions)
    start_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    path = [(0, start_energy, volumes(start), start if keep_path else None)]

    class Reporter(openmm.MinimizationReporter):
        def report(self, iteration, x, grad, args):
            xyz = np.asarray(x).reshape(-1, 3) * 10.0
            path.append((iteration + 1, args["system energy"], volumes(xyz), xyz if keep_path else None))
            return False

    simulation.minimizeEnergy(reporter=Reporter())
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    result = dict(split=split, sequence=sequence, prolines=[r + 1 for r in prolines], start=volumes(start),
                  minimized=volumes(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)),
                  iterations=len(path) - 1, start_energy_kj=start_energy,
                  final_energy_kj=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                  flip_iteration=[next((i for i, _, v, _ in path if v[k] < 0), None) for k in range(len(prolines))])
    if keep_path:
        result["path"] = dict(
            elements=[atom.element.atomic_number for atom in pdb.topology.atoms()],
            names=[[atom.residue.index + 1, atom.residue.name, atom.name] for atom in pdb.topology.atoms()],
            bonds=[[bond[0].index, bond[1].index] for bond in pdb.topology.bonds()],
            iterations=[i for i, _, _, _ in path], energies_kj=[float(e) for _, e, _, _ in path],
            volumes=[v[0] for _, _, v, _ in path], coordinates=[np.round(x, 3).tolist() for _, _, _, x in path])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdbs", type=Path, required=True, help="pdb_tarfiles/ with train.tar, val.tar, test.tar")
    parser.add_argument("--shards", type=Path, nargs="+", required=True, help="training webdataset shards")
    parser.add_argument("--subsampled", type=Path, required=True, help="trajectories_subsampled/ with val/ and test/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=200, help="training peptides without a clash to minimize")
    parser.add_argument("--example", default="FPL", help="training peptide whose minimization path is kept")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--no-minimize", action="store_true", help="skip the OpenMM minimization")
    parser.add_argument("--yfphagyt", nargs="+", action="append", default=[], metavar=("LABEL", "NPZ"),
                        help="label and YFPHAGYT rollout NPZs (repeatable)")
    args = parser.parse_args()

    # Starting structures: one record per proline.
    lengths, layouts, texts, records = {}, {}, {}, []
    for split in SPLITS:
        with tarfile.open(args.pdbs / f"{split}.tar") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".pdb")):
                    continue
                key = (split, Path(member.name).stem)
                texts[key] = tar.extractfile(member).read().decode()
                lines = [line for line in texts[key].splitlines() if line.startswith(("ATOM", "HETATM"))]
                atoms = [(int(line[22:26]), line[17:20].strip(), line[12:16].strip()) for line in lines]
                xyz = np.asarray([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])
                index = {(r, name): i for i, (r, _, name) in enumerate(atoms)}
                names = {r: residue for r, residue, _ in atoms}
                lengths[key], layouts[key] = len(names), (index, len(atoms))
                for position, residue in enumerate(sorted(names), start=1):
                    if names[residue] != "PRO":
                        continue
                    ring = np.asarray([xyz[index[(residue, name)]] for name in PRO_RING])
                    side = [i for i, (r, _, name) in enumerate(atoms)
                            if r == residue - 1 and name not in BACKBONE and not name.startswith("H")]
                    distance = float(np.min(np.linalg.norm(xyz[side][:, None] - ring[None], axis=2))) if side else np.inf
                    records.append(dict(split=split, sequence=key[1], length=len(names), position=position,
                                        residue=residue, prev=names.get(residue - 1, "^"), next=names.get(residue + 1, "$"),
                                        min_distance=round(distance, 3), clash=distance < 1.0,
                                        start_handedness=int(handedness(xyz[None], index, residue)[0]),
                                        frames=0, d_frames=0, failed_frames=0))
    by_peptide = defaultdict(list)
    for record in records:
        by_peptide[(record["split"], record["sequence"])].append(record)

    def count(record: dict, signs: np.ndarray) -> None:
        record["frames"] += int(signs.size)
        record["d_frames"] += int(np.sum(signs < 0))
        record["failed_frames"] += int(np.sum(signs == 0))

    # Dataset frames: training shards ({SEQ}_{TIME}.bin, float32 xyz) and every 100th val/test frame.
    example_frame = (None, None)
    for shard in args.shards:
        with tarfile.open(shard) as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".bin")):
                    continue
                sequence = Path(member.name).name.rsplit("_", 1)[0]
                if not by_peptide.get(("train", sequence)) and sequence != args.example:
                    continue
                index, n_atoms = layouts[("train", sequence)]
                frame = np.frombuffer(tar.extractfile(member).read(), dtype="<f4").reshape(1, n_atoms, 3).astype(np.float64)
                if sequence == args.example and example_frame[0] is None:
                    example_frame = (f"{shard.name}:{member.name}", frame[0])
                for record in by_peptide[("train", sequence)]:
                    count(record, handedness(frame, index, record["residue"]))
    for (split, sequence), prolines in by_peptide.items():
        if split != "train":
            path = args.subsampled / split / f"{lengths[(split, sequence)]}AA" / f"{sequence}_subsampled.npz"
            with np.load(path, allow_pickle=False) as arrays:
                frames = np.asarray(arrays["positions"])[::100].astype(np.float64)
            for record in prolines:
                count(record, handedness(frames.reshape(len(frames), -1, 3), layouts[(split, sequence)][0], record["residue"]))
    for record in records:
        record["artifact"] = record["d_frames"] > 0

    splits = []
    for split in SPLITS:
        for length in sorted({n for (s, _), n in lengths.items() if s == split}) + ["all"]:
            keys = [key for key, n in lengths.items() if key[0] == split and length in (n, "all")]
            with_pro = [key for key in keys if by_peptide.get(key)]
            pro = [record for key in with_pro for record in by_peptide[key]]
            splits.append(dict(split=split, length=length, peptides=len(keys), with_proline=len(with_pro),
                               with_d_proline=sum(any(r["artifact"] for r in by_peptide[key]) for key in with_pro),
                               prolines=len(pro), d_prolines=sum(r["artifact"] for r in pro)))

    # Minimization of every clashing peptide, evenly spaced clean training controls and the example.
    clashing = {(r["split"], r["sequence"]) for r in records if r["clash"]}
    clean = sorted({(r["split"], r["sequence"]) for r in records if r["split"] == "train"} - clashing)
    controls = set(clean[:: max(1, len(clean) // args.controls)][: args.controls])
    cases, minimization = [], None
    if not args.no_minimize:
        wanted = sorted(clashing | controls | {("train", args.example)})
        jobs = [(split, sequence, texts[(split, sequence)], (split, sequence) == ("train", args.example))
                for split, sequence in wanted]
        with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
            results = pool.map(minimize, jobs, chunksize=4)
        by_key = {(r["split"], r["sequence"], r["residue"]): r for r in records}
        for result in results:
            for k, residue in enumerate(result["prolines"]):
                audit = by_key[(result["split"], result["sequence"], residue)]
                cases.append(dict(split=result["split"], sequence=result["sequence"], residue=residue,
                                  prev=audit["prev"], overlap=audit["clash"], dataset_d=audit["artifact"],
                                  start_d=result["start"][k] < 0, minimized_d=result["minimized"][k] < 0,
                                  flip_iteration=result["flip_iteration"][k]))
        example = next(result["path"] for result in results if "path" in result)
        volumes = example["volumes"]
        member, frame = example_frame
        if frame is not None:  # nm to angstrom, judged by the N-CA bond of the first proline
            index = layouts[("train", args.example)][0]
            pro = next(r for r, residue, _ in example["names"] if residue == "PRO")
            frame = frame * (10.0 if np.linalg.norm(frame[index[(pro, "CA")]] - frame[index[(pro, "N")]]) < 0.3 else 1.0)
        minimization = dict(
            prolines=len(cases), agree=sum(c["dataset_d"] == c["minimized_d"] for c in cases),
            overlap_peptides=len(clashing), control_peptides=len(controls),
            summary=[dict(overlap=o, dataset_d=d, minimized_d=m, prolines=sum(
                (c["overlap"], c["dataset_d"], c["minimized_d"]) == (o, d, m) for c in cases))
                for o, d, m in sorted({(c["overlap"], c["dataset_d"], c["minimized_d"]) for c in cases})],
            example=dict(sequence=args.example, start_energy_kj=example["energies_kj"][0],
                         steps=example["iterations"][-1],
                         first_cross_step=next((i for i, v in zip(example["iterations"], volumes) if v < 0), None),
                         flip_step=example["iterations"][max((i for i, v in enumerate(volumes) if v >= 0), default=-1) + 1]
                         if volumes[-1] < 0 else None,
                         training_frame=member,
                         training_frame_coordinates=None if frame is None else np.round(frame, 3).tolist(),
                         path=example))

    # Training prolines by the residue before them (Table tab:dpro-prev) and the Phe-Pro rule.
    train = [r for r in records if r["split"] == "train"]
    by_prev = []
    for prev in sorted({r["prev"] for r in train}):
        group = [r for r in train if r["prev"] == prev]
        minimized = [c for c in cases if c["split"] == "train" and c["prev"] == prev]
        by_prev.append(dict(prev=prev, prolines=len(group), overlap=sum(r["clash"] for r in group),
                            d_prolines=sum(r["artifact"] for r in group), minimized=len(minimized),
                            minimized_d=sum(c["minimized_d"] for c in minimized),
                            agree=sum(c["minimized_d"] == c["dataset_d"] for c in minimized)))
    by_prev.sort(key=lambda row: (-row["d_prolines"], -row["overlap"], row["prev"]))
    phe = next((row for row in by_prev if row["prev"] == "PHE"), None)
    d_total = sum(row["d_prolines"] for row in by_prev)
    rule = phe and dict(precision=phe["d_prolines"] / phe["prolines"],
                        recall=phe["d_prolines"] / d_total if d_total else None)

    # YFPHAGYT samples: stereo pattern relative to the SMILES; rollout atoms follow the PDB atom order.
    yfphagyt = []
    if args.yfphagyt:
        from pbg.core.stereo import (
            compile_stereo_spec,
            configured_stereo_key,
            stereo_keys,
        )

        pro_alpha = layouts[("test", "YFPHAGYT")][0][(3, "CA")]
        for label, *paths in args.yfphagyt:
            for path in paths:
                with np.load(path, allow_pickle=False) as arrays:
                    positions, smiles = np.asarray(arrays["positions"]), str(np.asarray(arrays["smiles"]).item())
                spec = compile_stereo_spec(smiles)
                centers = [int(c) for c in np.asarray(spec.atoms)[:, 0]]
                full, pro = (1 << len(centers)) - 1, 1 << centers.index(pro_alpha)
                config, resolved = (np.asarray(v) for v in stereo_keys(spec, positions[:, :, 1:4]))
                diff = np.where(resolved == full, config ^ configured_stereo_key(smiles, spec), -1)
                counts = {name: int(np.count_nonzero(diff == key)) for name, key in
                          (("reference", pro), ("reference_mirror", full ^ pro), ("smiles", 0), ("smiles_mirror", full))}
                yfphagyt.append(dict(label=label, rollout=str(path), n=int(diff.size), **counts,
                                     other=int(diff.size) - sum(counts.values()),
                                     d_pro3_share=(counts["reference"] + counts["reference_mirror"]) / diff.size,
                                     l_pro3_share=(counts["smiles"] + counts["smiles_mirror"]) / diff.size))
    labels = {row["label"]: [r for r in yfphagyt if r["label"] == row["label"]] for row in yfphagyt}

    report = dict(
        splits=splits, by_prev=by_prev, rule=rule, minimization=minimization,
        yfphagyt=dict(rollouts=yfphagyt, labels={label: dict(
            rollouts=len(rows), d_pro3_min=min(r["d_pro3_share"] for r in rows),
            d_pro3_max=max(r["d_pro3_share"] for r in rows), l_pro3_max=max(r["l_pro3_share"] for r in rows))
            for label, rows in labels.items()}),
        failed_frames=sum(r["failed_frames"] for r in records),
        partial=[r for r in records if 0 < r["d_frames"] < r["frames"]],
        start_structure_d=sum(r["start_handedness"] < 0 for r in records),
        artifacts_outside_train=[r for r in records if r["split"] != "train" and r["artifact"]],
        cases=cases, prolines=records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    summary = {key: report[key] for key in ("splits", "by_prev", "rule", "failed_frames", "start_structure_d")}
    summary["minimization"] = minimization and {key: value for key, value in minimization.items() if key != "example"}
    summary["yfphagyt"] = report["yfphagyt"]["labels"]
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
