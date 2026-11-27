"""Water placement baselines of the microsolvation task: AmberTools tleap and ORCA SOLVATOR.

Both place ``k`` TIP3P waters around fixed peptide conformers:

* ``tleap``: ``solvateBox`` with a TIP3P box, then the ``k`` waters nearest to the solute
  (``pbg.data.subsample.subsample_solvent``, which keeps the tleap order of the kept waters).
* ``orca``: ORCA 6 ``%SOLVATOR`` in stochastic cluster mode (GFN-xTB, ALPB water, fixed solute)
  with ``nsolv = k``, so the waters keep their addition order.

The programs come from the environment: tleap from ``$AMBERHOME/bin`` (default
``~/.local/ambertools``) or ``PATH``, ORCA from ``OPI_ORCA`` or ``PATH``, and xtb from ``XTBEXE``,
``PATH`` or ``~/.local/bin``. Per-frame work folders go to ``PBG_SOLVATE_TMP`` (node-local disk when
many workers run) or the system temp folder.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from typing import Literal

import numpy as np
from rdkit.Chem import GetPeriodicTable

from pbg.core.peptide import sequence_to_pdb_block
from pbg.data.subsample import subsample_solvent

__all__ = ["place_waters"]


def place_waters(
    solute: np.ndarray,
    sequence: str,
    k_waters: int,
    method: Literal["tleap", "orca"],
    *,
    padding_angstrom: float = 10.0,
    pal_workers: int = 1,
    timeout_s: int | None = None,
) -> np.ndarray:
    """Place ``k_waters`` waters around each solute frame; returns Zxyz frames (nm), solute first.

    ``solute`` holds Zxyz frames in nm in the ff14SB atom order of ``sequence``. The result has shape
    ``(frames, solute atoms + 3 k_waters, 4)`` with OHH waters. Both programs write the solute
    coordinates rounded, so callers that need the exact solute put it back. ``padding_angstrom`` is
    the tleap box padding; ``pal_workers`` and the per-frame ``timeout_s`` apply to ORCA. A failed
    ORCA frame keeps its work folder as ``orca_failed_frame<index>`` in the work-folder root.
    """
    from openmm import app, unit

    topology = app.PDBFile(StringIO(sequence_to_pdb_block(sequence))).topology
    atomic_numbers = np.asarray([atom.element.atomic_number for atom in topology.atoms()], dtype=np.float64)
    n_solute = atomic_numbers.size
    solute = np.asarray(solute, dtype=np.float64)
    if solute.ndim != 3 or solute.shape[1:] != (n_solute, 4) or np.any(solute[:, :, 0] != atomic_numbers):
        raise ValueError(f"solute must be Zxyz frames in the ff14SB atom order of {sequence}, got {solute.shape}")
    if not np.isfinite(solute).all():
        raise ValueError("non-finite solute coordinates")
    if method not in ("tleap", "orca") or k_waters <= 0:
        raise ValueError(f"need method 'tleap' or 'orca' and k_waters > 0, got {method!r} and {k_waters}")

    workspace = Path(os.environ.get("PBG_SOLVATE_TMP", tempfile.gettempdir()))
    workspace.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if method == "tleap":
        amberhome = Path(env.get("AMBERHOME", Path.home() / ".local" / "ambertools"))
        env["PATH"] = f"{amberhome / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        program = shutil.which("tleap", path=env["PATH"])
    else:
        # Without XTBEXE, SOLVATOR looks for xtb next to the orca binary.
        local_xtb = Path.home() / ".local" / "bin" / "xtb"
        xtb = env.get("XTBEXE") or shutil.which("xtb") or (str(local_xtb) if local_xtb.is_file() else None)
        if xtb is not None:
            env["XTBEXE"] = xtb
        program = env.get("OPI_ORCA") or shutil.which("orca")
    if not program:
        raise FileNotFoundError(f"{method} executable not found (AMBERHOME, OPI_ORCA or PATH)")
    table = GetPeriodicTable()

    frames = []
    for index, frame in enumerate(solute):
        with tempfile.TemporaryDirectory(prefix=f"pbg_solvate_{method}_", dir=workspace) as folder:
            work = Path(folder)
            if method == "tleap":
                with (work / "solute.pdb").open("w") as handle:
                    app.PDBFile.writeFile(topology, unit.Quantity(frame[:, 1:4], unit.nanometer), handle, keepIds=True)
                (work / "leap.in").write_text(
                    "source leaprc.protein.ff14SB\n"
                    "source leaprc.water.tip3p\n"
                    "addPdbAtomMap {{ H H1 }}\n"
                    "mol = loadpdb solute.pdb\n"
                    f"solvateBox mol TIP3PBOX {float(padding_angstrom):.8f}\n"
                    "saveamberparm mol solvated.prmtop solvated.inpcrd\n"
                    "savepdb mol solvated.pdb\n"
                    "quit\n"
                )
                result = subprocess.run([program, "-f", "leap.in"], cwd=work, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
                if result.returncode != 0 or not (work / "solvated.pdb").is_file():
                    raise RuntimeError(f"tleap failed on frame {index} (exit {result.returncode}):\n{result.stdout[-2000:]}")
                pdb = app.PDBFile(str(work / "solvated.pdb"))
                atoms = list(pdb.topology.atoms())
                water = np.asarray([atom.residue.name.upper() in {"WAT", "HOH", "TIP3P"} for atom in atoms])
                if water[:n_solute].any() or not water[n_solute:].all():
                    raise RuntimeError(f"tleap frame {index}: expected {n_solute} solute atoms, then only waters")
                placed = np.concatenate([
                    np.asarray([[atom.element.atomic_number] for atom in atoms], dtype=np.float64),
                    np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64),
                ], axis=1)
                placed = subsample_solvent(placed, n_solute=n_solute, k=k_waters)
            else:
                pal = f"%pal\n    nprocs {int(pal_workers)}\nend\n\n" if pal_workers > 1 else ""
                atom_lines = "\n".join(
                    f"{table.GetElementSymbol(int(round(float(atom[0])))):2s} "
                    f"{float(atom[1]) * 10.0:20.12f} {float(atom[2]) * 10.0:20.12f} {float(atom[3]) * 10.0:20.12f}"
                    for atom in frame
                )
                (work / "pbg_solvate.inp").write_text(
                    "! XTB ALPB(WATER)\n"
                    f"{pal}"
                    "%solvator\n"
                    f"    nsolv {int(k_waters)}\n"
                    "    droplet False\n"
                    "    fixsolute True\n"
                    "    clustermode stochastic\n"
                    "end\n\n"
                    "%output\n"
                    "    jsonpropfile True\n"
                    "    jsongbwfile True\n"
                    "end\n\n"
                    "* xyz 0 1\n"
                    f"{atom_lines}\n"
                    "*\n",
                    encoding="utf-8",
                )
                output = work / "pbg_solvate.solvator.xyz"
                try:
                    with (work / "pbg_solvate.out").open("w", encoding="utf-8") as log:
                        result = subprocess.run([program, "pbg_solvate.inp"], cwd=work, env=env, stdout=log,
                                                stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
                    if result.returncode != 0 or not output.is_file():
                        tail = (work / "pbg_solvate.out").read_text(errors="replace")[-2000:]
                        raise RuntimeError(f"ORCA failed on frame {index} (exit {result.returncode}, "
                                           f"XTBEXE={env.get('XTBEXE', '')!r}):\n{tail}")
                except (RuntimeError, subprocess.TimeoutExpired):
                    keep = workspace / f"orca_failed_frame{index}"
                    shutil.rmtree(keep, ignore_errors=True)
                    shutil.copytree(work, keep)
                    raise
                lines = output.read_text(encoding="utf-8").splitlines()
                rows = [line.split() for line in lines[2:] if line.strip()]
                if len(rows) != int(lines[0].strip()) or any(len(row) < 4 for row in rows):
                    raise RuntimeError(f"malformed ORCA SOLVATOR output on frame {index}")
                placed = np.concatenate([
                    np.asarray([[table.GetAtomicNumber(row[0])] for row in rows], dtype=np.float64),
                    np.asarray([row[1:4] for row in rows], dtype=np.float64) / 10.0,
                ], axis=1)
        if placed.shape != (n_solute + 3 * k_waters, 4):
            raise RuntimeError(f"frame {index}: {method} gave {placed.shape[0]} atoms, expected {n_solute + 3 * k_waters}")
        frames.append(placed)
    return np.stack(frames)
