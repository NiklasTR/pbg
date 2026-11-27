"""Many Peptides metrics of the paper (Table 1, the starred table, the D-proline and box-volume appendices).

Table 1 drops the top 0.2% of importance weights. rESS is the end-to-end Kish rESS; E-W2 and TICA-W2
follow the legacy estimator on 1024 resampled frames (seeds 42, 43, 44); T-W2 is the exact backbone
torus W2 between 10^4 frames resampled from the weights (seed 42) and all 10^4 reference frames.
The starred metrics keep every weight: rESS*, the importance-weighted energy W2* and the sliced
global torus W2*. Table 1 cells average each system over the sampling seeds and then average the
systems; starred cells take the median over systems within each seed and then average the seed
medians. Both omit YFPHAGYT.

Each estimator forms its log weights with the same floating-point operations as the published
evaluation, so the values match the paper bitwise. The one exception is the D-proline appendix rESS,
whose published job divided by kT instead of multiplying by beta (a difference below 1e-16).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K
from pbg.core.ensemble import Trajectory
from pbg.core.stereo import (
    compile_stereo_spec,
    configured_stereo_key,
    intended_stereo_mask,
    stereo_keys,
)
from pbg.estimator import (
    backbone_torsions,
    global_torsions,
    importance_weights,
    relative_ess,
    sliced_torus_w2,
    tica_projection,
    w2,
    w2_1d,
)

__all__ = [
    "BOX_VOLUME_METRICS",
    "EXCLUDED_SYSTEMS",
    "STARRED_METRICS",
    "TABLE1_METRICS",
    "Rollout",
    "box_volume_floor",
    "box_volume_log_terms",
    "box_volume_metrics",
    "box_volume_rotations",
    "reference_stereo_smiles",
    "starred_metrics",
    "summarize",
    "table1_metrics",
]

TABLE1_METRICS = {
    "end_to_end_ress_drop_0_2pct": "rESS",
    "uniform_empirical_energy_w2_kbt_drop_0_2pct": "E-W2",
    "uniform_empirical_backbone_torus_w2_rad_drop_0_2pct_10k": "T-W2",
    "uniform_empirical_tica2_w2_drop_0_2pct": "TICA-W2",
}
STARRED_METRICS = {
    "end_to_end_ress": "rESS*",
    "importance_weighted_energy_w2_kbt": "E-W2*",
    "importance_weighted_global_torus_w2_sliced_rad": "T-W2*",
}
BOX_VOLUME_METRICS = {
    **{f"{metric}_box_volume_corrected": f"{label} box-corrected" for metric, label in STARRED_METRICS.items()},
    "box_volume_partial_fit_fraction": "F<1 fraction",
}
# PBG draws almost no conformers with the intended stereochemistry of YFPHAGYT (Appendix D.5).
EXCLUDED_SYSTEMS = ("YFPHAGYT",)
LEGACY_SEEDS = (42, 43, 44)
LEGACY_FRAMES = 1024
TORUS_FRAMES = 10_000
TORUS_SEED = 42
PERIOD = 2.0 * np.pi
SLICED = {"max_order": 2, "max_entry": 2, "max_nnz": 2, "max_slices": 4096, "n_random": 256}
SLICED_SEED = 0
# Training box of core_bin4096 and joint_bin4096: 4096 coordinate bins of 0.8 pm.
BOX_NM = 4096 * 0.8 / 1000.0
FIT_ATTEMPTS = 16  # pbg.core.augmenter._ROTATION_ATTEMPTS
ROTATION_SEED = 20261123


@dataclass(frozen=True, slots=True)
class Rollout:
    """One generated sample set: positions (frames, atoms, Zxyz in nm), energy (kJ/mol) and the
    log-likelihood of each generated conformer."""

    positions: np.ndarray
    energy: np.ndarray
    logq: np.ndarray
    smiles: str
    temperature: float
    frame_valid: np.ndarray | None
    n_reflected: int

    @classmethod
    def load(cls, path: Path, *, reflect: bool, smiles: str | None = None) -> Rollout:
        """Read a rollout NPZ.

        ``reflect`` mirrors fully resolved, fully inverted stereoisomers into the intended one (PBG
        rows; coordinates only, the generated logq stays). ``smiles`` replaces the intended
        stereochemistry of the stored SMILES (same atoms and order). The sampler's ``frame_valid``
        marks the intended stereochemistry of the stored frames, so it is dropped once reflection or
        ``smiles`` changes it.
        """
        with np.load(path, allow_pickle=False) as arrays:
            positions = np.asarray(arrays["positions"], dtype=np.float64)
            energy = np.asarray(arrays["energy"], dtype=np.float64).reshape(-1)
            logq = np.asarray(arrays["generator_log_likelihood"], dtype=np.float64).reshape(-1)
            stored = str(np.asarray(arrays["smiles"]).item())
            temperature = float(np.asarray(arrays["temperature"]).reshape(-1)[0])
            frame_valid = np.asarray(arrays["frame_valid"], dtype=bool) if "frame_valid" in arrays.files else None
        intended = stored if smiles is None else smiles
        if intended.replace("@", "") != stored.replace("@", ""):
            raise ValueError(f"{path}: SMILES {intended!r} does not match the rollout atoms {stored!r}")
        reflected = np.zeros(len(positions), dtype=bool)
        spec = compile_stereo_spec(intended) if reflect else None
        # A molecule without stereocenters has no mirror image to fold.
        if spec is not None and spec.atoms.size:
            config, resolved = (np.asarray(value) for value in stereo_keys(spec, positions[:, :, 1:4]))
            full = (1 << int(spec.atoms.shape[0])) - 1
            reflected = (resolved == full) & (config == (configured_stereo_key(intended, spec) ^ full))
            positions[reflected, :, 1:4] *= -1.0
        if reflected.any() or intended != stored:
            frame_valid = None
        return cls(positions, energy, logq, intended, temperature, frame_valid, int(reflected.sum()))


def table1_metrics(rollout: Rollout, reference: Trajectory, topology: Any) -> dict[str, float]:
    """Table 1 metrics of one system (top 0.2% of the weights dropped).

    E-W2 and TICA-W2 follow the legacy estimator: for each of the seeds 42, 43 and 44, draw
    min(reference frames, positive weights) energies and 1024 frames from the weights, and 1024
    reference frames without replacement. E-W2 compares the energy draw with the first reference
    energies, TICA-W2 the frame draws. Both are absent when fewer than 1024 frames keep a positive
    weight. The published T-W2 job did not apply the sampler's ``frame_valid``; the other metrics do.
    """
    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * rollout.temperature)
    log_weight = -beta * rollout.energy - rollout.logq
    finite = (
        intended_stereo_mask(rollout.smiles, rollout.positions[:, :, 1:4])
        & np.isfinite(rollout.energy)
        & np.isfinite(log_weight)
    )
    indices = np.flatnonzero(finite if rollout.frame_valid is None else finite & rollout.frame_valid)
    weights = importance_weights(log_weight[indices], drop_top_percent=0.2)
    metrics = {"end_to_end_ress_drop_0_2pct": relative_ess(weights, n_proposals=rollout.energy.size)}

    positive = np.flatnonzero(weights > 0.0)
    if positive.size >= LEGACY_FRAMES and reference.length >= LEGACY_FRAMES:
        probabilities = weights[positive] / float(weights[positive].sum())
        kbt = 1.0 / beta
        target_energy = rollout.energy[indices] / float(kbt)
        reference_energy = np.asarray(reference.energy, dtype=np.float64) / float(kbt)
        samples, reference_samples = rollout.positions[indices], np.asarray(reference.samples, dtype=np.float64)
        energy_w2, tica_w2 = [], []
        for seed in LEGACY_SEEDS:
            rng = np.random.default_rng(seed)
            energy_draw = rng.choice(positive, size=min(reference.length, positive.size), replace=True, p=probabilities)
            frame_draw = rng.choice(positive, size=LEGACY_FRAMES, replace=True, p=probabilities)
            reference_draw = rng.choice(reference.length, size=LEGACY_FRAMES, replace=False)
            energy_w2.append(w2_1d(reference_energy[: energy_draw.size], target_energy[energy_draw]))
            tica_w2.append(w2(
                tica_projection(reference_samples[reference_draw], reference),
                tica_projection(samples[frame_draw], reference),
            ))
        metrics["uniform_empirical_energy_w2_kbt_drop_0_2pct"] = float(np.mean(energy_w2))
        metrics["uniform_empirical_tica2_w2_drop_0_2pct"] = float(np.mean(tica_w2))

    torus_weights = importance_weights(log_weight[finite], drop_top_percent=0.2)
    support = np.flatnonzero(torus_weights > 0.0)
    draws = np.random.default_rng(TORUS_SEED).choice(
        support, size=TORUS_FRAMES, replace=True, p=torus_weights[support] / torus_weights[support].sum()
    )
    right = backbone_torsions(reference.samples, topology)
    if right.shape[0] != TORUS_FRAMES:
        raise ValueError(f"T-W2 needs {TORUS_FRAMES} reference frames, got {right.shape[0]}")
    metrics["uniform_empirical_backbone_torus_w2_rad_drop_0_2pct_10k"] = w2(
        backbone_torsions(rollout.positions[finite], topology)[draws], right, period=PERIOD
    )
    return metrics


def starred_metrics(
    rollout: Rollout, reference: Trajectory, *, log_weight_offset: np.ndarray | None = None
) -> dict[str, float]:
    """Starred metrics of one system with every weight kept (Table sota_modern).

    ``log_weight_offset`` (one value per frame) is added to the log weights, as in the box-volume
    correction. If the sampler marked no frame valid, rESS* and E-W2* fall back to the frames with
    the intended stereochemistry and a finite log weight. T-W2* never applies ``frame_valid``.
    """
    offset = 0.0 if log_weight_offset is None else np.asarray(log_weight_offset, dtype=np.float64).reshape(-1)
    stereo = intended_stereo_mask(rollout.smiles, rollout.positions[:, :, 1:4])
    kbt = GAS_CONSTANT_KJ_MOL_K * rollout.temperature
    finite = stereo & np.isfinite(rollout.energy) & np.isfinite(rollout.logq)
    for valid in ([] if rollout.frame_valid is None else [finite & rollout.frame_valid]) + [finite]:
        log_weight = np.where(valid, -rollout.energy / kbt - rollout.logq + offset, np.nan)
        if np.isfinite(log_weight).any():
            break
    weights = importance_weights(log_weight)
    energy_w2 = w2_1d(
        rollout.energy[valid] / kbt,
        np.asarray(reference.energy, dtype=np.float64).reshape(-1) / kbt,
        x_weights=weights[valid],
    )

    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * rollout.temperature)
    log_weight = -beta * rollout.energy - rollout.logq + offset
    valid = stereo & np.isfinite(log_weight) & np.isfinite(rollout.energy)
    torus = sliced_torus_w2(
        global_torsions(rollout.positions[valid], rollout.smiles),
        global_torsions(reference.samples, rollout.smiles),
        x_weights=importance_weights(log_weight[valid]),
        **SLICED,
        rng=np.random.default_rng(SLICED_SEED),
    )
    metrics = {
        "end_to_end_ress": relative_ess(weights, n_proposals=rollout.energy.size),
        "importance_weighted_energy_w2_kbt": float(energy_w2),
        "importance_weighted_global_torus_w2_sliced_rad": float(torus),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"nonfinite starred metric {metrics}")
    return metrics


def box_volume_rotations() -> np.ndarray:
    """The fixed set of 1024 uniform rotations that estimates the fitting fraction F."""
    from scipy.spatial.transform import Rotation

    return Rotation.random(1024, random_state=ROTATION_SEED).as_matrix()


def box_volume_log_terms(xyz: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame log V and log F of the box-volume correction (appendix).

    Training rotates each centred frame uniformly among the orientations whose axis extents fit the
    box of width ``BOX_NM`` and translates it uniformly inside the box, over a volume
    V = prod_a (BOX_NM - extent_a). A perfect model of these frames therefore gets weights
    proportional to F V under the plain target, where F is the fraction of rotations that fit.
    F is 1 whenever the largest interatomic distance is below the box width. ``xyz`` is in nm,
    frames x atoms x 3, as oriented.
    """
    extent = np.ptp(xyz, axis=1)
    log_v = np.log(np.clip(BOX_NM - extent, 1e-9, None)).sum(axis=1)
    log_f = np.zeros(len(xyz))
    for start in range(0, len(xyz), 128):
        chunk = xyz[start:start + 128]
        diameter = np.sqrt(((chunk[:, :, None, :] - chunk[:, None, :, :]) ** 2).sum(-1)).max(axis=(1, 2))
        for index in np.flatnonzero(diameter >= BOX_NM):
            fits = np.all(np.ptp(np.einsum("rij,aj->rai", rotations, chunk[index]), axis=1) < BOX_NM, axis=1)
            log_f[start + index] = np.log(max(int(fits.sum()), 1) / len(rotations))
    return log_v, log_f


def box_volume_metrics(rollout: Rollout, reference: Trajectory, rotations: np.ndarray) -> dict[str, float]:
    """Starred metrics with the box-volume corrected log weights -beta U - log q - log V - log F (appendix)."""
    log_v, log_f = box_volume_log_terms(rollout.positions[:, :, 1:4], rotations)
    metrics = starred_metrics(rollout, reference, log_weight_offset=-(log_v + log_f))
    return {
        **{f"{metric}_box_volume_corrected": value for metric, value in metrics.items()},
        "box_volume_partial_fit_fraction": float(np.mean(log_f < 0)),
    }


def box_volume_floor(
    reference: Trajectory, *, seed: int, rotations: np.ndarray, temperature_k: float = 310.0
) -> dict[str, float]:
    """Perfect-model floor of the box-volume bias on the reference MD (appendix).

    Each reference frame gets one training-like orientation: the first of up to 16 uniform rotations
    (``seed``) that fits the box. The perfect model's weights F V give the rESS ceiling and the
    energy and sliced torus W2 between the reference and its reweighted copy. The paper used the
    index of the system in the sorted 2AA, 4AA, 8AA test list without YFPHAGYT as ``seed``.
    """
    from scipy.spatial.transform import Rotation

    xyz = np.asarray(reference.samples, dtype=float)[:, :, 1:4]
    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    candidates = Rotation.random(
        len(xyz) * FIT_ATTEMPTS, random_state=np.random.default_rng(seed)
    ).as_matrix().reshape(len(xyz), FIT_ATTEMPTS, 3, 3)
    rotated = np.einsum("fkij,faj->fkai", candidates, xyz)
    fits = np.all(np.ptp(rotated, axis=2) < BOX_NM, axis=2)
    kept = fits.any(axis=1)
    log_v, log_f = box_volume_log_terms(rotated[np.arange(len(xyz)), np.argmax(fits, axis=1)][kept], rotations)
    log_w = log_v + log_f
    weights = importance_weights(log_w)
    energy = np.asarray(reference.energy, dtype=float).reshape(-1) / (GAS_CONSTANT_KJ_MOL_K * temperature_k)
    angles = global_torsions(xyz, reference.smiles)
    torus = sliced_torus_w2(
        angles[kept], angles, x_weights=weights, **SLICED, rng=np.random.default_rng(SLICED_SEED)
    )
    return {
        "box_volume_perfect_model_ress": relative_ess(weights, n_proposals=len(log_w)),
        "box_volume_floor_energy_w2_kbt": w2_1d(energy[kept], energy, x_weights=weights),
        "box_volume_floor_global_torus_w2_sliced_rad": float(torus),
        "box_volume_floor_mean_energy_shift_kbt": float(np.sum(weights * energy[kept]) - energy.mean()),
        "box_volume_log_weight_sd": float(np.std(log_w)),
        "box_volume_partial_fit_fraction": float(np.mean(log_f < 0)),
        "n_kept": int(kept.sum()),
    }


def reference_stereo_smiles(reference: Trajectory) -> str:
    """The reference SMILES with the stereochemistry of the reference MD frames (D-proline appendix).

    Every YFPHAGYT reference frame has D-Pro3, while its SMILES says L (Appendix D.5). The tag of
    each stereocenter where all frames differ from the SMILES is swapped; the result is checked
    against the frames.
    """
    spec = compile_stereo_spec(reference.smiles)
    centers = [int(center) for center in np.asarray(spec.atoms)[:, 0]]
    config, resolved = (np.asarray(value) for value in stereo_keys(spec, np.asarray(reference.samples)[:, :, 1:4]))
    if not np.all(resolved == (1 << len(centers)) - 1) or len(set(config.tolist())) != 1:
        raise ValueError("reference frames do not share one fully resolved stereoisomer")
    flips = int(config[0]) ^ configured_stereo_key(reference.smiles, spec)
    smiles = reference.smiles
    for bit, atom in enumerate(centers):
        if flips >> bit & 1:
            smiles, count = re.subn(
                rf"\[([A-Za-z]+)(@@|@)([^\]:]*):{atom + 1}\]",
                lambda match: f"[{match[1]}{'@' if match[2] == '@@' else '@@'}{match[3]}:{atom + 1}]",
                smiles,
            )
            if count != 1:
                raise ValueError(f"no unique stereo tag for atom {atom}")
    check = compile_stereo_spec(smiles)
    if [int(center) for center in np.asarray(check.atoms)[:, 0]] != centers or configured_stereo_key(
        smiles, check
    ) != int(config[0]):
        raise ValueError("reference-stereo SMILES does not match the reference frames")
    return smiles


def summarize(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Paper cells per size and metric from per-system results.

    ``results`` holds one mapping per system and sampling seed with ``sequence``, ``size``, ``seed``
    and ``metrics``. Table 1 cells: mean over systems of the per-system mean over seeds (systems with a
    value in every seed). Starred and box-volume cells: median over systems within each seed, then the
    mean of the seed medians. ``seed_min``/``seed_max`` are the extreme seed-level means or medians.
    """
    values: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for result in results:
        if result["sequence"] in EXCLUDED_SYSTEMS:
            continue
        for metric, value in result["metrics"].items():
            cell = values.setdefault((result["size"], metric), {})
            cell.setdefault(int(result["seed"]), {})[result["sequence"]] = float(value)
    labels = {**TABLE1_METRICS, **STARRED_METRICS, **BOX_VOLUME_METRICS}
    rows = []
    for (size, metric), by_seed in sorted(values.items(), key=lambda item: (item[0][0], list(labels).index(item[0][1]))):
        seeds = sorted(by_seed)
        if metric in TABLE1_METRICS:
            systems = sorted(set.intersection(*(set(by_seed[seed]) for seed in seeds)))
            per_seed = [float(np.mean([by_seed[seed][system] for system in systems])) for seed in seeds]
            value = float(np.mean([np.mean([by_seed[seed][system] for seed in seeds]) for system in systems]))
        else:
            systems = sorted(set.union(*(set(by_seed[seed]) for seed in seeds)))
            per_seed = [float(np.median(list(by_seed[seed].values()))) for seed in seeds]
            value = float(np.mean(per_seed))
        rows.append({
            "size": size, "metric": metric, "label": labels[metric], "value": value,
            "seed_min": min(per_seed), "seed_max": max(per_seed), "n_systems": len(systems), "n_seeds": len(seeds),
        })
    return rows
