r"""Joint Bayesian scaling model of training loss and sampling quality.

Paper, Appendix "Scaling Model on Many Peptides" (Eqs. scaling-resource-law and
scaling-model-adapters, prior table, compute-tilted posterior), "Sensitivity of the
Scaling Model" (Tables tab:scaling-ablation and tab:scaling-ablation-loso) and
"Scaling Model on MDQM9-nc" (Tables tab:mdqm9-scaling-ablation and -loso).

For checkpoint $i$ (non-embedding parameters $N_i$, loss tokens $D_i$, bin count
$b_i$) and test system $j$ (atom count $n_j$), with $N_{9i}=N_i/10^9$,
$D_{9i}=D_i/10^9$ and $x_j=\log(n_j/65)$:

$$G_i=\frac{G_0}{2}\left[(N_{9i}\sqrt{r})^{-\alpha}+(D_{9i}/\sqrt{r})^{-\alpha}\right],\qquad L_i\sim t_{\nu_L}(E+G_i+\delta_L[b_i],\sigma_L),$$

$$\eta_{mij}=a_m+p_m x_j+k_m\log G_i+\delta_m[b_i],\qquad Y_{mij}\sim t_{\nu_m}(\mu_{mij},\sigma_m),$$

with $\mu_{Eij}=\eta_{Eij}$, $\mu_{Sij}=\eta_{Sij}$ and
$\mu_{Rij}=\log(10^4)\,\mathrm{sigmoid}(\eta_{Rij})$ for $Y_E=\log E\text{-}W_2$,
$Y_S=\log T\text{-}SW_2$ and $Y_R=-\log\mathrm{rESS}$. Every likelihood term of
checkpoint $i$ is raised to $w_i=(6N_iD_i)^\gamma/\overline{(6ND)^\gamma}$ with
$\gamma=1/2$. The paper fit (law v23, row A4) draws 4 NUTS chains of 2000 warmup
and 2000 draws with target acceptance 0.99 and tree depth 12. The code keeps the
operation order of the implementation used for the paper, so log densities and
NUTS draws are bitwise equal on the same inputs.

Ablation rows (:data:`ABLATIONS`) change one modelling choice of the default
:class:`Variant` (row A4); D3 is the A3 fit.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import betainc
from numpyro.diagnostics import summary as diagnostic_summary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.special import expit, logit

from pbg.analysis import read_table

__all__ = ["ABLATIONS", "METRICS", "POST_EXTRAPOLATION", "PREFIXES", "SIZES", "ScalingProblem", "Variant", "analyze",
           "diagnose", "load_inputs", "metric_blocks", "model", "prepare", "sample"]

PREFIXES = ("energy", "sliced", "ress")
METRICS = ("importance_weighted_energy_w2_kbt", "importance_weighted_global_torus_w2_sliced_rad", "end_to_end_ress")
#: Reference size (85M non-embedding parameters, in billions) for $D^*/N$ when $\alpha\ne\beta$.
N_REF = 85543168 / 1e9
SIZES = {"8m": 8262016, "15m": 14947840, "30m": 29895168, "49m": 49164928, "85m": 85543168}
#: MDQM9-nc dataset constants: held-out size, atom reference and floor-prior offset (as in v22).
MDQM9_EXTRAPOLATION_SIZE = 159296256
MDQM9_ATOM_REFERENCE = 18.0
MDQM9_FLOOR_OFFSET = 3.5927834510803223 - 3.55
MDQM9_MINIMUM_TASKS = 90
#: The paper's MDQM9-nc cohort (n28) drops runs shorter than the shortest Many Peptides fitting run.
MDQM9_COHORT_MIN_STEPS = {"n36": 0, "n28": 3694}
#: Post-extrapolation law (v23_post_extrapolation, rate appendix figure): row A4 with these three
#: held-out Many Peptides checkpoints moved into the fit; the 160M checkpoint stays held out.
POST_EXTRAPOLATION = ("dipper_85m_core_bin4096_ws_scaling_d1280n_pristine_from640_v1_step_1150232",
                      "dipper_320m_core_bin4096_completep_many_pristine_d160n_v1_step_618977",
                      "dipper_640m_core_bin4096_completep_many_lumi_n4_b16_d48n_step_83846")


@dataclass(frozen=True)
class Variant:
    """One row of the scaling-model ablation. The defaults are the paper model (row A4).

    responses: 'joint' fits NLL and the three metric adapters; 'nll' fits only the NLL.
    tied: $\\beta=\\alpha$. prior: 'allocation' puts $r\\sim\\mathcal N^+(20,10)$ on $D^*/N$;
    'independent' puts independent priors on $\\log A$ and $\\log B$. gamma is the compute-tilting power.
    system_effects / lineage_effects add hierarchical test-system / training-lineage effects (A5, A6).
    held_out_size removes every fitting checkpoint of that size (leave one size out).
    atom_form, with $x=\\log(n/65)$: 'power' $p x$ (paper); 'exponential' $c(n/65-1)$;
    'both' $p x+q(n/65-1-x)$; 'extensive' $p x$ with $p_E=p_R=1$.
    held_out_min_residues removes peptides with at least that many residues from the fit.
    ress_censor_ess $c$ right-censors $-\\log\\mathrm{rESS}$ at $\\log(10^4/c)$ where $\\mathrm{ESS}\\le c$.
    MDQM9 only: terminal_nll 'local_linear' uses a low-LR local linear endpoint (N1);
    fit_extrapolation moves the 160M checkpoints into the fit (P1).
    dataset, atom_reference and floor_loc are dataset constants set by :func:`prepare`.
    """

    responses: str = "joint"
    tied: bool = True
    prior: str = "allocation"
    gamma: float = 0.5
    system_effects: bool = False
    lineage_effects: bool = False
    held_out_size: int | None = None
    atom_form: str = "power"
    held_out_min_residues: int | None = None
    ress_censor_ess: float | None = None
    terminal_nll: str = "logged"
    fit_extrapolation: bool = False
    dataset: str = "many_peptides"
    atom_reference: float = 65.0
    floor_loc: float = 3.55


ABLATIONS = {
    "A0": Variant(responses="nll", tied=False, prior="independent", gamma=0.0),
    "A1": Variant(tied=False, prior="independent", gamma=0.0),
    "A2": Variant(prior="independent", gamma=0.0),
    "A3": Variant(gamma=0.0),  # also D3
    "A4": Variant(),  # paper model
    "A5": Variant(system_effects=True),
    "A6": Variant(system_effects=True, lineage_effects=True),
    "D1": Variant(tied=False),
    "D2": Variant(prior="independent"),
    **{"A4_loso_" + k: Variant(held_out_size=v) for k, v in SIZES.items()},
    **{"A6_loso_" + k: Variant(system_effects=True, lineage_effects=True, held_out_size=v) for k, v in SIZES.items()},
    "F1": Variant(atom_form="exponential"),
    "F2": Variant(atom_form="both"),
    "F0_hold8": Variant(held_out_min_residues=8),
    "F1_hold8": Variant(atom_form="exponential", held_out_min_residues=8),
    "F2_hold8": Variant(atom_form="both", held_out_min_residues=8),
    "F3": Variant(atom_form="extensive"),
    "F3_hold8": Variant(atom_form="extensive", held_out_min_residues=8),
    "C1": Variant(ress_censor_ess=1.1),
    "C2": Variant(ress_censor_ess=2.0),
    "A4_nll": Variant(responses="nll"),
    "N1": Variant(terminal_nll="local_linear"),  # MDQM9 only
    "P1": Variant(fit_extrapolation=True),  # MDQM9 only
}
ABLATIONS.update({row + "_loso_" + k: replace(ABLATIONS[row], held_out_size=v)
                  for row in ("A0", "A1", "A2", "A3", "A5", "D1", "D2", "F1", "F2", "F3", "C1") for k, v in SIZES.items()})


@jax.custom_jvp
def student_t_log_survival(z, df):
    """$\\log(1-F_t(z))$ of the standard Student-t with df degrees of freedom (censored rESS)."""
    tail = 0.5 * betainc(0.5 * df, 0.5, df / (df + z**2))
    return jnp.log(jnp.where(z > 0, tail, 1.0 - tail))


@student_t_log_survival.defjvp
def student_t_log_survival_jvp(primals, tangents):
    # JAX betainc has no derivative in its first argument, so the df slope is a central difference.
    z, df = primals
    dz, ddf = tangents
    value = student_t_log_survival(z, df)
    h = 1e-5 * df
    slope_df = (student_t_log_survival(z, df + h) - student_t_log_survival(z, df - h)) / (2 * h)
    slope_z = -jnp.exp(dist.StudentT(df).log_prob(z) - value)
    return value, slope_z * dz + slope_df * ddf


def model(n, d, slot, nll, blocks, nbins, ref, weights, variant=None, lineage=None, nlineage=0, nsystems=0,
          system_projection=None, lineage_projection=None):
    """NumPyro model of Eqs. scaling-resource-law and scaling-model-adapters with the paper's priors.

    n, d: $N_9$ and $D_9$ of the fitting checkpoints; slot: bin index; nll: terminal training NLL;
    blocks: one dict per metric (PREFIXES order) with checkpoint index 'idx', 'atom' $x_j$, response 'y',
    'system' index, and for censored rESS 'censored'/'exact' positions; weights: $w_i$.
    With the allocation prior, $A=\\frac{G_0}{2}r^{-\\alpha/2}$ and $B=\\frac{G_0}{2}r^{\\alpha/2}$.
    """
    variant = variant or Variant()
    floor = numpyro.sample("floor", dist.Normal(variant.floor_loc, 0.15))
    log_alpha = numpyro.sample("log_alpha", dist.Normal(np.log(0.8), 0.75))
    alpha = jnp.exp(log_alpha)
    log_beta = log_alpha if variant.tied else numpyro.sample("log_beta", dist.Normal(np.log(0.8), 0.75))
    beta = jnp.exp(log_beta)
    if variant.prior == "allocation":
        ratio = numpyro.sample("allocation_ratio", dist.TruncatedNormal(loc=20.0, scale=np.sqrt(10.0), low=0.0))
        log_g0 = numpyro.sample("log_g0", dist.Normal(np.log(2) + 0.5 * (np.log(0.003) + np.log(0.05)), 1.5 / np.sqrt(2)))
        log_a = numpyro.deterministic("log_a", log_g0 - jnp.log(2.0) - 0.5 * alpha * jnp.log(ratio))
        if variant.tied:
            log_b = numpyro.deterministic("log_b", log_g0 - jnp.log(2.0) + 0.5 * alpha * jnp.log(ratio))
        else:
            # B such that D*(N_REF)/N_REF = r; reduces to the tied form when beta = alpha.
            log_b = numpyro.deterministic(
                "log_b", log_a + log_alpha - log_beta + beta * jnp.log(ratio) + (beta - alpha) * np.log(N_REF))
    else:
        log_a = numpyro.sample("log_a", dist.Normal(np.log(0.003), 1.5))
        log_b = numpyro.sample("log_b", dist.Normal(np.log(0.05), 1.5))
    g = jnp.exp(log_a) * n ** (-alpha) + jnp.exp(log_b) * d ** (-beta)
    if variant.lineage_effects:
        # One effect per training lineage (size x bin), projected orthogonal to the bin offsets.
        tau = numpyro.sample("lineage_tau", dist.HalfNormal(0.25))
        raw = numpyro.sample("lineage_raw", dist.Normal(0.0, tau).expand([nlineage]).to_event(1))
        zeta = numpyro.deterministic("lineage_effect", lineage_projection @ raw)
        g = g * jnp.exp(zeta[lineage])
    delta = numpyro.sample("loss_bin_offset", dist.Normal(0.0, 3.0).expand([nbins - 1]).to_event(1))
    offset = jnp.concatenate((delta[:ref], jnp.zeros(1), delta[ref:]))
    log_sigma = numpyro.sample("log_sigma_nll", dist.Normal(np.log(0.01), 1.0))
    log_nu = numpyro.sample("log_nu_nll", dist.Normal(np.log(10.0), 0.75))
    with numpyro.handlers.scale(scale=weights):
        numpyro.sample("nll_obs", dist.StudentT(jnp.exp(log_nu) + 2, floor + g + offset[slot], jnp.exp(log_sigma)), obs=nll)
    if variant.responses == "nll":
        return
    # Prior locations per metric: intercept a_m, atom exponent p_m (loc, scale), G-sensitivity k_m, scale sigma_m.
    for prefix, block, loc, atomloc, atomscale, kappaloc, scale in zip(
            PREFIXES, blocks, (5.0, 1.0, 4.5), (2.0, 0.7, 1.2), (0.5, 0.5, 0.8), (1.2, 0.7, 1.2), (0.5, 0.5, 1.5)):
        intercept = numpyro.sample(prefix + "_intercept", dist.Normal(loc, 2.0))
        delta = numpyro.sample(prefix + "_bin_offset", dist.Normal(0.0, 1.0).expand([nbins - 1]).to_event(1))
        offset = jnp.concatenate((delta[:ref], jnp.zeros(1), delta[ref:]))
        atom = (1.0 if variant.atom_form == "extensive" and prefix != "sliced"
                else numpyro.sample(prefix + "_atom", dist.Normal(atomloc, atomscale)))
        kappa = numpyro.sample(prefix + "_kappa", dist.Normal(kappaloc, 0.8))
        log_scale = numpyro.sample(prefix + "_log_scale", dist.Normal(np.log(scale), 0.75))
        log_df = numpyro.sample(prefix + "_log_df", dist.Normal(np.log(10.0), 0.75))
        x = block["atom"]
        covariate = jnp.exp(x) - 1.0 if variant.atom_form == "exponential" else x
        eta = intercept + offset[slot[block["idx"]]] + atom * covariate + kappa * jnp.log(g[block["idx"]])
        if variant.atom_form == "both":
            # Zero value and slope at the reference; q > 0 grows faster than any power of n.
            curvature = numpyro.sample(prefix + "_atom_curvature", dist.Normal(0.0, 2.0))
            eta = eta + curvature * (jnp.exp(x) - 1.0 - x)
        if variant.system_effects:
            # Test-system intercepts, projected to sum to zero.
            tau = numpyro.sample(prefix + "_system_tau", dist.HalfNormal(1.0 if prefix == "ress" else 0.5))
            raw = numpyro.sample(prefix + "_system_raw", dist.Normal(0.0, tau).expand([nsystems]).to_event(1))
            u = numpyro.deterministic(prefix + "_system_effect", system_projection @ raw)
            eta = eta + u[block["system"]]
        mean = jnp.log(10000.0) * jax.nn.sigmoid(eta) if prefix == "ress" else eta
        df = jnp.exp(log_df) + 2
        sigma = jnp.exp(log_scale)
        weight = weights[block["idx"]]
        y = block["y"]
        if prefix == "ress" and variant.ress_censor_ess is not None:
            # ESS <= c: y is only known to be at least log(1e4/c); the same checkpoint weight applies.
            threshold = np.log(1e4 / variant.ress_censor_ess)
            censored = block["censored"]
            with numpyro.handlers.scale(scale=weight[censored]):
                numpyro.factor(prefix + "_censored", student_t_log_survival((threshold - mean[censored]) / sigma, df))
            mean, weight, y = mean[block["exact"]], weight[block["exact"]], y[block["exact"]]
        with numpyro.handlers.scale(scale=weight):
            numpyro.sample(prefix + "_obs", dist.StudentT(df, mean, sigma), obs=y)


def lineage_of(model_id):
    """Training lineage of a checkpoint: model size and bin count share one WSD stable run."""
    match = re.match(r"dipper_(\d+m)_core_bin(\d+)_", model_id)
    if match is None:
        raise ValueError("no lineage in model id: " + model_id)
    return match.group(1) + "_" + match.group(2)


def residues_of(task_id):
    """Peptide length of a Many Peptides test task, e.g. many_peptides/test/8AA/... -> 8."""
    return int(task_id.split("/")[2].removesuffix("AA"))


def metric_blocks(checkpoints, observations, atom_reference=65.0):
    """Per-metric observation blocks: records sorted by (model_id, task_id), checkpoint index,
    $x=\\log(n/65)+\\log(65/n_\\mathrm{ref})$ and the transformed response (mean log value; negated for rESS).
    Non-positive values and rESS above 1 are dropped."""
    ids = {r["model_id"]: i for i, r in enumerate(checkpoints)}
    blocks = []
    for metric in METRICS:
        grouped = {}
        for o in observations:
            if o["model_id"] not in ids or o["metric_id"] != metric:
                continue
            if o["value"] is None or o["value"] <= 0 or (metric == "end_to_end_ress" and o["value"] > 1):
                continue
            grouped.setdefault((o["model_id"], o["task_id"]), []).append(o)
        records = []
        for (mid, task), oo in sorted(grouped.items()):
            target = float(np.mean(np.log([o["value"] for o in oo])))
            records.append({"model_id": mid, "task_id": task, "atom_count": oo[0]["atom_count"],
                            "target": -target if metric == "end_to_end_ress" else target})
        blocks.append({"metric": metric, "records": records,
                       "idx": np.array([ids[r["model_id"]] for r in records], dtype=int),
                       "atom": np.log([r["atom_count"] / 65 for r in records]) + np.log(65.0 / atom_reference),
                       "y": np.array([r["target"] for r in records])})
    return blocks


def load_inputs(data_dir, dataset="many_peptides", cohort=None):
    """Read the tidy tables of one dataset: (checkpoints, observations, starting values or None).

    many_peptides: the v23 cohort. mdqm9: the n36 cohort; cohort 'n28' (the paper's) keeps runs with
    at least 3694 optimizer steps.
    """
    data_dir = Path(data_dir)
    if dataset == "many_peptides":
        stem = "scaling_many_peptides_v23"
        initial = json.loads((data_dir / f"{stem}_init.json").read_text())["initial"]
        initial = {k: np.array(v) if isinstance(v, list) else v for k, v in initial.items()}
    else:
        stem, initial = "scaling_mdqm9_n36", None
    rows = read_table(data_dir / f"{stem}_checkpoints.csv.gz")
    observations = read_table(data_dir / f"{stem}_observations.csv.gz")
    if dataset == "mdqm9":
        minimum = MDQM9_COHORT_MIN_STEPS[cohort or "n28"]
        rows = [r for r in rows if r["step"] >= minimum]
        kept = {r["model_id"] for r in rows}
        observations = [o for o in observations if o["model_id"] in kept]
    return rows, observations, initial


@dataclass
class ScalingProblem:
    """Model inputs of one fit: all checkpoints (with roles), observations, bins, model kwargs, start."""

    rows: list
    observations: list
    bins: list
    kwargs: dict
    initial: dict
    variant: Variant


def _prepare_many_peptides(variant, rows, observations, initial):
    if variant.terminal_nll != "logged" or variant.fit_extrapolation:
        raise ValueError("this ablation row is for MDQM9 only")
    if variant.held_out_size is not None:
        if not any(r["role"] == "fit" and r["N"] == variant.held_out_size for r in rows):
            raise ValueError("no fitting checkpoints of that size")
        rows = [{**r, "role": "loso"} if r["role"] == "fit" and r["N"] == variant.held_out_size else r for r in rows]
    minimum = variant.held_out_min_residues
    fit_obs = observations if minimum is None else [o for o in observations if residues_of(o["task_id"]) < minimum]
    if minimum is not None and len(fit_obs) == len(observations):
        raise ValueError("no test systems to hold out")
    train = [r for r in rows if r["role"] == "fit"]
    bins = sorted({r["bin_tokens"] for r in train})
    ref = bins.index(4096)
    if any(r["bin_tokens"] not in bins for r in rows):
        raise ValueError("a held-out bin count has no fitting checkpoint")
    blocks = metric_blocks(train, fit_obs)
    expected = len({rec["task_id"] for b in blocks for rec in b["records"]})
    if minimum is None and expected != 91:
        raise ValueError("benchmark task set changed")
    for block in blocks:
        if set(Counter(r["model_id"] for r in block["records"]).values()) != {expected}:
            raise ValueError("unequal complete-system coverage")
    systems = {task: i for i, task in enumerate(sorted({rec["task_id"] for b in blocks for rec in b["records"]}))}
    lineages = {name: i for i, name in enumerate(sorted({lineage_of(r["model_id"]) for r in train}))}
    # Random effects are projected orthogonal to the fixed effects they would copy:
    # system effects sum to zero (metric intercept), lineage effects to the bin offsets.
    system_design = np.ones((len(systems), 1))
    lineage_design = np.array([[name.split("_")[1] == str(b) for b in bins] for name in lineages], dtype=float)
    kwargs = {"n": jnp.array([r["N"] / 1e9 for r in train]), "d": jnp.array([r["D"] / 1e9 for r in train]),
              "slot": jnp.array([bins.index(r["bin_tokens"]) for r in train]),
              "nll": jnp.array([r["final_nll"] for r in train]), "nbins": len(bins), "ref": ref,
              "blocks": [{"idx": jnp.array(b["idx"]), "atom": jnp.array(b["atom"]), "y": jnp.array(b["y"]),
                          "system": jnp.array([systems[rec["task_id"]] for rec in b["records"]])} for b in blocks],
              "lineage": jnp.array([lineages[lineage_of(r["model_id"])] for r in train]), "nlineage": len(lineages),
              "nsystems": len(systems),
              "system_projection": jnp.array(np.eye(len(systems)) - system_design @ np.linalg.pinv(system_design)),
              "lineage_projection": jnp.array(np.eye(len(lineages)) - lineage_design @ np.linalg.pinv(lineage_design))}
    return rows, bins, kwargs, dict(initial), variant


def _prepare_mdqm9(variant, rows, observations):
    if variant.lineage_effects:
        raise ValueError("lineage effects are not identifiable on MDQM9: all endpoints of a size share one trunk")
    if variant.held_out_min_residues is not None:
        raise ValueError("the residue hold-out applies to peptides only")
    if variant.held_out_size is not None and not any(
            r["N"] == variant.held_out_size and r["N"] != MDQM9_EXTRAPOLATION_SIZE for r in rows):
        raise ValueError("no MDQM9 fitting checkpoints of that size")
    rows = [dict(r) for r in rows]
    for r in rows:
        if variant.terminal_nll != "logged" and r["local_linear_nll"] is None:
            raise ValueError("fewer than 3 low-LR loss rows: " + r["model_id"])
        r.update(bin_tokens=4096, label=f"{r['model_id'].split('_')[1].upper()} {r['D'] / r['N']:.0f}N")
        r["final_nll"] = r["selected_nll"] = r["terminal_nll"] if variant.terminal_nll == "logged" else r["local_linear_nll"]
        r["role"] = ("extrapolation" if r["N"] == MDQM9_EXTRAPOLATION_SIZE and not variant.fit_extrapolation
                     else "loso" if r["N"] == variant.held_out_size else "fit")
    tasks = []
    if variant.responses == "joint":
        coverage = {}
        for o in observations:
            coverage.setdefault((o["model_id"], o["metric_id"]), set()).add(o["task_id"])
        # Keep the molecules with all metrics on every endpoint.
        complete = set.intersection(*(coverage.get((r["model_id"], metric), set()) for r in rows for metric in METRICS))
        tasks = sorted(complete)
        if len(tasks) < MDQM9_MINIMUM_TASKS:
            raise ValueError(f"only {len(tasks)} test molecules have complete metrics")
        observations = [o for o in observations if o["task_id"] in complete]
    else:
        observations = []
    train = [r for r in rows if r["role"] == "fit"]
    blocks = metric_blocks(train, observations, MDQM9_ATOM_REFERENCE)
    for block in blocks:
        if variant.responses == "joint" and set(Counter(r["model_id"] for r in block["records"]).values()) != {len(tasks)}:
            raise ValueError("unequal complete-molecule coverage")
    floor_loc = min(r["final_nll"] for r in train) - MDQM9_FLOOR_OFFSET
    variant = replace(variant, dataset="mdqm9", atom_reference=MDQM9_ATOM_REFERENCE, floor_loc=float(floor_loc))
    systems = {task: i for i, task in enumerate(tasks)}
    system_design = np.ones((len(systems), 1))
    n = np.array([r["N"] / 1e9 for r in train])
    d = np.array([r["D"] / 1e9 for r in train])
    y = np.array([r["final_nll"] for r in train])
    kwargs = {"n": jnp.array(n), "d": jnp.array(d), "slot": jnp.zeros(len(train), dtype=int), "nll": jnp.array(y),
              "nbins": 1, "ref": 0, "nsystems": len(systems),
              "blocks": [{"idx": jnp.array(b["idx"]), "atom": jnp.array(b["atom"]), "y": jnp.array(b["y"]),
                          "system": jnp.array([systems[rec["task_id"]] for rec in b["records"]], dtype=int)} for b in blocks],
              "system_projection": jnp.array(np.eye(len(systems)) - system_design @ np.linalg.pinv(system_design))}
    # Start at the A4 alpha, the prior-mean allocation, and the amplitude that puts the median fitting gap
    # through the observed losses; adapters start at their prior means with the matching intercept.
    alpha, ratio, floor = 0.216, 20.0, float(y.min() - 0.005)
    shape = 0.5 * ((n * np.sqrt(ratio)) ** -alpha + (d / np.sqrt(ratio)) ** -alpha)
    g0 = float(np.median((y - floor) / shape))
    g = g0 * shape
    initial = {"floor": floor, "log_alpha": np.log(alpha), "allocation_ratio": ratio, "log_g0": np.log(g0),
               "loss_bin_offset": np.zeros(0), "log_sigma_nll": np.log(0.003), "log_nu_nll": np.log(10.0)}
    if variant.responses == "joint":
        for prefix, block, (atom, kappa) in zip(PREFIXES, blocks, ((2.0, 1.2), (0.7, 0.7), (1.2, 1.2))):
            log_g = np.log(g[block["idx"]])
            eta = logit(np.clip(block["y"] / np.log(10000.0), 1e-4, 1 - 1e-4)) if prefix == "ress" else block["y"]
            intercept = float(np.mean(eta - atom * block["atom"] - kappa * log_g))
            mean = intercept + atom * block["atom"] + kappa * log_g
            if prefix == "ress":
                mean = np.log(10000.0) / (1 + np.exp(-mean))
            initial.update({prefix + "_intercept": intercept, prefix + "_bin_offset": np.zeros(0), prefix + "_atom": atom,
                            prefix + "_kappa": kappa, prefix + "_log_scale": float(np.log(np.std(block["y"] - mean))),
                            prefix + "_log_df": np.log(10.0)})
    return rows, [4096], kwargs, initial, variant, observations


def prepare(variant, rows, observations, initial=None, dataset="many_peptides"):
    """Model inputs, compute weights and starting values of one :class:`Variant`.

    rows/observations: the tidy tables of :func:`load_inputs`; initial: the Many Peptides starting
    values (v19 posterior medians); MDQM9 starting values are computed from the data.
    The weights are $w_i=(6N_iD_i)^\\gamma/\\overline{(6ND)^\\gamma}$ over the fitting checkpoints.
    """
    if dataset == "mdqm9":
        rows, bins, kwargs, initial, variant, observations = _prepare_mdqm9(variant, rows, observations)
    else:
        rows, bins, kwargs, initial, variant = _prepare_many_peptides(variant, rows, observations, initial)
    train = [r for r in rows if r["role"] == "fit"]
    logc = np.log([6.0 * r["N"] * r["D"] for r in train])
    weights = np.exp(variant.gamma * (logc - logc.max()))
    weights /= weights.mean()
    kwargs.update(weights=jnp.array(weights), variant=variant)
    # Convert the allocation-form starting values where the variant changes parameters.
    if variant.prior == "independent":
        ratio = initial.pop("allocation_ratio")
        log_g0 = initial.pop("log_g0")
        shift = 0.5 * np.exp(initial["log_alpha"]) * np.log(ratio)
        initial.update(log_a=log_g0 - np.log(2.0) - shift, log_b=log_g0 - np.log(2.0) + shift)
    if not variant.tied:
        initial["log_beta"] = initial["log_alpha"]
    if variant.responses == "nll":
        initial = {k: v for k, v in initial.items() if not k.startswith(PREFIXES)}
    if variant.system_effects:
        for prefix in PREFIXES:
            initial.update({prefix + "_system_tau": 0.5 if prefix == "ress" else 0.3,
                            prefix + "_system_raw": np.zeros(kwargs["nsystems"])})
    if variant.lineage_effects:
        initial.update(lineage_tau=0.05, lineage_raw=np.zeros(kwargs["nlineage"]))
    if variant.atom_form == "both":
        initial.update({prefix + "_atom_curvature": 0.0 for prefix in PREFIXES})
    if variant.atom_form == "extensive":
        for prefix in ("energy", "ress"):
            initial.pop(prefix + "_atom")
    if variant.ress_censor_ess is not None:
        # The tolerance keeps ESS exactly equal to c censored.
        block = kwargs["blocks"][PREFIXES.index("ress")]
        censored = np.asarray(block["y"]) >= np.log(1e4 / variant.ress_censor_ess) - 1e-9
        block.update(censored=jnp.array(np.flatnonzero(censored)), exact=jnp.array(np.flatnonzero(~censored)))
    return ScalingProblem(rows, observations, bins, kwargs, initial, variant)


def sample(problem, *, chains=4, warmup=2000, draws=2000, target_accept=0.99, seed=1800, chain_method="parallel"):
    """Dense-mass NUTS (tree depth 12) from the problem's starting values. Returns (draws, sampler fields)
    grouped by chain."""
    kernel = NUTS(model, init_strategy=init_to_value(values=problem.initial), dense_mass=True,
                  target_accept_prob=target_accept, max_tree_depth=12)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=chains, chain_method=chain_method,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **problem.kwargs,
             extra_fields=("diverging", "num_steps", "accept_prob", "potential_energy", "energy"))
    return mcmc.get_samples(group_by_chain=True), mcmc.get_extra_fields(group_by_chain=True)


def diagnose(grouped, extra):
    """Convergence gate of the paper fits: no divergences, split $\\hat R\\le1.01$, ESS $\\ge400$,
    under 1% of transitions at the tree-depth limit, and E-BFMI above 0.3 in every chain."""
    # Projected random effects are deterministic in the raw effects; empty sites have no diagnostics.
    diagvars = {k: v for k, v in grouped.items() if not k.endswith("_effect") and np.size(v)}
    diagvars["log_optimal_D_per_N"] = (grouped["log_b"] - grouped["log_a"]) / jnp.exp(grouped["log_alpha"])
    stats = diagnostic_summary(diagvars, group_by_chain=True)
    rhat = max(float(np.max(v["r_hat"])) for v in stats.values())
    ess = min(float(np.min(v["n_eff"])) for v in stats.values())
    div = int(np.sum(extra["diverging"]))
    depth = float(np.mean(np.asarray(extra["num_steps"]) >= 4095))
    energy = np.asarray(extra["energy"])
    bfmi = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(energy, axis=1)
    ok = div == 0 and rhat <= 1.01 and ess >= 400 and depth < 0.01 and np.min(bfmi) > 0.3
    return {"status": "ok" if ok else "failed", "divergences": div, "max_split_rhat": rhat,
            "min_effective_sample_size": ess, "ebfmi_by_chain": bfmi.tolist(), "tree_depth_limit_fraction": depth}


def interval(values):
    return {"median": float(np.median(values)), "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "q2.5": float(np.quantile(values, 0.025)), "q97.5": float(np.quantile(values, 0.975))}


def scores(y, p):
    error = p - y
    variance = float(np.var(y))
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
            "bias_predicted_minus_observed": float(np.mean(error)),
            "r2": float(1 - np.mean(error**2) / variance) if variance else None}


def eval_scores(evaluations, checkpoint_lookup):
    """Checkpoint-balanced RMSE, $R^2$ and predicted/observed ratio of the metric predictions per role."""
    output = {}
    for metric in METRICS:
        for role in ("fit", "extrapolation", "diagnostic_control"):
            rr = [r for r in evaluations if r["metric"] == metric and checkpoint_lookup[r["model_id"]]["role"] == role]
            if not rr:
                continue
            counts = Counter(r["model_id"] for r in rr)
            weights = np.array([1 / counts[r["model_id"]] for r in rr])
            weights /= weights.sum()
            y = np.array([r["target"] for r in rr])
            p = np.array([r["prediction"] for r in rr])
            error = p - y
            logerror = -error if metric == "end_to_end_ress" else error
            variance = np.sum(weights * (y - np.sum(weights * y)) ** 2)
            output.setdefault(metric, {})[role] = {
                "n_checkpoints": len(counts), "n_rows": len(rr), "rmse": float(np.sqrt(np.sum(weights * error**2))),
                "r2": float(1 - np.sum(weights * error**2) / variance),
                "predicted_over_observed": float(np.exp(np.sum(weights * logerror)))}
    return output


def analyze(grouped, problem):
    r"""Posterior summaries, derived quantities and held-out predictions from draws grouped by chain.

    Derived quantities (paper, Eq. scaling-optimal-ratio and Tables tab:scaling-ablation):
    $D^*(N)/N=(\beta B/(\alpha A))^{1/\beta}N^{\alpha/\beta-1}$ at $N=85.5$M ('optimal_D_per_N'),
    the frontier exponent $\alpha\beta/(\alpha+\beta)$ ('resource_compute_exponent'), and the compute
    exponent at fixed E-W2, $p_E/(k_E\,\alpha\beta/(\alpha+\beta))$ ('compute_parity_exponent').
    Returns (result, flat draws, metric predictions). Predictive draws use numpy seed 1801.
    """
    rows, observations, bins, variant = problem.rows, problem.observations, problem.bins, problem.variant
    flat = {k: np.asarray(v).reshape((np.shape(v)[0] * np.shape(v)[1], *np.shape(v)[2:])) for k, v in grouped.items()}
    if variant.atom_form == "extensive":
        for prefix in ("energy", "ress"):
            flat[prefix + "_atom"] = np.ones(len(flat["floor"]))
    alpha = np.exp(flat["log_alpha"])
    beta = np.exp(flat["log_beta"]) if "log_beta" in flat else alpha
    a = np.exp(flat["log_a"])
    b = np.exp(flat["log_b"])

    def log_tpp(size):
        return np.log(beta * b / (alpha * a)) / beta + (alpha / beta - 1) * np.log(size)

    logr = log_tpp(N_REF)
    ratio = np.exp(logr)
    rate = alpha * beta / (alpha + beta)  # G* scales as C^(-rate) on the compute-optimal frontier
    posterior = {k: interval(v) for k, v in {
        "E": flat["floor"], "A": a, "B": b, "alpha": alpha, "optimal_D_per_N": ratio, "log_optimal_D_per_N": logr,
        "resource_compute_exponent": rate, "resource_gap_factor_per_10x_compute": 10 ** (-rate),
        "sigma_nll": np.exp(flat["log_sigma_nll"]), "df_nll": 2 + np.exp(flat["log_nu_nll"])}.items()}
    if not variant.tied:
        posterior["beta"] = interval(beta)
        posterior["optimal_D_per_N_640m"] = interval(np.exp(log_tpp(0.64)))
    if variant.prior == "allocation":
        np.testing.assert_allclose(ratio, flat["allocation_ratio"], rtol=1e-8)
        posterior["G0"] = interval(np.exp(flat["log_g0"]))
    if variant.lineage_effects:
        posterior["lineage_tau"] = interval(flat["lineage_tau"])

    def atom_term(prefix, x, draws=slice(None)):
        term = flat[prefix + "_atom"][draws, None] * (np.exp(x) - 1.0 if variant.atom_form == "exponential" else x)
        if variant.atom_form == "both":
            term = term + flat[prefix + "_atom_curvature"][draws, None] * (np.exp(x) - 1.0 - x)
        return term

    if variant.responses == "joint":
        # At fixed E-W2, log C changes by (atom term)/(k_E*rate) as n moves away from the reference.
        scale = flat["energy_kappa"] * rate
        if variant.atom_form in ("power", "extensive"):
            posterior["compute_parity_exponent"] = interval(flat["energy_atom"] / scale)
        for size in (157, 300) if variant.dataset == "many_peptides" else ():
            log_factor = atom_term("energy", np.log([size / 65.0]))[:, 0] / scale
            posterior[f"compute_factor_n{size}"] = interval(np.exp(log_factor))
            if size == 157:
                posterior["compute_exponent_65_157"] = interval(log_factor / np.log(157 / 65.0))
    result = {"posterior": posterior,
              "allocation_probabilities": {str(r): float(np.mean(ratio > r)) for r in (20, 40, 80, 160, 203, 320)},
              "probability_alpha_above_beta": float(np.mean(alpha > beta)), "variant": asdict(variant),
              "adapters": {}, "loss_bin_offsets": {}, "loss": {}, "heldout_loss": [], "evaluation": {},
              "evaluation_8aa": {}, "heldout_metric": {}, "long_systems": {}}
    ref = bins.index(4096)
    offsets = np.insert(flat["loss_bin_offset"], ref, 0.0, axis=1)
    for i, bin_count in enumerate(bins):
        result["loss_bin_offsets"][str(bin_count)] = interval(offsets[:, i])
    for prefix in PREFIXES if variant.responses == "joint" else ():
        adapter = {k: interval(flat[prefix + "_" + k]) for k in ("intercept", "atom", "kappa")}
        adapter["sigma"] = interval(np.exp(flat[prefix + "_log_scale"]))
        adapter["df"] = interval(2 + np.exp(flat[prefix + "_log_df"]))
        adapter["probability_positive_kappa"] = float(np.mean(flat[prefix + "_kappa"] > 0))
        if variant.system_effects:
            adapter["system_tau"] = interval(flat[prefix + "_system_tau"])
        if variant.atom_form == "both":
            adapter["atom_curvature"] = interval(flat[prefix + "_atom_curvature"])
            adapter["probability_positive_curvature"] = float(np.mean(flat[prefix + "_atom_curvature"] > 0))
        result["adapters"][prefix] = adapter
    full = metric_blocks(rows, observations, variant.atom_reference)
    n = np.array([r["N"] / 1e9 for r in rows])
    d = np.array([r["D"] / 1e9 for r in rows])
    slots = np.array([bins.index(r["bin_tokens"]) for r in rows])
    g = a[:, None] * n[None, :] ** (-alpha[:, None]) + b[:, None] * d[None, :] ** (-beta[:, None])
    rng = np.random.default_rng(1801)
    if variant.lineage_effects:
        # A fitted lineage keeps its own effect; a lineage without fitting checkpoints draws a new one.
        names = [lineage_of(r["model_id"]) for r in rows]
        fitted = sorted({lineage_of(r["model_id"]) for r in rows if r["role"] == "fit"})
        order = fitted + sorted(set(names) - set(fitted))
        unseen = len(order) - len(fitted)
        effect = np.concatenate((flat["lineage_effect"], flat["lineage_tau"][:, None] * rng.standard_normal((len(alpha), unseen))), axis=1)
        g = g * np.exp(effect[:, [order.index(name) for name in names]])
    loss = flat["floor"][:, None] + g + offsets[:, slots]
    prediction = np.median(loss, axis=0)
    y = np.array([r["final_nll"] for r in rows])
    correction = np.log([r["bin_tokens"] / 4096 for r in rows])
    noisy = loss + rng.standard_t((2 + np.exp(flat["log_nu_nll"]))[:, None], size=loss.shape) * np.exp(flat["log_sigma_nll"])[:, None]
    lo, hi = np.quantile(noisy, [0.025, 0.975], axis=0)
    roles = [role for role in ("fit", "extrapolation", "loso") if any(r["role"] == role for r in rows)]
    for role in roles:
        mask = np.array([r["role"] == role for r in rows])
        result["loss"][role] = {**scores(y[mask] - correction[mask], prediction[mask] - correction[mask]),
                                "predictive_coverage": float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))}
    for i, r in enumerate(rows):
        if r["role"] != "fit":
            result["heldout_loss"].append({
                "model_id": r["model_id"], "label": r.get("label", r["model_id"]), "role": r["role"], "observed": float(y[i]),
                "predicted": float(prediction[i]), "mean_ci95": np.quantile(loss[:, i], [0.025, 0.975]).tolist(),
                "predictive_ci95": [float(lo[i]), float(hi[i])]})
    if variant.responses == "nll":
        return result, flat, []
    systems = {task: i for i, task in enumerate(sorted({rec["task_id"] for blk in full for rec in blk["records"]}))}
    selected = np.linspace(0, len(alpha) - 1, 1000, dtype=int)
    evals, checkpoint_eval, lookup = [], [], {r["model_id"]: r for r in rows}
    for prefix, block in zip(PREFIXES, full):
        offset = np.insert(flat[prefix + "_bin_offset"][selected], ref, 0.0, axis=1)
        idx = block["idx"]
        eta = (flat[prefix + "_intercept"][selected, None] + offset[:, slots[idx]] + atom_term(prefix, block["atom"], selected)
               + flat[prefix + "_kappa"][selected, None] * np.log(g[selected][:, idx]))
        if variant.system_effects:
            eta = eta + flat[prefix + "_system_effect"][selected][:, [systems[t["task_id"]] for t in block["records"]]]
        means = np.log(10000) * expit(eta) if prefix == "ress" else eta
        pred = np.median(means, axis=0)
        noise = rng.standard_t((2 + np.exp(flat[prefix + "_log_df"][selected]))[:, None], size=means.shape) * np.exp(flat[prefix + "_log_scale"][selected, None])
        lo, hi = np.quantile(means + noise, [0.025, 0.975], axis=0)
        for record, p, low, high in zip(block["records"], pred, lo, hi):
            evals.append({**record, "metric": block["metric"], "prediction": float(p), "covered": bool(low <= record["target"] <= high)})
        for r in rows:
            if r["role"] == "fit":
                continue
            for subset in ("all91", "8AA") if variant.dataset == "many_peptides" else ("all",):
                ii = [i for i, t in enumerate(block["records"]) if t["model_id"] == r["model_id"] and (subset != "8AA" or "/8AA/" in t["task_id"])]
                error = pred[ii] - block["y"][ii]
                sign = -1 if prefix == "ress" else 1
                checkpoint_eval.append({"model_id": r["model_id"], "label": r.get("label", r["model_id"]), "role": r["role"],
                                        "metric": block["metric"], "subset": subset, "n": len(ii),
                                        "rmse": float(np.sqrt(np.mean(error**2))),
                                        "predicted_over_observed": float(np.exp(sign * error.mean()))})
        del means, eta, noise
    result["evaluation"] = eval_scores(evals, lookup)
    result["evaluation_8aa"] = eval_scores([r for r in evals if "/8AA/" in r["task_id"]], lookup)
    result["heldout_evaluation"] = checkpoint_eval
    for role in roles[1:]:
        for metric in METRICS:
            # Every checkpoint has all systems, so a plain mean weights checkpoints equally.
            rr = [e for e in evals if e["metric"] == metric and lookup[e["model_id"]]["role"] == role]
            error = np.array([e["prediction"] - e["target"] for e in rr])
            result["heldout_metric"].setdefault(role, {})[metric] = {
                "n_checkpoints": len({e["model_id"] for e in rr}), "rmse": float(np.sqrt(np.mean(error**2))),
                "predictive_coverage": float(np.mean([e["covered"] for e in rr]))}
    for metric in METRICS if variant.dataset == "many_peptides" else ():
        # Peptides with 8 or more residues on the fitting checkpoints; out of sample when the variant holds them out.
        rr = [e for e in evals if e["metric"] == metric and lookup[e["model_id"]]["role"] == "fit" and residues_of(e["task_id"]) >= 8]
        error = np.array([e["prediction"] - e["target"] for e in rr])
        result["long_systems"][metric] = {
            "n_systems": len({e["task_id"] for e in rr}), "held_out": variant.held_out_min_residues == 8,
            "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(np.mean(error)),
            "predictive_coverage": float(np.mean([e["covered"] for e in rr]))}
    return result, flat, evals
