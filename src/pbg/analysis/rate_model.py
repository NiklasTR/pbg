r"""Empirical sampling-rate model of autoregressive and speculative decoding.

Paper, Appendix "Increasing Sampling Rate with Speculative Decoding",
Eq. rate-model and Eq. rate-model-likelihood, Tables tab:rate-model-parameters
and tab:rate-model-fit. For a batch of $B=512$ samples,

$$T_B=a+\alpha W+\beta\left(V_1+V_{\mathrm{write}}+\frac{D_{\mathrm{read}}}{q^\gamma}\right)+\tau(J+\kappa U),\qquad R=B/T_B,\qquad S=T_B^{\mathrm{AR}}/T_B^{\mathrm{SD}},$$

with work features $W$ (weight reads, TB), $V_1+V_{\mathrm{write}}$ (single-token cache
traffic and target writes, TB), $D_{\mathrm{read}}$ (packed target cache reads, TB),
$J$ (thousand layer calls), $U$ (thousand extra target layer-token positions) and
call width $q$ ($1$ for AR, $K+1$ for SD). The median rate $r_i$ of five repeats
gives $t_i=B/r_i$ and

$$\log t_i\sim\mathcal N(\log T_{B,i},\sigma_{m(i)}^2),\qquad m(i)\in\{\mathrm{AR},\mathrm{SD}\}.$$

The paper fit draws 4 vectorized NUTS chains (1000 warmup, 2000 draws, target
acceptance 0.95, tree depth 10, seed 20260925) and validates by five-fold
peptide cross-validation. :func:`build_features` derives the work features from
the model architectures, prompts and measured acceptance; the fit itself reads
the stored features of the published fit (columns ``x_*`` of the cell table).
"""

from __future__ import annotations

import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_median
from scipy.optimize import brentq

__all__ = ["FORMULA", "PRIORS", "RAW_COLUMNS", "analyze", "batch_work", "build_features", "cross_validate", "design",
           "expected_counts", "latency", "model", "sample", "scores", "speculative_features", "summarize"]

FORMULA = ("T_512 = a + alpha*W + beta*(V_1 + V_write + D_read/q^gamma) + tau*(J + kappa*U); "
           "samples_per_second=512/T_512; speedup=T_512(AR)/T_512(speculative)")
#: Work features in model order: constant, W, V_1+V_write, D_read, J, U, q.
RAW_COLUMNS = ("x_constant", "x_W", "x_V1_write", "x_D_read", "x_J", "x_U", "x_q")
#: Priors in seconds (a), seconds per TB (alpha, beta), seconds per thousand layer calls (tau); others dimensionless.
PRIORS = {
    "a": {"family": "HalfNormal", "scale": 0.1},
    "alpha": {"family": "LogNormal", "loc": 0.0, "scale": 1.0},
    "beta": {"family": "LogNormal", "loc": 0.0, "scale": 1.0},
    "tau": {"family": "LogNormal", "loc": float(np.log(0.1)), "scale": 1.0},
    "gamma": {"family": "LogNormal", "loc": 0.0, "scale": 0.75},
    "kappa": {"family": "LogNormal", "loc": 0.0, "scale": 0.75},
    "sigma_AR": {"family": "HalfNormal", "scale": 0.25},
    "sigma_SD": {"family": "HalfNormal", "scale": 0.25},
}
BATCH = 512


def latency(parameters, raw):
    """$T_B$ of Eq. rate-model; works with JAX tracers and broadcast NumPy posterior draws."""
    return (parameters["a"] + parameters["alpha"] * raw[..., 1]
            + parameters["beta"] * (raw[..., 2] + raw[..., 3] / raw[..., 6] ** parameters["gamma"])
            + parameters["tau"] * (raw[..., 4] + parameters["kappa"] * raw[..., 5]))


def model(raw, method_index, observed=None):
    """Gaussian log-latency likelihood of Eq. rate-model-likelihood; method_index 0 is AR, 1 is SD."""
    parameters = {}
    for name, prior in PRIORS.items():
        distribution = (dist.HalfNormal(prior["scale"]) if prior["family"] == "HalfNormal"
                        else dist.LogNormal(prior["loc"], prior["scale"]))
        parameters[name] = numpyro.sample(name, distribution)
    mean = jnp.log(latency(parameters, raw))
    sigma = jnp.where(method_index == 0, parameters["sigma_AR"], parameters["sigma_SD"])
    with numpyro.plate("cells", raw.shape[0]):
        numpyro.sample("log_latency", dist.Normal(mean, sigma), obs=observed)


def design(cells):
    """(raw features, method index, observed log latency, latency, method names) of the cell table."""
    raw = np.array([[c[k] for k in RAW_COLUMNS] for c in cells], dtype=float)
    methods = np.array([c["method"] for c in cells])
    times = BATCH / np.array([c["samples_per_second"] for c in cells])
    return raw, (methods == "speculative").astype(np.int32), np.log(times), times, methods


def summarize(grouped):
    """Posterior medians, 95% intervals, split $\\hat R$ and ESS of draws grouped by chain."""
    stats = summary(grouped, prob=0.95, group_by_chain=True)
    posterior = {}
    for name, values in grouped.items():
        q = np.quantile(values, [0.025, 0.5, 0.975])
        posterior[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "median": float(q[1]),
                           "q2.5": float(q[0]), "q97.5": float(q[2]), "split_rhat": float(stats[name]["r_hat"]),
                           "effective_sample_size": float(stats[name]["n_eff"])}
    return posterior


def sample(raw, method_index, observed, *, chains=4, warmup=1000, draws=2000, seed=20260925):
    """Dense-mass NUTS with the paper settings. Returns (draws by chain, posterior summary, diagnostics);
    the diagnostics pass with no divergences, split $\\hat R\\le1.01$, ESS $\\ge400$ and no tree-depth hits."""
    started = time.monotonic()
    sampler = MCMC(NUTS(model, dense_mass=True, target_accept_prob=0.95, max_tree_depth=10,
                        init_strategy=init_to_median(num_samples=20)),
                   num_warmup=warmup, num_samples=draws, num_chains=chains, chain_method="vectorized", progress_bar=False)
    sampler.run(jax.random.key(seed), jnp.asarray(raw), jnp.asarray(method_index), jnp.asarray(observed),
                extra_fields=("diverging", "num_steps"))
    grouped = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
    posterior = summarize(grouped)
    extra = sampler.get_extra_fields()
    diagnostics = {"chains": chains, "warmup": warmup, "draws_per_chain": draws,
                   "divergences": int(np.asarray(extra["diverging"]).sum()),
                   "max_split_rhat": max(v["split_rhat"] for v in posterior.values()),
                   "min_effective_sample_size": min(v["effective_sample_size"] for v in posterior.values()),
                   "tree_depth_limit_hits": int((np.asarray(extra["num_steps"]) >= 1023).sum()),
                   "elapsed_seconds": time.monotonic() - started}
    diagnostics["passed"] = (diagnostics["divergences"] == 0 and diagnostics["max_split_rhat"] <= 1.01
                             and diagnostics["min_effective_sample_size"] >= 400 and diagnostics["tree_depth_limit_hits"] == 0)
    return grouped, posterior, diagnostics


def scores(actual, prediction):
    """Throughput absolute percentage errors and latency $R^2$ (the $R_T^2$ of Table tab:rate-model-fit)."""
    relative = np.abs(actual / prediction - 1) * 100
    return {"throughput_mape_pct": float(relative.mean()), "throughput_median_ape_pct": float(np.median(relative)),
            "throughput_p90_ape_pct": float(np.quantile(relative, 0.9)),
            "time_r2": float(1 - np.sum((actual - prediction) ** 2) / np.sum((actual - actual.mean()) ** 2))}


def analyze(grouped, cells, rng):
    """In-sample fit, posterior predictive checks and derived effects from draws grouped by chain.

    The posterior predictive draws use ``rng`` (numpy seed 20260926 in the paper fit; the same
    generator then feeds :func:`cross_validate`). 'packed_read_factor_q{q}' is $q^{-\\gamma}$.
    """
    raw, method_index, observed, times, methods = design(cells)
    draws = {k: np.asarray(v).reshape(-1, 1) for k, v in grouped.items()}
    conditional = latency(draws, raw[None, :, :])
    median = np.median(conditional, axis=0)
    rate_intervals = np.quantile(BATCH / conditional, [0.025, 0.5, 0.975], axis=0)
    sigma = np.where(method_index[None, :] == 0, draws["sigma_AR"], draws["sigma_SD"])
    replicated = np.exp(np.log(conditional) + rng.normal(size=conditional.shape) * sigma)
    predictive = np.quantile(replicated, [0.025, 0.5, 0.975], axis=0)
    ppc = {}
    for method in ("AR", "speculative"):
        mask = methods == method
        ppc[method] = {
            "predictive_95_coverage": float(np.mean((times[mask] >= predictive[0, mask]) & (times[mask] <= predictive[2, mask]))),
            "log_residual_rmse": float(np.sqrt(np.mean((observed[mask] - np.log(median[mask])) ** 2))),
            "mean_standardized_log_residual": float(np.mean((observed[mask] - np.log(median[mask])) / np.median(sigma[:, mask], axis=0)))}
    return {"posterior": summarize({k: np.asarray(v) for k, v in grouped.items()}),
            "training": {m: scores(times[methods == m], median[methods == m]) for m in ("AR", "speculative")},
            "posterior_predictive": ppc,
            "probabilities": {f"{k}_gt_{v}": float(np.mean(grouped[k] > v)) for k in ("gamma", "kappa") for v in (1, 2)},
            "practical_effects": {f"packed_read_factor_q{q}": dict(zip(("q2.5", "median", "q97.5"),
                                  map(float, np.quantile(q ** (-np.asarray(grouped["gamma"])), [0.025, 0.5, 0.975]))))
                                  for q in (2, 3, 5)},
            "fitted_seconds": median.tolist(), "posterior_rate_intervals": rate_intervals.tolist(),
            "posterior_predictive_latency_intervals": predictive.tolist()}


def cross_validate(cells, rng, *, folds=5, chains=4, warmup=750, draws=1500, seed=20260930):
    """Five-fold peptide cross-validation of Table tab:rate-model-fit: refit without a fifth of the
    peptides (all targets, strides and methods) and predict their rates and SD speedups."""
    raw, method_index, observed, times, methods = design(cells)
    sequences = np.array([c["sequence"] for c in cells])
    labels = rng.permutation(np.unique(sequences))
    spec_indices = np.flatnonzero(method_index == 1)
    ar_index = {(c["model_id"], c["sequence"]): i for i, c in enumerate(cells) if c["method"] == "AR"}
    paired_ar = np.array([ar_index[cells[i]["model_id"], cells[i]["sequence"]] for i in spec_indices])
    actual_gains = np.array([cells[i]["samples_per_second"] / cells[i]["baseline_samples_per_second"] for i in spec_indices])
    oof = np.full(len(cells), np.nan)
    gains = np.full(len(spec_indices), np.nan)
    oof_interval = np.full((3, len(cells)), np.nan)
    records = []
    for fold, held in enumerate(np.array_split(labels, folds)):
        test = np.isin(sequences, held)
        fg, fp, fd = sample(raw[~test], method_index[~test], observed[~test], chains=chains, warmup=warmup, draws=draws,
                            seed=seed + fold)
        if not fd["passed"]:
            raise ValueError(f"fold {fold} posterior diagnostics failed: {json.dumps(fd)}")
        fdraws = {k: v.reshape(-1, 1) for k, v in fg.items()}
        ftime = latency(fdraws, raw[None, :, :])
        oof[test] = np.median(ftime[:, test], axis=0)
        oof_interval[:, test] = np.quantile(BATCH / ftime[:, test], [0.025, 0.5, 0.975], axis=0)
        chosen = test[spec_indices]
        gains[chosen] = np.median(ftime[:, paired_ar[chosen]] / ftime[:, spec_indices[chosen]], axis=0)
        fsigma = np.where(method_index[None, test] == 0, fdraws["sigma_AR"], fdraws["sigma_SD"])
        freplicated = np.exp(np.log(ftime[:, test]) + rng.normal(size=ftime[:, test].shape) * fsigma)
        flo, fhi = np.quantile(freplicated, [0.025, 0.975], axis=0)
        records.append({"fold": fold, "held_out_peptides": held.tolist(), "diagnostics": fd, "posterior": fp,
                        "predictive_95_coverage": float(np.mean((times[test] >= flo) & (times[test] <= fhi)))})
    return {"label": "Five-fold peptide CV", "folds": records,
            "sampling_rate": {m: scores(times[methods == m], oof[methods == m]) for m in ("AR", "speculative")},
            "speedup_mape_pct": float(100 * np.mean(np.abs(gains / actual_gains - 1))),
            "predicted_seconds": oof.tolist(), "predicted_speedups": gains.tolist(),
            "predicted_rate_intervals": oof_interval.tolist()}


def expected_counts(horizon, stride, probability):
    """Finite-horizon renewal means (accepted, proposed, rounds) under a stationary conditional acceptance."""
    accepted = np.zeros(horizon + 1)
    proposed = np.zeros(horizon + 1)
    rounds = np.zeros(horizon + 1)
    for remaining in range(1, horizon + 1):
        active = min(stride, remaining)
        prefix = np.arange(active + 1)
        law = (1 - probability) * probability**prefix
        law[-1] = probability**active
        emitted = np.minimum(prefix + 1, remaining)
        rest = remaining - emitted
        accepted[remaining] = law @ (prefix + accepted[rest])
        proposed[remaining] = active + law @ proposed[rest]
        rounds[remaining] = 1 + law @ rounds[rest]
    return accepted[-1], proposed[-1], rounds[-1]


def batch_work(horizon, prompt, stride, probability, batches, seed):
    """Simulate independent lanes of a 512 batch; charge full batches until all lanes finish.

    Attention dispatches on the longest live prefix; completed lanes still occupy tensor rows.
    """
    from pbg.models.pbg.transformer import prefix_bucket_lengths

    rng = np.random.default_rng(seed)
    outputs = np.zeros((batches, BATCH), dtype=np.int32)
    ladder = np.array(prefix_bucket_lengths(prompt + horizon + stride, 64))
    round_counts = np.zeros(batches)
    draft_prefix = np.zeros(batches)
    target_prefix = np.zeros(batches)
    mean_lane_rounds = np.zeros(batches)
    while np.any(outputs < horizon):
        live = outputs < horizon
        running = live.any(axis=1)
        prefix = np.where(live, prompt - 1 + outputs, 0).max(axis=1)
        for step in range(stride + 1):
            draft_prefix += running * ladder[np.searchsorted(ladder, prefix + step + 1)]
        target_prefix += running * ladder[np.searchsorted(ladder, prefix + stride + 1)]
        round_counts += running
        mean_lane_rounds += live.mean(axis=1)
        # Successes before the first rejection, capped at the active proposals.
        active = np.minimum(stride, horizon - outputs)
        if probability == 1:
            accepted = active
        elif probability == 0:
            accepted = np.zeros_like(active)
        else:
            draws = np.floor(np.log(np.maximum(rng.random(outputs.shape), 1e-300)) / np.log(probability))
            accepted = np.minimum(draws.astype(np.int32), active)
        outputs += np.minimum(accepted + 1, horizon - outputs)
    return {"batch_rounds": float(round_counts.mean()), "batch_rounds_se": float(round_counts.std(ddof=1) / np.sqrt(batches)),
            "mean_lane_rounds": float(mean_lane_rounds.mean()), "draft_prefix_sum": float(draft_prefix.mean()),
            "target_prefix_sum": float(target_prefix.mean()), "buckets": ladder.tolist()}


def speculative_features(target_model, draft_model, work, *, prompt, stride):
    """Architecture work counts of one speculative cell (13 candidate cost features; the model uses five of them).

    Each model is ``(kv_bytes_per_token, num_layers, dense_parameters)`` (see :func:`build_features`).
    """
    target_kv_bytes, target_layers, target_dense = target_model
    draft_kv_bytes, draft_layers, draft_dense = draft_model
    rounds = work["batch_rounds"]
    weight = 2 * (target_dense + (stride + 1) * draft_dense) * rounds / 1e12
    draft_kv = BATCH * draft_kv_bytes * (work["draft_prefix_sum"] + (stride + 1) * rounds + prompt - 1) / 1e12
    target_kv = BATCH * target_kv_bytes * (work["target_prefix_sum"] + (stride + 1) * rounds + prompt - 1) / 1e12
    layers = (target_layers + (stride + 1) * draft_layers) * rounds / 1000
    extra_compute = 2 * BATCH * target_dense * stride * rounds / 1e12
    extra_layer = target_layers * stride * rounds / 1000
    extra_kv = BATCH * target_kv_bytes * work["target_prefix_sum"] * stride / 1e12
    context_positions = BATCH * target_layers * stride * work["target_prefix_sum"] / 1e9
    return [1.0, weight, draft_kv + target_kv, layers, extra_compute, extra_layer, extra_kv, context_positions,
            (stride + 1) * rounds / 1000, 2 * target_dense * rounds / 1e12, target_kv, draft_kv, target_layers * rounds / 1000]


def build_features(cells, architectures, simulation_batches=128):
    """Work features (RAW_COLUMNS order) of every cell from architectures, prompts and acceptance.

    AR cells (listed first) set the prompt length $P$ and horizon $H=3n$ of each target and peptide.
    SD cells invert the measured accepted/proposed ratio to a stationary conditional acceptance
    probability and simulate ``simulation_batches`` batches of 512 lanes (seed from the cell key).
    Returns (raw features, per-cell dicts of prompt_tokens, horizon, probability and simulation).
    """
    from pbg.core.peptide import sequence_to_helm
    from pbg.core.tokeniser import load_tokeniser_default, make_tokeniser
    from pbg.models.pbg.transformer import (
        DEFAULT_ATTENTION_BUCKET_SIZE,
        prefix_bucket_lengths,
    )

    vocab = load_tokeniser_default(vocab_name="core_bin4096")
    vocab_size = int(vocab["meta"]["next_id"])
    configs = {a["model_id"]: a for a in architectures}
    shapes = {}
    for model_id, cfg in configs.items():
        if cfg["vocab_name"] != "core_bin4096" or not cfg["mixed_precision"]:
            raise ValueError(f"Unsupported architecture precision/vocabulary: {model_id}")
        if cfg["use_environment_tokens"] or cfg["use_temperature_tokens"]:
            raise ValueError(f"Unexpected prompt conditioning: {model_id}")
        dim, layers, head_dim = cfg["dim"], cfg["num_layers"], cfg["head_dim"]
        heads, kv_heads = cfg["num_heads"], cfg["num_kv_heads"]
        # Q, K, V and output projections plus the SwiGLU matrices of one block.
        block = 2 * dim * heads * head_dim + 2 * dim * kv_heads * head_dim + 3 * dim * cfg["intermediate_size"]
        # BF16 keys and values cached per token and sequence, summed over layers.
        kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 2
        for legal_head in (4096, vocab_size):
            # Executed dense parameters: blocks plus the output head over ``legal_head``
            # rows; no normalization scales or embedding lookups.
            shapes[model_id, legal_head] = (kv_bytes_per_token, layers, layers * block + legal_head * dim)
    raw, derived, contexts = [], [], {}
    for cell in cells:
        if cell["method"] == "AR":
            cfg = configs[cell["model_id"]]
            if cell["device"] != "NVIDIA H100 NVL" or cell["batch_size"] != BATCH:
                raise ValueError("AR hardware or batch mismatch")
            tokenizer = make_tokeniser(vocab, prompt_format=cfg["prompt_format"])
            prompt = 1 + len(tokenizer.encode_prompt(sequence_to_helm(cell["sequence"]), environment=cell["environment_prompt"],
                                                     temperature_K=cell["temperature_K"]))
            horizon = 3 * cell["atom_count"]
            kv_bytes_per_token, layers, dense = shapes[cell["model_id"], vocab_size]
            buckets = np.array(prefix_bucket_lengths(cfg["max_seq_len"], DEFAULT_ATTENTION_BUCKET_SIZE))
            padded = buckets[np.searchsorted(buckets, np.arange(prompt + 1, prompt + horizon + 1))]
            raw.append([1.0, 2 * dense * horizon / 1e12, BATCH * kv_bytes_per_token * (padded.sum() + horizon + prompt) / 1e12,
                        0.0, layers * horizon / 1000, 0.0, 1.0])
            contexts[cell["model_id"], cell["sequence"]] = (prompt, horizon)
            derived.append({"prompt_tokens": prompt, "horizon": horizon})
            continue
        prompt, horizon = contexts[cell["model_id"], cell["sequence"]]
        stride = cell["stride"]
        acceptance = cell["accepted"] / cell["proposed"]
        if not np.isclose(acceptance, cell["acceptance"], rtol=1e-10):
            raise ValueError("Stored acceptance differs from counters")
        if acceptance in (0.0, 1.0):
            probability = acceptance
        else:
            def residual(p):
                accepted, proposed, _ = expected_counts(horizon, stride, p)
                return accepted / proposed - acceptance

            probability = brentq(residual, 0.0, 1.0, xtol=1e-10)
        seed = int(hashlib.sha256(f"{cell['model_id']}:{cell['sequence']}:{stride}".encode()).hexdigest()[:8], 16)
        simulation = batch_work(horizon, prompt, stride, probability, simulation_batches, seed)
        target, draft = shapes[cell["model_id"], 4096], shapes[cell["draft_model_id"], 4096]
        features = speculative_features(target, draft, simulation, prompt=prompt, stride=stride)
        target_writes = BATCH * target[0] * ((stride + 1) * simulation["batch_rounds"] + prompt - 1) / 1e12
        raw.append([1.0, features[1], features[11] + target_writes, features[10] - target_writes, features[3], features[5],
                    float(stride + 1)])
        derived.append({"prompt_tokens": prompt, "horizon": horizon, "conditional_acceptance_probability": probability,
                        "simulation": simulation})
    return np.array(raw), derived
