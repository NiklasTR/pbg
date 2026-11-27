"""Fit one row of the joint Bayesian scaling model from the tidy tables in data/.

Paper: Appendix "Scaling Model on Many Peptides" (row A4 is the paper law v23),
"Sensitivity of the Scaling Model" (rows A0-A6, D1-D2, F*, C*, *_loso_*) and
"Scaling Model on MDQM9-nc" (--dataset mdqm9; the paper uses cohort n28).
--post-extrapolation refits row A4 with three of the four held-out checkpoints
(law v23_post_extrapolation, used in the speculative-decoding appendix).
See pbg.analysis.scaling_model for the equations.

Run MCMC through a CPU batch job, not on a shared login node, e.g.:

    sbatch -p cpu -t 02:00:00 -c 4 --mem 32G --wrap \\
        "python scripts/fit_scaling_model.py --row A4 --out outputs/scaling/A4"
    python scripts/fit_scaling_model.py --dataset mdqm9 --cohort n28 --row A5 --out outputs/mdqm9_n28/A5

With --draws-from DRAWS.npz the script skips sampling and analyses stored draws
(grouped by chain), e.g. the draws of a published fit.

Outputs in --out: run.json (inputs, variant, settings, likelihood check),
draws.npz and sampler.npz (by chain), diagnostics.json, posterior_results.json
(posterior summaries, $D^*/N$, compute parity exponent, held-out errors) and
evaluation_predictions.json (metric predictions per checkpoint and system).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", default="A4", help="ablation row (pbg.analysis.scaling_model.ABLATIONS); A4 is the paper model")
    parser.add_argument("--dataset", choices=("many_peptides", "mdqm9"), default="many_peptides")
    parser.add_argument("--cohort", choices=("n28", "n36"), default="n28", help="MDQM9-nc cohort (paper: n28)")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1800)
    parser.add_argument("--draws-from", type=Path, default=None, help="analyse stored draws instead of sampling")
    parser.add_argument("--post-extrapolation", action="store_true",
                        help="row A4 with three held-out checkpoints moved into the fit (law v23_post_extrapolation)")
    args = parser.parse_args()
    if args.post_extrapolation and (args.dataset != "many_peptides" or args.row != "A4"):
        parser.error("--post-extrapolation refits the Many Peptides row A4")
    # Parallel chains need one host device each; set before JAX starts.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={args.chains}")
    import jax.numpy as jnp
    import numpy as np
    from numpyro.infer.util import log_density

    from pbg.analysis import scaling_model as sm

    if args.row not in sm.ABLATIONS:
        parser.error(f"unknown row {args.row}")
    if args.draws_from is None and (args.out / "draws.npz").exists():
        raise SystemExit(f"{args.out} holds draws; choose a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, observations, initial = sm.load_inputs(args.data, args.dataset, args.cohort)
    if args.post_extrapolation:
        rows = [dict(r, role="fit") if r["model_id"] in sm.POST_EXTRAPOLATION else r for r in rows]
    problem =sm.prepare(sm.ABLATIONS[args.row], rows, observations, initial, dataset=args.dataset)
    kwargs = problem.kwargs
    # The tilted likelihood must equal the unweighted one plus sum_i (w_i - 1) log p_i.
    weighted, trace = log_density(sm.model, (), kwargs, problem.initial)
    unweighted, _ = log_density(sm.model, (), dict(kwargs, weights=jnp.ones_like(kwargs["weights"])), problem.initial)
    expected = sum(float(jnp.sum((site["scale"] - 1) * site["fn"].log_prob(site["value"])))
                   for site in trace.values() if site["type"] == "sample" and site.get("is_observed"))
    np.testing.assert_allclose(float(weighted - unweighted), expected, rtol=1e-9, atol=1e-7)
    train = [r for r in problem.rows if r["role"] == "fit"]
    save(args.out / "run.json", {
        "row": args.row + ("_post_extrapolation" if args.post_extrapolation else ""), "dataset": args.dataset, "cohort": args.cohort if args.dataset == "mdqm9" else "v23",
        "variant": asdict(problem.variant),
        "fit_checkpoints": len(train), "heldout_checkpoints": len(problem.rows) - len(train),
        "systems": kwargs.get("nsystems"), "bins": problem.bins,
        "weights": {r["model_id"]: float(w) for r, w in zip(train, np.asarray(kwargs["weights"]))},
        "likelihood_check": {"weighted_minus_unweighted": float(weighted - unweighted), "expected": expected},
        "sampler": {"chains": args.chains, "warmup": args.warmup, "draws": args.draws, "target_accept": args.target_accept,
                    "seed": args.seed, "max_tree_depth": 12, "dense_mass": True, "draws_from": str(args.draws_from or "")}})
    if args.draws_from is None:
        start = time.monotonic()
        grouped, extra = sm.sample(problem, chains=args.chains, warmup=args.warmup, draws=args.draws,
                                   target_accept=args.target_accept, seed=args.seed)
        grouped = {k: np.asarray(v) for k, v in grouped.items()}
        np.savez_compressed(args.out / "draws.npz", **grouped)
        np.savez_compressed(args.out / "sampler.npz", **{k: np.asarray(v) for k, v in extra.items()})
        diagnostics = {**sm.diagnose(grouped, extra), "elapsed_seconds": time.monotonic() - start}
        save(args.out / "diagnostics.json", diagnostics)
        print(json.dumps(diagnostics, indent=2), flush=True)
    else:
        with np.load(args.draws_from) as archive:
            grouped = {k: archive[k] for k in archive.files}
    result, _, evaluations = sm.analyze(grouped, problem)
    save(args.out / "posterior_results.json", result)
    if evaluations:
        save(args.out / "evaluation_predictions.json", evaluations)
    print(json.dumps({k: result["posterior"][k]["median"] for k in ("alpha", "optimal_D_per_N", "compute_parity_exponent")
                      if k in result["posterior"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
