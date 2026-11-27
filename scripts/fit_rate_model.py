"""Fit the empirical sampling-rate model from data/sampling_rate_cells.csv.gz.

Paper: Appendix "Increasing Sampling Rate with Speculative Decoding" (Eqs.
rate-model and rate-model-likelihood, Tables tab:rate-model-parameters and
tab:rate-model-fit). See pbg.analysis.rate_model for the equations.

Run MCMC through a CPU batch job, not on a shared login node, e.g.:

    sbatch -p cpu -t 01:00:00 -c 4 --mem 16G --wrap \\
        "python scripts/fit_rate_model.py --out outputs/rate_model"

Options: --no-cv skips the five-fold peptide cross-validation (Table
tab:rate-model-fit reports its MAPE); --draws-from DRAWS.npz analyses stored
draws (grouped by chain) instead of sampling; --check-features rebuilds the
work features from the architecture table, prompts and acceptance counts and
compares them with the stored features of the published fit.

Outputs in --out: rate_model.json (posterior, diagnostics, in-sample fit,
posterior predictive checks, cross-validation, fitted latencies) and draws.npz.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--draws-from", type=Path, default=None)
    parser.add_argument("--check-features", action="store_true")
    parser.add_argument("--simulation-batches", type=int, default=128)
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np

    from pbg.analysis import rate_model as rm
    from pbg.analysis import read_table

    args.out.mkdir(parents=True, exist_ok=True)
    cells = read_table(args.data / "sampling_rate_cells.csv.gz")
    raw, method_index, observed, _, _ = rm.design(cells)
    if args.check_features:
        architectures = read_table(args.data / "sampling_rate_architectures.csv.gz")
        rebuilt, derived = rm.build_features(cells, architectures, args.simulation_batches)
        error = np.abs(rebuilt - raw) / np.maximum(np.abs(raw), 1e-300)
        check = {"max_relative_error": float(error.max()), "max_absolute_error": float(np.abs(rebuilt - raw).max()),
                 "exact_cells": int(np.sum(np.all(rebuilt == raw, axis=1))), "cells": len(cells),
                 "prompt_and_horizon_match": all(d["prompt_tokens"] == c["prompt_tokens"] and d["horizon"] == c["horizon"]
                                                 for c, d in zip(cells, derived))}
        (args.out / "feature_check.json").write_text(json.dumps(check, indent=2) + "\n")
        print(json.dumps(check, indent=2), flush=True)
        return
    if args.draws_from is None:
        grouped, _, diagnostics = rm.sample(raw, method_index, observed, chains=args.chains, warmup=args.warmup,
                                            draws=args.draws, seed=args.seed)
        np.savez_compressed(args.out / "draws.npz", **grouped)
        print("Posterior diagnostics", json.dumps(diagnostics), flush=True)
        if not diagnostics["passed"]:
            raise SystemExit("MCMC diagnostics failed")
    else:
        with np.load(args.draws_from) as archive:
            grouped = {k: archive[k] for k in rm.PRIORS}
        diagnostics = {"draws_from": str(args.draws_from)}
    rng = np.random.default_rng(20260926)
    result = rm.analyze(grouped, cells, rng)
    if not args.no_cv:
        result["validation"] = {"peptide": rm.cross_validate(cells, rng)}
        print("Validation", json.dumps(result["validation"]["peptide"]["sampling_rate"]),
              "speedup MAPE", result["validation"]["peptide"]["speedup_mape_pct"], flush=True)
    payload = {"formula": rm.FORMULA, "priors": rm.PRIORS, "batch_size": rm.BATCH, "diagnostics": diagnostics, **result,
               "raw_feature_names": list(rm.RAW_COLUMNS)}
    (args.out / "rate_model.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("Training", json.dumps(result["training"]), flush=True)


if __name__ == "__main__":
    main()
