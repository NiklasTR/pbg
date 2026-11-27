#!/usr/bin/env python3
"""Train PBG models from one or more JSON configs, in order (see configs/README.md).

Each config is merged onto the defaults of ``pbg.models.pbg.train`` and
run to its ``training.num_steps``. A run resumes from the latest checkpoint of
its prefix when ``checkpoint.resume_latest`` is set, so an interrupted
sequence can be restarted with the same command. Multi-GPU data parallelism
uses all local devices; multi-node runs use ``jax.distributed`` (one process
per node). Example:

    PBG_DATA_BASE=/data PBG_CHECKPOINT_BASE=/ckpt python scripts/train.py \\
        configs/core_scaling/stable/dipper_15m_core_bin4096_ws_scaling.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--data-base", help="root of the relative data paths (default: $PBG_DATA_BASE or ./data)")
    parser.add_argument("--checkpoint-base", help="root of parents/ and checkpoints/ (default: $PBG_CHECKPOINT_BASE or .)")
    args = parser.parse_args()
    if args.data_base:
        os.environ["PBG_DATA_BASE"] = args.data_base
    if args.checkpoint_base:
        os.environ["PBG_CHECKPOINT_BASE"] = args.checkpoint_base
    # Import after the environment is set: the defaults read it at import time.
    from pbg.models.pbg.train import load_config, run

    for index, path in enumerate(args.configs, start=1):
        print(f"training config {index}/{len(args.configs)}: {path}", flush=True)
        run(load_config(path))


if __name__ == "__main__":
    main()
