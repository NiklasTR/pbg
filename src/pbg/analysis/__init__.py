"""Bayesian analysis models of the paper.

- :mod:`pbg.analysis.scaling_model`: the joint scaling model of training loss and
  sampling quality (Appendix "Scaling Model on Many Peptides", its sensitivity
  analysis, and "Scaling Model on MDQM9-nc").
- :mod:`pbg.analysis.rate_model`: the empirical sampling-rate model of
  autoregressive and speculative decoding (Appendix "Increasing Sampling Rate with
  Speculative Decoding").

Both models are fitted in float64, as in the paper: importing this package
enables JAX x64 mode. The fits read the tidy tables in ``data/`` (see ``data/README.md``) through
:func:`read_table`.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

__all__ = ["read_table"]


def read_table(path: str | Path) -> list[dict]:
    """Read a (gzipped) CSV table as a list of row dicts.

    Cells are typed by content: integers, then floats (written with their
    shortest round-trip repr, so values are exact), ``True``/``False``, and
    strings. Empty cells are ``None``.
    """

    def value(text: str):
        if text == "":
            return None
        if text in ("True", "False"):
            return text == "True"
        # A digit is required, so peptide sequences such as "NAN" or "INF" stay strings.
        for kind in (int, float) if any(c.isdigit() for c in text) else ():
            try:
                return kind(text)
            except ValueError:
                pass
        return text

    path = Path(path)
    with (gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")) as handle:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(handle)]
