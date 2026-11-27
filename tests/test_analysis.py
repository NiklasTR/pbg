"""Regression test of the scaling model and its shipped input tables.

The release model was checked bitwise against the research implementation
(log densities, derived quantities and short NUTS runs); this test pins the
log joint density at the shipped starting values. ``pbg.analysis`` turns on
JAX x64 when imported, so the import happens inside the test and the previous
setting is restored afterwards.
"""

from pathlib import Path

import jax
import pytest

pytest.importorskip("numpyro")

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize(
    ("row", "expected"),
    [("A4", -42257.752996786374), ("C1", -42203.75986102684)],
)
def test_log_density_at_starting_values(row: str, expected: float) -> None:
    previous = jax.config.jax_enable_x64
    try:
        import jax.numpy as jnp
        from numpyro.infer.util import log_density

        from pbg.analysis import scaling_model

        jax.config.update("jax_enable_x64", True)
        rows, observations, initial = scaling_model.load_inputs(DATA)
        problem = scaling_model.prepare(scaling_model.ABLATIONS[row], rows, observations, dict(initial))
        params = {key: jnp.asarray(value) for key, value in problem.initial.items()}
        value, _ = log_density(scaling_model.model, (), problem.kwargs, params)
    finally:
        jax.config.update("jax_enable_x64", previous)
    assert len(problem.rows) == 134
    assert float(value) == pytest.approx(expected, rel=1e-12)
