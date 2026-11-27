"""Estimators of the paper.

- ``ess``: self-normalised importance weights and rESS.
- ``free_energy``: Zwanzig and BAR free-energy differences.
- ``wasserstein``: W2 on the real line (E-W2), exact W2 in R^d or on a torus (T-W2, TICA-W2), and
  the sliced torus W2 (T-W2*).
- ``features``: the coordinates these distances compare (backbone and global torsions, tICA).
"""

from pbg.estimator.ess import importance_weights, relative_ess
from pbg.estimator.features import backbone_torsions, global_torsions, tica_projection
from pbg.estimator.free_energy import BarEstimate, ZwanzigEstimate, bar, zwanzig
from pbg.estimator.wasserstein import sliced_torus_w2, w2, w2_1d

__all__ = [
    "BarEstimate",
    "ZwanzigEstimate",
    "backbone_torsions",
    "bar",
    "global_torsions",
    "importance_weights",
    "relative_ess",
    "sliced_torus_w2",
    "tica_projection",
    "w2",
    "w2_1d",
    "zwanzig",
]
