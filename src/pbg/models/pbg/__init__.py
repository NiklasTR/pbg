"""PBG: model, sampling and training.

``config`` (``PBGConfig``), ``transformer`` (the Qwen-style decoder), ``model``
(``PBG``: init, load, loss), ``robin`` (ROBIN's atom conditioning on the PBG
trunk, for the prefix ablation), ``system`` (the molecular system and its
prompt), ``sample`` (decoding and scoring), ``rollout`` (rollout export),
``loader`` (training data), ``optimizer``, ``distributed`` and ``train``.

This ``__init__`` imports nothing: data-loader worker processes import
:mod:`pbg.models.pbg.loader` and must start without JAX.
"""
