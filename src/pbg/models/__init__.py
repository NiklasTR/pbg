"""Generative models: ``pbg`` (PBG model, training and sampling), ``robin`` (the
ROBIN baseline) and ``speculation`` (speculative decoding with a PBG draft and a
PBG target).

This ``__init__`` imports nothing: data-loader worker processes import
:mod:`pbg.models.pbg.loader` and must start without JAX.
"""
