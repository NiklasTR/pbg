"""Single frames: the raw ``DataFrame`` and the tokenised frames that rows pack."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np

from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_smiles


@dataclass(frozen=True, kw_only=True)
class BaseFrame(ABC):
    """Abstract single slice from an ensemble.

    This is the minimal raw frame contract before graph augmentation,
    tokenisation, or diffusion/augmentation-specific derivatives.
    """

    id: int | str = ""
    frame_id: int | None = None
    smiles: str = ""
    sequence: str | None = None
    helm: str | None = None
    environment: str | None = None
    temperature: float | None = None
    energy: float | None = None
    logq: float | None = None
    self_distillation_weight: float = 1.0
    behavior_token_log_probs: np.ndarray | None = None
    behavior_token_ids: np.ndarray | None = None
    frame_valid: bool = True
    stereo_valid: bool | None = None
    source: str | None = None
    generator_checkpoint: str | None = None
    alpha: float | None = None
    epsilon: float | None = None
    time: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.self_distillation_weight)
            or self.self_distillation_weight < 0.0
        ):
            raise ValueError("self_distillation_weight must be finite and non-negative")
        if self.behavior_token_log_probs is not None:
            log_probs = np.asarray(self.behavior_token_log_probs)
            if log_probs.ndim != 1 or not np.all(np.isfinite(log_probs)):
                raise ValueError("behavior_token_log_probs must be a finite vector")
        if self.behavior_token_ids is not None:
            token_ids = np.asarray(self.behavior_token_ids)
            if token_ids.ndim != 1:
                raise ValueError("behavior_token_ids must be a vector")
            if self.behavior_token_log_probs is None or token_ids.shape != np.asarray(
                self.behavior_token_log_probs
            ).shape:
                raise ValueError("behavior token ids and log probabilities must have equal shapes")


@dataclass(frozen=True, kw_only=True)
class DataFrame(BaseFrame):
    """Thin raw frame prior to augmentation or tokenisation."""

    conformer: np.ndarray | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.conformer is not None and (self.conformer.ndim != 2 or self.conformer.shape[1] != 4):
            raise ValueError(f"conformer must have shape (n_atoms, 4), got {self.conformer.shape}")

    @property
    def n_atoms(self) -> int:
        if self.conformer is None:
            return int(atom_numbers_from_smiles(self.smiles).shape[0])
        return int(self.conformer.shape[0])

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        frame_id: int | None = None,
        environment: str | None = "vacuum",
        temperature: float | None = None,
    ) -> DataFrame:
        """Build incomplete raw DataFrame from sequence alone."""
        sequence = sequence.strip()
        return cls(
            conformer=None,
            smiles=sequence_to_smiles(sequence),
            sequence=sequence,
            environment=environment,
            temperature=temperature,
            energy=None,
            logq=None,
            frame_id=frame_id,
        )

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        frame_id: int | None = None,
        environment: str | None = None,
        temperature: float | None = None,
    ) -> DataFrame:
        """Build incomplete raw frame from SMILES alone."""
        smiles = smiles.strip()
        return cls(
            conformer=None,
            smiles=smiles,
            sequence=None,
            environment=environment,
            temperature=temperature,
            energy=None,
            logq=None,
            frame_id=frame_id,
        )


@dataclass(frozen=True, kw_only=True)
class TokenisedFrame(BaseFrame):
    """Augmented and tokenised frame ready for row packing."""

    seg_len: int
    helm_ids: np.ndarray
    coord_ids: np.ndarray
    coord_atom: np.ndarray
    coord_slot: np.ndarray

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.behavior_token_log_probs is not None:
            expected = int(np.asarray(self.coord_ids).size) + 1
            if np.asarray(self.behavior_token_log_probs).shape != (expected,):
                raise ValueError(
                    "behavior_token_log_probs must align with coordinate tokens plus terminator"
                )


@dataclass(frozen=True, kw_only=True)
class AtomConditionedFrame(TokenisedFrame):
    """Tokenised frame with per-atom features in place of a text prompt.

    ``atom_features`` has one row per atom, in coordinate order; see
    ``pbg.models.pbg.robin``.
    """

    atom_features: np.ndarray
