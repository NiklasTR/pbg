from __future__ import annotations

import math
from dataclasses import dataclass

from pbg.data.bin import geometry_width


@dataclass(frozen=True)
class PBGConfig:
    # ── Tokeniser / coordinate geometry ─────────────────────────────────
    # Fields that earlier configs carried and this release fixes are listed in
    # ``pbg.checkpoint.RETIRED_OPTIONS["model"]``.
    # Coordinate-token vocabulary size: the per-axis bin count. Each atom has
    # one bin token per Cartesian axis, decoded at the bin centre.
    bin_tokens: int = 4096
    # Bin width, i.e. the spatial resolution. The box edge is
    # ``bin_width_pm * (bin_tokens - 1)``.
    bin_width_pm: float = 0.8
    max_seq_len: int = 1028
    # Name of the generated vocabulary: ``core_bin{bin_tokens}`` or
    # ``joint_bin4096`` (``pbg.core.tokeniser``).
    vocab_name: str | None = None

    # ── Qwen backbone architecture (owned directly, no translation layer) ─
    dim: int = 1024
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    # CompleteP alpha=1 residual prefactor. 1.0 leaves the unscaled add.
    residual_scale: float = 1.0
    # Tied μP readout. Input lookup stays E[x]; logits use unembedding_scale * E h.
    unembedding_scale: float = 1.0
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    mixed_precision: bool = False
    # Recompute each transformer block in the training backward pass
    # (jax.checkpoint): same math, far less activation memory.
    remat_blocks: bool = False

    # ── Positional encoding ──────────────────────────────────────────────
    atom_rope_frac: float = 0.5
    level_pe_enabled: bool = True
    level_pe_theta: float = 10_000.0

    # ── Tokeniser ────────────────────────────────────────────────────────
    prompt_format: str = "legacy"
    # Source-order checkpoints retain their old coordinate semantics. New
    # training can opt into the prompt-derived traversal "prompt_canonical_v1"
    # (pbg.core.topology): ff14SB linear peptides, whole-graph canonical
    # cycles and CHEM.
    atom_order: str = "source_v1"
    atom_order_implementation: str | None = None
    # "prefix" conditions on the text prompt. "atom" replaces the prompt with
    # ROBIN's per-atom features for the conditioning ablation; see
    # ``pbg.models.pbg.robin``.
    conditioning: str = "prefix"

    # ── Loss ─────────────────────────────────────────────────────────────
    # "token" averages the NLL over every loss token of the batch. "sample"
    # (ROBIN) averages within each molecule, then over the molecules.
    loss_reduction: str = "token"
    # False (ROBIN) drops the coordinate-block terminator from the targets.
    loss_terminator: bool = True

    # ── CPU collation policy ─────────────────────────────────────────────
    center: bool = True
    # With ``center``, True (ROBIN) subtracts the centre of mass instead of the
    # centroid. A separate key, so that configs that set ``center`` keep their
    # old meaning when they start from a checkpoint.
    center_of_mass: bool = False
    augment: bool = False
    translate: bool | None = None
    # Upper bound on per-batch peptide count for padded sample metadata:
    # ceil(max_samples_factor * batch_size * max_seq_len).
    max_samples_factor: float = 0.004

    # ── Rollout export ───────────────────────────────────────────────────
    stereo_minimum_count: int = 256

    def __post_init__(self) -> None:
        if self.conditioning not in {"prefix", "atom"}:
            raise ValueError(f"unknown conditioning: {self.conditioning!r}")
        if self.loss_reduction not in {"token", "sample"}:
            raise ValueError(f"unknown loss_reduction: {self.loss_reduction!r}")
        if not isinstance(self.loss_terminator, bool):
            raise ValueError(f"loss_terminator must be a bool, got {self.loss_terminator!r}")
        if self.conditioning == "atom" and self.atom_order != "source_v1":
            raise ValueError("atom conditioning needs atoms in source order")
        if self.atom_order == "source_v1":
            if self.atom_order_implementation is not None:
                raise ValueError("source_v1 does not use a canonical atom-traversal implementation")
        else:
            from pbg.core.topology import ATOM_ORDER_IMPLEMENTATIONS
            if self.atom_order not in ATOM_ORDER_IMPLEMENTATIONS:
                raise ValueError(f"unknown atom order: {self.atom_order!r}")
            implementation = ATOM_ORDER_IMPLEMENTATIONS[self.atom_order]
            if self.atom_order_implementation not in {None, implementation}:
                raise ValueError("Checkpoint atom-traversal implementation differs from the installed compiler")
            object.__setattr__(self, "atom_order_implementation", implementation)
        if self.prompt_format not in {"legacy", "compact_kelvin_v1"}:
            raise ValueError(f"unknown prompt format: {self.prompt_format!r}")
        if not isinstance(self.center, bool):
            raise TypeError("center must be a bool")
        if not isinstance(self.center_of_mass, bool):
            raise TypeError("center_of_mass must be a bool")
        if self.bin_tokens < 2:
            raise ValueError(f"bin_tokens must be at least 2, got {self.bin_tokens}")
        if self.bin_width_pm <= 0:
            raise ValueError(f"bin_width_pm must be positive, got {self.bin_width_pm}")
        if not (0.0 <= self.atom_rope_frac <= 1.0):
            raise ValueError(
                f"atom_rope_frac must be in [0, 1], got {self.atom_rope_frac}"
            )
        if self.num_kv_heads < 1:
            raise ValueError(f"num_kv_heads must be >= 1, got {self.num_kv_heads}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for GQA."
            )
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be a positive even integer, got {self.head_dim}"
            )
        if not self.residual_scale > 0.0:
            raise ValueError(
                f"residual_scale must be positive, got {self.residual_scale}"
            )
        if not self.unembedding_scale > 0.0:
            raise ValueError(
                f"unembedding_scale must be positive, got {self.unembedding_scale}"
            )
        if self.stereo_minimum_count < 1:
            raise ValueError("stereo_minimum_count must be positive")
        if int(self.max_seq_len) <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        object.__setattr__(self, "max_seq_len", int(self.max_seq_len))
        if self.max_samples_factor <= 0:
            raise ValueError(
                f"max_samples_factor must be positive, got {self.max_samples_factor}"
            )

    # ── Derived geometry ─────────────────────────────────────────────────

    @property
    def geom_width(self) -> int:
        """Per-atom coordinate token count ``D``."""
        return geometry_width(num_bins=self.bin_tokens, coordinate_method="bin")

    @property
    def max_conditioned_atoms(self) -> int:
        """Atom slots in one packed row under ``conditioning='atom'``."""
        return (self.max_seq_len - 2) // self.geom_width

    def max_samples_per_batch(self, batch_size: int) -> int:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        raw = self.max_samples_factor * float(batch_size) * float(self.max_seq_len)
        return max(1, math.ceil(raw))

    def sequence_length_for_frame(self, num_atoms: int, helm_token_len: int) -> int:
        """Length of ``[BOS | helm | coords | EOS]``."""
        return (
            2
            + int(helm_token_len)
            + int(num_atoms) * self.geom_width
        )

    def sequence_length_for_atoms(self, num_atoms: int, helm_token_len: int = 0) -> int:
        """Backward-compatible alias when HELM length is unknown."""
        return self.sequence_length_for_frame(num_atoms, helm_token_len)

    @property
    def bin_width_nm(self) -> float:
        return float(self.bin_width_pm) * 1.0e-3

    @property
    def box_width_nm(self) -> float:
        """Edge length of the representable cube in nanometres."""
        return self.bin_width_nm * (self.bin_tokens - 1)

    @property
    def half_box_nm(self) -> float:
        return self.box_width_nm / 2.0

    @property
    def coordinate_resolution_pm(self) -> float:
        """Width of the finest represented cell in picometres."""
        return self.bin_width_pm

    @property
    def min_val(self) -> float:
        return -self.half_box_nm

    @property
    def max_val(self) -> float:
        return self.half_box_nm
