"""Vocabularies, prompt tokenisation and coordinate tokens of PBG.

The core and joint vocabularies are deterministic functions of the
coordinate bin count (``core_vocab``, ``joint_vocab``); they equal the JSON
files that the paper checkpoints were trained with. A vocabulary holds
``core_special_tokens`` (``<BOS>`` = ``<PAD>``), the greedy longest-match
``core_tokens``, six ``dipper_added_tokens`` delimiters, and a contiguous
block of ``coordinate_tokens`` (token = bin index + ``coord_base_id``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from beartype import beartype

from pbg.data.bin import apply_discretise, undo_discretise

if TYPE_CHECKING:
    from pbg.core.frame import DataFrame

EXTRA_TOKENS = (
    "water",
    "vacuum",
    "PEPTIDE",
    "CHEM",
    "$$$$",
    "cc",
    "CC",
    "(=",
    "ccc",
    "@@",
    "NC",
    "CO",
    "Cl",
    "Br",
    "NH",
    "methanol",
    "ethanol",
    "acetone",
    "acetonitrile",
    "dimethyl sulfoxide",
    "octanol",
    "toluene",
    "hexane",
    "chloroform",
    "[MASK]",
)

DELIMITERS = (
    "<|env_start|>",
    "<|env_end|>",
    "<|helm_start|>",
    "<|helm_end|>",
    "<|coords_start|>",
    "<|coords_end|>",
)

# The joint vocabulary reuses unused core token IDs for force-field words, so
# that it keeps the 4224-row embedding of core_bin4096.
JOINT_TOKEN_REPLACEMENTS = (
    (4213, "methanol", "gaff2"),
    (4214, "ethanol", "am1bcc"),
    (4215, "acetone", "obc1"),
    (4216, "acetonitrile", "tip3p"),
    (4217, "dimethyl sulfoxide", "rf"),
    (4218, "octanol", "nm"),
    (4223, None, "ff14sb"),
)


def core_vocab(num_bins: int, *, alignment: int = 128) -> dict[str, object]:
    """Return the ``core_bin{num_bins}`` vocabulary."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if alignment < 1:
        raise ValueError("alignment must be positive")
    printable = "".join(chr(codepoint) for codepoint in range(32, 127))
    core_tokens = {token: index + 1 for index, token in enumerate(printable)}
    delimiter_base = 1 + len(core_tokens)
    delimiters = {
        token: delimiter_base + index for index, token in enumerate(DELIMITERS)
    }
    coordinate_base = delimiter_base + len(delimiters)
    width = max(4, len(str(num_bins - 1)))
    coordinates = {
        f"coord_bin_{index:0{width}d}": coordinate_base + index
        for index in range(num_bins)
    }
    extra_base = coordinate_base + num_bins
    core_tokens.update(
        {token: extra_base + index for index, token in enumerate(EXTRA_TOKENS)}
    )
    semantic_next_id = extra_base + len(EXTRA_TOKENS)
    next_id = ((semantic_next_id + alignment - 1) // alignment) * alignment
    padding_ranges = (
        [] if semantic_next_id == next_id else [[semantic_next_id, next_id]]
    )
    return {
        "core_special_tokens": {"<BOS>": 0, "<PAD>": 0},
        "core_tokens": core_tokens,
        "dipper_added_tokens": delimiters,
        "coordinate_tokens": coordinates,
        "meta": {
            "version": 4,
            "vocab_type": "core_ascii",
            "printable_ascii_count": 95,
            "extra_token_count": len(EXTRA_TOKENS),
            "semantic_next_id": semantic_next_id,
            "next_id": next_id,
            "padding_ranges": padding_ranges,
            "padding_alignment": alignment,
        },
    }


def joint_vocab() -> dict[str, object]:
    """Return the ``joint_bin4096`` vocabulary of PBG-Joint."""
    vocab = copy.deepcopy(core_vocab(4096))
    meta = vocab["meta"]
    assert isinstance(meta, dict)
    if meta["next_id"] != 4224 or meta["semantic_next_id"] != 4223:
        raise ValueError("expected the original 4224-row core_bin4096 allocation")
    tokens = vocab["core_tokens"]
    assert isinstance(tokens, dict)
    for token_id, old, new in JOINT_TOKEN_REPLACEMENTS:
        if old is not None and tokens.pop(old) != token_id:
            raise ValueError(f"unexpected token ID for {old}")
        if new in tokens:
            raise ValueError(f"target token already exists: {new}")
        tokens[new] = token_id
    meta.update(
        version=5,
        vocab_type="joint_ascii",
        vocab_family="joint",
        vocab_name="joint_bin4096",
        extra_token_count=26,
        semantic_next_id=4224,
        padding_ranges=[],
        prompt_format="compact_kelvin_v1",
        source_vocab_name="core_bin4096",
    )
    validate_vocab(vocab, expected_coordinate_count=4096)
    return vocab

def upgrade_core_mask_token(vocab: dict[str, object]) -> dict[str, object]:
    """Upgrade the original core-64k solvent rows to chloroform and ``[MASK]``."""
    core_tokens = vocab.get("core_tokens")
    if not isinstance(core_tokens, dict):
        return vocab
    if core_tokens.get("decane") == 65662 and core_tokens.get("chloroform") == 65663:
        del core_tokens["decane"]
        core_tokens["chloroform"] = 65662
        core_tokens["[MASK]"] = 65663
        meta = vocab.get("meta")
        if isinstance(meta, dict):
            meta["version"] = 4
    return vocab


def validate_vocab(
    vocab: dict[str, object], *, expected_coordinate_count: int | None = None
) -> None:
    """Validate the ID invariants relied on by arithmetic coordinate encoding."""
    coordinate_tokens = vocab.get("coordinate_tokens")
    meta = vocab.get("meta")
    if not isinstance(coordinate_tokens, dict) or not coordinate_tokens:
        raise ValueError("vocab must contain non-empty coordinate_tokens")
    if not isinstance(meta, dict):
        raise ValueError("vocab must contain a meta object")

    coord_ids = [int(value) for value in coordinate_tokens.values()]
    if len(set(coord_ids)) != len(coord_ids):
        raise ValueError("coordinate token IDs must be unique")
    ordered = sorted(coord_ids)
    if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
        raise ValueError("coordinate token IDs must be contiguous")
    if (
        expected_coordinate_count is not None
        and len(ordered) != expected_coordinate_count
    ):
        raise ValueError(
            f"vocab has {len(ordered)} coordinate tokens, expected "
            f"{expected_coordinate_count}"
        )

    next_id = int(meta["next_id"])
    semantic_next_id = int(meta.get("semantic_next_id", next_id))
    if ordered[-1] >= semantic_next_id or semantic_next_id > next_id:
        raise ValueError(
            "coordinate, semantic, and model vocabulary bounds are inconsistent"
        )

    delimiters = vocab.get("dipper_added_tokens")
    if not isinstance(delimiters, dict) or set(delimiters) != set(DELIMITERS):
        raise ValueError("vocab must contain the six required PBG delimiters")
    core_special = vocab.get("core_special_tokens")
    core_tokens = vocab.get("core_tokens")
    if not isinstance(core_special, dict) or not isinstance(core_tokens, dict):
        raise ValueError("vocab must be a core or joint PBG vocabulary")
    if set(core_special) != {"<BOS>", "<PAD>"}:
        raise ValueError("core vocab must contain BOS and PAD special tokens")
    if int(core_special["<BOS>"]) != int(core_special["<PAD>"]):
        raise ValueError("core BOS and PAD must share one ID")
    delimiter_ids = [int(value) for value in delimiters.values()]
    if len(set(delimiter_ids)) != len(delimiter_ids):
        raise ValueError("PBG delimiter IDs must be unique")
    if max(delimiter_ids) >= ordered[0]:
        raise ValueError("PBG delimiters must precede the coordinate block")
    core_ids = [int(value) for value in core_tokens.values()]
    if len(set(core_ids)) != len(core_ids):
        raise ValueError("core token IDs must be unique")
    allocated = core_ids + delimiter_ids + ordered + [int(core_special["<BOS>"])]
    if len(set(allocated)) != len(allocated):
        raise ValueError("core token groups must have disjoint IDs")
    if min(allocated) < 0 or max(allocated) >= semantic_next_id:
        raise ValueError("core token IDs are outside semantic vocabulary bounds")
    if set(allocated) != set(range(semantic_next_id)):
        raise ValueError("core semantic token IDs must cover a contiguous range")

    padding_ranges = meta.get("padding_ranges", [])
    expected_ranges = [] if semantic_next_id == next_id else [[semantic_next_id, next_id]]
    if padding_ranges != expected_ranges:
        raise ValueError(
            "padding_ranges must describe exactly the unnamed terminal model rows"
        )


@dataclass(frozen=True)
class VocabSchema:
    """Terminal IDs and id groups of one core or joint vocabulary."""

    bos_id: int
    pad_id: int
    coord_base_id: int
    coord_ids: tuple[int, ...]
    #: Token that terminates the coordinate block (``<|coords_end|>``).
    terminator_id: int
    mask_id: int

    @staticmethod
    def of(vocab: dict[str, object]) -> VocabSchema:
        validate_vocab(vocab)
        coord_ids = tuple(sorted(int(v) for v in vocab["coordinate_tokens"].values()))
        special = vocab["core_special_tokens"]
        return VocabSchema(
            int(special["<BOS>"]),
            int(special["<PAD>"]),
            coord_ids[0],
            coord_ids,
            int(vocab["dipper_added_tokens"]["<|coords_end|>"]),
            mask_id=int(vocab["core_tokens"]["[MASK]"]),
        )

    def decode_bias(self, vocab_size: int, *, allow_terminator: bool) -> np.ndarray:
        """``-inf`` everywhere except coordinate tokens (and optionally terminator)."""
        bias = np.full((vocab_size,), -np.inf, dtype=np.float32)
        bias[np.asarray(self.coord_ids, dtype=np.int32)] = 0.0
        if allow_terminator:
            bias[self.terminator_id] = 0.0
        return bias

class CoreTokenizer:
    """Greedy longest-match tokenizer with printable-ASCII fallback."""

    def __init__(self, vocab: dict[str, object]) -> None:
        tokens = vocab.get("core_tokens")
        if not isinstance(tokens, dict) or not tokens:
            raise ValueError("core vocab must contain core_tokens")
        self.encoder = {str(token): int(token_id) for token, token_id in tokens.items()}
        self._terms = tuple(
            sorted(self.encoder, key=lambda token: (-len(token), token))
        )
        # Keep the exact longest-match order, but inspect only candidates
        # that can start at the current character.
        by_first: dict[str, list[str]] = {}
        for token in self._terms:
            if not token:
                raise ValueError("core vocabulary must not contain an empty token")
            by_first.setdefault(token[0], []).append(token)
        self._terms_by_first = {key: tuple(values) for key, values in by_first.items()}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            token = next(
                (term for term in self._terms_by_first.get(text[index], ()) if text.startswith(term, index)),
                None,
            )
            if token is None:
                raise ValueError(
                    f"core vocabulary cannot encode character {text[index]!r} "
                    f"at position {index}"
                )
            ids.append(self.encoder[token])
            index += len(token)
        return ids


class QwenJointTokeniser:
    """Encodes a frame's non-coordinate prompt prefix for the joint vocab.

    Prefix layout: ``<|env_start|> core(env text) <|env_end|> <|helm_start|>
    core(helm) <|helm_end|>``. All other vocab IDs are read via
    :class:`VocabSchema`, so this class only
    holds the four delimiters its two encode methods need.
    """

    def __init__(self, vocab: dict[str, object], *, prompt_format: str = "legacy") -> None:
        if prompt_format not in {"legacy", "compact_kelvin_v1"}:
            raise ValueError(f"unknown prompt format: {prompt_format!r}")
        if "core_tokens" not in vocab:
            raise ValueError("the prompt tokeniser requires a core vocabulary (core_tokens)")
        self.prompt_format = prompt_format
        delim = vocab["dipper_added_tokens"]
        assert isinstance(delim, dict)
        self._env = (int(delim["<|env_start|>"]), int(delim["<|env_end|>"]))
        self._helm = (int(delim["<|helm_start|>"]), int(delim["<|helm_end|>"]))
        self._tokenizer = CoreTokenizer(vocab)
        # Instances now live for a worker's lifetime. Bound retained strings
        # and arrays instead of caching every representation in the corpus.
        self._helm_cache: dict[str, np.ndarray] = {}
        self._environment_cache: dict[str, list[int]] = {}

    def encode_helm(self, helm: str) -> np.ndarray:
        """Return <|helm_start|> + core(helm) + <|helm_end|> ids (cached)."""
        cached = self._helm_cache.get(helm)
        if cached is not None:
            return cached
        start, end = self._helm
        ids = np.asarray([start, *self._tokenizer.encode(helm), end], dtype=np.int32)
        ids.flags.writeable = False
        if len(self._helm_cache) >= 4096:
            self._helm_cache.pop(next(iter(self._helm_cache)))
        self._helm_cache[helm] = ids
        return ids

    def encode_prompt(
        self, helm: str, *, environment: str | None, temperature_K: float | None
    ) -> np.ndarray:
        """Format conditioning text without changing the physical temperature.

        Legacy checkpoints retain their exact prompt convention. Compact core
        prompts remove environment whitespace and round half kelvins upward.
        HELM chemistry and every vocabulary ID remain unchanged.
        """
        temperature = None if temperature_K is None else f"{float(temperature_K):g}K"
        description = ", ".join(filter(None, (temperature, environment)))
        if self.prompt_format == "compact_kelvin_v1":
            import math

            if temperature_K is not None:
                value = float(temperature_K)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("temperature must be finite and positive")
                temperature = f"{math.floor(value + 0.5)}K"
            description = ",".join(filter(None, (temperature, environment)))
            description = "".join(description.split())
        env_ids = self._environment_cache.get(description)
        if env_ids is None:
            env_ids = self._tokenizer.encode(description)
            if len(self._environment_cache) >= 512:
                self._environment_cache.pop(next(iter(self._environment_cache)))
            self._environment_cache[description] = env_ids
        start, end = self._env
        out = [start, *env_ids, end, *self.encode_helm(helm).tolist()]
        arr = np.asarray(out, dtype=np.int32)
        arr.flags.writeable = False
        return arr

def make_tokeniser(vocab: dict[str, object], *, prompt_format: str = "legacy") -> QwenJointTokeniser:
    """Return the prompt tokeniser for a core or joint vocabulary."""
    return QwenJointTokeniser(vocab, prompt_format=prompt_format)

def apply_tokeniser_conformer(
    frame: "DataFrame",
    *,
    method: str,
    num_bins: int,
    min_val: float,
    max_val: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin the coordinates of one frame.

    Returns the flattened 0-based bin indices (three per atom, x/y/z) and,
    for each index, its atom and axis position.
    """
    if method != "bin":
        raise ValueError(f"Unknown coordinate tokeniser method {method!r}; use 'bin'")
    if frame.conformer is None:
        raise ValueError("coordinate tokenisation needs a conformer")
    conformer = np.asarray(frame.conformer)
    if conformer.ndim != 2 or conformer.shape[1] != 4:
        raise ValueError("frame.conformer must have shape (N, 4) as Zxyz")
    geometry = np.asarray(
        apply_discretise(conformer[None, :, 1:4], num_bins=num_bins, min_val=min_val, max_val=max_val)
    )[0].astype(np.int32)
    n_atoms, width = geometry.shape
    return (
        geometry.reshape(-1),
        np.repeat(np.arange(n_atoms, dtype=np.int32), width),
        np.tile(np.arange(width, dtype=np.int32), n_atoms),
    )


@beartype
def undo_tokeniser_coordinate(
    token_ids: np.ndarray,
    method: str,
    *,
    num_bins: int,
    min_val: float,
    max_val: float,
    decode_method: str = "center",
    key: object | None = None,
) -> np.ndarray:
    """Invert bin coordinate tokenisation."""
    ids = np.asarray(token_ids)
    if ids.ndim == 0:
        raise ValueError("token_ids must have at least one dimension")
    if method != "bin":
        raise ValueError(f"Unknown coordinate tokeniser method {method!r}; use 'bin'")
    if ids.shape[-1] != 3:
        raise ValueError("bin coordinate tokens must have final width 3")
    return undo_discretise(ids, num_bins=num_bins, min_val=min_val, max_val=max_val, method=decode_method, key=key)


def load_tokeniser_default(vocab_name: str) -> dict[str, object]:
    """Return a new copy of the vocabulary named ``core_bin{N}`` or ``joint_bin4096``."""
    if vocab_name == "joint_bin4096":
        return joint_vocab()
    if vocab_name.startswith("core_bin") and vocab_name[len("core_bin"):].isdigit():
        return core_vocab(int(vocab_name[len("core_bin"):]))
    raise ValueError(f"unknown vocabulary {vocab_name!r}; use core_bin<N> or joint_bin4096")
