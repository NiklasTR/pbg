"""Configuration for the historical converted ROBIN checkpoints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RobinConfig:
    num_bins: int
    bin_width_pm: float
    dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float
    max_num_atoms: int
    coordinate_method: str = 'bin'
    undiscretize_method: str = 'center'
    mixed_precision: bool = False
    token_mode: str = 'per_dim'
    loss_reduction: str = 'per_sample'

    def __post_init__(self):
        if self.num_bins < 2 or self.bin_width_pm <= 0:
            raise ValueError('ROBIN requires at least two bins and a positive bin width')
        if self.dim <= 0 or self.dim % 2 or self.num_heads <= 0 or self.dim % self.num_heads:
            raise ValueError('ROBIN requires an even dimension divisible by num_heads')
        if self.num_layers < 1 or self.max_num_atoms < 1:
            raise ValueError('num_layers and max_num_atoms must be positive')
        if self.coordinate_method != 'bin' or self.token_mode not in ('per_dim', 'combined'):
            raise ValueError('Unsupported ROBIN coordinate or token layout')
        if self.undiscretize_method not in ('center', 'uniform', 'uniform_noise'):
            raise ValueError('Unsupported ROBIN reconstruction')

    @property
    def min_val(self):
        return -self.bin_width_pm * 1e-3 * (self.num_bins - 1) / 2

    @property
    def max_val(self):
        return -self.min_val

    @classmethod
    def from_checkpoint(cls, values):
        """Apply the same legacy normalization as BoltzmannGenerator.load."""
        values = dict(values)
        if values.get('coordinate_method') is None:
            binning = values.get('binning')
            values['coordinate_method'] = 'bin' if binning in (None, 'linear_scale', 'bin') else binning
        if 'bin_width_pm' not in values:
            values['bin_width_pm'] = (
                (values['max_val'] - values['min_val'])
                * values.get('standardize_std', 1.0) / (values['num_bins'] - 1) * 1e3
            )
        for name in (
            'discretise', 'bin_width', 'min_val', 'max_val', 'standardize_std',
            'temperature', 'clip_reweighting_logits', 'm_spectrum', 'use_spectrum',
            'model_type', 'backbone', 'fragment_config_path', 'bos_token_id',
            'eos_token_id', 'mask_token_id', 'pad_token_id', 'max_atomic_number',
            'decoder_atom_shuffle', 'num_laplacian_modes', 'graph_rope_frac', 'binning',
        ):
            values.pop(name, None)
        return cls(**values)
