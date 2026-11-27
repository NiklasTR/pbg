# ROBIN

ROBIN is the transferable autoregressive Boltzmann generator of Rehman et al.,
[Autoregressive Boltzmann Generators](https://icml.cc/virtual/2026/poster/66096)
(ICML 2026, arXiv:2606.27361). It uses a CGPT decoder with atom/residue
conditioning and global cross-attention.

```python
import jax
from pbg.models.robin import Robin, sample_robin_conformers
from pbg.models.pbg.system import MolecularSystem

model = Robin.load("robin_epoch949_jax.ckpt")
system = MolecularSystem.from_peptide("AA", environment="water", temperature_K=310.0)
batch = sample_robin_conformers(
    model, system, jax.random.key(42), num_samples=512, batch_size=512,
)
```


Loading expects a trusted converted JAX pickle with `params` and `config`.
Native PyTorch checkpoints are not supported.

## Citation

```bibtex
@inproceedings{rehmann2025autoregressive,
  title        = {{A}utoregressive {B}oltzmann {G}enerators},
  author       = {Rehman, Danyal and Tan, Charlie B. and Bengio, Yoshua and Bose, Avishek Joey and Tong, Alexander},
  booktitle    = {International Conference on Machine Learning (ICML)},
  year         = {2026},
  eprint       = {2606.27361},
  archivePrefix = {arXiv},
  url          = {https://icml.cc/virtual/2026/poster/66096}
}
```
