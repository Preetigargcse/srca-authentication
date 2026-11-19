# SRCA: Risk-Calibrated Selective Image Authentication

Code, data and results for the paper:

> P. Garg, V. Jain, G. Manteghi, M. Gautam. *SRCA: Risk-Calibrated Selective Image Authentication through Joint Payload Feasibility and Diagnostic-Loss Control.* Submitted to the Journal of Information Security and Applications.

SRCA is a blind, block-based selective fragile watermark whose operating point
λ = (τ, ω, b, r) (protection threshold, carrier fraction, LSB planes, recovery bits) is chosen
from calibration data with Learn-then-Test. Three risks are certified jointly: the
relevance-weighted tamper-miss risk, the probability that the payload does not fit, and the
probability of exceeding a relevance-weighted distortion budget.

## Repository layout

| Path | Contents |
| --- | --- |
| `code/srca_core.py` | Relevance map, embedding, blind verification (keyed BLAKE2b MACs, border header) |
| `code/data_attacks.py` | Patch pool construction and the tampering threat model (5 operations) |
| `code/precompute.py` | Runs embed → tamper → verify for every image and all 384 configurations |
| `code/analysis.py` | Hoeffding–Bentkus / binomial p-values, LTT selection, evaluation over splits |
| `code/run_experiments.py` | All experiments in Section 5 (main comparison, sweeps, shift) |
| `code/breakdown.py` | Per-operation and per-source analysis (Table 6) |
| `code/fig_*.py` | Figures 1–6 |
| `code/export_dataset.py` | Writes / verifies the 760-patch dataset in `data/` |
| `data/patches/` | The 760 evaluation patches (256 × 256, 8-bit grayscale PNG) |
| `data/manifest.csv` | Patch id, domain, source image, file name, SHA-256 of the pixels |
| `results/loss_tables/` | Per-image losses for every configuration (targeted and uniform threat models) |
| `results/*.json` | Numbers reported in the paper |
| `figures/` | Figures 1–6 as used in the manuscript |

## Installation

```bash
git clone <this-repository-url>
cd srca-authentication
pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, scikit-image 0.26, Matplotlib 3.11 and Pillow 12.3 on CPU.
No GPU is required.

## Reproducing the paper

All commands are run from `code/`.

```bash
cd code
./run_all.sh            # full reproduction, about 16 min on a 2-core CPU
```

Or step by step:

```bash
python precompute.py 1.0 ../results/loss_tables/tgt_0.npz 0 2   # targeted threat model, shard 0 of 2
python precompute.py 1.0 ../results/loss_tables/tgt_1.npz 1 2
python precompute.py 0.0 ../results/loss_tables/uni_0.npz 0 2   # uniform threat model
python precompute.py 0.0 ../results/loss_tables/uni_1.npz 1 2
python run_experiments.py      # -> results/results.json   (Tables 3-5, Figs. 3, 5, 6)
python breakdown.py            # -> results/results_breakdown.json (Table 6)
python fig_framework.py; python fig_example.py; python fig_results.py   # -> figures/
```

The stored loss tables let you skip the first four commands; `run_experiments.py` then takes
about 40 s. A quick check of the pipeline on three images:

```bash
SRCA_LIMIT=3 python precompute.py 1.0 /tmp/smoke.npz 0 2
```

All random choices (patch crops, attacks, splits, keys) use fixed seeds, so the numbers in
`results/` are reproduced exactly.

## Mapping to the paper

| Paper item | Script | Output key |
| --- | --- | --- |
| Fig. 1 | `fig_framework.py` | `figures/fig1_framework.pdf` |
| Fig. 2, risk landscape | `fig_results.py` | `results.json → landscape` |
| Table 3, Fig. 3 | `run_experiments.py` | `results.json → main`, `main_samples` |
| Fig. 4, qualitative example | `fig_example.py` | `figures/fig2_example.png` |
| Table 4, Fig. 5 | `run_experiments.py` | `alpha_sweep`, `ncal_sweep`, `dmax_sweep` |
| Table 5, Fig. 6 | `run_experiments.py` | `shift` |
| Table 6 | `breakdown.py` | `results_breakdown.json` |

## Using SRCA on your own images

```python
from srca_core import relevance_map, embed, verify
w = relevance_map(img)                          # img: 256x256 uint8 grayscale
wm, meta = embed(img, w, (0.15, 0.2, 1, 0))     # deployed configuration from the paper
res = verify(received_img)                      # blind: needs only the key
res['flags']                                    # per-block tamper flags (1024 blocks)
```

The key is set by `KEY` in `srca_core.py`; replace it with your own secret before any real use.
To calibrate on a new image source, add the images to the pool in `data_attacks.py` and rerun
the pipeline.

## Data

The 760 patches are random 256 × 256 crops (with rotation and flipping, seed 0) of 17 test
images distributed with scikit-image: 13 natural images (520 patches) and 4 biomedical images
(240 patches). `python export_dataset.py --verify` checks that a local regeneration matches
the manifest. Source images and their licences are listed in `data/SOURCES.md`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

Code: MIT licence (see `LICENSE`). Data: see `data/SOURCES.md`; the source images are
CC0 or public domain unless stated otherwise there.
