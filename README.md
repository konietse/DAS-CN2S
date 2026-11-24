# DAS-CN2S

Coherent Noise2Self (CN2S): self-supervised removal of coherent traffic noise from Distributed Acoustic Sensing (DAS) data.

## Installation

```bash
git clone https://github.com/konietse/DAS-CN2S.git
cd DAS-CN2S
pip install .             # models, inference and training
pip install ".[all]"      # additionally preprocessing and the figure notebooks
```

This only installs the dependencies. The modules `data`, `masks`, `models` and `inference` are imported from the repository root.

## Pretrained models

The checkpoints of the paper are attached to the [GitHub releases](https://github.com/konietse/DAS-CN2S/releases) as plain PyTorch state dicts.
Download them into `ckpt/`. The model recommended for real data is the fine-tuned CN2S model for 4 m channel spacing:

```bash
mkdir -p ckpt
curl -L -o ckpt/CN2S-finetuned4-complement-128x512-16x180_200.pt \
  https://github.com/konietse/DAS-CN2S/releases/latest/download/CN2S-finetuned4-complement-128x512-16x180_200.pt
```

```python
import numpy as np, torch
import data, inference, models

model = models.CN2SUNet(1, 1, hidden_ch=32, n_layers=4)
model.load_state_dict(torch.load('ckpt/CN2S-finetuned4-complement-128x512-16x180_200.pt', map_location='cpu'))
model.eval()

# x: (channels, samples) strain rate at 50 Hz and 4 m channel spacing
gutter = 1000
x = data.bandpass(np.pad(x[None], ((0, 0), (0, 0), (gutter, gutter))), low=1.0, high=10.0, fs=50, gutter=gutter)[0]
x = x / x.std(axis=-1, keepdims=True)
rec = inference.xreconstruct(model, torch.from_numpy(x), 128, 512, 16, nx_stride=6, nt_stride=50, batch_size=128, dx=4, fs=50)
```

| Checkpoint | Model | Training data | Channel spacing |
|---|---|---|---|
| `CN2S-synthetic4-complement-128x512-16x180_200.pt` | CN2S | synthetic | 4 m |
| `CN2S-synthetic4-center-128x512-16x180_200.pt` | CN2S, center loss (Fig. 7) | synthetic | 4 m |
| `CN2S-finetuned4-complement-128x512-16x180_200.pt` | CN2S | synthetic, fine-tuned on real | 4 m |
| `CN2S-real4-complement-128x512-16x180_200.pt` | CN2S (Fig. A3), retrained with `scripts/train.py` | real only | 4 m |
| `CN2S-synthetic20-complement-32x512-4x180_200.pt` | CN2S | synthetic | 20 m |
| `CN2S-finetuned20-complement-32x512-4x180_200.pt` | CN2S | synthetic, fine-tuned on real | 20 m |
| `N2S-synthetic4-44x2048-4x2048_200.pt` | N2S baseline | synthetic | 4 m |
| `N2S-finetuned4-44x2048-4x2048_200.pt` | N2S baseline | synthetic, fine-tuned on real | 4 m |
| `N2S-synthetic20-11x2048-1x2048_200.pt` | N2S baseline | synthetic | 20 m |
| `N2S-finetuned20-11x2048-1x2048_200.pt` | N2S baseline | synthetic, fine-tuned on real | 20 m |

## Reproducing the paper

### 1. Data

The DAS and seismic data are published at the [ANU Data Commons](https://doi.org/10.25911/rnhb-vw10) (29.5 GB, 341 files).

```bash
python scripts/download_data.py --out data
```

### 2. Preprocessing

```bash
python scripts/preprocess_data.py --data-dir data
```

This simulates strain rates from pairs of nodal seismometers (station coordinates in `metadata/stations_2B_SISSLE.csv`, exported from AusPass),
extracts the manually picked traffic signals and stacks the real DAS recordings into `data/preprocessed/`.

### 3. Training

```bash
# CN2S, 4 m channel spacing
python scripts/train.py --model cn2s --stage synthetic
python scripts/train.py --model cn2s --stage finetune
# variants used in the paper
python scripts/train.py --model cn2s --stage synthetic --target center   # Fig. 7
python scripts/train.py --model cn2s --stage real                        # Fig. A3, real data only
python scripts/train.py --model cn2s --stage synthetic --dx 20
python scripts/train.py --model cn2s --stage finetune --dx 20
# N2S baseline
python scripts/train.py --model n2s --stage synthetic [--dx 20]
python scripts/train.py --model n2s --stage finetune [--dx 20]
```

Each run trains 200 epochs of 10,000 random patches, writes a checkpoint to `ckpt/` every 10 epochs and
logs losses and test reconstructions to `runs/<name>/`. See `python scripts/train.py --help` for all options.

## Citation

```bibtex
@article{konietzny2024cn2s,
  title   = {Unsupervised Coherent Noise Removal From Seismological Distributed Acoustic Sensing Data},
  author  = {Konietzny, Sebastian and Lai, Voon Hui and Miller, Meghan S. and Townend, John and Harmeling, Stefan},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  volume  = {1},
  number  = {4},
  pages   = {e2024JH000356},
  year    = {2024},
  doi     = {10.1029/2024JH000356}
}
```

## License

The code is released under the [MIT License](LICENSE). The data at the [ANU Data Commons](https://doi.org/10.25911/rnhb-vw10) are released under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements

This project incorporates code from the following repositories:
- [A Self-Supervised Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing data](https://doi.org/10.6084/m9.figshare.14152277.v1) by Martijn van den Ende, Itzhak Lior, Jean-Paul Ampuero, Anthony Sladen, André Ferrari, Cédric Richard, used under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
- [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) by Haris Iqbal, used under the MIT License.
