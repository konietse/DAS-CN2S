# Figures

The notebooks write to `figures/out/`. They expect the data in `data/` and the checkpoints in `ckpt/` (see [README](../README.md)).

| Notebook | Figures | Notes |
|---|---|---|
| `figure_01.ipynb` | 1 | downloads map tiles |
| `figures_02-05.ipynb` | 2 to 5 | |
| `figure_06.ipynb` | 6 | |
| `figure_07.ipynb` | 7 | the evaluation takes about 8 h per loss on one GPU and is cached in `out/quantitative_evaluation_*.csv`, delete them to recompute |
| `figure_08.ipynb`, `figure_09.ipynb` | 8, 9 | |
| `figures_10-11.ipynb` | 10, 11 | run with `CN2S_DX=20` (Fig. 10, default) and `CN2S_DX=4` (Fig. 11) |
| `figure_12.ipynb` | 12 | requires a clone of [Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior), see `dip.py`. The Deep Image Prior baseline is not seeded |
| `figure_13.ipynb` | 13 | |
| `figure_A1.ipynb` | A1 | |
| `figure_A2.ipynb` | A2 | |
| `figure_A3.ipynb` | A3 | |

The Alpine Fault traces in Fig. 1 are from the New Zealand Active Faults Database (GNS Science, `NZAFD_250K_Oct_2023_WGS84.*`).
