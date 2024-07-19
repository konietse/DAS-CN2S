# DAS-CN2S

## Getting Started

### Data Access
- Download the DAS and seismological data used in this study from the [Australian National University Data Commons](doi link under review).

### Preprocessing
- Preprocess the data using the provided Jupyter notebook: `preprocess_data.ipynb`.

### Training Models
- Train the models using the appropriate Jupyter notebook based on your data type:
  - `train_CDAS_synthetic.ipynb` for synthetic DAS data.
  - `train_CDAS_real.ipynb` for finetuning on real DAS data.
  - `train_N2S.ipynb` for other specific training setups.

## Acknowledgements

This project incorporates code from the following repositories:
- [A Self-Supervised Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing data](https://doi.org/10.6084/m9.figshare.14152277.v1) by Martijn van den Ende, Itzhak Lior, Jean-Paul Ampuero, Anthony Sladen, André Ferrari, Cédric Richard, used under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
- [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) by Haris Iqbal, used under the MIT License.