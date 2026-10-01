# CrossHONA

<p align="center">
  <img src="figures/CrossHONA.png" alt="CrossHONA overview" width="900">
</p>

CrossHONA is a deep learning framework for cross-species single-cell RNA-seq and spatial transcriptomics integration and cell type annotation. The model jointly incorporates homologous and non-homologous genes within a unified latent embedding space to capture both conserved and species-specific biological signals.

## Repository Structure

- `CrossHONA/`
  - Core model implementation and training pipeline
  - Includes staged training scripts, dataset loaders, and preprocessing utilities

- `CrossHONA-Agent/`
  - Optional Gradio-based interface and utility scripts for experiment management and visualization
  - See `CrossHONA-Agent/DOCKER.md` for the containerized deployment alternative

- `SETUP_AGENT.md`
  - Full setup, daily start-up, troubleshooting, and example chat-tab session for the agent + web UI

- `environment.yml`
  - Conda environment specification

## Installation

```bash
git clone https://github.com/YMa-lab/CrossHONA.git
cd CrossHONA

conda env create -f environment.yml
conda activate crossspecies
```
## Usage
Example training scripts are provided in:

```bash
CrossHONA/run_*.sh
```
Users should prepare their own datasets following the preprocessing procedure described in the manuscript. Before running, edit codedir, datadir, resultdir placeholders to point at your local paths.

## Requirements
- Linux
- Python 3.10
- CUDA-enabled GPU
- Conda or Miniforge

## Source data and plotting scripts

The source data required to reproduce the figures are available through the following [Google Drive link](https://drive.google.com/drive/folders/1j4lI7GLqMaKO8P4jh1TSAqnZmFjc9a8q?usp=sharing). The corresponding plotting scripts are provided in the [`plot_scripts/`](plot_scripts/) directory of this repository. Please download the source data and follow the instructions in the relevant scripts to reproduce each plot.

## License

See [LICENSE](LICENSE).

## Citation
@article{wang2026crosshona,<p>
  author    = {Wang, Ruohan and Zhu, Yu and Gao, Zixiao and Ma, Ying},<p>
  title     = {CrossHONA: Cross-species HOmologous and Non-homologous gene-aware framework for transcriptomics integration and Annotation},<p>
  journal   = {OpenReview},<p>
  year      = {2026},<p>
  url       = {[https://openreview.net/forum?id=hZqR47CiBp#discussion]{https://openreview.net/forum?id=hZqR47CiBp#discussion}}<p>
}
