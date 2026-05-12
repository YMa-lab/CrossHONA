# CrossHONA

Cross-species single-cell integration with a hierarchical staged trainer,
wrapped behind a local-LLM agent + Gradio web UI. Everything runs locally —
no data or prompts leave the host.

## What's in this repo

- **`CrossHONA/`** — the training code: a 4-stage VAE-based
  cross-species integration pipeline (`staged_trainer.py`, `datasets.py`,
  `run_*.sh`).
- **`CrossHONA-Agent/`** — a Gradio web app and a tool-calling agent
  (`api.py`, `agent.py`, `app.py`, `tools/`) that drive preprocessing,
  training, monitoring, and visualization through chat or form input.
- **`SETUP_AGENT.md`** — full setup and usage guide (conda-based). Start there.
- **`CrossHONA-Agent/DOCKER.md`** — alternative containerized setup.

## Quick start

```bash
git clone https://github.com/<org>/CrossHONA.git
cd CrossHONA
export REPO_ROOT=$PWD
conda env create -f environment.yml
conda activate crossspecies
```

Then follow [SETUP_AGENT.md](SETUP_AGENT.md) for Ollama install, staging
data, and launching the web app. No datasets are bundled with the
repository — see §4 for how to bring your own. The MERFISH human ↔ mouse
demo (referenced in §8 and §9) runs in ~3 minutes on a single GPU once the
data is staged.

## Requirements

- Linux workstation or single-GPU node on an HPC cluster
- One CUDA-capable GPU
- ~10 GB free on a data volume for the LLM weights
- Conda / Miniforge

## License

See [LICENSE](LICENSE).
