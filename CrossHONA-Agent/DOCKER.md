# CrossHONA Agent — Docker deployment

Containerized alternative to the conda-based setup in
[../SETUP_AGENT.md](../SETUP_AGENT.md). Everything (data, model, LLM) stays
on the host — the containers have no outbound network needs after the
initial model pull.

## Requirements

- Linux workstation with one NVIDIA GPU (≥16 GB recommended)
- Docker 24+ with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## First-time setup

```bash
cd $REPO_ROOT/CrossHONA-Agent

# 1) Build the app image (5–10 min the first time)
docker compose build

# 2) Start the LLM service so we can pull a model
docker compose up -d ollama
docker exec -it crossspecies_ollama ollama pull qwen2.5:7b   # ~5 GB

# 3) Start the app
docker compose up -d app
```

Open <http://localhost:7860> in a browser.

> **Hardening (optional):** after the model is pulled, edit `docker-compose.yml`
> and uncomment `internal: true` under `networks.internal` to cut all outbound
> traffic from both containers.

## Two ways to use the running app

### Form tab

Upload `ref.h5ad`, `target.h5ad`, the homologous-gene `.csv`, name a project, and
click **Run preprocess**. Then set epochs / latent dim / etc. and **Start
training**. The Monitor section shows live progress.

### Chat tab

Plain English. The agent calls the same tools the Form uses.

> *Inspect uploads/mouse_visium.h5ad. Then preprocess it against
> uploads/human_visium.h5ad using uploads/homo.csv into project demo.*

> *Train demo for 50 epochs as run_a, latent_dim 64.*

> *What's the status of demo/run_a?*

> *Plot the UMAP for demo/run_a.*

The agent never executes shell commands or arbitrary code — only the tools in
`tools/`. Every tool call is logged to `results/audit.log`.

See [../SETUP_AGENT.md](../SETUP_AGENT.md) §9 for a fuller chat-tab walkthrough.

## GPU sharing

There is exactly one GPU and both training and the LLM want it.

- `qwen2.5:7b` (Q4) sits at ~5–6 GB. Most training runs fit alongside it.
- If a training run OOMs, stop the LLM first:
  `docker exec crossspecies_ollama ollama stop qwen2.5:7b`

## Troubleshooting

- **"no preprocessed data found"** — run preprocess for that project first.
- **Status shows `not_started` but `pid` exists** — the subprocess crashed early;
  check `results/<project>/<run>/train.log`.
- **Chat replies "I cannot..."** — usually means the model didn't pick a tool.
  Be more explicit: name the project and run.

## Privacy guarantees

- App port is bound to `127.0.0.1` only (see `docker-compose.yml`).
- `share=False` is hard-coded in `app.py` — Gradio will not create a public tunnel.
- `HF_HUB_OFFLINE`, `WANDB_MODE=offline`, `GRADIO_ANALYTICS_ENABLED=False` set
  in the image.
- LLM is local (Ollama). The OpenAI SDK only ever talks to `http://ollama:11434`.
