# Cross-Species Agent — Setup Guide

How to deploy the local-LLM agent + Gradio UI on a single-GPU Linux machine.
Everything runs locally; no data or prompts leave the host.

Do sections **1 → 2 → 3 → 4 → 5 → 7**. Section 6 is the daily start-up routine.
Section 9 is a copy-pasteable transcript of using the chat tab.

HPC users (SLURM, reverse-proxy access to the web UI): see **Appendix A**.

---

## 0. Repository layout

After cloning, the repo looks like:

```
CrossHONA/                      # $REPO_ROOT (the cloned repo)
├── README.md
├── SETUP_AGENT.md              # this file (conda-based setup)
├── LICENSE
├── .gitignore
├── environment.yml
├── CrossHONA/                  # training code (Python package)
│   ├── staged_trainer.py
│   ├── datasets.py
│   └── run_*.sh
└── CrossHONA-Agent/            # agent + Gradio web app
    ├── api.py / agent.py / app.py
    ├── tools/{data,training,viz}.py
    ├── run_merfish_demo.py     # MERFISH demo as a Python script (alt to §8)
    ├── Dockerfile / docker-compose.yml / .dockerignore
    ├── DOCKER.md               # containerized setup (alternative to this file)
    ├── data/                   # bring your own datasets (see §4); gitignored
    └── results/                # outputs land here; gitignored
```

> Prefer Docker? See [CrossHONA-Agent/DOCKER.md](CrossHONA-Agent/DOCKER.md)
> for the containerized path — same UI, simpler dependency story.

Pick a location and clone:

```bash
git clone https://github.com/<org>/CrossHONA.git
cd CrossHONA
export REPO_ROOT=$PWD
```

`$REPO_ROOT` is referenced throughout this guide. Add it to your `~/.bashrc`
if you'd rather not re-export each session.

---

## 1. Conda env (`crossspecies`)

Required packages are defined in `environment.yml` (Python 3.10).

```bash
cd $REPO_ROOT
conda env create -f environment.yml          # ~5 min, downloads ~4 GB
conda activate crossspecies
```

Sanity check:
```bash
cd $REPO_ROOT/CrossHONA-Agent
export PYTHONNOUSERSITE=1                                  # see §7 gotcha
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # see §7 gotcha
python -c "from api import CrossSpeciesModel; print('OK')"
```

`api` is `CrossHONA-Agent/api.py`. The `from api import …` only works when
the working directory is `CrossHONA-Agent/` (it's not on `PYTHONPATH`).
If you'd rather not `cd` each time, set
`export PYTHONPATH=$REPO_ROOT/CrossHONA-Agent:$PYTHONPATH`.

---

## 2. Install Ollama (local LLM runtime)

If you don't have sudo (e.g. shared HPC), install to `~/.local`.

```bash
cd /tmp
curl -fL --progress-bar \
  https://github.com/ollama/ollama/releases/download/v0.5.7/ollama-linux-amd64.tgz \
  -o ollama.tgz

ls -lh ollama.tgz       # ~1.5 GB; if it's a few bytes, the URL changed
file ollama.tgz         # must say "gzip compressed data"

mkdir -p ~/.local
tar -xzf /tmp/ollama.tgz -C ~/.local
~/.local/bin/ollama --version
```

> v0.5.7 is what's been verified end-to-end. Newer Ollama releases should
> work — bump the URL accordingly.

Persist `PATH` and put models on a volume with space (`~/` often has tight
quota; each model is ~5 GB):

```bash
export OLLAMA_MODELS=$HOME/ollama_models   # or any path with ≥10 GB free
mkdir -p "$OLLAMA_MODELS"
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo "export OLLAMA_MODELS=$OLLAMA_MODELS" >> ~/.bashrc
source ~/.bashrc
# verify
echo $OLLAMA_MODELS
which ollama
```

> Setting these in `.bashrc` matters: every fresh SSH session needs them.
> If you only `export` in one shell, the next session re-downloads models
> into `~/.ollama/` and silently eats your home quota.

---

## 3. Get a GPU and pull the model

Ollama needs a GPU for usable speed (CPU works but is ~10× slower).

```bash
nvidia-smi    # confirm one GPU is visible
hostname      # remember the node name — referenced later in Appendix A
```

Start the daemon and pull `qwen2.5:7b`:

```bash
nohup ollama serve > ~/ollama.log 2>&1 &
sleep 3
curl http://localhost:11434/api/tags
# expect {"models":[]} on first run

ollama pull qwen2.5:7b       # ~5 GB, only needed once
ollama list
export OLLAMA_MODEL=qwen2.5:7b
```

> Why `qwen2.5:7b` and not `llama3.1:8b`? Tool-calling accuracy. Llama 3.1
> hallucinates parameters and invents filenames frequently enough that
> the agent stops being usable. Qwen2.5 is the model that's actually used.

If port 11434 is taken:
```bash
OLLAMA_HOST=127.0.0.1:11500 nohup ollama serve > ~/ollama.log 2>&1 &
export OLLAMA_BASE_URL=http://localhost:11500/v1
```

---

## 4. Stage the data

**No data is shipped with this repository.** For safety, the agent can only
read files under `$REPO_ROOT/CrossHONA-Agent/data/`, so that's where you
need to put (or symlink) your `.h5ad` and `.tsv`/`.csv` files. The agent's
`list_datasets` walks the tree.

Drop files directly:

```bash
cd $REPO_ROOT/CrossHONA-Agent/data
mkdir -p my_dataset
cp /path/to/sample.h5ad my_dataset/
```

Or symlink — preferred when the data lives on shared storage and you don't
want to duplicate it:

```bash
cd $REPO_ROOT/CrossHONA-Agent/data
mkdir -p my_dataset
ln -sfn /path/to/shared/storage/sample.h5ad my_dataset/sample.h5ad
```

**To run the MERFISH demo (§8 and §9)** you need three files placed at
`CrossHONA-Agent/data/merfish/`:

```
merfish/human_H19_STG_4000.h5ad
merfish/mouse_mouse1_242.h5ad
merfish/Human_Mouse.tsv
```

These are not in the repo. Obtain them separately (see the project page or
contact the maintainers) and stage them with `cp` or `ln -sfn` as above.
Without these files, sections §8 and §9 will not run.

---

## 5. Daily start-up (every time you log in)

```bash
# 1. Activate env + set required env vars (see §7 for why)
conda activate crossspecies
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 2. Start ollama if it isn't already
pgrep -f "ollama serve" >/dev/null || nohup ollama serve > ~/ollama.log 2>&1 &
sleep 2
curl -s http://localhost:11434/api/tags | python -m json.tool

# 3. Launch the web app
cd $REPO_ROOT/CrossHONA-Agent

export CROSSSPECIES_SCRIPTS_DIR=$REPO_ROOT/CrossHONA
export CROSSSPECIES_DATA_DIR=$PWD/data
export CROSSSPECIES_RESULTS_DIR=$PWD/results
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=qwen2.5:7b

python app.py
# wait for: "Running on local URL:  http://0.0.0.0:7860"
```

Open `http://localhost:7860` in a browser. If you're on a remote machine,
forward the port however your environment supports it — SSH tunnel,
JupyterHub proxy, Open OnDemand or another reverse proxy, etc. See
**Appendix A** for HPC tips.

To avoid retyping env vars each session, drop this in `~/start_app.sh`:

```bash
#!/bin/bash
cd "$REPO_ROOT/CrossHONA-Agent"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CROSSSPECIES_SCRIPTS_DIR=$REPO_ROOT/CrossHONA
export CROSSSPECIES_DATA_DIR=$PWD/data
export CROSSSPECIES_RESULTS_DIR=$PWD/results
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=qwen2.5:7b
pkill -9 -f "python.*app.py" 2>/dev/null
sleep 2
nohup python -u app.py > ~/app.log 2>&1 &
echo "starting... tail ~/app.log to watch"
```

`chmod +x ~/start_app.sh`, then run it. Always use `python -u` (unbuffered)
so any later traceback actually makes it into `~/app.log`.

---

## 6. Quick "is it alive" checklist

```bash
# 1. conda env exists
conda env list | grep crossspecies

# 2. ollama installed
~/.local/bin/ollama --version

# 3. ollama daemon up and a model is loaded
curl -s http://localhost:11434/api/tags | python -m json.tool

# 4. app importable
cd $REPO_ROOT/CrossHONA-Agent
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -c "from api import CrossSpeciesModel; from agent import Agent; print('OK')"

# 5. data dir exists and is reachable (must be staged separately — see §4)
ls -lH data/
```

---

## 7. Common gotchas (alphabetical)

- **`ECONNREFUSED 0.0.0.0:7860` from a reverse proxy** — `python app.py` isn't
  running, or it crashed. Check the terminal/log where you launched it.
- **`GLIBCXX_3.4.30 not found`** — system libstdc++ is too old for
  numba/llvmlite. The conda env ships a newer one; put it first:
  `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`
- **`gradio.exceptions.Error: Data incompatible with messages format`** — old
  Chatbot tuple format vs Gradio 6.x. The shipped `app.py` uses dict messages.
- **`ModuleNotFoundError: No module named 'api'`** — you're not in
  `CrossHONA-Agent/`. Either `cd` there or
  `export PYTHONPATH=$REPO_ROOT/CrossHONA-Agent:$PYTHONPATH`.
- **`ModuleNotFoundError: No module named 'scib'` or `skmisc`** — env wasn't
  built from the latest `environment.yml`. Quick fix:
  `pip install scib scikit-misc`.
- **`Not Found` ollama.tgz (9 bytes)** — used a stale URL or missed `-L`. See §2.
- **`sudo: ... not in the sudoers file`** — you never needed sudo here.
  Everything installs to `~/.local` or the conda env.
- **`undefined symbol: _ZNK3c107SymBool10guard_bool...`** — PyTorch ABI
  mismatch. An old torch in `~/.local/lib/python3.10/site-packages` (user
  site) shadows the conda one. Fix: `export PYTHONNOUSERSITE=1` **before**
  launching anything that imports torch (the training subprocess inherits
  this from the parent shell).
- **Agent invents fake dataset names like "dataset_2022-01-01"** — that's
  the LLM hallucinating. Verify it actually called the tool by looking at
  `results/audit.log`. If it didn't, reset the conversation; if it persists,
  re-pull the model: `ollama rm qwen2.5:7b && ollama pull qwen2.5:7b`.
- **GPU not visible** — you're on a login node, not a GPU node.

---

## 8. End-to-end demo test (≈3 minutes on a GPU)

This is the exact command sequence to confirm the pipeline works after a
fresh install. It runs preprocess + a tiny 4-stage training
(5 epochs/stage = 20 epochs) on the MERFISH demo:

```bash
cd $REPO_ROOT/CrossHONA-Agent

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CROSSSPECIES_SCRIPTS_DIR=$REPO_ROOT/CrossHONA
export CROSSSPECIES_DATA_DIR=$PWD/data
export CROSSSPECIES_RESULTS_DIR=$PWD/results

# preprocess (sync, ~10s)
python -c "
from tools.training import preprocess
print(preprocess(
    ref_h5ad='merfish/human_H19_STG_4000.h5ad',
    target_h5ad='merfish/mouse_mouse1_242.h5ad',
    homo_table='merfish/Human_Mouse.tsv',
    species1_name='Human', species2_name='Mouse',
    project_name='merfish_demo', skip_QC=True,
    nonhomo_hvg=1000, identity_graph=True,
    homo_gene_id_ref='Gene name', homo_gene_id_target='Gene name',
))"

# train (async, 2–3 min)
python -c "
from tools.training import train
print(train(
    project_name='merfish_demo', run_name='run1',
    epochs_per_stage=5, lr=3e-4, batch_size=20,
    use_nonhomo=True, VAE=True, denoise=True,
    condition=True, identity_graph=True,
    beta_intra=1.0, beta_homo=1.0, beta_bridge=1.0,
))"

# poll until done
until grep -q "TRAINING COMPLETE" results/merfish_demo/run1/train.log 2>/dev/null; do
  sleep 10
  python -c "
from tools.training import check_status
import json
print(json.dumps(check_status('merfish_demo', 'run1'), indent=2))
"
done
```

A successful run produces `results/merfish_demo/run1/` with:
- `Stage{0,1,2,3}_*` subdirectories
- `ref_latent.npy`, `target_latent.npy`, etc.
- `all_stage_metrics.json`
- `train.log` ending in "TRAINING COMPLETE"

---

## 9. Using the chat tab — example session

Open the UI from §5 and click the **Chat** tab. Paste each message below in
order. Each is followed by what the agent should do and roughly what it should
reply. If you see "(stopped after too many tool rounds)" or the agent invents
filenames, see the troubleshooting bullets at the end.

### Setup expectations

- **Model**: the chat is driven by `qwen2.5:7b` (set in §5).
- **Latency**: the first message in a fresh session takes ~10–30 s while the
  model is loaded onto the GPU. Subsequent turns are 1–3 s.
- **History**: the chat keeps full conversation history. Click
  **Reset conversation** if the agent starts repeating itself or refuses to
  re-check state.

### 9.1 Inspect what data is available

> **You:** *List the datasets I have available.*

→ Calls `list_datasets`. Reply lists files under `data/`:

```
data_root: .../CrossHONA-Agent/data
h5ad: merfish/human_H19_STG_4000.h5ad, merfish/mouse_mouse1_242.h5ad
csv: (none)
tsv: merfish/Human_Mouse.tsv
```

> **You:** *Inspect merfish/human_H19_STG_4000.h5ad.*

→ Calls `inspect_dataset`. Reply gives cell count, gene count, obs columns,
celltype-column candidates, top cell types, and whether spatial info is present.
Roughly:

```
n_cells: 4835, n_genes: 3999
obs_columns: ['cell_type']
obsm_keys: ['spatial']
top_cell_types: lOGC=2117, EXC=960, lASC=421, INC=360, oENDO=279
```

### 9.2 Preprocess

> **You:** *Preprocess merfish/human_H19_STG_4000.h5ad and merfish/mouse_mouse1_242.h5ad with homo file merfish/Human_Mouse.tsv into project merfish_demo. Use Human and Mouse as species, skip_QC, identity_graph on, Gene name for both homo gene IDs.*

→ Calls `preprocess`. The tool defends against the agent passing
`null`/`""`/`0` for unspecified params, so a short reply like the following
is normal:

```
Preprocessing complete for project `merfish_demo`.
Files created: inverse_dict_ref.pkl, inverse_dict_target.pkl,
ref_data.pt, target_data.pt
```

If you instead see *"An error occurred during preprocessing..."*, check
`results/audit.log` to see what arguments were passed; if the response
mentions `error: preprocess finished but did not produce ...`, the agent
got file paths or column names wrong — re-state them more explicitly.

### 9.3 Launch training

> **You:** *Train merfish_demo as run1 for 5 epochs per stage, batch size 20, lr 3e-4, beta_bridge 1.0.*

→ Calls `train`. Reply within ~1 s:

```
Training started: run_id=run1, pid=NNNNNN, total_epochs=20.
Ask "status of merfish_demo/run1" anytime.
```

The training itself runs in a detached subprocess (4 stages × 5 epochs ≈ 2–3
min on a single GPU). The chat stays responsive while it runs.

### 9.4 Check status

> **You:** *Status of merfish_demo/run1?*

This message is **intercepted** before the LLM sees it — `cb_chat` recognises
the `status` keyword and a `project/run` reference, calls `check_status`
directly, and formats the result. So the reply is **deterministic**, not
generated:

```
**merfish_demo/run1** — ▶ `running`
- stage: `Stage2_CrossSpeciesHomo` · epoch 3/5
- loss: 4.1230 · recon: 2.5012 · cls: 0.1480
- pid: 4118387 · alive: True
```

When training finishes, the icon flips to `✓ done` and `alive: False`. Other
phrases that trigger the same intercept: "progress", "is it done", "how far",
"which stage", "list runs".

(For runs that haven't been mentioned in the message, the intercept also looks
back through prior messages to find the last `project/run` you typed.)

### 9.5 Visualize

> **You:** *Plot the UMAP for merfish_demo/run1.*

→ Calls `plot_umap`. The agent responds in text *and* the chat displays the
PNG inline (Gradio ChatMessage with a `{"path": ...}` content). The image is
also saved at `results/merfish_demo/run1/umap_ref_latent.png`.

> **You:** *Plot the loss curves for merfish_demo/run1.*

→ Calls `plot_loss_curves`. Renders one panel per stage (Stage0 → Stage3),
showing `loss`, `recon`, and `cls` over epochs. Saved at
`results/merfish_demo/run1/loss_curves.png`.

### 9.6 Other useful prompts

| Prompt | Tool called |
|---|---|
| *List my runs.* | `list_runs` |
| *Stop merfish_demo/run1.* | `stop_run` (SIGTERM → SIGKILL) |
| *What runs do I have under merfish_demo?* | `list_runs(project_name=...)` |

### 9.7 When the agent goes off the rails

The 7-billion-parameter LLM is the weakest link. Symptoms and fixes:

- **Agent reports a different stage/epoch than what's actually in the log.**
  It paraphrased an earlier `check_status` result instead of re-calling.
  Re-phrase your question with explicit keywords ("status", "progress") to
  trigger the deterministic intercept (§9.4), or click **Reset conversation**.
- **Agent invents column names like "CellType_Human" or paths without `merfish/`.**
  qwen2.5 sometimes "normalises" your input. Re-state values in **backticks**
  to make them more literal: *Use `Gene name` (with a space, exactly).*
- **`(stopped after too many tool rounds — try rephrasing)`.** Agent tried
  ≥6 tool calls without converging. Inspect `results/audit.log` to see what
  it kept trying; usually one parameter is mis-spelled. Restart the
  conversation and be more explicit.
- **Agent prints Python code instead of running a tool.** It hallucinated
  the call. Reset the conversation and ask again. If it persists across
  resets, the model file may be corrupted —
  `ollama rm qwen2.5:7b && ollama pull qwen2.5:7b`.

When in doubt, the **Form tab** does the same things deterministically with
sliders and text fields, no LLM involved.

---

## Appendix A: HPC / remote machines

If you're on an HPC cluster, you need (a) a GPU node and (b) a way to
access the Gradio UI from your laptop's browser. Adapt the snippets below to
your site's conventions.

### Get a GPU node (SLURM example)

```bash
srun -p gpu --gres=gpu:1 --mem=32G -t 4:00:00 --pty bash
hostname        # remember the node name; you'll need it for port forwarding
```

Non-SLURM sites: use whatever your scheduler provides (`qsub -I`, `bsub -Is`,
`oarsub -I`, etc.) and request one GPU.

### Get the UI into your browser

Three common patterns, in rough order of preference:

1. **SSH tunnel** (works on most clusters that allow ssh to compute nodes
   from the login node):
   ```bash
   # from your laptop
   ssh -L 7860:<gpu-node>:7860 <user>@<login-node>
   ```
   Then open `http://localhost:7860` locally. If the login node blocks
   forwards or can't resolve the GPU node hostname, try the FQDN, or fall
   back to option 2 or 3.

2. **Open OnDemand reverse proxy** (if your cluster runs OOD). The URL
   shape is typically
   `https://<ood-host>/rnode/<gpu-node-fqdn>/<session-port>/proxy/7860/`.
   Gradio behind a path-prefixed proxy needs `GRADIO_ROOT_PATH` to match:
   ```bash
   export GRADIO_ROOT_PATH="/rnode/<gpu-node-fqdn>/<session-port>/proxy/7860"
   nohup python -u app.py > ~/app.log 2>&1 &
   ```
   If the chat shows *"Could not parse server response..."*, that prefix has
   changed (OOD session ports often rotate per session) — restart `app.py`
   with the new value.

3. **JupyterHub / VS Code Server / cluster-specific proxy.** If your
   cluster ships an in-browser environment, launch `app.py` from inside it
   and use whatever port-forwarding feature the environment provides.

### HPC gotchas

- **Home quota.** Many clusters cap `~/` at a few GB. `OLLAMA_MODELS` must
  point to a larger data volume — see §2.
- **Shared data, no duplication.** If your input data lives on a shared
  read-only volume, symlink it into `CrossHONA-Agent/data/` instead of
  copying:
  ```bash
  mkdir -p $REPO_ROOT/CrossHONA-Agent/data/merfish
  cd $REPO_ROOT/CrossHONA-Agent/data/merfish
  ln -sfn /path/to/shared/Merfish_brain/human_H19_STG_4000.h5ad .
  ln -sfn /path/to/shared/Merfish_brain/mouse_mouse1_242.h5ad   .
  ln -sfn /path/to/shared/homolog_genes/Human_Mouse.tsv         .
  ```
  The agent treats symlinks the same as real files.
- **Same-shell env vars.** All `export` lines in §5 must run in the same
  shell that launches `python app.py`; the training subprocess inherits
  them.
