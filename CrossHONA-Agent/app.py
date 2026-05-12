"""
Gradio web app — two tabs:
  * Form: manual preprocess + train, parameter sliders, status panel.
  * Chat: natural-language agent backed by a local Ollama model.

Bind to 127.0.0.1 in production (handled by docker-compose port mapping).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import gradio as gr

import re

from agent import Agent
from tools import DATA_ROOT, RESULTS_ROOT
from tools.data import inspect_dataset, list_datasets
from tools.training import check_status, list_runs, preprocess, stop_run, train
from tools.viz import plot_loss_curves, plot_umap


# ---------------------------------------------------------------------------
# Form-tab callbacks
# ---------------------------------------------------------------------------
def _stage_upload(file_obj, kind: str) -> str:
    """Copy an uploaded file into DATA_ROOT/uploads/ and return its rel path."""
    if file_obj is None:
        return ""
    upload_dir = DATA_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    src = Path(file_obj.name)
    dst = upload_dir / src.name
    if str(src) != str(dst):
        shutil.copy2(src, dst)
    return f"uploads/{src.name}"


def cb_preprocess(ref_file, tgt_file, homo_file, sp1, sp2, project,
                  hvg, k_ref, k_tgt, ct_ref, ct_tgt, gid_ref, gid_tgt, skip_qc):
    if not all([ref_file, tgt_file, homo_file, sp1, sp2, project]):
        return "Missing required field.", ""
    ref = _stage_upload(ref_file, "ref")
    tgt = _stage_upload(tgt_file, "target")
    homo = _stage_upload(homo_file, "homo")
    out = preprocess(
        ref_h5ad=ref, target_h5ad=tgt, homo_csv=homo,
        species1_name=sp1, species2_name=sp2, project_name=project,
        hvg=int(hvg), k_ref=int(k_ref), k_target=int(k_tgt),
        celltype_name_ref=ct_ref, celltype_name_target=ct_tgt,
        homo_gene_id_ref=gid_ref, homo_gene_id_target=gid_tgt,
        skip_QC=skip_qc,
    )
    return json.dumps(out, indent=2), project


def cb_train(project, run_name, epochs, lr, latent, hidden, batch_size,
             use_nonhomo, seed):
    if not project or not run_name:
        return "Need project and run name."
    out = train(
        project_name=project, run_name=run_name,
        epochs_per_stage=int(epochs), lr=float(lr),
        latent_dim=int(latent), hidden_dim=int(hidden),
        batch_size=int(batch_size), use_nonhomo=use_nonhomo, seed=int(seed),
    )
    return json.dumps(out, indent=2)


def cb_status(project, run_name):
    if not project or not run_name:
        return "Need project and run name."
    return json.dumps(check_status(project, run_name), indent=2)


def cb_stop(project, run_name):
    return json.dumps(stop_run(project, run_name), indent=2)


def cb_umap(project, run_name, embedding):
    out = plot_umap(project, run_name, embedding)
    if "error" in out:
        return None, json.dumps(out, indent=2)
    return out["image_path"], json.dumps(out, indent=2)


def cb_losses(project, run_name):
    out = plot_loss_curves(project, run_name)
    if "error" in out:
        return None, json.dumps(out, indent=2)
    return out["image_path"], json.dumps(out, indent=2)


def cb_refresh_runs():
    return json.dumps(list_runs(), indent=2)


# ---------------------------------------------------------------------------
# Chat-tab state
# ---------------------------------------------------------------------------
_STATUS_TRIGGERS = re.compile(
    r"\b(status|progress|done|finished|complete|alive|how\s+far|"
    r"what\s+epoch|which\s+stage|how\s+is)\b",
    re.IGNORECASE,
)
_RUN_REF_RE = re.compile(r"([A-Za-z0-9_.\-]+)\s*/\s*([A-Za-z0-9_.\-]+)")


def _maybe_status_intercept(message: str, history: list) -> str | None:
    """Detect status questions and answer from check_status directly."""
    if not _STATUS_TRIGGERS.search(message):
        return None

    # Try project/run from current message, fall back to last seen pair.
    for source in [message] + [m.get("content", "") for m in reversed(history)
                                if m.get("role") == "user"]:
        m = _RUN_REF_RE.search(source or "")
        if m:
            project, run = m.group(1), m.group(2)
            try:
                result = check_status(project, run)
            except Exception as e:
                return f"`check_status` failed: {e}"
            return _format_status(project, run, result)
    return None


def _format_status(project: str, run: str, r: dict) -> str:
    if "error" in r:
        return f"**{project}/{run}**: {r['error']}"
    status = r.get("status", "unknown")
    icon = {"done": "✓", "running": "▶", "stopped": "■",
            "crashed": "✗", "not_started": "·"}.get(status, "?")
    lines = [f"**{project}/{run}** — {icon} `{status}`"]
    p = r.get("progress")
    if p:
        lines.append(
            f"- stage: `{p.get('stage')}` · epoch {p.get('epoch')}/{p.get('epochs_total')}"
        )
        lines.append(
            f"- loss: {p.get('loss'):.4f} · recon: {p.get('recon'):.4f} "
            f"· cls: {p.get('cls'):.4f}"
        )
    if r.get("pid"):
        lines.append(f"- pid: {r['pid']} · alive: {r.get('alive')}")
    return "\n".join(lines)


def _extract_image_paths(agent: Agent, before_len: int) -> list[str]:
    """Scan tool messages added during this turn for image_path fields."""
    paths = []
    for msg in agent.history[before_len:]:
        if msg.get("role") != "tool":
            continue
        try:
            content = json.loads(msg.get("content", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(content, dict) and content.get("image_path"):
            paths.append(content["image_path"])
    return paths


def cb_chat(message, history, agent_state):
    if agent_state is None:
        agent_state = Agent()

    # --- Plan B: deterministic short-circuit for status questions ---
    intercepted = _maybe_status_intercept(message, history)
    if intercepted is not None:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": intercepted},
        ]
        return history, agent_state, ""

    # --- Normal LLM path ---
    before_len = len(agent_state.history)
    reply = agent_state.chat(message)
    image_paths = _extract_image_paths(agent_state, before_len)

    new_msgs = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    # Attach any plot images the agent generated this turn as separate messages.
    # Gradio 6.x renders `{"path": ...}` as an inline file/image in the chatbot.
    for path in image_paths:
        if Path(path).exists():
            new_msgs.append({
                "role": "assistant",
                "content": {"path": str(Path(path).resolve())},
            })
    return history + new_msgs, agent_state, ""


def cb_reset_chat(agent_state):
    if agent_state is not None:
        agent_state.reset()
    return [], agent_state


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(title="Cross-Species Aligner") as app:
        agent_state = gr.State(value=None)

        gr.Markdown(
            "# Cross-Species Hierarchical Alignment\n"
            "All compute and the chat LLM run **locally**. No data leaves this machine."
        )

        with gr.Tabs():
            # ------------------- Form tab -------------------
            with gr.Tab("Form"):
                with gr.Accordion("1. Preprocess", open=True):
                    with gr.Row():
                        f_ref = gr.File(label="Reference .h5ad", file_types=[".h5ad"])
                        f_tgt = gr.File(label="Target .h5ad", file_types=[".h5ad"])
                        f_homo = gr.File(label="Homologous gene CSV", file_types=[".csv"])
                    with gr.Row():
                        sp1 = gr.Textbox(label="Species 1 name", value="mouse")
                        sp2 = gr.Textbox(label="Species 2 name", value="human")
                        proj = gr.Textbox(label="Project name", value="my_project")
                    with gr.Row():
                        hvg = gr.Slider(1000, 10000, value=5000, step=500, label="HVG")
                        k_ref = gr.Slider(3, 30, value=11, step=1, label="k_ref")
                        k_tgt = gr.Slider(3, 30, value=11, step=1, label="k_target")
                    with gr.Row():
                        ct_ref = gr.Textbox(label="celltype col (ref)", value="cell_type")
                        ct_tgt = gr.Textbox(label="celltype col (tgt)", value="cell_type")
                        gid_ref = gr.Textbox(label="homo gene id (ref)", value="Gene name")
                        gid_tgt = gr.Textbox(label="homo gene id (tgt)", value="gene name")
                    skip_qc = gr.Checkbox(label="Skip QC", value=False)
                    btn_pre = gr.Button("Run preprocess", variant="primary")
                    out_pre = gr.Code(label="Preprocess result", language="json")

                with gr.Accordion("2. Train", open=True):
                    with gr.Row():
                        proj_t = gr.Textbox(label="Project name")
                        run_name = gr.Textbox(label="Run name", value="run1")
                    with gr.Row():
                        epochs = gr.Slider(5, 200, value=30, step=5, label="Epochs / stage")
                        lr = gr.Number(value=1e-4, label="Learning rate")
                        latent = gr.Slider(32, 256, value=128, step=32, label="latent_dim")
                    with gr.Row():
                        hidden = gr.Slider(128, 1024, value=512, step=128, label="hidden_dim")
                        bs = gr.Slider(2, 50, value=10, step=1, label="batch_size")
                        seed = gr.Number(value=42, label="seed")
                    use_nh = gr.Checkbox(label="Use non-homologous branch", value=True)
                    btn_train = gr.Button("Start training", variant="primary")
                    out_train = gr.Code(label="Train result", language="json")

                with gr.Accordion("3. Monitor", open=True):
                    with gr.Row():
                        proj_s = gr.Textbox(label="Project")
                        run_s = gr.Textbox(label="Run")
                    with gr.Row():
                        btn_status = gr.Button("Check status")
                        btn_stop = gr.Button("Stop run", variant="stop")
                    out_status = gr.Code(language="json")
                    btn_runs = gr.Button("List all runs")
                    out_runs = gr.Code(language="json")

                with gr.Accordion("4. Visualize", open=False):
                    with gr.Row():
                        proj_v = gr.Textbox(label="Project")
                        run_v = gr.Textbox(label="Run")
                        emb = gr.Dropdown(
                            ["ref_latent", "target_latent",
                             "ref_latent_homo", "tgt_latent_homo"],
                            value="ref_latent", label="Embedding",
                        )
                    with gr.Row():
                        btn_umap = gr.Button("UMAP")
                        btn_loss = gr.Button("Loss curves")
                    img = gr.Image(label="Plot")
                    out_viz = gr.Code(language="json")

            # ------------------- Chat tab -------------------
            with gr.Tab("Chat"):
                gr.Markdown(
                    "Ask the local agent in plain English. Try:\n"
                    "- *List the datasets I have.*\n"
                    "- *Inspect uploads/mouse.h5ad.*\n"
                    "- *Preprocess uploads/mouse.h5ad and uploads/human.h5ad with "
                    "homo file uploads/homo.csv into project demo.*\n"
                    "- *Train demo for 30 epochs as run1.*\n"
                    "- *Status of demo/run1.*"
                )
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(label="Message", placeholder="Type and press Enter...")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    reset = gr.Button("Reset conversation")

        # ---------------- wire callbacks ----------------
        btn_pre.click(cb_preprocess,
                      [f_ref, f_tgt, f_homo, sp1, sp2, proj,
                       hvg, k_ref, k_tgt, ct_ref, ct_tgt, gid_ref, gid_tgt, skip_qc],
                      [out_pre, proj_t])
        btn_train.click(cb_train,
                        [proj_t, run_name, epochs, lr, latent, hidden, bs, use_nh, seed],
                        [out_train])
        btn_status.click(cb_status, [proj_s, run_s], [out_status])
        btn_stop.click(cb_stop, [proj_s, run_s], [out_status])
        btn_runs.click(cb_refresh_runs, [], [out_runs])
        btn_umap.click(cb_umap, [proj_v, run_v, emb], [img, out_viz])
        btn_loss.click(cb_losses, [proj_v, run_v], [img, out_viz])

        msg.submit(cb_chat, [msg, chatbot, agent_state], [chatbot, agent_state, msg])
        send.click(cb_chat, [msg, chatbot, agent_state], [chatbot, agent_state, msg])
        reset.click(cb_reset_chat, [agent_state], [chatbot, agent_state])

    return app


if __name__ == "__main__":
    # Telemetry off
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    app = build_app()
    launch_kwargs = dict(
        server_name="0.0.0.0",   # in container; host-side port is bound to 127.0.0.1
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,             # never create a public tunnel
        theme=gr.themes.Soft(),
        # Gradio sandboxes file serving by default. The chat tab embeds
        # PNGs from RESULTS_ROOT, so we have to whitelist it.
        allowed_paths=[str(RESULTS_ROOT)],
    )
    # When fronted by a path-prefixed reverse proxy (e.g. Open OnDemand at
    # /rnode/HOST/PORT/proxy/7860, JupyterHub, nginx subpath, …), set
    # GRADIO_ROOT_PATH so the client knows the URL prefix for /queue/join etc.
    root_path = os.environ.get("GRADIO_ROOT_PATH")
    if root_path:
        launch_kwargs["root_path"] = root_path
    app.queue(default_concurrency_limit=2).launch(**launch_kwargs)
