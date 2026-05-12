"""
Local LLM agent. Connects to an Ollama server (OpenAI-compatible API),
exposes the tool registry via function calling, and dispatches calls.

No data ever leaves the machine: OLLAMA_BASE_URL points to localhost.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
import typing
from pathlib import Path
from typing import Any, Callable, get_type_hints

from openai import OpenAI

from tools import RESULTS_ROOT, TOOLS

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

SYSTEM_PROMPT = """You are an assistant for a cross-species single-cell / spatial \
transcriptomics alignment tool. You help the user inspect data, run preprocessing, \
launch training, and visualize results.

RULES:
- Only use the provided tools. Never suggest shell commands, raw Python, or web URLs.
- Do not invent file names or capabilities. If the user is vague, call list_datasets \
or list_runs first.
- You CANNOT upload files. To add data, the user must place .h5ad / .csv / .tsv \
files into the data directory themselves. Tell them this when they ask.
- The available tools are exactly: list_datasets, inspect_dataset, preprocess, train, \
check_status, stop_run, list_runs, plot_umap, plot_loss_curves. Do not reference any \
other function names.

PARAMETER RULES (very important):
- When calling a tool, ONLY pass parameters the user explicitly mentioned. Omit all \
others — the tool has correct defaults that you must NOT override.
- Use file paths EXACTLY as the user gave them (case, slashes, prefix all matter). \
Do not "normalize", uppercase, or strip directory prefixes.
- Use string values EXACTLY as the user gave them (e.g. "Gene name" stays as \
"Gene name", not "GeneName" or "gene_name").
- Never invent column names like "CellType_Human" — if you do not know a column \
name, call inspect_dataset first.
- If a tool call returns an error, read the error and fix that one issue. Do NOT \
retry with different random parameters.

EXECUTION RULES:
- Training is long-running. After calling `train`, report the run_id and tell the \
user they can ask for status later. Do not poll on your own.
- For ANY question about current state ("status", "is it done", "progress", \
"what stage", "list runs"), ALWAYS call the relevant tool fresh. NEVER answer \
from memory or earlier tool results — state changes between turns.
- When reporting file paths from tool results, copy them VERBATIM. The path \
"merfish/human_H19_STG_4000.h5ad" must NOT be shortened to \
"human_H19_STG_4000.h5ad" — the user needs the directory prefix to call other \
tools correctly.
- Be terse. Report numbers, not paragraphs."""


# ---------------------------------------------------------------------------
# Audit log — every tool invocation is appended here
# ---------------------------------------------------------------------------
_LOG_PATH = RESULTS_ROOT / "audit.log"
logging.basicConfig(
    filename=_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
_audit = logging.getLogger("audit")


# ---------------------------------------------------------------------------
# Type → JSON-schema mapping for tool signatures
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object",
}


def _py_type_to_schema(tp) -> dict:
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        # treat Optional[X] as X
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _py_type_to_schema(args[0]) if args else {"type": "string"}
    if tp in _TYPE_MAP:
        return {"type": _TYPE_MAP[tp]}
    return {"type": "string"}


def _fn_to_schema(name: str, fn: Callable) -> dict:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        schema = _py_type_to_schema(hints.get(pname, str))
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(pname)
        props[pname] = schema
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (fn.__doc__ or "").strip().split("\n\n")[0],
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS = [_fn_to_schema(n, f) for n, f in TOOLS.items()]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_msg: str, max_tool_rounds: int = 6) -> str:
        """Send a user message, run any tool calls, return final assistant text."""
        self.history.append({"role": "user", "content": user_msg})
        _audit.info("USER %s", json.dumps(user_msg)[:500])

        for _ in range(max_tool_rounds):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            self.history.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or ""

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._dispatch(name, args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps(result, default=str)[:8000],
                })

        return "(stopped after too many tool rounds — try rephrasing)"

    def _dispatch(self, name: str, args: dict) -> Any:
        fn = TOOLS.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        _audit.info("TOOL %s args=%s", name, json.dumps(args, default=str)[:500])
        t0 = time.time()
        try:
            out = fn(**args)
        except TypeError as e:
            out = {"error": f"bad arguments: {e}"}
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        _audit.info("TOOL %s done in %.2fs", name, time.time() - t0)
        return out
