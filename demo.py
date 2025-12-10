"""Minimal llama.cpp tool-calling demonstration runnable from IPython."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import inspect
from typing import Any, Dict, List

from openai import OpenAI
from textwrap import dedent
import psutil


LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "Qwen2.5-VL-7B-Instruct-Q4_K_M")
LOG_LEVEL = os.environ.get("DEMO_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url=f"{LLAMA_SERVER_URL}/v1",
    api_key=os.environ.get("LLAMA_API_KEY", "not-needed"),
)

# -------------------------------
# Tool schema for the model
# -------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "top_processes",
            "description": "Show the most CPU hungry processes using the ps command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many processes to return (default 5).",
                        "minimum": 1,
                        "maximum": 20,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_usage",
            "description": "Report memory usage using psutil.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disk_usage",
            "description": "Report disk usage for the given path using df -h.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filesystem path to check (default /).",
                    }
                },
                "required": [],
            },
        },
    },
]

# -------------------------------
# Actual Python implementations
# -------------------------------
def top_processes(limit: int = 5) -> str:
    """Return the top processes using the ps command, excluding the ps process itself."""
    logger.info("Running top_processes with limit=%s", limit)
    try:
        cmd = ["ps", "axo", "pid,ppid,pcpu,pmem,comm", "--sort=-pcpu"]
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        lines = completed.stdout.strip().splitlines()
        if not lines:
            return "No processes found."
        header = lines[0]
        rows = lines[1:]
        filtered_rows = []
        for line in rows:
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            comm = parts[4]
            if comm == "ps":
                continue
            filtered_rows.append(line)
            if len(filtered_rows) >= limit:
                break
        return "\n".join([header] + filtered_rows)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to run top_processes: %s", e)
        return f"Error: {e}"
    except Exception as e:
        logger.error("Unexpected error in top_processes: %s", e)
        return f"Error: {e}"

def disk_usage(path: str = "/") -> str:
    """Return human-readable disk usage for the given path."""
    logger.info("Running disk_usage for path=%s", path)
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    try:
        cmd = ["df", "-h", path]
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return completed.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error("Failed to run disk_usage: %s", e)
        return f"Error: {e}"
    except Exception as e:
        logger.error("Unexpected error in disk_usage: %s", e)
        return f"Error: {e}"

def memory_usage() -> str:
    """Return human-readable memory usage using psutil."""
    logger.info("Running memory_usage")
    try:
        # Get memory info
        memory_info = psutil.virtual_memory()
        total = memory_info.total / (1024 ** 3)  # Convert bytes to GB
        available = memory_info.available / (1024 ** 3)  # Convert bytes to GB
        used = memory_info.used / (1024 ** 3)  # Convert bytes to GB
        percent = memory_info.percent
        return (
            f"Total memory: {total:.2f} GB\n"
            f"Used memory: {used:.2f} GB\n"
            f"Available memory: {available:.2f} GB\n"
            f"Memory usage: {percent}%"
        )
    except Exception as e:
        logger.error("Unexpected error in memory_usage: %s", e)
        return f"Error: {e}"

# Mapping from tool name to implementation
TOOL_IMPLS: Dict[str, Any] = {
    "top_processes": top_processes,
    "disk_usage": disk_usage,
    "memory_usage": memory_usage
}

# -------------------------------
# Generic **kwargs adapter
# -------------------------------
def call_tool_with_filtered_kwargs(name: str, args: Dict[str, Any]) -> str:
    """Call a tool implementation, keeping only args that match its signature."""
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        raise ValueError(f"Unknown tool: {name}")
    if not isinstance(args, dict):
        logger.warning("Tool args for %s were not a dict: %r; treating as empty.", name, args)
        args = {}
    sig = inspect.signature(impl)
    filtered: Dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param_name in args:
            filtered[param_name] = args[param_name]
    logger.info("Calling tool %s with filtered args=%s (raw=%s)", name, filtered, args)
    return impl(**filtered)

# -------------------------------
# Llama.cpp call wrapper
# -------------------------------
def _call_llama(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Sending %s message(s) to llama.cpp", len(messages))
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    logger.info("Received response with %s choice(s)", len(response.choices))
    return response.model_dump()

# -------------------------------
# System prompt
# -------------------------------
SYSTEM_PROMPT = dedent("""
    You are a helpful assistant that can optionally inspect the local system using tools.
    You have access to:
    - top_processes: inspect CPU-hungry processes.
    - disk_usage: inspect disk usage.
    - memory_usage: inspect the memory usage.
    Only call these tools when the user explicitly asks about CPU, processes,
    disk usage, filesystem space, or similar diagnostics.
    After receiving tool outputs, integrate them into a final natural-language answer.
    Do not call tools again unless the user explicitly asks.
""")

# -------------------------------
# Demo runner
# -------------------------------
def run_demo(
    user_message: str = "What is currently using the most CPU?",
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """Run a multi-turn tool calling conversation until the model finishes."""
    logger.info("Starting demo with user message: %s", user_message)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    last_response: Dict[str, Any] | None = None
    for iteration in range(1, max_iterations + 1):
        logger.info("Requesting model response (iteration %s)", iteration)
        last_response = _call_llama(messages)
        choice_obj = last_response["choices"][0]
        choice = choice_obj["message"]
        finish_reason = choice_obj.get("finish_reason")
        logger.info("finish_reason=%s", finish_reason)
        messages.append(choice)
        tool_calls = choice.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            logger.warning("tool_calls had unexpected type %r — ignoring", type(tool_calls))
            tool_calls = []
        if not tool_calls:
            logger.info("Model provided final answer without tools.")
            logger.info(choice.get("content"))
            return last_response
        for tool_call in tool_calls:
            func_desc = tool_call.get("function", {})
            name = func_desc.get("name")
            raw_args = func_desc.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning("Bad JSON tool arguments %r — using empty dict", raw_args)
                args = {}
            if not name:
                logger.warning("Tool call without name: %r", tool_call)
                continue
            try:
                tool_output = call_tool_with_filtered_kwargs(name, args)
            except Exception as e:
                logger.warning("Error while running tool %s: %s", name, e)
                tool_output = f"Error: {e}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": name,
                    "content": tool_output,
                }
            )
        logger.info("Completed tool calls; continuing conversation.")
    logger.warning("Reached max_iterations=%s without final answer.", max_iterations)
    return last_response or {}
