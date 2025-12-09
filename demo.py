"""Minimal llama.cpp tool-calling demonstration runnable from IPython."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List

from openai import OpenAI
from textwrap import dedent


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
# Tools definition for the model
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
    logger.info("Running top_processes with limit=%s", limit)
    cmd = ["ps", "axo", "pid,ppid,pcpu,pmem,comm", "--sort=-pcpu"]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = completed.stdout.strip().splitlines()
    return "\n".join(lines[: limit + 1])  # include header


def disk_usage(path: str = "/") -> str:
    logger.info("Running disk_usage for path=%s", path)
    cmd = ["df", "-h", path]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


# Dispatch table for dynamic tool execution
TOOL_IMPLS = {
    "top_processes": top_processes,
    "disk_usage": disk_usage,
}


# -------------------------------
# Llama.cpp call wrapper
# -------------------------------

def _call_llama(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Sending %s message(s) to llama.cpp", len(messages))
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=TOOLS,
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

    Only call these tools when the user explicitly asks about CPU, processes,
    disk usage, filesystem space, or similar diagnostics.

    After receiving tool outputs, integrate them into a final natural-language answer.
    Do not call tools again unless the user explicitly asks.
""")


# -------------------------------
# Demo Runner
# -------------------------------

def run_demo(
    user_message: str = "What is currently using the most CPU?",
    max_iterations: int = 5,
) -> Dict[str, Any]:

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

        # Ensure tool_calls is always a list
        tool_calls = choice.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            logger.warning("tool_calls had unexpected type %r — ignoring", type(tool_calls))
            tool_calls = []

        # If no tool calls: final answer
        if not tool_calls:
            logger.info("Model provided final answer without tools.")
            logger.info(choice.get("content"))
            return last_response

        # Execute each requested tool
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name")
            raw_args = func.get("arguments") or "{}"

            # Parse arguments
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning("Bad JSON tool arguments %r — using empty dict", raw_args)
                args = {}

            impl = TOOL_IMPLS.get(name)
            if not impl:
                logger.warning("Unknown tool requested: %s", name)
                continue

            # Run tool with flexible **args
            logger.info("Executing tool '%s' with args=%s", name, args)
            try:
                tool_output = impl(**args)
            except TypeError:
                # If model passes unexpected arguments, discard them
                tool_output = impl()

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": name,
                    "content": tool_output,
                }
            )

        logger.info("Completed tool calls; continuing conversation.")

    logger.warning(
        "Reached max_iterations=%s without final answer.", max_iterations
    )
    return last_response or {}
