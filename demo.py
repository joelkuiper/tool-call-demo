"""Minimal llama.cpp tool-calling demonstration runnable from IPython."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List

from openai import OpenAI


LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "Qwen2.5-VL-7B-Instruct-Q4_K_M")
LOG_LEVEL = os.environ.get("DEMO_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url=f"{LLAMA_SERVER_URL}/v1",
    api_key=os.environ.get("LLAMA_API_KEY", "not-needed"),
)


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


def top_processes(limit: int = 5) -> str:
    """Return the top processes using the ps command."""
    logger.info("Running top_processes with limit=%s", limit)
    cmd = ["ps", "axo", "pid,ppid,pcpu,pmem,comm", "--sort=-pcpu"]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    header_and_rows = completed.stdout.strip().splitlines()
    limited_rows = header_and_rows[: limit + 1]  # header + rows
    logger.info("top_processes completed; returning %s rows", len(limited_rows) - 1)
    return "\n".join(limited_rows)


def disk_usage(path: str = "/") -> str:
    """Return human-readable disk usage for the given path."""
    logger.info("Running disk_usage for path=%s", path)
    cmd = ["df", "-h", path]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    logger.info("disk_usage completed")
    return completed.stdout.strip()


def _call_llama(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Sending %s message(s) to llama.cpp", len(messages))
    logger.debug("Messages payload: %s", json.dumps(messages, indent=2))
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )
    logger.info("Received response with %s choice(s)", len(response.choices))
    return response.model_dump()


def run_demo(user_message: str = "What is currently using the most CPU?") -> Dict[str, Any]:
    """Run a two-turn tool calling conversation and return the final response.

    This is intentionally small so it can be copied into an IPython session:

    >>> import demo
    >>> final = demo.run_demo()
    >>> final = demo.run_demo("Check disk usage for /tmp")
    >>> print(json.dumps(final, indent=2))
    """

    logger.info("Starting demo with user message: %s", user_message)

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Use the available tools to inspect the system before answering."
                " Call top_processes to inspect CPU and disk_usage to inspect storage."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    first_response = _call_llama(messages)
    choice = first_response["choices"][0]["message"]
    tool_calls = choice.get("tool_calls", [])

    if tool_calls:
        messages.append(choice)

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name")
        args = function.get("arguments") or "{}"
        parsed_args = json.loads(args)

        if name == "top_processes":
            limit = int(parsed_args.get("limit", 5))
            logger.info("Model requested top_processes with limit=%s", limit)
            tool_output = top_processes(limit=limit)
        elif name == "disk_usage":
            path = parsed_args.get("path", "/")
            logger.info("Model requested disk_usage for path=%s", path)
            tool_output = disk_usage(path=path)
        else:
            logger.warning("Unknown tool requested: %s", name)
            continue

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": tool_output,
            }
        )

    if not tool_calls:
        logger.info("Model responded without requesting any tools.")

    logger.info("Requesting final answer after tool execution")
    final_response = _call_llama(messages)
    logger.info(final_response["choices"][0]["message"]["content"])
    return final_response
