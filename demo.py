"""Minimal llama.cpp tool-calling demonstration runnable from IPython."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List
from textwrap import dedent

from openai import OpenAI


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

SYSTEM_PROMPT = dedent("""
    You are a helpful assistant that can optionally inspect the local system using tools.

    You have access to:
    - top_processes: to inspect CPU-hungry processes.
    - disk_usage: to inspect disk usage for a given path.

    Only call these tools when the user explicitly asks about CPU usage, running processes,
    disk operations, disk space, filesystem usage, or similar system-level diagnostics.

    If you have already called a tool for the current question, do not call the same tool again
    unless the user provides a new argument or explicitly asks for a repeat.

    After receiving tool output, integrate it into a final natural-language answer.
""")


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


def run_demo(
    user_message: str = "What is currently using the most CPU?",
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """Run a multi-turn tool calling conversation until the model finishes.

    The loop will continue issuing tool calls and feeding results back into the
    model until an assistant message arrives with no tool_calls (or the
    ``max_iterations`` safeguard is hit).

    This is intentionally small so it can be copied into an IPython session:

    >>> import demo
    >>> final = demo.run_demo()
    >>> final = demo.run_demo("Check disk usage for /tmp")
    >>> print(json.dumps(final, indent=2))
    """

    logger.info("Starting demo with user message: %s", user_message)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_response: Dict[str, Any] | None = None

    for iteration in range(1, max_iterations + 1):
        logger.info("Requesting model response (iteration %s)", iteration)
        last_response = _call_llama(messages)
        choice = last_response["choices"][0]["message"]

        # Normalize tool_calls: if missing or None, treat as []
        raw_tool_calls = choice.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            logger.warning(
                "Unexpected tool_calls type %r; treating as no tool calls",
                type(raw_tool_calls),
            )
            raw_tool_calls = []

        tool_calls = raw_tool_calls
        messages.append(choice)

        # If there are no tool calls, this is our final answer
        if not tool_calls:
            logger.info(
                "Model provided final answer without tool calls on iteration %s",
                iteration,
            )
            logger.info(choice.get("content"))
            return last_response

        # Execute requested tools
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

        logger.info(
            "Completed tool calls for iteration %s; continuing conversation", iteration
        )

        # Simple guardrail: na de eerste tool-ronde expliciet zeggen dat het nu klaar is met tools
        if iteration == 1:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You have now received the requested tool outputs. "
                        "Use them to answer the user's question directly. "
                        "Do not call any further tools unless the user explicitly asks "
                        "for another system inspection."
                    ),
                }
            )

    logger.warning(
        "Reached max_iterations=%s without a final model message lacking tool calls.",
        max_iterations,
    )
    return last_response or {}
