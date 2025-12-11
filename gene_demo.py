"""Minimal llama.cpp tool-calling demonstration runnable from IPython."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import inspect
from typing import Dict, List, Any, Optional
import yaml
from Bio import Entrez, Medline

from openai import OpenAI
from textwrap import dedent

# Open and load the YAML file
with open('secret.yaml', 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)
    Entrez.api_key = data["key"]
    Entrez.email = data["email"]

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "Qwen2.5-VL-7B-Instruct-Q4_K_M")
LOG_LEVEL = os.environ.get("DEMO_LOG_LEVEL", "INFO").upper()
MIN_TOOL_CALLS = 4     # enforce at least N tool calls before final answer
PMID_RECALL_THRESHOLD = 3  # if a single tool call returns <2 PMIDs, force a refinement call

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
            "name": "article_info",
            "description": "Get summaries of articles from pubmed",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "requested search term",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gene_info",
            "description": "Get gene info from NCBI",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene": {
                        "type": "string",
                        "description": "requested gene name",
                    }
                },
                "required": [],
            },
        },
    }
]

# -------------------------------
# Actual Python implementations
# -------------------------------

def get_article_info(term: str) -> str:
    """Retrieve PubMed titles + abstracts for a search term."""
    with Entrez.esearch(db="pubmed", term=term, retmax=5) as stream:
        record = Entrez.read(stream)

    ids = record["IdList"]
    if not ids:
        return ""

    with Entrez.efetch(db="pubmed", id=",".join(ids), rettype="medline", retmode="text") as stream:
        medline_records = Medline.parse(stream)
        result_lines = []
        for r in medline_records:
            title = r.get("LOCUS") or ""
            abstract = r.get("AB") or ""
            pmid = r.get("PMID") or ""
            result_lines.append(f"PMID: {pmid}\nTITLE: {title}\nABSTRACT:\n{abstract}\n")

        result = "\n".join(result_lines)
        return result[:3500] #cutoff at 3500
    

def _esearch_gene_id(symbol: str) -> Optional[str]:
    """
    Find the NCBI Gene ID for an official symbol in Homo sapiens.
    Returns the first GeneID (as string) or None.
    """
    query = f'{symbol}[Gene Name] AND "Homo sapiens"[Organism]'
    with Entrez.esearch(db="gene", term=query, retmax=1, retmode="xml") as h:
        rec = Entrez.read(h)
    ids = rec.get("IdList", [])
    return ids[0] if ids else None


def _efetch_gene_xml(gene_id: str) -> List[Dict[str, Any]]:
    """
    Fetch NCBI Gene XML (Entrez.read result is a list of dicts).
    """
    with Entrez.efetch(db="gene", id=gene_id, retmode="xml") as h:
        data = Entrez.read(h)
    # data is typically a list with one element per gene
    return data


def _safe_get(d: Dict[str, Any], path: List[Any], default=None):
    """
    Safely descend nested dict/list structure by path (keys or indices).
    """
    cur = d
    for key in path:
        try:
            if isinstance(key, int) and isinstance(cur, list):
                cur = cur[key]
            else:
                cur = cur.get(key) if isinstance(cur, dict) else None
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _gene_basic_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract official symbol, full name, summary, aliases.
    """
    res = {"symbol": "", "name": "", "summary": "", "aliases": []}

    # Official Symbol and Name
    gene_ref = _safe_get(doc, ["Entrezgene_gene", "Gene-ref"], {})
    res["symbol"] = gene_ref.get("Gene-ref_locus", "") or gene_ref.get("Gene-ref_locus-tag", "") or ""
    res["name"] = gene_ref.get("Gene-ref_desc", "") or ""
    res["location"] = gene_ref.get("Gene-ref_maploc", "") or ""

    # Summary
    res["summary"] = doc.get("Entrezgene_summary", "") or ""

    # Aliases
    syn = gene_ref.get("Gene-ref_syn", []) or []
    res["aliases"] = [s for s in syn if isinstance(s, str)]

    return res

def get_gene_info(gene: str) -> Dict[str, Any]:
    """
    High-level wrapper:
    - Resolve gene -> GeneID (human)
    - Fetch Gene XML
    - Extract: basic fields, GO, pathways, expression
    - Link MedGen -> extract HPO cross-refs (if available)
    Returns a dict with these sections.
    """
    gene_id = _esearch_gene_id(gene)
    if not gene_id:
        return f"error: No Homo sapiens gene found for gene '{gene}'."

    gene_docs = _efetch_gene_xml(gene_id)
    if not gene_docs:
        return f"error: Could not fetch Gene record for GeneID {gene_id}."

    doc = gene_docs[0]
    basic = _gene_basic_fields(doc)
    gene_info_dict = {
        "gene_id": {gene_id},
        "symbol": basic.get("symbol"),
        "location": basic.get("location"),
        "name": basic.get("name"),
        "summary": basic.get("summary"),
        "aliases": basic.get("aliases", []),
    }

    gene_info_text = '\n'.join([f"{k}: {v}"for k,v in gene_info_dict.items()])
    return gene_info_text

# Mapping from tool name to implementation
TOOL_IMPLS: Dict[str, Any] = {
    "article_info": get_article_info,
    "gene_info": get_gene_info,
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
# --- Modify your llama.cpp call wrapper to accept optional tool_choice ---
def _call_llama(messages: List[Dict[str, Any]], tool_choice: Any = "auto") -> Dict[str, Any]:
    logger.info("Sending %s message(s) to llama.cpp", len(messages))
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice=tool_choice,    # now configurable
        temperature=0,
        # If your llama.cpp server supports it, enabling parallel tool calls helps:
        # parallel_tool_calls=True,
    )
    logger.info("Received response with %s choice(s)", len(response.choices))
    return response.model_dump()

# -------------------------------
# System prompt
# -------------------------------

# Strengthen the system prompt to make expectations explicit:
SYSTEM_PROMPT = dedent("""
    You are a helpful assistant that searches PubMed abstracts and Gene information and provide answers based on these.
    Refinement strategies include: using MeSH terms, official HGNC gene symbols, common aliases,
    adding species (e.g., human), disease names, and broader/related keywords.
    Clearly state the PMIDs of the abstracts used. 
    !In as few words as possible!

    Tools available:
    - article_info(term: string): Summaries of PubMed articles about the search term used.
    - gene_info(gene: string): Gene records from the NCBI nucleotide database.
                       
    
""")

# --- Helper to check recall in tool outputs ---
def count_pmids(tool_output: str) -> int:
    # Our get_article_info() prints "PMID: " per record; counting is a simple proxy for recall
    if not tool_output:
        return 0
    return tool_output.count("PMID:")


# --- Updated run_demo enforcing multiple tool calls when needed ---
def run_demo(
    user_message: str = "What can you tell me about the NEMO gene, and what would be the best name to use for this gene?",
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """Run a multi-turn tool calling conversation until the model finishes, enforcing multiple calls."""
    logger.info("Starting demo with user message: %s", user_message)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    total_tool_calls = 0
    last_response: Dict[str, Any] | None = None

    for iteration in range(1, max_iterations + 1):
        logger.info("Requesting model response (iteration %s)", iteration)
        # Default attempt: let the model decide (auto)
        last_response = _call_llama(messages, tool_choice="auto")
        choice_obj = last_response["choices"][0]
        choice = choice_obj["message"]
        finish_reason = choice_obj.get("finish_reason")
        logger.info("finish_reason=%s", finish_reason)
        messages.append(choice)

        tool_calls = choice.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            logger.warning("tool_calls had unexpected type %r — treating as empty", type(tool_calls))
            tool_calls = []

        if tool_calls:
            # Execute all tool calls the model requested
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

                total_tool_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": tool_output,
                    }
                )

                # Evaluate simple recall; if low, nudge refinement explicitly
                if name == "article_info":
                    pmids = count_pmids(tool_output)
                    if pmids < PMID_RECALL_THRESHOLD and total_tool_calls < MIN_TOOL_CALLS:
                        messages.append({
                            "role": "system",
                            "content": (
                                "Recall from the last query appears low. Refine the PubMed query using "
                                "MeSH terms, common aliases, and species context. Then call article_info again. "
                                "Do not finalize the answer yet. "
                                "Additionally, if information on other genes should be looked into, call gene_info for said gene"
                            ),
                        })
                        # Force the next assistant turn to issue a tool call to article_info
                        last_response = _call_llama(messages, tool_choice={"type": "function", "function": {"name": "article_info"}})
                        choice_obj = last_response["choices"][0]
                        choice = choice_obj["message"]
                        messages.append(choice)

                        forced_calls = choice.get("tool_calls") or []
                        for forced_call in forced_calls:
                            func_desc = forced_call.get("function", {})
                            name2 = func_desc.get("name")
                            raw_args2 = func_desc.get("arguments") or "{}"
                            try:
                                args2 = json.loads(raw_args2)
                            except json.JSONDecodeError:
                                logger.warning("Bad JSON tool arguments %r — using empty dict", raw_args2)
                                args2 = {}
                            if not name2:
                                logger.warning("Forced tool call without name: %r", forced_call)
                                continue
                            try:
                                tool_output2 = call_tool_with_filtered_kwargs(name2, args2)
                            except Exception as e:
                                logger.warning("Error while running tool %s: %s", name2, e)
                                tool_output2 = f"Error: {e}"

                            total_tool_calls += 1
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": forced_call["id"],
                                    "name": name2,
                                    "content": tool_output2,
                                }
                            )

            # After executing tools, continue the conversation
            logger.info("Completed tool calls; continuing conversation.")
            continue

        # If the assistant attempted to answer without tools:
        if total_tool_calls < MIN_TOOL_CALLS:
            logger.info("Assistant tried to finish before reaching MIN_TOOL_CALLS=%s; enforcing another call.", MIN_TOOL_CALLS)
            messages.append({
                "role": "system",
                "content": (
                    f"You must call article_info at least {MIN_TOOL_CALLS} times before finalizing. "
                    "Refine the query and call the tool again."
                ),
            })
            # Force a tool call next turn
            last_response = _call_llama(messages, tool_choice={"type": "function", "function": {"name": "article_info"}})
            choice_obj = last_response["choices"][0]
            choice = choice_obj["message"]
            messages.append(choice)
            enforced_calls = choice.get("tool_calls") or []
            for tool_call in enforced_calls:
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
                total_tool_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": tool_output,
                    }
                )
            logger.info("Enforced extra tool call; continuing.")
            continue

        # Otherwise the assistant provided a final answer and we have enough tool calls
        logger.info("Model provided final answer after %s tool call(s).", total_tool_calls)
        logger.info(choice.get("content"))
        return last_response

    logger.warning("Reached max_iterations=%s without final answer.", max_iterations)
    return last_response or {}
