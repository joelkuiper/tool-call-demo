# tool-call-demo

A minimal end-to-end demonstration of llama.cpp tool calling. It assumes a running llama.cpp server and uses the official `openai` client, e.g.:

```bash
llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M --jinja
```

## Run from IPython
There are two demonstrations avaiable. 

### System stats 
This demo uses `psutil`, `ps` and `df` to inspect system stats.

It defines two tools:
  * `top_processes` shells out to `ps` to list the busiest processes.
  * `disk_usage` uses `df -h` to summarize storage for a path.
Then: 
  * Sends a chat completion request with that tool registered.
  * Executes the model's tool call and returns the model's final message with the tool output attached.

 
```python
import demo
result = demo.run_demo()  # defaults to asking about CPU usage
print(result)

# try a different user message without editing the module
custom = demo.run_demo("Check disk usage for /tmp")
print(custom)
```

### BioPython 
This demo searches PubMed and NCBI to retrieve data about genes.
It implements the tools `article_info` (Get summaries of articles from PubMed) and `gene_info` (Get gene info from NCBI). Install the optional bioportal package `uv sync --extra biopython`. 

```python
import gene_demo 
result = gene_demo.run_demo("What can you tell me about the NEMO gene, and what would be the best name to use for this gene?")
print(result)
```

Set `LLAMA_SERVER_URL` and `LLAMA_MODEL` if your server is different from the defaults (`http://127.0.0.1:8080` and `Qwen2.5-VL-7B-Instruct-Q4_K_M`). You can also provide `LLAMA_API_KEY` if your server requires an API token; otherwise the default dummy value is used.

Verbose logging is enabled by default so you can see each step of the tool-calling flow. Override with `DEMO_LOG_LEVEL=DEBUG` to see full payloads or `DEMO_LOG_LEVEL=WARNING` to reduce noise.
