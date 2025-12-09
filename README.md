# tool-call-demo

A minimal end-to-end demonstration of llama.cpp tool calling. It assumes a running llama.cpp server and uses the official `openai` client, e.g.:

```bash
llama-server --model Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --jinja -fa on -ngl 999
```

## Run from IPython

```python
import demo
result = demo.run_demo()  # defaults to asking about CPU usage
print(result)

# try a different user message without editing the module
custom = demo.run_demo("Check disk usage for /tmp")
print(custom)
```

Set `LLAMA_SERVER_URL` and `LLAMA_MODEL` if your server is different from the defaults (`http://127.0.0.1:8080` and `Qwen2.5-VL-7B-Instruct-Q4_K_M`). You can also provide `LLAMA_API_KEY` if your server requires an API token; otherwise the default dummy value is used.

Verbose logging is enabled by default so you can see each step of the tool-calling flow. Override with `DEMO_LOG_LEVEL=DEBUG` to see full payloads or `DEMO_LOG_LEVEL=WARNING` to reduce noise.

## What it does

* Defines two tools:
  * `top_processes` shells out to `ps` to list the busiest processes.
  * `disk_usage` uses `df -h` to summarize storage for a path.
* Sends a chat completion request with that tool registered.
* Executes the model's tool call and returns the model's final message with the tool output attached.

## CLI

You can also run the demo directly:

```bash
python -m main
```
