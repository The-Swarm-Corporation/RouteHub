
![Banner Logo](banner.png)

# RouteHub

<p align="left">
  <a href="https://swarms.ai">Swarms Website</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://docs.swarms.world/routehub/overview">Documentation</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://swarms.world">Swarms Marketplace</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="docs/routehub.pdf">Paper</a>
</p>

<p align="left">
  <a href="https://swarms.ai"><img src="https://img.shields.io/badge/Built%20by-Swarms.ai-3670A0?style=for-the-badge" alt="Built by Swarms.ai"></a>
  <a href="https://pypi.org/project/routehub/"><img src="https://img.shields.io/pypi/v/routehub?style=for-the-badge&color=3670A0" alt="PyPI version"></a>
  <a href="https://github.com/The-Swarm-Corporation/RouteHub"><img src="https://img.shields.io/github/stars/The-Swarm-Corporation/RouteHub?style=for-the-badge&color=3670A0" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-3670A0?style=for-the-badge" alt="License: Apache 2.0"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3670A0?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/openai/openai-python"><img src="https://img.shields.io/badge/Built%20on-OpenAI%20SDK-3670A0?style=for-the-badge&logo=openai&logoColor=white" alt="Built on the OpenAI SDK"></a>
  <a href="https://twitter.com/swarms_corp/"><img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter"></a>
  <a href="https://discord.gg/EamjgSaEQf"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
</p>

**RouteHub is the Swarms LLM gateway, built for raw speed.** litellm is slow: importing it takes over a second, loads 2,332 modules and calls GitHub before your code runs. RouteHub imports in 5.5 ms (about 190x faster), adds 0.004 ms to each call against litellm's 0.74 ms, and reaches 20+ providers through the official OpenAI SDK with the same function names. See [Performance](#performance).

## Quickstart

Install from PyPI:

```bash
pip install routehub

# Or with UV
uv add routehub

# with orjson for faster JSON handling
pip install "routehub[fast]"
```

Set a provider key and make a call:

```bash
export OPENAI_API_KEY="sk-..."
```

```python
import routehub

response = routehub.completion(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Summarize our Q3 risks in three bullets."}],
)
print(response.choices[0].message.content)
print(response.usage.total_tokens)
```

Change the model string to change providers. The call and the response type stay the same:

```python
routehub.completion(model="claude-sonnet-4-6", messages=messages)
routehub.completion(model="gemini/gemini-3-flash-preview", messages=messages)
routehub.completion(model="groq/llama-3.3-70b-versatile", messages=messages)
routehub.completion(model="openrouter/anthropic/claude-sonnet-4.6", messages=messages)
```

Stream, run async, or call tools with the same function:

```python
import asyncio

for chunk in routehub.completion(model="claude-sonnet-4-6", messages=messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="")

response = asyncio.run(
    routehub.acompletion(model="gpt-5.4-mini", messages=messages, tools=[tool])
)
```

[example.py](example.py) walks through every feature. Python 3.10+.

## Features

| Feature | What it means |
|---|---|
| **One API, every major provider** | OpenAI, Anthropic, Gemini, Groq, xAI, DeepSeek, OpenRouter, Together, Mistral, Fireworks, Azure OpenAI, Ollama, vLLM and any OpenAI-compatible server. Requests use the OpenAI chat format, and every response is the OpenAI SDK's own `ChatCompletion`, `ChatCompletionChunk` or `CreateEmbeddingResponse`. |
| **Built on the official OpenAI SDK** | Retries, timeouts, connection pooling and response types come from OpenAI's client rather than a reimplementation. Works with openai 2.x and 3.x. |
| **First-class Claude support** | Anthropic's OpenAI-compatible endpoint drops prompt caching, thinking output and cached-token usage, so RouteHub calls Claude's native Messages API and translates both ways. `cache_control` markers, extended thinking, tool use, images, PDFs and cached-token accounting all work. |
| **No global state** | Retries, TLS verification, timeouts and parameter dropping are arguments to each call. One tenant's `ssl_verify=False` or retry policy never leaks into another's, which matters when many teams or agents share a process. |
| **Predictable errors** | Every provider failure is raised as a typed exception (`RateLimitError`, `ContextWindowExceededError`, `AuthenticationError` and so on) carrying the status code, provider and model. Each subclasses the matching OpenAI SDK exception, so existing handlers keep working. |
| **Usage and cost visibility** | Token usage comes back in one shape for every provider, including cached input tokens and, where the provider reports them, reasoning tokens. `get_model_info` returns per-token prices, so spend can be computed per call. |
| **Live model catalog** | `get_model_info` reads context windows, output limits, prices and capabilities from OpenRouter's live model list, and `get_all_models` queries each provider's own models API. Both cache for five minutes, the last catalog is kept on disk for offline runs, and private or fine-tuned models can be registered alongside. |
| **Small supply-chain surface** | Three direct dependencies, `openai`, `pydantic` and `tiktoken`, for 20 installed packages in total, against 58 for litellm. API keys come from the environment or the call and are never written to logs. |
| **Testable** | `mock_response` returns realistic responses, streams included, without a network call or an API key. RouteHub itself ships with 324 offline tests. |
| **Async throughout** | `acompletion`, `aembedding`, async streams and async model listing, with HTTP clients cached per event loop. |
| **litellm-compatible** | The same function names and module paths (`routehub.utils`, `routehub.exceptions`), so migrating is mostly a change of import. See [Migrating from litellm](#migrating-from-litellm). |

## Performance

Measured on the same machine (macOS, Python 3.12) against litellm 1.76.1.

| | litellm | RouteHub |
|---|---|---|
| `import` time | 1,027 ms | **5.5 ms** |
| Modules loaded by `import` | 2,332 | **7** |
| Gateway overhead per call | 0.74 ms | **0.004 ms** |
| Installed packages | 58 | **20** |
| Network calls at import | 1 (model price list from GitHub, 5 s timeout) | **none** |

- **Import time** is the median of 7 fresh processes. litellm's GitHub fetch was turned off for its measurement, so the gap is pure import cost. RouteHub loads the OpenAI SDK on the first call, which brings import plus first call to 232 ms.
- **Overhead per call** is the time each gateway adds to a bare OpenAI SDK call (0.416 ms on its own), measured against an instant mock server so network time is excluded. It is the median of 300 calls.
- **Installed packages** counts the fully resolved dependency tree of each package.

Fast startup matters most for serverless functions, CLI tools, test suites and short-lived agent processes, where litellm's import alone can outweigh the work being done.

For the full benchmark, covering cold start, per-call overhead, streaming, async throughput and connection reuse against LiteLLM, any-llm, aisuite and the bare OpenAI and Anthropic SDKs, see the [paper](#paper).

## Supported providers

Prefix the model with its provider. Bare names starting with `gpt-`, `o1`/`o3`/`o4`, `claude-`, `gemini-`, `grok-`, `deepseek-` or `mistral-` need no prefix.

| Provider | Model string | API key variable |
|---|---|---|
| OpenAI | `gpt-5.4`, `openai/gpt-4o` | `OPENAI_API_KEY` |
| Anthropic (native API) | `claude-sonnet-4-6`, `anthropic/claude-opus-4-7` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-3-flash-preview` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Azure OpenAI | `azure/<deployment>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| Groq | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| xAI | `xai/grok-4` | `XAI_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-sonnet-4.6` | `OPENROUTER_API_KEY` |
| Together AI | `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHERAI_API_KEY` or `TOGETHER_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| Fireworks | `fireworks_ai/accounts/fireworks/models/...` | `FIREWORKS_API_KEY` |
| Perplexity | `perplexity/sonar` | `PERPLEXITYAI_API_KEY` |
| Cerebras | `cerebras/...` | `CEREBRAS_API_KEY` |
| DeepInfra | `deepinfra/...` | `DEEPINFRA_API_KEY` |
| SambaNova | `sambanova/...` | `SAMBANOVA_API_KEY` |
| NVIDIA NIM | `nvidia_nim/...` | `NVIDIA_NIM_API_KEY` |
| Moonshot | `moonshot/...` | `MOONSHOT_API_KEY` |
| Alibaba DashScope | `dashscope/...` | `DASHSCOPE_API_KEY` |
| Hugging Face | `huggingface/...` | `HF_TOKEN` |
| Ollama | `ollama/llama3` | none; `OLLAMA_API_BASE` for a remote server |
| LM Studio | `lm_studio/...` | none; `LM_STUDIO_API_BASE` |
| vLLM | `hosted_vllm/...` | `HOSTED_VLLM_API_BASE` |
| Any OpenAI-compatible server | any name with `api_base=` | optional |

Most providers also read a `<PROVIDER>_API_BASE` variable for a proxy or private endpoint, and `api_base=` (or `base_url=`) overrides it per call.

## Usage guide

### Streaming

```python
stream = routehub.completion(
    model="claude-sonnet-4-6",
    messages=messages,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.usage:
        print("\nusage:", chunk.usage)
    elif chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Streams yield `ChatCompletionChunk` objects and can be used in a `with` block. With `include_usage`, the final chunk carries `usage` and no choices, exactly as OpenAI does.

### Async

```python
import asyncio

async def main():
    response = await routehub.acompletion(model="gpt-5.4-mini", messages=messages)
    stream = await routehub.acompletion(model="gpt-5.4-mini", messages=messages, stream=True)
    async for chunk in stream:
        ...

asyncio.run(main())
```

`acompletion` takes exactly the same arguments as `completion`.

### Tool calling

```python
weather = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

response = routehub.completion(model="claude-sonnet-4-6", messages=messages, tools=[weather])
call = response.choices[0].message.tool_calls[0]
print(call.function.name, call.function.arguments)
```

Return results as `{"role": "tool", "tool_call_id": call.id, "content": "..."}` messages. For Claude, tool definitions, calls and results are translated to Anthropic's format and back, including parallel calls.

### Structured output

```python
from pydantic import BaseModel

class RiskReport(BaseModel):
    title: str
    severity: int

response = routehub.completion(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Assess the main risk of a single-region deployment."}],
    response_format=RiskReport,
)
report = RiskReport.model_validate_json(response.choices[0].message.content)
```

A pydantic model becomes a strict `json_schema` response format. For Claude, the schema is enforced through a forced tool call and returned as the message content.

### Reasoning

`reasoning_effort` accepts `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"` and `"xhigh"`. OpenAI-compatible providers receive it directly. For Claude it becomes a thinking budget (1,024 tokens for low, 2,048 for medium, 4,096 for high), with `max_tokens` raised above the budget when needed. Pass `thinking={...}` to configure Claude's thinking directly.

Claude's thinking is returned on the message as `reasoning_content` (text) and `thinking_blocks` (the signed blocks). Send `thinking_blocks` back on the assistant message for multi-turn tool use, as Anthropic requires. DeepSeek's `reasoning_content` comes through the same way.

Thinking tokens are reported in `usage.completion_tokens_details.reasoning_tokens` and counted in `completion_tokens`, as OpenAI does. Gemini's OpenAI-compatible endpoint bills them in `total_tokens` only, so RouteHub fills in the difference; an answer that spent its whole `max_tokens` budget thinking comes back with `finish_reason == "length"`, empty content, and a usage block that adds up.

### Prompt caching with Claude

```python
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": policy_handbook, "cache_control": {"type": "ephemeral"}}
        ],
    },
    {"role": "user", "content": "Which policies cover vendor onboarding?"},
]
response = routehub.completion(model="claude-sonnet-4-6", messages=messages)
print(response.usage.prompt_tokens_details.cached_tokens)   # tokens read from cache
print(response.usage.cache_creation_input_tokens)           # tokens written to cache
```

`cache_control` markers on messages and tools go to Anthropic unchanged. `prompt_tokens` includes cache reads and writes, so cost accounting stays correct. Markers pass through to OpenRouter and are stripped for every other provider. `supports_prompt_caching(model)` answers `True` only where markers are honoured, which is Claude, directly or through OpenRouter. OpenAI caches long prompts on its own and Gemini has a separate caching API, so both answer `False` even though their catalog entries carry a cache-read price.

### Embeddings

```python
response = routehub.embedding(model="text-embedding-3-small", input=["first text", "second text"])
vectors = [item.embedding for item in response.data]
```

`aembedding` is the async form. Any OpenAI-compatible embeddings endpoint works.

### Batches

```python
results = routehub.batched_completions(
    model="gpt-5.4-mini",
    requests=[
        {"messages": [{"role": "user", "content": "Classify: ..."}]},
        {"messages": [{"role": "user", "content": "Classify: ..."}], "max_tokens": 50},
    ],
)
```

`batched_completions` sends the requests through OpenAI's Batch API, Groq's Batch API for `groq/` models, or Anthropic's Message Batches API for Claude models. All three cost about half as much as regular calls and finish within 24 hours. Each request takes the same arguments as `completion`; keyword arguments such as `api_key` or `temperature` apply to every request. It polls every `poll_interval` seconds (30 by default) and returns one `ChatCompletion` per request in request order, or the RouteHub exception for a request that failed, without failing the rest. It raises `Timeout` after `timeout` seconds (24 hours by default) while the batch keeps running.

Pass `wait=False` to get a `BatchJob` back at once, with `status()`, `results()` and `cancel()`. Another process can collect the work with `routehub.BatchJob("openai", job.id).results()`, `routehub.BatchJob("groq", job.id)` for a Groq batch, or `routehub.BatchJob("anthropic", job.id)` for a Claude batch. Other providers raise `BadRequestError` for now.

`await routehub.abatched_completions(...)` takes the same arguments and keeps the event loop free while the batch runs.

### Testing without a provider

```python
response = routehub.completion(model="gpt-5.4-mini", messages=messages, mock_response="Approved.")
routehub.completion(model="gpt-5.4-mini", messages=messages, mock_response=TimeoutError("simulated outage"))
```

`mock_response` returns a real `ChatCompletion`, or a stream when `stream=True`, without any network call or API key. Pass an exception to rehearse failure handling.

## Configuration

Every setting is a parameter of the call; nothing is configured globally.

| Parameter | Default | Effect |
|---|---|---|
| `drop_params` | `False` | Drop parameters a model rejects instead of sending them: sampling parameters on OpenAI reasoning models, `reasoning_effort` on models without reasoning, thinking on older Claude models, and unknown keyword arguments. |
| `num_retries` | `None` | Retries on rate limits, 5xx errors and connection failures, with exponential backoff that honours `retry-after`. `None` means 2. |
| `ssl_verify` | `True` | TLS verification, or a path to a corporate CA bundle. |
| `gzip` | `None` | Gzip the request body sent to Claude's Messages API, which saves upload time on long histories. OpenAI-compatible providers always get an uncompressed body, because OpenAI rejects compressed ones. `None` reads `ROUTEHUB_GZIP_ON`. |
| `set_verbose` | `False` | Print the provider, model, parameter names and timing to stderr. API keys and message content are never printed. |
| `request_timeout` | `600.0` | Request timeout in seconds. `timeout=` overrides it. |

Also per call: `api_key`, `base_url` or `api_base`, `api_version` (Azure), `custom_llm_provider`, `extra_headers`, `extra_body` (always sent), `client` (your own configured OpenAI or httpx client, for example with a proxy or mTLS), and `mock_response`. Any other OpenAI parameter (`stop`, `seed`, `n`, `logprobs`, `prompt_cache_key` and so on) can be passed by name. Unrecognized keyword arguments are sent in the request body unless `drop_params` is on.

### Environment variables

| Variable | Effect |
|---|---|
| Provider keys and bases | See [Supported providers](#supported-providers). |
| `ROUTEHUB_USE_ORJSON` | RouteHub uses `orjson` for JSON when it is installed (the `fast` extra). Set this to `0`, `false`, `no` or `off` to use the standard `json` module instead. Read once, on first use. |
| `ROUTEHUB_GZIP_ON` | Set to `1`, `true`, `yes` or `on` to gzip Claude request bodies when a call doesn't pass `gzip=`. An explicit `gzip=False` still wins. Read on each call. |
| `ROUTEHUB_CACHE_DIR` | Directory for the on-disk copy of the model catalog, used when OpenRouter is unreachable. Defaults to `$XDG_CACHE_HOME/routehub` or `~/.cache/routehub`. Set to `off` to keep nothing on disk. Read on each fetch. |

## Error handling

| Exception | Raised when |
|---|---|
| `BadRequestError` | the provider returns 400 |
| `ContextWindowExceededError` | the prompt is longer than the model's context window |
| `ContentPolicyViolationError` | the provider refuses on content-policy grounds |
| `AuthenticationError` | the key is invalid (401), or no key is configured |
| `PermissionDeniedError` | the key lacks access (403) |
| `NotFoundError` | the model or endpoint does not exist (404) |
| `UnprocessableEntityError` | 422 |
| `RateLimitError` | 429 |
| `InternalServerError` | 500 and above |
| `ServiceUnavailableError` | 503, or Anthropic's 529 overloaded |
| `Timeout` | the request timed out |
| `APIConnectionError` | the provider could not be reached |

```python
try:
    routehub.completion(model="gpt-5.4-mini", messages=messages, num_retries=3)
except routehub.ContextWindowExceededError:
    ...  # trim the conversation and retry
except routehub.RateLimitError as error:
    print(error.status_code, error.llm_provider, error.model)
```

Each exception subclasses the matching OpenAI SDK exception, and the original error is chained as `__cause__`.

## Model catalog

```python
info = routehub.get_model_info("claude-sonnet-4-6")
info["max_input_tokens"]        # 1000000
info["max_output_tokens"]       # 128000
info["input_cost_per_token"]    # 3e-06
info["supports_reasoning"]      # True

routehub.get_max_tokens("gpt-5.4-mini")       # 128000
routehub.supports_vision("gpt-5.4-mini")      # True
routehub.supports_function_calling("deepseek/deepseek-chat")
"claude-sonnet-4-6" in routehub.model_list    # True
```

This data comes from [OpenRouter's model list](https://openrouter.ai/api/v1/models), fetched on first use and cached for five minutes; if a refresh fails, the previous data is kept. Each successful fetch is also written to `~/.cache/routehub/openrouter_models.json` (or `$XDG_CACHE_HOME/routehub`), and that file is used when OpenRouter cannot be reached, so an offline or air-gapped run keeps the catalog it last saw. `ROUTEHUB_CACHE_DIR` moves the file; set it to `off` to disable the disk cache. The `supports_*` functions return `False` for unknown models, and `get_model_info` raises `ModelNotMappedError`.

Names are matched the way providers spell them:

- Anthropic-style names find OpenRouter's (`claude-opus-4-7-20251001` finds `anthropic/claude-opus-4.7`).
- Serving providers' ids find the vendor's entry. `groq/llama-3.3-70b-versatile`, `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo`, `fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct` and `cerebras/llama3.3-70b` all resolve to `meta-llama/llama-3.3-70b-instruct`. The match strips serving suffixes (`-versatile`, `-instant`, `-turbo`, context sizes) and then shortens the name while it still carries a version or size number, so `no-such-model` stays unmapped. Output limits come from the vendor's catalog entry and may differ from what the serving provider allows.
- Embedding models, which OpenRouter does not list, come from a small built-in table: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`, `gemini-embedding-001` and `mistral-embed`, with `mode == "embedding"`.

Every entry carries `provider`, the RouteHub provider that serves it. A provider-prefixed lookup reports that prefix (`groq/llama-3.3-70b-versatile` reports `groq`), a bare name reports the vendor's provider. `litellm_provider` holds the same value for code written against litellm.

Register private, fine-tuned or self-hosted models. Registered entries take priority and never expire:

```python
routehub.register_model({
    "acme-support-ft": {
        "max_input_tokens": 32768,
        "max_output_tokens": 4096,
        "supports_function_calling": True,
        "input_cost_per_token": 0.000002,
    }
})
```

### Listing models from every provider

`routehub.get_all_models` asks each provider's own models API what it serves, concurrently, and returns one consistent shape:

```python
from routehub.get_all_models import get_all_models, get_model

models = get_all_models()   # every provider with a key configured, plus OpenRouter
chat = [m for m in models if m["type"] == "chat"]

get_all_models(["anthropic", "gemini"])          # selected providers only
get_model("anthropic", "claude-sonnet-4-6")      # one model, from the per-model endpoint
get_model("openrouter", "anthropic/claude-sonnet-4.6")["endpoints"]  # every host serving it
```

Sources: OpenRouter (no key needed), OpenAI, Anthropic, Gemini, Groq, xAI, DeepSeek, Mistral, Together, Fireworks, Cerebras, Cohere, and a local Ollama server when `OLLAMA_API_BASE` is set. Each entry has `id`, `provider`, `model`, `name`, `type`, `context_window`, `max_output_tokens`, `input_modalities`, `output_modalities`, `supports_vision`, `supports_function_calling`, `supports_reasoning`, `supports_structured_output`, `input_cost_per_token`, `output_cost_per_token`, `created`, `endpoints` and the provider's `raw` record, with `None` where a provider does not report a field.

Results are cached for five minutes, per provider and, for `get_model`, per model, failures included, so an unavailable provider is retried once per window rather than on every call. Pass `refresh=True` to the list functions to fetch again, or call `clear_cache()`, which also clears `get_model`. `aget_all_models` and `aget_models` are the async forms.

## Token counting

```python
routehub.encode(model="gpt-5.4-mini", text="hello world")   # token ids
routehub.token_counter(model="gpt-5.4-mini", messages=messages)
```

Counting uses tiktoken with the model's encoding, or `o200k_base` for models tiktoken does not know, so counts for non-OpenAI models are estimates. When no tokenizer can be loaded (offline, before tiktoken has cached its files), counts fall back to four bytes per token.

## Migrating from litellm

| litellm | RouteHub |
|---|---|
| `pip install litellm` | `pip install routehub` |
| `from litellm import completion, acompletion, embedding` | `from routehub import completion, acompletion, embedding` |
| `from litellm.utils import get_model_info, supports_vision` | `from routehub.utils import get_model_info, supports_vision` |
| `from litellm.exceptions import AuthenticationError` | `from routehub.exceptions import AuthenticationError` |
| `from litellm import model_list, encode` | `from routehub import model_list, encode` |
| `litellm.drop_params = True` | `completion(..., drop_params=True)` |
| `litellm.num_retries = 3` | `completion(..., num_retries=3)` |
| `litellm.ssl_verify = False` | `completion(..., ssl_verify=False)` |
| `litellm.set_verbose = True` | `completion(..., set_verbose=True)` |
| `ModelResponse` | `openai.types.chat.ChatCompletion` |

Other differences to plan for:

- Responses are OpenAI SDK objects, so use attribute access or `.model_dump()`; dictionary access (`response["choices"]`) is not supported.
- Settings apply to one call, not the whole process.
- Requests to OpenAI with `max_tokens` are sent as `max_completion_tokens`, which every current OpenAI chat model accepts and reasoning models require.
- litellm-only keyword arguments such as `metadata` and `caching` are accepted and ignored.

## Agent skill

[skills/routehub](skills/routehub/SKILL.md) is an [Agent Skill](https://agentskills.io) that shows coding agents how to call RouteHub correctly, including which litellm habits break here. To install it for Claude Code, copy the folder into `~/.claude/skills/` for every project, or into a project's `.claude/skills/`:

```bash
git clone https://github.com/The-Swarm-Corporation/RouteHub.git
mkdir -p ~/.claude/skills
cp -r RouteHub/skills/routehub ~/.claude/skills/
```


## Development

Commands, invariants and PR conventions for contributors and coding agents are in [CLAUDE.md](CLAUDE.md).

```bash
uv sync
uv run pytest                        # 324 offline tests, no API keys needed
uv run --extra fast pytest           # with orjson installed
uv run --with "openai<3" pytest      # against openai 2.x
uv run ruff check .                  # lint
```

`tests/live_smoke.py` exercises real providers and needs their API keys in the environment:

```bash
uv run python tests/live_smoke.py
```

## Paper

[RouteHub: A Low-Overhead, SDK-Native LLM Gateway for Agentic Workloads](docs/routehub.pdf) describes RouteHub's design and measures it against LiteLLM, any-llm, aisuite and the bare provider SDKs, each in its own environment against an instant mock server, plus connection reuse over a real network.

If you use RouteHub in your research, please cite:

```bibtex
@misc{gomez2026routehub,
  title        = {RouteHub: A Low-Overhead, SDK-Native LLM Gateway for Agentic Workloads},
  author       = {Gomez, Kye and Grandhi, Shryuk and Gazali, Ayaan},
  year         = {2026},
  month        = oct,
  howpublished = {\url{https://github.com/The-Swarm-Corporation/RouteHub/blob/main/docs/routehub.pdf}},
  note         = {The Swarm Corporation}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
