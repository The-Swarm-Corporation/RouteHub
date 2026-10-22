---
name: routehub
description: Call 20+ LLM providers (OpenAI, Anthropic, Gemini, Groq, OpenRouter, Ollama and more) through RouteHub's litellm-compatible Python API. Use when writing, reviewing or migrating Python code that calls routehub, or replacing litellm with routehub.
---

# RouteHub

RouteHub has litellm's function names, but responses are OpenAI SDK objects and every setting is an argument to the call. Code written from litellm habits often runs and then does the wrong thing. Use the API as written here. [providers.md](providers.md) lists every provider prefix, key variable and base-URL variable; [migrating-from-litellm.md](migrating-from-litellm.md) maps litellm imports, settings and objects to RouteHub.

## Install

```bash
pip install routehub
pip install "routehub[fast]"
```

The `fast` extra adds orjson. The `aiohttp` extra lets async calls use openai's aiohttp transport. Python 3.10+.

## Model strings

Write `provider/model`, such as `groq/llama-3.3-70b-versatile` or `openrouter/anthropic/claude-sonnet-4.6`. Bare names are inferred from their prefix (full list in [providers.md](providers.md)): `gpt-`, `o1`, `o3`, `o4`, `chatgpt-`, `codex-`, `text-embedding-` (OpenAI), `claude-` (Anthropic), `gemini-`, `gemma-` (Gemini), `grok-` (xAI), `deepseek-` (DeepSeek), `mistral-`, `codestral`, `magistral-`, `ministral-`, `pixtral-`, `devstral-` (Mistral), `command-` (Cohere). Any other name needs a prefix, `custom_llm_provider=`, or `api_base=` (an unknown name with `api_base` is sent to an OpenAI-compatible server). Otherwise `BadRequestError` is raised before any request. The key comes from `api_key=` or the provider's variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, ...). The base URL comes from `api_base=` (or `base_url=`) or the provider's `*_API_BASE` variable. See [providers.md](providers.md).

```python
import routehub

routehub.get_llm_provider("groq/llama-3.3-70b-versatile")
routehub.get_llm_provider("claude-sonnet-4-6")
```

`get_llm_provider` returns `(model, provider, api_key, api_base)` without sending anything. Use it to check a model string.

## Completion

```python
messages = [{"role": "user", "content": "Summarize our Q3 risks."}]

response = routehub.completion(
    model="gpt-5.4-mini", messages=messages
)
print(response.choices[0].message.content)
print(response.usage.total_tokens)
```

Every provider, Claude included, returns `openai.types.chat.ChatCompletion`. `acompletion` is the async form and takes exactly the same arguments:

```python
import asyncio

response = asyncio.run(
    routehub.acompletion(model="claude-sonnet-4-6", messages=messages)
)
```

## Streaming

```python
stream = routehub.completion(
    model="claude-sonnet-4-6",
    messages=messages,
    stream=True,
    stream_options={"include_usage": True},
)
with stream:
    for chunk in stream:
        if chunk.usage:
            print(chunk.usage.total_tokens)
        elif chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

Chunks are `ChatCompletionChunk`. With `include_usage`, the last chunk carries `usage` and an empty `choices` list, so check `chunk.choices` before indexing it. With `acompletion(..., stream=True)`, use `async for` and `async with`.

## Tool calling

```python
import json


def get_weather(city):
    return f"Sunny in {city}"


weather_tool = {
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
chat = [{"role": "user", "content": "Weather in Paris?"}]
response = routehub.completion(
    model="claude-sonnet-4-6", messages=chat, tools=[weather_tool]
)
message = response.choices[0].message
chat.append(message.model_dump(exclude_none=True))
for call in message.tool_calls or []:
    result = get_weather(**json.loads(call.function.arguments))
    chat.append(
        {"role": "tool", "tool_call_id": call.id, "content": result}
    )
```

Tools use the OpenAI format for every provider; RouteHub translates them for Claude, parallel calls included. `tool_calls` is `None` when the model answers in text. Claude 5.1 and later (Opus 5.5, Sonnet 5.5, Fable 5.1) reject a forced `tool_choice` (`"required"` or a named tool): RouteHub raises `UnsupportedParamsError`, or sends `"auto"` when `drop_params=True`.

## Structured output

```python
from pydantic import BaseModel


class RiskReport(BaseModel):
    title: str
    severity: int


response = routehub.completion(
    model="gpt-5.4-mini",
    messages=messages,
    response_format=RiskReport,
)
report = RiskReport.model_validate_json(
    response.choices[0].message.content
)
```

A pydantic class becomes a strict `json_schema` response format; a dict is sent as given. The JSON arrives as the message content string, so parse it yourself. For Claude the schema is enforced through a forced tool call (native structured output on Claude 5.1 and later) and still returned as the content.

## Reasoning

```python
response = routehub.completion(
    model="claude-sonnet-4-6",
    messages=messages,
    reasoning_effort="medium",
)
message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None)
```

`reasoning_effort` takes one of `routehub.REASONING_EFFORTS`: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`. OpenAI-compatible providers receive it as is. For Claude before 4.7 it becomes a thinking budget (1,024 tokens for minimal and low, 2,048 for medium, 4,096 for high, 8,192 for xhigh), with `max_tokens` raised above the budget when needed. Claude 4.7 and later (Opus 4.7, Claude 5 models) get adaptive thinking at that effort. Those models reject `temperature`, `top_p` and `top_k` on every call: RouteHub raises `UnsupportedParamsError`, or drops them with `drop_params=True`. `thinking={"type": "enabled", "budget_tokens": 2048}` sets Claude's thinking directly and wins over `reasoning_effort`.

`reasoning_content` (text) and `thinking_blocks` (signed blocks) exist on the message only when the provider returned reasoning, so read them with `getattr`. In a Claude tool loop with thinking, append `message.model_dump(exclude_none=True)` as above: it keeps `thinking_blocks`, which Anthropic requires back, and RouteHub strips them for other providers.

## Prompt caching with Claude

```python
handbook = "Vendor onboarding policy: ..."
cached_chat = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": handbook,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {"role": "user", "content": "Which policies cover vendors?"},
]
response = routehub.completion(
    model="claude-sonnet-4-6", messages=cached_chat
)
details = response.usage.prompt_tokens_details
cache_reads = details.cached_tokens if details else 0
```

`cache_control` on content blocks and tool definitions goes to Anthropic unchanged and passes through to OpenRouter. It is stripped for every other provider. For Claude, `prompt_tokens` includes cache reads and writes, and usage also carries `cache_read_input_tokens` and `cache_creation_input_tokens`.

## Embeddings

```python
response = routehub.embedding(
    model="text-embedding-3-small",
    input=["first text", "second text"],
)
vectors = [item.embedding for item in response.data]
```

Returns `CreateEmbeddingResponse`. `aembedding` is the async form. Any OpenAI-compatible embeddings endpoint works; Anthropic has none and raises `BadRequestError`.

## Token counts and model info

```python
prompt_tokens = routehub.token_counter(
    model="gpt-5.4-mini", messages=messages
)

info = routehub.get_model_info("claude-sonnet-4-6")
window = info["max_input_tokens"]
price_in = info["input_cost_per_token"]
price_out = info["output_cost_per_token"]

routehub.register_model(
    {"acme-support-ft": {"max_input_tokens": 32768}}
)
routehub.supports_vision("gpt-5.4-mini")
```

`token_counter(model=..., text=...)` counts a plain string. It uses tiktoken, with `o200k_base` for models tiktoken does not know, so non-OpenAI counts are estimates. `get_model_info` reads OpenRouter's live model list on first use (a network call, cached five minutes) and raises `ModelNotMappedError` for unknown models; prices can be `None`. Registered models take priority and never expire. The `supports_*` functions return `False` for unknown models.

## Per-call settings

```python
import openai

response = routehub.completion(
    model="gpt-5.4-mini",
    messages=messages,
    num_retries=3,
    drop_params=True,
    ssl_verify="/etc/ssl/certs/corp-ca.pem",
    timeout=30,
    keepalive_expiry=30.0,
)
proxy = openai.OpenAI(
    base_url="https://llm-proxy.internal/v1", api_key="sk-proxy"
)
response = routehub.completion(
    model="gpt-5.4-mini", messages=messages, client=proxy
)
```

There are no module-level settings. `routehub.drop_params = True` sets an attribute nothing reads.

| Argument | Default | Effect |
|---|---|---|
| `num_retries` | 2 | Retries on rate limits, server errors, timeouts and connection failures, with backoff that honours `retry-after`. |
| `drop_params` | `False` | Drop what the model rejects instead of failing: sampling parameters on OpenAI reasoning models, `reasoning_effort` on other OpenAI models, thinking on pre-thinking Claude models, sampling on Claude 4.7+, forced `tool_choice` on Claude 5.1+. Unknown keyword arguments are then not sent either. |
| `ssl_verify` | `True` | TLS verification, or a path to a CA bundle. |
| `request_timeout` / `timeout` | `600.0` / `None` | Seconds. `timeout` wins when set. |
| `keepalive_expiry` | `60.0` | Seconds an idle pooled connection stays open. |
| `gzip` | `None` | Gzip Claude request bodies; other providers are always sent uncompressed. `None` reads `ROUTEHUB_GZIP_ON` (`1`, `true`, `yes`, `on`). |
| `set_verbose` | `False` | Provider, model, parameter names and timing to stderr. Never keys or message content. |
| `client` | `None` | Your own client: `openai.OpenAI` / `AsyncOpenAI` for OpenAI-compatible providers, an httpx `Client` / `AsyncClient` for Claude. No API key is required, and the client's own TLS, pool and (for OpenAI SDK clients) retry settings apply. |

Also per call: `api_key`, `api_base` or `base_url`, `api_version` (Azure), `custom_llm_provider`, `extra_headers`, `extra_body` (always sent, even with `drop_params`). Other OpenAI parameters (`stop`, `seed`, `n`, `logprobs`, `prompt_cache_key`, ...) can be passed by name.

## Errors

| Exception | Raised when |
|---|---|
| `BadRequestError` | 400, or a bad model string or empty `messages` before any request |
| `UnsupportedParamsError` | a parameter the model rejects, before any request (subclass of `BadRequestError`) |
| `ContextWindowExceededError` | 413, or a 400 saying the prompt is too long |
| `ContentPolicyViolationError` | a 400 citing content policy |
| `AuthenticationError` | 401, or no API key configured (before any request) |
| `PermissionDeniedError` | 403 |
| `NotFoundError` | 404 |
| `UnprocessableEntityError` | 422 |
| `RateLimitError` | 429 |
| `ServiceUnavailableError` | 503, or Anthropic's 529 |
| `InternalServerError` | any other 5xx |
| `APIError` | any other error status |
| `Timeout` | the request timed out |
| `APIConnectionError` | the provider could not be reached |
| `ModelNotMappedError` | `get_model_info` or `get_max_tokens` on an unknown model (a plain `Exception`) |

Every exception except `ModelNotMappedError` subclasses the matching OpenAI SDK exception (`openai.RateLimitError` and so on), so existing `openai` handlers keep working. Status errors carry `status_code`, `llm_provider`, `model` and `message`. `Timeout` is not a `routehub.APIConnectionError`, and `ServiceUnavailableError` is not a `routehub.InternalServerError`; catch `openai.APIConnectionError` or `openai.InternalServerError` to cover both.

## Testing without a provider

```python
reply = routehub.completion(
    model="gpt-5.4-mini",
    messages=messages,
    mock_response="Approved.",
)
assert reply.choices[0].message.content == "Approved."

try:
    routehub.completion(
        model="gpt-5.4-mini",
        messages=messages,
        mock_response=routehub.RateLimitError("simulated"),
    )
except routehub.RateLimitError as error:
    assert error.status_code == 429
```

`mock_response` returns a real `ChatCompletion`, or with `stream=True` a stream of chunks split on spaces (async too), with no network call and no API key. An exception is raised as given. The model string is not resolved, so a mock does not catch a wrong model name, and usage counts are rough estimates. `embedding` has no `mock_response`: the argument is ignored and the real API is called, so pass `client=` a fake OpenAI client instead.

## Pitfalls

- Responses are OpenAI SDK objects. Use attribute access (`response.choices[0].message.content`) or `response.model_dump()`; `response["choices"]` raises `TypeError`.
- `max_tokens` is sent as `max_completion_tokens` to OpenAI, and to Azure for reasoning models (`o1`, `o3`, `o4`, `gpt-5`, `gpt-6`, except `-chat` variants). Claude gets 4,096 output tokens when neither is set.
- `cache_control` markers are kept only for Anthropic and OpenRouter.
- litellm-only keyword arguments (`metadata`, `caching`, `cache`, `fallbacks`, `retry_policy`, `logger_fn`, any `litellm_*` name, ...) are accepted and ignored. `fallbacks` does nothing: catch the error and call the next model yourself.
- Any other unknown keyword argument is sent to the provider in the request body unless `drop_params=True`. A misspelt argument is not caught locally: it reaches the provider, which may reject or ignore it.
- Settings are per call. Pass `num_retries`, `drop_params`, `ssl_verify` and the rest to each call, or wrap `completion` in your own function.
