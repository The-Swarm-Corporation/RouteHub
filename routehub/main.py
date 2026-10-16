"""completion, acompletion and embeddings across providers."""

import asyncio
import inspect
import random
import sys
import time
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
)

from routehub._json import dumps, loads
from routehub.clients import http_client, openai_client
from routehub.providers import PROVIDERS, get_llm_provider

ReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh"
]
REASONING_EFFORTS: Tuple[str, ...] = get_args(ReasoningEffort)

DEFAULT_NUM_RETRIES = 2

# Chat parameters sent to OpenAI-compatible APIs as they are.
OPENAI_PARAMS = frozenset(
    {
        "audio",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "modalities",
        "moderation",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
    }
)

# Arguments that configure the gateway rather than the request.
_OPTIONS = (
    "api_key",
    "api_base",
    "base_url",
    "api_version",
    "custom_llm_provider",
    "extra_headers",
    "extra_body",
    "thinking",
    "drop_params",
    "num_retries",
    "ssl_verify",
    "keepalive_expiry",
    "set_verbose",
    "request_timeout",
    "timeout",
    "mock_response",
    "client",
)

# litellm keyword arguments that only configure litellm and are never sent.
_IGNORED_KWARGS = frozenset(
    {
        "acompletion",
        "additional_drop_params",
        "allowed_openai_params",
        "base_model",
        "cache",
        "caching",
        "context_window_fallback_dict",
        "deployment_id",
        "fallbacks",
        "input_cost_per_token",
        "logger_fn",
        "metadata",
        "model_info",
        "model_list",
        "no-log",
        "num_retries_per_request",
        "output_cost_per_token",
        "preset_cache_key",
        "proxy_server_request",
        "retry_policy",
    }
)

_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")
_REASONING_REJECTS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    "logit_bias",
)
# Providers that pass Anthropic cache_control markers through to Claude.
_CACHE_CONTROL_PROVIDERS = frozenset({"openrouter"})
_RESPONSE_ONLY_KEYS = (
    "thinking_blocks",
    "reasoning_content",
    "provider_specific_fields",
)
_RETRYABLE_STATUS = frozenset(
    {408, 409, 429, 500, 502, 503, 504, 529}
)


def get_reasoning_efforts() -> Tuple[str, ...]:
    """Return the reasoning_effort values completion accepts.

    Returns:
        Tuple[str, ...]: The accepted values.
    """
    return REASONING_EFFORTS


@dataclass
class _Call:
    """Everything needed to send one chat request."""

    model: str
    bare: str
    provider: str
    api_key: Optional[str]
    api_base: Optional[str]
    api_version: Optional[str]
    organization: Optional[str]
    messages: List[dict]
    params: Dict[str, Any]
    extras: Dict[str, Any]
    extra_headers: Optional[dict]
    extra_body: Optional[dict]
    drop_params: bool
    max_retries: int
    ssl_verify: Union[bool, str]
    keepalive_expiry: float
    timeout: Any
    verbose: bool
    client: Any


def _log(verbose: bool, message: str) -> None:
    """Print a debug line when set_verbose is on.

    Args:
        verbose (bool): Whether verbose output is on.
        message (str): The line to print.
    """
    if verbose:
        print(f"routehub: {message}", file=sys.stderr)


def _response_format_param(response_format: Any) -> Any:
    """Turn a pydantic model class into a json_schema response_format.

    Args:
        response_format (Any): A response_format dict or a pydantic model class.

    Returns:
        Any: A response_format dict.
    """
    if not isinstance(response_format, type):
        return response_format
    try:
        from pydantic import BaseModel
    except ImportError:
        return response_format
    if not issubclass(response_format, BaseModel):
        return response_format
    try:
        from openai.lib._pydantic import to_strict_json_schema

        schema = to_strict_json_schema(response_format)
    except Exception:
        schema = response_format.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_format.__name__,
            "schema": schema,
            "strict": True,
        },
    }


def _split_args(
    args: Dict[str, Any],
) -> Tuple[str, Any, dict, dict, dict]:
    """Separate completion arguments into request parameters and options.

    Args:
        args (Dict[str, Any]): The completion call's locals.

    Returns:
        Tuple[str, Any, dict, dict, dict]: Model, messages, OpenAI params,
        gateway options, and remaining keyword arguments.
    """
    kwargs = dict(args.pop("kwargs") or {})
    model = args.pop("model")
    messages = args.pop("messages")
    options = {name: args.pop(name) for name in _OPTIONS}
    params = {k: v for k, v in args.items() if v is not None}
    for name in list(kwargs):
        if name in OPENAI_PARAMS and kwargs[name] is not None:
            params[name] = kwargs.pop(name)
    return model, messages, params, options, kwargs


def _build_call(
    model: str,
    messages: Any,
    params: dict,
    options: dict,
    kwargs: dict,
) -> _Call:
    """Resolve the provider and settle every setting for one request.

    Args:
        model (str): Model string as passed.
        messages (Any): Chat messages.
        params (dict): OpenAI parameters that were set.
        options (dict): Gateway options.
        kwargs (dict): Remaining keyword arguments.

    Returns:
        _Call: The resolved request.
    """
    from routehub.exceptions import (
        AuthenticationError,
        BadRequestError,
    )

    if not model:
        raise BadRequestError("model is required")
    if not messages:
        raise BadRequestError(
            "messages is required and cannot be empty", model=model
        )
    bare, provider, api_key, api_base = get_llm_provider(
        model,
        options["custom_llm_provider"],
        options["api_base"] or options["base_url"],
        options["api_key"],
    )
    api_version = options["api_version"]
    if provider == "azure" and not api_version:
        import os

        api_version = os.environ.get(
            "AZURE_API_VERSION"
        ) or os.environ.get("OPENAI_API_VERSION")
    if not api_key and options["client"] is None:
        names = " or ".join(PROVIDERS[provider].key_env) or "api_key"
        raise AuthenticationError(
            f"No API key for provider {provider!r}. Set {names}, or pass api_key.",
            llm_provider=provider,
            model=model,
        )

    headers = dict(options["extra_headers"] or {})
    headers.update(kwargs.pop("headers", None) or {})
    num_retries = options["num_retries"]
    if num_retries is None:
        num_retries = kwargs.pop("max_retries", None)
    if num_retries is None:
        num_retries = DEFAULT_NUM_RETRIES
    timeout = options["timeout"]
    if timeout is None:
        timeout = options["request_timeout"]
    organization = kwargs.pop("organization", None)

    if params.get("response_format") is not None:
        params["response_format"] = _response_format_param(
            params["response_format"]
        )
    extras = {
        k: v
        for k, v in kwargs.items()
        if v is not None
        and k not in _IGNORED_KWARGS
        and not k.startswith("litellm_")
    }
    if options["thinking"] is not None:
        extras["thinking"] = options["thinking"]

    return _Call(
        model=model,
        bare=bare,
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        organization=organization,
        messages=list(messages),
        params=params,
        extras=extras,
        extra_headers=headers or None,
        extra_body=options["extra_body"],
        drop_params=bool(options["drop_params"]),
        max_retries=int(num_retries),
        ssl_verify=options["ssl_verify"],
        keepalive_expiry=options["keepalive_expiry"],
        timeout=timeout,
        verbose=bool(options["set_verbose"]),
        client=options["client"],
    )


def _is_openai_reasoning(model: str) -> bool:
    """Whether an OpenAI model belongs to a reasoning family.

    Args:
        model (str): Bare OpenAI model name.

    Returns:
        bool: True for o-series and gpt-5 and later, except chat variants.
    """
    name = model.lower().removeprefix("ft:")
    return (
        name.startswith(_OPENAI_REASONING_PREFIXES)
        and "-chat" not in name
    )


def _strip_cache_control(item: Any) -> Any:
    """Return a copy of a tool or content block without cache_control.

    Args:
        item (Any): A tool definition or content block.

    Returns:
        Any: The item without the marker.
    """
    if isinstance(item, dict) and "cache_control" in item:
        return {k: v for k, v in item.items() if k != "cache_control"}
    return item


def _clean_messages(
    messages: List[dict], keep_cache_control: bool
) -> List[dict]:
    """Remove fields OpenAI-compatible APIs reject from chat messages.

    Args:
        messages (List[dict]): Chat messages, possibly carrying response-only
            fields or Anthropic cache markers.
        keep_cache_control (bool): Leave cache_control markers in place.

    Returns:
        List[dict]: Messages safe to send; the originals when nothing changed.
    """
    cleaned = []
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        new = message
        if any(key in message for key in _RESPONSE_ONLY_KEYS):
            new = {
                k: v
                for k, v in message.items()
                if k not in _RESPONSE_ONLY_KEYS
            }
        content = new.get("content")
        if (
            not keep_cache_control
            and isinstance(content, list)
            and any(
                isinstance(p, dict) and "cache_control" in p
                for p in content
            )
        ):
            new = dict(new)
            new["content"] = [
                _strip_cache_control(p) for p in content
            ]
        if not keep_cache_control and "cache_control" in new:
            new = _strip_cache_control(new)
        changed = changed or new is not message
        cleaned.append(new)
    return cleaned if changed else messages


_SDK_PARAMS: Dict[type, frozenset] = {}


def _sdk_params(completions: Any) -> frozenset:
    """Return the keyword arguments this SDK version's create() accepts.

    Args:
        completions (Any): The client's chat.completions resource.

    Returns:
        frozenset: Parameter names.
    """
    kind = type(completions)
    if kind not in _SDK_PARAMS:
        _SDK_PARAMS[kind] = frozenset(
            inspect.signature(completions.create).parameters
        )
    return _SDK_PARAMS[kind]


def _drop_rejected(call: _Call, request: dict, effort: str) -> None:
    """Remove what an OpenAI model rejects, when drop_params is on.

    Args:
        call (_Call): The resolved request.
        request (dict): The create() arguments, changed in place.
        effort (str): The reasoning parameter, "reasoning_effort" for
            chat or "reasoning" for the Responses API.
    """
    if call.drop_params and call.provider in ("openai", "azure"):
        if _is_openai_reasoning(call.bare):
            for name in _REASONING_REJECTS:
                if name == "temperature" and request.get(name) == 1:
                    continue
                request.pop(name, None)
        else:
            request.pop(effort, None)


def _openai_request(call: _Call) -> Tuple[dict, dict]:
    """Build the chat.completions.create arguments for a provider.

    Args:
        call (_Call): The resolved request.

    Returns:
        Tuple[dict, dict]: The create() arguments and the extra body.
    """
    keep_markers = call.provider in _CACHE_CONTROL_PROVIDERS
    request = dict(call.params)
    request["model"] = call.bare
    request["messages"] = _clean_messages(call.messages, keep_markers)

    reasoning_model = call.provider in (
        "openai",
        "azure",
    ) and _is_openai_reasoning(call.bare)
    # OpenAI accepts max_completion_tokens on every chat model, reasoning models require it.
    if call.provider == "openai" or reasoning_model:
        if (
            "max_tokens" in request
            and "max_completion_tokens" not in request
        ):
            request["max_completion_tokens"] = request.pop(
                "max_tokens"
            )
    if request.get("reasoning_effort") == "None":
        request.pop("reasoning_effort")
    if not request.get("stream"):
        request.pop("stream_options", None)
    if not request.get("tools") and not request.get("functions"):
        request.pop("tool_choice", None)
        request.pop("parallel_tool_calls", None)
    if request.get("tools") and not keep_markers:
        request["tools"] = [
            _strip_cache_control(t) for t in request["tools"]
        ]
    _drop_rejected(call, request, "reasoning_effort")

    extra_body = dict(call.extra_body or {})
    if not call.drop_params:
        for key, value in call.extras.items():
            extra_body.setdefault(key, value)
    return request, extra_body


def _create_kwargs(
    completions: Any, request: dict, extra_body: dict, call: _Call
) -> dict:
    """Fit the request to this SDK version's create() signature.

    Parameters newer than the installed SDK go into the extra body.

    Args:
        completions (Any): The client's chat.completions resource.
        request (dict): The create() arguments.
        extra_body (dict): Fields to send outside the typed parameters.
        call (_Call): The resolved request.

    Returns:
        dict: Keyword arguments for create().
    """
    accepted = _sdk_params(completions)
    kwargs = {}
    for key, value in request.items():
        if key in accepted:
            kwargs[key] = value
        else:
            extra_body.setdefault(key, value)
    kwargs["extra_headers"] = call.extra_headers
    kwargs["extra_body"] = extra_body or None
    kwargs["timeout"] = call.timeout
    return kwargs


def _openai_client(call: _Call, is_async: bool) -> Any:
    """Return the caller's client or a cached SDK client.

    Args:
        call (_Call): The resolved request.
        is_async (bool): Whether to use the async client.

    Returns:
        Any: An OpenAI SDK client.
    """
    if call.client is not None:
        return call.client
    return openai_client(
        call.provider,
        call.api_key,
        call.api_base,
        is_async=is_async,
        max_retries=call.max_retries,
        ssl_verify=call.ssl_verify,
        keepalive_expiry=call.keepalive_expiry,
        api_version=call.api_version,
        organization=call.organization,
    )


def _openai_complete(call: _Call) -> Any:
    """Send a chat request through the OpenAI SDK.

    Args:
        call (_Call): The resolved request.

    Returns:
        Any: A ChatCompletion, or a ChatStream when streaming.
    """
    from routehub.exceptions import map_exception
    from routehub.streaming import ChatStream

    request, extra_body = _openai_request(call)
    try:
        client = _openai_client(call, is_async=False)
        completions = client.chat.completions
        response = completions.create(
            **_create_kwargs(completions, request, extra_body, call)
        )
    except Exception as error:
        mapped = map_exception(error, call.provider, call.model)
        if mapped is error:
            raise
        raise mapped from error
    if request.get("stream"):
        return ChatStream(
            iter(response), response.close, call.provider, call.model
        )
    return response


async def _openai_acomplete(call: _Call) -> Any:
    """Send a chat request through the async OpenAI SDK.

    Args:
        call (_Call): The resolved request.

    Returns:
        Any: A ChatCompletion, or an AsyncChatStream when streaming.
    """
    from routehub.exceptions import map_exception
    from routehub.streaming import AsyncChatStream

    request, extra_body = _openai_request(call)
    try:
        client = _openai_client(call, is_async=True)
        completions = client.chat.completions
        response = await completions.create(
            **_create_kwargs(completions, request, extra_body, call)
        )
    except Exception as error:
        mapped = map_exception(error, call.provider, call.model)
        if mapped is error:
            raise
        raise mapped from error
    if request.get("stream"):
        return AsyncChatStream(
            response.__aiter__(),
            response.close,
            call.provider,
            call.model,
        )
    return response


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter.

    Args:
        attempt (int): Zero-based retry number.

    Returns:
        float: Seconds to wait.
    """
    return min(0.5 * 2**attempt, 8.0) * (0.75 + random.random() * 0.5)


def _retry_delay(response: Any, attempt: int) -> float:
    """Seconds to wait before retrying, honouring retry-after.

    Args:
        response (Any): The failed HTTP response.
        attempt (int): Zero-based retry number.

    Returns:
        float: Seconds to wait.
    """
    header = response.headers.get("retry-after")
    try:
        return min(float(header), 60.0)
    except (TypeError, ValueError):
        return _backoff(attempt)


def _anthropic_prepare(call: _Call) -> Tuple[str, dict, dict, bool]:
    """Build the Messages API URL, headers and body.

    Args:
        call (_Call): The resolved request.

    Returns:
        Tuple[str, dict, dict, bool]: URL, headers, body, and whether a JSON
        schema is enforced through a forced tool call.
    """
    from routehub import anthropic

    body, json_mode = anthropic.build_request(
        call.bare,
        call.messages,
        call.params,
        call.extras,
        call.drop_params,
    )
    if call.extra_body:
        body.update(call.extra_body)
    return (
        anthropic.messages_url(call.api_base),
        anthropic.headers(call.api_key, call.extra_headers),
        body,
        json_mode,
    )


def _anthropic_send(
    client: Any, url: str, headers: dict, body: dict, call: _Call
) -> Any:
    """POST to the Messages API, retrying retryable failures.

    Args:
        client (Any): An httpx-compatible client.
        url (str): The Messages endpoint.
        headers (dict): Request headers.
        body (dict): Request body.
        call (_Call): The resolved request.

    Returns:
        Any: The successful HTTP response, still open when streaming.
    """
    from routehub import anthropic
    from routehub._http import httpx
    from routehub.exceptions import map_exception

    stream = bool(body.get("stream"))
    attempt = 0
    while True:
        try:
            request = client.build_request(
                "POST",
                url,
                headers=headers,
                content=dumps(body),
                timeout=call.timeout,
            )
            response = client.send(request, stream=stream)
        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as error:
            if attempt < call.max_retries:
                time.sleep(_backoff(attempt))
                attempt += 1
                continue
            raise map_exception(
                error, call.provider, call.model
            ) from error
        if response.status_code < 400:
            return response
        if stream:
            response.read()
        if (
            response.status_code in _RETRYABLE_STATUS
            and attempt < call.max_retries
        ):
            delay = _retry_delay(response, attempt)
            response.close()
            time.sleep(delay)
            attempt += 1
            continue
        error = anthropic.response_error(response, call.model)
        response.close()
        raise error


async def _anthropic_asend(
    client: Any, url: str, headers: dict, body: dict, call: _Call
) -> Any:
    """POST to the Messages API asynchronously, retrying retryable failures.

    Args:
        client (Any): An httpx-compatible async client.
        url (str): The Messages endpoint.
        headers (dict): Request headers.
        body (dict): Request body.
        call (_Call): The resolved request.

    Returns:
        Any: The successful HTTP response, still open when streaming.
    """
    from routehub import anthropic
    from routehub._http import httpx
    from routehub.exceptions import map_exception

    stream = bool(body.get("stream"))
    attempt = 0
    while True:
        try:
            request = client.build_request(
                "POST",
                url,
                headers=headers,
                content=dumps(body),
                timeout=call.timeout,
            )
            response = await client.send(request, stream=stream)
        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as error:
            if attempt < call.max_retries:
                await asyncio.sleep(_backoff(attempt))
                attempt += 1
                continue
            raise map_exception(
                error, call.provider, call.model
            ) from error
        if response.status_code < 400:
            return response
        if stream:
            await response.aread()
        if (
            response.status_code in _RETRYABLE_STATUS
            and attempt < call.max_retries
        ):
            delay = _retry_delay(response, attempt)
            await response.aclose()
            await asyncio.sleep(delay)
            attempt += 1
            continue
        error = anthropic.response_error(response, call.model)
        await response.aclose()
        raise error


def _include_usage(call: _Call) -> bool:
    """Whether the caller asked for a trailing usage chunk.

    Args:
        call (_Call): The resolved request.

    Returns:
        bool: True when stream_options.include_usage is set.
    """
    options = call.params.get("stream_options") or {}
    return bool(options.get("include_usage"))


def _anthropic_complete(call: _Call) -> Any:
    """Send a chat request to the Anthropic Messages API.

    Args:
        call (_Call): The resolved request.

    Returns:
        Any: A ChatCompletion, or a ChatStream when streaming.
    """
    from routehub import anthropic
    from routehub.streaming import ChatStream

    url, headers, body, json_mode = _anthropic_prepare(call)
    client = call.client or http_client(
        is_async=False,
        ssl_verify=call.ssl_verify,
        keepalive_expiry=call.keepalive_expiry,
    )
    response = _anthropic_send(client, url, headers, body, call)
    if not body.get("stream"):
        return anthropic.to_chat_completion(
            loads(response.content), call.bare, json_mode
        )
    translator = anthropic.StreamTranslator(
        call.bare, _include_usage(call), json_mode
    )

    def chunks():
        for event, raw in anthropic.iter_sse(response.iter_lines()):
            yield from translator.handle(event, loads(raw))

    return ChatStream(
        chunks(), response.close, call.provider, call.model
    )


async def _anthropic_acomplete(call: _Call) -> Any:
    """Send a chat request to the Anthropic Messages API asynchronously.

    Args:
        call (_Call): The resolved request.

    Returns:
        Any: A ChatCompletion, or an AsyncChatStream when streaming.
    """
    from routehub import anthropic
    from routehub.streaming import AsyncChatStream

    url, headers, body, json_mode = _anthropic_prepare(call)
    client = call.client or http_client(
        is_async=True,
        ssl_verify=call.ssl_verify,
        keepalive_expiry=call.keepalive_expiry,
    )
    response = await _anthropic_asend(
        client, url, headers, body, call
    )
    if not body.get("stream"):
        return anthropic.to_chat_completion(
            loads(response.content), call.bare, json_mode
        )
    translator = anthropic.StreamTranslator(
        call.bare, _include_usage(call), json_mode
    )

    async def chunks():
        async for event, raw in anthropic.aiter_sse(
            response.aiter_lines()
        ):
            for chunk in translator.handle(event, loads(raw)):
                yield chunk

    return AsyncChatStream(
        chunks(), response.aclose, call.provider, call.model
    )


def _estimate_tokens(text: str) -> int:
    """Rough token count used only for mock responses.

    Args:
        text (str): The text to measure.

    Returns:
        int: About one token per four characters.
    """
    return max(1, len(text) // 4)


def _mock_chunks(
    model: str, content: str, include_usage: bool
) -> List[Any]:
    """Build the chunks of a mocked streaming response.

    Args:
        model (str): Model name to report.
        content (str): The mocked reply.
        include_usage (bool): Append a usage chunk.

    Returns:
        List[Any]: ChatCompletionChunks.
    """
    from openai.types.chat import ChatCompletionChunk

    base = {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    words = content.split(" ")
    deltas = [{"role": "assistant", "content": ""}]
    deltas += [
        {"content": word if i == len(words) - 1 else word + " "}
        for i, word in enumerate(words)
    ]
    chunks = [
        ChatCompletionChunk.model_validate(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": d, "finish_reason": None}
                ],
            }
        )
        for d in deltas
    ]
    chunks.append(
        ChatCompletionChunk.model_validate(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
        )
    )
    if include_usage:
        completion = _estimate_tokens(content)
        chunks.append(
            ChatCompletionChunk.model_validate(
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": completion,
                        "total_tokens": completion,
                    },
                }
            )
        )
    return chunks


def _mock_completion(
    model: str,
    messages: Any,
    mock_response: Any,
    stream: bool,
    include_usage: bool,
    is_async: bool,
) -> Any:
    """Return a canned response without calling any provider.

    Args:
        model (str): Model name to report.
        messages (Any): The request messages, used for the token estimate.
        mock_response (Any): The reply text, or an exception to raise.
        stream (bool): Return a stream instead of a completion.
        include_usage (bool): Append a usage chunk when streaming.
        is_async (bool): Return an async stream.

    Returns:
        Any: A ChatCompletion, ChatStream or AsyncChatStream.
    """
    if isinstance(mock_response, BaseException):
        raise mock_response
    content = str(mock_response)
    if stream:
        from routehub.streaming import AsyncChatStream, ChatStream

        chunks = _mock_chunks(model, content, include_usage)
        if is_async:

            async def agen():
                for chunk in chunks:
                    yield chunk

            return AsyncChatStream(agen(), None, "mock", model)
        return ChatStream(iter(chunks), None, "mock", model)

    from openai.types.chat import ChatCompletion

    from routehub.tokenizer import _content_text

    prompt = sum(
        _estimate_tokens(_content_text(m.get("content")))
        for m in messages or []
        if isinstance(m, dict)
    )
    completion = _estimate_tokens(content)
    return ChatCompletion.model_validate(
        {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            },
        }
    )


def completion(
    model: str,
    messages: Optional[List[dict]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop: Optional[Union[str, List[str]]] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    modalities: Optional[List[str]] = None,
    prediction: Optional[dict] = None,
    audio: Optional[dict] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    reasoning_effort: Optional[ReasoningEffort] = None,
    verbosity: Optional[Literal["low", "medium", "high"]] = None,
    response_format: Optional[Union[dict, type]] = None,
    seed: Optional[int] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    web_search_options: Optional[dict] = None,
    functions: Optional[List[dict]] = None,
    function_call: Optional[Union[str, dict]] = None,
    thinking: Optional[dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    extra_body: Optional[dict] = None,
    drop_params: bool = False,
    num_retries: Optional[int] = None,
    ssl_verify: Union[bool, str] = True,
    keepalive_expiry: float = 60.0,
    set_verbose: bool = False,
    request_timeout: float = 600.0,
    timeout: Optional[Union[float, Any]] = None,
    mock_response: Optional[Union[str, BaseException]] = None,
    client: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Call any supported chat model with OpenAI-style arguments.

    Args:
        model (str): Provider-prefixed or bare model name, such as
            "gpt-5.4", "claude-sonnet-4-6" or "groq/llama-3.3-70b-versatile".
        messages (Optional[List[dict]]): OpenAI-format chat messages.
        temperature (Optional[float]): Sampling temperature.
        top_p (Optional[float]): Nucleus sampling cutoff.
        n (Optional[int]): Number of choices to generate.
        stream (Optional[bool]): Stream the reply as chunks.
        stream_options (Optional[dict]): Stream settings, such as include_usage.
        stop (Optional[Union[str, List[str]]]): Stop sequences.
        max_completion_tokens (Optional[int]): Output token cap.
        max_tokens (Optional[int]): Output token cap, older name.
        modalities (Optional[List[str]]): Output modalities.
        prediction (Optional[dict]): Predicted output.
        audio (Optional[dict]): Audio output settings.
        presence_penalty (Optional[float]): Presence penalty.
        frequency_penalty (Optional[float]): Frequency penalty.
        logit_bias (Optional[dict]): Token bias map.
        user (Optional[str]): End-user identifier.
        reasoning_effort (Optional[ReasoningEffort]): Reasoning depth; mapped
            to Claude thinking: adaptive effort on Claude 5 and Opus 4.7
            and later, a token budget on older models.
        verbosity (Optional[str]): Output verbosity for models that support it.
        response_format (Optional[Union[dict, type]]): Output format, or a
            pydantic model class for JSON-schema output.
        seed (Optional[int]): Sampling seed.
        tools (Optional[List[dict]]): Tool definitions.
        tool_choice (Optional[Union[str, dict]]): Tool selection.
        logprobs (Optional[bool]): Return log probabilities.
        top_logprobs (Optional[int]): Number of top log probabilities.
        parallel_tool_calls (Optional[bool]): Allow several tool calls per turn.
        web_search_options (Optional[dict]): Built-in web search settings.
        functions (Optional[List[dict]]): Legacy function definitions.
        function_call (Optional[Union[str, dict]]): Legacy function selection.
        thinking (Optional[dict]): Anthropic thinking configuration.
        api_key (Optional[str]): API key; read from the provider's env var
            when omitted.
        api_base (Optional[str]): Base URL override.
        base_url (Optional[str]): Base URL override, alternative name.
        api_version (Optional[str]): Azure API version.
        custom_llm_provider (Optional[str]): Provider override.
        extra_headers (Optional[dict]): Additional HTTP headers.
        extra_body (Optional[dict]): Additional request body fields, always sent.
        drop_params (bool): Drop parameters the model cannot accept instead
            of sending them.
        num_retries (Optional[int]): Retries on rate limits, server errors and
            connection failures; 2 when omitted.
        ssl_verify (Union[bool, str]): TLS verification, or a CA bundle path.
        keepalive_expiry (float): Seconds an idle pooled connection stays open.
        set_verbose (bool): Print request and timing details to stderr.
        request_timeout (float): Request timeout in seconds.
        timeout (Optional[Union[float, Any]]): Request timeout; overrides
            request_timeout when set.
        mock_response (Optional[Union[str, BaseException]]): Return this text,
            or raise this exception, without calling a provider.
        client (Optional[Any]): A preconfigured OpenAI or httpx client to use.
        **kwargs (Any): Other provider parameters, sent in the request body
            unless drop_params is on.

    Returns:
        Any: An OpenAI ChatCompletion, or a ChatStream of ChatCompletionChunks
        when stream is on.
    """
    args = dict(locals())
    model, messages, params, options, extra = _split_args(args)
    if options["mock_response"] is not None:
        return _mock_completion(
            model,
            messages,
            options["mock_response"],
            bool(params.get("stream")),
            bool(
                (params.get("stream_options") or {}).get(
                    "include_usage"
                )
            ),
            is_async=False,
        )
    call = _build_call(model, messages, params, options, extra)
    _log(
        call.verbose,
        f"{call.provider} {call.bare} at {call.api_base or 'default base'} "
        f"params={sorted(call.params)} extras={sorted(call.extras)}",
    )
    started = time.perf_counter()
    if PROVIDERS[call.provider].native:
        response = _anthropic_complete(call)
    else:
        response = _openai_complete(call)
    _log(
        call.verbose,
        f"{call.model} answered in {time.perf_counter() - started:.2f}s",
    )
    return response


async def acompletion(
    model: str,
    messages: Optional[List[dict]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop: Optional[Union[str, List[str]]] = None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    modalities: Optional[List[str]] = None,
    prediction: Optional[dict] = None,
    audio: Optional[dict] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    reasoning_effort: Optional[ReasoningEffort] = None,
    verbosity: Optional[Literal["low", "medium", "high"]] = None,
    response_format: Optional[Union[dict, type]] = None,
    seed: Optional[int] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    web_search_options: Optional[dict] = None,
    functions: Optional[List[dict]] = None,
    function_call: Optional[Union[str, dict]] = None,
    thinking: Optional[dict] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    extra_body: Optional[dict] = None,
    drop_params: bool = False,
    num_retries: Optional[int] = None,
    ssl_verify: Union[bool, str] = True,
    keepalive_expiry: float = 60.0,
    set_verbose: bool = False,
    request_timeout: float = 600.0,
    timeout: Optional[Union[float, Any]] = None,
    mock_response: Optional[Union[str, BaseException]] = None,
    client: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Async form of completion, with the same arguments.

    Args:
        model (str): Provider-prefixed or bare model name.
        messages (Optional[List[dict]]): OpenAI-format chat messages.
        temperature (Optional[float]): Sampling temperature.
        top_p (Optional[float]): Nucleus sampling cutoff.
        n (Optional[int]): Number of choices to generate.
        stream (Optional[bool]): Stream the reply as chunks.
        stream_options (Optional[dict]): Stream settings, such as include_usage.
        stop (Optional[Union[str, List[str]]]): Stop sequences.
        max_completion_tokens (Optional[int]): Output token cap.
        max_tokens (Optional[int]): Output token cap, older name.
        modalities (Optional[List[str]]): Output modalities.
        prediction (Optional[dict]): Predicted output.
        audio (Optional[dict]): Audio output settings.
        presence_penalty (Optional[float]): Presence penalty.
        frequency_penalty (Optional[float]): Frequency penalty.
        logit_bias (Optional[dict]): Token bias map.
        user (Optional[str]): End-user identifier.
        reasoning_effort (Optional[ReasoningEffort]): Reasoning depth.
        verbosity (Optional[str]): Output verbosity.
        response_format (Optional[Union[dict, type]]): Output format.
        seed (Optional[int]): Sampling seed.
        tools (Optional[List[dict]]): Tool definitions.
        tool_choice (Optional[Union[str, dict]]): Tool selection.
        logprobs (Optional[bool]): Return log probabilities.
        top_logprobs (Optional[int]): Number of top log probabilities.
        parallel_tool_calls (Optional[bool]): Allow several tool calls per turn.
        web_search_options (Optional[dict]): Built-in web search settings.
        functions (Optional[List[dict]]): Legacy function definitions.
        function_call (Optional[Union[str, dict]]): Legacy function selection.
        thinking (Optional[dict]): Anthropic thinking configuration.
        api_key (Optional[str]): API key.
        api_base (Optional[str]): Base URL override.
        base_url (Optional[str]): Base URL override, alternative name.
        api_version (Optional[str]): Azure API version.
        custom_llm_provider (Optional[str]): Provider override.
        extra_headers (Optional[dict]): Additional HTTP headers.
        extra_body (Optional[dict]): Additional request body fields.
        drop_params (bool): Drop parameters the model cannot accept.
        num_retries (Optional[int]): Retries on retryable failures.
        ssl_verify (Union[bool, str]): TLS verification, or a CA bundle path.
        keepalive_expiry (float): Seconds an idle pooled connection stays open.
        set_verbose (bool): Print request and timing details to stderr.
        request_timeout (float): Request timeout in seconds.
        timeout (Optional[Union[float, Any]]): Overrides request_timeout.
        mock_response (Optional[Union[str, BaseException]]): Canned reply.
        client (Optional[Any]): A preconfigured async client to use.
        **kwargs (Any): Other provider parameters.

    Returns:
        Any: An OpenAI ChatCompletion, or an AsyncChatStream when stream is on.
    """
    args = dict(locals())
    model, messages, params, options, extra = _split_args(args)
    if options["mock_response"] is not None:
        return _mock_completion(
            model,
            messages,
            options["mock_response"],
            bool(params.get("stream")),
            bool(
                (params.get("stream_options") or {}).get(
                    "include_usage"
                )
            ),
            is_async=True,
        )
    call = _build_call(model, messages, params, options, extra)
    _log(
        call.verbose,
        f"{call.provider} {call.bare} at {call.api_base or 'default base'} "
        f"params={sorted(call.params)} extras={sorted(call.extras)}",
    )
    started = time.perf_counter()
    if PROVIDERS[call.provider].native:
        response = await _anthropic_acomplete(call)
    else:
        response = await _openai_acomplete(call)
    _log(
        call.verbose,
        f"{call.model} answered in {time.perf_counter() - started:.2f}s",
    )
    return response


def _embedding_request(
    model: str,
    input: Any,
    dimensions: Optional[int],
    encoding_format: Optional[str],
    user: Optional[str],
    options: dict,
    is_async: bool,
) -> Tuple[Any, dict, str]:
    """Resolve the client and arguments for an embeddings call.

    Args:
        model (str): Model string.
        input (Any): Text or list of texts.
        dimensions (Optional[int]): Output dimensions.
        encoding_format (Optional[str]): "float" or "base64".
        user (Optional[str]): End-user identifier.
        options (dict): Gateway options.
        is_async (bool): Whether to use the async client.

    Returns:
        Tuple[Any, dict, str]: The embeddings resource, create() arguments,
        and the provider.
    """
    from routehub.exceptions import (
        AuthenticationError,
        BadRequestError,
    )

    bare, provider, api_key, api_base = get_llm_provider(
        model,
        options["custom_llm_provider"],
        options["api_base"] or options["base_url"],
        options["api_key"],
    )
    if PROVIDERS[provider].native:
        raise BadRequestError(
            f"{provider} has no embeddings API.",
            llm_provider=provider,
            model=model,
        )
    if not api_key and options["client"] is None:
        names = " or ".join(PROVIDERS[provider].key_env) or "api_key"
        raise AuthenticationError(
            f"No API key for provider {provider!r}. Set {names}, or pass api_key.",
            llm_provider=provider,
            model=model,
        )
    retries = options["num_retries"]
    client = options["client"] or openai_client(
        provider,
        api_key,
        api_base,
        is_async=is_async,
        max_retries=(
            DEFAULT_NUM_RETRIES if retries is None else retries
        ),
        ssl_verify=options["ssl_verify"],
        keepalive_expiry=options["keepalive_expiry"],
        api_version=options["api_version"],
    )
    request = {"model": bare, "input": input}
    if dimensions is not None:
        request["dimensions"] = dimensions
    if encoding_format is not None:
        request["encoding_format"] = encoding_format
    if user is not None:
        request["user"] = user
    timeout = options["timeout"]
    request["timeout"] = (
        options["request_timeout"] if timeout is None else timeout
    )
    request["extra_headers"] = options["extra_headers"]
    request["extra_body"] = options["extra_body"]
    return client.embeddings, request, provider


def embedding(
    model: str,
    input: Union[str, List[str]],
    dimensions: Optional[int] = None,
    encoding_format: Optional[str] = None,
    user: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    extra_body: Optional[dict] = None,
    num_retries: Optional[int] = None,
    ssl_verify: Union[bool, str] = True,
    keepalive_expiry: float = 60.0,
    request_timeout: float = 600.0,
    timeout: Optional[Union[float, Any]] = None,
    client: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Embed text with any OpenAI-compatible embeddings model.

    Args:
        model (str): Provider-prefixed or bare embedding model name.
        input (Union[str, List[str]]): Text or texts to embed.
        dimensions (Optional[int]): Output dimensions, where supported.
        encoding_format (Optional[str]): "float" or "base64".
        user (Optional[str]): End-user identifier.
        api_key (Optional[str]): API key.
        api_base (Optional[str]): Base URL override.
        base_url (Optional[str]): Base URL override, alternative name.
        api_version (Optional[str]): Azure API version.
        custom_llm_provider (Optional[str]): Provider override.
        extra_headers (Optional[dict]): Additional HTTP headers.
        extra_body (Optional[dict]): Additional request body fields.
        num_retries (Optional[int]): Retries on retryable failures.
        ssl_verify (Union[bool, str]): TLS verification, or a CA bundle path.
        keepalive_expiry (float): Seconds an idle pooled connection stays open.
        request_timeout (float): Request timeout in seconds.
        timeout (Optional[Union[float, Any]]): Overrides request_timeout.
        client (Optional[Any]): A preconfigured OpenAI client to use.
        **kwargs (Any): Ignored litellm options.

    Returns:
        Any: An OpenAI CreateEmbeddingResponse.
    """
    options = dict(locals())
    from routehub.exceptions import map_exception

    resource, request, provider = _embedding_request(
        model,
        input,
        dimensions,
        encoding_format,
        user,
        options,
        False,
    )
    try:
        return resource.create(**request)
    except Exception as error:
        mapped = map_exception(error, provider, model)
        if mapped is error:
            raise
        raise mapped from error


async def aembedding(
    model: str,
    input: Union[str, List[str]],
    dimensions: Optional[int] = None,
    encoding_format: Optional[str] = None,
    user: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    extra_body: Optional[dict] = None,
    num_retries: Optional[int] = None,
    ssl_verify: Union[bool, str] = True,
    keepalive_expiry: float = 60.0,
    request_timeout: float = 600.0,
    timeout: Optional[Union[float, Any]] = None,
    client: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Async form of embedding, with the same arguments.

    Args:
        model (str): Provider-prefixed or bare embedding model name.
        input (Union[str, List[str]]): Text or texts to embed.
        dimensions (Optional[int]): Output dimensions.
        encoding_format (Optional[str]): "float" or "base64".
        user (Optional[str]): End-user identifier.
        api_key (Optional[str]): API key.
        api_base (Optional[str]): Base URL override.
        base_url (Optional[str]): Base URL override, alternative name.
        api_version (Optional[str]): Azure API version.
        custom_llm_provider (Optional[str]): Provider override.
        extra_headers (Optional[dict]): Additional HTTP headers.
        extra_body (Optional[dict]): Additional request body fields.
        num_retries (Optional[int]): Retries on retryable failures.
        ssl_verify (Union[bool, str]): TLS verification, or a CA bundle path.
        keepalive_expiry (float): Seconds an idle pooled connection stays open.
        request_timeout (float): Request timeout in seconds.
        timeout (Optional[Union[float, Any]]): Overrides request_timeout.
        client (Optional[Any]): A preconfigured async OpenAI client to use.
        **kwargs (Any): Ignored litellm options.

    Returns:
        Any: An OpenAI CreateEmbeddingResponse.
    """
    options = dict(locals())
    from routehub.exceptions import map_exception

    resource, request, provider = _embedding_request(
        model, input, dimensions, encoding_format, user, options, True
    )
    try:
        return await resource.create(**request)
    except Exception as error:
        mapped = map_exception(error, provider, model)
        if mapped is error:
            raise
        raise mapped from error


def _responses_request(
    model: str, kwargs: dict, is_async: bool
) -> Tuple[_Call, Any, dict]:
    """Resolve the client and create() arguments for a Responses call.

    Args:
        model (str): Model string.
        kwargs (dict): Gateway options and Responses API parameters.
        is_async (bool): Whether to use the async client.

    Returns:
        Tuple[_Call, Any, dict]: Call, responses resource, create() args.
    """
    spec = inspect.signature(completion).parameters
    options = {n: kwargs.pop(n, spec[n].default) for n in _OPTIONS}
    if options["mock_response"] is not None:
        raise NotImplementedError("mock_response is not supported")
    input = kwargs.get("input")
    call = _build_call(model, [input], {}, options, kwargs)
    resource = _openai_client(call, is_async).responses
    request = {**call.extras, "model": call.bare}
    if kwargs.get("metadata") is not None:
        request["metadata"] = kwargs["metadata"]
    if call.drop_params:
        accepted = _sdk_params(resource)
        request = {k: v for k, v in request.items() if k in accepted}
    _drop_rejected(call, request, "reasoning")
    _log(call.verbose, f"responses {call.provider} {call.bare}")
    extra_body = dict(call.extra_body or {})
    request = _create_kwargs(resource, request, extra_body, call)
    return call, resource, request


def responses(model: str, **kwargs: Any) -> Any:
    """Call a model through the OpenAI Responses API.

    Args:
        model (str): Provider-prefixed or bare model name.
        **kwargs (Any): completion's options and Responses API parameters.

    Returns:
        Any: An OpenAI Response, or its event stream when stream is on.
    """
    from routehub.exceptions import map_exception

    call, resource, request = _responses_request(model, kwargs, False)
    try:
        return resource.create(**request)
    except Exception as error:
        mapped = map_exception(error, call.provider, call.model)
        if mapped is error:
            raise
        raise mapped from error


async def aresponses(model: str, **kwargs: Any) -> Any:
    """Async form of responses, with the same arguments.

    Args:
        model (str): Provider-prefixed or bare model name.
        **kwargs (Any): Gateway options and Responses API parameters.

    Returns:
        Any: An OpenAI Response, or its async event stream.
    """
    from routehub.exceptions import map_exception

    call, resource, request = _responses_request(model, kwargs, True)
    try:
        return await resource.create(**request)
    except Exception as error:
        mapped = map_exception(error, call.provider, call.model)
        if mapped is error:
            raise
        raise mapped from error
