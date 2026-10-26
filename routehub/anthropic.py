"""Native adapter for the Anthropic Messages API, speaking the OpenAI chat format.

Anthropic's OpenAI-compatible endpoint drops prompt caching, thinking output and
cached-token usage, so Claude is called through its own API and the results are
returned as OpenAI SDK objects.
"""

import re
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from routehub._json import dumps_str, loads

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
JSON_TOOL_NAME = "json_tool_call"

REASONING_BUDGETS = {
    "minimal": 1024,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 8192,
    "max": 16000,
    "ultra": 16000,
}

ADAPTIVE_EFFORTS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
    "ultra": "max",
}

WEB_SEARCH_MAX_USES = {"low": 1, "medium": 5, "high": 10}

_CLAUDE_VERSION = re.compile(
    r"claude-(?:[a-z]+-)?(\d+)(?:-(\d{1,2})(?!\d))?"
)

STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

# Request fields Anthropic accepts that are not OpenAI parameters.
NATIVE_EXTRAS = frozenset(
    {
        "top_k",
        "container",
        "mcp_servers",
        "service_tier",
        "output_config",
        "context_management",
        "inference_geo",
    }
)

# Claude models released before extended thinking existed.
_NO_THINKING_PREFIXES = (
    "claude-2",
    "claude-instant",
    "claude-3-haiku",
    "claude-3-sonnet",
    "claude-3-opus",
    "claude-3-5",
)

_NATIVE_BLOCKS = frozenset(
    {
        "image",
        "document",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "search_result",
    }
)


def _supports_thinking(model: str) -> bool:
    """Whether a Claude model accepts the thinking parameter.

    Args:
        model (str): Bare Claude model name.

    Returns:
        bool: False for models that predate extended thinking.
    """
    return not model.lower().startswith(_NO_THINKING_PREFIXES)


def _adaptive_thinking_only(model: str) -> bool:
    """Whether a Claude model rejects fixed thinking budgets.

    Claude 5 models and Claude Opus 4.7 and later accept only adaptive
    thinking, with depth set by ``output_config.effort``.

    Args:
        model (str): Bare Claude model name.

    Returns:
        bool: True when ``budget_tokens`` would be rejected.
    """
    match = _CLAUDE_VERSION.search(model.lower())
    if not match:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= (4, 7)


def _rejects_forced_tool_choice(model: str) -> bool:
    """Whether a Claude model rejects a forced tool_choice.

    Claude Opus 5.5, Sonnet 5.5, Fable 5.1 and later return a 400 for a
    ``tool_choice`` of type ``any`` or ``tool``.

    Args:
        model (str): Bare Claude model name.

    Returns:
        bool: True when only ``auto`` and ``none`` are accepted.
    """
    match = _CLAUDE_VERSION.search(model.lower())
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2) or 0)) >= (5, 1)


def _parse_data_uri(url: str) -> Optional[Tuple[str, str]]:
    """Split a data URI into media type and base64 payload.

    Args:
        url (str): A URL that may be a data URI.

    Returns:
        Optional[Tuple[str, str]]: Media type and payload, or None for other URLs.
    """
    if not url.startswith("data:"):
        return None
    header, _, data = url.partition(",")
    media_type = header[5:].split(";")[0] or "image/jpeg"
    return media_type, data


def _image_block(part: dict) -> Optional[dict]:
    """Translate an OpenAI image_url part into an Anthropic image block.

    Args:
        part (dict): The OpenAI content part.

    Returns:
        Optional[dict]: The Anthropic block, or None when there is no URL.
    """
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if not url:
        return None
    parsed = _parse_data_uri(url)
    if parsed:
        source = {
            "type": "base64",
            "media_type": parsed[0],
            "data": parsed[1],
        }
    else:
        source = {"type": "url", "url": url}
    return {"type": "image", "source": source}


def _file_block(part: dict) -> Optional[dict]:
    """Translate an OpenAI file part into an Anthropic document or image block.

    Args:
        part (dict): The OpenAI content part.

    Returns:
        Optional[dict]: The Anthropic block, or None when it carries no data.
    """
    file = part.get("file") or {}
    value = file.get("file_data") or file.get("file_id")
    if isinstance(value, str) and value.startswith(
        ("http://", "https://")
    ):
        return {
            "type": "document",
            "source": {"type": "url", "url": value},
        }
    data = file.get("file_data")
    if data:
        media_type, payload = _parse_data_uri(data) or (
            "application/pdf",
            data,
        )
        kind = (
            "image" if media_type.startswith("image/") else "document"
        )
        return {
            "type": kind,
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": payload,
            },
        }
    if file.get("file_id"):
        return {
            "type": "document",
            "source": {"type": "file", "file_id": file["file_id"]},
        }
    return None


def _content_blocks(content: Any) -> List[dict]:
    """Translate OpenAI message content into Anthropic content blocks.

    Args:
        content (Any): A string, a list of OpenAI content parts, or None.

    Returns:
        List[dict]: Anthropic content blocks, with cache_control markers kept.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return (
            [{"type": "text", "text": content}]
            if content.strip()
            else []
        )
    blocks = []
    for part in content:
        if isinstance(part, str):
            if part.strip():
                blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            if not (part.get("text") or "").strip():
                continue
            block = {"type": "text", "text": part["text"]}
        elif kind == "image_url":
            block = _image_block(part)
        elif kind == "file":
            block = _file_block(part)
        elif kind in _NATIVE_BLOCKS:
            blocks.append(dict(part))
            continue
        else:
            continue
        if block is None:
            continue
        if part.get("cache_control"):
            block["cache_control"] = part["cache_control"]
        blocks.append(block)
    return blocks


def _tool_use_input(arguments: Any) -> dict:
    """Decode tool-call arguments into the dict Anthropic expects.

    Args:
        arguments (Any): A JSON string or an already-decoded dict.

    Returns:
        dict: The decoded arguments, empty when they are not valid JSON.
    """
    if isinstance(arguments, dict):
        return arguments
    try:
        decoded = loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _translate_messages(
    messages: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """Split OpenAI messages into an Anthropic system prompt and turns.

    Args:
        messages (List[dict]): OpenAI-format chat messages.

    Returns:
        Tuple[List[dict], List[dict]]: System blocks and Anthropic messages.
    """
    system: List[dict] = []
    turns: List[dict] = []
    for message in messages:
        role = message.get("role")
        blocks: List[dict] = []
        if role in ("system", "developer"):
            blocks = _content_blocks(message.get("content"))
            system.extend(blocks)
        elif role == "user":
            blocks = _content_blocks(message.get("content"))
            if blocks:
                turns.append({"role": "user", "content": blocks})
        elif role == "assistant":
            blocks = [
                dict(b) for b in message.get("thinking_blocks") or []
            ]
            blocks.extend(_content_blocks(message.get("content")))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": function.get("name"),
                        "input": _tool_use_input(
                            function.get("arguments")
                        ),
                    }
                )
            if blocks:
                turns.append({"role": "assistant", "content": blocks})
        elif role in ("tool", "function"):
            result = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id")
                or message.get("name"),
            }
            content = _content_blocks(message.get("content"))
            if content:
                result["content"] = content
            if message.get("is_error"):
                result["is_error"] = True
            blocks = [result]
            previous = turns[-1] if turns else None
            # Results of one parallel tool turn must share a user message.
            if (
                previous
                and previous["role"] == "user"
                and all(
                    b.get("type") == "tool_result"
                    for b in previous["content"]
                )
            ):
                previous["content"].append(result)
            else:
                turns.append({"role": "user", "content": [result]})
        marker = message.get("cache_control")
        if marker and blocks:
            blocks[-1].setdefault("cache_control", marker)
    return system, turns


def _translate_tools(tools: List[dict]) -> List[dict]:
    """Translate OpenAI tool definitions into Anthropic tools.

    Args:
        tools (List[dict]): OpenAI function tools, or native Anthropic tools.

    Returns:
        List[dict]: Anthropic tool definitions, with cache_control kept.
    """
    translated = []
    for tool in tools:
        function = tool.get("function")
        if function is None:
            translated.append(dict(tool))
            continue
        native = {
            "name": function["name"],
            "input_schema": function.get("parameters")
            or {"type": "object", "properties": {}},
        }
        if function.get("description"):
            native["description"] = function["description"]
        marker = tool.get("cache_control") or function.get(
            "cache_control"
        )
        if marker:
            native["cache_control"] = marker
        translated.append(native)
    return translated


def _translate_tool_choice(
    choice: Any, parallel_tool_calls: Optional[bool]
) -> Optional[dict]:
    """Translate an OpenAI tool_choice into Anthropic's form.

    Args:
        choice (Any): "auto", "required", "none", or a named function dict.
        parallel_tool_calls (Optional[bool]): False disables parallel calls.

    Returns:
        Optional[dict]: The Anthropic tool_choice, or None to use the default.
    """
    if choice == "auto":
        result = {"type": "auto"}
    elif choice == "required":
        result = {"type": "any"}
    elif choice == "none":
        return {"type": "none"}
    elif isinstance(choice, dict):
        name = (choice.get("function") or {}).get(
            "name"
        ) or choice.get("name")
        result = (
            {"type": "tool", "name": name} if name else dict(choice)
        )
    else:
        result = None
    if parallel_tool_calls is False:
        result = result or {"type": "auto"}
        result["disable_parallel_tool_use"] = True
    return result


def _json_schema(response_format: Any) -> Optional[dict]:
    """Extract the JSON schema from an OpenAI json_schema response_format.

    Args:
        response_format (Any): The response_format parameter.

    Returns:
        Optional[dict]: The schema, or None when no schema was requested.
    """
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    spec = response_format.get("json_schema") or {}
    return spec.get("schema") or {"type": "object"}


def build_request(
    model: str,
    messages: List[dict],
    params: Dict[str, Any],
    extras: Dict[str, Any],
    drop_params: bool,
) -> Tuple[dict, bool]:
    """Build an Anthropic Messages API body from OpenAI-style arguments.

    Args:
        model (str): Bare Claude model name.
        messages (List[dict]): OpenAI-format chat messages.
        params (Dict[str, Any]): OpenAI parameters that were set.
        extras (Dict[str, Any]): Non-OpenAI keyword arguments.
        drop_params (bool): Drop parameters the model cannot accept instead
            of sending them.

    Returns:
        Tuple[dict, bool]: The request body, and whether a JSON schema is
        being enforced through a forced tool call.
    """
    system, turns = _translate_messages(messages)
    max_tokens = (
        params.get("max_tokens")
        or params.get("max_completion_tokens")
        or DEFAULT_MAX_TOKENS
    )
    body: Dict[str, Any] = {"model": model, "messages": turns}
    if system:
        body["system"] = system

    thinking = extras.get("thinking")
    effort = params.get("reasoning_effort")
    if effort in ("none", "None"):
        effort = None
    if thinking is None and effort:
        if _adaptive_thinking_only(model):
            thinking = {"type": "adaptive"}
        else:
            thinking = {
                "type": "enabled",
                "budget_tokens": REASONING_BUDGETS.get(effort, 4096),
            }
    if thinking and drop_params and not _supports_thinking(model):
        thinking = None
    if (
        drop_params
        and thinking
        and thinking.get("type") == "enabled"
        and _adaptive_thinking_only(model)
    ):
        thinking = {**thinking, "type": "adaptive"}
        thinking.pop("budget_tokens", None)
    if thinking:
        body["thinking"] = thinking
        if effort and thinking.get("type") == "adaptive":
            body["output_config"] = {
                "effort": ADAPTIVE_EFFORTS.get(effort, "high"),
                **(extras.get("output_config") or {}),
            }
        budget = thinking.get("budget_tokens") or 0
        # Anthropic rejects a budget that is not below max_tokens.
        if budget and max_tokens <= budget:
            max_tokens = budget + 1024
    body["max_tokens"] = max_tokens

    adaptive_only = _adaptive_thinking_only(model)
    thinking_type = (thinking or {}).get("type")
    if adaptive_only or thinking_type in ("enabled", "adaptive"):
        sampling = [
            key
            for key, value, floor in (
                ("temperature", params.get("temperature"), 1),
                ("top_p", params.get("top_p"), 0.95),
                ("top_k", extras.get("top_k"), None),
            )
            if value is not None
            and (adaptive_only or floor is None or value < floor)
        ]
        if sampling and not drop_params:
            from routehub.exceptions import UnsupportedParamsError

            raise UnsupportedParamsError(
                f"{model} does not accept {', '.join(sampling)}"
                f"{'' if adaptive_only else ' with thinking on'}. "
                "Remove them or pass drop_params=True.",
                llm_provider="anthropic",
                model=model,
            )
        params = {
            k: v for k, v in params.items() if k not in sampling
        }
        extras = {
            k: v for k, v in extras.items() if k not in sampling
        }

    temperature = params.get("temperature")
    if temperature is not None:
        body["temperature"] = min(float(temperature), 1.0)
    if params.get("top_p") is not None:
        if not (drop_params and "temperature" in body):
            body["top_p"] = params["top_p"]
    stop = params.get("stop")
    if stop:
        body["stop_sequences"] = (
            [stop] if isinstance(stop, str) else stop
        )
    if params.get("user"):
        body["metadata"] = {"user_id": params["user"]}
    if params.get("stream"):
        body["stream"] = True

    tools = params.get("tools")
    if not tools and params.get("functions"):
        tools = [
            {"type": "function", "function": f}
            for f in params["functions"]
        ]
    if tools:
        body["tools"] = _translate_tools(tools)
    choice = _translate_tool_choice(
        params.get("tool_choice"), params.get("parallel_tool_calls")
    )
    if choice and tools:
        always = _rejects_forced_tool_choice(model)
        if choice.get("type") in ("any", "tool") and (
            always or thinking_type == "enabled"
        ):
            if not drop_params:
                from routehub.exceptions import UnsupportedParamsError

                raise UnsupportedParamsError(
                    f"{model} does not accept a forced tool_choice "
                    f"({params.get('tool_choice')!r})"
                    f"{'' if always else ' with thinking on'}. Use "
                    "tool_choice='auto', or set drop_params=True to "
                    "send 'auto' instead.",
                    llm_provider="anthropic",
                    model=model,
                )
            choice = {
                "type": "auto",
                **{
                    k: v
                    for k, v in choice.items()
                    if k == "disable_parallel_tool_use"
                },
            }
        body["tool_choice"] = choice

    search = params.get("web_search_options")
    if search is not None:
        tool = {"type": "web_search_20250305", "name": "web_search"}
        location = (search.get("user_location") or {}).get(
            "approximate"
        )
        if location:
            tool["user_location"] = {
                **location,
                "type": "approximate",
            }
        uses = WEB_SEARCH_MAX_USES.get(
            search.get("search_context_size")
        )
        if uses:
            tool["max_uses"] = uses
        body.setdefault("tools", []).append(tool)

    schema = _json_schema(params.get("response_format"))
    json_mode = schema is not None
    if json_mode and _rejects_forced_tool_choice(model):
        body["output_config"] = {
            "format": {"type": "json_schema", "schema": schema},
            **(
                body.get("output_config")
                or extras.get("output_config")
                or {}
            ),
        }
        json_mode = False
    if json_mode:
        body.setdefault("tools", []).append(
            {
                "name": JSON_TOOL_NAME,
                "description": "Respond with a JSON object matching this schema.",
                "input_schema": schema,
            }
        )
        # Thinking cannot be combined with a forced tool call.
        body["tool_choice"] = (
            {"type": "auto"}
            if thinking
            else {"type": "tool", "name": JSON_TOOL_NAME}
        )

    for key, value in extras.items():
        if key == "thinking" or (
            key == "output_config" and key in body
        ):
            continue
        if key in NATIVE_EXTRAS or not drop_params:
            body[key] = value
    return body, json_mode


def messages_url(api_base: str) -> str:
    """Return the Messages endpoint for a base URL.

    Args:
        api_base (str): The Anthropic base URL, with or without /v1.

    Returns:
        str: The full /v1/messages URL.
    """
    base = api_base.rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def headers(api_key: str, extra_headers: Optional[dict]) -> dict:
    """Build the request headers.

    Args:
        api_key (str): Anthropic API key.
        extra_headers (Optional[dict]): Additional headers, such as anthropic-beta.

    Returns:
        dict: The headers to send.
    """
    result = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    if extra_headers:
        result.update(extra_headers)
    return result


def usage(raw: Optional[dict]) -> dict:
    """Convert Anthropic usage into the OpenAI usage shape.

    Args:
        raw (Optional[dict]): Anthropic usage block.

    Returns:
        dict: OpenAI-shaped usage, with prompt tokens including cache reads
        and writes, and the Anthropic cache counts kept alongside.
    """
    raw = raw or {}
    cache_read = raw.get("cache_read_input_tokens") or 0
    cache_write = raw.get("cache_creation_input_tokens") or 0
    prompt = (raw.get("input_tokens") or 0) + cache_read + cache_write
    completion = raw.get("output_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {
            "cached_tokens": cache_read,
            "cache_creation_tokens": cache_write,
        },
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }


def to_chat_completion(
    data: dict, model: str, json_mode: bool
) -> Any:
    """Convert an Anthropic message into an OpenAI ChatCompletion.

    Args:
        data (dict): The decoded Messages API response.
        model (str): Model name to report when the response omits it.
        json_mode (bool): Whether the JSON tool carries the answer.

    Returns:
        ChatCompletion: The OpenAI SDK response object.
    """
    from openai.types.chat import ChatCompletion

    text, tool_calls, thinking_blocks, reasoning = [], [], [], []
    for block in data.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            text.append(block.get("text", ""))
        elif kind == "tool_use":
            if json_mode and block.get("name") == JSON_TOOL_NAME:
                text.append(dumps_str(block.get("input") or {}))
                continue
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": dumps_str(
                            block.get("input") or {}
                        ),
                    },
                }
            )
        elif kind == "thinking":
            thinking_blocks.append(
                {
                    "type": "thinking",
                    "thinking": block.get("thinking", ""),
                    "signature": block.get("signature"),
                }
            )
            reasoning.append(block.get("thinking", ""))
        elif kind == "redacted_thinking":
            thinking_blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": block.get("data"),
                }
            )

    finish_reason = STOP_REASONS.get(data.get("stop_reason"), "stop")
    if finish_reason == "tool_calls" and not tool_calls:
        finish_reason = "stop"
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if thinking_blocks:
        message["thinking_blocks"] = thinking_blocks
        message["reasoning_content"] = "".join(reasoning)
    return ChatCompletion.model_validate(
        {
            "id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model") or model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message,
                    "logprobs": None,
                }
            ],
            "usage": usage(data.get("usage")),
        }
    )


def iter_sse(
    lines: Iterator[str],
) -> Iterator[Tuple[Optional[str], str]]:
    """Group server-sent-event lines into (event, data) pairs.

    Args:
        lines (Iterator[str]): Decoded lines of the event stream.

    Returns:
        Iterator[Tuple[Optional[str], str]]: Event name and data payload.
    """
    event: Optional[str] = None
    data: List[str] = []
    for line in lines:
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


async def aiter_sse(lines: Any) -> Any:
    """Group server-sent-event lines into (event, data) pairs, asynchronously.

    Args:
        lines (Any): Async iterator of decoded lines of the event stream.

    Returns:
        Any: Async iterator of event name and data payload.
    """
    event: Optional[str] = None
    data: List[str] = []
    async for line in lines:
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


class StreamTranslator:
    """Turns Anthropic stream events into OpenAI ChatCompletionChunks."""

    def __init__(
        self, model: str, include_usage: bool, json_mode: bool
    ) -> None:
        """Start a translation.

        Args:
            model (str): Model name to report until the stream names it.
            include_usage (bool): Emit a final usage chunk with no choices.
            json_mode (bool): Whether the JSON tool carries the answer.
        """
        from openai.types.chat import ChatCompletionChunk

        self._chunk_cls = ChatCompletionChunk
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.model = model
        self.created = int(time.time())
        self.include_usage = include_usage
        self.json_mode = json_mode
        self.usage: dict = {}
        self._tool_index = -1
        self._block_kind: Dict[int, str] = {}
        self._block_tool: Dict[int, int] = {}
        self._unstreamed_input: Dict[int, Any] = {}

    def _chunk(
        self,
        delta: dict,
        finish_reason: Optional[str] = None,
    ) -> Any:
        """Build one chunk carrying a delta.

        Args:
            delta (dict): The choice delta.
            finish_reason (Optional[str]): Set on the closing chunk.

        Returns:
            ChatCompletionChunk: The chunk.
        """
        return self._chunk_cls.model_validate(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                        "logprobs": None,
                    }
                ],
            }
        )

    def _usage_chunk(self) -> Any:
        """Build the trailing usage chunk.

        Returns:
            ChatCompletionChunk: A chunk with usage and no choices.
        """
        return self._chunk_cls.model_validate(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage(self.usage),
            }
        )

    def handle(self, event: Optional[str], data: dict) -> List[Any]:
        """Translate one stream event.

        Args:
            event (Optional[str]): The SSE event name.
            data (dict): The decoded event payload.

        Returns:
            List[Any]: Zero or more ChatCompletionChunks.
        """
        kind = data.get("type") or event
        if kind == "message_start":
            message = data.get("message") or {}
            self.id = message.get("id") or self.id
            self.model = message.get("model") or self.model
            self.usage = dict(message.get("usage") or {})
            return [self._chunk({"role": "assistant", "content": ""})]
        if kind == "content_block_start":
            return self._block_start(data)
        if kind == "content_block_delta":
            return self._block_delta(data)
        if kind == "message_delta":
            delta = data.get("delta") or {}
            finish = STOP_REASONS.get(
                delta.get("stop_reason"), "stop"
            )
            if finish == "tool_calls" and self._tool_index < 0:
                finish = "stop"
            self.usage.update(
                {
                    k: v
                    for k, v in (data.get("usage") or {}).items()
                    if v is not None
                }
            )
            return [self._chunk({}, finish_reason=finish)]
        if kind == "content_block_stop":
            index = data.get("index", 0)
            if index not in self._unstreamed_input:
                return []
            arguments = dumps_str(self._unstreamed_input.pop(index))
            if self._block_kind.get(index) == "json_tool":
                return [self._chunk({"content": arguments})]
            return [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self._block_tool[index],
                                "function": {"arguments": arguments},
                            }
                        ]
                    }
                )
            ]
        if kind == "message_stop":
            return [self._usage_chunk()] if self.include_usage else []
        if kind == "error":
            raise stream_error(data.get("error") or {}, self.model)
        return []

    def _block_start(self, data: dict) -> List[Any]:
        """Translate a content_block_start event.

        Args:
            data (dict): The event payload.

        Returns:
            List[Any]: Zero or more chunks.
        """
        block = data.get("content_block") or {}
        index = data.get("index", 0)
        kind = block.get("type")
        self._block_kind[index] = kind
        if kind == "tool_use":
            self._unstreamed_input[index] = block.get("input") or {}
            if self.json_mode and block.get("name") == JSON_TOOL_NAME:
                self._block_kind[index] = "json_tool"
                return []
            self._tool_index += 1
            self._block_tool[index] = self._tool_index
            return [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self._tool_index,
                                "id": block.get("id"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": "",
                                },
                            }
                        ]
                    }
                )
            ]
        if kind == "text" and block.get("text"):
            return [self._chunk({"content": block["text"]})]
        if kind == "redacted_thinking":
            return [
                self._chunk(
                    {
                        "thinking_blocks": [
                            {
                                "type": "redacted_thinking",
                                "data": block.get("data"),
                            }
                        ]
                    }
                )
            ]
        return []

    def _block_delta(self, data: dict) -> List[Any]:
        """Translate a content_block_delta event.

        Args:
            data (dict): The event payload.

        Returns:
            List[Any]: Zero or more chunks.
        """
        index = data.get("index", 0)
        delta = data.get("delta") or {}
        kind = delta.get("type")
        if kind == "text_delta":
            return [self._chunk({"content": delta.get("text", "")})]
        if kind == "input_json_delta":
            partial = delta.get("partial_json", "")
            if partial:
                self._unstreamed_input.pop(index, None)
            if self._block_kind.get(index) == "json_tool":
                return (
                    [self._chunk({"content": partial})]
                    if partial
                    else []
                )
            if index not in self._block_tool:
                return []
            return [
                self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self._block_tool[index],
                                "function": {"arguments": partial},
                            }
                        ]
                    }
                )
            ]
        if kind == "thinking_delta":
            text = delta.get("thinking", "")
            return [
                self._chunk(
                    {
                        "reasoning_content": text,
                        "thinking_blocks": [
                            {"type": "thinking", "thinking": text}
                        ],
                    }
                )
            ]
        if kind == "signature_delta":
            return [
                self._chunk(
                    {
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "",
                                "signature": delta.get("signature"),
                            }
                        ]
                    }
                )
            ]
        return []


_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}


def stream_error(error: dict, model: str) -> Exception:
    """Build the exception for an error event inside a stream.

    Args:
        error (dict): The error object from the event.
        model (str): Model that was called.

    Returns:
        Exception: The matching RouteHub exception.
    """
    from routehub.exceptions import exception_for_status

    status = _ERROR_STATUS.get(error.get("type"), 500)
    return exception_for_status(
        status,
        error.get("message") or "Anthropic stream error",
        "anthropic",
        model,
        body=error,
    )


def response_error(response: Any, model: str) -> Exception:
    """Build the exception for a non-2xx Messages API response.

    Args:
        response (Any): The HTTP response, with its body already read.
        model (str): Model that was called.

    Returns:
        Exception: The matching RouteHub exception.
    """
    from routehub.exceptions import exception_for_status

    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    message = (
        error.get("message")
        if isinstance(error, dict)
        else response.text or f"HTTP {response.status_code}"
    )
    return exception_for_status(
        response.status_code,
        message,
        "anthropic",
        model,
        response=response,
        body=body,
    )
