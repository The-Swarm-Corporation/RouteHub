"""AWS Bedrock support through the Converse API.

Requests are signed with SigV4 from the standard AWS environment
variables, or sent with a Bedrock API key as an Authorization: Bearer
header. Responses come back as OpenAI SDK objects, like the native
Anthropic adapter.
"""

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

_SERVICE = "bedrock"
_DUMMY_KEYS = ("", "bedrock")
_INFERENCE_MAP = {
    "max_tokens": "maxTokens",
    "max_completion_tokens": "maxTokens",
    "temperature": "temperature",
    "top_p": "topP",
}
_SUPPORTED_PARAMS = frozenset(
    {
        "stream",
        "aws_region",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
    }
)
_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "content_filtered": "content_filter",
}


def _content_blocks(content: Any) -> List[dict]:
    """Turn one message's OpenAI content into Converse content blocks.

    Args:
        content (Any): A string, a list of OpenAI content parts, or None.

    Returns:
        List[dict]: Converse content blocks.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    blocks = []
    for part in content:
        if part.get("type") != "text":
            raise ValueError(
                "Bedrock supports text content parts only; "
                "send images through the Anthropic or OpenAI path"
            )
        blocks.append({"text": part.get("text", "")})
    return blocks


def _tool_result(message: dict) -> List[dict]:
    """Turn a tool-role message into Converse toolResult blocks.

    Args:
        message (dict): An OpenAI tool message with tool_call_id/content.

    Returns:
        List[dict]: Converse toolResult blocks.
    """
    content = message.get("content")
    if isinstance(content, str):
        body = [{"text": content}]
    elif isinstance(content, list):
        body = _content_blocks(content)
    else:
        body = [{"text": json.dumps(content)}]
    return [
        {
            "toolResult": {
                "toolUseId": message.get("tool_call_id"),
                "content": body,
            }
        }
    ]


def _tool_use_blocks(tool_calls: List[dict]) -> List[dict]:
    """Turn OpenAI tool calls into Converse toolUse blocks.

    Args:
        tool_calls (List[dict]): OpenAI-shaped tool calls.

    Returns:
        List[dict]: Converse toolUse blocks.
    """
    blocks = []
    for call in tool_calls:
        fn = call.get("function") or {}
        blocks.append(
            {
                "toolUse": {
                    "toolUseId": call.get("id"),
                    "name": fn.get("name"),
                    "input": json.loads(fn.get("arguments") or "{}"),
                }
            }
        )
    return blocks


def build_request(
    bare: str,
    messages: List[dict],
    params: Dict[str, Any],
) -> dict:
    """Build a Converse request body from an OpenAI chat request.

    Args:
        bare (str): The bare Bedrock model id.
        messages (List[dict]): OpenAI chat messages.
        params (Dict[str, Any]): OpenAI request parameters.

    Returns:
        dict: The Converse request body.
    """
    body: Dict[str, Any] = {"modelId": bare}
    out: List[dict] = []
    system: List[dict] = []
    for message in messages:
        role = message.get("role", "user")
        if role in ("system", "developer"):
            system.extend(_content_blocks(message.get("content")))
            continue
        if role == "tool":
            role, blocks = "user", _tool_result(message)
        else:
            blocks = _content_blocks(message.get("content"))
            if message.get("tool_calls"):
                blocks.extend(_tool_use_blocks(message["tool_calls"]))
            role = "assistant" if role == "assistant" else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    if system:
        body["system"] = system
    body["messages"] = out
    config = {}
    for name, target in _INFERENCE_MAP.items():
        if params.get(name) is not None:
            config[target] = params[name]
    if params.get("stop"):
        stop = params["stop"]
        config["stopSequences"] = (
            [stop] if isinstance(stop, str) else stop
        )
    if config:
        body["inferenceConfig"] = config
    tools = params.get("tools") or []
    specs = []
    for tool in tools:
        fn = tool.get("function") or tool
        if not fn.get("name"):
            continue
        specs.append(
            {
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn.get("description") or "",
                    "inputSchema": {
                        "json": fn.get("parameters")
                        or {"type": "object", "properties": {}}
                    },
                }
            }
        )
    if specs and params.get("tool_choice") != "none":
        body["toolConfig"] = {"tools": specs}
        choice = params.get("tool_choice")
        if choice == "auto":
            body["toolConfig"]["toolChoice"] = {"auto": {}}
        elif choice == "required":
            body["toolConfig"]["toolChoice"] = {"any": {}}
        elif isinstance(choice, dict) and choice.get(
            "function", {}
        ).get("name"):
            body["toolConfig"]["toolChoice"] = {
                "tool": {"name": choice["function"]["name"]}
            }
    return body


def region_for(extras: Dict[str, Any]) -> Optional[str]:
    """Resolve the Bedrock region for one call.

    Args:
        extras (Dict[str, Any]): Gateway extras, which may carry aws_region.

    Returns:
        Optional[str]: The region, or None.
    """
    import os

    if extras.get("aws_region"):
        return extras["aws_region"]
    return os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION"
    )


def converse_url(
    api_base: Optional[str],
    region: str,
    bare: str,
) -> str:
    """Build the Converse endpoint URL.

    Args:
        api_base (Optional[str]): Explicit base URL override.
        region (str): AWS region for the default endpoint.
        bare (str): The bare Bedrock model id.

    Returns:
        str: The full URL.
    """
    base = (
        api_base or f"https://bedrock-runtime.{region}.amazonaws.com"
    )
    model = quote(bare, safe="")
    return f"{base.rstrip('/')}/model/{model}/converse"


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def sigv4_headers(
    method: str,
    url: str,
    body: bytes,
    region: str,
    access_key: str,
    secret_key: str,
    session_token: Optional[str] = None,
    amz_date: Optional[str] = None,
) -> dict:
    """Sign one request with SigV4 for the Bedrock service.

    Args:
        method (str): HTTP method.
        url (str): The full request URL.
        body (bytes): The encoded request body.
        region (str): AWS region.
        access_key (str): AWS access key id.
        secret_key (str): AWS secret access key.
        session_token (Optional[str]): AWS session token, if any.
        amz_date (Optional[str]): Frozen x-amz-date for tests.

    Returns:
        dict: Headers carrying the signature.
    """
    now = amz_date or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    date = now[:8]
    split = urlsplit(url)
    host = split.netloc
    canonical_uri = quote(split.path, safe="/")
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": now,
    }
    if session_token:
        headers["x-amz-security-token"] = session_token
    signed = sorted(headers.items())
    canonical_headers = "".join(
        f"{name}:{value}\n" for name, value in signed
    )
    header_names = ";".join(name for name, _ in signed)
    canonical = "\n".join(
        [
            method,
            canonical_uri,
            split.query,
            canonical_headers,
            header_names,
            payload_hash,
        ]
    )
    scope = f"{date}/{region}/{_SERVICE}/aws4_request"
    to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            now,
            scope,
            hashlib.sha256(canonical.encode()).hexdigest(),
        ]
    )
    key = _hmac(
        _hmac(
            _hmac(_hmac(f"AWS4{secret_key}".encode(), date), region),
            _SERVICE,
        ),
        "aws4_request",
    )
    signature = hmac.new(
        key, to_sign.encode(), hashlib.sha256
    ).hexdigest()
    auth = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={header_names}, Signature={signature}"
    )
    out = {
        "authorization": auth,
        "content-type": "application/json",
    }
    out.update({k: v for k, v in headers.items() if k != "host"})
    return out


def auth_headers(
    call: Any, method: str, url: str, body: bytes, region: str
) -> dict:
    """Build the auth headers for one call: API key or SigV4.

    Args:
        call (Any): The resolved request.
        method (str): HTTP method.
        url (str): The full request URL.
        body (bytes): The encoded request body.
        region (str): AWS region.

    Returns:
        dict: Headers for the request.
    """
    import os

    from routehub.exceptions import AuthenticationError

    extra = dict(call.extra_headers or {})
    if call.api_key and call.api_key not in _DUMMY_KEYS:
        headers = {
            "authorization": f"Bearer {call.api_key}",
            "content-type": "application/json",
        }
        headers.update(extra)
        return headers
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        headers = sigv4_headers(
            method,
            url,
            body,
            region,
            access_key,
            secret_key,
            os.environ.get("AWS_SESSION_TOKEN"),
        )
        headers.update(extra)
        return headers
    raise AuthenticationError(
        "Bedrock needs a BEDROCK_API_KEY or AWS credentials "
        "(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)",
        llm_provider="bedrock",
        model=call.model,
    )


def _usage(raw: Optional[dict]) -> dict:
    """Convert Converse usage into the OpenAI usage shape.

    Args:
        raw (Optional[dict]): Converse usage block.

    Returns:
        dict: OpenAI-shaped usage, with prompt tokens including cache reads
        and writes, and the Anthropic cache counts kept alongside.
    """
    raw = raw or {}
    cache_read = raw.get("cacheReadInputTokens") or 0
    cache_write = raw.get("cacheWriteInputTokens") or 0
    prompt = (raw.get("inputTokens") or 0) + cache_read + cache_write
    completion = raw.get("outputTokens") or 0
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
    data: dict, model: str, request_id: Optional[str]
) -> Any:
    """Convert a Converse response into an OpenAI ChatCompletion.

    Args:
        data (dict): The decoded Converse response.
        model (str): Model name to report when the response omits it.
        request_id (Optional[str]): The x-amzn-requestid response header.

    Returns:
        ChatCompletion: The OpenAI SDK response object.
    """
    from openai.types.chat import ChatCompletion

    text, tool_calls = [], []
    for block in (
        data.get("output", {}).get("message", {}).get("content") or []
    ):
        if block.get("text") is not None:
            text.append(block["text"])
        tool_use = block.get("toolUse")
        if tool_use:
            tool_calls.append(
                {
                    "id": tool_use.get("toolUseId"),
                    "type": "function",
                    "function": {
                        "name": tool_use.get("name"),
                        "arguments": json.dumps(
                            tool_use.get("input") or {}
                        ),
                    },
                }
            )
    finish = _STOP_REASONS.get(data.get("stopReason"), "stop")
    if finish == "tool_calls" and not tool_calls:
        finish = "stop"
    return ChatCompletion.model_construct(
        id=request_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": "".join(text) or None,
                    **(
                        {"tool_calls": tool_calls}
                        if tool_calls
                        else {}
                    ),
                },
            }
        ],
        usage=_usage(data.get("usage")),
    )


def response_error(response: Any, model: str) -> Exception:
    """Build the exception for a non-2xx Converse response.

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
    message = (
        body.get("message")
        if isinstance(body, dict)
        else response.text or f"HTTP {response.status_code}"
    )
    return exception_for_status(
        response.status_code,
        message,
        "bedrock",
        model,
        response=response,
        body=body,
    )


def _prepare(call: Any) -> Tuple[str, bytes, str]:
    """Build the URL and encoded body for one call.

    Args:
        call (Any): The resolved request.

    Returns:
        Tuple[str, bytes, str]: URL, body bytes and region.
    """
    from routehub.exceptions import BadRequestError

    region = region_for(call.extras)
    if not region:
        raise BadRequestError(
            "Bedrock needs a region: set AWS_REGION or AWS_DEFAULT_REGION, "
            "or pass aws_region=",
            llm_provider="bedrock",
            model=call.model,
        )
    rejected = sorted(
        name
        for name, value in {**call.params, **call.extras}.items()
        if value is not None and name not in _SUPPORTED_PARAMS
    )
    if rejected and not call.drop_params:
        raise BadRequestError(
            f"Bedrock does not support {', '.join(rejected)}; "
            "pass drop_params=True to ignore them",
            llm_provider="bedrock",
            model=call.model,
        )
    body = build_request(call.bare, call.messages, call.params)
    if call.extra_body:
        body.update(call.extra_body)
    content = json.dumps(body).encode()
    return (
        converse_url(call.api_base, region, call.bare),
        content,
        region,
    )


def complete(call: Any, client: Any) -> Any:
    """Run one synchronous Converse call, retrying retryable failures.

    Args:
        call (Any): The resolved request.
        client (Any): An httpx-compatible sync client.

    Returns:
        Any: The ChatCompletion for the call.
    """
    from routehub._http import httpx
    from routehub.exceptions import map_exception
    from routehub.main import (
        _RETRYABLE_STATUS,
        _backoff,
        _retry_delay,
    )

    url, content, region = _prepare(call)
    attempt = 0
    while True:
        headers = auth_headers(call, "POST", url, content, region)
        try:
            response = client.post(
                url,
                headers=headers,
                content=content,
                timeout=call.timeout,
            )
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
            return to_chat_completion(
                response.json(),
                call.model,
                response.headers.get("x-amzn-requestid"),
            )
        if (
            response.status_code in _RETRYABLE_STATUS
            and attempt < call.max_retries
        ):
            delay = _retry_delay(response, attempt)
            response.close()
            time.sleep(delay)
            attempt += 1
            continue
        error = response_error(response, call.model)
        response.close()
        raise error


async def acomplete(call: Any, client: Any) -> Any:
    """Run one asynchronous Converse call, retrying retryable failures.

    Args:
        call (Any): The resolved request.
        client (Any): An httpx-compatible async client.

    Returns:
        Any: The ChatCompletion for the call.
    """
    import asyncio

    from routehub._http import httpx
    from routehub.exceptions import map_exception
    from routehub.main import (
        _RETRYABLE_STATUS,
        _backoff,
        _retry_delay,
    )

    url, content, region = _prepare(call)
    attempt = 0
    while True:
        headers = auth_headers(call, "POST", url, content, region)
        try:
            response = await client.post(
                url,
                headers=headers,
                content=content,
                timeout=call.timeout,
            )
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
            return to_chat_completion(
                response.json(),
                call.model,
                response.headers.get("x-amzn-requestid"),
            )
        if (
            response.status_code in _RETRYABLE_STATUS
            and attempt < call.max_retries
        ):
            delay = _retry_delay(response, attempt)
            await response.aclose()
            await asyncio.sleep(delay)
            attempt += 1
            continue
        error = response_error(response, call.model)
        await response.aclose()
        raise error
