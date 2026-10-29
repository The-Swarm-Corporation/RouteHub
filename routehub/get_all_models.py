"""List models straight from each provider's API, in one common shape.

This replaces the model file litellm bundles. get_all_models calls every
configured provider's models endpoint concurrently; get_model calls the
per-model endpoint, which for OpenRouter also lists every upstream endpoint
serving the model. Results are cached for CACHE_TTL_SECONDS, and providers
without a configured API key are skipped.

Each entry is a dict with these keys, None where the provider does not say:
id, provider, model, name, type, context_window, max_output_tokens,
input_modalities, output_modalities, supports_vision,
supports_function_calling, supports_reasoning, supports_structured_output,
input_cost_per_token, output_cost_per_token, created, endpoints, raw.
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger("routehub")

CACHE_TTL_SECONDS = 300
TIMEOUT_SECONDS = 10

_lock = threading.Lock()
_cache: Dict[Tuple[str, Optional[str]], Tuple[float, List[dict]]] = {}


def _entry(
    provider: str, model: str, raw: dict, **fields: Any
) -> dict:
    """Build one model entry in the common shape.

    Args:
        provider (str): RouteHub provider name.
        model (str): The model id as the provider names it.
        raw (dict): The provider's original record.
        **fields (Any): Known values for the common keys.

    Returns:
        dict: The entry, with every common key present.
    """
    entry = {
        "id": f"{provider}/{model}",
        "provider": provider,
        "model": model,
        "name": None,
        "type": None,
        "context_window": None,
        "max_output_tokens": None,
        "input_modalities": None,
        "output_modalities": None,
        "supports_vision": None,
        "supports_function_calling": None,
        "supports_reasoning": None,
        "supports_structured_output": None,
        "input_cost_per_token": None,
        "output_cost_per_token": None,
        "created": None,
        "endpoints": None,
    }
    entry.update(fields)
    entry["raw"] = raw
    return entry


def _guess_type(model: str) -> str:
    """Infer a model's type from its id, for providers that do not report one.

    Args:
        model (str): The model id.

    Returns:
        str: One of chat, embedding, image, audio, moderation or realtime.
    """
    name = model.lower()
    if "embed" in name:
        return "embedding"
    if any(
        k in name
        for k in ("dall-e", "gpt-image", "imagen", "image-gen")
    ):
        return "image"
    if any(
        k in name
        for k in ("whisper", "tts", "transcribe", "audio", "speech")
    ):
        return "audio"
    if "moderation" in name or "guard" in name:
        return "moderation"
    if "realtime" in name:
        return "realtime"
    return "chat"


def _price(value: Any, scale: float = 1.0) -> Optional[float]:
    """Parse a price into USD per token.

    Args:
        value (Any): The provider's price, as a number or string.
        scale (float): Divisor that converts the provider's unit to per token.

    Returns:
        Optional[float]: The price per token, or None when absent.
    """
    try:
        return float(value) / scale if value is not None else None
    except (TypeError, ValueError):
        return None


def _iso_epoch(value: Optional[str]) -> Optional[int]:
    """Convert an ISO 8601 timestamp to Unix seconds.

    Args:
        value (Optional[str]): The timestamp.

    Returns:
        Optional[int]: Seconds since the epoch, or None.
    """
    if not value:
        return None
    try:
        return int(
            datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).timestamp()
        )
    except ValueError:
        return None


def _supported(capabilities: dict, name: str) -> Optional[bool]:
    """Read an Anthropic capability flag.

    Args:
        capabilities (dict): The model's capabilities object.
        name (str): Capability name.

    Returns:
        Optional[bool]: Whether it is supported, or None when not reported.
    """
    value = capabilities.get(name)
    return value.get("supported") if isinstance(value, dict) else None


def _parse_openai_list(
    provider: str,
) -> Callable[[List[dict]], List[dict]]:
    """Parser for OpenAI-style records that carry only ids.

    Args:
        provider (str): RouteHub provider name.

    Returns:
        Callable[[List[dict]], List[dict]]: The parser.
    """

    def parse(items: List[dict]) -> List[dict]:
        return [
            _entry(
                provider,
                item["id"],
                item,
                type=_guess_type(item["id"]),
                created=item.get("created"),
            )
            for item in items
            if item.get("id")
        ]

    return parse


def _parse_anthropic(items: List[dict]) -> List[dict]:
    """Parse Anthropic's model records.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    entries = []
    for item in items:
        caps = item.get("capabilities") or {}
        vision = _supported(caps, "image_input")
        inputs = ["text"]
        if vision:
            inputs.append("image")
        if _supported(caps, "pdf_input"):
            inputs.append("file")
        entries.append(
            _entry(
                "anthropic",
                item["id"],
                item,
                name=item.get("display_name"),
                type="chat",
                context_window=item.get("max_input_tokens"),
                max_output_tokens=item.get("max_tokens"),
                input_modalities=inputs if caps else None,
                output_modalities=["text"],
                supports_vision=vision,
                supports_function_calling=True,
                supports_reasoning=_supported(caps, "thinking"),
                supports_structured_output=_supported(
                    caps, "structured_outputs"
                ),
                created=_iso_epoch(item.get("created_at")),
            )
        )
    return entries


def _parse_gemini(items: List[dict]) -> List[dict]:
    """Parse Gemini's model records.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    entries = []
    for item in items:
        model = (item.get("name") or "").removeprefix("models/")
        if not model:
            continue
        methods = item.get("supportedGenerationMethods") or []
        if "generateContent" in methods:
            kind = "chat"
        elif "embedContent" in methods:
            kind = "embedding"
        elif "bidiGenerateContent" in methods:
            kind = "realtime"
        else:
            kind = _guess_type(model)
        entries.append(
            _entry(
                "gemini",
                model,
                item,
                name=item.get("displayName"),
                type=kind,
                context_window=item.get("inputTokenLimit"),
                max_output_tokens=item.get("outputTokenLimit"),
                supports_reasoning=item.get("thinking"),
            )
        )
    return entries


def _parse_groq(items: List[dict]) -> List[dict]:
    """Parse Groq's model records, skipping inactive models.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    return [
        _entry(
            "groq",
            item["id"],
            item,
            type=_guess_type(item["id"]),
            context_window=item.get("context_window"),
            max_output_tokens=item.get("max_completion_tokens"),
            created=item.get("created"),
        )
        for item in items
        if item.get("id") and item.get("active", True)
    ]


def _parse_deepseek(items: List[dict]) -> List[dict]:
    """Parse DeepSeek's model records.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    entries = []
    for item in items:
        inputs = item.get("input_modalities")
        entries.append(
            _entry(
                "deepseek",
                item["id"],
                item,
                name=item.get("name"),
                type="chat",
                context_window=item.get("context_window"),
                max_output_tokens=item.get("max_output_tokens"),
                input_modalities=inputs,
                output_modalities=item.get("output_modalities"),
                supports_vision=(
                    ("image" in inputs) if inputs else None
                ),
                supports_function_calling=True,
                supports_reasoning=(
                    True if item.get("effort") else None
                ),
            )
        )
    return entries


def _parse_mistral(items: List[dict]) -> List[dict]:
    """Parse Mistral's model records, one entry per id.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    entries, seen = [], set()
    for item in items:
        model = item.get("id")
        if not model or model in seen:
            continue
        seen.add(model)
        caps = item.get("capabilities") or {}
        entries.append(
            _entry(
                "mistral",
                model,
                item,
                name=item.get("name"),
                type=(
                    "chat"
                    if caps.get("completion_chat")
                    else _guess_type(model)
                ),
                context_window=item.get("max_context_length"),
                supports_vision=caps.get("vision"),
                supports_function_calling=caps.get(
                    "function_calling"
                ),
                created=item.get("created"),
            )
        )
    return entries


def _parse_together(items: List[dict]) -> List[dict]:
    """Parse Together's model records; prices are per million tokens.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    entries = []
    for item in items:
        pricing = item.get("pricing") or {}
        kind = item.get("type")
        entries.append(
            _entry(
                "together_ai",
                item["id"],
                item,
                name=item.get("display_name"),
                type=(
                    "chat"
                    if kind in ("chat", "language", "code")
                    else kind
                ),
                context_window=item.get("context_length") or None,
                input_cost_per_token=_price(
                    pricing.get("input"), 1e6
                ),
                output_cost_per_token=_price(
                    pricing.get("output"), 1e6
                ),
                created=item.get("created"),
            )
        )
    return entries


def _parse_fireworks(items: List[dict]) -> List[dict]:
    """Parse Fireworks' model records.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    return [
        _entry(
            "fireworks_ai",
            item["id"],
            item,
            type=(
                "chat"
                if item.get("supports_chat")
                else _guess_type(item["id"])
            ),
            context_window=item.get("context_length"),
            supports_vision=item.get("supports_image_input"),
            supports_function_calling=item.get("supports_tools"),
            created=item.get("created"),
        )
        for item in items
        if item.get("id")
    ]


def _parse_cohere(items: List[dict]) -> List[dict]:
    """Parse Cohere's model records.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    entries = []
    for item in items:
        endpoints = item.get("endpoints") or []
        features = item.get("features") or []
        if "chat" in endpoints:
            kind = "chat"
        elif "rerank" in endpoints:
            kind = "rerank"
        else:
            kind = _guess_type(item["name"])
        entries.append(
            _entry(
                "cohere",
                item["name"],
                item,
                type=kind,
                context_window=item.get("context_length") or None,
                supports_vision="vision" in features,
                supports_function_calling="tools" in features,
                supports_reasoning="reasoning" in features,
                supports_structured_output="json_schema" in features,
            )
        )
    return entries


def _openrouter_fields(
    item: dict, params: set, pricing: dict
) -> dict:
    """Common capability and price fields for an OpenRouter record.

    Args:
        item (dict): The model record.
        params (set): Supported request parameters.
        pricing (dict): Per-token prices as strings.

    Returns:
        dict: Fields for _entry.
    """
    architecture = item.get("architecture") or {}
    inputs = architecture.get("input_modalities") or []
    outputs = architecture.get("output_modalities") or []
    return {
        "name": item.get("name"),
        "type": (
            "chat" if "text" in outputs or not outputs else "image"
        ),
        "input_modalities": inputs or None,
        "output_modalities": outputs or None,
        "supports_vision": "image" in inputs,
        "supports_function_calling": "tools" in params,
        "supports_reasoning": "reasoning" in params
        or item.get("reasoning") is not None,
        "supports_structured_output": "structured_outputs" in params,
        "input_cost_per_token": _price(pricing.get("prompt")),
        "output_cost_per_token": _price(pricing.get("completion")),
        "created": item.get("created"),
    }


def _parse_openrouter(items: List[dict]) -> List[dict]:
    """Parse OpenRouter's model records; prices are per token.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    entries = []
    for item in items:
        top = item.get("top_provider") or {}
        entries.append(
            _entry(
                "openrouter",
                item["id"],
                item,
                context_window=item.get("context_length")
                or top.get("context_length"),
                max_output_tokens=top.get("max_completion_tokens"),
                **_openrouter_fields(
                    item,
                    set(item.get("supported_parameters") or []),
                    item.get("pricing") or {},
                ),
            )
        )
    return entries


def _parse_openrouter_endpoints(items: List[dict]) -> List[dict]:
    """Parse OpenRouter's per-model endpoints record.

    The model's limits are the best any upstream endpoint offers, and its
    parameters are those any endpoint supports.

    Args:
        items (List[dict]): The single record from /models/{id}/endpoints.

    Returns:
        List[dict]: One common entry, with the endpoints attached.
    """
    entries = []
    for item in items:
        endpoints = item.get("endpoints") or []
        params = {
            p
            for e in endpoints
            for p in e.get("supported_parameters") or []
        }
        contexts = [
            e["context_length"]
            for e in endpoints
            if e.get("context_length")
        ]
        outputs = [
            e["max_completion_tokens"]
            for e in endpoints
            if e.get("max_completion_tokens")
        ]
        entries.append(
            _entry(
                "openrouter",
                item["id"],
                item,
                context_window=max(contexts) if contexts else None,
                max_output_tokens=max(outputs) if outputs else None,
                endpoints=endpoints,
                **_openrouter_fields(
                    item,
                    params,
                    (
                        endpoints[0].get("pricing")
                        if endpoints
                        else None
                    )
                    or {},
                ),
            )
        )
    return entries


def _parse_ollama(items: List[dict]) -> List[dict]:
    """Parse a local Ollama server's model records.

    Args:
        items (List[dict]): Raw records.

    Returns:
        List[dict]: Common entries.
    """
    return [
        _entry(
            "ollama",
            item["name"],
            item,
            type=_guess_type(item["name"]),
        )
        for item in items
        if item.get("name")
    ]


@dataclass(frozen=True)
class ModelSource:
    """Where and how to list one provider's models.

    Attributes:
        provider (str): RouteHub provider name.
        url (str): The model-listing endpoint.
        parse (Callable[[List[dict]], List[dict]]): Turns raw records into entries.
        key_env (Tuple[str, ...]): Environment variables holding the API key;
            empty when the endpoint is public.
        auth (str): How the key is sent: bearer, x-api-key, goog or none.
        list_key (Optional[str]): Response field holding the records; None
            when the response is a bare list.
        params (Dict[str, Any]): Query parameters for the listing.
        headers (Dict[str, str]): Extra headers.
        page_param (Optional[str]): Query parameter that requests the next page.
        detail_url (Optional[str]): Per-model endpoint, with {model} in place
            of the model id; None when the provider has none.
        detail_key (Optional[str]): Response field holding the per-model record.
        detail_parse (Optional[Callable]): Parser for the per-model record,
            when it differs from the listing parser.
        base_env (Tuple[str, ...]): Environment variables overriding the base
            URL; a public source with base_env is only listed by default when
            one of them is set.
    """

    provider: str
    url: str
    parse: Callable[[List[dict]], List[dict]]
    key_env: Tuple[str, ...] = ()
    auth: str = "bearer"
    list_key: Optional[str] = "data"
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    page_param: Optional[str] = None
    detail_url: Optional[str] = None
    detail_key: Optional[str] = None
    detail_parse: Optional[Callable[[List[dict]], List[dict]]] = None
    base_env: Tuple[str, ...] = ()


SOURCES: Dict[str, ModelSource] = {
    s.provider: s
    for s in (
        ModelSource(
            "openrouter",
            "https://openrouter.ai/api/v1/models",
            _parse_openrouter,
            auth="none",
            detail_url="https://openrouter.ai/api/v1/models/{model}/endpoints",
            detail_key="data",
            detail_parse=_parse_openrouter_endpoints,
        ),
        ModelSource(
            "openai",
            "https://api.openai.com/v1/models",
            _parse_openai_list("openai"),
            ("OPENAI_API_KEY",),
            detail_url="https://api.openai.com/v1/models/{model}",
        ),
        ModelSource(
            "anthropic",
            "https://api.anthropic.com/v1/models",
            _parse_anthropic,
            ("ANTHROPIC_API_KEY",),
            auth="x-api-key",
            params={"limit": 1000},
            headers={"anthropic-version": "2023-06-01"},
            page_param="after_id",
            detail_url="https://api.anthropic.com/v1/models/{model}",
        ),
        ModelSource(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/models",
            _parse_gemini,
            ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            auth="goog",
            list_key="models",
            params={"pageSize": 1000},
            page_param="pageToken",
            detail_url="https://generativelanguage.googleapis.com/v1beta/models/{model}",
        ),
        ModelSource(
            "groq",
            "https://api.groq.com/openai/v1/models",
            _parse_groq,
            ("GROQ_API_KEY",),
            detail_url="https://api.groq.com/openai/v1/models/{model}",
        ),
        ModelSource(
            "xai",
            "https://api.x.ai/v1/models",
            _parse_openai_list("xai"),
            ("XAI_API_KEY",),
            detail_url="https://api.x.ai/v1/models/{model}",
        ),
        ModelSource(
            "deepseek",
            "https://api.deepseek.com/models",
            _parse_deepseek,
            ("DEEPSEEK_API_KEY",),
            detail_url="https://api.deepseek.com/models/{model}",
        ),
        ModelSource(
            "mistral",
            "https://api.mistral.ai/v1/models",
            _parse_mistral,
            ("MISTRAL_API_KEY",),
            detail_url="https://api.mistral.ai/v1/models/{model}",
        ),
        ModelSource(
            "together_ai",
            "https://api.together.xyz/v1/models",
            _parse_together,
            (
                "TOGETHERAI_API_KEY",
                "TOGETHER_API_KEY",
                "TOGETHER_AI_TOKEN",
            ),
            list_key=None,
        ),
        ModelSource(
            "fireworks_ai",
            "https://api.fireworks.ai/inference/v1/models",
            _parse_fireworks,
            ("FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY"),
        ),
        ModelSource(
            "cerebras",
            "https://api.cerebras.ai/v1/models",
            _parse_openai_list("cerebras"),
            ("CEREBRAS_API_KEY",),
            detail_url="https://api.cerebras.ai/v1/models/{model}",
        ),
        ModelSource(
            "cohere",
            "https://api.cohere.com/v1/models",
            _parse_cohere,
            ("COHERE_API_KEY", "CO_API_KEY"),
            list_key="models",
            params={"page_size": 1000},
            page_param="page_token",
        ),
        ModelSource(
            "ollama",
            "http://localhost:11434/api/tags",
            _parse_ollama,
            auth="none",
            list_key="models",
            base_env=("OLLAMA_API_BASE",),
        ),
    )
}


def _source(provider: str) -> ModelSource:
    """Look up a provider's source.

    Args:
        provider (str): RouteHub provider name.

    Returns:
        ModelSource: The source.
    """
    try:
        return SOURCES[provider]
    except KeyError:
        raise ValueError(
            f"No model list for provider {provider!r}. "
            f"Available: {', '.join(SOURCES)}"
        ) from None


def _api_key(
    source: ModelSource, api_key: Optional[str]
) -> Optional[str]:
    """Pick the explicit key or the first configured environment variable.

    Args:
        source (ModelSource): The provider's source.
        api_key (Optional[str]): Explicit key.

    Returns:
        Optional[str]: The key, or None.
    """
    if api_key:
        return api_key
    for name in source.key_env:
        if os.environ.get(name):
            return os.environ[name]
    return None


def _should_fetch(
    source: ModelSource, key: Optional[str], named: bool
) -> bool:
    """Whether a provider can be listed with the current configuration.

    Args:
        source (ModelSource): The provider's source.
        key (Optional[str]): The API key, if any.
        named (bool): Whether the caller asked for this provider by name.

    Returns:
        bool: True when it should be fetched.
    """
    if source.key_env:
        return key is not None
    if source.base_env and not named:
        return any(os.environ.get(name) for name in source.base_env)
    return True


def _url(source: ModelSource, url: str) -> str:
    """Apply a base-URL environment override to a source URL.

    Args:
        source (ModelSource): The provider's source.
        url (str): The default URL.

    Returns:
        str: The URL to request.
    """
    for name in source.base_env:
        base = os.environ.get(name)
        if base:
            path = url.split("://", 1)[1].split("/", 1)[1]
            return f"{base.rstrip('/').removesuffix('/v1')}/{path}"
    return url


def _headers(
    source: ModelSource, key: Optional[str]
) -> Dict[str, str]:
    """Build request headers, including authentication.

    Args:
        source (ModelSource): The provider's source.
        key (Optional[str]): The API key.

    Returns:
        Dict[str, str]: The headers.
    """
    headers = dict(source.headers)
    if key and source.auth == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif key and source.auth == "x-api-key":
        headers["x-api-key"] = key
    elif key and source.auth == "goog":
        headers["x-goog-api-key"] = key
    return headers


def _records(
    source: ModelSource, body: Any
) -> Tuple[List[dict], Optional[str]]:
    """Extract records and the next-page cursor from a listing response.

    Args:
        source (ModelSource): The provider's source.
        body (Any): The decoded response.

    Returns:
        Tuple[List[dict], Optional[str]]: Records and the cursor, if any.
    """
    if source.list_key is None:
        return (body if isinstance(body, list) else []), None
    records = body.get(source.list_key) or []
    cursor = None
    if source.page_param == "after_id" and body.get("has_more"):
        cursor = body.get("last_id")
    elif source.page_param == "pageToken":
        cursor = body.get("nextPageToken")
    elif source.page_param == "page_token":
        cursor = body.get("next_page_token")
    return records, cursor


def _cached(
    cache_key: Tuple[str, Optional[str]],
) -> Optional[List[dict]]:
    """Return fresh cached entries for a provider, if any.

    Args:
        cache_key (Tuple[str, Optional[str]]): Provider and API key.

    Returns:
        Optional[List[dict]]: The entries, or None when missing or expired.
    """
    with _lock:
        hit = _cache.get(cache_key)
    if hit and time.monotonic() < hit[0]:
        return hit[1]
    return None


def _store(
    cache_key: Tuple[str, Optional[str]], entries: List[dict]
) -> None:
    """Cache a provider's entries for CACHE_TTL_SECONDS, even when empty.

    A failed fetch caches an empty list too, so a provider that is down is
    retried once per TTL rather than on every call.

    Args:
        cache_key (Tuple[str, Optional[str]]): Provider and API key.
        entries (List[dict]): The entries.
    """
    with _lock:
        _cache[cache_key] = (
            time.monotonic() + CACHE_TTL_SECONDS,
            entries,
        )


def clear_cache() -> None:
    """Forget every cached model list and model so the next call fetches."""
    with _lock:
        _cache.clear()


def _describe(error: Exception) -> str:
    """Summarize a request failure in one line.

    Args:
        error (Exception): The failure.

    Returns:
        str: The HTTP status, or the error type and message.
    """
    response = getattr(error, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}"
    return f"{type(error).__name__}: {error}"


def _fetch_list(
    source: ModelSource,
    key: Optional[str],
    client: Any,
    timeout: float,
) -> List[dict]:
    """Download every page of one provider's model list.

    Args:
        source (ModelSource): The provider's source.
        key (Optional[str]): The API key.
        client (Any): An httpx-compatible client.
        timeout (float): Per-request timeout in seconds.

    Returns:
        List[dict]: Common entries; empty when the request fails.
    """
    from routehub._json import loads

    headers, params = _headers(source, key), dict(source.params)
    records: List[dict] = []
    try:
        while True:
            response = client.get(
                _url(source, source.url),
                headers=headers,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            page, cursor = _records(source, loads(response.content))
            records.extend(page)
            if not cursor:
                break
            params[source.page_param] = cursor
        return source.parse(records)
    except Exception as error:
        logger.warning(
            f"Could not list {source.provider} models: {_describe(error)}"
        )
        return []


async def _afetch_list(
    source: ModelSource,
    key: Optional[str],
    client: Any,
    timeout: float,
) -> List[dict]:
    """Download every page of one provider's model list asynchronously.

    Args:
        source (ModelSource): The provider's source.
        key (Optional[str]): The API key.
        client (Any): An httpx-compatible async client.
        timeout (float): Per-request timeout in seconds.

    Returns:
        List[dict]: Common entries; empty when the request fails.
    """
    from routehub._json import loads

    headers, params = _headers(source, key), dict(source.params)
    records: List[dict] = []
    try:
        while True:
            response = await client.get(
                _url(source, source.url),
                headers=headers,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            page, cursor = _records(source, loads(response.content))
            records.extend(page)
            if not cursor:
                break
            params[source.page_param] = cursor
        return source.parse(records)
    except Exception as error:
        logger.warning(
            f"Could not list {source.provider} models: {_describe(error)}"
        )
        return []


def _plan(providers: Optional[Iterable[str]], refresh: bool) -> Tuple[
    List[ModelSource],
    Dict[str, List[dict]],
    List[Tuple[ModelSource, Optional[str]]],
]:
    """Split providers into cached results and ones still to fetch.

    Args:
        providers (Optional[Iterable[str]]): Providers asked for, or None for all.
        refresh (bool): Ignore the cache.

    Returns:
        Tuple: The sources in order, results already cached, and the sources
        and keys that need fetching.
    """
    named = providers is not None
    sources = [_source(p) for p in (providers if named else SOURCES)]
    cached: Dict[str, List[dict]] = {}
    to_fetch: List[Tuple[ModelSource, Optional[str]]] = []
    for source in sources:
        key = _api_key(source, None)
        if not _should_fetch(source, key, named):
            continue
        hit = None if refresh else _cached((source.provider, key))
        if hit is not None:
            cached[source.provider] = hit
        else:
            to_fetch.append((source, key))
    return sources, cached, to_fetch


def _flatten(
    sources: List[ModelSource], results: Dict[str, List[dict]]
) -> List[dict]:
    """Concatenate results in source order.

    Args:
        sources (List[ModelSource]): Sources in the order asked for.
        results (Dict[str, List[dict]]): Entries per provider.

    Returns:
        List[dict]: All entries, grouped by provider.
    """
    return [e for s in sources for e in results.get(s.provider, [])]


def get_models(
    provider: str,
    api_key: Optional[str] = None,
    timeout: float = TIMEOUT_SECONDS,
    refresh: bool = False,
) -> List[dict]:
    """List one provider's models from its API.

    Args:
        provider (str): RouteHub provider name, such as "anthropic".
        api_key (Optional[str]): API key; read from the provider's env var
            when omitted.
        timeout (float): Per-request timeout in seconds.
        refresh (bool): Ignore the cache and fetch again.

    Returns:
        List[dict]: Model entries; empty when no key is configured or the
        request fails.
    """
    from routehub._http import httpx

    source = _source(provider)
    key = _api_key(source, api_key)
    if not _should_fetch(source, key, named=True):
        return []
    if not refresh:
        hit = _cached((provider, key))
        if hit is not None:
            return hit
    with httpx.Client(follow_redirects=True) as client:
        entries = _fetch_list(source, key, client, timeout)
    _store((provider, key), entries)
    return entries


def get_all_models(
    providers: Optional[Iterable[str]] = None,
    timeout: float = TIMEOUT_SECONDS,
    refresh: bool = False,
) -> List[dict]:
    """List models from every configured provider concurrently.

    The call takes about as long as the slowest provider. A provider that
    fails or has no API key contributes nothing.

    Args:
        providers (Optional[Iterable[str]]): Providers to list; every
            provider in SOURCES when omitted.
        timeout (float): Per-request timeout in seconds.
        refresh (bool): Ignore the cache and fetch again.

    Returns:
        List[dict]: Model entries from all providers, grouped by provider.
    """
    from routehub._http import httpx

    sources, results, to_fetch = _plan(providers, refresh)
    if to_fetch:
        with httpx.Client(follow_redirects=True) as client:
            with ThreadPoolExecutor(
                max_workers=len(to_fetch)
            ) as pool:
                futures = [
                    (
                        source,
                        key,
                        pool.submit(
                            _fetch_list, source, key, client, timeout
                        ),
                    )
                    for source, key in to_fetch
                ]
                for source, key, future in futures:
                    entries = future.result()
                    _store((source.provider, key), entries)
                    results[source.provider] = entries
    return _flatten(sources, results)


async def aget_models(
    provider: str,
    api_key: Optional[str] = None,
    timeout: float = TIMEOUT_SECONDS,
    refresh: bool = False,
) -> List[dict]:
    """Async form of get_models, sharing its cache.

    Args:
        provider (str): RouteHub provider name.
        api_key (Optional[str]): API key; read from the provider's env var
            when omitted.
        timeout (float): Per-request timeout in seconds.
        refresh (bool): Ignore the cache and fetch again.

    Returns:
        List[dict]: Model entries.
    """
    from routehub._http import httpx

    source = _source(provider)
    key = _api_key(source, api_key)
    if not _should_fetch(source, key, named=True):
        return []
    if not refresh:
        hit = _cached((provider, key))
        if hit is not None:
            return hit
    async with httpx.AsyncClient(follow_redirects=True) as client:
        entries = await _afetch_list(source, key, client, timeout)
    _store((provider, key), entries)
    return entries


async def aget_all_models(
    providers: Optional[Iterable[str]] = None,
    timeout: float = TIMEOUT_SECONDS,
    refresh: bool = False,
) -> List[dict]:
    """Async form of get_all_models, sharing its cache.

    Args:
        providers (Optional[Iterable[str]]): Providers to list; every
            provider in SOURCES when omitted.
        timeout (float): Per-request timeout in seconds.
        refresh (bool): Ignore the cache and fetch again.

    Returns:
        List[dict]: Model entries from all providers, grouped by provider.
    """
    from routehub._http import httpx

    sources, results, to_fetch = _plan(providers, refresh)
    if to_fetch:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            fetched = await asyncio.gather(
                *(
                    _afetch_list(s, k, client, timeout)
                    for s, k in to_fetch
                )
            )
        for (source, key), entries in zip(to_fetch, fetched):
            _store((source.provider, key), entries)
            results[source.provider] = entries
    return _flatten(sources, results)


def get_model(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    timeout: float = TIMEOUT_SECONDS,
) -> Optional[dict]:
    """Fetch one model's details from the provider's per-model endpoint.

    Falls back to searching the provider's model list when it has no
    per-model endpoint. For OpenRouter the entry includes every upstream
    endpoint serving the model, with its own limits and prices. The answer
    is cached for CACHE_TTL_SECONDS, as the model lists are.

    Args:
        provider (str): RouteHub provider name.
        model (str): The model id as the provider names it.
        api_key (Optional[str]): API key; read from the provider's env var
            when omitted.
        timeout (float): Per-request timeout in seconds.

    Returns:
        Optional[dict]: The model entry, or None when the provider does not
        know the model or cannot be reached.
    """
    from routehub._http import httpx
    from routehub._json import loads

    source = _source(provider)
    key = _api_key(source, api_key)
    if not _should_fetch(source, key, named=True):
        return None
    if source.detail_url is None:
        return next(
            (
                e
                for e in get_models(provider, key, timeout)
                if e["model"] == model
            ),
            None,
        )
    cache_key = (f"{provider}/{model}", key)
    hit = _cached(cache_key)
    if hit is not None:
        return hit[0] if hit else None

    entry = None
    try:
        response = httpx.get(
            _url(source, source.detail_url.format(model=model)),
            headers=_headers(source, key),
            timeout=timeout,
            follow_redirects=True,
        )
        if response.status_code != 404:
            response.raise_for_status()
            record = loads(response.content)
            if source.detail_key:
                record = record.get(source.detail_key) or {}
            entries = (source.detail_parse or source.parse)([record])
            entry = entries[0] if entries else None
    except Exception as error:
        logger.warning(
            f"Could not fetch {provider} model {model!r}: {_describe(error)}"
        )
    _store(cache_key, [entry] if entry is not None else [])
    return entry


def configured_providers() -> List[str]:
    """List the providers whose models can be listed right now.

    Returns:
        List[str]: Providers with an API key configured, plus public ones.
    """
    return [
        name
        for name, source in SOURCES.items()
        if _should_fetch(source, _api_key(source, None), named=False)
    ]
