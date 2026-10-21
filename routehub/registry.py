"""Model metadata from OpenRouter's live model list, fetched on first use.

Context windows, output limits, prices and capability flags come from
https://openrouter.ai/api/v1/models and are cached for CACHE_TTL_SECONDS.
"""

import logging
import re
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger("routehub")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = 300
TIMEOUT_SECONDS = 10

# RouteHub provider to OpenRouter vendor prefix.
_VENDORS = {
    "openai": "openai",
    "azure": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "xai": "x-ai",
    "mistral": "mistralai",
    "deepseek": "deepseek",
    "moonshot": "moonshotai",
    "dashscope": "qwen",
}
_PROVIDERS_BY_VENDOR = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "x-ai": "xai",
    "mistralai": "mistral",
    "deepseek": "deepseek",
    "moonshotai": "moonshot",
    "qwen": "dashscope",
}

_SUPPORT_FLAGS = (
    "supports_vision",
    "supports_function_calling",
    "supports_parallel_function_calling",
    "supports_tool_choice",
    "supports_response_schema",
    "supports_system_messages",
    "supports_prompt_caching",
    "supports_reasoning",
    "supports_pdf_input",
    "supports_audio_input",
    "supports_audio_output",
    "supports_web_search",
    "supports_computer_use",
    "supports_assistant_prefill",
)

_DATE_SUFFIX = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")
_VERSION_HYPHEN = re.compile(r"(?<=\d)-(?=\d)")

_lock = threading.Lock()


class ModelNotMappedError(Exception):
    """The model is in neither OpenRouter's list nor the registered models."""


_catalog: Dict[str, dict] = {}
_aliases: Dict[str, str] = {}
_info_cache: Dict[str, dict] = {}
_expires_at = 0.0
_registered: Dict[str, dict] = {}

__all__ = [
    "ModelNotMappedError",
    "OPENROUTER_MODELS_URL",
    "clear_model_cache",
    "get_max_tokens",
    "get_model_info",
    "model_cost",  # noqa: F822  served by the module __getattr__
    "model_list",  # noqa: F822  served by the module __getattr__
    "register_model",
    *_SUPPORT_FLAGS,
]


def _fetch() -> List[dict]:
    """Download OpenRouter's model list.

    Returns:
        List[dict]: The raw model entries, or an empty list on failure.
    """
    from routehub._http import httpx

    try:
        response = httpx.get(
            OPENROUTER_MODELS_URL, timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        from routehub._json import loads

        return loads(response.content).get("data") or []
    except Exception as error:
        logger.warning(f"Could not fetch OpenRouter models: {error}")
        return []


def _normalize(name: str) -> str:
    """Rewrite a model name into OpenRouter's naming style.

    Args:
        name (str): A model name such as "claude-opus-4-7-20251001".

    Returns:
        str: The normalized name, such as "claude-opus-4.7".
    """
    name = name.lower()
    if name.endswith("-latest"):
        name = name[: -len("-latest")]
    name = _DATE_SUFFIX.sub("", name)
    return _VERSION_HYPHEN.sub(".", name)


def _build_aliases(catalog: Dict[str, dict]) -> Dict[str, str]:
    """Index every OpenRouter id under the names callers use for it.

    Args:
        catalog (Dict[str, dict]): OpenRouter id to raw entry.

    Returns:
        Dict[str, str]: Lookup key to OpenRouter id.
    """
    aliases: Dict[str, str] = {}
    normalized: Dict[str, str] = {}
    for model_id in catalog:
        lowered = model_id.lower()
        aliases[lowered] = model_id
        # Variants such as ":free" or ":batch" are reachable by their full id only.
        if ":" in lowered or "/" not in lowered:
            continue
        name = lowered.split("/", 1)[1]
        aliases.setdefault(name, model_id)
        normalized.setdefault(_normalize(name), model_id)
    return {**normalized, **aliases}


def _ensure_catalog() -> Dict[str, dict]:
    """Return the cached catalog, fetching it when missing or expired.

    Returns:
        Dict[str, dict]: OpenRouter id to raw entry.
    """
    global _catalog, _aliases, _expires_at
    if time.monotonic() < _expires_at:
        return _catalog
    with _lock:
        if time.monotonic() < _expires_at:
            return _catalog
        entries = _fetch()
        if entries:
            _catalog = {e["id"]: e for e in entries if e.get("id")}
            _aliases = _build_aliases(_catalog)
            _info_cache.clear()
        # Expire even on failure so a dead endpoint is retried once a TTL.
        _expires_at = time.monotonic() + CACHE_TTL_SECONDS
    return _catalog


def clear_model_cache() -> None:
    """Forget the cached OpenRouter list so the next lookup fetches again."""
    global _catalog, _aliases, _expires_at
    with _lock:
        _catalog, _aliases = {}, {}
        _info_cache.clear()
        _expires_at = 0.0


def _price(value: Optional[str]) -> Optional[float]:
    """Parse an OpenRouter per-token price string.

    Args:
        value (Optional[str]): Price in USD per token, as a string.

    Returns:
        Optional[float]: The price, or None when absent or unparseable.
    """
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_info(model_id: str, entry: dict) -> dict:
    """Convert an OpenRouter entry into litellm's model-info shape.

    Args:
        model_id (str): The OpenRouter id.
        entry (dict): The raw OpenRouter entry.

    Returns:
        dict: Model info with context window, output limit, prices and flags.
    """
    cached = _info_cache.get(model_id)
    if cached is not None:
        return cached
    architecture = entry.get("architecture") or {}
    inputs = architecture.get("input_modalities") or []
    outputs = architecture.get("output_modalities") or []
    params = set(entry.get("supported_parameters") or [])
    pricing = entry.get("pricing") or {}
    top = entry.get("top_provider") or {}
    context = entry.get("context_length") or top.get("context_length")
    max_output = top.get("max_completion_tokens")
    vendor = model_id.split("/", 1)[0]
    info = {
        "key": model_id,
        "name": entry.get("name"),
        "litellm_provider": _PROVIDERS_BY_VENDOR.get(
            vendor, "openrouter"
        ),
        "mode": (
            "chat"
            if "text" in outputs or not outputs
            else "image_generation"
        ),
        "max_input_tokens": context,
        "max_output_tokens": max_output,
        "max_tokens": max_output or context,
        "input_cost_per_token": _price(pricing.get("prompt")),
        "output_cost_per_token": _price(pricing.get("completion")),
        "cache_read_input_token_cost": _price(
            pricing.get("input_cache_read")
        ),
        "cache_creation_input_token_cost": _price(
            pricing.get("input_cache_write")
        ),
        "supports_vision": "image" in inputs,
        "supports_pdf_input": "file" in inputs,
        "supports_audio_input": "audio" in inputs,
        "supports_video_input": "video" in inputs,
        "supports_audio_output": "audio" in outputs,
        "supports_function_calling": "tools" in params,
        "supports_parallel_function_calling": "tools" in params,
        "supports_tool_choice": "tool_choice" in params,
        "supports_response_schema": "structured_outputs" in params,
        "supports_reasoning": "reasoning" in params
        or "include_reasoning" in params
        or entry.get("reasoning") is not None,
        "supports_reasoning_effort": "reasoning_effort" in params,
        "supports_prompt_caching": "input_cache_read" in pricing,
        "supports_web_search": "web_search_options" in params
        or "web_search" in pricing,
        "supports_system_messages": True,
        "supports_computer_use": None,
        "supports_assistant_prefill": None,
        "supported_openai_params": sorted(params),
        "knowledge_cutoff": entry.get("knowledge_cutoff"),
    }
    _info_cache[model_id] = info
    return info


def _candidates(
    model: str, custom_llm_provider: Optional[str]
) -> List[str]:
    """List the lookup keys to try for a model string, most specific first.

    Args:
        model (str): A bare or provider-prefixed model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        List[str]: Lowercased lookup keys.
    """
    from routehub.providers import split_model

    lowered = model.lower()
    if lowered.startswith("openrouter/"):
        return [lowered[len("openrouter/") :]]
    keys = [lowered]
    bare, provider = split_model(model, custom_llm_provider)
    bare = bare.lower()
    vendor = _VENDORS.get(provider or "")
    if vendor:
        keys += [f"{vendor}/{bare}", f"{vendor}/{_normalize(bare)}"]
    keys += [bare, _normalize(bare)]
    return keys


def _registered_entry(
    model: str, custom_llm_provider: Optional[str]
) -> Optional[Tuple[str, dict]]:
    """Find a model added with register_model.

    Args:
        model (str): A bare or provider-prefixed model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        Optional[Tuple[str, dict]]: The key and entry, or None.
    """
    if not _registered:
        return None
    keys = [model]
    if custom_llm_provider:
        keys.insert(0, f"{custom_llm_provider}/{model}")
    if "/" in model:
        keys.append(model.split("/", 1)[1])
    for key in keys:
        if key in _registered:
            return key, _registered[key]
    return None


def _lookup(
    model: str, custom_llm_provider: Optional[str] = None
) -> Optional[Tuple[str, dict]]:
    """Find the model info for a model string.

    Args:
        model (str): A bare or provider-prefixed model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        Optional[Tuple[str, dict]]: The matched key and model info, or None.
    """
    found = _registered_entry(model, custom_llm_provider)
    if found:
        return found
    catalog = _ensure_catalog()
    for key in _candidates(model, custom_llm_provider):
        model_id = _aliases.get(key)
        if model_id is not None:
            return model_id, _to_info(model_id, catalog[model_id])
    return None


class ModelList(list):
    """OpenRouter model ids whose membership test resolves aliases.

    Iterating yields OpenRouter ids such as "anthropic/claude-sonnet-4.6",
    while "claude-sonnet-4-6" in model_list is also True.
    """

    def __contains__(self, model: object) -> bool:
        """Whether the registry knows the model under any of its names.

        Args:
            model (object): The model name to test.

        Returns:
            bool: True when known.
        """
        if list.__contains__(self, model):
            return True
        return isinstance(model, str) and _lookup(model) is not None


def __getattr__(name: str):
    """Expose model_cost and model_list, fetched on first access.

    Args:
        name (str): The attribute being looked up.

    Returns:
        Any: The model-info dict or the model list.
    """
    if name == "model_cost":
        catalog = _ensure_catalog()
        merged = {
            model_id: _to_info(model_id, entry)
            for model_id, entry in catalog.items()
        }
        merged.update(_registered)
        return merged
    if name == "model_list":
        catalog = _ensure_catalog()
        return ModelList([*catalog, *_registered])
    raise AttributeError(
        f"module 'routehub.registry' has no attribute {name!r}"
    )


def register_model(model_cost: Union[dict, str]) -> dict:
    """Add or override models, in litellm's model-info shape.

    Registered entries take priority over OpenRouter data and never expire.

    Args:
        model_cost (Union[dict, str]): Model name to info, or a path to a
            JSON file of it. Fields merge into an earlier registration.

    Returns:
        dict: All registered models.
    """
    if isinstance(model_cost, str):
        import json

        with open(model_cost, "rb") as f:
            model_cost = json.load(f)
    with _lock:
        for name, entry in model_cost.items():
            _registered[name] = {**_registered.get(name, {}), **entry}
    return dict(_registered)


def get_model_info(
    model: str, custom_llm_provider: Optional[str] = None
) -> dict:
    """Return the metadata for a model.

    Args:
        model (str): A bare or provider-prefixed model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        dict: Context window, output limit, prices and capability flags, with
        a key field naming the matched model.
    """
    found = _lookup(model, custom_llm_provider)
    if found is None:
        raise ModelNotMappedError(
            f"This model isn't mapped yet. model={model}, "
            f"custom_llm_provider={custom_llm_provider}. "
            "It is not in OpenRouter's model list; add it with "
            "routehub.register_model()."
        )
    key, entry = found
    info = dict(entry)
    info.setdefault("key", key)
    info.setdefault("max_input_tokens", info.get("max_tokens"))
    info.setdefault("max_output_tokens", info.get("max_tokens"))
    for flag in _SUPPORT_FLAGS:
        info.setdefault(flag, None)
    return info


def get_max_tokens(model: str) -> Optional[int]:
    """Return a model's maximum output tokens.

    Args:
        model (str): A bare or provider-prefixed model name.

    Returns:
        Optional[int]: The output token limit.
    """
    info = get_model_info(model)
    return info.get("max_output_tokens") or info.get("max_tokens")


def _supports(
    flag: str, model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Check one capability flag, treating unknown models as unsupported.

    Args:
        flag (str): The model-info flag name.
        model (str): A bare or provider-prefixed model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when the model is known to support it.
    """
    found = _lookup(model, custom_llm_provider)
    return bool(found and found[1].get(flag))


def supports_vision(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model accepts image input.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports("supports_vision", model, custom_llm_provider)


def supports_function_calling(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model supports tool calls.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_function_calling", model, custom_llm_provider
    )


def supports_parallel_function_calling(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model can return several tool calls in one turn.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_parallel_function_calling",
        model,
        custom_llm_provider,
    )


def supports_tool_choice(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model accepts the tool_choice parameter.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_tool_choice", model, custom_llm_provider
    )


def supports_response_schema(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model supports JSON-schema structured output.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_response_schema", model, custom_llm_provider
    )


def supports_system_messages(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model accepts system messages.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_system_messages", model, custom_llm_provider
    )


def supports_prompt_caching(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the provider caches prompts for this model.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_prompt_caching", model, custom_llm_provider
    )


def supports_reasoning(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model is a reasoning model.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports("supports_reasoning", model, custom_llm_provider)


def supports_pdf_input(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model accepts file input such as PDFs.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports("supports_pdf_input", model, custom_llm_provider)


def supports_audio_input(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model accepts audio input.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_audio_input", model, custom_llm_provider
    )


def supports_audio_output(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model can produce audio.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_audio_output", model, custom_llm_provider
    )


def supports_web_search(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model has built-in web search.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True when supported.
    """
    return _supports(
        "supports_web_search", model, custom_llm_provider
    )


def supports_computer_use(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model supports computer use; OpenRouter does not report it.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True only when a registered entry says so.
    """
    return _supports(
        "supports_computer_use", model, custom_llm_provider
    )


def supports_assistant_prefill(
    model: str, custom_llm_provider: Optional[str] = None
) -> bool:
    """Whether the model continues a prefilled assistant message; OpenRouter does not report it.

    Args:
        model (str): Model name.
        custom_llm_provider (Optional[str]): Explicit provider.

    Returns:
        bool: True only when a registered entry says so.
    """
    return _supports(
        "supports_assistant_prefill", model, custom_llm_provider
    )
