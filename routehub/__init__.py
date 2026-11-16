"""RouteHub: a fast, lightweight LLM gateway with a litellm-compatible API."""

import importlib
from typing import TYPE_CHECKING

_LAZY = {
    "completion": "routehub.main",
    "acompletion": "routehub.main",
    "embedding": "routehub.main",
    "aembedding": "routehub.main",
    "responses": "routehub.main",
    "aresponses": "routehub.main",
    "batched_completions": "routehub.batches",
    "abatched_completions": "routehub.batches",
    "BatchJob": "routehub.batches",
    "get_reasoning_efforts": "routehub.main",
    "REASONING_EFFORTS": "routehub.main",
    "get_llm_provider": "routehub.providers",
    "PROVIDERS": "routehub.providers",
    "get_model_info": "routehub.registry",
    "get_max_tokens": "routehub.registry",
    "register_model": "routehub.registry",
    "model_cost": "routehub.registry",
    "model_list": "routehub.registry",
    "clear_model_cache": "routehub.registry",
    "supports_vision": "routehub.registry",
    "supports_reasoning": "routehub.registry",
    "supports_function_calling": "routehub.registry",
    "supports_parallel_function_calling": "routehub.registry",
    "supports_response_schema": "routehub.registry",
    "supports_prompt_caching": "routehub.registry",
    "supports_tool_choice": "routehub.registry",
    "supports_system_messages": "routehub.registry",
    "supports_pdf_input": "routehub.registry",
    "supports_audio_input": "routehub.registry",
    "supports_web_search": "routehub.registry",
    "supports_audio_output": "routehub.registry",
    "supports_computer_use": "routehub.registry",
    "supports_assistant_prefill": "routehub.registry",
    "encode": "routehub.tokenizer",
    "decode": "routehub.tokenizer",
    "token_counter": "routehub.tokenizer",
    "APIError": "routehub.exceptions",
    "APIConnectionError": "routehub.exceptions",
    "AuthenticationError": "routehub.exceptions",
    "BadRequestError": "routehub.exceptions",
    "ConflictError": "routehub.exceptions",
    "ContentPolicyViolationError": "routehub.exceptions",
    "ContextWindowExceededError": "routehub.exceptions",
    "InternalServerError": "routehub.exceptions",
    "ModelNotMappedError": "routehub.registry",
    "NotFoundError": "routehub.exceptions",
    "OpenAIError": "routehub.exceptions",
    "PermissionDeniedError": "routehub.exceptions",
    "RateLimitError": "routehub.exceptions",
    "ServiceUnavailableError": "routehub.exceptions",
    "Timeout": "routehub.exceptions",
    "UnprocessableEntityError": "routehub.exceptions",
    "UnsupportedParamsError": "routehub.exceptions",
}

_UNCACHED = frozenset({"model_list", "model_cost"})

__all__ = sorted(_LAZY)

if TYPE_CHECKING:  # pragma: no cover
    from routehub.exceptions import *  # noqa: F401,F403
    from routehub.main import (  # noqa: F401
        REASONING_EFFORTS,
        acompletion,
        aembedding,
        completion,
        embedding,
        get_reasoning_efforts,
    )
    from routehub.providers import (
        PROVIDERS,
        get_llm_provider,
    )  # noqa: F401
    from routehub.registry import *  # noqa: F401,F403
    from routehub.tokenizer import (
        decode,
        encode,
        token_counter,
    )  # noqa: F401


def _version() -> str:
    """Read the installed version from the package metadata.

    The version lives in pyproject.toml only, so the attribute can never
    drift from the released package.

    Returns:
        str: The version string, or "0.0.0" when RouteHub is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("routehub")
    except PackageNotFoundError:
        found = "0.0.0"
    globals()["__version__"] = found
    return found


def __getattr__(name: str):
    """Load a public name from its submodule on first access.

    Args:
        name (str): The attribute being looked up.

    Returns:
        Any: The requested function, class or value.
    """
    if name == "__version__":
        return _version()
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(
            f"module 'routehub' has no attribute {name!r}"
        )
    value = getattr(importlib.import_module(module), name)
    # model_list and model_cost refresh with the registry, so they are not pinned.
    if name not in _UNCACHED:
        globals()[name] = value
    return value


def __dir__() -> list:
    """List the module's public names, including lazy ones.

    Returns:
        list: Attribute names.
    """
    return sorted(set(globals()) | set(_LAZY) | {"__version__"})
