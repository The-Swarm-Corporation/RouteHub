"""RouteHub: a fast, lightweight LLM gateway with a litellm-compatible API."""

import importlib
import os
import sys
import threading
from typing import TYPE_CHECKING, Optional

__version__ = "0.1.0"

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

__all__ = sorted([*_LAZY, "preload"])

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


_preload_thread: Optional[threading.Thread] = None


def preload() -> Optional[threading.Thread]:
    """Start importing the OpenAI SDK on a daemon thread.

    Opt-in, also enabled by ROUTEHUB_PRELOAD=1. It only helps when the main
    thread has other work to overlap with (building agents, reading config,
    network I/O); pure CPU work competes for the GIL. completion and
    acompletion wait for the thread to finish, so the SDK is imported once.
    Code outside routehub that imports openai submodules meanwhile should
    join the returned thread first: a concurrent submodule import can see a
    partially initialized package. An interpreter that exits mid-import
    abandons it.

    Returns:
        Optional[threading.Thread]: The import thread, or None when the SDK
        is already imported.
    """
    global _preload_thread
    if _preload_thread is not None and _preload_thread.is_alive():
        return _preload_thread
    if "openai" in sys.modules:
        return None
    _preload_thread = threading.Thread(
        target=importlib.import_module,
        args=("openai",),
        name="routehub-preload",
        daemon=True,
    )
    _preload_thread.start()
    return _preload_thread


def __getattr__(name: str):
    """Load a public name from its submodule on first access.

    Args:
        name (str): The attribute being looked up.

    Returns:
        Any: The requested function, class or value.
    """
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
    return sorted(set(globals()) | set(_LAZY))


if os.environ.get("ROUTEHUB_PRELOAD") == "1":
    preload()
