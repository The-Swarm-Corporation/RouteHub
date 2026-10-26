"""Provider table and model-string resolution."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Provider:
    """How to reach one provider.

    Attributes:
        name (str): The provider prefix used in model strings.
        base_url (Optional[str]): Default API base URL.
        key_env (Tuple[str, ...]): Environment variables holding the API key.
        base_env (Tuple[str, ...]): Environment variables overriding the base URL.
        native (bool): True when the provider uses a native adapter instead of
            the OpenAI-compatible endpoint.
        default_key (Optional[str]): Key to send when none is configured, for
            local servers that ignore it.
        base_suffix (str): Path appended to a base URL from api_base or
            base_env when it is missing.
    """

    name: str
    base_url: Optional[str]
    key_env: Tuple[str, ...] = ()
    base_env: Tuple[str, ...] = ()
    native: bool = False
    default_key: Optional[str] = None
    base_suffix: str = ""


PROVIDERS = {
    p.name: p
    for p in (
        Provider(
            "openai",
            None,
            ("OPENAI_API_KEY",),
            ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
        ),
        Provider(
            "azure",
            None,
            ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
            ("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT"),
        ),
        Provider(
            "anthropic",
            "https://api.anthropic.com",
            ("ANTHROPIC_API_KEY",),
            ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"),
            native=True,
        ),
        Provider(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            ("GEMINI_API_BASE",),
        ),
        Provider(
            "groq",
            "https://api.groq.com/openai/v1",
            ("GROQ_API_KEY",),
            ("GROQ_API_BASE",),
        ),
        Provider(
            "xai",
            "https://api.x.ai/v1",
            ("XAI_API_KEY",),
            ("XAI_API_BASE",),
        ),
        Provider(
            "deepseek",
            "https://api.deepseek.com/v1",
            ("DEEPSEEK_API_KEY",),
            ("DEEPSEEK_API_BASE",),
        ),
        Provider(
            "openrouter",
            "https://openrouter.ai/api/v1",
            ("OPENROUTER_API_KEY", "OR_API_KEY"),
            ("OPENROUTER_API_BASE",),
        ),
        Provider(
            "together_ai",
            "https://api.together.xyz/v1",
            (
                "TOGETHERAI_API_KEY",
                "TOGETHER_API_KEY",
                "TOGETHER_AI_TOKEN",
            ),
            ("TOGETHERAI_API_BASE",),
        ),
        Provider(
            "mistral",
            "https://api.mistral.ai/v1",
            ("MISTRAL_API_KEY",),
            ("MISTRAL_API_BASE",),
        ),
        Provider(
            "cohere",
            "https://api.cohere.ai/compatibility/v1",
            ("COHERE_API_KEY", "CO_API_KEY"),
        ),
        Provider(
            "fireworks_ai",
            "https://api.fireworks.ai/inference/v1",
            ("FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY"),
            ("FIREWORKS_API_BASE",),
        ),
        Provider(
            "perplexity",
            "https://api.perplexity.ai",
            ("PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY"),
        ),
        Provider(
            "cerebras",
            "https://api.cerebras.ai/v1",
            ("CEREBRAS_API_KEY",),
        ),
        Provider(
            "deepinfra",
            "https://api.deepinfra.com/v1/openai",
            ("DEEPINFRA_API_KEY",),
        ),
        Provider(
            "sambanova",
            "https://api.sambanova.ai/v1",
            ("SAMBANOVA_API_KEY",),
        ),
        Provider(
            "nvidia_nim",
            "https://integrate.api.nvidia.com/v1",
            ("NVIDIA_NIM_API_KEY",),
            ("NVIDIA_NIM_API_BASE",),
        ),
        Provider(
            "moonshot",
            "https://api.moonshot.ai/v1",
            ("MOONSHOT_API_KEY",),
        ),
        Provider(
            "dashscope",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            ("DASHSCOPE_API_KEY",),
        ),
        Provider(
            "huggingface",
            "https://router.huggingface.co/v1",
            ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
        ),
        Provider(
            "ollama",
            "http://localhost:11434/v1",
            (),
            ("OLLAMA_API_BASE",),
            default_key="ollama",
            base_suffix="/v1",
        ),
        Provider(
            "ollama_chat",
            "http://localhost:11434/v1",
            (),
            ("OLLAMA_API_BASE",),
            default_key="ollama",
            base_suffix="/v1",
        ),
        Provider(
            "lm_studio",
            "http://localhost:1234/v1",
            ("LM_STUDIO_API_KEY",),
            ("LM_STUDIO_API_BASE",),
            default_key="lm-studio",
        ),
        Provider(
            "hosted_vllm",
            None,
            ("HOSTED_VLLM_API_KEY",),
            ("HOSTED_VLLM_API_BASE",),
            default_key="EMPTY",
        ),
        Provider(
            "openai_like",
            None,
            ("OPENAI_LIKE_API_KEY",),
            ("OPENAI_LIKE_API_BASE",),
            default_key="EMPTY",
        ),
    )
}

# Aliases litellm accepts for the same endpoints.
_ALIASES = {
    "custom_openai": "openai_like",
    "text-completion-openai": "openai",
    "together": "together_ai",
    "fireworks": "fireworks_ai",
    "nvidia": "nvidia_nim",
    "google": "gemini",
    "google_ai_studio": "gemini",
    "x-ai": "xai",
    "vllm": "hosted_vllm",
    "cohere_chat": "cohere",
}

_PREFIX_PROVIDERS = (
    (
        (
            "gpt-",
            "o1",
            "o3",
            "o4",
            "chatgpt-",
            "text-embedding-",
            "davinci",
            "babbage",
            "omni-moderation",
            "codex-",
        ),
        "openai",
    ),
    (("claude-",), "anthropic"),
    (("gemini-", "gemma-"), "gemini"),
    (("grok-",), "xai"),
    (("deepseek-",), "deepseek"),
    (
        (
            "mistral-",
            "codestral",
            "magistral-",
            "ministral-",
            "pixtral-",
            "devstral-",
        ),
        "mistral",
    ),
    (("command-",), "cohere"),
)


def _canonical(name: str) -> str:
    """Resolve a provider alias to its canonical name.

    Args:
        name (str): Provider name or alias.

    Returns:
        str: The canonical provider name.
    """
    return _ALIASES.get(name, name)


def _guess_provider(model: str) -> Optional[str]:
    """Infer the provider of a bare model name.

    Args:
        model (str): Model name without a provider prefix.

    Returns:
        Optional[str]: The provider, or None when it cannot be inferred.
    """
    lowered = model.lower()
    for prefixes, provider in _PREFIX_PROVIDERS:
        if lowered.startswith(prefixes):
            return provider
    return None


def _first_env(names: Tuple[str, ...]) -> Optional[str]:
    """Return the first set environment variable among names.

    Args:
        names (Tuple[str, ...]): Variable names in priority order.

    Returns:
        Optional[str]: The value, or None when none are set.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def split_model(
    model: str, custom_llm_provider: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Split a model string into its bare model name and provider.

    Args:
        model (str): A model string such as "groq/llama-3.3-70b-versatile".
        custom_llm_provider (Optional[str]): Explicit provider override.

    Returns:
        Tuple[str, Optional[str]]: The bare model name and the provider, or
        None when the provider cannot be determined.
    """
    if custom_llm_provider:
        provider = _canonical(custom_llm_provider)
        prefix = f"{custom_llm_provider}/"
        if model.startswith(prefix):
            model = model[len(prefix) :]
        return model, provider
    if "/" in model:
        prefix, rest = model.split("/", 1)
        canonical = _canonical(prefix)
        if canonical in PROVIDERS:
            return rest, canonical
    return model, _guess_provider(model)


def get_llm_provider(
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Resolve a model string to its provider, API key and base URL.

    Args:
        model (str): A model string such as "anthropic/claude-sonnet-4-6".
        custom_llm_provider (Optional[str]): Explicit provider override.
        api_base (Optional[str]): Explicit base URL, which wins over env vars.
        api_key (Optional[str]): Explicit API key, which wins over env vars.

    Returns:
        Tuple[str, str, Optional[str], Optional[str]]: The bare model name,
        provider, API key and base URL.
    """
    bare, provider = split_model(model, custom_llm_provider)
    if provider is None and api_base:
        provider = "openai_like"
    if provider is None or provider not in PROVIDERS:
        # Imported here: exceptions loads the OpenAI SDK, which lookups never need.
        from routehub.exceptions import BadRequestError

        raise BadRequestError(
            f"LLM provider not provided or not supported for model={model!r}. "
            "Prefix the model with its provider, e.g. 'groq/llama-3.3-70b-versatile', "
            f"or pass custom_llm_provider. Supported: {', '.join(sorted(PROVIDERS))}",
            llm_provider=str(provider or ""),
            model=model,
        )
    spec = PROVIDERS[provider]
    key = api_key or _first_env(spec.key_env) or spec.default_key
    base = api_base or _first_env(spec.base_env)
    if (
        base
        and spec.base_suffix
        and not base.rstrip("/").endswith(spec.base_suffix)
    ):
        base = base.rstrip("/") + spec.base_suffix
    base = base or spec.base_url
    return bare, provider, key, base
