"""Model-string resolution into provider, key and base URL."""

import pytest

import routehub as rh
from routehub.providers import PROVIDERS, split_model


@pytest.mark.parametrize(
    "model, provider, bare",
    [
        ("gpt-5.4", "openai", "gpt-5.4"),
        ("o3-mini", "openai", "o3-mini"),
        (
            "text-embedding-3-small",
            "openai",
            "text-embedding-3-small",
        ),
        ("claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("gemini-2.5-pro", "gemini", "gemini-2.5-pro"),
        ("grok-4", "xai", "grok-4"),
        ("deepseek-chat", "deepseek", "deepseek-chat"),
        ("mistral-large-latest", "mistral", "mistral-large-latest"),
        ("openai/gpt-4o", "openai", "gpt-4o"),
        (
            "anthropic/claude-sonnet-4-6",
            "anthropic",
            "claude-sonnet-4-6",
        ),
        (
            "groq/llama-3.3-70b-versatile",
            "groq",
            "llama-3.3-70b-versatile",
        ),
        (
            "openrouter/anthropic/claude-sonnet-4.6",
            "openrouter",
            "anthropic/claude-sonnet-4.6",
        ),
        (
            "together_ai/meta-llama/Llama-3.3-70B",
            "together_ai",
            "meta-llama/Llama-3.3-70B",
        ),
        (
            "together/meta-llama/Llama-3.3-70B",
            "together_ai",
            "meta-llama/Llama-3.3-70B",
        ),
        ("x-ai/grok-4", "xai", "grok-4"),
        ("ollama/llama3", "ollama", "llama3"),
    ],
)
def test_split_model(model, provider, bare):
    assert split_model(model) == (bare, provider)


def test_unknown_bare_model_has_no_provider():
    assert split_model("mystery-model") == ("mystery-model", None)


def test_custom_llm_provider_wins():
    assert split_model("my-model", "groq") == ("my-model", "groq")
    assert split_model("groq/my-model", "groq") == (
        "my-model",
        "groq",
    )


def test_keys_and_bases_come_from_the_environment():
    bare, provider, key, base = rh.get_llm_provider(
        "groq/llama-3.3-70b-versatile"
    )
    assert (bare, provider) == ("llama-3.3-70b-versatile", "groq")
    assert key == "test-groq_api_key"
    assert base == "https://api.groq.com/openai/v1"


def test_explicit_key_and_base_win(monkeypatch):
    monkeypatch.setenv("GROQ_API_BASE", "https://env.local/v1")
    _, _, key, base = rh.get_llm_provider(
        "groq/x", api_key="explicit", api_base="https://arg.local/v1"
    )
    assert key == "explicit" and base == "https://arg.local/v1"
    _, _, _, base = rh.get_llm_provider("groq/x")
    assert base == "https://env.local/v1"


def test_openai_uses_the_sdk_default_base():
    _, provider, key, base = rh.get_llm_provider("gpt-4o")
    assert (
        provider == "openai"
        and key == "test-openai_api_key"
        and base is None
    )


def test_fallback_key_variables(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert (
        rh.get_llm_provider("gemini/gemini-2.5-pro")[2]
        == "google-key"
    )


def test_cohere_uses_the_compatibility_endpoint(monkeypatch):
    monkeypatch.setenv("CO_API_KEY", "co-key")
    for model in (
        "cohere/command-a-03-2025",
        "cohere_chat/command-a-03-2025",
        "command-a-03-2025",
    ):
        assert rh.get_llm_provider(model) == (
            "command-a-03-2025",
            "cohere",
            "co-key",
            "https://api.cohere.ai/compatibility/v1",
        )
    monkeypatch.setenv("COHERE_API_KEY", "cohere-key")
    assert rh.get_llm_provider("command-r")[2] == "cohere-key"


def test_local_servers_get_a_placeholder_key_and_v1_suffix(
    monkeypatch,
):
    _, _, key, base = rh.get_llm_provider("ollama/llama3")
    assert key == "ollama" and base == "http://localhost:11434/v1"
    monkeypatch.setenv("OLLAMA_API_BASE", "http://gpu-box:11434")
    assert (
        rh.get_llm_provider("ollama/llama3")[3]
        == "http://gpu-box:11434/v1"
    )


def test_explicit_api_base_gets_the_same_v1_suffix_as_the_env_var(
    monkeypatch,
):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://gpu-box:11434")
    from_env = rh.get_llm_provider("ollama/llama3")[3]
    monkeypatch.delenv("OLLAMA_API_BASE")
    explicit = rh.get_llm_provider(
        "ollama/llama3", api_base="http://gpu-box:11434"
    )[3]
    assert explicit == from_env == "http://gpu-box:11434/v1"
    assert (
        rh.get_llm_provider(
            "ollama/llama3", api_base="http://gpu-box:11434/v1/"
        )[3]
        == "http://gpu-box:11434/v1/"
    )


def test_api_base_without_a_provider_means_openai_compatible():
    bare, provider, _, base = rh.get_llm_provider(
        "my-finetune", api_base="http://vllm.local/v1"
    )
    assert (
        provider == "openai_like"
        and bare == "my-finetune"
        and base == "http://vllm.local/v1"
    )


def test_unknown_provider_raises():
    with pytest.raises(rh.BadRequestError, match="not supported"):
        rh.get_llm_provider("mystery-model")
    with pytest.raises(rh.BadRequestError):
        rh.get_llm_provider("x", custom_llm_provider="bedrock")


def test_only_anthropic_is_native():
    assert [
        name for name, spec in PROVIDERS.items() if spec.native
    ] == ["anthropic"]
