"""Listing models from every provider's API."""

import pytest

from routehub import get_all_models as gam
from routehub._http import httpx

ROUTES = {
    "openrouter.ai/api/v1/models": {
        "data": [
            {
                "id": "anthropic/claude-sonnet-4.6",
                "name": "Anthropic: Claude Sonnet 4.6",
                "context_length": 1000000,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                },
                "top_provider": {"max_completion_tokens": 128000},
                "supported_parameters": [
                    "tools",
                    "reasoning",
                    "structured_outputs",
                ],
            }
        ]
    },
    "openrouter.ai/api/v1/models/anthropic/claude-sonnet-4.6/endpoints": {
        "data": {
            "id": "anthropic/claude-sonnet-4.6",
            "name": "Anthropic: Claude Sonnet 4.6",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "endpoints": [
                {
                    "provider_name": "Anthropic",
                    "context_length": 200000,
                    "max_completion_tokens": 64000,
                    "pricing": {
                        "prompt": "0.000003",
                        "completion": "0.000015",
                    },
                    "supported_parameters": ["tools"],
                },
                {
                    "provider_name": "Amazon Bedrock",
                    "context_length": 1000000,
                    "max_completion_tokens": 128000,
                    "pricing": {"prompt": "0.000004"},
                    "supported_parameters": ["reasoning"],
                },
            ],
        }
    },
    "api.openai.com/v1/models": {
        "object": "list",
        "data": [
            {"id": "gpt-5.4-mini", "created": 1},
            {"id": "text-embedding-3-small", "created": 2},
            {"id": "whisper-1"},
        ],
    },
    "api.openai.com/v1/models/gpt-5.4-mini": {
        "id": "gpt-5.4-mini",
        "created": 1,
        "owned_by": "system",
    },
    "api.anthropic.com/v1/models": {
        "data": [
            {
                "type": "model",
                "id": "claude-sonnet-4-6",
                "display_name": "Claude Sonnet 4.6",
                "created_at": "2026-02-17T00:00:00Z",
                "max_input_tokens": 1000000,
                "max_tokens": 128000,
                "capabilities": {
                    "image_input": {"supported": True},
                    "pdf_input": {"supported": True},
                    "structured_outputs": {"supported": True},
                    "thinking": {"supported": True},
                },
            }
        ],
        "has_more": False,
        "last_id": "claude-sonnet-4-6",
    },
    "generativelanguage.googleapis.com/v1beta/models": {
        "models": [
            {
                "name": "models/gemini-3-flash-preview",
                "displayName": "Gemini 3 Flash",
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 65536,
                "supportedGenerationMethods": ["generateContent"],
                "thinking": True,
            },
            {
                "name": "models/gemini-embedding-001",
                "supportedGenerationMethods": ["embedContent"],
                "inputTokenLimit": 2048,
            },
        ]
    },
    "api.groq.com/openai/v1/models": {
        "data": [
            {
                "id": "llama-3.3-70b-versatile",
                "context_window": 131072,
                "max_completion_tokens": 32768,
                "active": True,
            },
            {"id": "retired-model", "active": False},
        ]
    },
    "api.deepseek.com/models": {
        "data": [
            {
                "id": "deepseek-flash",
                "name": "DeepSeek Flash",
                "context_window": 1048576,
                "max_output_tokens": 393216,
                "input_modalities": ["text", "image"],
                "effort": {"supported_levels": ["low"]},
            }
        ]
    },
    "api.mistral.ai/v1/models": {
        "data": [
            {
                "id": "mistral-large-latest",
                "max_context_length": 131072,
                "capabilities": {
                    "completion_chat": True,
                    "function_calling": True,
                    "vision": False,
                },
            },
            {
                "id": "mistral-large-latest",
                "max_context_length": 131072,
            },
        ]
    },
    "api.together.xyz/v1/models": [
        {
            "id": "meta-llama/Llama-3.3-70B",
            "type": "chat",
            "display_name": "Llama 3.3 70B",
            "context_length": 131072,
            "pricing": {"input": 0.88, "output": 0.88},
        }
    ],
    "api.cohere.com/v1/models": {
        "models": [
            {
                "name": "command-a-plus-05-2026",
                "endpoints": ["generate", "chat"],
                "context_length": 128000,
                "features": [
                    "json_schema",
                    "tools",
                    "reasoning",
                    "vision",
                ],
            },
            {
                "name": "embed-english-v3.0-image",
                "endpoints": ["embed_image"],
                "context_length": 0,
                "features": None,
            },
            {
                "name": "rerank-v3.5",
                "endpoints": ["rerank"],
                "context_length": 4096,
                "features": None,
            },
        ]
    },
}


@pytest.fixture
def provider_api(monkeypatch):
    """Route every provider request to ROUTES and record what was asked."""
    calls = []

    def handler(request):
        path = f"{request.url.host}{request.url.path}"
        calls.append(
            (path, dict(request.headers), dict(request.url.params))
        )
        if (
            path == "generativelanguage.googleapis.com/v1beta/models"
            and "pageToken" not in request.url.params
        ):
            body = dict(ROUTES[path])
            first, second = body["models"]
            return httpx.Response(
                200, json={"models": [first], "nextPageToken": "p2"}
            )
        if path == "generativelanguage.googleapis.com/v1beta/models":
            return httpx.Response(
                200, json={"models": [ROUTES[path]["models"][1]]}
            )
        if path in ROUTES:
            return httpx.Response(200, json=ROUTES[path])
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    real_client, real_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **k: real_client(*a, transport=transport, **k),
    )
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real_async(*a, transport=transport, **k),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **k: real_client(transport=transport).get(
            url,
            **{x: v for x, v in k.items() if x != "follow_redirects"},
        ),
    )
    for name in (
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.setenv(name, f"test-{name.lower()}")
    return calls


def by_id(entries):
    return {e["id"]: e for e in entries}


class TestParsers:
    def test_every_entry_has_the_common_keys(self, provider_api):
        keys = set(gam._entry("p", "m", {}))
        for entry in gam.get_all_models():
            assert set(entry) == keys

    def test_anthropic_capabilities(self, provider_api):
        entry = by_id(gam.get_models("anthropic"))[
            "anthropic/claude-sonnet-4-6"
        ]
        assert entry["name"] == "Claude Sonnet 4.6"
        assert entry["context_window"] == 1000000
        assert entry["max_output_tokens"] == 128000
        assert entry["supports_vision"] is True
        assert entry["supports_reasoning"] is True
        assert entry["supports_structured_output"] is True
        assert entry["input_modalities"] == ["text", "image", "file"]
        assert entry["created"] == 1771286400

    def test_gemini_types_limits_and_pagination(self, provider_api):
        entries = by_id(gam.get_models("gemini"))
        assert (
            entries["gemini/gemini-3-flash-preview"]["context_window"]
            == 1048576
        )
        assert (
            entries["gemini/gemini-3-flash-preview"][
                "supports_reasoning"
            ]
            is True
        )
        assert (
            entries["gemini/gemini-embedding-001"]["type"]
            == "embedding"
        )
        gemini_calls = [
            c
            for c in provider_api
            if c[0].startswith("generativelanguage")
        ]
        assert len(gemini_calls) == 2
        assert gemini_calls[1][2]["pageToken"] == "p2"

    def test_openai_ids_and_guessed_types(self, provider_api):
        entries = by_id(gam.get_models("openai"))
        assert entries["openai/gpt-5.4-mini"]["type"] == "chat"
        assert (
            entries["openai/text-embedding-3-small"]["type"]
            == "embedding"
        )
        assert entries["openai/whisper-1"]["type"] == "audio"
        assert (
            entries["openai/gpt-5.4-mini"]["context_window"] is None
        )

    def test_groq_skips_inactive_models(self, provider_api):
        entries = by_id(gam.get_models("groq"))
        assert list(entries) == ["groq/llama-3.3-70b-versatile"]
        assert (
            entries["groq/llama-3.3-70b-versatile"][
                "max_output_tokens"
            ]
            == 32768
        )

    def test_deepseek_mistral_together(self, provider_api):
        deepseek = by_id(gam.get_models("deepseek"))[
            "deepseek/deepseek-flash"
        ]
        assert (
            deepseek["max_output_tokens"] == 393216
            and deepseek["supports_vision"] is True
            and deepseek["supports_reasoning"] is True
        )
        mistral = gam.get_models("mistral")
        assert (
            len(mistral) == 1
            and mistral[0]["supports_function_calling"] is True
            and mistral[0]["context_window"] == 131072
        )
        together = by_id(gam.get_models("together_ai"))[
            "together_ai/meta-llama/Llama-3.3-70B"
        ]
        assert together["input_cost_per_token"] == pytest.approx(
            0.88e-6
        )

    def test_cohere_types_and_features(
        self, provider_api, monkeypatch
    ):
        monkeypatch.setenv("CO_API_KEY", "test-co")
        entries = by_id(gam.get_models("cohere"))
        plus = entries["cohere/command-a-plus-05-2026"]
        assert plus["type"] == "chat"
        assert plus["context_window"] == 128000
        assert plus["supports_function_calling"] is True
        assert plus["supports_vision"] is True
        assert plus["supports_reasoning"] is True
        assert plus["supports_structured_output"] is True
        embed = entries["cohere/embed-english-v3.0-image"]
        assert embed["type"] == "embedding"
        assert embed["context_window"] is None
        assert embed["supports_function_calling"] is False
        assert entries["cohere/rerank-v3.5"]["type"] == "rerank"
        assert (
            provider_api[-1][1]["authorization"] == "Bearer test-co"
        )
        assert provider_api[-1][2]["page_size"] == "1000"

    def test_openrouter_needs_no_key(self, provider_api, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        entry = gam.get_models("openrouter")[0]
        assert entry["id"] == "openrouter/anthropic/claude-sonnet-4.6"
        assert entry["supports_reasoning"] is True and entry[
            "input_cost_per_token"
        ] == pytest.approx(0.000003)
        assert "authorization" not in provider_api[-1][1]


class TestAuth:
    def test_each_provider_gets_its_auth_header(self, provider_api):
        gam.get_all_models(["openai", "anthropic", "gemini"])
        headers = {path: h for path, h, _ in provider_api}
        assert (
            headers["api.openai.com/v1/models"]["authorization"]
            == "Bearer test-openai_api_key"
        )
        assert (
            headers["api.anthropic.com/v1/models"]["x-api-key"]
            == "test-anthropic_api_key"
        )
        assert (
            headers["api.anthropic.com/v1/models"][
                "anthropic-version"
            ]
            == "2023-06-01"
        )
        assert (
            headers[
                "generativelanguage.googleapis.com/v1beta/models"
            ]["x-goog-api-key"]
            == "test-gemini_api_key"
        )

    def test_explicit_api_key(self, provider_api):
        gam.get_models("openai", api_key="explicit")
        assert (
            provider_api[-1][1]["authorization"] == "Bearer explicit"
        )


class TestGetAllModels:
    def test_lists_every_configured_provider(self, provider_api):
        providers = {e["provider"] for e in gam.get_all_models()}
        assert providers == {
            "openrouter",
            "openai",
            "anthropic",
            "gemini",
            "groq",
            "deepseek",
            "mistral",
            "together_ai",
        }

    def test_skips_providers_without_keys(
        self, provider_api, monkeypatch
    ):
        monkeypatch.delenv("GROQ_API_KEY")
        assert "groq" not in {
            e["provider"] for e in gam.get_all_models()
        }
        assert not any(
            c[0].startswith("api.groq.com") for c in provider_api
        )

    def test_ollama_only_when_configured_or_named(self, provider_api):
        assert "ollama" not in gam.configured_providers()

    def test_selected_providers_in_order(self, provider_api):
        entries = gam.get_all_models(["anthropic", "openai"])
        assert [e["provider"] for e in entries][0] == "anthropic"
        assert {e["provider"] for e in entries} == {
            "anthropic",
            "openai",
        }

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="No model list"):
            gam.get_models("nope")

    def test_a_failing_provider_does_not_break_the_rest(
        self, provider_api, monkeypatch
    ):
        monkeypatch.setenv("XAI_API_KEY", "bad")
        entries = gam.get_all_models(["xai", "openai"])
        assert {e["provider"] for e in entries} == {"openai"}

    async def test_async_matches_sync(self, provider_api):
        sync_ids = [e["id"] for e in gam.get_all_models()]
        gam.clear_cache()
        async_ids = [e["id"] for e in await gam.aget_all_models()]
        assert async_ids == sync_ids
        gam.clear_cache()
        assert [
            e["id"] for e in await gam.aget_models("anthropic")
        ] == ["anthropic/claude-sonnet-4-6"]


class TestCaching:
    def test_results_are_cached_per_provider(self, provider_api):
        gam.get_all_models()
        first = len(provider_api)
        gam.get_all_models()
        gam.get_models("anthropic")
        assert len(provider_api) == first

    def test_failures_are_cached_too(self, provider_api, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "bad")
        gam.get_models("xai")
        gam.get_models("xai")
        assert (
            len(
                [
                    c
                    for c in provider_api
                    if c[0].startswith("api.x.ai")
                ]
            )
            == 1
        )

    def test_refresh_and_clear_cache_refetch(self, provider_api):
        gam.get_models("openai")
        gam.get_models("openai", refresh=True)
        gam.clear_cache()
        gam.get_models("openai")
        assert (
            len(
                [
                    c
                    for c in provider_api
                    if c[0] == "api.openai.com/v1/models"
                ]
            )
            == 3
        )

    def test_expired_entries_refetch(self, provider_api, monkeypatch):
        gam.get_models("openai")
        monkeypatch.setattr(gam, "CACHE_TTL_SECONDS", -1)
        gam.clear_cache()
        gam.get_models("openai")
        gam.get_models("openai")
        assert (
            len(
                [
                    c
                    for c in provider_api
                    if c[0] == "api.openai.com/v1/models"
                ]
            )
            == 3
        )


class TestGetModel:
    def test_per_model_endpoint(self, provider_api):
        entry = gam.get_model("openai", "gpt-5.4-mini")
        assert entry["id"] == "openai/gpt-5.4-mini"
        assert (
            provider_api[-1][0]
            == "api.openai.com/v1/models/gpt-5.4-mini"
        )

    def test_a_fetched_model_is_cached(self, provider_api):
        first = gam.get_model("openai", "gpt-5.4-mini")
        calls = len(provider_api)

        assert gam.get_model("openai", "gpt-5.4-mini") == first
        assert len(provider_api) == calls

    def test_openrouter_includes_every_endpoint(self, provider_api):
        entry = gam.get_model(
            "openrouter", "anthropic/claude-sonnet-4.6"
        )
        assert len(entry["endpoints"]) == 2
        assert entry["context_window"] == 1000000
        assert entry["max_output_tokens"] == 128000
        assert (
            entry["supports_function_calling"] is True
            and entry["supports_reasoning"] is True
        )

    def test_falls_back_to_the_list_without_a_detail_endpoint(
        self, provider_api
    ):
        entry = gam.get_model(
            "together_ai", "meta-llama/Llama-3.3-70B"
        )
        assert entry["context_window"] == 131072

    def test_unknown_model_is_none(self, provider_api):
        assert gam.get_model("openai", "no-such-model") is None
        assert gam.get_model("together_ai", "no-such-model") is None
