"""Model metadata from the OpenRouter catalog."""

import os

import pytest
from conftest import CATALOG

import routehub as rh
from routehub import registry
from routehub._http import httpx

# Captured at collection, before the autouse fixture swaps in the fake catalog.
REAL_FETCH = registry._fetch


class TestLookups:
    def test_maps_openrouter_fields(self):
        info = rh.get_model_info("gpt-5.4-mini")
        assert info["key"] == "openai/gpt-5.4-mini"
        assert info["provider"] == "openai"
        assert info["litellm_provider"] == "openai"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 400000
        assert info["max_output_tokens"] == 128000
        assert info["input_cost_per_token"] == pytest.approx(
            0.00000075
        )
        assert info["cache_read_input_token_cost"] == pytest.approx(
            0.000000075
        )
        assert info["supports_vision"] is True
        assert info["supports_pdf_input"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_function_calling"] is True
        assert info["supports_response_schema"] is True
        # OpenAI caches on its own; cache_control markers are not honoured.
        assert info["supports_prompt_caching"] is False

    @pytest.mark.parametrize(
        "name",
        [
            "claude-sonnet-4-6",
            "anthropic/claude-sonnet-4-6",
            "claude-sonnet-4-6-20260101",
            "anthropic/claude-sonnet-4.6",
            "openrouter/anthropic/claude-sonnet-4.6",
            "claude-sonnet-4.6",
        ],
    )
    def test_native_and_openrouter_names_resolve(self, name):
        assert (
            rh.get_model_info(name)["key"]
            == "anthropic/claude-sonnet-4.6"
        )

    def test_provider_prefix_maps_to_vendor(self):
        info = rh.get_model_info("gemini/gemini-2.5-pro")
        assert info["key"] == "google/gemini-2.5-pro"
        assert info["provider"] == "gemini"
        assert info["litellm_provider"] == "gemini"
        assert info["supports_audio_input"] is True

    def test_variants_need_their_full_id(self):
        assert (
            rh.get_model_info(
                "openrouter/anthropic/claude-sonnet-4.6:batch"
            )["key"]
            == "anthropic/claude-sonnet-4.6:batch"
        )
        assert (
            rh.get_model_info("claude-sonnet-4-6")["key"]
            == "anthropic/claude-sonnet-4.6"
        )

    def test_reasoning_false_when_not_reported(self):
        assert (
            rh.supports_reasoning("deepseek/deepseek-chat") is False
        )
        assert rh.supports_vision("deepseek/deepseek-chat") is False
        assert (
            rh.supports_function_calling("deepseek/deepseek-chat")
            is True
        )

    def test_unknown_model_raises_and_supports_is_false(self):
        with pytest.raises(rh.ModelNotMappedError):
            rh.get_model_info("no-such-model")
        with pytest.raises(rh.ModelNotMappedError):
            rh.get_max_tokens("no-such-model")
        assert rh.supports_vision("no-such-model") is False

    def test_get_max_tokens(self):
        assert rh.get_max_tokens("claude-sonnet-4-6") == 128000

    def test_utils_module_mirrors_litellm_utils(self):
        from routehub.utils import (
            get_model_info,
            supports_parallel_function_calling,
        )

        assert (
            get_model_info("gpt-5.4-mini")["max_input_tokens"]
            == 400000
        )
        assert (
            supports_parallel_function_calling("gpt-5.4-mini") is True
        )


class TestModelList:
    def test_contains_resolves_aliases(self):
        models = rh.model_list
        assert "openai/gpt-5.4-mini" in list(models)
        assert "claude-sonnet-4-6" in models
        assert "gpt-5.4-mini" in models
        assert "made-up-model" not in models

    def test_model_cost_is_keyed_by_openrouter_id(self):
        cost = rh.model_cost
        assert (
            cost["anthropic/claude-sonnet-4.6"]["max_output_tokens"]
            == 128000
        )
        assert set(cost) == {entry["id"] for entry in CATALOG} | set(
            registry._BUILTIN
        )


class TestCaching:
    def test_catalog_is_fetched_once_per_ttl(self, catalog):
        rh.get_model_info("gpt-5.4-mini")
        rh.supports_vision("claude-sonnet-4-6")
        list(rh.model_list)
        assert catalog["count"] == 1

    def test_clear_model_cache_refetches(self, catalog):
        rh.get_model_info("gpt-5.4-mini")
        rh.clear_model_cache()
        rh.get_model_info("gpt-5.4-mini")
        assert catalog["count"] == 2

    def test_expired_cache_refetches(self, catalog, monkeypatch):
        rh.get_model_info("gpt-5.4-mini")
        monkeypatch.setattr(registry, "_expires_at", 0.0)
        rh.get_model_info("gpt-5.4-mini")
        assert catalog["count"] == 2

    def test_failed_refresh_keeps_stale_data(self, monkeypatch):
        rh.get_model_info("gpt-5.4-mini")
        monkeypatch.setattr(registry, "_fetch", lambda: [])
        monkeypatch.setattr(registry, "_expires_at", 0.0)
        assert (
            rh.get_model_info("gpt-5.4-mini")["max_input_tokens"]
            == 400000
        )

    def test_failed_first_fetch_means_unknown(self, monkeypatch):
        monkeypatch.setattr(registry, "_fetch", lambda: [])
        assert rh.supports_vision("gpt-5.4-mini") is False

    def test_fetch_handles_network_errors(self, monkeypatch):
        def offline(*args, **kwargs):
            raise httpx.ConnectError("offline")

        monkeypatch.setattr(httpx, "get", offline)
        assert REAL_FETCH() == []

    def test_fetch_parses_the_openrouter_response(self, monkeypatch):
        def serve(url, **kwargs):
            assert url == registry.OPENROUTER_MODELS_URL
            return httpx.Response(
                200,
                json={"data": CATALOG},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx, "get", serve)
        assert [e["id"] for e in REAL_FETCH()] == [
            e["id"] for e in CATALOG
        ]


class TestServingProviders:
    """Provider-specific ids map onto the vendor's catalog entry."""

    @pytest.mark.parametrize(
        "name",
        [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.3-70b-8192",
            "cerebras/llama-3.3-70b",
            "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
            "cerebras/llama3.3-70b",
            "meta-llama/llama-3.3-70b-instruct",
        ],
    )
    def test_serving_ids_resolve_to_the_vendor_entry(self, name):
        info = rh.get_model_info(name)
        assert info["key"] == "meta-llama/llama-3.3-70b-instruct"
        assert info["max_output_tokens"] == 16384
        assert rh.get_max_tokens(name) == 16384
        assert name in rh.model_list

    def test_provider_names_the_serving_provider(self):
        info = rh.get_model_info("groq/llama-3.3-70b-versatile")
        assert info["provider"] == "groq"
        assert info["litellm_provider"] == "groq"
        assert (
            rh.get_model_info("meta-llama/llama-3.3-70b-instruct")[
                "provider"
            ]
            == "openrouter"
        )

    def test_capabilities_follow_the_vendor_entry(self):
        assert (
            rh.supports_function_calling(
                "groq/llama-3.3-70b-versatile"
            )
            is True
        )
        assert (
            rh.supports_vision("groq/llama-3.3-70b-versatile")
            is False
        )

    def test_preview_suffixes_fall_back_to_the_family(self):
        assert (
            rh.get_model_info("gemini/gemini-2.5-pro-preview-06-05")[
                "key"
            ]
            == "google/gemini-2.5-pro"
        )

    def test_vendor_prefix_is_not_crossed(self):
        with pytest.raises(rh.ModelNotMappedError):
            rh.get_model_info("openai/llama-3.3-70b-versatile")

    @pytest.mark.parametrize(
        "name",
        [
            "no-such-model",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral",
            "llama",
        ],
    )
    def test_nothing_close_still_raises(self, name):
        with pytest.raises(rh.ModelNotMappedError):
            rh.get_model_info(name)
        assert rh.supports_vision(name) is False

    def test_strip_serving_suffixes(self):
        strip = registry._strip_serving_suffixes
        assert strip("llama-3.3-70b-versatile") == "llama-3.3-70b"
        assert strip("llama-3-70b-8192") == "llama-3-70b"
        assert (
            strip("llama-3.1-70b-instruct-turbo") == "llama-3.1-70b"
        )
        assert strip("gemma2-9b-it") == "gemma2-9b"
        assert strip("mistral-large") == "mistral-large"


class TestEmbeddingModels:
    def test_builtin_embedding_models(self):
        info = rh.get_model_info("text-embedding-3-small")
        assert info["mode"] == "embedding"
        assert info["provider"] == "openai"
        assert info["max_input_tokens"] == 8191
        assert info["input_cost_per_token"] == pytest.approx(2e-8)
        assert info["output_vector_size"] == 1536
        assert "text-embedding-3-small" in rh.model_list
        assert "text-embedding-3-small" in rh.model_cost

    def test_prefixed_and_azure_names(self):
        assert (
            rh.get_model_info("openai/text-embedding-3-large")["key"]
            == "text-embedding-3-large"
        )
        info = rh.get_model_info("azure/text-embedding-3-large")
        assert info["provider"] == "azure"
        assert (
            rh.get_model_info("mistral-embed")["provider"]
            == "mistral"
        )

    def test_registration_overrides_a_builtin(self):
        rh.register_model(
            {"text-embedding-3-small": {"max_input_tokens": 10}}
        )
        assert (
            rh.get_model_info("text-embedding-3-small")[
                "max_input_tokens"
            ]
            == 10
        )

    def test_builtins_survive_an_empty_catalog(self, monkeypatch):
        monkeypatch.setattr(registry, "_fetch", lambda: [])
        assert rh.get_model_info("text-embedding-3-small")[
            "mode"
        ] == ("embedding")


class TestPromptCaching:
    @pytest.mark.parametrize(
        "name",
        [
            "claude-sonnet-4-6",
            "anthropic/claude-sonnet-4-6",
            "openrouter/anthropic/claude-sonnet-4.6",
        ],
    )
    def test_claude_honours_cache_control(self, name):
        assert rh.supports_prompt_caching(name) is True

    @pytest.mark.parametrize(
        "name", ["gemini/gemini-2.5-pro", "gpt-5.4-mini"]
    )
    def test_providers_with_their_own_caching_answer_false(
        self, name
    ):
        # Both list a cache-read price, but neither accepts cache_control.
        info = rh.get_model_info(name)
        assert info["cache_read_input_token_cost"] is not None
        assert info["supports_prompt_caching"] is False
        assert rh.supports_prompt_caching(name) is False

    def test_registered_entries_decide_for_themselves(self):
        rh.register_model(
            {"gemini/private": {"supports_prompt_caching": True}}
        )
        assert rh.supports_prompt_caching("gemini/private") is True

    @pytest.mark.parametrize(
        "name, provider, caching",
        [
            ("claude-sonnet-latest", "anthropic", True),
            (
                "openrouter/~anthropic/claude-sonnet-latest",
                "openrouter",
                True,
            ),
            ("gemini-pro-latest", "gemini", False),
        ],
    )
    def test_latest_aliases_keep_their_vendor(
        self, monkeypatch, name, provider, caching
    ):
        by_id = {entry["id"]: entry for entry in CATALOG}
        aliases = [
            {
                **by_id["anthropic/claude-sonnet-4.6"],
                "id": "~anthropic/claude-sonnet-latest",
            },
            {
                **by_id["google/gemini-2.5-pro"],
                "id": "~google/gemini-pro-latest",
            },
        ]
        monkeypatch.setattr(
            registry, "_fetch", lambda: [*CATALOG, *aliases]
        )
        info = rh.get_model_info(name)
        assert info["provider"] == provider
        assert info["supports_prompt_caching"] is caching
        assert rh.supports_prompt_caching(name) is caching


class TestProviderKey:
    def test_provider_and_litellm_provider_match(self):
        info = rh.get_model_info("claude-sonnet-4-6")
        assert info["provider"] == "anthropic"
        assert info["litellm_provider"] == info["provider"]

    def test_openrouter_prefix_names_openrouter(self):
        info = rh.get_model_info(
            "openrouter/anthropic/claude-sonnet-4.6"
        )
        assert info["provider"] == "openrouter"

    def test_custom_llm_provider_wins(self):
        info = rh.get_model_info(
            "claude-sonnet-4-6", custom_llm_provider="openrouter"
        )
        assert info["provider"] == "openrouter"

    def test_registered_litellm_provider_is_mirrored(self):
        rh.register_model({"m": {"litellm_provider": "groq"}})
        info = rh.get_model_info("m")
        assert info["provider"] == "groq"
        assert info["litellm_provider"] == "groq"
        rh.register_model({"n": {"provider": "cohere"}})
        assert rh.get_model_info("n")["litellm_provider"] == "cohere"

    def test_registered_without_a_provider(self):
        rh.register_model({"p": {"max_input_tokens": 1}})
        info = rh.get_model_info("p")
        assert info["provider"] is None
        assert rh.get_model_info("ollama/p")["provider"] == "ollama"


class TestDiskCache:
    """The last successful fetch is kept on disk for offline runs."""

    def serve(self, url, **kwargs):
        return httpx.Response(
            200,
            json={"data": CATALOG},
            request=httpx.Request("GET", url),
        )

    def offline(self, *args, **kwargs):
        raise httpx.ConnectError("offline")

    def test_fetch_writes_and_reads_the_disk_cache(
        self, monkeypatch, tmp_path
    ):
        path = registry._cache_path()
        assert path.startswith(str(tmp_path))
        monkeypatch.setattr(httpx, "get", self.serve)
        assert len(REAL_FETCH()) == len(CATALOG)
        assert os.path.exists(path)
        monkeypatch.setattr(httpx, "get", self.offline)
        assert [e["id"] for e in REAL_FETCH()] == [
            e["id"] for e in CATALOG
        ]

    def test_offline_start_uses_the_disk_cache(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", self.serve)
        REAL_FETCH()
        monkeypatch.setattr(registry, "_fetch", REAL_FETCH)
        monkeypatch.setattr(httpx, "get", self.offline)
        registry.clear_model_cache()
        assert rh.supports_vision("gpt-5.4-mini") is True
        assert rh.get_max_tokens("claude-sonnet-4-6") == 128000

    def test_no_disk_cache_means_unknown(self, monkeypatch):
        monkeypatch.setattr(registry, "_fetch", REAL_FETCH)
        monkeypatch.setattr(httpx, "get", self.offline)
        assert rh.supports_vision("gpt-5.4-mini") is False

    @pytest.mark.parametrize("value", ["off", "0", "false", "no"])
    def test_disk_cache_can_be_disabled(self, monkeypatch, value):
        monkeypatch.setenv(registry.CACHE_DIR_ENV, value)
        assert registry._cache_path() is None
        monkeypatch.setattr(httpx, "get", self.serve)
        REAL_FETCH()
        monkeypatch.setattr(httpx, "get", self.offline)
        assert REAL_FETCH() == []

    def test_default_location_is_under_the_cache_home(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv(registry.CACHE_DIR_ENV)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert registry._cache_path() == os.path.join(
            str(tmp_path), "routehub", "openrouter_models.json"
        )

    def test_corrupt_disk_cache_is_ignored(self, monkeypatch):
        path = registry._cache_path()
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write("not json")
        monkeypatch.setattr(httpx, "get", self.offline)
        assert REAL_FETCH() == []

    def test_unwritable_cache_dir_does_not_break_fetching(
        self, monkeypatch, tmp_path
    ):
        blocker = tmp_path / "file"
        blocker.write_text("")
        monkeypatch.setenv(registry.CACHE_DIR_ENV, str(blocker / "x"))
        monkeypatch.setattr(httpx, "get", self.serve)
        assert len(REAL_FETCH()) == len(CATALOG)


class TestRegisterModel:
    def test_registered_models_override_and_never_expire(self):
        rh.register_model(
            {
                "my-model": {
                    "max_input_tokens": 8192,
                    "max_output_tokens": 1024,
                    "supports_function_calling": True,
                }
            }
        )
        info = rh.get_model_info("my-model")
        assert info["key"] == "my-model"
        assert info["max_output_tokens"] == 1024
        assert info["supports_vision"] is None
        assert rh.supports_function_calling("my-model") is True
        assert "my-model" in rh.model_list
        assert rh.model_cost["my-model"]["max_input_tokens"] == 8192

    def test_registration_merges_fields(self):
        rh.register_model({"m": {"max_input_tokens": 1}})
        rh.register_model({"m": {"max_output_tokens": 2}})
        info = rh.get_model_info("m")
        assert (
            info["max_input_tokens"] == 1
            and info["max_output_tokens"] == 2
        )

    def test_registration_beats_the_catalog(self):
        rh.register_model(
            {"gpt-5.4-mini": {"max_input_tokens": 1234}}
        )
        assert (
            rh.get_model_info("gpt-5.4-mini")["max_input_tokens"]
            == 1234
        )

    def test_provider_prefixed_lookup_finds_registration(self):
        rh.register_model({"local-llama": {"max_input_tokens": 4096}})
        assert (
            rh.get_model_info("ollama/local-llama")[
                "max_input_tokens"
            ]
            == 4096
        )

    def test_register_from_json_file(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text('{"file-model": {"max_input_tokens": 77}}')
        rh.register_model(str(path))
        assert (
            rh.get_model_info("file-model")["max_input_tokens"] == 77
        )


def test_normalize_names():
    assert (
        registry._normalize("claude-opus-4-7-20251001")
        == "claude-opus-4.7"
    )
    assert (
        registry._normalize("claude-3-5-haiku-latest")
        == "claude-3.5-haiku"
    )
    assert registry._normalize("gpt-4o-2024-08-06") == "gpt-4o"
    assert registry._normalize("gemini-2.5-pro") == "gemini-2.5-pro"
