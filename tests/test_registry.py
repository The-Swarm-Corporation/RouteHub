"""Model metadata from the OpenRouter catalog."""

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
        assert info["supports_prompt_caching"] is True

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

    def test_exact_name_wins_over_a_normalized_variant(
        self, monkeypatch
    ):
        catalog = [
            {"id": "openai/gpt-4o-2024-11-20"},
            {"id": "~deepseek/deepseek-v4-flash-latest"},
            {"id": "openai/gpt-4o"},
            {"id": "deepseek/deepseek-v4-flash"},
        ]
        monkeypatch.setattr(registry, "_fetch", lambda: catalog)
        assert rh.get_model_info("gpt-4o")["key"] == "openai/gpt-4o"
        assert (
            rh.get_model_info("deepseek-v4-flash")["key"]
            == "deepseek/deepseek-v4-flash"
        )
        assert (
            rh.get_model_info("gpt-4o-2024-11-20")["key"]
            == "openai/gpt-4o-2024-11-20"
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
        assert set(cost) == {entry["id"] for entry in CATALOG}


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
