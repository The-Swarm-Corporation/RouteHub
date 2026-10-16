"""JSON helper, tokenizer, lazy imports, exceptions and stream wrappers."""

import importlib
import subprocess
import sys

import openai
import pytest

import routehub as rh
from routehub import exceptions, tokenizer


class TestJson:
    def reload(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("ROUTEHUB_USE_ORJSON", raising=False)
        else:
            monkeypatch.setenv("ROUTEHUB_USE_ORJSON", value)
        import routehub._json as module

        return importlib.reload(module)

    @pytest.mark.parametrize("value", ["0", "false", "No", " off "])
    def test_env_var_forces_stdlib(self, monkeypatch, value):
        assert self.reload(monkeypatch, value).USING_ORJSON is False

    def test_uses_orjson_only_when_installed(self, monkeypatch):
        module = self.reload(monkeypatch, None)
        try:
            import orjson  # noqa: F401

            assert module.USING_ORJSON is True
        except ImportError:
            assert module.USING_ORJSON is False

    @pytest.mark.parametrize("value", [None, "false"])
    def test_round_trip_matches_either_way(self, monkeypatch, value):
        module = self.reload(monkeypatch, value)
        data = {
            "a": [1, 2.5, None, True],
            "b": {"c": "é"},
            1: "int key",
            "big": 2**70,
        }
        encoded = module.dumps(data)
        assert isinstance(encoded, bytes)
        assert module.loads(encoded) == {
            "a": [1, 2.5, None, True],
            "b": {"c": "é"},
            "1": "int key",
            "big": 2**70,
        }
        assert module.dumps_str({"x": 1}) == '{"x":1}'
        assert module.loads('{"y": 2}') == {"y": 2}

    def teardown_method(self):
        import routehub._json as module

        importlib.reload(module)


class TestTokenizer:
    def test_encode_counts_tokens(self):
        tokens = rh.encode(model="gpt-4o", text="hello world")
        assert isinstance(tokens, list) and 1 <= len(tokens) <= 4

    def test_encode_falls_back_to_an_estimate(self, monkeypatch):
        monkeypatch.setattr(
            tokenizer, "_encoding_for", lambda model: None
        )
        assert len(rh.encode(model="anything", text="a" * 40)) == 10

    def test_unknown_models_use_the_default_encoding(self):
        assert rh.encode(
            model="claude-sonnet-4-6", text="hi"
        ) == rh.encode(model="gpt-4o", text="hi")

    def test_custom_tokenizer(self):
        class Words:
            def encode(self, text):
                return text.split()

        assert rh.encode(text="a b c", custom_tokenizer=Words()) == [
            "a",
            "b",
            "c",
        ]
        assert rh.encode(
            text="a b", custom_tokenizer={"tokenizer": Words()}
        ) == ["a", "b"]

    def test_decode_round_trips(self):
        assert (
            rh.decode(
                model="gpt-4o",
                tokens=rh.encode(model="gpt-4o", text="round trip"),
            )
            == "round trip"
        )

    def test_token_counter_for_text_and_messages(self):
        assert rh.token_counter(
            model="gpt-4o", text="hello world"
        ) == len(rh.encode("gpt-4o", "hello world"))
        messages = [
            {"role": "system", "content": "Be brief."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            },
        ]
        count = rh.token_counter(model="gpt-4o", messages=messages)
        text_only = len(rh.encode("gpt-4o", "Be brief.")) + len(
            rh.encode("gpt-4o", "Hi")
        )
        assert count == text_only + 2 * 3 + 3

    def test_empty_messages_count_zero(self):
        assert rh.token_counter(messages=[]) == 0


class TestLazyImport:
    def run(self, code):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_import_loads_nothing_heavy(self):
        out = self.run(
            "import sys, routehub; "
            "print(any(m.startswith(('openai', 'tiktoken', 'httpx')) for m in sys.modules))"
        )
        assert out == "False"

    def test_model_lookups_never_import_the_openai_sdk(self):
        out = self.run(
            "import sys, routehub as rh; "
            "rh.get_llm_provider('groq/llama-3.3-70b-versatile', api_key='k'); "
            "print('openai' in sys.modules)"
        )
        assert out == "False"

    def test_preload_imports_the_sdk_once_while_completion_runs(self):
        out = self.run(
            "import sys, time, routehub as rh\n"
            "from routehub import completion\n"
            "hits = []\n"
            "class Count:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        hits.extend([1] if name == 'openai' else [])\n"
            "sys.meta_path.insert(0, Count())\n"
            "thread = rh.preload()\n"
            "while 'openai' not in sys.modules: time.sleep(0.001)\n"
            "reply = completion(model='gpt-5.4-mini', messages=[{'role': 'user', 'content': 'hi'}], mock_response='ok')\n"
            "thread.join(30)\n"
            "print(reply.choices[0].message.content, len(hits), thread.is_alive(), rh.preload())"
        )
        assert out == "ok 1 False None"

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            rh.not_a_real_name

    def test_dir_lists_lazy_names(self):
        assert {
            "completion",
            "acompletion",
            "embedding",
            "get_model_info",
        } <= set(dir(rh))


class TestExceptions:
    @pytest.mark.parametrize(
        "gate, sdk",
        [
            (rh.AuthenticationError, openai.AuthenticationError),
            (rh.BadRequestError, openai.BadRequestError),
            (rh.ContextWindowExceededError, openai.BadRequestError),
            (rh.RateLimitError, openai.RateLimitError),
            (rh.NotFoundError, openai.NotFoundError),
            (rh.InternalServerError, openai.InternalServerError),
            (rh.ServiceUnavailableError, openai.InternalServerError),
            (rh.Timeout, openai.APITimeoutError),
            (rh.APIConnectionError, openai.APIConnectionError),
        ],
    )
    def test_gate_exceptions_subclass_the_sdk(self, gate, sdk):
        assert issubclass(gate, sdk)

    def test_constructible_without_a_response(self):
        error = rh.RateLimitError(
            "slow down", llm_provider="groq", model="m"
        )
        assert (
            error.status_code == 429
            and error.llm_provider == "groq"
            and error.model == "m"
        )
        assert str(error) == "slow down"

    def test_litellm_style_import_paths(self):
        from routehub.exceptions import (
            AuthenticationError,
            BadRequestError,
            InternalServerError,
        )

        assert AuthenticationError is rh.AuthenticationError
        assert BadRequestError is rh.BadRequestError
        assert InternalServerError is rh.InternalServerError

    @pytest.mark.parametrize(
        "status, cls",
        [
            (400, rh.BadRequestError),
            (401, rh.AuthenticationError),
            (413, rh.ContextWindowExceededError),
            (418, rh.APIError),
            (502, rh.InternalServerError),
            (529, rh.ServiceUnavailableError),
        ],
    )
    def test_exception_for_status(self, status, cls):
        assert (
            type(exceptions.exception_for_status(status, "msg"))
            is cls
        )

    def test_map_exception_leaves_other_errors_alone(self):
        error = KeyError("x")
        assert exceptions.map_exception(error) is error
        gate = rh.BadRequestError("already mapped")
        assert exceptions.map_exception(gate) is gate


class TestStreamWrappers:
    def test_close_runs_once_on_exhaustion(self):
        from routehub.streaming import ChatStream

        closed = []
        stream = ChatStream(
            iter([1, 2]), lambda: closed.append(True), "p", "m"
        )
        assert list(stream) == [1, 2]
        stream.close()
        assert closed == [True]

    def test_errors_mid_stream_are_mapped(self):
        from routehub._http import httpx
        from routehub.streaming import ChatStream

        def chunks():
            yield 1
            raise httpx.ReadTimeout(
                "stalled", request=httpx.Request("GET", "https://x")
            )

        stream = ChatStream(chunks(), None, "openai", "gpt-4o")
        assert next(stream) == 1
        with pytest.raises(rh.Timeout):
            next(stream)

    async def test_async_close_runs_once(self):
        from routehub.streaming import AsyncChatStream

        closed = []

        async def chunks():
            yield "a"

        async def close():
            closed.append(True)

        stream = AsyncChatStream(chunks(), close, "p", "m")
        assert [c async for c in stream] == ["a"]
        await stream.aclose()
        assert closed == [True]
