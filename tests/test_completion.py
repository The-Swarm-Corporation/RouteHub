"""completion and acompletion through OpenAI-compatible providers."""

import functools
import inspect
import json
import typing

import openai
import pytest
from conftest import chat_completion
from pydantic import BaseModel

import routehub as rh
import routehub.main as main
from routehub._http import httpx

USER = [{"role": "user", "content": "Hi"}]


def ok(**message):
    return httpx.Response(200, json=chat_completion(**message))


def chunk(delta=None, finish=None, usage=None, choices=True):
    body = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": (
            [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish,
                }
            ]
            if choices
            else []
        ),
    }
    if usage:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


def event_stream(*deltas, usage=True):
    """An OpenAI chat-completion event stream response."""
    parts = [chunk(d) for d in deltas] + [chunk(finish="stop")]
    if usage:
        parts.append(
            chunk(
                choices=False,
                usage={
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            )
        )
    parts.append("data: [DONE]\n\n")
    return httpx.Response(
        200,
        content="".join(parts).encode(),
        headers={"content-type": "text/event-stream"},
    )


class TestBasics:
    def test_returns_openai_chat_completion(self, mock_openai):
        client, recorder = mock_openai(ok(content="Hello"))
        response = rh.completion(
            model="gpt-5.4-mini", messages=USER, client=client
        )
        assert isinstance(response, openai.types.chat.ChatCompletion)
        assert response.choices[0].message.content == "Hello"
        assert response.usage.total_tokens == 6
        assert recorder.last_json["model"] == "gpt-5.4-mini"
        assert recorder.last_json["messages"] == USER

    def test_provider_prefix_is_stripped(self, mock_openai):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=USER,
            client=client,
        )
        assert (
            recorder.last_json["model"] == "llama-3.3-70b-versatile"
        )

    def test_provider_specific_fields_are_kept(self, mock_openai):
        client, _ = mock_openai(
            ok(content="144", reasoning_content="12 * 12")
        )
        response = rh.completion(
            model="deepseek/deepseek-reasoner",
            messages=USER,
            client=client,
            api_key="k",
        )
        assert (
            response.choices[0].message.reasoning_content == "12 * 12"
        )

    def test_model_dump_works_like_litellm(self, mock_openai):
        client, _ = mock_openai(ok(content="Hi"))
        dumped = rh.completion(
            model="gpt-4o", messages=USER, client=client
        ).model_dump()
        assert dumped["choices"][0]["message"]["content"] == "Hi"

    def test_requires_model_and_messages(self):
        with pytest.raises(rh.BadRequestError):
            rh.completion(model="gpt-5.4-mini", messages=[])
        with pytest.raises(rh.BadRequestError):
            rh.completion(model="", messages=USER)

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(rh.BadRequestError, match="not supported"):
            rh.completion(model="mystery-model", messages=USER)

    def test_missing_api_key_fails_before_any_request(
        self, monkeypatch
    ):
        monkeypatch.delenv("GROQ_API_KEY")
        with pytest.raises(
            rh.AuthenticationError, match="GROQ_API_KEY"
        ):
            rh.completion(
                model="groq/llama-3.3-70b-versatile", messages=USER
            )

    def test_reasoning_effort_signature_lists_values(self):
        annotation = (
            inspect.signature(rh.completion)
            .parameters["reasoning_effort"]
            .annotation
        )
        assert (
            typing.get_args(typing.get_args(annotation)[0])
            == rh.REASONING_EFFORTS
        )
        assert rh.get_reasoning_efforts() == (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        )

    def test_completion_and_acompletion_share_a_signature(self):
        assert (
            inspect.signature(rh.acompletion).parameters.keys()
            == inspect.signature(rh.completion).parameters.keys()
        )

    def test_settings_are_parameters_with_documented_defaults(self):
        params = inspect.signature(rh.completion).parameters
        assert params["drop_params"].default is False
        assert params["num_retries"].default is None
        assert params["ssl_verify"].default is True
        assert params["set_verbose"].default is False
        assert params["request_timeout"].default == 600.0


class TestRequestShaping:
    def test_openai_max_tokens_becomes_max_completion_tokens(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            max_tokens=50,
            client=client,
        )
        assert recorder.last_json["max_completion_tokens"] == 50
        assert "max_tokens" not in recorder.last_json

    def test_other_providers_keep_max_tokens(self, mock_openai):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=USER,
            max_tokens=50,
            client=client,
        )
        assert recorder.last_json["max_tokens"] == 50

    def test_max_completion_tokens_becomes_max_tokens_where_required(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok(), ok())
        rh.completion(
            model="mistral/mistral-small-latest",
            messages=USER,
            max_tokens=10,
            max_completion_tokens=50,
            client=client,
        )
        assert recorder.last_json["max_tokens"] == 50
        assert "max_completion_tokens" not in recorder.last_json
        rh.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=USER,
            max_completion_tokens=50,
            client=client,
        )
        assert recorder.last_json["max_completion_tokens"] == 50

    def test_drop_params_strips_sampling_for_reasoning_models(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-5.4-mini",
            messages=USER,
            temperature=0.2,
            top_p=0.9,
            presence_penalty=0.1,
            reasoning_effort="low",
            drop_params=True,
            client=client,
        )
        body = recorder.last_json
        assert (
            "temperature" not in body
            and "top_p" not in body
            and "presence_penalty" not in body
        )
        assert body["reasoning_effort"] == "low"

    def test_reasoning_models_keep_temperature_one(self, mock_openai):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="o3",
            messages=USER,
            temperature=1,
            drop_params=True,
            client=client,
        )
        assert recorder.last_json["temperature"] == 1

    def test_without_drop_params_everything_is_sent(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-5.4-mini",
            messages=USER,
            temperature=0.2,
            client=client,
        )
        assert recorder.last_json["temperature"] == 0.2

    def test_drop_params_removes_reasoning_effort_for_non_reasoning_models(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            reasoning_effort="high",
            temperature=0.3,
            drop_params=True,
            client=client,
        )
        assert "reasoning_effort" not in recorder.last_json
        assert recorder.last_json["temperature"] == 0.3

    def test_string_none_reasoning_effort_is_dropped(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-5.4-mini",
            messages=USER,
            reasoning_effort="None",
            client=client,
        )
        assert "reasoning_effort" not in recorder.last_json

    def test_stream_options_and_tool_settings_need_their_feature(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            stream_options={"include_usage": True},
            tool_choice="auto",
            parallel_tool_calls=True,
            client=client,
        )
        body = recorder.last_json
        assert (
            "stream_options" not in body
            and "tool_choice" not in body
            and "parallel_tool_calls" not in body
        )

    def test_tools_are_sent(self, mock_openai):
        tool = {
            "type": "function",
            "function": {
                "name": "a",
                "parameters": {"type": "object"},
            },
        }
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            tools=[tool],
            tool_choice="required",
            parallel_tool_calls=False,
            client=client,
        )
        body = recorder.last_json
        assert (
            body["tools"] == [tool]
            and body["tool_choice"] == "required"
            and body["parallel_tool_calls"] is False
        )

    def test_cache_control_is_stripped_for_openai(self, mock_openai):
        client, recorder = mock_openai(ok())
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Sys",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "Hi"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "parameters": {"type": "object"},
                },
                "cache_control": {"type": "ephemeral"},
            }
        ]
        rh.completion(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            client=client,
        )
        body = recorder.last_json
        assert body["messages"][0]["content"] == [
            {"type": "text", "text": "Sys"}
        ]
        assert "cache_control" not in body["tools"][0]
        assert "cache_control" in messages[0]["content"][0], (
            "the caller's messages must not be mutated"
        )

    def test_cache_control_is_kept_for_openrouter(self, mock_openai):
        client, recorder = mock_openai(ok())
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        rh.completion(
            model="openrouter/anthropic/claude-sonnet-4.6",
            messages=messages,
            client=client,
        )
        assert recorder.last_json["messages"][0]["content"][0][
            "cache_control"
        ] == {"type": "ephemeral"}

    def test_response_only_fields_are_removed_from_history(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        history = USER + [
            {
                "role": "assistant",
                "content": "Hey",
                "reasoning_content": "r",
                "thinking_blocks": [],
            },
            {"role": "user", "content": "More"},
        ]
        rh.completion(
            model="deepseek/deepseek-chat",
            messages=history,
            client=client,
            api_key="k",
        )
        assert recorder.last_json["messages"][1] == {
            "role": "assistant",
            "content": "Hey",
        }

    def test_response_message_objects_are_accepted_as_history(
        self, mock_openai, mock_http
    ):
        message = openai.types.chat.ChatCompletionMessage(
            role="assistant",
            content="Hey",
            reasoning_content="r",
            thinking_blocks=[
                {
                    "type": "thinking",
                    "thinking": "r",
                    "signature": "s",
                }
            ],
        )
        history = USER + [
            message,
            {"role": "user", "content": "More"},
        ]
        client, recorder = mock_openai(ok())
        rh.completion(
            model="deepseek/deepseek-chat",
            messages=history,
            client=client,
            api_key="k",
        )
        assert recorder.last_json["messages"][1] == {
            "role": "assistant",
            "content": "Hey",
        }
        reply = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        claude, recorder = mock_http(httpx.Response(200, json=reply))
        rh.completion(
            model="claude-sonnet-4-6",
            messages=history,
            client=claude,
            api_key="k",
        )
        sent = recorder.last_json["messages"][1]["content"]
        assert [b["type"] for b in sent] == ["thinking", "text"]
        assert rh.token_counter(
            model="gpt-4o", messages=[message]
        ) == rh.token_counter(
            model="gpt-4o",
            messages=[message.model_dump(exclude_none=True)],
        )

    def test_pydantic_response_format_becomes_json_schema(
        self, mock_openai
    ):
        class Answer(BaseModel):
            value: int

        client, recorder = mock_openai(ok(content='{"value": 4}'))
        rh.completion(
            model="gpt-4o",
            messages=USER,
            response_format=Answer,
            client=client,
        )
        fmt = recorder.last_json["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "Answer"
        assert (
            fmt["json_schema"]["schema"]["properties"]["value"][
                "type"
            ]
            == "integer"
        )

    def test_unknown_kwargs_go_to_the_body_unless_dropped(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=USER,
            top_k=5,
            client=client,
        )
        assert recorder.last_json["top_k"] == 5
        rh.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=USER,
            top_k=5,
            drop_params=True,
            client=client,
        )
        assert "top_k" not in recorder.last_json

    def test_extra_body_is_always_sent(self, mock_openai):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            extra_body={"custom": True},
            drop_params=True,
            client=client,
        )
        assert recorder.last_json["custom"] is True

    def test_litellm_only_kwargs_are_ignored(self, mock_openai):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            caching=False,
            metadata={"trace": 1},
            litellm_call_id="x",
            client=client,
        )
        body = recorder.last_json
        assert (
            "caching" not in body
            and "metadata" not in body
            and "litellm_call_id" not in body
        )

    def test_openai_params_passed_as_kwargs_are_forwarded(
        self, mock_openai
    ):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            stop=["END"],
            seed=7,
            prompt_cache_key="abc",
            client=client,
        )
        body = recorder.last_json
        assert (
            body["stop"] == ["END"]
            and body["seed"] == 7
            and body["prompt_cache_key"] == "abc"
        )

    def test_extra_headers_and_headers_are_merged(self, mock_openai):
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            extra_headers={"x-a": "1"},
            headers={"x-b": "2"},
            client=client,
        )
        request = recorder.requests[-1]
        assert (
            request.headers["x-a"] == "1"
            and request.headers["x-b"] == "2"
        )


class TestStreaming:
    def test_stream_yields_chunks_and_usage(self, mock_openai):
        client, _ = mock_openai(
            event_stream(
                {"role": "assistant", "content": ""},
                {"content": "Hel"},
                {"content": "lo"},
            )
        )
        stream = rh.completion(
            model="gpt-4o",
            messages=USER,
            stream=True,
            stream_options={"include_usage": True},
            client=client,
        )
        chunks = list(stream)
        assert all(
            isinstance(c, openai.types.chat.ChatCompletionChunk)
            for c in chunks
        )
        assert (
            "".join(
                c.choices[0].delta.content or ""
                for c in chunks
                if c.choices
            )
            == "Hello"
        )
        assert chunks[-1].usage.total_tokens == 5

    def test_stream_requests_stream_options(self, mock_openai):
        client, recorder = mock_openai(event_stream({"content": "x"}))
        list(
            rh.completion(
                model="gpt-4o",
                messages=USER,
                stream=True,
                stream_options={"include_usage": True},
                client=client,
            )
        )
        assert recorder.last_json["stream"] is True
        assert recorder.last_json["stream_options"] == {
            "include_usage": True
        }

    def test_stream_is_a_context_manager(self, mock_openai):
        client, _ = mock_openai(event_stream({"content": "x"}))
        with rh.completion(
            model="gpt-4o", messages=USER, stream=True, client=client
        ) as stream:
            first = next(iter(stream))
        assert first.choices[0].delta.content == "x"

    async def test_async_stream(self, mock_openai):
        client, _ = mock_openai(
            event_stream({"content": "As"}, {"content": "ync"}),
            is_async=True,
        )
        stream = await rh.acompletion(
            model="gpt-4o", messages=USER, stream=True, client=client
        )
        text = ""
        async for item in stream:
            if item.choices:
                text += item.choices[0].delta.content or ""
        assert text == "Async"


class TestAsync:
    async def test_acompletion_matches_completion(self, mock_openai):
        client, recorder = mock_openai(
            ok(content="Async hello"), is_async=True
        )
        response = await rh.acompletion(
            model="gpt-4o",
            messages=USER,
            max_tokens=10,
            client=client,
        )
        assert response.choices[0].message.content == "Async hello"
        assert recorder.last_json["max_completion_tokens"] == 10

    async def test_acompletion_maps_errors(self, mock_openai):
        client, _ = mock_openai(
            httpx.Response(
                429, json={"error": {"message": "Rate limit reached"}}
            ),
            is_async=True,
        )
        with pytest.raises(rh.RateLimitError):
            await rh.acompletion(
                model="gpt-4o", messages=USER, client=client
            )

    async def test_acompletion_validates_like_completion(self):
        with pytest.raises(rh.BadRequestError):
            await rh.acompletion(model="gpt-4o", messages=[])


class TestErrors:
    @pytest.mark.parametrize(
        "status, message, expected",
        [
            (
                400,
                "Invalid value for temperature",
                rh.BadRequestError,
            ),
            (
                400,
                "This model's maximum context length is 128000 tokens",
                rh.ContextWindowExceededError,
            ),
            (
                400,
                "Your request was rejected by our safety system: content_policy_violation",
                rh.ContentPolicyViolationError,
            ),
            (
                401,
                "Incorrect API key provided",
                rh.AuthenticationError,
            ),
            (
                403,
                "Project does not have access",
                rh.PermissionDeniedError,
            ),
            (404, "The model does not exist", rh.NotFoundError),
            (422, "Unprocessable", rh.UnprocessableEntityError),
            (429, "Rate limit reached", rh.RateLimitError),
            (500, "Server error", rh.InternalServerError),
            (503, "Overloaded", rh.ServiceUnavailableError),
        ],
    )
    def test_status_codes_map_to_gate_exceptions(
        self, mock_openai, status, message, expected
    ):
        client, _ = mock_openai(
            httpx.Response(
                status, json={"error": {"message": message}}
            )
        )
        with pytest.raises(expected) as raised:
            rh.completion(
                model="gpt-4o", messages=USER, client=client
            )
        error = raised.value
        assert type(error) is expected
        assert error.status_code == status
        assert error.llm_provider == "openai"
        assert error.model == "gpt-4o"
        assert message in str(error)

    def test_gate_exceptions_are_openai_exceptions(self, mock_openai):
        client, _ = mock_openai(
            httpx.Response(
                401, json={"error": {"message": "bad key"}}
            )
        )
        with pytest.raises(openai.AuthenticationError):
            rh.completion(
                model="gpt-4o", messages=USER, client=client
            )

    def test_original_error_is_chained(self, mock_openai):
        client, _ = mock_openai(
            httpx.Response(
                401, json={"error": {"message": "bad key"}}
            )
        )
        with pytest.raises(rh.AuthenticationError) as raised:
            rh.completion(
                model="gpt-4o", messages=USER, client=client
            )
        assert isinstance(
            raised.value.__cause__, openai.AuthenticationError
        )

    def test_connection_failure_maps(self, mock_openai):
        def fail(request):
            raise httpx.ConnectError("refused", request=request)

        client, _ = mock_openai(fail)
        with pytest.raises(rh.APIConnectionError):
            rh.completion(
                model="gpt-4o", messages=USER, client=client
            )

    def test_timeout_maps(self, mock_openai):
        def slow(request):
            raise httpx.ReadTimeout("too slow", request=request)

        client, _ = mock_openai(slow)
        with pytest.raises(rh.Timeout):
            rh.completion(
                model="gpt-4o", messages=USER, client=client
            )

    def test_error_event_inside_a_stream_maps(self, mock_openai):
        error = {"error": {"code": 429, "message": "Rate limit"}}
        client, _ = mock_openai(
            httpx.Response(
                200,
                content=(
                    chunk({"content": "Hi"})
                    + f"data: {json.dumps(error)}\n\n"
                ).encode(),
                headers={"content-type": "text/event-stream"},
            )
        )
        stream = rh.completion(
            model="gpt-4o", messages=USER, stream=True, client=client
        )
        assert next(stream).choices[0].delta.content == "Hi"
        with pytest.raises(rh.RateLimitError):
            next(stream)


class TestMockResponse:
    def test_mock_text(self):
        response = rh.completion(
            model="any-model", messages=USER, mock_response="Mocked"
        )
        assert response.choices[0].message.content == "Mocked"
        assert response.usage.completion_tokens >= 1

    def test_mock_needs_no_provider_or_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY")
        assert (
            rh.completion(
                model="unknown/thing",
                messages=USER,
                mock_response="x",
            )
            .choices[0]
            .message.content
            == "x"
        )

    def test_mock_stream(self):
        stream = rh.completion(
            model="any-model",
            messages=USER,
            mock_response="a b c",
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)
        assert (
            "".join(
                c.choices[0].delta.content or ""
                for c in chunks
                if c.choices
            )
            == "a b c"
        )
        assert chunks[-1].usage is not None

    def test_mock_exception_is_raised(self):
        with pytest.raises(ValueError, match="boom"):
            rh.completion(
                model="any-model",
                messages=USER,
                mock_response=ValueError("boom"),
            )

    async def test_mock_async(self):
        response = await rh.acompletion(
            model="any-model",
            messages=USER,
            mock_response="Async mock",
        )
        assert response.choices[0].message.content == "Async mock"
        stream = await rh.acompletion(
            model="any-model",
            messages=USER,
            mock_response="x y",
            stream=True,
        )
        assert (
            "".join(
                [
                    c.choices[0].delta.content or ""
                    async for c in stream
                    if c.choices
                ]
            )
            == "x y"
        )


class TestSettings:
    @pytest.fixture
    def captured_client(self, monkeypatch, mock_openai):
        """Capture the arguments completion uses to build its SDK client."""
        seen = {}
        client, recorder = mock_openai(ok())

        def fake_openai_client(provider, api_key, api_base, **kwargs):
            seen.update(
                provider=provider,
                api_key=api_key,
                api_base=api_base,
                **kwargs,
            )
            return client

        monkeypatch.setattr(main, "openai_client", fake_openai_client)
        return seen

    def test_num_retries_defaults_to_two(self, captured_client):
        rh.completion(model="gpt-4o", messages=USER)
        assert captured_client["max_retries"] == 2

    def test_num_retries_is_passed_to_the_client(
        self, captured_client
    ):
        rh.completion(model="gpt-4o", messages=USER, num_retries=5)
        assert captured_client["max_retries"] == 5

    def test_max_retries_is_accepted_as_an_alias(
        self, captured_client
    ):
        rh.completion(model="gpt-4o", messages=USER, max_retries=1)
        assert captured_client["max_retries"] == 1

    def test_ssl_verify_and_credentials_are_passed(
        self, captured_client
    ):
        rh.completion(
            model="groq/llama-3.3-70b-versatile",
            messages=USER,
            ssl_verify=False,
            api_key="explicit",
            base_url="https://proxy.local/v1",
        )
        assert captured_client["ssl_verify"] is False
        assert captured_client["api_key"] == "explicit"
        assert captured_client["api_base"] == "https://proxy.local/v1"

    def test_env_key_is_used_by_default(self, captured_client):
        rh.completion(
            model="groq/llama-3.3-70b-versatile", messages=USER
        )
        assert captured_client["api_key"] == "test-groq_api_key"
        assert (
            captured_client["api_base"]
            == "https://api.groq.com/openai/v1"
        )

    def test_request_timeout_and_timeout(
        self, mock_openai, monkeypatch
    ):
        seen = []
        client, _ = mock_openai(ok())
        original = client.chat.completions.create

        @functools.wraps(original)
        def spy(*args, **kwargs):
            seen.append(kwargs["timeout"])
            return original(*args, **kwargs)

        monkeypatch.setattr(client.chat.completions, "create", spy)
        rh.completion(
            model="gpt-4o",
            messages=USER,
            client=client,
            request_timeout=12.5,
        )
        rh.completion(
            model="gpt-4o",
            messages=USER,
            client=client,
            request_timeout=12.5,
            timeout=3,
        )
        rh.completion(model="gpt-4o", messages=USER, client=client)
        assert seen == [12.5, 3, 600.0]

    def test_set_verbose_logs_to_stderr_without_secrets(
        self, mock_openai, capsys
    ):
        client, _ = mock_openai(ok())
        rh.completion(
            model="gpt-4o",
            messages=USER,
            client=client,
            set_verbose=True,
            api_key="sk-secret",
        )
        err = capsys.readouterr().err
        assert "routehub: openai gpt-4o" in err
        assert "answered in" in err
        assert "sk-secret" not in err

    def test_quiet_by_default(self, mock_openai, capsys):
        client, _ = mock_openai(ok())
        rh.completion(model="gpt-4o", messages=USER, client=client)
        assert capsys.readouterr().err == ""

    def test_gzip_leaves_openai_compatible_bodies_alone(
        self, mock_openai, monkeypatch
    ):
        monkeypatch.setenv("ROUTEHUB_GZIP_ON", "1")
        client, recorder = mock_openai(ok())
        rh.completion(
            model="gpt-4o", messages=USER, gzip=True, client=client
        )
        request = recorder.requests[0]
        assert "content-encoding" not in request.headers
        assert json.loads(request.content)["model"] == "gpt-4o"


class TestClients:
    def test_clients_are_cached_by_settings(self):
        from routehub import clients

        clients._clients.clear()
        a = clients.openai_client(
            "openai",
            "k",
            None,
            is_async=False,
            max_retries=4,
            ssl_verify=True,
        )
        b = clients.openai_client(
            "openai",
            "k",
            None,
            is_async=False,
            max_retries=4,
            ssl_verify=True,
        )
        c = clients.openai_client(
            "openai",
            "k",
            None,
            is_async=False,
            max_retries=1,
            ssl_verify=True,
        )
        assert a is b and a is not c
        assert a.max_retries == 4 and c.max_retries == 1

    def test_clients_keep_idle_connections_for_a_minute(self):
        from routehub import clients

        clients._clients.clear()
        sdk = clients.openai_client(
            "openai",
            "k",
            None,
            is_async=False,
            max_retries=0,
            ssl_verify=True,
        )
        native = clients.http_client(is_async=False, ssl_verify=True)
        for pool in (
            sdk._client._transport._pool,
            native._transport._pool,
        ):
            assert pool._keepalive_expiry == 60.0
            assert pool._max_keepalive_connections == 100
            assert pool._max_connections == 1000
        short = clients.http_client(
            is_async=False, ssl_verify=True, keepalive_expiry=5.0
        )
        assert short is not native
        assert short._transport._pool._keepalive_expiry == 5.0

    def test_azure_builds_an_azure_client(self):
        from routehub import clients

        client = clients.openai_client(
            "azure",
            "k",
            "https://x.openai.azure.com",
            is_async=False,
            max_retries=0,
            ssl_verify=True,
            api_version="2024-10-21",
        )
        assert isinstance(client, openai.AzureOpenAI)

    async def test_async_clients_are_per_event_loop(self):
        from routehub import clients

        a = clients.openai_client(
            "openai",
            "k",
            None,
            is_async=True,
            max_retries=0,
            ssl_verify=True,
        )
        b = clients.openai_client(
            "openai",
            "k",
            None,
            is_async=True,
            max_retries=0,
            ssl_verify=True,
        )
        assert a is b
        assert isinstance(a, openai.AsyncOpenAI)


def test_own_client_needs_no_env_key(monkeypatch, mock_openai):
    monkeypatch.delenv("OPENAI_API_KEY")
    client, _ = mock_openai(ok(content="from my client"))
    response = rh.completion(
        model="gpt-4o", messages=USER, client=client
    )
    assert response.choices[0].message.content == "from my client"


def test_responses_returns_a_response_and_maps_errors(mock_openai):
    client, recorder = mock_openai(
        httpx.Response(200, json={"id": "resp_1", "output": []}),
        httpx.Response(429, json={"error": {"message": "Busy"}}),
    )
    reply = rh.responses("openai/gpt-5.4", input="hi", client=client)
    assert isinstance(reply, openai.types.responses.Response)
    assert recorder.last_json == {"model": "gpt-5.4", "input": "hi"}
    with pytest.raises(rh.RateLimitError):
        rh.responses("gpt-5.4", input="hi", client=client)


def test_responses_drop_params_drops_what_the_model_rejects(
    mock_openai,
):
    client, recorder = mock_openai(
        httpx.Response(200, json={"id": "resp_1", "output": []}),
        httpx.Response(200, json={"id": "resp_2", "output": []}),
    )
    rh.responses(
        "o3",
        input="hi",
        temperature=0.2,
        top_p=0.5,
        drop_params=True,
        client=client,
    )
    assert recorder.last_json == {"model": "o3", "input": "hi"}
    rh.responses(
        "gpt-4o",
        input="hi",
        temperature=0.2,
        reasoning={"effort": "low"},
        drop_params=True,
        client=client,
    )
    assert recorder.last_json == {
        "model": "gpt-4o",
        "input": "hi",
        "temperature": 0.2,
    }
