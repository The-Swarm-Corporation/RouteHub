import hashlib
import time
import types

import pytest

import routehub as rh
from routehub import bedrock
from routehub._http import httpx
from routehub.exceptions import BadRequestError

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather lookup",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}


@pytest.fixture(autouse=True)
def aws_region_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")


def converse_body(**over):
    body = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "hi there"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 5,
            "outputTokens": 2,
            "totalTokens": 7,
            "cacheReadInputTokens": 3,
            "cacheWriteInputTokens": 1,
        },
        "ResponseMetadata": {"RequestId": "r-1"},
        "model": "anthropic.claude-sonnet-4-6",
    }
    body.update(over)
    return body


def reply(status=200, body=None):
    return httpx.Response(status, json=body or converse_body())


def make_call(**over):
    fields = {
        "model": "bedrock/amazon.nova-lite-v1",
        "bare": "amazon.nova-lite-v1",
        "provider": "bedrock",
        "api_key": "bk-1",
        "api_base": None,
        "messages": [{"role": "user", "content": "hi"}],
        "params": {},
        "extras": {},
        "extra_headers": None,
        "extra_body": None,
        "drop_params": False,
        "max_retries": 0,
        "timeout": None,
    }
    fields.update(over)
    return types.SimpleNamespace(**fields)


def tool_call(id, city):
    return {
        "id": id,
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": f'{{"city": "{city}"}}',
        },
    }


class TestBuildRequest:
    def test_maps_openai_to_converse(self):
        body = bedrock.build_request(
            "anthropic.claude-sonnet-4-6",
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
            ],
            {
                "max_tokens": 128,
                "temperature": 0.2,
                "stop": ["\n\n"],
                "tools": [WEATHER_TOOL],
            },
        )
        assert body["modelId"] == "anthropic.claude-sonnet-4-6"
        assert body["system"] == [{"text": "Be brief."}]
        assert body["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]
        assert body["inferenceConfig"] == {
            "maxTokens": 128,
            "temperature": 0.2,
            "stopSequences": ["\n\n"],
        }
        spec = body["toolConfig"]["tools"][0]["toolSpec"]
        assert spec["name"] == "get_weather"
        assert spec["inputSchema"]["json"]["properties"] == {
            "city": {"type": "string"}
        }

    def test_tool_call_and_result_become_blocks(self):
        body = bedrock.build_request(
            "m",
            [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "tool_calls": [tool_call("t-1", "Paris")],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t-1",
                    "content": "sunny",
                },
            ],
            {},
        )
        messages = body["messages"]
        assert messages[1]["content"][0]["toolUse"] == {
            "toolUseId": "t-1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
        assert messages[2]["content"][0]["toolResult"] == {
            "toolUseId": "t-1",
            "content": [{"text": "sunny"}],
        }

    def test_parallel_tool_results_share_one_user_turn(self):
        body = bedrock.build_request(
            "m",
            [
                {
                    "role": "user",
                    "content": "weather in Paris and Rome?",
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        tool_call("t-1", "Paris"),
                        tool_call("t-2", "Rome"),
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t-1",
                    "content": "sunny",
                },
                {
                    "role": "tool",
                    "tool_call_id": "t-2",
                    "content": "rainy",
                },
                {"role": "user", "content": "thanks"},
            ],
            {},
        )
        messages = body["messages"]
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "user",
        ]
        results = messages[2]["content"]
        assert len(results) == 3
        assert results[1]["toolResult"]["toolUseId"] == "t-2"
        assert results[2] == {"text": "thanks"}

    def test_developer_and_list_system_content(self):
        body = bedrock.build_request(
            "m",
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Be brief."}
                    ],
                },
                {"role": "developer", "content": "Use metric."},
                {"role": "user", "content": "hello"},
            ],
            {},
        )
        assert body["system"] == [
            {"text": "Be brief."},
            {"text": "Use metric."},
        ]
        assert body["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]

    def test_stop_string_and_max_completion_tokens(self):
        body = bedrock.build_request(
            "m",
            [{"role": "user", "content": "hello"}],
            {"stop": "END", "max_completion_tokens": 50},
        )
        assert body["inferenceConfig"] == {
            "maxTokens": 50,
            "stopSequences": ["END"],
        }

    def test_tool_choice_without_tools_is_ignored(self):
        body = bedrock.build_request(
            "m",
            [{"role": "user", "content": "hello"}],
            {"tool_choice": "auto"},
        )
        assert "toolConfig" not in body

    def test_tool_choice_none_omits_tool_config(self):
        body = bedrock.build_request(
            "m",
            [{"role": "user", "content": "hello"}],
            {"tools": [WEATHER_TOOL], "tool_choice": "none"},
        )
        assert "toolConfig" not in body


class TestToChatCompletion:
    def test_text_and_usage_with_cache_tokens(self):
        result = bedrock.to_chat_completion(
            converse_body(), "bedrock/amazon.nova-lite-v1", "r-1"
        )
        assert result.id == "r-1"
        assert result.choices[0].message.content == "hi there"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 9
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 11
        assert result.usage.cache_read_input_tokens == 3
        assert result.usage.cache_creation_input_tokens == 1

    def test_tool_use_becomes_tool_calls(self):
        body = converse_body(
            output={
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t-1",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        }
                    ],
                }
            },
            stopReason="tool_use",
        )
        result = bedrock.to_chat_completion(body, "m", None)
        assert result.choices[0].finish_reason == "tool_calls"
        call = result.choices[0].message.tool_calls[0]
        assert call.id == "t-1"
        assert call.function.name == "get_weather"
        assert call.function.arguments == '{"city": "Paris"}'


class TestComplete:
    def test_bearer_key_reaches_converse(self, mock_http):
        client, recorder = mock_http(reply())
        call = make_call(
            model="bedrock/anthropic.claude-sonnet-4-6",
            bare="anthropic.claude-sonnet-4-6",
        )
        result = bedrock.complete(call, client)
        request = recorder.requests[0]
        assert request.headers["authorization"] == "Bearer bk-1"
        assert (
            request.url.host
            == "bedrock-runtime.us-east-1.amazonaws.com"
        )
        assert request.url.path.startswith(
            "/model/anthropic.claude-sonnet-4-6/converse"
        )
        assert recorder.last_json["messages"] == [
            {"role": "user", "content": [{"text": "hi"}]}
        ]
        assert result.choices[0].message.content == "hi there"

    def test_sigv4_used_for_placeholder_key(
        self, mock_http, monkeypatch
    ):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        client, recorder = mock_http(reply())
        bedrock.complete(make_call(api_key="bedrock"), client)
        auth = recorder.requests[0].headers["authorization"]
        assert auth.startswith("AWS4-HMAC-SHA256")

    def test_aws_region_extra_overrides_env(self, mock_http):
        client, recorder = mock_http(reply())
        bedrock.complete(
            make_call(extras={"aws_region": "ap-south-1"}), client
        )
        assert (
            recorder.requests[0].url.host
            == "bedrock-runtime.ap-south-1.amazonaws.com"
        )

    def test_no_credentials_raise_before_sending(
        self, mock_http, monkeypatch
    ):
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ):
            monkeypatch.delenv(name, raising=False)
        client, recorder = mock_http(reply())
        with pytest.raises(Exception) as err:
            bedrock.complete(make_call(api_key=None), client)
        assert "AWS" in str(err.value)
        assert recorder.requests == []

    def test_error_status_maps_to_exception(self, mock_http):
        client, _ = mock_http(
            reply(status=400, body={"message": "bad model"})
        )
        with pytest.raises(BadRequestError) as err:
            bedrock.complete(make_call(), client)
        assert "bad model" in str(err.value)

    def test_unsupported_param_raises_unless_dropped(self, mock_http):
        client, recorder = mock_http(reply())
        params = {"response_format": {"type": "json_object"}}
        with pytest.raises(BadRequestError) as err:
            bedrock.complete(make_call(params=params), client)
        assert "response_format" in str(err.value)
        assert recorder.requests == []
        result = bedrock.complete(
            make_call(params=params, drop_params=True), client
        )
        assert result.choices[0].message.content == "hi there"

    def test_missing_region_is_a_clear_error(
        self, mock_http, monkeypatch
    ):
        monkeypatch.delenv("AWS_REGION")
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        client, recorder = mock_http(reply())
        with pytest.raises(BadRequestError) as err:
            bedrock.complete(make_call(), client)
        assert "AWS_REGION" in str(err.value)
        assert recorder.requests == []

    def test_retry_is_signed_with_a_fresh_clock(
        self, mock_http, monkeypatch
    ):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        ticks = iter([0, 3600])
        monkeypatch.setattr(
            bedrock,
            "time",
            types.SimpleNamespace(
                strftime=time.strftime,
                gmtime=lambda: time.gmtime(next(ticks)),
                time=time.time,
                sleep=time.sleep,
            ),
        )
        client, recorder = mock_http(
            reply(status=503, body={"message": "busy"}), reply()
        )
        bedrock.complete(
            make_call(api_key=None, max_retries=1), client
        )
        first, second = recorder.requests
        assert (
            first.headers["x-amz-date"]
            != second.headers["x-amz-date"]
        )

    async def test_acomplete_sends_the_same_request(self, mock_http):
        client, recorder = mock_http(reply(), is_async=True)
        result = await bedrock.acomplete(make_call(), client)
        assert result.choices[0].message.content == "hi there"
        assert (
            recorder.requests[0].headers["authorization"]
            == "Bearer bk-1"
        )


class TestSigV4:
    def test_signature_is_deterministic(self):
        body = b'{"modelId": "m"}'
        args = (
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/m/converse",
            body,
            "us-east-1",
            "AKIDEXAMPLE",
            "secret",
        )
        headers = bedrock.sigv4_headers(
            *args, session_token="tok", amz_date="20260101T000000Z"
        )
        again = bedrock.sigv4_headers(
            *args, session_token="tok", amz_date="20260101T000000Z"
        )
        assert headers == again
        auth = headers["authorization"]
        assert auth.startswith(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/"
            "us-east-1/bedrock/aws4_request"
        )
        assert "Signature=" in auth
        assert headers["x-amz-security-token"] == "tok"
        assert (
            headers["x-amz-content-sha256"]
            == hashlib.sha256(body).hexdigest()
        )

    def test_colon_model_id_matches_aws_signature(self):
        headers = bedrock.sigv4_headers(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/anthropic.claude-3-5-sonnet-20240620-v1%3A0"
            "/converse",
            b'{"modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0"}',
            "us-east-1",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            amz_date="20260101T000000Z",
        )
        assert headers["authorization"] == (
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260101/"
            "us-east-1/bedrock/aws4_request, "
            "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
            "Signature=0e39903b089c85eec3562677e9fdc9edb62ec203a2"
            "3905352b5dc92af75af5dc"
        )

    def test_converse_url_quotes_model_ids(self):
        assert bedrock.converse_url(
            None, "us-east-1", "anthropic.claude-3-5-sonnet-v1:0"
        ) == (
            "https://bedrock-runtime.us-east-1.amazonaws.com"
            "/model/anthropic.claude-3-5-sonnet-v1%3A0/converse"
        )


class TestRouting:
    def chat(self, client, **kwargs):
        return rh.completion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
            **{"api_key": "bk-1", **kwargs},
        )

    def test_bearer_key_reaches_converse(self, mock_http):
        client, recorder = mock_http(reply())
        result = rh.completion(
            model="bedrock/anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
            max_tokens=128,
        )
        request = recorder.requests[0]
        assert request.headers["authorization"] == "Bearer bk-1"
        assert "x-api-key" not in request.headers
        assert (
            request.url.host
            == "bedrock-runtime.us-east-1.amazonaws.com"
        )
        assert request.url.path.startswith(
            "/model/anthropic.claude-sonnet-4-6/converse"
        )
        assert recorder.last_json["inferenceConfig"] == {
            "maxTokens": 128
        }
        assert result.choices[0].message.content == "hi there"
        assert result.usage.prompt_tokens == 9

    def test_sigv4_used_without_an_api_key(
        self, mock_http, monkeypatch
    ):
        monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        client, recorder = mock_http(reply())
        result = rh.completion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
        )
        assert result.choices[0].message.content == "hi there"
        auth = recorder.requests[0].headers["authorization"]
        assert auth.startswith("AWS4-HMAC-SHA256")

    async def test_acompletion_awaits_the_same_request(
        self, mock_http
    ):
        client, recorder = mock_http(reply(), is_async=True)
        result = await rh.acompletion(
            model="bedrock/amazon.nova-lite-v1",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bk-1",
            client=client,
        )
        assert result.choices[0].message.content == "hi there"
        assert (
            recorder.requests[0].headers["authorization"]
            == "Bearer bk-1"
        )

    def test_region_comes_from_env_and_aws_region(
        self, mock_http, monkeypatch
    ):
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        client, recorder = mock_http(reply(), reply())
        self.chat(client)
        self.chat(client, aws_region="ap-south-1")
        hosts = [r.url.host for r in recorder.requests]
        assert hosts == [
            "bedrock-runtime.eu-central-1.amazonaws.com",
            "bedrock-runtime.ap-south-1.amazonaws.com",
        ]

    def test_missing_region_is_a_clear_error(
        self, mock_http, monkeypatch
    ):
        monkeypatch.delenv("AWS_REGION")
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        client, recorder = mock_http(reply())
        with pytest.raises(BadRequestError) as err:
            self.chat(client)
        assert "AWS_REGION" in str(err.value)
        assert recorder.requests == []

    def test_error_status_maps_to_exception(self, mock_http):
        client, _ = mock_http(
            reply(status=400, body={"message": "bad model"})
        )
        with pytest.raises(BadRequestError) as err:
            self.chat(client, num_retries=0)
        assert "bad model" in str(err.value)

    def test_unsupported_param_raises_unless_dropped(self, mock_http):
        client, recorder = mock_http(reply())
        fmt = {"type": "json_object"}
        with pytest.raises(BadRequestError) as err:
            self.chat(client, response_format=fmt)
        assert "response_format" in str(err.value)
        assert recorder.requests == []
        result = self.chat(
            client, response_format=fmt, drop_params=True
        )
        assert result.choices[0].message.content == "hi there"

    def test_num_retries_re_signs_each_attempt(
        self, mock_http, monkeypatch
    ):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        ticks = iter([0, 3600])
        monkeypatch.setattr(
            bedrock,
            "time",
            types.SimpleNamespace(
                strftime=time.strftime,
                gmtime=lambda: time.gmtime(next(ticks)),
                time=time.time,
                sleep=time.sleep,
            ),
        )
        client, recorder = mock_http(
            reply(status=503, body={"message": "busy"}), reply()
        )
        self.chat(client, api_key="bedrock", num_retries=1)
        first, second = recorder.requests
        assert (
            first.headers["x-amz-date"]
            != second.headers["x-amz-date"]
        )

    def test_streaming_rejected_before_any_request(self, mock_http):
        client, recorder = mock_http(reply())
        with pytest.raises(BadRequestError) as err:
            self.chat(client, stream=True)
        assert "streaming" in str(err.value).lower()
        assert recorder.requests == []

    def test_embeddings_rejected(self, mock_http):
        client, recorder = mock_http(reply())
        with pytest.raises(BadRequestError, match="embeddings"):
            rh.embedding(
                model="bedrock/amazon.nova-lite-v1",
                input=["hi"],
                api_key="bk-1",
                client=client,
            )
        assert recorder.requests == []

    def test_responses_api_rejected_before_any_request(
        self, mock_openai
    ):
        client, recorder = mock_openai(reply())
        with pytest.raises(BadRequestError, match="Responses"):
            rh.responses(
                model="bedrock/amazon.nova-lite-v1",
                input="hi",
                api_key="bk-1",
                client=client,
            )
        assert recorder.requests == []
