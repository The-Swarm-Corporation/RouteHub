"""The native Anthropic adapter: request, response and stream translation."""

import base64
import gzip
import json

import openai
import pytest
from conftest import sse

import routehub as rh
from routehub import anthropic
from routehub._http import httpx

CACHE = {"type": "ephemeral"}


def build(messages, drop_params=False, extras=None, **params):
    return anthropic.build_request(
        "claude-sonnet-4-6",
        messages,
        params,
        extras or {},
        drop_params,
    )[0]


def message_response(content, stop_reason="end_turn", usage=None):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }


class TestBuildRequest:
    def test_system_messages_are_hoisted_with_cache_markers(self):
        body = build(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Be brief.",
                            "cache_control": CACHE,
                        }
                    ],
                },
                {"role": "developer", "content": "Use metric units."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert body["system"] == [
            {
                "type": "text",
                "text": "Be brief.",
                "cache_control": CACHE,
            },
            {"type": "text", "text": "Use metric units."},
        ]
        assert body["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}],
            }
        ]
        assert body["max_tokens"] == anthropic.DEFAULT_MAX_TOKENS

    def test_message_level_cache_markers_reach_the_last_block(self):
        body = build(
            [
                {
                    "role": "system",
                    "content": "Handbook",
                    "cache_control": CACHE,
                },
                {
                    "role": "user",
                    "content": "Which?",
                    "cache_control": CACHE,
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "a"}}
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t1",
                    "content": "Policy 7",
                    "cache_control": CACHE,
                },
            ]
        )
        assert body["system"] == [
            {
                "type": "text",
                "text": "Handbook",
                "cache_control": CACHE,
            }
        ]
        user, _, results = body["messages"]
        assert user["content"] == [
            {"type": "text", "text": "Which?", "cache_control": CACHE}
        ]
        assert results["content"][0]["cache_control"] == CACHE

    def test_images_become_base64_or_url_sources(self):
        body = build(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,AAAA"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/cat.jpg"
                            },
                        },
                    ],
                }
            ]
        )
        blocks = body["messages"][0]["content"]
        assert blocks[1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        }
        assert blocks[2] == {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/cat.jpg",
            },
        }

    def test_image_media_type_comes_from_the_bytes(self):
        images = [
            ("image/jpeg", b"\x89PNG\r\n\x1a\n" + bytes(8)),
            ("image/jpg", b"\xff\xd8\xff\xe0" + bytes(12)),
            ("image/png", b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            ("image/jpeg", b"GIF89a" + bytes(10)),
        ]
        uris = [
            f"data:{label};base64,{base64.b64encode(raw).decode()}"
            for label, raw in images
        ]
        body = build(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": u}}
                        for u in uris[:3]
                    ]
                    + [
                        {
                            "type": "file",
                            "file": {"file_data": uris[3]},
                        }
                    ],
                }
            ]
        )
        assert [
            (b["type"], b["source"]["media_type"])
            for b in body["messages"][0]["content"]
        ] == [
            ("image", "image/png"),
            ("image", "image/jpeg"),
            ("image", "image/webp"),
            ("image", "image/gif"),
        ]

    def test_file_urls_become_url_document_sources(self):
        url = "https://example.com/report.pdf"
        body = build(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_id": url}},
                        {"type": "file", "file": {"file_data": url}},
                        {
                            "type": "file",
                            "file": {"file_id": "file_011"},
                        },
                        {
                            "type": "file",
                            "file": {
                                "file_data": "JVBERi0=",
                                "file_id": url,
                            },
                        },
                    ],
                }
            ]
        )
        assert [
            b["source"] for b in body["messages"][0]["content"]
        ] == [
            {"type": "url", "url": url},
            {"type": "url", "url": url},
            {"type": "file", "file_id": "file_011"},
            {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0=",
            },
        ]

    def test_empty_text_blocks_are_dropped(self):
        body = build(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "x"},
                    ],
                }
            ]
        )
        assert body["messages"][0]["content"] == [
            {"type": "text", "text": "x"}
        ]

    def test_tool_calls_and_results_round_trip(self):
        body = build(
            [
                {
                    "role": "user",
                    "content": "Weather in Paris and Rome?",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "Two calls.",
                            "signature": "sig",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        },
                        {
                            "id": "t2",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": "not json",
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "t1",
                    "content": "Sunny",
                },
                {
                    "role": "tool",
                    "tool_call_id": "t2",
                    "content": [
                        {
                            "type": "text",
                            "text": "Rain",
                            "cache_control": CACHE,
                        }
                    ],
                },
            ]
        )
        assistant = body["messages"][1]
        assert assistant["content"][0] == {
            "type": "thinking",
            "thinking": "Two calls.",
            "signature": "sig",
        }
        assert assistant["content"][1] == {
            "type": "tool_use",
            "id": "t1",
            "name": "weather",
            "input": {"city": "Paris"},
        }
        assert assistant["content"][2]["input"] == {}
        results = body["messages"][2]
        assert len(body["messages"]) == 3
        assert results["role"] == "user"
        assert results["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [{"type": "text", "text": "Sunny"}],
        }
        assert results["content"][1]["content"] == [
            {"type": "text", "text": "Rain", "cache_control": CACHE}
        ]

    def test_tools_and_tool_choice_translate(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "description": "A tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "b", "parameters": None},
                "cache_control": CACHE,
            },
        ]
        body = build(
            [{"role": "user", "content": "Go"}],
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
        )
        assert body["tools"][0] == {
            "name": "a",
            "description": "A tool",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        }
        assert body["tools"][1] == {
            "name": "b",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": CACHE,
        }
        assert body["tool_choice"] == {
            "type": "any",
            "disable_parallel_tool_use": True,
        }

    @pytest.mark.parametrize(
        "choice, expected",
        [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            (
                {"type": "function", "function": {"name": "a"}},
                {"type": "tool", "name": "a"},
            ),
        ],
    )
    def test_tool_choice_values(self, choice, expected):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "parameters": {"type": "object"},
                },
            }
        ]
        assert (
            build(
                [{"role": "user", "content": "Go"}],
                tools=tools,
                tool_choice=choice,
            )["tool_choice"]
            == expected
        )

    def test_reasoning_effort_becomes_thinking_budget(self):
        body = build(
            [{"role": "user", "content": "Think"}],
            reasoning_effort="medium",
            max_tokens=1000,
        )
        assert body["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }
        assert body["max_tokens"] == 2048 + 1024

    @pytest.mark.parametrize(
        "model",
        ["claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-4-8"],
    )
    def test_reasoning_effort_becomes_adaptive_thinking(self, model):
        body, _ = anthropic.build_request(
            model,
            [{"role": "user", "content": "Think"}],
            {"reasoning_effort": "minimal", "max_tokens": 1000},
            {"output_config": {"format": {"type": "json_schema"}}},
            False,
        )
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {
            "effort": "low",
            "format": {"type": "json_schema"},
        }
        assert body["max_tokens"] == 1000

    def test_reasoning_effort_none_sends_no_thinking(self):
        assert "thinking" not in build(
            [{"role": "user", "content": "Hi"}],
            reasoning_effort="none",
        )

    def test_explicit_thinking_wins(self):
        body = build(
            [{"role": "user", "content": "Hi"}],
            reasoning_effort="high",
            max_tokens=20000,
            extras={
                "thinking": {"type": "enabled", "budget_tokens": 5000}
            },
        )
        assert body["thinking"]["budget_tokens"] == 5000
        assert body["max_tokens"] == 20000

    @pytest.mark.parametrize(
        "model", ["claude-opus-4-7", "claude-opus-5-5"]
    )
    def test_drop_params_turns_a_thinking_budget_adaptive(
        self, model
    ):
        def request(drop_params):
            return anthropic.build_request(
                model,
                [{"role": "user", "content": "Think"}],
                {"reasoning_effort": "low", "max_tokens": 1000},
                {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 4000,
                        "display": "summarized",
                    }
                },
                drop_params,
            )[0]

        body = request(True)
        assert body["thinking"] == {
            "type": "adaptive",
            "display": "summarized",
        }
        assert body["output_config"] == {"effort": "low"}
        assert body["max_tokens"] == 1000
        assert request(False)["thinking"]["budget_tokens"] == 4000

    @pytest.mark.parametrize(
        ("output_config", "expected"),
        [
            (None, {"effort": "high"}),
            ({"effort": "low"}, {"effort": "low"}),
        ],
    )
    def test_reasoning_effort_sets_effort_for_explicit_adaptive_thinking(
        self, output_config, expected
    ):
        extras = {"thinking": {"type": "adaptive"}}
        if output_config:
            extras["output_config"] = output_config
        body, _ = anthropic.build_request(
            "claude-opus-4-8",
            [{"role": "user", "content": "Think"}],
            {"reasoning_effort": "high", "max_tokens": 1000},
            extras,
            False,
        )
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == expected

    def test_drop_params_removes_thinking_for_old_models(self):
        body, _ = anthropic.build_request(
            "claude-3-5-haiku-20241022",
            [{"role": "user", "content": "Hi"}],
            {"reasoning_effort": "low"},
            {},
            True,
        )
        assert "thinking" not in body

    def test_sampling_parameters(self):
        body = build(
            [{"role": "user", "content": "Hi"}],
            temperature=1.7,
            stop="END",
            user="u-1",
            max_completion_tokens=50,
        )
        assert body["temperature"] == 1.0
        assert body["stop_sequences"] == ["END"]
        assert body["metadata"] == {"user_id": "u-1"}
        assert body["max_tokens"] == 50

    def test_drop_params_drops_top_p_alongside_temperature(self):
        assert "top_p" not in build(
            [{"role": "user", "content": "Hi"}],
            drop_params=True,
            temperature=0.5,
            top_p=0.9,
        )
        assert (
            build(
                [{"role": "user", "content": "Hi"}],
                temperature=0.5,
                top_p=0.9,
            )["top_p"]
            == 0.9
        )

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-4-7"],
    )
    def test_sampling_parameters_are_refused_by_new_models(
        self, model
    ):
        def request(drop_params):
            return anthropic.build_request(
                model,
                [{"role": "user", "content": "Hi"}],
                {"temperature": 0.2, "top_p": 0.9},
                {"top_k": 5},
                drop_params,
            )[0]

        body = request(True)
        assert not {"temperature", "top_p", "top_k"} & body.keys()
        with pytest.raises(
            rh.UnsupportedParamsError,
            match="temperature, top_p, top_k",
        ):
            request(False)
        old = build(
            [{"role": "user", "content": "Hi"}],
            extras={"top_k": 5},
            temperature=0.2,
            top_p=0.9,
        )
        assert (old["temperature"], old["top_p"], old["top_k"]) == (
            0.2,
            0.9,
            5,
        )

    def test_thinking_limits_sampling_and_forced_tool_choice(self):
        turns = [{"role": "user", "content": "Go"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "parameters": {"type": "object"},
                },
            }
        ]
        params = {
            "reasoning_effort": "low",
            "temperature": 0.2,
            "top_p": 0.9,
            "tools": tools,
            "tool_choice": "required",
        }
        body = build(turns, True, {"top_k": 5}, **params)
        assert not {"temperature", "top_p", "top_k"} & body.keys()
        assert body["tool_choice"] == {"type": "auto"}
        with pytest.raises(
            rh.UnsupportedParamsError,
            match="temperature, top_p, top_k with thinking on",
        ):
            build(turns, False, {"top_k": 5}, **params)
        with pytest.raises(
            rh.UnsupportedParamsError, match="forced tool_choice"
        ):
            build(
                turns,
                reasoning_effort="low",
                tools=tools,
                tool_choice="required",
            )
        body = build(
            turns, reasoning_effort="low", temperature=1, top_p=0.95
        )
        assert (body["temperature"], body["top_p"]) == (1.0, 0.95)
        body = build(
            turns,
            extras={"thinking": {"type": "adaptive"}},
            tools=tools,
            tool_choice="required",
        )
        assert body["tool_choice"] == {"type": "any"}

    def test_extras_respect_drop_params(self):
        extras = {"top_k": 5, "unknown_field": 1}
        assert build(
            [{"role": "user", "content": "Hi"}],
            drop_params=True,
            extras=extras,
        ) == {
            **build([{"role": "user", "content": "Hi"}]),
            "top_k": 5,
        }
        assert (
            build([{"role": "user", "content": "Hi"}], extras=extras)[
                "unknown_field"
            ]
            == 1
        )

    def test_web_search_options_add_the_web_search_tool(self):
        tool = {"type": "function", "function": {"name": "a"}}
        body = build(
            [{"role": "user", "content": "News?"}],
            tools=[tool],
            web_search_options={
                "search_context_size": "high",
                "user_location": {
                    "type": "approximate",
                    "approximate": {"city": "Paris", "country": "FR"},
                },
            },
        )
        assert [t["name"] for t in body["tools"]] == [
            "a",
            "web_search",
        ]
        assert body["tools"][1] == {
            "type": "web_search_20250305",
            "name": "web_search",
            "user_location": {
                "type": "approximate",
                "city": "Paris",
                "country": "FR",
            },
            "max_uses": 10,
        }
        assert build(
            [{"role": "user", "content": "News?"}],
            web_search_options={},
        )["tools"] == [
            {"type": "web_search_20250305", "name": "web_search"}
        ]

    @pytest.mark.parametrize(
        "model", ["claude-sonnet-4-6", "claude-opus-5"]
    )
    def test_json_schema_uses_forced_tool(self, model):
        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "x",
                "schema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                },
            },
        }
        body, json_mode = anthropic.build_request(
            model,
            [{"role": "user", "content": "Hi"}],
            {"response_format": fmt},
            {},
            False,
        )
        assert json_mode
        assert body["tools"][-1]["name"] == anthropic.JSON_TOOL_NAME
        assert body["tool_choice"] == {
            "type": "tool",
            "name": anthropic.JSON_TOOL_NAME,
        }

    @pytest.mark.parametrize(
        "choice",
        ["required", {"type": "function", "function": {"name": "a"}}],
    )
    def test_forced_tool_choice_on_models_that_reject_it(
        self, choice
    ):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "parameters": {"type": "object"},
                },
            }
        ]
        params = {
            "tools": tools,
            "tool_choice": choice,
            "parallel_tool_calls": False,
        }
        turns = [{"role": "user", "content": "Go"}]
        body, _ = anthropic.build_request(
            "claude-sonnet-5-5", turns, params, {}, True
        )
        assert body["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }
        with pytest.raises(
            rh.UnsupportedParamsError, match="claude-opus-5-5"
        ):
            anthropic.build_request(
                "claude-opus-5-5", turns, params, {}, False
            )
        body, _ = anthropic.build_request(
            "claude-fable-5", turns, params, {}, False
        )
        assert body["tool_choice"]["type"] in ("any", "tool")

    def test_messages_url(self):
        assert (
            anthropic.messages_url("https://api.anthropic.com")
            == "https://api.anthropic.com/v1/messages"
        )
        assert (
            anthropic.messages_url("https://proxy.local/v1/")
            == "https://proxy.local/v1/messages"
        )


class TestResponseTranslation:
    def test_text_tool_use_and_thinking(self):
        completion = anthropic.to_chat_completion(
            message_response(
                [
                    {
                        "type": "thinking",
                        "thinking": "Plan.",
                        "signature": "sig",
                    },
                    {"type": "text", "text": "Calling a tool."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "weather",
                        "input": {"city": "Paris"},
                    },
                ],
                stop_reason="tool_use",
            ),
            "claude-sonnet-4-6",
            False,
        )
        choice = completion.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content == "Calling a tool."
        assert choice.message.tool_calls[0].function.name == "weather"
        assert json.loads(
            choice.message.tool_calls[0].function.arguments
        ) == {"city": "Paris"}
        assert choice.message.reasoning_content == "Plan."
        assert choice.message.thinking_blocks == [
            {
                "type": "thinking",
                "thinking": "Plan.",
                "signature": "sig",
            }
        ]
        assert completion.model_dump()["choices"][0]["message"][
            "thinking_blocks"
        ]

    def test_usage_counts_cache_reads_and_writes_as_prompt_tokens(
        self,
    ):
        completion = anthropic.to_chat_completion(
            message_response(
                [{"type": "text", "text": "OK"}],
                usage={
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 100,
                },
            ),
            "claude-sonnet-4-6",
            False,
        )
        usage = completion.usage
        assert usage.prompt_tokens == 1010
        assert usage.completion_tokens == 3
        assert usage.total_tokens == 1013
        assert usage.prompt_tokens_details.cached_tokens == 900
        assert usage.cache_creation_input_tokens == 100

    @pytest.mark.parametrize(
        "stop_reason, finish",
        [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
            ("refusal", "content_filter"),
        ],
    )
    def test_finish_reasons(self, stop_reason, finish):
        completion = anthropic.to_chat_completion(
            message_response(
                [{"type": "text", "text": "x"}],
                stop_reason=stop_reason,
            ),
            "m",
            False,
        )
        assert completion.choices[0].finish_reason == finish

    def test_json_mode_returns_tool_input_as_content(self):
        completion = anthropic.to_chat_completion(
            message_response(
                [
                    {
                        "type": "tool_use",
                        "id": "t",
                        "name": anthropic.JSON_TOOL_NAME,
                        "input": {"a": 1},
                    }
                ],
                stop_reason="tool_use",
            ),
            "m",
            True,
        )
        assert json.loads(completion.choices[0].message.content) == {
            "a": 1
        }
        assert completion.choices[0].message.tool_calls is None
        assert completion.choices[0].finish_reason == "stop"


STREAM_EVENTS = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_9",
                "model": "claude-sonnet-4-6",
                "usage": {
                    "input_tokens": 20,
                    "cache_read_input_tokens": 5,
                    "output_tokens": 1,
                },
            },
        },
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Hmm."},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
    ),
    (
        "content_block_stop",
        {"type": "content_block_stop", "index": 0},
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    ("ping", {"type": "ping"}),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Hel"},
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "lo"},
        },
    ),
    (
        "content_block_stop",
        {"type": "content_block_stop", "index": 1},
    ),
    (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "t1",
                "name": "weather",
                "input": {},
            },
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"city": ',
            },
        },
    ),
    (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '"Paris"}',
            },
        },
    ),
    (
        "content_block_stop",
        {"type": "content_block_stop", "index": 2},
    ),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 42},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


def collect(chunks):
    text, reasoning, tool_args, tool_names, finish, usage = (
        "",
        "",
        {},
        {},
        None,
        None,
    )
    for chunk in chunks:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        text += delta.content or ""
        reasoning += getattr(delta, "reasoning_content", None) or ""
        for call in delta.tool_calls or []:
            if call.function and call.function.name:
                tool_names[call.index] = call.function.name
            tool_args[call.index] = tool_args.get(call.index, "") + (
                (call.function and call.function.arguments) or ""
            )
        finish = choice.finish_reason or finish
    return text, reasoning, tool_names, tool_args, finish, usage


class TestStreamTranslation:
    def test_full_event_sequence(self):
        translator = anthropic.StreamTranslator(
            "claude-sonnet-4-6", include_usage=True, json_mode=False
        )
        chunks = [
            c
            for event, data in STREAM_EVENTS
            for c in translator.handle(event, data)
        ]
        text, reasoning, names, args, finish, usage = collect(chunks)
        assert chunks[0].choices[0].delta.role == "assistant"
        assert chunks[0].id == "msg_9"
        assert text == "Hello"
        assert reasoning == "Hmm."
        assert names == {0: "weather"}
        assert json.loads(args[0]) == {"city": "Paris"}
        assert finish == "tool_calls"
        assert usage.prompt_tokens == 25
        assert usage.completion_tokens == 42
        assert usage.prompt_tokens_details.cached_tokens == 5

    @pytest.mark.parametrize(
        "json_mode,name",
        [(False, "now"), (True, anthropic.JSON_TOOL_NAME)],
    )
    def test_tool_without_arguments_streams_empty_object(
        self, json_mode, name
    ):
        translator = anthropic.StreamTranslator(
            "m", include_usage=False, json_mode=json_mode
        )
        block = {"type": "tool_use", "id": "t1", "name": name}
        events = [
            (
                "content_block_start",
                {"index": 0, "content_block": {**block, "input": {}}},
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": "",
                    },
                },
            ),
            ("content_block_stop", {"index": 0}),
        ]
        chunks = [
            c for e, d in events for c in translator.handle(e, d)
        ]
        text, _, names, args, _, _ = collect(chunks)
        if json_mode:
            assert names == {}
            assert json.loads(text) == {}
        else:
            assert names == {0: "now"}
            assert json.loads(args[0]) == {}

    def test_server_tool_input_is_not_streamed_as_a_tool_call(self):
        translator = anthropic.StreamTranslator(
            "m", include_usage=False, json_mode=False
        )
        events = []
        for index, kind, name, partial in (
            (0, "server_tool_use", "web_search", '{"query": "x"}'),
            (1, "tool_use", "book_trip", '{"city": "Paris"}'),
        ):
            block = {"type": kind, "id": f"b{index}", "name": name}
            delta = {
                "type": "input_json_delta",
                "partial_json": partial,
            }
            events += [
                (
                    "content_block_start",
                    {"index": index, "content_block": block},
                ),
                (
                    "content_block_delta",
                    {"index": index, "delta": delta},
                ),
                ("content_block_stop", {"index": index}),
            ]
        chunks = [
            c for e, d in events for c in translator.handle(e, d)
        ]
        _, _, names, args, _, _ = collect(chunks)
        assert names == {0: "book_trip"}
        assert args == {0: '{"city": "Paris"}'}

    def test_no_usage_chunk_unless_requested(self):
        translator = anthropic.StreamTranslator(
            "m", include_usage=False, json_mode=False
        )
        chunks = [
            c
            for event, data in STREAM_EVENTS
            for c in translator.handle(event, data)
        ]
        assert all(c.choices for c in chunks)

    def test_error_event_raises_mapped_exception(self):
        translator = anthropic.StreamTranslator(
            "m", include_usage=False, json_mode=False
        )
        with pytest.raises(
            rh.ServiceUnavailableError, match="Overloaded"
        ):
            translator.handle(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Overloaded",
                    },
                },
            )

    def test_sse_parser_handles_comments_and_multiline_data(self):
        lines = [
            ": keepalive",
            "event: a",
            'data: {"x":',
            "data: 1}",
            "",
            "data: {}",
            "",
        ]
        assert list(anthropic.iter_sse(iter(lines))) == [
            ("a", '{"x":\n1}'),
            (None, "{}"),
        ]


def messages(text="Hi"):
    return [{"role": "user", "content": text}]


class TestCompletionThroughAnthropic:
    def test_non_streaming_call(self, mock_http):
        client, recorder = mock_http(
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": "Hello"}]
                ),
            )
        )
        response = rh.completion(
            model="claude-sonnet-4-6",
            messages=messages(),
            max_tokens=100,
            client=client,
            extra_headers={"anthropic-beta": "x"},
        )
        assert response.choices[0].message.content == "Hello"
        request = recorder.requests[0]
        assert (
            str(request.url)
            == "https://api.anthropic.com/v1/messages"
        )
        assert (
            request.headers["x-api-key"] == "test-anthropic_api_key"
        )
        assert (
            request.headers["anthropic-version"]
            == anthropic.ANTHROPIC_VERSION
        )
        assert request.headers["anthropic-beta"] == "x"
        assert recorder.last_json["model"] == "claude-sonnet-4-6"
        assert recorder.last_json["max_tokens"] == 100

    def test_json_schema_uses_structured_outputs_on_claude_5_5(
        self, mock_http
    ):
        client, recorder = mock_http(
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": '{"a": 1}'}]
                ),
            )
        )
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        response = rh.completion(
            model="claude-sonnet-5-5",
            messages=messages(),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": schema},
            },
            reasoning_effort="high",
            client=client,
        )
        assert json.loads(response.choices[0].message.content) == {
            "a": 1
        }
        sent = recorder.last_json
        assert "tools" not in sent
        assert "tool_choice" not in sent
        assert sent["output_config"] == {
            "format": {"type": "json_schema", "schema": schema},
            "effort": "high",
        }

    def test_provider_prefix_is_stripped(self, mock_http):
        client, recorder = mock_http(
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": "x"}]
                ),
            )
        )
        rh.completion(
            model="anthropic/claude-sonnet-4-6",
            messages=messages(),
            client=client,
        )
        assert recorder.last_json["model"] == "claude-sonnet-4-6"

    def test_streaming_call(self, mock_http):
        client, _ = mock_http(
            httpx.Response(
                200,
                content=sse(STREAM_EVENTS),
                headers={"content-type": "text/event-stream"},
            )
        )
        stream = rh.completion(
            model="claude-sonnet-4-6",
            messages=messages(),
            stream=True,
            stream_options={"include_usage": True},
            client=client,
        )
        text, _, names, _, finish, usage = collect(stream)
        assert text == "Hello"
        assert names == {0: "weather"}
        assert finish == "tool_calls"
        assert usage.completion_tokens == 42

    def test_error_status_maps_to_exception(self, mock_http):
        client, _ = mock_http(
            httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "prompt is too long: 300000 tokens",
                    },
                },
            )
        )
        with pytest.raises(rh.ContextWindowExceededError) as raised:
            rh.completion(
                model="claude-sonnet-4-6",
                messages=messages(),
                client=client,
            )
        assert isinstance(raised.value, openai.BadRequestError)
        assert raised.value.status_code == 400
        assert raised.value.llm_provider == "anthropic"
        assert "prompt is too long" in str(raised.value)

    def test_auth_error(self, mock_http):
        client, _ = mock_http(
            httpx.Response(
                401,
                json={
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "invalid x-api-key",
                    },
                },
            )
        )
        with pytest.raises(rh.AuthenticationError):
            rh.completion(
                model="claude-sonnet-4-6",
                messages=messages(),
                client=client,
                num_retries=3,
            )

    def test_retries_rate_limits_then_succeeds(self, mock_http):
        client, recorder = mock_http(
            httpx.Response(
                429,
                json={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "slow down",
                    },
                },
                headers={"retry-after": "1"},
            ),
            httpx.Response(
                529,
                json={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "busy",
                    },
                },
            ),
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": "Finally"}]
                ),
            ),
        )
        response = rh.completion(
            model="claude-sonnet-4-6",
            messages=messages(),
            client=client,
            num_retries=2,
        )
        assert response.choices[0].message.content == "Finally"
        assert len(recorder.requests) == 3

    def test_retries_exhausted_raises(self, mock_http):
        client, recorder = mock_http(
            httpx.Response(
                529,
                json={
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "busy",
                    },
                },
            )
        )
        with pytest.raises(rh.ServiceUnavailableError):
            rh.completion(
                model="claude-sonnet-4-6",
                messages=messages(),
                client=client,
                num_retries=1,
            )
        assert len(recorder.requests) == 2

    def test_connection_error_maps(self, mock_http):
        def fail(request):
            raise httpx.ConnectError("refused", request=request)

        client, _ = mock_http(fail)
        with pytest.raises(rh.APIConnectionError):
            rh.completion(
                model="claude-sonnet-4-6",
                messages=messages(),
                client=client,
                num_retries=0,
            )

    async def test_async_streaming_call(self, mock_http):
        client, _ = mock_http(
            httpx.Response(200, content=sse(STREAM_EVENTS)),
            is_async=True,
        )
        stream = await rh.acompletion(
            model="claude-sonnet-4-6",
            messages=messages(),
            stream=True,
            client=client,
        )
        text = ""
        async for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].delta.content or ""
        assert text == "Hello"

    async def test_async_non_streaming_call(self, mock_http):
        client, _ = mock_http(
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": "Async"}]
                ),
            ),
            is_async=True,
        )
        response = await rh.acompletion(
            model="claude-sonnet-4-6",
            messages=messages(),
            client=client,
        )
        assert response.choices[0].message.content == "Async"

    def test_gzip_compresses_the_body_on_every_attempt(
        self, mock_http
    ):
        client, recorder = mock_http(
            httpx.Response(529, json={"error": {"type": "x"}}),
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": "Hi"}]
                ),
            ),
        )
        response = rh.completion(
            model="claude-sonnet-4-6",
            messages=messages(),
            gzip=True,
            client=client,
        )
        assert response.choices[0].message.content == "Hi"
        assert len(recorder.requests) == 2
        for request in recorder.requests:
            assert request.headers["content-encoding"] == "gzip"
            body = json.loads(gzip.decompress(request.content))
            assert body["model"] == "claude-sonnet-4-6"

    @pytest.mark.parametrize(
        "env, argument, compressed",
        [
            (None, None, False),
            ("1", None, True),
            ("on", None, True),
            ("0", None, False),
            ("1", False, False),
            (None, True, True),
        ],
    )
    def test_gzip_argument_overrides_routehub_gzip_on(
        self, mock_http, monkeypatch, env, argument, compressed
    ):
        if env is not None:
            monkeypatch.setenv("ROUTEHUB_GZIP_ON", env)
        client, recorder = mock_http(
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": "x"}]
                ),
            )
        )
        rh.completion(
            model="claude-sonnet-4-6",
            messages=messages(),
            gzip=argument,
            client=client,
        )
        encoding = recorder.requests[0].headers.get(
            "content-encoding"
        )
        assert (encoding == "gzip") is compressed
        assert recorder.last_json["model"] == "claude-sonnet-4-6"

    async def test_async_gzip(self, mock_http):
        client, recorder = mock_http(
            httpx.Response(
                200,
                json=message_response(
                    [{"type": "text", "text": "Async"}]
                ),
            ),
            is_async=True,
        )
        await rh.acompletion(
            model="claude-sonnet-4-6",
            messages=messages(),
            gzip=True,
            client=client,
        )
        request = recorder.requests[0]
        assert request.headers["content-encoding"] == "gzip"
        assert recorder.last_json["model"] == "claude-sonnet-4-6"
