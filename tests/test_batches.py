import json

import openai
import pytest
from conftest import chat_completion

import routehub as rh
from routehub._http import httpx

USER = [{"role": "user", "content": "Hi"}]


class TestBatches:
    def test_results_come_back_in_request_order(self, mock_openai):
        def batch(status, **fields):
            return httpx.Response(
                200,
                json={"id": "batch_1", "status": status, **fields},
            )

        def jsonl(*lines):
            return httpx.Response(
                200,
                content="".join(
                    json.dumps(line) + "\n" for line in lines
                ).encode(),
            )

        done = batch(
            "completed",
            output_file_id="file-out",
            error_file_id="file-err",
            request_counts={"total": 2, "completed": 1, "failed": 1},
        )
        client, recorder = mock_openai(
            httpx.Response(200, json={"id": "file-in"}),
            batch("validating"),
            batch("in_progress"),
            done,
            done,
            jsonl(
                {
                    "custom_id": "1",
                    "response": {
                        "status_code": 200,
                        "body": chat_completion(content="Second"),
                    },
                    "error": None,
                }
            ),
            jsonl(
                {
                    "custom_id": "0",
                    "response": {
                        "status_code": 400,
                        "body": {
                            "error": {
                                "message": "maximum context length is 8 tokens"
                            }
                        },
                    },
                    "error": None,
                }
            ),
        )
        results = rh.batched_completions(
            model="gpt-5.4-mini",
            requests=[
                {"messages": USER, "max_tokens": 5},
                {"messages": USER},
            ],
            poll_interval=0,
            client=client,
        )
        upload = recorder.requests[0].content
        assert b'"custom_id":"0"' in upload
        assert b'"max_completion_tokens":5' in upload
        assert recorder.requests[1].url.path == "/v1/batches"
        assert isinstance(results[0], rh.ContextWindowExceededError)
        assert isinstance(
            results[1], openai.types.chat.ChatCompletion
        )
        assert results[1].choices[0].message.content == "Second"

    async def test_abatched_completions_collects_results(
        self, mock_openai
    ):
        done = httpx.Response(
            200,
            json={
                "id": "batch_1",
                "status": "completed",
                "output_file_id": "file-out",
                "request_counts": {"total": 1, "completed": 1},
            },
        )
        line = {
            "custom_id": "0",
            "response": {
                "status_code": 200,
                "body": chat_completion(content="Async"),
            },
        }
        client, _ = mock_openai(
            httpx.Response(200, json={"id": "file-in"}),
            httpx.Response(
                200, json={"id": "batch_1", "status": "validating"}
            ),
            done,
            done,
            httpx.Response(200, content=json.dumps(line).encode()),
        )
        results = await rh.abatched_completions(
            model="gpt-5.4-mini",
            requests=[{"messages": USER}],
            poll_interval=0,
            client=client,
        )
        assert results[0].choices[0].message.content == "Async"

    def test_requests_an_expired_batch_never_ran_are_none(
        self, mock_openai
    ):
        def jsonl(line):
            return httpx.Response(
                200, content=(json.dumps(line) + "\n").encode()
            )

        client, _ = mock_openai(
            httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "expired",
                    "output_file_id": "file-out",
                    "error_file_id": "file-err",
                    "request_counts": {
                        "total": 2,
                        "completed": 1,
                        "failed": 0,
                    },
                },
            ),
            jsonl(
                {
                    "custom_id": "0",
                    "response": {
                        "status_code": 200,
                        "body": chat_completion(content="First"),
                    },
                    "error": None,
                }
            ),
            jsonl(
                {
                    "custom_id": "1",
                    "response": None,
                    "error": {
                        "code": "batch_expired",
                        "message": "This request could not be executed before the completion window expired.",
                    },
                }
            ),
        )
        results = rh.BatchJob("openai", "batch_1", client).results()
        assert results[0].choices[0].message.content == "First"
        assert results[1] is None

    def test_claude_batch_runs_on_message_batches(self, mock_http):
        def batch(status):
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_1",
                    "processing_status": status,
                    "request_counts": {
                        "processing": 0,
                        "succeeded": 1,
                        "errored": 1,
                        "canceled": 0,
                        "expired": 0,
                    },
                },
            )

        lines = [
            {
                "custom_id": "1",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [
                            {"type": "text", "text": "Second"}
                        ],
                        "stop_reason": "end_turn",
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 1,
                        },
                    },
                },
            },
            {
                "custom_id": "0",
                "result": {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "prompt is too long: 9 tokens > 8 maximum",
                        },
                    },
                },
            },
        ]
        client, recorder = mock_http(
            batch("in_progress"),
            batch("in_progress"),
            batch("ended"),
            batch("ended"),
            httpx.Response(
                200,
                content="".join(
                    json.dumps(line) + "\n" for line in lines
                ).encode(),
            ),
        )
        results = rh.batched_completions(
            model="claude-sonnet-4-6",
            requests=[
                {"messages": USER, "max_tokens": 5},
                {"messages": USER},
            ],
            poll_interval=0,
            client=client,
        )
        submitted = json.loads(recorder.requests[0].content)
        assert recorder.requests[0].url.path == "/v1/messages/batches"
        assert [r["custom_id"] for r in submitted["requests"]] == [
            "0",
            "1",
        ]
        assert submitted["requests"][0]["params"]["max_tokens"] == 5
        assert (
            submitted["requests"][0]["params"]["model"]
            == "claude-sonnet-4-6"
        )
        assert (
            recorder.requests[0].headers["x-api-key"]
            == "test-anthropic_api_key"
        )
        assert (
            recorder.requests[-1].url.path
            == "/v1/messages/batches/msgbatch_1/results"
        )
        assert isinstance(results[0], rh.ContextWindowExceededError)
        assert isinstance(
            results[1], openai.types.chat.ChatCompletion
        )
        assert results[1].choices[0].message.content == "Second"

    def test_groq_batch_runs_on_groq_batch_api(self, mock_openai):
        client, recorder = mock_openai(
            httpx.Response(200, json={"id": "file-in"}),
            httpx.Response(
                200, json={"id": "batch_1", "status": "validating"}
            ),
        )
        job = rh.batched_completions(
            model="groq/llama-3.1-8b-instant",
            requests=[{"messages": USER, "max_tokens": 5}],
            wait=False,
            client=client,
        )
        upload = recorder.requests[0].content
        assert b'"model":"llama-3.1-8b-instant"' in upload
        assert b'"url":"/v1/chat/completions"' in upload
        assert b'"max_tokens":5' in upload
        created = json.loads(recorder.requests[1].content)
        assert created["endpoint"] == "/v1/chat/completions"
        assert (job.provider, job.id) == ("groq", "batch_1")
        rebuilt = rh.BatchJob("groq", "batch_1")
        assert (
            str(rebuilt.client.base_url)
            == "https://api.groq.com/openai/v1/"
        )
        assert rebuilt.api_key == "test-groq_api_key"

    def test_groq_batch_results_and_failure(self, mock_openai):
        body = {**chat_completion("Hi"), "service_tier": "on_demand"}
        client, _ = mock_openai(
            httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "completed",
                    "output_file_id": "file-out",
                    "request_counts": {"total": 1},
                },
            ),
            httpx.Response(
                200,
                json={
                    "custom_id": "0",
                    "response": {"status_code": 200, "body": body},
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "failed",
                    "errors": [{"message": "bad method"}],
                },
            ),
        )
        job = rh.BatchJob("groq", "batch_1", client)
        [result] = job.results()
        assert result.choices[0].message.content == "Hi"
        with pytest.raises(rh.BadRequestError, match="bad method"):
            job.results()

    def test_groq_context_error_matches_direct_call(
        self, mock_openai
    ):
        error = {
            "error": {
                "message": "Please reduce the length of the messages or completion.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }
        client, _ = mock_openai(
            httpx.Response(400, json=error),
            httpx.Response(
                200,
                json={
                    "id": "batch_1",
                    "status": "completed",
                    "error_file_id": "file-err",
                    "request_counts": {"total": 1},
                },
            ),
            httpx.Response(
                200,
                json={
                    "custom_id": "0",
                    "response": {"status_code": 400, "body": error},
                },
            ),
        )
        with pytest.raises(rh.ContextWindowExceededError):
            rh.completion(
                model="groq/llama-3.1-8b-instant",
                messages=USER,
                client=client,
            )
        [result] = rh.BatchJob("groq", "batch_1", client).results()
        assert isinstance(result, rh.ContextWindowExceededError)

    def test_claude_batch_is_gzipped_when_asked(self, mock_http):
        client, recorder = mock_http(
            httpx.Response(
                200,
                json={"id": "msgbatch_1", "processing_status": "x"},
            )
        )
        job = rh.batched_completions(
            model="claude-sonnet-4-6",
            requests=[{"messages": USER}],
            wait=False,
            gzip=True,
            client=client,
        )
        assert job.id == "msgbatch_1"
        request = recorder.requests[0]
        assert request.headers["content-encoding"] == "gzip"
        submitted = recorder.last_json["requests"]
        assert submitted[0]["params"]["model"] == "claude-sonnet-4-6"
