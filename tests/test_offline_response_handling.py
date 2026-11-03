"""Runnable application response-handling examples with no provider access."""

import asyncio
import json
import os
import socket
from typing import Any, NoReturn

import pytest
from openai.types.chat import ChatCompletionChunk

import routehub

MODEL = "gpt-5.4-mini"
MESSAGES = [
    {"role": "user", "content": "Return the review decision."}
]


@pytest.fixture(autouse=True)
def offline_only(monkeypatch, fake_env):
    """Clear the shared fixture's fake keys and reject socket access.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restore patches after each test.
        fake_env (None): Run the repository's environment isolation first.
    """
    for name in tuple(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)

    def reject_network(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail before a socket operation can reach a provider.

        Args:
            *args (Any): Socket operation arguments.
            **kwargs (Any): Socket operation keyword arguments.

        Raises:
            AssertionError: Always; this module must stay offline.
        """
        raise AssertionError(
            "Unexpected network access in an offline test"
        )

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)


def chunk_text(chunk: ChatCompletionChunk) -> str:
    """Extract text while allowing role, finish, and usage chunks.

    Args:
        chunk (ChatCompletionChunk): One streamed SDK response.

    Returns:
        str: The content, or an empty string for non-text chunks.
    """
    if not chunk.choices:
        return ""
    return chunk.choices[0].delta.content or ""


@pytest.mark.parametrize(
    "reply", ["Approved.", "", "Approved.  Next step.\n"]
)
def test_completion_content(reply: str) -> None:
    """Preserve the text and stop reason, including empty replies.

    Args:
        reply (str): The canned response to return.
    """
    result = routehub.completion(
        model=MODEL, messages=MESSAGES, mock_response=reply
    )
    assert result.choices[0].message.content == reply
    assert result.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("reply", ["", "Approved.  Next step.\n"])
def test_stream_with_usage_chunk(reply: str) -> None:
    """Reassemble a stream without indexing an empty choices list.

    Args:
        reply (str): The text to reconstruct, including whitespace.
    """
    with routehub.completion(
        model=MODEL,
        messages=MESSAGES,
        mock_response=reply,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        chunks = list(stream)
    assert "".join(chunk_text(chunk) for chunk in chunks) == reply
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert any(
        chunk.choices and chunk.choices[0].finish_reason == "stop"
        for chunk in chunks
    )


def test_async_completion() -> None:
    """Await a non-streaming mock response."""

    async def run() -> None:
        """Await the canned SDK completion without a provider."""
        result = await routehub.acompletion(
            model=MODEL, messages=MESSAGES, mock_response="Approved."
        )
        assert result.choices[0].message.content == "Approved."

    asyncio.run(run())


@pytest.mark.parametrize("reply", ["", "Approved after review."])
def test_async_stream(reply: str) -> None:
    """Consume asynchronous streams, including empty replies.

    Args:
        reply (str): The canned response to consume asynchronously.
    """

    async def run() -> None:
        """Consume the canned stream and close its context."""
        stream = await routehub.acompletion(
            model=MODEL,
            messages=MESSAGES,
            mock_response=reply,
            stream=True,
        )
        async with stream:
            parts = [chunk_text(chunk) async for chunk in stream]
        assert "".join(parts) == reply

    asyncio.run(run())


def test_application_catches_rate_limit() -> None:
    """Catch the exception supplied to mock_response unchanged."""
    simulated = routehub.RateLimitError("simulated limit")
    with pytest.raises(routehub.RateLimitError) as caught:
        routehub.completion(
            model=MODEL, messages=MESSAGES, mock_response=simulated
        )
    assert caught.value is simulated
    assert caught.value.status_code == 429


@pytest.mark.parametrize("reply", ['{"approved": true}', "not JSON"])
def test_application_parses_json(reply: str) -> None:
    """Check application parsing of valid and malformed JSON.

    Args:
        reply (str): Valid JSON or malformed application output.
    """
    result = routehub.completion(
        model=MODEL, messages=MESSAGES, mock_response=reply
    )
    content = result.choices[0].message.content
    assert isinstance(content, str)
    if reply == "not JSON":
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
    else:
        assert json.loads(content) == {"approved": True}


def test_network_guard_is_active() -> None:
    """Verify the guard raises before connecting."""
    with pytest.raises(
        AssertionError, match="Unexpected network access"
    ):
        socket.create_connection(("example.invalid", 443))
