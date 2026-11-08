"""Chat completions through OpenAI, Groq and Anthropic batch APIs."""

import inspect
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

from routehub._json import dumps, loads
from routehub.clients import http_client, openai_client
from routehub.main import (
    _RETRYABLE_STATUS,
    DEFAULT_NUM_RETRIES,
    _anthropic_prepare,
    _anthropic_send,
    _build_call,
    _openai_client,
    _openai_request,
    _retry_delay,
    _split_args,
    completion,
)
from routehub.providers import get_llm_provider

_BATCH_DONE = ("completed", "failed", "expired", "cancelled", "ended")


@contextmanager
def _mapped_errors(provider: str, model: str = "") -> Iterator[None]:
    """Re-raise SDK errors inside the block as RouteHub exceptions.

    Args:
        provider (str): Provider that was called.
        model (str): Model that was called.

    Yields:
        None: Control to the block.
    """
    from routehub.exceptions import map_exception

    try:
        yield
    except Exception as error:
        mapped = map_exception(error, provider, model)
        if mapped is error:
            raise
        raise mapped from error


@dataclass
class BatchJob:
    """A submitted batch, rebuildable from its provider and id.

    Args:
        provider (str): Provider the batch runs on, "openai",
            "groq" or "anthropic".
        id (str): The provider's batch id.
        client (Any): An OpenAI client for openai and groq, or an
            httpx client for anthropic; built when omitted.
        api_key (Optional[str]): API key; read from the provider's
            environment variable when omitted.
        api_base (Optional[str]): Base URL; read from the provider's
            environment variable or default when omitted.
    """

    provider: str
    id: str
    client: Any = None
    api_key: Optional[str] = field(default=None, repr=False)
    api_base: Optional[str] = None

    def __post_init__(self) -> None:
        """Resolve the key and base URL, and build a missing client."""
        _, self.provider, self.api_key, self.api_base = (
            get_llm_provider(
                self.provider,
                self.provider,
                self.api_base,
                self.api_key,
            )
        )
        if self.client is not None:
            return
        if self.provider == "anthropic":
            self.client = http_client(is_async=False, ssl_verify=True)
        else:
            self.client = openai_client(
                self.provider,
                self.api_key,
                self.api_base,
                is_async=False,
                max_retries=DEFAULT_NUM_RETRIES,
                ssl_verify=True,
            )

    def _anthropic(self, method: str, path: str = "") -> Any:
        """Send a Message Batches request, retrying retryable statuses.

        Args:
            method (str): HTTP method.
            path (str): Path after the batch id, such as "/cancel".

        Returns:
            Any: The successful HTTP response.
        """
        from routehub import anthropic

        for attempt in range(DEFAULT_NUM_RETRIES + 1):
            response = self.client.request(
                method,
                f"{anthropic.messages_url(self.api_base)}/batches/"
                f"{self.id}{path}",
                headers=anthropic.headers(self.api_key, None),
            )
            if (
                response.status_code not in _RETRYABLE_STATUS
                or attempt == DEFAULT_NUM_RETRIES
            ):
                break
            time.sleep(_retry_delay(response, attempt))
        if response.status_code >= 400:
            raise anthropic.response_error(response, "")
        return response

    def status(self) -> str:
        """Return the batch's current status.

        Returns:
            str: The provider's status, such as "in_progress" or
            "completed", or "ended" for anthropic.
        """
        with _mapped_errors(self.provider):
            if self.provider == "anthropic":
                response = self._anthropic("GET")
                return loads(response.content)["processing_status"]
            return self.client.batches.retrieve(self.id).status

    def cancel(self) -> str:
        """Ask the provider to stop the batch.

        Returns:
            str: The status after the request, usually "cancelling",
            or "canceling" for anthropic.
        """
        with _mapped_errors(self.provider):
            if self.provider == "anthropic":
                response = self._anthropic("POST", "/cancel")
                return loads(response.content)["processing_status"]
            return self.client.batches.cancel(self.id).status

    def results(self) -> List[Any]:
        """Download the results of a finished batch.

        Returns:
            List[Any]: A ChatCompletion, or the RouteHub exception for a
            failed request, per request in submission order; None for a
            request with no result, such as one a cancelled batch never
            ran.
        """
        from openai.types.chat import ChatCompletion

        from routehub.exceptions import (
            BadRequestError,
            exception_for_status,
        )

        if self.provider == "anthropic":
            from routehub import anthropic

            with _mapped_errors(self.provider):
                batch = loads(self._anthropic("GET").content)
                if batch["processing_status"] != "ended":
                    raise BadRequestError(
                        f"Batch {self.id} has status "
                        f"{batch['processing_status']}.",
                        llm_provider=self.provider,
                    )
                lines = self._anthropic("GET", "/results").content
            results = [None] * sum(batch["request_counts"].values())
            for raw in lines.splitlines():
                line = loads(raw)
                result = line["result"]
                if result["type"] == "succeeded":
                    results[int(line["custom_id"])] = (
                        anthropic.to_chat_completion(
                            result["message"], "", True
                        )
                    )
                elif result["type"] == "errored":
                    results[int(line["custom_id"])] = (
                        anthropic.stream_error(
                            result["error"]["error"], ""
                        )
                    )
            return results
        with _mapped_errors(self.provider):
            batch = self.client.batches.retrieve(self.id)
            if (
                batch.status == "failed"
                or batch.status not in _BATCH_DONE
            ):
                reasons = (
                    getattr(batch.errors, "data", batch.errors) or []
                )
                raise BadRequestError(
                    f"Batch {self.id} has status {batch.status}. "
                    + " ".join(e.message or "" for e in reasons),
                    llm_provider=self.provider,
                )
            files = [batch.output_file_id, batch.error_file_id]
            texts = [
                self.client.files.content(f).text for f in files if f
            ]
        results: List[Any] = [None] * batch.request_counts.total
        for raw in (r for t in texts for r in t.splitlines()):
            line = loads(raw)
            response = line.get("response")
            if response is None:
                continue
            status = response.get("status_code", 500)
            body = response.get("body") or {}
            if status < 400:
                result = ChatCompletion.construct(**body)
            else:
                error = line.get("error") or body.get("error") or {}
                result = exception_for_status(
                    status,
                    error.get("message", ""),
                    self.provider,
                    body=error,
                )
            results[int(line["custom_id"])] = result
        return results


def batched_completions(
    model: str,
    requests: List[dict],
    *,
    wait: bool = True,
    poll_interval: float = 30,
    timeout: float = 24 * 3600,
    **kwargs: Any,
) -> Any:
    """Run chat completions through OpenAI, Groq or Anthropic batches.

    Args:
        model (str): OpenAI, Groq or Claude model name, such as
            "gpt-5.4-mini", "groq/llama-3.1-8b-instant" or
            "claude-sonnet-4-6".
        requests (List[dict]): completion arguments for each request,
            such as messages and max_tokens.
        wait (bool): Poll until the batch finishes and return its
            results; when False, return a BatchJob at once.
        poll_interval (float): Seconds between status checks.
        timeout (float): Seconds to wait before raising Timeout; the
            batch keeps running and BatchJob can still collect it.
        **kwargs (Any): completion arguments shared by every request,
            such as api_key, temperature or client.

    Returns:
        Any: The results of BatchJob.results when wait is on,
        otherwise the BatchJob.
    """
    from routehub.exceptions import BadRequestError, Timeout

    if not requests:
        raise BadRequestError(
            "requests is required and cannot be empty", model=model
        )
    spec = inspect.signature(completion).parameters
    lines = []
    for i, request in enumerate(requests):
        args = {**kwargs, **request, "model": model}
        named = {
            n: args.pop(n, p.default)
            for n, p in spec.items()
            if n != "kwargs"
        }
        call = _build_call(*_split_args({**named, "kwargs": args}))
        if call.provider == "anthropic":
            url, headers, body, _ = _anthropic_prepare(call)
            lines.append({"custom_id": str(i), "params": body})
            continue
        body, extra_body = _openai_request(call)
        lines.append(
            {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {**body, **extra_body},
            }
        )
    if call.provider == "anthropic":
        client = call.client or http_client(
            is_async=False,
            ssl_verify=call.ssl_verify,
            keepalive_expiry=call.keepalive_expiry,
        )
        response = _anthropic_send(
            client,
            url + "/batches",
            headers,
            {"requests": lines},
            call,
        )
        job = BatchJob(
            call.provider,
            loads(response.content)["id"],
            client,
            call.api_key,
            call.api_base,
        )
    elif call.provider not in ("openai", "groq"):
        raise BadRequestError(
            "batched_completions supports openai, groq and "
            f"anthropic, not {call.provider}.",
            llm_provider=call.provider,
            model=model,
        )
    else:
        with _mapped_errors(call.provider, model):
            client = _openai_client(call, is_async=False)
            upload = client.files.create(
                file=("batch.jsonl", b"\n".join(map(dumps, lines))),
                purpose="batch",
            )
            batch = client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        job = BatchJob(call.provider, batch.id, client)
    if not wait:
        return job
    deadline = time.monotonic() + timeout
    while job.status() not in _BATCH_DONE:
        if time.monotonic() > deadline:
            raise Timeout(
                f"Batch {job.id} is still running after {timeout}s.",
                call.provider,
                model,
            )
        time.sleep(poll_interval)
    return job.results()


async def abatched_completions(
    model: str,
    requests: List[dict],
    *,
    wait: bool = True,
    poll_interval: float = 30,
    timeout: float = 24 * 3600,
    **kwargs: Any,
) -> Any:
    """Async form of batched_completions, with the same arguments.

    Each provider call runs in a worker thread and the wait between
    status checks is an asyncio.sleep, so the event loop stays free.

    Args:
        model (str): OpenAI, Groq or Claude model name.
        requests (List[dict]): completion arguments for each request.
        wait (bool): Poll until the batch finishes; when False, return
            a BatchJob at once.
        poll_interval (float): Seconds between status checks.
        timeout (float): Seconds to wait before raising Timeout.
        **kwargs (Any): completion arguments shared by every request,
            with a sync client as in batched_completions.

    Returns:
        Any: The results, or the BatchJob when wait is off.
    """
    import asyncio

    from routehub.exceptions import Timeout

    job = await asyncio.to_thread(
        batched_completions, model, requests, wait=False, **kwargs
    )
    if not wait:
        return job
    deadline = time.monotonic() + timeout
    while await asyncio.to_thread(job.status) not in _BATCH_DONE:
        if time.monotonic() > deadline:
            raise Timeout(
                f"Batch {job.id} is still running after {timeout}s.",
                job.provider,
                model,
            )
        await asyncio.sleep(poll_interval)
    return await asyncio.to_thread(job.results)
