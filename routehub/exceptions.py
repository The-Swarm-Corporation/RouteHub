"""Provider-agnostic exceptions, named and shaped like litellm's."""

from typing import Any, Optional

import openai
from openai import OpenAIError

from routehub._http import httpx
from routehub.registry import ModelNotMappedError

_PLACEHOLDER_URL = "https://routehub.invalid/v1/chat/completions"

_CONTEXT_WINDOW_PHRASES = (
    "context length",
    "context_length_exceeded",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "input is too long",
    "exceeds the maximum",
)

_CONTENT_POLICY_PHRASES = (
    "content_policy_violation",
    "content policy",
    "safety system",
    "content management policy",
)


def _response(status_code: int, response: Any = None) -> Any:
    """Return the given response, or a placeholder carrying the status code.

    Args:
        status_code (int): HTTP status for the placeholder.
        response (Any): An existing response to reuse.

    Returns:
        Any: An httpx-compatible response.
    """
    if response is not None:
        return response
    return httpx.Response(
        status_code, request=httpx.Request("POST", _PLACEHOLDER_URL)
    )


class _GateStatusError:
    """Shared constructor for status-code exceptions."""

    status_code_default = 500

    def __init__(
        self,
        message: str,
        llm_provider: str = "",
        model: str = "",
        response: Any = None,
        body: Optional[object] = None,
    ) -> None:
        """Build the exception.

        Args:
            message (str): Human-readable error message.
            llm_provider (str): Provider that raised it.
            model (str): Model that was called.
            response (Any): The HTTP response, when there was one.
            body (Optional[object]): The decoded error body.
        """
        self.llm_provider = llm_provider
        self.model = model
        self.message = message
        super().__init__(
            message,
            response=_response(self.status_code_default, response),
            body=body,
        )


class APIError(_GateStatusError, openai.APIStatusError):
    """Any provider error without a more specific class."""


class BadRequestError(_GateStatusError, openai.BadRequestError):
    """The request was malformed or rejected (400)."""

    status_code_default = 400


class UnsupportedParamsError(BadRequestError):
    """A parameter the provider does not support was passed."""


class ContextWindowExceededError(BadRequestError):
    """The prompt is longer than the model's context window."""


class ContentPolicyViolationError(BadRequestError):
    """The provider refused the request on content-policy grounds."""


class AuthenticationError(
    _GateStatusError, openai.AuthenticationError
):
    """Missing or invalid credentials (401)."""

    status_code_default = 401


class PermissionDeniedError(
    _GateStatusError, openai.PermissionDeniedError
):
    """The credentials lack access to the resource (403)."""

    status_code_default = 403


class NotFoundError(_GateStatusError, openai.NotFoundError):
    """The model or endpoint does not exist (404)."""

    status_code_default = 404


class UnprocessableEntityError(
    _GateStatusError, openai.UnprocessableEntityError
):
    """The request was well formed but could not be processed (422)."""

    status_code_default = 422


class RateLimitError(_GateStatusError, openai.RateLimitError):
    """The provider's rate limit was hit (429)."""

    status_code_default = 429


class InternalServerError(
    _GateStatusError, openai.InternalServerError
):
    """The provider failed internally (500 and above)."""

    status_code_default = 500


class ServiceUnavailableError(
    _GateStatusError, openai.InternalServerError
):
    """The provider is overloaded or down (503, 529)."""

    status_code_default = 503


class Timeout(openai.APITimeoutError):
    """The request timed out."""

    def __init__(
        self,
        message: str = "Request timed out.",
        llm_provider: str = "",
        model: str = "",
        request: Any = None,
    ) -> None:
        """Build the exception.

        Args:
            message (str): Human-readable error message.
            llm_provider (str): Provider that was called.
            model (str): Model that was called.
            request (Any): The HTTP request that timed out.
        """
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(
            request=request or httpx.Request("POST", _PLACEHOLDER_URL)
        )
        self.message = message


class APIConnectionError(openai.APIConnectionError):
    """The provider could not be reached."""

    def __init__(
        self,
        message: str = "Connection error.",
        llm_provider: str = "",
        model: str = "",
        request: Any = None,
    ) -> None:
        """Build the exception.

        Args:
            message (str): Human-readable error message.
            llm_provider (str): Provider that was called.
            model (str): Model that was called.
            request (Any): The HTTP request that failed.
        """
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(
            message=message,
            request=request
            or httpx.Request("POST", _PLACEHOLDER_URL),
        )


_BY_STATUS = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    413: ContextWindowExceededError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    503: ServiceUnavailableError,
    529: ServiceUnavailableError,
}

__all__ = [
    "APIConnectionError",
    "APIError",
    "AuthenticationError",
    "BadRequestError",
    "ContentPolicyViolationError",
    "ContextWindowExceededError",
    "InternalServerError",
    "ModelNotMappedError",
    "NotFoundError",
    "OpenAIError",
    "PermissionDeniedError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
    "UnprocessableEntityError",
    "UnsupportedParamsError",
    "exception_for_status",
    "map_exception",
]


def exception_for_status(
    status_code: int,
    message: str,
    llm_provider: str = "",
    model: str = "",
    response: Any = None,
    body: Optional[object] = None,
) -> Exception:
    """Build the exception class that matches an HTTP status and message.

    Args:
        status_code (int): HTTP status returned by the provider.
        message (str): The provider's error message.
        llm_provider (str): Provider that raised it.
        model (str): Model that was called.
        response (Any): The HTTP response.
        body (Optional[object]): The decoded error body; its code
            is matched along with the message.

    Returns:
        Exception: An instance of the matching RouteHub exception.
    """
    cls = _BY_STATUS.get(status_code)
    if cls is None:
        cls = InternalServerError if status_code >= 500 else APIError
    code = body.get("code") if isinstance(body, dict) else None
    lowered = f"{message or ''} {code or ''}".lower()
    if cls is BadRequestError:
        if any(p in lowered for p in _CONTEXT_WINDOW_PHRASES):
            cls = ContextWindowExceededError
        elif any(p in lowered for p in _CONTENT_POLICY_PHRASES):
            cls = ContentPolicyViolationError
    return cls(
        message,
        llm_provider=llm_provider,
        model=model,
        response=_response(status_code, response),
        body=body,
    )


def map_exception(
    error: BaseException, llm_provider: str = "", model: str = ""
) -> BaseException:
    """Convert an OpenAI SDK error into the matching RouteHub exception.

    Args:
        error (BaseException): The error raised by the SDK or HTTP layer.
        llm_provider (str): Provider that was called.
        model (str): Model that was called.

    Returns:
        BaseException: A RouteHub exception, or the error unchanged when it
        is not a provider error.
    """
    if type(error).__module__ == __name__:
        return error
    if isinstance(error, openai.APITimeoutError):
        return Timeout(
            str(error), llm_provider, model, request=error.request
        )
    if isinstance(error, openai.APIConnectionError):
        return APIConnectionError(
            str(error), llm_provider, model, request=error.request
        )
    if isinstance(error, openai.APIStatusError):
        return exception_for_status(
            error.status_code,
            error.message,
            llm_provider,
            model,
            response=error.response,
            body=error.body,
        )
    if type(error) is openai.APIError:
        body = error.body if isinstance(error.body, dict) else {}
        code = str(body.get("code"))
        status = int(code) if code.isdecimal() else 500
        return exception_for_status(
            status if 400 <= status < 600 else 500,
            error.message,
            llm_provider,
            model,
            body=error.body,
        )
    if isinstance(error, httpx.TimeoutException):
        return Timeout(
            str(error) or "Request timed out.",
            llm_provider,
            model,
            request=getattr(error, "_request", None),
        )
    if isinstance(error, httpx.TransportError):
        return APIConnectionError(
            str(error) or "Connection error.",
            llm_provider,
            model,
            request=getattr(error, "_request", None),
        )
    return error
