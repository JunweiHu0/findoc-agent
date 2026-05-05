"""Error recovery — transient-error retry decorator with exponential backoff.

Uses tenacity (already a transitive dependency via langchain) — zero new packages.
Only retries on transient errors (429/5xx/Timeout/ConnectError); fatal errors
(401/400) fail immediately to avoid wasting API credits.
"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import openai

_TRANSIENT = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_FATAL = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
)


def is_transient(exc: BaseException) -> bool:
    """Check whether an exception is transient (retryable)."""
    return isinstance(exc, _TRANSIENT)


def is_fatal(exc: BaseException) -> bool:
    """Check whether an exception is fatal (should not retry)."""
    return isinstance(exc, _FATAL)


def with_retry(max_attempts: int = 3, base: float = 1.0, cap: float = 30.0):
    """Decorator: retry on transient errors with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts (including the initial call).
        base: Initial wait multiplier in seconds.
        cap: Maximum wait time in seconds.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base, max=cap),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )


def classify_error(exc: Exception) -> dict:
    """Classify an exception for error_log recording.

    Returns a dict with {error_type, message, retryable, fatal}.
    """
    error_type = type(exc).__name__
    return {
        "error_type": error_type,
        "message": str(exc),
        "retryable": is_transient(exc),
        "fatal": is_fatal(exc),
    }
