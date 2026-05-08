"""Error recovery — transient-error retry decorator with exponential backoff / 错误恢复——瞬时错误指数退避重试装饰器。

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
    """Check whether an exception is transient (retryable) / 检查异常是否为瞬时的（可重试）。"""
    return isinstance(exc, _TRANSIENT)


def is_fatal(exc: BaseException) -> bool:
    """Check whether an exception is fatal (should not retry) / 检查异常是否为致命的（不应重试）。"""
    return isinstance(exc, _FATAL)


def with_retry(max_attempts: int = 3, base: float = 1.0, cap: float = 30.0):
    """Decorator: retry on transient errors with exponential backoff / 装饰器：瞬时错误指数退避重试。

    Args / 参数:
        max_attempts: Maximum retry attempts (including the initial call) / 最大重试次数（含首次调用）。
        base: Initial wait multiplier in seconds / 初始等待时间乘数（秒）。
        cap: Maximum wait time in seconds / 最大等待时间（秒）。
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base, max=cap),
        retry=retry_if_exception_type(_TRANSIENT),
        reraise=True,
    )


def classify_error(exc: Exception) -> dict:
    """Classify an exception for error_log recording / 为error_log记录分类异常。

    Returns a dict with {error_type, message, retryable, fatal} / 返回包含 {error_type, message, retryable, fatal} 的字典。
    """
    error_type = type(exc).__name__
    return {
        "error_type": error_type,
        "message": str(exc),
        "retryable": is_transient(exc),
        "fatal": is_fatal(exc),
    }
