"""Token-aware rate limiting for API calls."""

import asyncio
import os
import warnings
from collections import deque
from collections.abc import Callable

# Token limits
MAX_TOKENS_PER_MINUTE = int(os.getenv("MAX_TOKENS_PER_MINUTE", "20000"))
REQUEST_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "10.0"))

# Retry configuration
MAX_RETRIES = int(os.getenv("RATE_LIMIT_RETRIES", "5"))
BASE_RETRY_DELAY = float(os.getenv("RATE_LIMIT_BASE_DELAY", "15.0"))
TOKEN_ESTIMATE_MULTIPLIER = float(os.getenv("TOKEN_ESTIMATE_MULTIPLIER", "1.5"))

# Track (tokens, timestamp) pairs for usage in the last minute
_token_usage: deque[tuple[int, float]] = deque(maxlen=10000)

# Async lock to prevent race conditions in check-and-record
_lock: asyncio.Lock | None = None


async def _get_lock() -> asyncio.Lock:
    """Get or create the global lock for rate limiting."""
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def _record_token_usage(tokens: int) -> None:
    """Record token usage with current timestamp."""
    now = asyncio.get_running_loop().time()
    _token_usage.append((tokens, now))


def _get_current_token_usage(now: float) -> int:
    """Get total tokens used in the last 60 seconds."""
    cutoff = now - 60.0
    total = 0
    for tokens, ts in _token_usage:
        if ts >= cutoff:
            total += tokens
    return total


async def _cleanup_old_entries(now: float) -> None:
    """Remove entries older than 60 seconds."""
    cutoff = now - 60.0
    # Remove old entries from the front of the deque
    while _token_usage and _token_usage[0][1] < cutoff:
        _token_usage.popleft()


async def wait_for_token_capacity(
    estimated_tokens: int,
    max_wait: float = 30.0,
) -> None:
    """
    Wait until there's capacity for the estimated token usage.

    Args:
        estimated_tokens: Number of tokens we expect to use
        max_wait: Maximum seconds to wait (default 30s)
    """
    if estimated_tokens <= 0:
        return

    if estimated_tokens > MAX_TOKENS_PER_MINUTE:
        warnings.warn(
            f"Estimated tokens ({estimated_tokens}) exceed per-minute limit ({MAX_TOKENS_PER_MINUTE}). "
            "This request may fail."
        )

    now = asyncio.get_running_loop().time()

    await _cleanup_old_entries(now)

    current_usage = _get_current_token_usage(now)
    available = MAX_TOKENS_PER_MINUTE - current_usage

    if estimated_tokens > available:
        tokens_needed = estimated_tokens - available
        # Calculate wait time: tokens drain at MAX_TOKENS_PER_MINUTE/60 per second
        drain_rate = MAX_TOKENS_PER_MINUTE / 60.0
        seconds_to_wait = tokens_needed / drain_rate

        if seconds_to_wait > max_wait:
            warnings.warn(
                f"Rate limit wait ({seconds_to_wait:.1f}s) exceeds max ({max_wait}s). "
                f"Proceeding anyway - request may fail."
            )
            seconds_to_wait = max_wait

        if seconds_to_wait > 0:
            await asyncio.sleep(seconds_to_wait)


async def with_token_rate_limit(
    func: Callable,
    *args,
    estimated_input_tokens: int = 1000,
    estimated_output_tokens: int = 500,
    request_delay: float | None = None,
    **kwargs,
):
    """
    Execute an async function with token rate limiting and retry on 429 errors.

    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        estimated_input_tokens: Estimated input tokens for the request
        estimated_output_tokens: Estimated output tokens for the response
        request_delay: Delay between requests (uses RATE_LIMIT_DELAY if None)
        **kwargs: Keyword arguments for the function
    """
    if request_delay is None:
        request_delay = REQUEST_DELAY

    safe_input_tokens = int(estimated_input_tokens * TOKEN_ESTIMATE_MULTIPLIER)
    safe_output_tokens = int(estimated_output_tokens * TOKEN_ESTIMATE_MULTIPLIER)

    last_exception = None
    for attempt in range(MAX_RETRIES + 1):
        # Wait for capacity and record usage (under lock)
        # Don't hold lock during sleep or function execution
        async with await _get_lock():
            try:
                # Wait for token capacity for input (with safety buffer)
                await wait_for_token_capacity(safe_input_tokens)

                # Record input token usage immediately (still under lock)
                await _record_token_usage(safe_input_tokens)
            except Exception as e:
                # Error during rate limiting - release lock and retry
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise

        # Delay between requests (outside lock to allow concurrency)
        if request_delay > 0:
            await asyncio.sleep(request_delay)

        # Execute the function (outside lock to allow concurrent API calls)
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            # Check if this is a 429 rate limit error
            error_str = str(e).lower()
            is_rate_limit = (
                "429" in error_str or
                "rate limit" in error_str or
                "rate_limit" in error_str or
                "toomanyrequests" in error_str
            )

            if is_rate_limit:
                last_exception = e
                if attempt < MAX_RETRIES:
                    # Exponential backoff: wait longer each retry
                    wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                    warnings.warn(
                        f"Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Waiting {wait_time:.1f}s before retry."
                    )
                    await asyncio.sleep(wait_time)
                    # Clear old token usage to be conservative on retry
                    now = asyncio.get_running_loop().time()
                    await _cleanup_old_entries(now)
                    continue
                else:
                    warnings.warn(
                        f"Rate limit retries exhausted. Last error: {e}"
                    )
            # Re-raise non-rate-limit errors or exhausted retries
            raise last_exception or e
            return None  # type: ignore  # Should never reach here

        # Record output token usage
        await _record_token_usage(safe_output_tokens)

        return result

    return None  # Should never reach here


def estimate_tokens(text: str) -> int:
    """
    Roughly estimate token count from text.
    Conservative estimate: ~3 characters per token for English text.
    """
    if not text:
        return 0
    return len(text) // 3


def estimate_json_tokens(data: dict, max_depth: int = 3) -> int:
    """
    Estimate token count for JSON-serializable data.
    Truncates nested data to avoid counting everything.
    """
    import json

    def truncate_data(d, depth):
        if depth <= 0:
            return {}

        if isinstance(d, dict):
            return {k: truncate_data(v, depth - 1) for k, v in list(d.items())[:10]}
        elif isinstance(d, list):
            return [truncate_data(item, depth - 1) for item in d[:5]]
        elif isinstance(d, str):
            return d[:500]
        else:
            return d

    truncated = truncate_data(data, max_depth)
    return estimate_tokens(json.dumps(truncated))


def get_log_token_estimate(messages: list) -> int:
    """Estimate token count for a log's messages."""

    # Create a simplified version for estimation
    simplified = []
    for m in messages:
        parts = []
        for part in m["parts"]:
            kind = part.get("part_kind", "")
            if kind in ("user-prompt", "text"):
                content = part.get("content", "")
                parts.append({"kind": kind, "content": content[:500]})
            elif kind == "tool-call":
                parts.append({
                    "kind": kind,
                    "function_name": part.get("function_name", "unknown")
                })
            elif kind == "tool-return":
                parts.append({"kind": kind, "count": len(part.get("content", []))})

        if parts:
            simplified.append({"kind": m["kind"], "parts": parts})

    return estimate_json_tokens(simplified)
