"""
Retry strategies with exponential backoff, jitter, and saga compensation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable


# Stripe error codes that are safe to retry
_RETRYABLE_CODES = frozenset(
    {
        "rate_limit",
        "lock_timeout",
        "api_connection_error",
        "api_error",
        "idempotency_error",
        "temporary_service_error",
        "server_error",
        "timeout",
        "503",
        "429",
    }
)

# Stripe error codes that must NOT be retried (permanent failures)
_NON_RETRYABLE_CODES = frozenset(
    {
        "card_declined",
        "insufficient_funds",
        "invalid_card_number",
        "expired_card",
        "incorrect_cvc",
        "do_not_honor",
        "fraudulent",
        "lost_card",
        "stolen_card",
        "pickup_card",
        "restricted_card",
        "authentication_required",
        "currency_not_supported",
        "invalid_account",
        "account_closed",
        "amount_too_large",
        "amount_too_small",
    }
)


def exponential_backoff(
    attempt: int,
    base: float = 0.5,
    maximum: float = 32.0,
) -> float:
    """
    Compute exponential backoff delay in seconds for a given attempt number.

    delay = min(base * 2^attempt, maximum)

    Args:
        attempt: zero-indexed attempt number (0 = first retry)
        base: base delay in seconds (default 0.5)
        maximum: maximum delay in seconds (default 32.0)

    Returns:
        delay in seconds, capped at maximum
    """
    if attempt < 0:
        raise ValueError(f"attempt must be non-negative, got {attempt}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    if maximum <= 0:
        raise ValueError(f"maximum must be positive, got {maximum}")
    return min(base * (2.0 ** attempt), maximum)


def jitter(delay: float, factor: float = 0.25) -> float:
    """
    Apply full jitter to a delay to avoid thundering-herd effects.

    jittered_delay = delay * uniform(1 - factor, 1 + factor)

    Args:
        delay: base delay in seconds
        factor: jitter fraction in [0, 1] (default 0.25 = ±25%)

    Returns:
        jittered delay, always non-negative
    """
    if delay < 0:
        raise ValueError(f"delay must be non-negative, got {delay}")
    if not (0.0 <= factor <= 1.0):
        raise ValueError(f"factor must be in [0, 1], got {factor}")
    jitter_amount = delay * factor * (2.0 * random.random() - 1.0)
    return max(0.0, delay + jitter_amount)


def is_retryable(error_code: str) -> bool:
    """
    Determine whether a Stripe error code represents a retryable condition.

    Non-retryable errors represent permanent declines or invalid state that
    will not succeed on retry.

    Args:
        error_code: Stripe error code string (case-insensitive)

    Returns:
        True if the error is retryable
    """
    normalized = error_code.lower().strip()
    if normalized in _NON_RETRYABLE_CODES:
        return False
    # Explicit retryable list OR unknown codes (fail-safe: treat unknown as retryable)
    return True


@dataclass(frozen=True)
class SagaStep:
    """
    A single step in a saga transaction with a forward action and compensating action.
    """

    name: str
    compensate: Callable[[], Any]


def saga_compensate(steps: list[SagaStep], failed_step: int) -> list[str]:
    """
    Execute compensating transactions for all completed steps up to (but not
    including) the failed step, in reverse order.

    Args:
        steps: ordered list of SagaStep objects representing the saga
        failed_step: zero-indexed position of the step that failed

    Returns:
        list of step names for which compensation was attempted (in execution order)

    Raises:
        ValueError: if failed_step is out of bounds
    """
    if not steps:
        return []
    if not (0 <= failed_step < len(steps)):
        raise ValueError(
            f"failed_step {failed_step} out of bounds for {len(steps)} steps"
        )

    compensated: list[str] = []
    # Compensate steps 0..failed_step-1 in reverse order
    for i in range(failed_step - 1, -1, -1):
        steps[i].compensate()
        compensated.append(steps[i].name)

    return compensated
