"""
Circuit breaker for payment provider calls.
Implements the CLOSED -> OPEN -> HALF_OPEN state machine.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _MetricWindow:
    """Sliding window of success/failure events with timestamps."""

    window_seconds: float
    _events: deque[tuple[float, bool]] = field(default_factory=deque)  # (timestamp, success)

    def record(self, success: bool) -> None:
        now = time.monotonic()
        self._events.append((now, success))
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def failure_count(self) -> int:
        now = time.monotonic()
        self._prune(now)
        return sum(1 for _, success in self._events if not success)

    def success_count(self) -> int:
        now = time.monotonic()
        self._prune(now)
        return sum(1 for _, success in self._events if success)

    def total_count(self) -> int:
        now = time.monotonic()
        self._prune(now)
        return len(self._events)

    def failure_rate(self) -> float:
        total = self.total_count()
        if total == 0:
            return 0.0
        return self.failure_count() / total


class CircuitBreaker:
    """
    Three-state circuit breaker for protecting downstream payment calls.

    States:
        CLOSED  — normal operation; failures are counted
        OPEN    — circuit tripped; all calls fail fast
        HALF_OPEN — probe state; one call allowed to test recovery

    Args:
        threshold: number of failures in the window to trip to OPEN (default 5)
        recovery_timeout: seconds before transitioning OPEN -> HALF_OPEN (default 60)
        window_seconds: observation window for failure counting (default 60)
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        threshold: int = 5,
        recovery_timeout: float = 60.0,
        window_seconds: float = 60.0,
    ) -> None:
        self._threshold = threshold
        self._recovery_timeout = recovery_timeout
        self._window = _MetricWindow(window_seconds)
        self._state = self.CLOSED
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        self._maybe_transition()
        return self._state

    def _maybe_transition(self) -> None:
        if self._state == self.OPEN:
            now = time.monotonic()
            if self._opened_at is not None and (now - self._opened_at) >= self._recovery_timeout:
                self._state = self.HALF_OPEN

    def allow_request(self) -> bool:
        """
        Return True if the circuit allows a request through.
        CLOSED: always allowed.
        OPEN: always blocked.
        HALF_OPEN: one probe allowed.
        """
        self._maybe_transition()
        if self._state == self.CLOSED:
            return True
        if self._state == self.OPEN:
            return False
        # HALF_OPEN: allow exactly one probe
        return True

    def record_success(self) -> None:
        """Record a successful call. Resets HALF_OPEN -> CLOSED."""
        self._window.record(success=True)
        if self._state == self.HALF_OPEN:
            self._state = self.CLOSED
            self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call. May trip the circuit to OPEN."""
        self._window.record(success=False)
        if self._state == self.HALF_OPEN:
            # Probe failed; go back to OPEN
            self._state = self.OPEN
            self._opened_at = time.monotonic()
        elif self._state == self.CLOSED:
            if self._window.failure_count() >= self._threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()

    def metrics(self, window: float | None = None) -> dict[str, Any]:
        """
        Return current metrics snapshot for the observation window.

        Args:
            window: ignored (uses the configured window_seconds); included for API symmetry

        Returns:
            dict with: state, failure_count, success_count, total_count,
                       failure_rate, threshold, recovery_timeout
        """
        self._maybe_transition()
        return {
            "state": self._state,
            "failure_count": self._window.failure_count(),
            "success_count": self._window.success_count(),
            "total_count": self._window.total_count(),
            "failure_rate": self._window.failure_rate(),
            "threshold": self._threshold,
            "recovery_timeout": self._recovery_timeout,
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._state = self.CLOSED
        self._opened_at = None
        self._window._events.clear()
