"""
Idempotency key management for payment operations.
Ensures duplicate requests return the same result without re-executing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredResult:
    key: str
    result: Any
    created_at: float
    expires_at: float


@dataclass
class IdempotencyKey:
    """
    Manages idempotency keys for safe payment retries.
    Thread-safety note: not safe for concurrent access without external locking.
    """

    _store: dict[str, StoredResult] = field(default_factory=dict)

    def generate(self, payload: dict[str, Any]) -> str:
        """
        Generate a deterministic idempotency key from a payload dict.
        Uses SHA-256 of the sorted JSON representation.
        """
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def store(self, key: str, result: Any, ttl: float = 86400.0) -> None:
        """
        Persist the result for the given idempotency key with a TTL in seconds.
        Overwrites any existing entry.
        """
        now = time.monotonic()
        self._store[key] = StoredResult(
            key=key,
            result=result,
            created_at=now,
            expires_at=now + ttl,
        )

    def retrieve(self, key: str) -> Any | None:
        """
        Retrieve the stored result for a key, or None if not found / expired.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.result

    def is_duplicate(self, key: str) -> bool:
        """
        Return True if the key exists and has not expired.
        """
        return self.retrieve(key) is not None

    def cleanup_expired(self, ttl: float | None = None) -> int:
        """
        Remove all expired entries from the store.
        If ttl is provided, also remove entries older than ttl seconds
        regardless of their individual expiry.

        Returns the number of entries removed.
        """
        now = time.monotonic()
        expired_keys = [
            k for k, v in self._store.items()
            if now > v.expires_at or (ttl is not None and now - v.created_at > ttl)
        ]
        for k in expired_keys:
            del self._store[k]
        return len(expired_keys)

    def __len__(self) -> int:
        return len(self._store)
