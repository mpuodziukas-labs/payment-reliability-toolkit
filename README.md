# Stripe Payment Reliability

Production-grade reliability primitives for payment infrastructure: idempotency, retry strategies, and circuit breaking.

## Modules

- **`reliability/idempotency.py`** — `IdempotencyKey`: deterministic key generation, TTL-backed storage, duplicate detection, expired entry cleanup
- **`reliability/retry.py`** — Exponential backoff with jitter, Stripe error code classification, saga compensation pattern
- **`reliability/circuit_breaker.py`** — `CircuitBreaker`: CLOSED/OPEN/HALF_OPEN state machine, sliding window metrics, configurable threshold and recovery timeout

## Quick Start

```bash
pip install pytest
pytest tests/ -v
```

## Key Design Decisions

- **Idempotency keys**: SHA-256 of sorted JSON payload — deterministic, order-independent, collision-resistant
- **Exponential backoff**: `min(base * 2^attempt, max)` with ±25% jitter to prevent thundering herd
- **Retryable errors**: non-retryable set (card declines, fraud flags) is explicit; all unknown codes default to retryable
- **Saga compensation**: compensating transactions execute in strict reverse order for consistent rollback
- **Circuit breaker**: sliding window failure counting with automatic HALF_OPEN probe after recovery timeout
