from collections import defaultdict, deque
import threading
import time
from typing import Deque, Dict, Tuple
from fastapi import HTTPException, status
from app.config import settings


class SlidingWindowRateLimiter:
    """
    Rate limiter em janela deslizante (Sliding Window) thread-safe
    por user_id e endereço IP, protegendo contra abuso de LLM e DoS.
    """

    def __init__(self, max_requests_per_minute: int = settings.RATE_LIMIT_PER_MINUTE):
        self.max_requests = max_requests_per_minute
        self.window_seconds = 60.0
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_and_consume(self, key: str, limit_override: int = 0) -> Tuple[bool, int, int]:
        """
        Retorna (allowed, remaining, retry_after_seconds).
        """
        limit = limit_override if limit_override > 0 else self.max_requests
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                oldest = bucket[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest)))
                return False, 0, retry_after

            bucket.append(now)
            remaining = max(0, limit - len(bucket))
            return True, remaining, 0

    def enforce(self, key: str, limit_override: int = 0) -> int:
        allowed, remaining, retry_after = self.check_and_consume(key, limit_override)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Limite de requisições excedido para '{key}'. Tente novamente em {retry_after}s.",
                headers={"Retry-After": str(retry_after), "X-RateLimit-Remaining": "0"},
            )
        return remaining

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


rate_limiter = SlidingWindowRateLimiter()
