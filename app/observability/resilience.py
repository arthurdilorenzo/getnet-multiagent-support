import random
import threading
import time
from typing import Any, Callable, Dict, TypeVar
from app.config import settings
from app.observability.logger import logger

T = TypeVar("T")


class CircuitBreakerOpenError(RuntimeError):
    """Exceção disparada quando o Circuit Breaker está no estado OPEN."""


class ExternalServiceUnavailableError(RuntimeError):
    """Exceção disparada quando uma API externa falha após todas as tentativas."""


class CircuitBreaker:
    """
    Implementação de Circuit Breaker (CLOSED -> OPEN -> HALF_OPEN) thread-safe
    para proteger chamadas a APIs externas (Open-Meteo, AwesomeAPI, DuckDuckGo, Gemini).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = settings.CB_FAILURE_THRESHOLD,
        recovery_timeout_sec: float = settings.CB_RECOVERY_TIMEOUT_SECONDS,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "OPEN":
                if (time.monotonic() - self._last_failure_time) >= self.recovery_timeout_sec:
                    self._state = "HALF_OPEN"
            return self._state

    def allow_request(self) -> bool:
        current = self.state
        return current in ("CLOSED", "HALF_OPEN")

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
                logger.warning(
                    "circuit_breaker_tripped",
                    breaker=self.name,
                    failure_count=self._failure_count,
                    state="OPEN",
                )

    def reset(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._state = "CLOSED"


class CircuitBreakerRegistry:
    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {
            "weather_api": CircuitBreaker("weather_api"),
            "currency_api": CircuitBreaker("currency_api"),
            "web_search_api": CircuitBreaker("web_search_api"),
            "gemini_api": CircuitBreaker("gemini_api"),
        }

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name)
        return self._breakers[name]

    def states(self) -> Dict[str, str]:
        return {name: cb.state for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        for cb in self._breakers.values():
            cb.reset()


breaker_registry = CircuitBreakerRegistry()


def call_with_resilience(
    breaker_name: str,
    fn: Callable[[], T],
    max_retries: int = settings.HTTP_MAX_RETRIES,
    base_delay_sec: float = 0.05,
) -> T:
    """
    Executa `fn` sob proteção de Circuit Breaker e Retries com Exponential Backoff + Jitter.
    """
    breaker = breaker_registry.get(breaker_name)
    if not breaker.allow_request():
        raise CircuitBreakerOpenError(
            f"Circuit Breaker '{breaker_name}' está ABERTO devido a falhas consecutivas recentes."
        )

    last_exc: Exception = RuntimeError("Unknown error")
    for attempt in range(1, max_retries + 2):
        try:
            result = fn()
            breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "external_call_retry_failed",
                breaker=breaker_name,
                attempt=attempt,
                max_attempts=max_retries + 1,
                error_type=type(exc).__name__,
            )
            if attempt <= max_retries:
                sleep_sec = base_delay_sec * (2 ** (attempt - 1)) + random.uniform(0.0, 0.02)
                time.sleep(sleep_sec)

    breaker.record_failure()
    raise ExternalServiceUnavailableError(
        f"Serviço externo '{breaker_name}' indisponível após {max_retries + 1} tentativas."
    ) from last_exc
