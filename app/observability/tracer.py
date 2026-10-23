import time
import uuid
from typing import Any, Dict, List, Optional
from app.observability.logger import logger
from app.schemas import TokenUsage, TraceStep
from app.security.pii import PIIRedactor


class SystemTracer:
    """
    Rastreador distribuído por requisição (OpenTelemetry-compatible structure).
    Registra trace_id único, spans com latência individual e acumulada,
    consumo de tokens, estimativa de custo (USD) e mascaramento de PII.
    """

    # Preços de referência Gemini 2.5 Flash por 1M tokens (Input: $0.15 / Output: $0.60)
    INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
    OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000

    def __init__(self, trace_id: Optional[str] = None, session_id: str = "", user_id: str = ""):
        self.trace_id: str = trace_id or f"trc_{uuid.uuid4().hex[:16]}"
        self.session_id: str = session_id
        self.user_id: str = user_id
        self.steps: List[TraceStep] = []
        self._start_time: float = time.perf_counter()
        self._last_step_time: float = self._start_time
        self._step_counter: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / 3.8))

    def record_tokens(self, prompt_text: str = "", completion_text: str = "", prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        p_tok = prompt_tokens if prompt_tokens > 0 else self.estimate_tokens(prompt_text)
        c_tok = completion_tokens if completion_tokens > 0 else self.estimate_tokens(completion_text)
        self._prompt_tokens += p_tok
        self._completion_tokens += c_tok

    def get_token_usage(self) -> TokenUsage:
        total = self._prompt_tokens + self._completion_tokens
        cost = (self._prompt_tokens * self.INPUT_COST_PER_TOKEN) + (
            self._completion_tokens * self.OUTPUT_COST_PER_TOKEN
        )
        return TokenUsage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=total,
            estimated_cost_usd=round(cost, 6),
        )

    def add_step(
        self,
        agent: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ) -> TraceStep:
        now = time.perf_counter()
        self._step_counter += 1
        cumulative_ms = round((now - self._start_time) * 1000, 2)
        step_delta_ms = round((now - self._last_step_time) * 1000, 2)
        self._last_step_time = now

        masked_details = PIIRedactor.mask_structure(details or {})

        step = TraceStep(
            step_id=self._step_counter,
            agent=agent,
            action=action,
            status=status,
            details=masked_details,
            latency_ms=cumulative_ms,
            step_duration_ms=step_delta_ms,
        )
        self.steps.append(step)

        logger.info(
            "trace_span_completed",
            trace_id=self.trace_id,
            session_id=self.session_id,
            user_id=self.user_id,
            step_id=self._step_counter,
            agent=agent,
            action=action,
            status=status,
            step_duration_ms=step_delta_ms,
            cumulative_ms=cumulative_ms,
            details=masked_details,
        )
        return step

    def get_total_execution_time_ms(self) -> float:
        return round((time.perf_counter() - self._start_time) * 1000, 2)

    def get_steps(self) -> List[TraceStep]:
        return self.steps
