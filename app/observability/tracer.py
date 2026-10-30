import time
import uuid
from typing import Any, Dict, List, Optional
from app.observability.logger import logger
from app.schemas import TokenUsage, TraceStep
from app.security.pii import PIIRedactor


class SystemTracer:
    """
    Rastreador local de execução com geração de trace_id, spans locais estruturados em dicionários,
    telemetria de tokens, estimativa de custo e mascaramento de PII.
    Nota: opera exclusivamente em memória/logs locais; não implementa exportador OpenTelemetry
    distribuído nem integração com collector OTel externo.
    """

    # Coeficientes ilustrativos do provedor
    INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
    OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000

    def __init__(self, trace_id: Optional[str] = None, session_id: str = "", user_id: str = ""):
        self.trace_id: str = trace_id or f"trc_{uuid.uuid4().hex}"
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

    def to_local_span_dicts(self) -> List[Dict[str, Any]]:
        """Serializador local experimental que formata passos em dicionários tipo span (não exporta a collector externo)."""
        spans = []
        for step in self.steps:
            spans.append({
                "trace_id": self.trace_id,
                "span_id": f"span_{step.step_id:04x}_{uuid.uuid4().hex[:8]}",
                "parent_span_id": f"span_{step.step_id-1:04x}" if step.step_id > 1 else None,
                "name": f"{step.agent}:{step.action}",
                "kind": "SPAN_KIND_INTERNAL",
                "status": {"code": "STATUS_CODE_OK" if step.status == "success" else "STATUS_CODE_ERROR"},
                "duration_ms": step.step_duration_ms,
                "attributes": {
                    "session.id": self.session_id,
                    "user.id": self.user_id,
                    "agent.name": step.agent,
                    "action.name": step.action,
                    **(step.details or {}),
                },
            })
        return spans

    def to_opentelemetry_spans(self) -> List[Dict[str, Any]]:
        """Alias legado descontinuado; mantido para compatibilidade. Redireciona para to_local_span_dicts."""
        return self.to_local_span_dicts()

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
