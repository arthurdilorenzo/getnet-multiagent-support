import time
import uuid
from typing import Any, Dict, List, Optional
from app.observability.logger import logger
from app.observability.trace_store import TraceRepository
from app.schemas import TokenUsage, TraceStep
from app.security.pii import PIIRedactor


class SystemTracer:
    """
    Rastreador de execução com geração de trace_id, spans estáveis persistidos em SQLite,
    serialização compatível com OpenTelemetry (OTLP/JSON), telemetria de tokens,
    estimativa de custo e mascaramento de PII.
    """

    # Coeficientes ilustrativos do provedor
    INPUT_COST_PER_TOKEN = 0.15 / 1_000_000
    OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000
    TOTAL_PERSISTENCE_FAILURES: int = 0

    def __init__(
        self,
        trace_id: Optional[str] = None,
        session_id: str = "",
        user_id: str = "",
        repository: Optional[TraceRepository] = None,
        parent_trace_id: Optional[str] = None,
    ):
        self.trace_id: str = trace_id or f"trc_{uuid.uuid4().hex}"
        self.session_id: str = session_id
        self.user_id: str = user_id
        self.parent_trace_id: Optional[str] = parent_trace_id
        self._repository: Optional[TraceRepository] = repository
        self.steps: List[TraceStep] = []
        self._span_ids: Dict[int, str] = {}
        self.persistence_failures: int = 0
        self._start_time: float = time.perf_counter()
        self._last_step_time: float = self._start_time
        self._step_counter: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        try:
            self.repo.start_trace(
                trace_id=self.trace_id,
                session_id=self.session_id,
                user_id=self.user_id,
                parent_trace_id=self.parent_trace_id,
            )
        except Exception as exc:
            self._record_persistence_failure("start_trace", exc)

    def _record_persistence_failure(self, phase: str, exc: Exception) -> None:
        self.persistence_failures += 1
        type(self).TOTAL_PERSISTENCE_FAILURES += 1
        logger.warning(
            "trace_persistence_degraded",
            trace_id=self.trace_id,
            phase=phase,
            error_type=type(exc).__name__,
            failures=self.persistence_failures,
        )

    @property
    def repo(self) -> TraceRepository:
        return self._repository or TraceRepository.default()

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
        """Serializador de spans com IDs estáveis por etapa e atributos mascarados."""
        spans = []
        for step in self.steps:
            span_id = self._span_ids.get(step.step_id) or f"span_{self.trace_id[-8:]}_{step.step_id:04x}"
            self._span_ids[step.step_id] = span_id
            parent_span_id = self._span_ids.get(step.step_id - 1) if step.step_id > 1 else None
            spans.append({
                "trace_id": self.trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
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
        """Retorna spans com IDs estáveis em formato estruturado."""
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
        span_id = f"span_{self.trace_id[-8:]}_{self._step_counter:04x}_{uuid.uuid4().hex[:6]}"
        self._span_ids[self._step_counter] = span_id
        parent_span_id = self._span_ids.get(self._step_counter - 1) if self._step_counter > 1 else None

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

        try:
            self.repo.record_span(
                trace_id=self.trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                step_id=self._step_counter,
                agent=agent,
                action=action,
                status=status,
                duration_ms=step_delta_ms,
                cumulative_ms=cumulative_ms,
                attributes={
                    "session.id": self.session_id,
                    "user.id": self.user_id,
                    "agent.name": agent,
                    "action.name": action,
                    **masked_details,
                },
                session_id=self.session_id,
                user_id=self.user_id,
            )
        except Exception as exc:
            self._record_persistence_failure("record_span", exc)

        logger.info(
            "trace_span_completed",
            trace_id=self.trace_id,
            span_id=span_id,
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

    def finish(
        self,
        status: str = "completed",
        answer_status: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        usage = self.get_token_usage()
        try:
            self.repo.finish_trace(
                self.trace_id,
                status=status,
                answer_status=answer_status,
                total_duration_ms=self.get_total_execution_time_ms(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=usage.estimated_cost_usd,
                metadata=metadata,
            )
        except Exception as exc:
            self._record_persistence_failure("finish_trace", exc)

    def get_total_execution_time_ms(self) -> float:
        return round((time.perf_counter() - self._start_time) * 1000, 2)

    def get_steps(self) -> List[TraceStep]:
        return self.steps
