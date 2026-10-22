import time
from typing import List, Dict, Any, Optional
from app.schemas import TraceStep

class SystemTracer:
    """
    Rastreador de observabilidade para o sistema multi-agente.
    Mede latências de cada etapa, decisões tomadas, ferramentas acionadas e histórico de execução.
    """
    def __init__(self):
        self.steps: List[TraceStep] = []
        self._start_time: float = time.perf_counter()
        self._step_counter: int = 0

    def add_step(self, agent: str, action: str, details: Optional[Dict[str, Any]] = None, status: str = "success") -> TraceStep:
        self._step_counter += 1
        elapsed = (time.perf_counter() - self._start_time) * 1000
        step = TraceStep(
            step_id=self._step_counter,
            agent=agent,
            action=action,
            status=status,
            details=details or {},
            latency_ms=round(elapsed, 2)
        )
        self.steps.append(step)
        return step

    def get_total_execution_time_ms(self) -> float:
        return round((time.perf_counter() - self._start_time) * 1000, 2)

    def get_steps(self) -> List[TraceStep]:
        return self.steps
