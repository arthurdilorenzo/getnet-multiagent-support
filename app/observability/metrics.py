from collections import Counter, deque
import threading
from typing import Any, Deque, Dict, List
from app.observability.resilience import breaker_registry
from app.schemas import TokenUsage


class MetricsCollector:
    """
    Coletor central de telemetria operacional, custos de tokens e percentis de latência (p50/p95/p99).
    """

    def __init__(self, max_samples: int = 1000) -> None:
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.blocked_guardrail_requests: int = 0
        self.escalated_requests: int = 0
        self.abstained_requests: int = 0
        self.agent_counts: Counter = Counter()
        self.tool_counts: Counter = Counter()
        self.tool_errors: Counter = Counter()
        self._completion_groups: Dict[str, Dict[str, Counter]] = {}
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._latencies_ms: Deque[float] = deque(maxlen=max_samples)

    def record_request(
        self,
        agent_used: str,
        tools_called: List[str],
        latency_ms: float,
        token_usage: TokenUsage,
        blocked: bool = False,
        escalated: bool = False,
        abstained: bool = False,
        failed: bool = False,
        semantic_mode: str = '',
        answer_status: str = '',
    ) -> None:
        with self._lock:
            self.total_requests += 1
            mode = semantic_mode if semantic_mode in {'native_semantic', 'limited_offline'} else 'unclassified'
            status = answer_status if answer_status in {'complete', 'partial', 'abstained'} else 'unclassified'
            group = self._completion_groups.setdefault(mode, {}).setdefault(
                status, Counter(total=0, successful=0, failed=0, abstained=0))
            group['total'] += 1
            group['failed' if failed else 'successful'] += 1
            group['abstained'] += int(abstained)
            if failed:
                self.failed_requests += 1
            else:
                self.successful_requests += 1
            if blocked:
                self.blocked_guardrail_requests += 1
            if escalated:
                self.escalated_requests += 1
            if abstained:
                self.abstained_requests += 1

            self.agent_counts[agent_used] += 1
            for tool in tools_called:
                self.tool_counts[tool] += 1

            self.total_prompt_tokens += token_usage.prompt_tokens
            self.total_completion_tokens += token_usage.completion_tokens
            self.total_cost_usd = round(self.total_cost_usd + token_usage.estimated_cost_usd, 6)
            self._latencies_ms.append(float(latency_ms))

    def record_tool_error(self, tool_name: str) -> None:
        with self._lock:
            self.tool_errors[tool_name] += 1

    @staticmethod
    def _percentile(sorted_vals: List[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return round(sorted_vals[idx], 2)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            sorted_lat = sorted(self._latencies_ms)
            avg_lat = round(sum(sorted_lat) / len(sorted_lat), 2) if sorted_lat else 0.0
            return {
                "requests": {
                    "total": self.total_requests,
                    "successful": self.successful_requests,
                    "failed": self.failed_requests,
                    "blocked_by_guardrails": self.blocked_guardrail_requests,
                    "escalated_to_hitl": self.escalated_requests,
                    "abstained": self.abstained_requests,
                },
                "requests_by_mode_and_status": {
                    mode: {status: dict(counts) for status, counts in statuses.items()}
                    for mode, statuses in self._completion_groups.items()
                },
                "completion_metadata_limitations": (
                    "Modo e status são metadados declarados pelo pipeline; successful conta "
                    "execuções sem falha, não comprova resolução ou correção semântica."
                ),
                "latency_ms": {
                    "avg_ms": avg_lat,
                    "p50_ms": self._percentile(sorted_lat, 50),
                    "p95_ms": self._percentile(sorted_lat, 95),
                    "p99_ms": self._percentile(sorted_lat, 99),
                    "samples_count": len(sorted_lat),
                },
                "token_and_cost_telemetry": {
                    "prompt_tokens": self.total_prompt_tokens,
                    "completion_tokens": self.total_completion_tokens,
                    "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                    "estimated_total_cost_usd": round(self.total_cost_usd, 6),
                },
                "agent_distribution": dict(self.agent_counts),
                "tool_invocations": dict(self.tool_counts),
                "tool_errors": dict(self.tool_errors),
                "circuit_breakers": breaker_registry.states(),
            }


metrics_collector = MetricsCollector()
