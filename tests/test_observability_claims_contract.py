import pytest
from pathlib import Path
from app.observability.metrics import MetricsCollector
from app.observability.tracer import SystemTracer
from app.schemas import TokenUsage


def test_cost_metric_when_zero_successful_requests():
    collector = MetricsCollector()
    snap = collector.snapshot()
    # Quando há 0 requests sem falha, custo médio por request sem falha DEVE ser None/indisponível
    cost_info = snap["token_and_cost_telemetry"]
    assert cost_info["estimated_cost_per_nonfailed_request_usd"] is None
    # Metadados de depreciação do alias antigo
    assert "deprecations" in cost_info
    assert "estimated_cost_per_successful_resolution_usd" in cost_info["deprecations"]


def test_cost_metric_with_abstention_counts_as_nonfailed_not_confirmed_resolution():
    collector = MetricsCollector()
    collector.record_request(
        agent_used="KNOWLEDGE_AGENT",
        tools_called=["search_getnet_knowledge"],
        latency_ms=10.0,
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost_usd=0.00015),
        abstained=True,
        failed=False,
    )
    snap = collector.snapshot()
    cost_info = snap["token_and_cost_telemetry"]
    assert cost_info["estimated_cost_per_nonfailed_request_usd"] == 0.00015
    # Confirma que o snapshot avisa que conta execuções sem falha, não resolução confirmada
    assert "não comprova resolução" in snap["completion_metadata_limitations"].lower()


def test_tracer_does_not_claim_exported_otel_pipeline():
    tracer = SystemTracer(session_id="sess_trace_test", user_id="cliente1988")
    tracer.add_step(agent="ROUTER", action="plan", status="success")
    # O método local é um serializador de dicionários local, não exportador OTel de rede
    spans = tracer.to_local_span_dicts()
    assert len(spans) == 1
    assert spans[0]["name"] == "ROUTER:plan"
    # Alias legado redireciona para o mesmo resultado estrutural
    otel_spans = tracer.to_opentelemetry_spans()
    assert len(otel_spans) == 1
    assert otel_spans[0]["name"] == spans[0]["name"]
    assert otel_spans[0]["trace_id"] == spans[0]["trace_id"]

    # Docstring não promete exportador OTel interoperável
    assert "exportador opentelemetry distribuído" not in (SystemTracer.__doc__ or "").lower() or "não implementa exportador" in (SystemTracer.__doc__ or "").lower()


def test_readme_does_not_claim_active_hitl_resumption_or_otel_persistence():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "retomada ativa pelo agente" not in readme
    assert "OpenTelemetry Trace Persist" not in readme
    assert "observabilidade completa" not in readme
