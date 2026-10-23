"""A dependent lookup needs a typed observation of the requested terminal."""

import pytest

from app.schemas import ToolExecutionResult
from app.security.conditional_dependencies import (
    evaluate_terminal_condition,
    parse_terminal_condition,
)


TARGET = "GET-CLASSICA-QA-9627"
OTHER = "GET-SMART-QA-9513"
REQUEST = (
    f"If firmware of {TARGET} is qa-12.08 and status is Online, "
    "tell me the eligible benefits."
)


def diagnosis(terminal_id=TARGET, *, firmware="qa-12.08", status="Online", success=True,
              found=True, tool_name="diagnose_terminal_issue"):
    return ToolExecutionResult(tool_name=tool_name, success=success, data={
        "found": found,
        "requested_terminal_id": terminal_id,
        "user_id": "synthetic",
        "terminal": {"terminal_id": terminal_id, "firmware_version": firmware, "status": status},
    })


def test_exact_firmware_and_online_authorize_only_matching_terminal():
    condition = parse_terminal_condition(REQUEST, TARGET)
    assert condition is not None
    assert evaluate_terminal_condition(condition, [diagnosis()]) is True


@pytest.mark.parametrize("firmware,status", [
    ("qa-12.07", "Online"), ("qa-12.08", "Offline"),
    (None, "Online"), ("qa-12.08", None),
    ("", "Online"),
])
def test_false_or_null_predicate_does_not_authorize(firmware, status):
    condition = parse_terminal_condition(REQUEST, TARGET)
    assert not evaluate_terminal_condition(condition, [diagnosis(firmware=firmware, status=status)])


def test_missing_field_does_not_authorize():
    condition = parse_terminal_condition(REQUEST, TARGET)
    result = diagnosis()
    del result.data["terminal"]["firmware_version"]
    assert not evaluate_terminal_condition(condition, [result])


def test_wrong_target_cannot_donate_matching_fields():
    condition = parse_terminal_condition(REQUEST, TARGET)
    assert not evaluate_terminal_condition(condition, [diagnosis(OTHER)])
    assert not evaluate_terminal_condition(condition, [diagnosis(OTHER), diagnosis(firmware=None)])


def test_failed_missing_or_untyped_observation_cannot_authorize():
    condition = parse_terminal_condition(REQUEST, TARGET)
    assert not evaluate_terminal_condition(condition, [])
    assert not evaluate_terminal_condition(condition, [diagnosis(success=False)])
    assert not evaluate_terminal_condition(condition, [diagnosis(found=False)])
    assert not evaluate_terminal_condition(condition, [diagnosis(tool_name="get_customer_profile")])
    assert not evaluate_terminal_condition(condition, [diagnosis().model_dump()])


def test_failed_exact_target_without_terminal_conflicts_with_success():
    condition = parse_terminal_condition(REQUEST, TARGET)
    failed = ToolExecutionResult(
        tool_name="diagnose_terminal_issue", success=False, error_code="TERMINAL_NOT_FOUND",
        data={"found": False, "requested_terminal_id": TARGET},
    )
    assert not evaluate_terminal_condition(condition, [diagnosis(), failed])
    assert not evaluate_terminal_condition(condition, [failed, diagnosis()])


@pytest.mark.parametrize("data", [None, {}, {"requested_terminal_id": TARGET}])
@pytest.mark.parametrize("failure", [
    {"success": False},
    {"success": True, "outcome_state": "failed"},
    {"success": True, "outcome_state": "unknown"},
    {"success": True, "outcome_state": "pending"},
    {"success": True, "error_code": "DIAGNOSTIC_READ_FAILED"},
])
def test_failed_diagnostic_without_terminal_invalidates_positive_read(data, failure):
    condition = parse_terminal_condition(REQUEST, TARGET)
    failed = ToolExecutionResult(tool_name="diagnose_terminal_issue", data=data, **failure)
    assert evaluate_terminal_condition(condition, [diagnosis(), failed]) is False
    assert evaluate_terminal_condition(condition, [failed, diagnosis()]) is False


@pytest.mark.parametrize("conflict", [
    {"firmware": "qa-12.07"},
    {"status": "Offline"},
    {"firmware": None},
    {"found": False},
])
def test_conflicting_exact_target_reads_never_authorize(conflict):
    condition = parse_terminal_condition(REQUEST, TARGET)
    conflicting = diagnosis(**conflict)
    assert evaluate_terminal_condition(condition, [diagnosis(), conflicting]) is False
    assert evaluate_terminal_condition(condition, [conflicting, diagnosis()]) is False


@pytest.mark.parametrize("message", [
    f"If firmware of {TARGET} is qa-12.08 or status is Online, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online and certified, tell me benefits.",
    f"If firmware of {TARGET} is at least qa-12.08 and status is Online, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online or Offline, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, and certified, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, provided that it is certified, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits only if certified.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits if certified.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits unless certified.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits when certified.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits whenever certified.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits assuming certification.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, informe os benefícios caso certificado.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits in case of certification.",
    f"For certified devices, if firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits.",
    f"Provided certified and if firmware of {TARGET} is qa-12.08 and status is Online, tell me benefits.",
    f"Unless certified, explain the benefit quando o {TARGET} tiver firmware qa-12.08 com conexão online.",
    f"If firmware of {OTHER} is qa-12.08 and status is Online, tell me benefits.",
    "If firmware is current and terminal is online, tell me benefits.",
])
def test_unsupported_or_ambiguous_condition_has_no_authorization(message):
    assert parse_terminal_condition(message, TARGET) is None


def test_portuguese_exact_conjunction_is_supported():
    request = f"Se o firmware do terminal {TARGET} for qa-12.08 e o status for Online, informe os benefícios."
    condition = parse_terminal_condition(request, TARGET)
    assert condition is not None
    assert evaluate_terminal_condition(condition, [diagnosis()])


@pytest.mark.parametrize("message", [
    f"O recebimento do Crediário só deve ser explicado quando o {TARGET} tiver firmware qa-12.08 com conexão online.",
    f"O prazo do Crediário só deve ser explicado quando o firmware do meu {TARGET} for exatamente qa-12.08, com conexão ativa.",
])
def test_portuguese_explicit_prerequisite_forms_from_native_cases(message):
    condition = parse_terminal_condition(message, TARGET)
    assert condition is not None
    assert evaluate_terminal_condition(condition, [diagnosis()])


# An earlier exact positive must never mask a later incomplete business read.
INVALID_DIAGNOSTIC_READS = [
    pytest.param(None, True, id="null-payload"),
    pytest.param(None, False, id="failed-null-payload"),
    pytest.param({}, True, id="empty-payload"),
    pytest.param(diagnosis(OTHER).data, True, id="different-terminal-payload"),
    pytest.param({"found": True}, True, id="missing-terminal-and-requested-id"),
    pytest.param({"found": True, "requested_terminal_id": TARGET, "terminal": None}, True, id="null-terminal"),
    pytest.param({"found": True, "requested_terminal_id": TARGET, "terminal": []}, True, id="untyped-terminal"),
    pytest.param({"found": True, "requested_terminal_id": TARGET, "terminal": {}}, True, id="empty-terminal"),
]
for field, value, case in [
    ("found", False, "not-found"), ("found", 1, "numeric-found"),
    ("found", "true", "string-found"), ("found", None, "missing-found"),
    ("requested_terminal_id", None, "missing-requested-id"),
    ("requested_terminal_id", OTHER, "other-requested-id"),
    ("requested_terminal_id", [TARGET], "untyped-requested-id"),
    ("terminal_id", None, "missing-terminal-id"),
    ("terminal_id", OTHER, "other-terminal-id"),
    ("terminal_id", [TARGET], "untyped-terminal-id"),
    ("firmware_version", None, "missing-firmware"),
    ("firmware_version", [], "untyped-firmware"),
    ("firmware_version", "", "empty-firmware"),
    ("status", None, "missing-status"), ("status", True, "untyped-status"),
    ("status", "", "empty-status"),
]:
    payload = diagnosis().data
    section = payload if field in {"found", "requested_terminal_id"} else payload["terminal"]
    if value is None:
        del section[field]
    else:
        section[field] = value
    INVALID_DIAGNOSTIC_READS.append(pytest.param(payload, True, id=case))


@pytest.mark.parametrize("data,success", INVALID_DIAGNOSTIC_READS)
def test_every_diagnostic_needs_complete_typed_target_payload(data, success):
    condition = parse_terminal_condition(REQUEST, TARGET)
    incomplete = ToolExecutionResult(tool_name="diagnose_terminal_issue", success=success, data=data)
    assert evaluate_terminal_condition(condition, [incomplete]) is False
    assert evaluate_terminal_condition(condition, [diagnosis(), incomplete]) is False
    assert evaluate_terminal_condition(condition, [incomplete, diagnosis()]) is False


def test_multiple_complete_exact_target_diagnostics_authorize():
    condition = parse_terminal_condition(REQUEST, TARGET)
    assert evaluate_terminal_condition(condition, [diagnosis(), diagnosis()]) is True


@pytest.mark.parametrize("data,success", INVALID_DIAGNOSTIC_READS)
@pytest.mark.parametrize("invalid_first", [False, True])
def test_incomplete_diagnostic_blocks_orchestrator_dependent_knowledge_and_web(
        monkeypatch, tmp_path, data, success, invalid_first):
    import importlib

    from app.agents.orchestrator import MultiAgentOrchestrator
    from app.config import settings
    from app.memory.store import StateStore
    from app.schemas import ChatRequest, PlanStep, RoutingPlan, SpecialistOutput
    from app.security.auth import SecurityContext

    module = importlib.import_module("app.agents.orchestrator")
    monkeypatch.setattr(module, "state_store", StateStore(tmp_path / "read-integrity.sqlite"))
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "AUTH_ENFORCE_STRICT", False)
    orch = MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router, "plan", lambda **kwargs: RoutingPlan(
        primary_agent="KNOWLEDGE_AGENT", confidence=.9, reasoning="synthetic dependent proposal",
        plan_steps=[PlanStep(step_order=1, target_agent="KNOWLEDGE_AGENT", sub_query=REQUEST,
                            rationale="conditional benefits", request_unit_ids=["u1"])],
    ))
    incomplete = ToolExecutionResult(tool_name="diagnose_terminal_issue", success=success, data=data)
    results = [incomplete, diagnosis()] if invalid_first else [diagnosis(), incomplete]
    calls = []

    def delegate(**kwargs):
        calls.append(kwargs["target_agent"])
        assert kwargs["target_agent"] == "CUSTOMER_SUPPORT_AGENT"
        assert TARGET in kwargs["sub_query"]
        return SpecialistOutput(agent_name="CUSTOMER_SUPPORT_AGENT", response="diagnostic reads",
            tools_called=["diagnose_terminal_issue"], tool_results=results)

    # Substitute only the specialist boundary that supplies corrupted telemetry.
    monkeypatch.setattr(orch, "_delegate_to_specialist", delegate)
    out = orch.process(ChatRequest(message=REQUEST, user_id="synthetic"), SecurityContext("synthetic"))
    assert calls == ["CUSTOMER_SUPPORT_AGENT"]
    assert not {"search_getnet_knowledge", "web_search_general"} & set(out.tools_called)
    gate = next(step for step in out.trace if step.action == "evaluate_conditional_dependency")
    assert gate.details["predicate_authorized"] is False
    assert out.answer_status == "abstained"
    assert all(row["state"] == "unresolved" for row in out.request_coverage)
