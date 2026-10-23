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


@pytest.mark.parametrize("message", [
    f"If firmware of {TARGET} is qa-12.08 or status is Online, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online and certified, tell me benefits.",
    f"If firmware of {TARGET} is at least qa-12.08 and status is Online, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online or Offline, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, and certified, tell me benefits.",
    f"If firmware of {TARGET} is qa-12.08 and status is Online, provided that it is certified, tell me benefits.",
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
