import pytest
from pydantic import ValidationError

from app.schemas import ChatResponse, SpecialistOutput, ToolExecutionResult


def test_tool_timeout_preserves_reconciliation_contract():
    result = ToolExecutionResult(tool_name="write", success=False,
                                 operation_id="op-test", outcome_state="unknown")
    assert result.model_dump()["operation_id"] == "op-test"
    assert result.outcome_state == "unknown"
    with pytest.raises(ValidationError):
        ToolExecutionResult(tool_name="write", success=False, outcome_state="guessed")


def test_evidence_survives_specialist_and_http_serialization():
    evidence = [{"kind": "tool_fact", "tool_name": "profile", "fact_path": "status", "value": "active"}]
    specialist = SpecialistOutput(agent_name="support", response="active", evidence=evidence)
    response = ChatResponse(response="active", user_id="owner", agent_used="support",
                            evidence=specialist.model_dump().get("evidence"),
                            action_proposals=[{"proposal_id": "proposal-test"}])
    assert response.model_dump()["evidence"] == evidence
    assert response.action_proposals == [{"proposal_id": "proposal-test"}]
