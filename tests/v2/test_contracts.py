"""Provider-free unit tests for Getnet Núcleo 2.0 consolidation contracts."""
import hashlib
import pytest
from pydantic import ValidationError

from app.schemas import ToolExecutionResult
from app.v2.contracts import (
    AnswerDocument,
    Binding,
    BindingProposal,
    ContextBlock,
    DocumentaryBlock,
    Entities,
    EvidenceSpan,
    Fact,
    HumanBlock,
    OperationalBlock,
    PendingBlock,
    PlanProposal,
    Requirement,
    RequirementDecision,
    RequirementProposal,
    ResumePlan,
    StepProposal,
    answer_status,
    canonicalize_plan,
)
from app.v2.knowledge import EvidenceError, format_fact, human_span, render_final_answer, tool_span, validate_answer


def test_requirements_precede_final_answer_and_empty_are_forbidden():
    fields = list(AnswerDocument.model_fields)
    assert fields.index("requirements") < fields.index("final_answer")

    with pytest.raises(ValidationError):
        AnswerDocument(requirements=[], final_answer=[PendingBlock(requirement_id="r0001")])

    with pytest.raises(ValidationError):
        AnswerDocument(
            requirements=[RequirementDecision(requirement_id="r0001", status="unanswerable")],
            final_answer=[],
        )


def test_canonicalize_plan_assigns_stable_server_ids():
    proposal = PlanProposal(
        requirements=[
            RequirementProposal(
                kind="documentary",
                summary="Regras de Crediário",
                constraints=["sem juros"],
            ),
            RequirementProposal(
                kind="operational",
                summary="Status da máquina",
                bindings=[
                    BindingProposal(
                        tool_name="diagnose_terminal_issue",
                        resource_type="terminal",
                        resource_id="POS-9941",
                        field_pointer="/terminal/status",
                    )
                ],
            ),
        ],
        steps=[
            StepProposal(kind="knowledge", query="crediario", requirement_indexes=[0]),
            StepProposal(kind="tool", tool_name="diagnose_terminal_issue", arguments={"terminal_id": "POS-9941"}, requirement_indexes=[1]),
        ],
    )

    reqs, steps = canonicalize_plan(proposal)
    assert len(reqs) == 2
    assert reqs[0].requirement_id == "r0001"
    assert reqs[0].kind == "documentary"
    assert reqs[1].requirement_id == "r0002"
    assert reqs[1].kind == "operational"
    assert len(reqs[1].bindings) == 1
    assert reqs[1].bindings[0].binding_id == "r0002:b0001"

    # Step requirement indices validation
    assert steps[0].requirement_indexes == [0]
    assert steps[1].requirement_indexes == [1]


def test_canonicalize_plan_rejects_invalid_step_indices():
    proposal = PlanProposal(
        requirements=[RequirementProposal(kind="documentary", summary="Prazo")],
        steps=[StepProposal(kind="knowledge", requirement_indexes=[5])],  # out of bounds
    )
    with pytest.raises(ValueError, match="índice de requisito inválido"):
        canonicalize_plan(proposal)

    proposal_dup = PlanProposal(
        requirements=[RequirementProposal(kind="documentary", summary="Prazo")],
        steps=[StepProposal(kind="knowledge", requirement_indexes=[0, 0])],  # duplicated
    )
    with pytest.raises(ValueError, match="índices de requisitos duplicados"):
        canonicalize_plan(proposal_dup)


def test_answer_status_derivation():
    dec_supported = RequirementDecision(requirement_id="r0001", status="supported")
    dec_contradicted = RequirementDecision(requirement_id="r0002", status="contradicted")
    dec_unanswerable = RequirementDecision(requirement_id="r0003", status="unanswerable")

    # All supported -> complete
    assert answer_status([dec_supported]) == "complete"
    assert answer_status([dec_supported, dec_contradicted]) == "complete"

    # All unanswerable -> abstained
    assert answer_status([dec_unanswerable]) == "abstained"

    # Mixed -> partial
    assert answer_status([dec_supported, dec_unanswerable]) == "partial"


def test_tool_span_and_human_span_creation():
    # Failed tool call produces no span
    failed_tool = ToolExecutionResult(tool_name="test", success=False, outcome_state="failed")
    assert tool_span(failed_tool) is None

    # tool call with found=False produces no span
    not_found = ToolExecutionResult(tool_name="test", success=True, outcome_state="completed", data={"found": False})
    assert tool_span(not_found) is None

    # Successful completed tool call produces EvidenceSpan
    success = ToolExecutionResult(tool_name="get_customer_profile", success=True, outcome_state="completed", data={"found": True, "name": "Loja X"})
    span = tool_span(success)
    assert span is not None
    assert span.kind == "tool"
    assert "Loja X" in span.text

    # Human span
    h_span = human_span("TICK-1", "operador_joao", "Conferência manual realizada.")
    assert h_span.kind == "human"
    assert h_span.source == "hitl:TICK-1:operador_joao"


def test_operational_block_renders_server_formatted_fact():
    fact_saldo = Fact(
        fact_id="f1",
        operation_id="OP-1",
        resource_type="customer",
        resource_id="user_1",
        field_pointer="/sales_data/saldo_disponivel_brl",
        availability="available",
        value=1415.80,
        unit="BRL",
    )
    formatted = format_fact(fact_saldo)
    assert "BRL 1.415,80" in formatted

    doc = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[OperationalBlock(requirement_id="r0001", fact_ids=["f1"])],
    )
    rendered = render_final_answer(doc, [], [fact_saldo], {})
    assert "saldo_disponivel_brl (user_1): BRL 1.415,80" in rendered
