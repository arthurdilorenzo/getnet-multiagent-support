"""Tests for V2.1 evolution: Semantic Evaluation V2, Adaptive Replanning, Domain Presentation, Revocation Store, and Persistent Traces."""
import json
from pathlib import Path
from fastapi.testclient import TestClient

import app.main as main_module
from app.evals.v2.datasets import (
    default_calibration_records,
    default_canary_pairs,
    default_eval_cases,
)
from app.evals.v2.judge import IndependentSemanticJudge
from app.evals.v2.runner import SemanticEvalCampaignRunner
from app.hitl.queue import HITLQueueManager
from app.memory.store import StateStore
from app.observability.trace_store import TraceRepository
from app.security.auth import AuthManager, SecurityContext
from app.v2.adapters import StateAdapter, ToolGateway
from app.v2.contracts import (
    AnswerDocument,
    BindingProposal,
    DocumentaryBlock,
    ExecutionBudget,
    Fact,
    OperationalBlock,
    PlanProposal,
    ReplanProposal,
    RequirementDecision,
    RequirementProposal,
    StepProposal,
)
from app.v2.knowledge import KnowledgeService, format_fact
from app.v2.runtime import Runtime
from scripts.eval_v2 import build_campaign_runtime


def test_domain_presenter_renders_natural_portuguese_with_canonical_audit_token():
    fact = Fact(
        fact_id="f1",
        binding_id="r0001:b0001",
        operation_id="op1",
        resource_type="terminal",
        resource_id="TERM_001",
        field_pointer="/terminal/status",
        availability="available",
        value="OFFLINE",
        value_type="str",
    )
    rendered = format_fact(fact)
    assert "A consulta técnica indica que o terminal TERM_001 encontra-se offline" in rendered
    assert "[status (TERM_001): OFFLINE]" in rendered


def test_adaptive_replan_enriches_documentary_lookup_after_diagnostic_observation(tmp_path: Path):
    runtime = build_campaign_runtime(tmp_path)
    ctx = SecurityContext(
        authenticated_user_id="cliente1988",
        role="merchant",
        scopes=["chat:write", "customer:read_own", "terminal:action_own"],
    )
    response = runtime.process(
        "Meu terminal está com falha; consulte o status técnico e oriente o procedimento.",
        ctx,
        session_id="sess_adaptive_diag",
    )
    assert response.answer_status == "complete"
    assert response.tools_called == ["diagnose_terminal_issue", "search_getnet_knowledge"]
    actions = [step.action for step in response.trace]
    assert "observe" in actions
    assert "adaptive_replan" in actions


def test_adaptive_replan_model_proposal_and_loop_detection(tmp_path: Path):
    class ReplanningModel:
        supports_replan = True

        def __init__(self):
            self.replan_calls = 0

        def generate(self, schema, payload, *, phase: str, tracer):
            if schema is PlanProposal:
                return PlanProposal(
                    requirements=[
                        RequirementProposal(
                            kind="operational",
                            summary="Diagnosticar terminal",
                            bindings=[
                                BindingProposal(
                                    tool_name="diagnose_terminal_issue",
                                    resource_type="terminal",
                                    resource_id="TERM_001",
                                    field_pointer="/terminal/status",
                                )
                            ],
                        ),
                        RequirementProposal(kind="documentary", summary="Procedimento de conectividade"),
                    ],
                    steps=[
                        StepProposal(
                            kind="tool",
                            tool_name="diagnose_terminal_issue",
                            arguments={"terminal_id": "TERM_001"},
                            requirement_indexes=[0],
                        ),
                        # Duplicate tool step to verify loop detection blocks it
                        StepProposal(
                            kind="tool",
                            tool_name="diagnose_terminal_issue",
                            arguments={"terminal_id": "TERM_001"},
                            requirement_indexes=[0],
                        ),
                    ],
                )
            if schema is ReplanProposal:
                self.replan_calls += 1
                if self.replan_calls == 1:
                    return ReplanProposal(
                        action="append_steps",
                        reason_code="lookup_observed_error_code",
                        steps=[
                            StepProposal(
                                kind="knowledge",
                                query="erro E_SIM_NO_SIGNAL Get Smart",
                                requirement_indexes=[1],
                            )
                        ],
                    )
                return ReplanProposal(action="continue")
            if schema is AnswerDocument:
                facts = payload["observed_facts"]
                spans = [s for s in payload["evidence_spans"] if s["kind"] == "document"]
                return AnswerDocument(
                    requirements=[
                        RequirementDecision(requirement_id="r0001", status="supported"),
                        RequirementDecision(requirement_id="r0002", status="supported"),
                    ],
                    final_answer=[
                        OperationalBlock(requirement_id="r0001", fact_ids=[facts[0]["fact_id"]]),
                        DocumentaryBlock(
                            requirement_id="r0002",
                            text="Verifique o chip e reconecte o Wi-Fi.",
                            span_ids=[spans[0]["span_id"]],
                        ),
                    ],
                )
            raise ValueError(schema)

    base_rt = build_campaign_runtime(tmp_path)
    model = ReplanningModel()
    knowledge = KnowledgeService(model, retrieve=base_rt.knowledge.retrieve)
    runtime = Runtime(
        model,
        knowledge,
        base_rt.state,
        base_rt.gateway,
        base_rt.queue,
        budget=ExecutionBudget(max_steps=6, max_replans=2),
    )
    ctx = SecurityContext(
        authenticated_user_id="cliente1988",
        role="merchant",
        scopes=["chat:write", "customer:read_own"],
    )
    resp = runtime.process("Verifique meu terminal TERM_001", ctx, session_id="sess_loop_guard")
    assert resp.answer_status == "complete"
    trace_actions = [s.action for s in resp.trace]
    assert "loop_detected" in trace_actions
    assert "adaptive_replan" in trace_actions
    assert resp.tools_called.count("diagnose_terminal_issue") == 1


def test_persisted_traces_and_otlp_payload_survive_across_repository_instances(tmp_path: Path, monkeypatch):
    store = StateStore(tmp_path / "traces_test.db")
    monkeypatch.setattr(main_module, "state_store", store)
    TraceRepository._STORE_BY_PATH.clear()

    runtime = build_campaign_runtime(tmp_path)
    # Point default trace repository to the same DB
    repo = TraceRepository(db_path=store.db_path)
    monkeypatch.setattr(TraceRepository, "default", classmethod(lambda cls: repo))

    ctx = SecurityContext(
        authenticated_user_id="cliente1988",
        role="merchant",
        scopes=["chat:write", "customer:read_own"],
    )
    resp = runtime.process("Qual é a diferença entre a Get Clássica e a Get Smart?", ctx, session_id="sess_trace_1")

    fresh_repo = TraceRepository(db_path=store.db_path)
    persisted = fresh_repo.get_trace(resp.trace_id)
    assert persisted is not None
    assert persisted["trace_id"] == resp.trace_id
    assert persisted["status"] == "completed"
    assert persisted["answer_status"] == "complete"
    assert len(persisted["spans"]) >= 2
    assert "resourceSpans" in persisted["otlp_resource_spans"]
    otlp_spans = persisted["otlp_resource_spans"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(otlp_spans) >= 2
    for sp in otlp_spans:
        assert len(sp["traceId"]) == 32
        int(sp["traceId"], 16)
        assert len(sp["spanId"]) == 16
        int(sp["spanId"], 16)
        assert int(sp["startTimeUnixNano"]) > 0
        assert int(sp["endTimeUnixNano"]) >= int(sp["startTimeUnixNano"])
        assert sp["status"]["code"] in (1, 2)
        assert "durationMs" not in sp

    client = TestClient(main_module.app)
    op_token = AuthManager.create_jwt_token("operator-audit", role="operator")
    http_res = client.get(f"/api/traces/{resp.trace_id}", headers={"Authorization": f"Bearer {op_token}"})
    assert http_res.status_code == 200
    assert http_res.json()["trace_id"] == resp.trace_id

    merchant_token = AuthManager.create_jwt_token("cliente1988", role="merchant")
    denied = client.get(f"/api/traces/{resp.trace_id}", headers={"Authorization": f"Bearer {merchant_token}"})
    assert denied.status_code == 403

    # Verify tracer tracks persistence_failures when repository fails
    from app.observability.tracer import SystemTracer
    broken_tracer = SystemTracer(session_id="sess_broken", user_id="cliente1988")
    monkeypatch.setattr(broken_tracer.repo, "record_span", lambda **kw: (_ for _ in ()).throw(RuntimeError("sqlite locked")))
    broken_tracer.add_step(agent="V2_SUPERVISOR", action="test_fail")
    assert broken_tracer.persistence_failures >= 1
    assert SystemTracer.TOTAL_PERSISTENCE_FAILURES >= 1


def test_runtime_confidence_gate_and_budget_enforcement(tmp_path: Path):
    class LowConfidenceAndCostlyModel:
        supports_replan = True

        def __init__(self):
            self.replan_calls = 0

        def generate(self, schema, payload, *, phase: str, tracer):
            tracer.record_tokens(prompt_text="x" * 200, completion_text="y" * 200)
            if schema is PlanProposal:
                return PlanProposal(
                    requirements=[RequirementProposal(kind="documentary", summary="Buscar FAQ")],
                    steps=[
                        StepProposal(kind="knowledge", query="consulta 1", requirement_indexes=[0]),
                        StepProposal(kind="knowledge", query="consulta 2", requirement_indexes=[0]),
                    ],
                )
            if schema is ReplanProposal:
                self.replan_calls += 1
                # Even returning 'continue' must consume max_replans budget
                return ReplanProposal(action="continue")
            if schema is AnswerDocument:
                spans = payload["evidence_spans"]
                return AnswerDocument(
                    requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
                    final_answer=[DocumentaryBlock(requirement_id="r0001", text="A Get Clássica possui teclado físico.", span_ids=[spans[0]["span_id"]])],
                )
            raise ValueError(schema)

    base_rt = build_campaign_runtime(tmp_path)
    model = LowConfidenceAndCostlyModel()
    knowledge = KnowledgeService(model, retrieve=base_rt.knowledge.retrieve)
    # 1 span retrieved => confidence = 0.65 < threshold 0.8 => triggers _maybe_replan
    # 2 knowledge steps returning same span => triggers duplicate_retrieval_evidence loop_detected
    runtime = Runtime(
        model,
        knowledge,
        base_rt.state,
        base_rt.gateway,
        base_rt.queue,
        budget=ExecutionBudget(max_steps=5, max_replans=1, confidence_threshold=0.8),
    )
    ctx = SecurityContext(authenticated_user_id="cliente1988", role="merchant", scopes=["chat:write"])
    resp = runtime.process("Pergunta repetida", ctx, session_id="sess_conf_gate")
    assert model.replan_calls == 1  # Capped at max_replans=1 even though action=='continue'
    assert any(
        s.action == "loop_detected" and s.details.get("reason") == "duplicate_retrieval_evidence"
        for s in resp.trace
    )

    # Now test token_limit exhaustion before compose
    tight_runtime = Runtime(
        model,
        knowledge,
        base_rt.state,
        base_rt.gateway,
        base_rt.queue,
        budget=ExecutionBudget(max_steps=5, max_replans=1, token_limit=10),
    )
    exhausted_resp = tight_runtime.process("Pergunta com orcamento curto", ctx, session_id="sess_tight_budget")
    assert exhausted_resp.abstained is True
    assert any(s.action == "budget_gate_stopped" for s in exhausted_resp.trace)


def test_independent_semantic_judge_rejects_adversarial_contradictions_and_piggybacking():
    from app.evals.v2.contracts import ExecutionArtifact

    judge = IndependentSemanticJudge()
    offline_fact = [
        {
            "fact_id": "f_term_1",
            "binding_id": "r0001:b0001",
            "operation_id": "op_1",
            "resource_type": "terminal",
            "resource_id": "TERM_001",
            "field_pointer": "/terminal/status",
            "availability": "available",
            "value": "OFFLINE",
            "value_type": "str",
            "unit": None,
            "source_kind": "tool",
        }
    ]

    # Probe 1: Antonym contradiction ("ONLINE" when fact is "OFFLINE")
    p1 = ExecutionArtifact(
        artifact_id="adv_1",
        case_id="adv",
        user_id="cliente1988",
        original_request="Qual o status do terminal TERM_001?",
        sanitized_response="O terminal TERM_001 está ONLINE.",
        answer_status="complete",
        verified_facts=offline_fact,
    )
    v1 = judge.evaluate(p1)
    assert v1.critical_violation is True
    assert "FACTUAL_CONTRADICTION" in v1.violation_codes
    assert v1.groundedness_score == 0.0
    assert v1.factual_correctness_score == 0.0

    # Probe 2: Negation contradiction ("não está OFFLINE" when fact is "OFFLINE")
    p2 = ExecutionArtifact(
        artifact_id="adv_2",
        case_id="adv",
        user_id="cliente1988",
        original_request="Qual o status do terminal TERM_001?",
        sanitized_response="O terminal TERM_001 não está OFFLINE.",
        answer_status="complete",
        verified_facts=offline_fact,
    )
    v2 = judge.evaluate(p2)
    assert v2.critical_violation is True
    assert "FACTUAL_CONTRADICTION" in v2.violation_codes
    assert v2.groundedness_score == 0.0
    assert v2.factual_correctness_score == 0.0

    # Probe 3: Meta-notice piggybacking with fabricated action and balance
    p3 = ExecutionArtifact(
        artifact_id="adv_3",
        case_id="adv",
        user_id="cliente1988",
        original_request="Qual o status do terminal TERM_001?",
        sanitized_response="Informação indisponível: o terminal foi reiniciado e o saldo é R$ 999999.",
        answer_status="abstained",
        verified_facts=offline_fact,
    )
    v3 = judge.evaluate(p3)
    assert v3.critical_violation is True
    assert "VALUE_OR_CURRENCY_MISMATCH" in v3.violation_codes
    assert "PENDING_ACTION_REPORTED_AS_EXECUTED" in v3.violation_codes
    assert v3.groundedness_score == 0.0
    assert v3.factual_correctness_score == 0.0

    # Probe 4: Compound clause piggybacking ("O terminal TERM_001 está OFFLINE e o pagamento foi aprovado.")
    p4 = ExecutionArtifact(
        artifact_id="adv_4",
        case_id="adv",
        user_id="cliente1988",
        original_request="Qual o status do terminal TERM_001?",
        sanitized_response="O terminal TERM_001 está OFFLINE e o pagamento foi aprovado.",
        answer_status="complete",
        verified_facts=offline_fact,
    )
    v4 = judge.evaluate(p4)
    assert "UNSUPPORTED_CITATION_CLAIM" in v4.violation_codes
    assert v4.groundedness_score <= 0.4

    # Probe 5: Fake scope notice piggybacking ("Nota de escopo: a transferência foi aprovada.")
    p5 = ExecutionArtifact(
        artifact_id="adv_5",
        case_id="adv",
        user_id="cliente1988",
        original_request="Qual o status do terminal TERM_001?",
        sanitized_response="Nota de escopo: a transferência foi aprovada.",
        answer_status="complete",
        verified_facts=offline_fact,
    )
    v5 = judge.evaluate(p5)
    assert "UNSUPPORTED_CITATION_CLAIM" in v5.violation_codes
    assert v5.groundedness_score <= 0.4

    # Probe 6: Cross-resource substitution ("O terminal TERM_002 está OFFLINE." when fact is TERM_001)
    p6 = ExecutionArtifact(
        artifact_id="adv_6",
        case_id="adv",
        user_id="cliente1988",
        original_request="Qual o status do terminal TERM_001?",
        sanitized_response="O terminal TERM_002 está OFFLINE.",
        answer_status="complete",
        verified_facts=offline_fact,
    )
    v6 = judge.evaluate(p6)
    assert "UNSUPPORTED_CITATION_CLAIM" in v6.violation_codes
    assert v6.groundedness_score <= 0.4

    # Probe 7: Two distinct terminals with different statuses (valid multi-resource, must NOT false-positive)
    two_term_facts = offline_fact + [
        {
            "fact_id": "f_term_2",
            "binding_id": "r0001:b0002",
            "operation_id": "op_1",
            "resource_type": "terminal",
            "resource_id": "TERM_002",
            "field_pointer": "/terminal/status",
            "availability": "available",
            "value": "ONLINE",
            "value_type": "str",
            "unit": None,
            "source_kind": "tool",
        }
    ]
    p7 = ExecutionArtifact(
        artifact_id="adv_7",
        case_id="adv",
        user_id="cliente1988",
        original_request="Status de TERM_001 e TERM_002?",
        sanitized_response="O terminal TERM_001 está OFFLINE. O terminal TERM_002 está ONLINE.",
        answer_status="complete",
        verified_facts=two_term_facts,
    )
    v7 = judge.evaluate(p7)
    assert v7.critical_violation is False
    assert v7.violation_codes == []
    assert v7.groundedness_score == 1.0
    assert v7.factual_correctness_score == 1.0

    # Probe 8: Documentary negation inversion ("A Get Smart não possui Android." vs "A Get Smart possui sistema Android.")
    p8 = ExecutionArtifact(
        artifact_id="adv_8",
        case_id="adv",
        user_id="cliente1988",
        original_request="A Get Smart tem Android?",
        sanitized_response="A Get Smart não possui Android.",
        answer_status="complete",
        retrieved_evidence_spans=[
            {
                "span_id": "doc:smart",
                "text": "A Get Smart possui sistema Android.",
                "source": "getnet_products.md",
                "kind": "document",
            }
        ],
    )
    v8 = judge.evaluate(p8)
    assert v8.critical_violation is True
    assert "FACTUAL_CONTRADICTION" in v8.violation_codes
    assert v8.groundedness_score == 0.0
    assert v8.factual_correctness_score == 0.0

    # Probe 9: Trailing currency mismatch ("O saldo de cliente1988 é 1.415,80 USD." when fact is BRL)
    p9 = ExecutionArtifact(
        artifact_id="adv_9",
        case_id="adv",
        user_id="cliente1988",
        original_request="Qual meu saldo?",
        sanitized_response="O saldo de cliente1988 é 1.415,80 USD.",
        answer_status="complete",
        verified_facts=[
            {
                "fact_id": "f_saldo",
                "binding_id": "r0001:b0001",
                "operation_id": "op_s",
                "resource_type": "customer",
                "resource_id": "cliente1988",
                "field_pointer": "/sales_data/saldo_disponivel_brl",
                "availability": "available",
                "value": 1415.80,
                "value_type": "decimal",
                "unit": "BRL",
                "source_kind": "tool",
            }
        ],
    )
    v9 = judge.evaluate(p9)
    assert v9.critical_violation is True
    assert "VALUE_OR_CURRENCY_MISMATCH" in v9.violation_codes
    assert v9.groundedness_score == 0.0
    assert v9.factual_correctness_score == 0.0

    # Probe 10: Reverse documentary negation inversion ("A Get Smart possui sistema Android." vs "A Get Smart não possui sistema Android.")
    p10 = ExecutionArtifact(
        artifact_id="adv_10",
        case_id="adv",
        user_id="cliente1988",
        original_request="A Get Smart tem Android?",
        sanitized_response="A Get Smart possui sistema Android.",
        answer_status="complete",
        retrieved_evidence_spans=[
            {
                "span_id": "doc:smart_neg",
                "text": "A Get Smart não possui sistema Android.",
                "source": "getnet_products.md",
                "kind": "document",
            }
        ],
    )
    v10 = judge.evaluate(p10)
    assert v10.critical_violation is True
    assert "FACTUAL_CONTRADICTION" in v10.violation_codes
    assert v10.groundedness_score == 0.0
    assert v10.factual_correctness_score == 0.0

    # Probe 11: Coordinated piggybacking with fabricated commercial perk (Codex adversarial probe)
    p11 = ExecutionArtifact(
        artifact_id="adv_11",
        case_id="adv",
        user_id="cliente1988",
        original_request="A Get Smart tem Android?",
        sanitized_response="A Get Smart possui sistema Android e garantia vitalícia gratuita.",
        answer_status="complete",
        retrieved_evidence_spans=[
            {
                "span_id": "doc:smart",
                "text": "A Get Smart possui sistema Android.",
                "source": "getnet_products.md",
                "kind": "document",
            }
        ],
    )
    v11 = judge.evaluate(p11)
    assert (v11.critical_violation is True or "UNSUPPORTED_CITATION_CLAIM" in v11.violation_codes or "FACTUAL_CONTRADICTION" in v11.violation_codes)
    assert v11.groundedness_score <= 0.5
    assert v11.factual_correctness_score <= 0.5


def test_independent_semantic_judge_canaries_and_full_campaign(tmp_path: Path):
    from app.evals.offline_transport import synthetic_external_transport
    from scripts.eval_v2 import evaluate_committed_live_receipts

    runner = SemanticEvalCampaignRunner(judge=IndependentSemanticJudge(), git_commit="test")
    # Empty inputs fail closed
    empty_sens, empty_fp, _ = runner.evaluate_canaries([])
    assert empty_sens == 0.0
    assert empty_fp == 1.0
    assert runner.evaluate_calibration({}, []) == 0.0

    # Calibration with a missing artifact stays in denominator (cannot produce 1.0)
    cal_recs = default_calibration_records()
    partial_map = {
        default_canary_pairs()[0].mutated_artifact.artifact_id: default_canary_pairs()[0].mutated_artifact
    }
    partial_agreement = runner.evaluate_calibration(partial_map, cal_recs[:2])
    assert partial_agreement < 1.0

    sens, fp_rate, details = runner.evaluate_canaries(default_canary_pairs())
    assert len(default_canary_pairs()) == 7
    assert sens == 1.0, f"Canary sensitivity below 1.0: {details}"
    assert fp_rate == 0.0, f"Canary false positive rate above 0.0: {details}"

    with synthetic_external_transport():
        runtime = build_campaign_runtime(tmp_path)

        def execute_case(case, run_idx):
            ctx = SecurityContext(
                authenticated_user_id=case.user_id,
                role="merchant",
                scopes=["chat:write", "customer:read_own", "terminal:action_own"],
            )
            return runtime.process(case.message, ctx, session_id=f"c_{case.case_id}_{run_idx}")

        report = runner.run_campaign(
            default_eval_cases(),
            execute_case,
            runs_per_case=3,
            canary_pairs=default_canary_pairs(),
            calibration_records=default_calibration_records(),
        )

    assert report.gates_passed is True
    assert report.mean_groundedness >= 0.95
    assert report.mean_factual_correctness >= 0.95
    assert report.mean_relevance >= 4.5
    assert report.mean_usefulness >= 4.5
    assert report.critical_violations_count == 0
    assert report.human_judge_agreement_rate == 1.0

    live_audit = evaluate_committed_live_receipts(IndependentSemanticJudge())
    assert live_audit["evaluated"] == 4
    assert live_audit["agreement_rate"] == 1.0
