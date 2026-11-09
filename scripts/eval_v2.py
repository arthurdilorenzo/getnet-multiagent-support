#!/usr/bin/env python3
"""Reproducible Semantic Evaluation V2 Campaign Runner for the integrated V2 Runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evals.v2.contracts import EvalCaseV2, ExecutionArtifact
from app.evals.v2.datasets import (
    default_calibration_records,
    default_canary_pairs,
    default_eval_cases,
)
from app.evals.v2.judge import IndependentSemanticJudge
from app.evals.v2.runner import SemanticEvalCampaignRunner
from app.hitl.queue import HITLQueueManager
from app.memory.store import StateStore
from app.security.auth import SecurityContext
from app.v2.adapters import StateAdapter, ToolGateway
from app.v2.contracts import (
    AnswerDocument,
    BindingProposal,
    DocumentaryBlock,
    EvidenceSpan,
    OperationalBlock,
    PendingBlock,
    PlanProposal,
    RequirementDecision,
    RequirementProposal,
    StepProposal,
)
from app.v2.knowledge import KnowledgeService
from app.v2.runtime import Runtime


class DeterministicCampaignModel:
    """Deterministic model driving the real V2 Runtime, ToolGateway, KnowledgeService, and DomainPresenter."""

    def generate(self, schema, payload, *, phase: str, tracer):
        req_text = payload.get("original_request", "")
        tracer.record_tokens(prompt_text=json.dumps(payload), completion_text=req_text)

        if schema is PlanProposal:
            if "Get Clássica" in req_text:
                return PlanProposal(
                    requirements=[RequirementProposal(kind="documentary", summary="Comparar Get Clássica e Get Smart")],
                    steps=[StepProposal(kind="knowledge", query="Get Clássica Get Smart diferenças", requirement_indexes=[0])],
                )
            if "saldo disponível" in req_text:
                return PlanProposal(
                    requirements=[
                        RequirementProposal(
                            kind="operational",
                            summary="Consultar saldo disponível para liquidação",
                            bindings=[
                                BindingProposal(
                                    tool_name="get_sales_settlement",
                                    resource_type="customer",
                                    resource_id="current_customer",
                                    field_pointer="/sales_data/saldo_disponivel_brl",
                                    unit="BRL",
                                )
                            ],
                        )
                    ],
                    steps=[StepProposal(kind="tool", tool_name="get_sales_settlement", arguments={}, requirement_indexes=[0])],
                )
            if "status técnico" in req_text:
                return PlanProposal(
                    requirements=[
                        RequirementProposal(
                            kind="operational",
                            summary="Diagnosticar status do terminal",
                            bindings=[
                                BindingProposal(
                                    tool_name="diagnose_terminal_issue",
                                    resource_type="terminal",
                                    resource_id="TERM_001",
                                    field_pointer="/terminal/status",
                                )
                            ],
                        ),
                        RequirementProposal(kind="documentary", summary="Orientar procedimento técnico"),
                    ],
                    steps=[
                        StepProposal(
                            kind="tool",
                            tool_name="diagnose_terminal_issue",
                            arguments={"terminal_id": "TERM_001"},
                            requirement_indexes=[0],
                        )
                    ],
                )
            if "reinício remoto" in req_text:
                return PlanProposal(
                    requirements=[
                        RequirementProposal(
                            kind="operational",
                            summary="Propor reinício remoto do terminal",
                            bindings=[
                                BindingProposal(
                                    tool_name="execute_terminal_remote_action",
                                    resource_type="terminal",
                                    resource_id="TERM_001",
                                    field_pointer="/status",
                                )
                            ],
                        )
                    ],
                    steps=[
                        StepProposal(
                            kind="tool",
                            tool_name="execute_terminal_remote_action",
                            arguments={"terminal_id": "TERM_001", "action_type": "remote_reboot"},
                            requirement_indexes=[0],
                        )
                    ],
                )
            if "POS-9941" in req_text:
                return PlanProposal(
                    requirements=[
                        RequirementProposal(
                            kind="operational",
                            summary="Consultar último erro de transação do terminal POS-9941",
                            bindings=[
                                BindingProposal(
                                    tool_name="diagnose_terminal_issue",
                                    resource_type="terminal",
                                    resource_id="POS-9941",
                                    field_pointer="/ultimo_erro_transacao",
                                )
                            ],
                        )
                    ],
                    steps=[
                        StepProposal(
                            kind="tool",
                            tool_name="diagnose_terminal_issue",
                            arguments={"terminal_id": "POS-9941"},
                            requirement_indexes=[0],
                        )
                    ],
                )
            return PlanProposal(
                requirements=[
                    RequirementProposal(
                        kind="operational",
                        summary="Consultar cotação EUR/BRL",
                        bindings=[
                            BindingProposal(
                                tool_name="get_currency_exchange_rate",
                                resource_type="currency",
                                resource_id="EUR/BRL",
                                field_pointer="/rate",
                                unit="BRL",
                            )
                        ],
                    )
                ],
                steps=[
                    StepProposal(
                        kind="tool",
                        tool_name="get_currency_exchange_rate",
                        arguments={"base_currency": "EUR", "target_currency": "BRL"},
                        requirement_indexes=[0],
                    )
                ],
            )

        if schema is AnswerDocument:
            planned = payload.get("planned_requirements", [])
            spans = payload.get("evidence_spans", [])
            facts = payload.get("observed_facts", [])
            doc_spans = [s for s in spans if s.get("kind") == "document"]
            decisions = []
            blocks = []
            for req in planned:
                rid = req["requirement_id"]
                if req["kind"] == "documentary":
                    sid = doc_spans[0]["span_id"] if doc_spans else spans[0]["span_id"]
                    decisions.append(RequirementDecision(requirement_id=rid, status="supported"))
                    if "Get Clássica" in req_text:
                        text = (
                            "A Get Clássica é o terminal tradicional com foco em operações essenciais de balcão e conectividade "
                            "chip/Wi-Fi, enquanto a Get Smart opera com sistema Android, tela sensível ao toque e suporte a aplicativos de gestão."
                        )
                    else:
                        text = (
                            "Para o terminal offline com erro de conectividade, oriente verificar o sinal do chip e a conexão Wi-Fi "
                            "antes de solicitar nova tentativa."
                        )
                    blocks.append(DocumentaryBlock(requirement_id=rid, text=text, span_ids=[sid]))
                else:
                    req_binding_ids = {b["binding_id"] for b in req.get("bindings", [])}
                    matching_facts = [f for f in facts if f.get("binding_id") in req_binding_ids]
                    decisions.append(RequirementDecision(requirement_id=rid, status="supported"))
                    blocks.append(OperationalBlock(requirement_id=rid, fact_ids=[f["fact_id"] for f in matching_facts]))
            return AnswerDocument(requirements=decisions, final_answer=blocks)

        raise ValueError(f"Unsupported schema: {schema}")


def build_campaign_runtime(tmp_dir: Path) -> Runtime:
    customers_data = {
        "customers": {
            "cliente1988": {
                "user_id": "cliente1988",
                "status": "ACTIVE",
                "segment": "Varejo Alimentar",
                "terminals": [
                    {
                        "terminal_id": "TERM_001",
                        "model": "Get Smart",
                        "status": "OFFLINE",
                        "connectivity": "3G/Wi-Fi",
                        "ultimo_erro_transacao": "E_SIM_NO_SIGNAL",
                    },
                    {
                        "terminal_id": "POS-9941",
                        "model": "Get Clássica",
                        "status": "ONLINE",
                        "connectivity": "GPRS",
                    },
                ],
                "sales_settlement": {
                    "yesterday_sales": {
                        "date": "2026-10-20",
                        "saldo_disponivel_brl": 1415.80,
                        "gross_amount_brl": 1500.00,
                        "net_amount_brl": 1415.80,
                    }
                },
            }
        }
    }
    customers_path = tmp_dir / "customers.json"
    customers_path.write_text(json.dumps(customers_data, ensure_ascii=False), encoding="utf-8")

    store = StateStore(tmp_dir / "eval_v2.db")
    queue = HITLQueueManager(store)
    gateway = ToolGateway(customers_path, store=store, queue=queue)
    gateway.currency = lambda base_currency="EUR", target_currency="BRL": {
        "pair": f"{base_currency}/{target_currency}",
        "rate_type": "daily_reference",
        "rate": 6.12345,
        "date": "2026-10-28",
        "source": "ECB/Frankfurter",
    }

    def retrieve_fn(query: str):
        return {
            "retrieval_mode": "hybrid_neural_bm25",
            "candidate_chunks": [
                {
                    "chunk_id": "pos_catalog_01",
                    "source": "getnet_maquininhas.md",
                    "source_url": "https://site.getnet.com.br/maquininhas",
                    "content": (
                        "A Get Clássica é o terminal tradicional com foco em operações essenciais de balcão e conectividade "
                        "chip/Wi-Fi. A Get Smart opera com sistema Android, tela sensível ao toque e aplicativos de gestão. "
                        "Em caso de terminal offline com erro E_SIM_NO_SIGNAL, verifique o sinal do chip e a conexão Wi-Fi."
                    ),
                }
            ],
        }

    model = DeterministicCampaignModel()
    knowledge = KnowledgeService(model, retrieve=retrieve_fn)
    return Runtime(model, knowledge, StateAdapter(store), gateway, queue)


def build_live_campaign_runtime(tmp_dir: Path) -> Runtime:
    from app.v2.model import GeminiModel

    customers_data = {
        "customers": {
            "cliente1988": {
                "user_id": "cliente1988",
                "status": "ACTIVE",
                "segment": "Varejo Alimentar",
                "terminals": [
                    {
                        "terminal_id": "TERM_001",
                        "model": "Get Smart",
                        "status": "OFFLINE",
                        "connectivity": "3G/Wi-Fi",
                        "ultimo_erro_transacao": "E_SIM_NO_SIGNAL",
                    },
                    {
                        "terminal_id": "POS-9941",
                        "model": "Get Clássica",
                        "status": "ONLINE",
                        "connectivity": "GPRS",
                    },
                ],
                "sales_settlement": {
                    "yesterday_sales": {
                        "date": "2026-10-20",
                        "saldo_disponivel_brl": 1415.80,
                        "gross_amount_brl": 1500.00,
                        "net_amount_brl": 1415.80,
                    }
                },
            }
        }
    }
    customers_path = tmp_dir / "customers.json"
    customers_path.write_text(json.dumps(customers_data, ensure_ascii=False), encoding="utf-8")

    store = StateStore(tmp_dir / "eval_v2_live.db")
    queue = HITLQueueManager(store)
    gateway = ToolGateway(customers_path, store=store, queue=queue)
    model = GeminiModel(supports_replan=True)
    knowledge = KnowledgeService(model)
    return Runtime(model, knowledge, StateAdapter(store), gateway, queue)


def evaluate_committed_live_receipts(judge: IndependentSemanticJudge) -> dict:
    """Evaluates committed live Gemini V2 HTTP transcripts against human adjudications."""
    receipt_dir = ROOT / "archive_audits" / "getnet_final_new_questions_2026-10-29" / "receipts"
    http_path = receipt_dir / "http.jsonl"
    adj_path = receipt_dir / "adjudication.jsonl"
    if not http_path.exists() or not adj_path.exists():
        return {"evaluated": 0, "agreement_rate": 0.0, "cases": []}

    requests_by_case = {}
    responses_by_case = {}
    for line in http_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "request":
            requests_by_case[row["case"]] = row["body"]
        elif row.get("event") == "response":
            responses_by_case[row["case"]] = row["body"]

    adjudications = {}
    for line in adj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        adjudications[row["case"]] = row

    live_expectations = {
        "N01_recommendation": EvalCaseV2(
            case_id="N01_recommendation",
            category="live_v2_receipt",
            user_id="new_papelaria",
            message=requests_by_case.get("N01_recommendation", {}).get("message", ""),
            expected_answer_status="complete",
            required_facts_or_terms=["papelaria", "Get Clássica", "Get Smart", "estoque"],
        ),
        "N02_missing_diagnostic": EvalCaseV2(
            case_id="N02_missing_diagnostic",
            category="live_v2_receipt",
            user_id="new_terminal",
            message=requests_by_case.get("N02_missing_diagnostic", {}).get("message", ""),
            expected_answer_status="partial",
            required_facts_or_terms=["SIM1", "ativo", "Informação indisponível"],
        ),
        "N03_snapshot_zero": EvalCaseV2(
            case_id="N03_snapshot_zero",
            category="live_v2_receipt",
            user_id="new_snapshot",
            message=requests_by_case.get("N03_snapshot_zero", {}).get("message", ""),
            expected_answer_status="abstained",
            required_facts_or_terms=["0,00", "Snapshot histórico", "Informação indisponível"],
        ),
        "N04_public_pair": EvalCaseV2(
            case_id="N04_public_pair",
            category="live_v2_receipt",
            user_id="new_public",
            message=requests_by_case.get("N04_public_pair", {}).get("message", ""),
            expected_answer_status="complete",
            required_facts_or_terms=["5.3849", "Campinas", "26", "18.1"],
        ),
    }

    case_results = []
    agreements = 0
    for case_id, ref_case in live_expectations.items():
        resp_body = responses_by_case.get(case_id)
        adj = adjudications.get(case_id)
        if not resp_body or not adj:
            continue
        ev_list = resp_body.get("evidence", [])
        artifact = ExecutionArtifact(
            artifact_id=f"live_{case_id}",
            case_id=case_id,
            user_id=ref_case.user_id,
            original_request=ref_case.message,
            sanitized_response=resp_body.get("response", ""),
            answer_status=resp_body.get("answer_status", "abstained"),
            abstained=bool(resp_body.get("abstained", False)),
            escalated=bool(resp_body.get("escalated", False)),
            tools_called=list(resp_body.get("tools_called", [])),
            retrieved_evidence_spans=[e for e in ev_list if isinstance(e, dict) and "span_id" in e],
            verified_facts=[e for e in ev_list if isinstance(e, dict) and "fact_id" in e],
            action_proposals=list(resp_body.get("action_proposals", [])),
            trace_id=resp_body.get("trace_id", ""),
        )
        verdict = judge.evaluate(artifact, reference=ref_case)
        judge_pass = (
            not verdict.critical_violation
            and "PARTIAL_REQUEST_OMISSION" not in verdict.violation_codes
            and verdict.factual_correctness_score >= 0.8
        )
        human_pass = adj.get("verdict") in {"PASS", "EXPECTED_PARTIAL"}
        agreed = judge_pass == human_pass
        if agreed:
            agreements += 1
        case_results.append(
            {
                "case_id": case_id,
                "prior_adjudicator_verdict": adj.get("verdict"),
                "prior_adjudicator_reviewer": adj.get("reviewer", "prior_ai_adjudicator"),
                "judge_pass": judge_pass,
                "agreed": agreed,
                "violations": verdict.violation_codes,
                "groundedness": verdict.groundedness_score,
                "factual_correctness": verdict.factual_correctness_score,
            }
        )

    total = len(case_results)
    return {
        "source_receipt": "archive_audits/getnet_final_new_questions_2026-10-29/receipts/http.jsonl",
        "adjudication_provenance": "prior_ai_adjudication_not_independent_human",
        "evaluated": total,
        "agreement_rate": round(agreements / total, 4) if total else 0.0,
        "cases": case_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2 Semantic Evaluation Campaign")
    parser.add_argument("--runs", type=int, default=None, help="Runs per case (defaults to 1 for live, 3 for offline)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run campaign live against configured Gemini provider",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write campaign receipt JSON",
    )
    args = parser.parse_args()

    runs_per_case = args.runs if args.runs is not None else (1 if args.live else 3)
    default_output = (
        str(ROOT / "data" / "v2_live_campaign_receipt.json")
        if args.live
        else str(ROOT / "data" / "v2_semantic_eval_campaign_receipt.json")
    )
    output_path = args.output or default_output

    import tempfile
    import time
    from contextlib import nullcontext

    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
        dirty_out = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
        git_dirty = bool(dirty_out)
    except Exception:
        git_sha = "HEAD"
        git_dirty = False

    from app.evals.offline_transport import synthetic_external_transport

    judge = IndependentSemanticJudge()
    transport_cm = nullcontext() if args.live else synthetic_external_transport()

    with tempfile.TemporaryDirectory() as tmp, transport_cm:
        tmp_path = Path(tmp)
        runtime = (
            build_live_campaign_runtime(tmp_path)
            if args.live
            else build_campaign_runtime(tmp_path)
        )
        runner = SemanticEvalCampaignRunner(
            judge=judge,
            git_commit=git_sha,
            git_dirty=git_dirty,
            runtime_version="v2.1-live" if args.live else "v2.1",
        )

        def execute_case(case: EvalCaseV2, run_idx: int):
            if args.live:
                time.sleep(5.0)
            ctx = SecurityContext(
                authenticated_user_id=case.user_id,
                role="merchant",
                scopes=["chat:write", "customer:read_own", "terminal:action_own"],
            )
            for attempt in range(3):
                resp = runtime.process(
                    case.message,
                    ctx,
                    session_id=f"eval_{case.case_id}_run_{run_idx}_{attempt}" if attempt > 0 else f"eval_{case.case_id}_run_{run_idx}",
                )
                is_transient_cloud_error = any(
                    s.action == "failed" and "ExternalServiceUnavailableError" in str(s.details)
                    for s in resp.trace
                )
                if not args.live or not is_transient_cloud_error:
                    return resp
                time.sleep(4.0)
            return resp

        live_audit = evaluate_committed_live_receipts(judge)
        report = runner.run_campaign(
            default_eval_cases(),
            execute_case,
            runs_per_case=runs_per_case,
            canary_pairs=default_canary_pairs(),
            calibration_records=default_calibration_records(),
            baseline_metrics={
                "mean_groundedness": 0.92,
                "mean_factual_correctness": 0.92,
                "mean_relevance": 4.2,
                "mean_usefulness": 3.8,
            },
            live_receipt_audit=live_audit,
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if report.gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
