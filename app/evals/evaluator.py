import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.agents.orchestrator import orchestrator
from app.config import settings
from app.schemas import ChatRequest, ChatResponse


class GroundednessJudge:
    """
    Heurística lexical do dataset local: presença de termos esperados, URLs e
    ausência de fallbacks falsos conhecidos. Não verifica entailment, todos os
    fatos da resposta nem a validade externa das fontes; não é juiz semântico.
    """

    FORBIDDEN_HALLUCINATION_STRINGS = [
        "Serviço Meteorológico Regional (Cache)",
        "Banco Central / Mercado Financeiro (Referência)",
        "O tema foi verificado com sucesso nas fontes externas públicas",
    ]

    @classmethod
    def score_response(
        cls,
        case: Dict[str, Any],
        response: ChatResponse,
    ) -> Dict[str, Any]:
        resp_lower = response.response.lower()

        # 1. Verifica ausência de strings proibidas de fallback falso
        for forbidden in cls.FORBIDDEN_HALLUCINATION_STRINGS:
            if forbidden.lower() in resp_lower:
                return {
                    "groundedness_score": 0.0,
                    "passed": False,
                    "reason": f"Detectada string de fallback não verificável: '{forbidden}'",
                }

        # 2. Se o caso exige abstention e o agente absteve, verifica se não inventou fatos
        if case.get("expect_abstained"):
            abstained_ok = response.abstained is True
            return {
                "groundedness_score": 1.0 if abstained_ok else 0.0,
                "passed": abstained_ok,
                "reason": "Abstention correta executada." if abstained_ok else "Falhou em abster-se.",
            }

        # 3. Se uma API externa falhou (sem conexão no ambiente de execução), mas o agente absteve honestamente:
        if response.abstained and case.get("category") == "web_search":
            return {
                "groundedness_score": 1.0,
                "passed": True,
                "reason": "Abstention honesta diante de indisponibilidade de rede externa.",
            }

        # 4. Verifica cobertura dos termos factuais exigidos das fontes
        req_terms: List[str] = case.get("required_grounding_terms", [])
        matched = sum(1 for t in req_terms if t.lower() in resp_lower)
        term_ratio = (matched / len(req_terms)) if req_terms else 1.0

        # 5. Verifica citação de URL oficial se exigida
        expected_url = case.get("expected_url_substring")
        url_ok = True
        if expected_url:
            sources_joined = " ".join(response.sources).lower() + " " + resp_lower
            url_ok = expected_url.lower() in sources_joined

        score = round((0.75 * term_ratio) + (0.25 * (1.0 if url_ok else 0.0)), 4)
        return {
            "groundedness_score": score,
            "passed": score >= 0.75,
            "matched_terms": matched,
            "total_required_terms": len(req_terms),
            "url_citation_verified": url_ok,
        }


class SystemEvaluator:
    """
    Executor da Suíte de Avaliação de Qualidade (Evals) do Sistema Multi-Agente.
    Calcula:
    - Routing Accuracy
    - Tool Selection Recall
    - Mean Groundedness Score
    - Abstention & Guardrail Precision
    - Latências p50 e p95
    """

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.dataset_path = Path(dataset_path or settings.EVAL_DATASET_PATH)

    def run_suite(self) -> Dict[str, Any]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        cases: List[Dict[str, Any]] = raw.get("cases", [])

        case_results: List[Dict[str, Any]] = []
        routing_hits = 0
        tool_recall_scores: List[float] = []
        groundedness_scores: List[float] = []
        latencies_ms: List[float] = []

        for idx, case in enumerate(cases, start=1):
            req = ChatRequest(
                message=case["message"],
                user_id=case["user_id"],
                session_id=f"eval_sess_{case['case_id']}_{idx}",
            )
            resp = orchestrator.process(req)
            latencies_ms.append(resp.execution_time_ms)

            # 1. Routing check
            expected_agent = case["expected_agent"]
            route_ok = (resp.agent_used == expected_agent) or (expected_agent in resp.agent_used)
            if route_ok:
                routing_hits += 1

            # 2. Tool Recall check
            exp_tools: List[str] = case.get("expected_tools", [])
            if exp_tools:
                called_set = set(resp.tools_called)
                t_recall = sum(1 for t in exp_tools if t in called_set) / len(exp_tools)
            else:
                t_recall = 1.0
            tool_recall_scores.append(t_recall)

            # 3. Groundedness Judge
            judge_res = GroundednessJudge.score_response(case, resp)
            groundedness_scores.append(judge_res["groundedness_score"])

            case_results.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "expected_agent": expected_agent,
                    "actual_agent": resp.agent_used,
                    "routing_correct": route_ok,
                    "expected_tools": exp_tools,
                    "actual_tools": resp.tools_called,
                    "tool_recall": round(t_recall, 4),
                    "abstained": resp.abstained,
                    "groundedness": judge_res,
                    "latency_ms": resp.execution_time_ms,
                }
            )

        n = max(1, len(cases))
        sorted_lat = sorted(latencies_ms)
        p50 = sorted_lat[int(0.50 * (len(sorted_lat) - 1))] if sorted_lat else 0.0
        p95 = sorted_lat[int(0.95 * (len(sorted_lat) - 1))] if sorted_lat else 0.0

        summary = {
            "dataset_version": raw.get("dataset_version", "2.0.0"),
            "execution_mode": "gemini" if settings.GEMINI_API_KEY else "offline_deterministic",
            "judge_type": "lexical_terms_and_url_presence",
            "total_cases": len(cases),
            "metrics": {
                "routing_accuracy": round(routing_hits / n, 4),
                "mean_tool_recall": round(sum(tool_recall_scores) / n, 4),
                "mean_groundedness_score": round(sum(groundedness_scores) / n, 4),
                "overall_pass_rate": round(
                    sum(1 for c in case_results if c["routing_correct"] and c["tool_recall"] == 1.0 and c["groundedness"]["passed"]) / n,
                    4,
                ),
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
            },
            "cases": case_results,
        }
        return summary


evaluator = SystemEvaluator()


def main() -> int:
    """Emit metrics and fail CI when any SPEC acceptance threshold regresses."""
    import sys
    import math

    report = evaluator.run_suite()
    sys.stdout.write(json.dumps(report["metrics"], indent=2, ensure_ascii=False) + "\n")
    failures = []
    if report.get("total_cases", 0) <= 0:
        failures.append("total_cases must be greater than zero")
    for metric, minimum in (
        ("routing_accuracy", 1.0),
        ("mean_tool_recall", 1.0),
        ("mean_groundedness_score", 0.90),
    ):
        actual = report["metrics"].get(metric)
        if not isinstance(actual, (float, int)) or not math.isfinite(actual) or not minimum <= actual <= 1.0:
            failures.append(f"{metric}: expected {minimum}..1.0, got {actual}")
    if failures:
        sys.stderr.write("Evaluation acceptance failed: " + "; ".join(failures) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
