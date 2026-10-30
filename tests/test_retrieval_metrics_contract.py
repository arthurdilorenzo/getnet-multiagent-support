import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from app.evals.evaluator import SystemEvaluator, validate_evaluation_report
from app.rag.retriever import HybridKnowledgeRetriever


def test_real_system_evaluator_retrieval_benchmark_mathematics():
    """
    Exercita o método de produção SystemEvaluator.run_retrieval_benchmark diretamente.
    Stub apenas na fronteira de retrieve_hybrid para fornecer ranks controlados:
    - rank1: relevante no rank 1 -> hit@1=1, hit@3=1, hit@5=1, rr=1.0
    - rank4: relevante no rank 4 -> hit@1=0, hit@3=0, hit@5=1, rr=0.25
    - missing: não encontrado -> hit@1=0, hit@3=0, hit@5=0, rr=0.0
    Esperado:
    hit_at_1 = 1/3 = 0.3333
    hit_at_3 = 1/3 = 0.3333
    hit_at_5 = 2/3 = 0.6667
    mrr_at_5 = (1.0 + 0.25 + 0.0)/3 = 0.4167
    """
    cases = [
        {"message": "rank1", "category": "rag_test", "expected_tools": ["search_getnet_knowledge"], "expected_chunk_ids": ["good_chunk"]},
        {"message": "rank4", "category": "rag_test", "expected_tools": ["search_getnet_knowledge"], "expected_chunk_ids": ["good_chunk"]},
        {"message": "missing", "category": "rag_test", "expected_tools": ["search_getnet_knowledge"], "expected_chunk_ids": ["good_chunk"]},
    ]

    def fake_retrieve_hybrid(query, **kwargs):
        rank_mapping = {
            "rank1": ["good_chunk"],
            "rank4": ["other_1", "other_2", "other_3", "good_chunk"],
            "missing": ["other_1", "other_2"],
        }
        ids = rank_mapping.get(query, [])
        return {"results": [{"chunk_id": cid, "source_url": "https://site.getnet.com.br/"} for cid in ids]}

    with patch.object(HybridKnowledgeRetriever, "retrieve_hybrid", side_effect=fake_retrieve_hybrid):
        real_math = SystemEvaluator.run_retrieval_benchmark(cases)

    assert real_math["cases_evaluated"] == 3
    assert real_math["hit_at_1"] == pytest.approx(0.3333, abs=1e-4)
    assert real_math["hit_at_3"] == pytest.approx(0.3333, abs=1e-4)
    assert real_math["hit_at_5"] == pytest.approx(0.6667, abs=1e-4)
    assert real_math["mrr_at_5"] == pytest.approx(0.4167, abs=1e-4)
    # Metadados de depreciação de recall presentes
    assert "deprecations" in real_math
    assert "recall_at_3" in real_math["deprecations"]


def test_empty_rag_cases_returns_unavailable_not_vacuous_perfection():
    """Benchmark sem casos retorna indisponível no método (None), não perfeição artificial."""
    res = SystemEvaluator.run_retrieval_benchmark([])
    assert res["cases_evaluated"] == 0
    assert res["hit_at_1"] is None
    assert res["hit_at_3"] is None
    assert res["hit_at_5"] is None
    assert res["mrr_at_5"] is None


def test_gate_validator_rejects_missing_or_degraded_retrieval():
    """Validador puro de aceitação reprova métricas ausentes, inválidas ou abaixo dos pisos."""
    base_report = {
        "total_cases": 25,
        "execution_mode": "offline_deterministic",
        "metrics": {
            "routing_accuracy": 1.0,
            "mean_tool_recall": 1.0,
            "mean_tool_precision": 1.0,
            "safe_tool_selection_rate": 1.0,
            "mean_groundedness_score": 1.0,
            "overall_pass_rate": 0.96,
            "hit_at_3": 1.0,
            "hit_at_5": 1.0,
            "mrr_at_5": 1.0,
            "retrieval_cases_evaluated": 12,
        },
    }

    # Caso válido: passa sem falhas
    assert validate_evaluation_report(base_report) == []

    # Rejeita retrieval_cases_evaluated = 0 ou ausente
    rep_missing = copy.deepcopy(base_report)
    rep_missing["metrics"]["retrieval_cases_evaluated"] = 0
    rep_missing["metrics"]["hit_at_3"] = None
    failures = validate_evaluation_report(rep_missing)
    assert any("retrieval_cases_evaluated" in f for f in failures)

    # Rejeita retrieval_cases_evaluated booleano
    rep_bool = copy.deepcopy(base_report)
    rep_bool["metrics"]["retrieval_cases_evaluated"] = True
    assert any("retrieval_cases_evaluated" in f for f in validate_evaluation_report(rep_bool))

    # Rejeita métricas abaixo dos pisos (Hit@3 < 0.90)
    rep_low = copy.deepcopy(base_report)
    rep_low["metrics"]["hit_at_3"] = 0.85
    failures_low = validate_evaluation_report(rep_low)
    assert any("hit_at_3: expected 0.9" in f for f in failures_low)

    # Rejeita total_cases <= 0
    rep_no_cases = copy.deepcopy(base_report)
    rep_no_cases["total_cases"] = 0
    assert any("total_cases" in f for f in validate_evaluation_report(rep_no_cases))


def test_cli_subprocess_rejects_empty_retrieval_dataset(tmp_path):
    """
    Subprocesso do CLI real (sem mocks no processo avaliado):
    Copia o dataset de 25 casos para temporário, preservando os 25 casos,
    mas removendo rótulos de relevância de retrieval. O CLI deve falhar com exit não-zero.
    """
    root = Path(__file__).resolve().parents[1]
    original_dataset = json.loads((root / "data" / "eval_dataset.json").read_text(encoding="utf-8"))
    modified_cases = []
    for c in original_dataset["cases"]:
        m = copy.deepcopy(c)
        m.pop("expected_url_substring", None)
        m.pop("expected_chunk_ids", None)
        if m.get("category", "").startswith("rag_"):
            m["category"] = "general_support"
        modified_cases.append(m)

    tmp_dataset = tmp_path / "dataset_no_retrieval.json"
    tmp_dataset.write_text(json.dumps({"cases": modified_cases}), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["GEMINI_API_KEY"] = ""
    env["GETNET_LIVE_EMBEDDINGS"] = "0"
    env["EVAL_DATASET_PATH"] = str(tmp_dataset)

    proc = subprocess.run(
        [sys.executable, "-m", "app.evals.evaluator"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0, f"Expected non-zero exit code, got {proc.returncode}"
    assert "Evaluation acceptance failed" in proc.stderr
    assert "retrieval_cases_evaluated" in proc.stderr
