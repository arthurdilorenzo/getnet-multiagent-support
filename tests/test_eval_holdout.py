"""Automated regression gate for the unmemorized holdout evaluation dataset."""

from pathlib import Path
from app.evals.evaluator import SystemEvaluator, validate_evaluation_report
from app.evals.offline_transport import synthetic_external_transport


def test_holdout_evaluation_acceptance_gate():
    repo_root = Path(__file__).resolve().parent.parent
    holdout_path = repo_root / "data" / "eval_holdout_dataset.json"
    assert holdout_path.exists(), "Holdout dataset file missing"

    with synthetic_external_transport():
        evaluator = SystemEvaluator(dataset_path=holdout_path)
        report = evaluator.run_suite()

    failures = validate_evaluation_report(report)
    assert not failures, f"Holdout acceptance failures: {failures}"

    metrics = report["metrics"]
    assert metrics["routing_accuracy"] == 1.0
    assert metrics["mean_tool_recall"] == 1.0
    assert metrics["mean_tool_precision"] == 1.0
    assert metrics["safe_tool_selection_rate"] == 1.0
    assert metrics["mean_groundedness_score"] >= 0.90
    assert metrics["overall_pass_rate"] >= 0.90
    assert metrics["hit_at_3"] >= 0.90
