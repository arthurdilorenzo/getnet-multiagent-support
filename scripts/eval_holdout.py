"""Independent Holdout Evaluation Runner.

Evaluates an unmemorized holdout dataset against the multi-agent system
and RAG retrieval pipeline to prove model/pipeline generalization beyond
the regression test set.
"""

import os
import sys

# Default to offline deterministic evaluation unless explicit live flag requested
if os.environ.get("GETNET_LIVE_TESTS") != "1":
    os.environ["GEMINI_API_KEY"] = ""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from app.evals.evaluator import SystemEvaluator, validate_evaluation_report
from app.evals.offline_transport import synthetic_external_transport


def run_holdout_evaluation(
    dataset_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    evaluator = SystemEvaluator(dataset_path=dataset_path)

    # Disable noisy info logs during test batch
    logging.disable(logging.WARNING)

    with synthetic_external_transport():
        report = evaluator.run_suite()

    logging.disable(logging.NOTSET)

    # Save artifact receipt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / "data" / "eval_holdout_dataset.json"
    output_path = repo_root / "docs" / "eval_results" / "holdout_evaluation.json"

    if not dataset_path.exists():
        sys.stderr.write(f"Holdout dataset not found at: {dataset_path}\n")
        return 1

    report = run_holdout_evaluation(dataset_path=dataset_path, output_path=output_path)
    metrics = report.get("metrics", {})

    summary = {
        "dataset": str(dataset_path.name),
        "total_cases": report.get("total_cases", 0),
        "routing_accuracy": metrics.get("routing_accuracy"),
        "mean_tool_recall": metrics.get("mean_tool_recall"),
        "mean_tool_precision": metrics.get("mean_tool_precision"),
        "safe_tool_selection_rate": metrics.get("safe_tool_selection_rate"),
        "mean_groundedness_score": metrics.get("mean_groundedness_score"),
        "overall_pass_rate": metrics.get("overall_pass_rate"),
        "hit_at_3": metrics.get("hit_at_3"),
        "mrr_at_5": metrics.get("mrr_at_5"),
        "receipt_saved": str(output_path),
    }

    sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    failures = validate_evaluation_report(report)
    if failures:
        sys.stderr.write("Holdout acceptance gates failed: " + "; ".join(failures) + "\n")
        return 1

    sys.stdout.write("All holdout evaluation acceptance gates PASSED successfully.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
