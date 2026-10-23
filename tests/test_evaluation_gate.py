"""The CLI must fail CI when any published acceptance metric regresses."""

import json
import importlib
import pytest

eval_module = importlib.import_module("app.evals.evaluator")


@pytest.mark.parametrize(
    "metric,value",
    [("routing_accuracy", 0.99), ("mean_tool_recall", 0.99), ("mean_groundedness_score", 0.8999)],
)
def test_cli_fails_when_an_acceptance_metric_is_below_spec(monkeypatch, capsys, metric, value):
    report = {
        "total_cases": 13,
        "metrics": {
            "routing_accuracy": 1.0,
            "mean_tool_recall": 1.0,
            "mean_groundedness_score": 0.90,
        },
    }
    report["metrics"][metric] = value
    monkeypatch.setattr(eval_module.evaluator, "run_suite", lambda: report)
    assert eval_module.main() == 1
    captured = capsys.readouterr()
    assert metric in captured.err
    assert json.loads(captured.out)[metric] == value


def test_cli_accepts_exact_thresholds(monkeypatch, capsys):
    report = {
        "total_cases": 13,
        "metrics": {"routing_accuracy": 1.0, "mean_tool_recall": 1.0, "mean_groundedness_score": 0.90},
    }
    monkeypatch.setattr(eval_module.evaluator, "run_suite", lambda: report)
    assert eval_module.main() == 0
    assert json.loads(capsys.readouterr().out)["mean_groundedness_score"] == 0.90


def test_cli_rejects_empty_evaluation_dataset(monkeypatch, capsys):
    monkeypatch.setattr(eval_module.evaluator, "run_suite", lambda: {
        "total_cases": 0,
        "metrics": {"routing_accuracy": 1.0, "mean_tool_recall": 1.0, "mean_groundedness_score": 1.0},
    })
    assert eval_module.main() == 1
    assert "total_cases" in capsys.readouterr().err
