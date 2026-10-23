# Frente M08 A3 — plain postfix binding and Router retrieval boundary

Date: 2026-10-23. Baseline: `de1f750f25b8c766eecb67e9eb56c22183d3ff5d`.

## Scope and evidence

Read `SPEC_V11_ADDENDUM_M08_ROUTER_PROBE.md`, `agent-a3.md`, and `agent-a2-report.md` before code changes. Changed only `app/agents/orchestrator.py`, `app/agents/router_agent.py`, `tests/test_conditional_orchestration_v11.py`, and this report. The root-owned addendum is excluded from this commit. No P05/KnowledgeAgent implementation, corpus, evaluator, freeze, labels, budgets, API/network calls, push, promotion, or subagents.

## Root causes and correction

1. The dependency detector linked the preceding sentence only under `elif prefix and sentence_start`. A bare `If`, `Provided that`, or `Desde que` has an empty prefix. It therefore blocked the predicate but left the preceding benefits request runnable. Detection now links the previous sentence even with an empty prefix. Earlier unknown qualifiers remain attached; clear independent requests remain outside the dependent span. The detector also recognizes the explicitly specified `Since` and `As long as` markers.
2. The native Router called `_prefer_local_documentary_evidence` before the orchestrator evaluated its gate. That checker sent each Web-assigned unit to `retriever.retrieve_candidates`, including dependent disclosure and predicate text. A later specialist gate cannot undo this retrieval. The single shared detector now lives on `RouterAgent.conditional_documentary_dependency`; the original orchestrator method delegates to it. The real relevance checker computes the dependent unit IDs and declines candidate retrieval for those IDs. Ordinary independent units retain the existing official-source relevance/provenance checks and route selection. No circular imports or duplicate detector grammars were introduced.

Postfix normalization is restricted to the existing literal benefits request plus an enumerated lexical marker (`Only if`, `If`, `Provided that`, `As long as`, `Since`, `Somente se`, `Se`, `Desde que`, with the existing optional contrast prefix). It passes the predicate unchanged to `parse_terminal_condition`. The parser and `evaluate_terminal_condition` are unchanged: exact literal firmware/Online conjunction, one explicit terminal, completed successful typed `diagnose_terminal_issue`, exact target, expected firmware/status, and no extra conditions. Unknown ranges/predicates/extra conditions do not gain authorization from lexical linking. Direct EN/PT status interrogatives preserve their bounded exemption.

## TDD and integration boundary

Added 41 collected cases, with the full native `RouterAgent.plan` implementation and `_prefer_local_documentary_evidence` active. A thin test wrapper enables synthetic native configuration only while calling the original bound `plan`; the provider client returns native Google SDK delegation objects without contacting a service. It does not replace planning, request-unit binding, protocol validation, relevance selection, specialist dispatch, or registry validation.

The `retrieve_candidates` spy forwards to the real local retriever (deterministic hash/BM25, no network). The real local `search_getnet_knowledge` handler is retained. Real specialists use their offline path; the diagnostic handler supplies controlled data and the external Web handler fails synthetically if dispatched. Registry recording forwards to real `ToolRegistry.execute`; a second wrapper records completed typed results, so an attempted or rejected diagnostic cannot satisfy the positive ordering assertion.

Coverage:

- 24 cases: six plain postfixes × wrong firmware/missing firmware/Offline/diagnostic failure; zero retriever calls and only an exact-target diagnostic registry read.
- Six cases: an extra unknown predicate with each marker; zero candidate retrieval and zero registry reads, all coverage remains unresolved.
- Eight positives: the original Crediário form, the earlier Only-if postfix, and six new postfixes. The first event after the real Router checker is a completed, successful typed diagnostic with exact terminal, firmware and Online status. Only then does the real local retriever receive the entire canonical dependent request; no earlier candidate query occurs.
- One mixed-scope case: independent Pix question still reaches the real native relevance probe, selects local Knowledge and retrieves its canonical independent query; unsupported subsequent benefits remain blocked.
- Two direct Support cases: `Check if my terminal ... is Online` and `Verifique se o terminal ... está Online`; exact diagnostic succeeds without candidate retrieval or a conditional-disclosure block.

## Exact commands and results

Commands ran from `/private/tmp/getnet-v11-worktree`. Pytest prefix:

`GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/pytest`

RED, before any production changes:

`-q tests/test_conditional_orchestration_v11.py -k native_router`

Exact terminal summary:

```text
39 failed, 2 passed, 99 deselected in 1.86s
```

Log: `/private/tmp/getnet-v11-agent-a3-red.txt`. Failures showed actual pre-gate candidate queries such as `Tell me the eligible benefits.`, the firmware predicate and `and status is Online.`. The registry also received the leaked Web benefits query. Two direct Support cases passed as characterization coverage.

GREEN, after the shared detector and probe guard:

Same command. Exact output:

```text
.........................................                                [100%]
41 passed, 99 deselected in 1.25s
```

Log: `/private/tmp/getnet-v11-agent-a3-green.txt`.

Focused regressions:

`-q tests/test_conditional_orchestration_v11.py tests/test_conditional_dependencies_v11.py tests/test_pix_crediario_partial_v11.py tests/test_pix_getnet_routing_v11.py tests/test_replan_scope_v8.py tests/test_action_integrity.py tests/test_orchestrator_contracts.py tests/test_unit_rag_and_router.py`

Exact output:

```text
........................................................................ [ 30%]
........................................................................ [ 60%]
........................................................................ [ 90%]
........................                                                 [100%]
240 passed in 2.82s
```

Log: `/private/tmp/getnet-v11-agent-a3-focused.txt`.

Full suite:

`-v`

Exact terminal summary:

```text
======================= 1220 passed, 1 warning in 7.88s ========================
```

Warning: existing upstream `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` No test failures. Log: `/private/tmp/getnet-v11-agent-a3-full.txt`.

Offline evaluator:

`GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python -m app.evals.evaluator`

Exit 0. Exact final JSON:

```json
{
  "routing_accuracy": 1.0,
  "mean_tool_recall": 1.0,
  "mean_tool_precision": 1.0,
  "safe_tool_selection_rate": 1.0,
  "mean_groundedness_score": 1.0,
  "overall_pass_rate": 0.9231,
  "latency_p50_ms": 14.57,
  "latency_p95_ms": 22.48,
  "external_services_mode": "synthetic_external_api_fixtures",
  "execution_mode": "offline_deterministic",
  "metrics_by_mode": {
    "limited_offline": {
      "case_count": 12,
      "non_abstention_rate": 0.8333,
      "declared_complete_rate": 0.8333,
      "limited_consistency_pass_rate": 0.9167
    },
    "unclassified": {
      "case_count": 1,
      "non_abstention_rate": 0.0,
      "declared_complete_rate": 0.0,
      "limited_consistency_pass_rate": 1.0
    }
  },
  "mode_metrics_limitations": "Metadados e checks limitados não medem utilidade, falsa suficiência ou entailment; exigem adjudicação independente por necessidade."
}
```

Log: `/private/tmp/getnet-v11-agent-a3-evaluator.txt`.

Static checks:

- `/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python scripts/check_no_print.py app`: exit 0, empty stdout.
- `git diff --check`: exit 0, empty stdout.

## Limits and handoff

This is deterministic offline evidence of dataflow and authorization, including real native Router planning protocol with a synthetic provider, not an executed Gemini run or a V4 quality score. No Docker or exact-SHA hosted CI claim. Marker/diagnostic language remains deliberately finite; unrecognized conditions stay blocked rather than inferred. The Router does no dependent retrieval even for a parseable condition until the orchestrator obtains typed proof; the authorized specialist then performs its ordinary documentary lookup.

Root must independently review the whole candidate, append the operational WORKLOG summary linking this report, rebind any native freeze to the final reviewed SHA, and apply its existing promotion criteria. This scoped agent made no operational WORKLOG changes to avoid editing root-owned files.
