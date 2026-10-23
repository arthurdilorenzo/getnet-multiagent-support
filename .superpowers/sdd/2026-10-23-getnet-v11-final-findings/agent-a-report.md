# Frente A — M08 final-review remediation

Date: 2026-10-23. Baseline: `dc5ce0a20d1a7347a8dbeff4b3391d1f259e4639`.

## Scope and contract

Read `SPEC_V11_ADDENDUM_FINAL_REVIEW.md`, `agent-a.md`, and the workspace final-review report before implementation. Changed only `app/agents/orchestrator.py`, `tests/test_conditional_orchestration_v11.py`, and this report. Knowledge/P05 changes belong to the parallel Frente B and were not edited by Frente A. No Gemini/API calls, subagents, push, main changes, or promotion.

The server-owned dependency detector now associates a postfix `Only if` / `Somente se` sentence with its preceding content span. A semicolon before the postfix likewise associates the preceding disclosure clause, while earlier clearly independent request clauses remain assignable. Bare qualification/context fragments such as `Unless certified` and `For certified devices` remain attached instead of being silently discarded as independent requests.

Authorization is still separate from detection. Only a literal English/Portuguese benefits request followed by the exact supported postfix is reordered into the existing finite parser grammar. The predicate is not rewritten or relaxed; unknown/range/additional predicates or unsupported consequence syntax fail closed. The existing exact-target typed diagnostic evaluation still controls access. Firmware mismatch, absent firmware, Offline, and tool failure block both local and Web dependent retrieval.

A complete bounded diagnostic question (`Check/Verify/Confirm if [the] terminal GET-… is Online/Offline`) bypasses disclosure-condition detection and reaches CustomerSupportAgent. A question followed by a documentary consequence in the same sentence does not receive that exemption.

## Root causes

1. The old detector used only the sentence containing the conditional marker. It blocked `Only if firmware …` but left the earlier `Tell me the eligible benefits.` assignment runnable.
2. The marker scan treated interrogative `if` as a conditional disclosure, deleting the legitimate Support assignment for `Check if terminal … is Online.`

## Regression evidence and TDD

New tests exercise the real orchestrator, Support/Knowledge dispatch, and ToolRegistry validation/dispatch. A recording wrapper forwards to the real registry; only handlers are synthetic. Diagnostics return controlled typed data; retrieval handlers terminate with a deliberate synthetic exception after dispatch, so tests cannot call external services. Assertions check registry tool names, target arguments, query text, order, and ledger state, not just the in-memory plan.

Added 24 collected cases:

- Eight local/Web route × false/unknown/error observation cases: only the diagnostic read executes, with zero dependent retrieval calls.
- Two unsupported postfix grammar cases: no registry reads execute.
- A direct status-question case: the exact terminal diagnostic executes.
- Eight independent-prefix cases: local/Web, period/semicolon separators, imperative/question/polite request forms preserve only the independent canonical query.
- Two exact positive postfix cases (EN/PT): diagnostic precedes retrieval and the product request survives in the actual query.
- Two unknown-prefix qualifier cases: unsupported qualification cannot be dropped to authorize the postfix.
- A direct-question-plus-consequence negative case: the diagnostic exemption cannot authorize dependent disclosure.

All pytest commands used this interpreter/prefix:

`GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/pytest`

RED runs, before the corresponding production changes:

1. `-q tests/test_conditional_orchestration_v11.py -k 'sentence_postfix or direct_check_if'`: **11 failed**, as expected. Earlier benefits retrieval escaped; direct status read was absent. Log: `/private/tmp/getnet-v11-agent-a-red.txt`.
2. `-q tests/test_conditional_orchestration_v11.py -k 'postfix_keeps or postfix_retrieval'`: **5 failed**, as expected. Independent queries still carried the benefits span; the positive postfix did not execute diagnostic first. Log: `/private/tmp/getnet-v11-agent-a-independent-red.txt`.
3. `-q tests/test_conditional_orchestration_v11.py -k 'unknown_qualifier or direct_status_question or positive_registry'`: **2 failed, 3 passed**. Unknown preceding qualifiers were incorrectly discarded; EN/PT positive and consequence-exemption checks were characterization cases. Log: `/private/tmp/getnet-v11-agent-a-qualifier-red.txt`.
4. `-q tests/test_conditional_orchestration_v11.py -k postfix_keeps`: **4 failed, 4 passed**. Natural questions and polite independent requests were conservatively overblocked. Log: `/private/tmp/getnet-v11-agent-a-question-red.txt`.

During parallel integration, Frente B's first canonical binder temporarily dropped the benefits clause on the positive EN/PT path, sending only the condition to retrieval. The new M08 boundary tests caught both failures. Frente B corrected Knowledge binding; Frente A did not edit that file. An early Web-prefix fixture also accidentally supplied a Knowledge rather than Web plan; its explicit plan was corrected, without changing production routing.

## Final verification

- `-q tests/test_conditional_orchestration_v11.py tests/test_conditional_dependencies_v11.py tests/test_orchestrator_contracts.py tests/test_replan_scope_v8.py tests/test_action_integrity.py`: **100 passed**. Log: `/private/tmp/getnet-v11-agent-a-focused.txt`.
- `-v`: **1122 passed, 1 warning**, 6.65 s on the shared integration working tree. Warning: existing upstream Starlette/httpx deprecation. Log: `/private/tmp/getnet-v11-agent-a-full.txt`.
- `/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python scripts/check_no_print.py app`: exit 0.
- `git diff --check`: exit 0.

## Limits and handoff

These are deterministic/offline contracts, not evidence of native Gemini quality or a V4 score. Exact postfix authorization remains deliberately finite; other syntax is blocked instead of inferred. The diagnostic exemption is likewise bounded to the complete status-question form above. No Knowledge source selection, corpus, budgets, provider behavior, labels, or policy gates were modified by Frente A.

Root must finish integration review, rerun final checks on the frozen SHA, update the workspace operational WORKLOG with this report, and obtain the required independent review/audit before considering any promotion. Shared full-suite success here includes the then-current Frente B working-tree changes; it is not an exact-SHA CI claim.
