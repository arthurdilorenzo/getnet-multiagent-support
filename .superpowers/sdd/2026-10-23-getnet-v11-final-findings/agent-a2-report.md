# Frente A2 — M08 connective and diagnostic variants

Date: 2026-10-23. Baseline: `0745d6ccbc054d3be4a4f2f854b314607936a776`.

## Scope and evidence read

Read `SPEC_V11_ADDENDUM_M08_VARIANTS.md`, `agent-a2.md`, and prior implementation report `agent-a-report.md` before changes. Changed only `app/agents/orchestrator.py`, M08 tests in `tests/test_conditional_orchestration_v11.py`, and this report. No KnowledgeAgent/P05 implementation changes, Gemini/API calls, push, promotion, or subagents.

## Root causes and final behavior

1. Postfix linking accepted only `Only`/`Somente` (and existing bare qualifier variants) before `if`/`se`. With `But only if` / `However, only if`, only the predicate sentence was blocked; the earlier benefits unit escaped to the real retrieval registry. The same defect occurred for `Mas somente se` / `No entanto, somente se`.
2. The complete bounded diagnostic-question exemption allowed only `[the] terminal` in English and no Portuguese status question. Legitimate `Check if my terminal … is Online.` and `Verifique se o terminal … está Online.` therefore became unsupported disclosure conditions and lost their Support assignments.

The detector now admits a finite optional contrast prefix (`But`, `However`, `Mas`, `No entanto`, with an optional comma) when linking the immediately preceding disclosure span. The same finite prefix is admitted in postfix canonicalization. Only the existing literal benefits/conjunction grammar is reordered; the predicate itself is passed unchanged to `parse_terminal_condition`. Exact target, firmware, status, completed typed observation, and all other authorization checks remain unchanged.

The diagnostic exemption now additionally recognizes `my terminal` and `Verifique/Confirme se o terminal … está/esta Online/Offline`. It still requires the entire bounded sentence. A same-sentence documentary consequence cannot borrow the exemption. A separate diagnostic sentence remains executable while a later unsupported conditional disclosure stays blocked.

## TDD and registry boundary

Added 52 collected cases:

- 32 cases: four EN/PT contrast postfixes × firmware mismatch/missing firmware/Offline/diagnostic failure × Knowledge/Web route. Only `diagnose_terminal_issue` executes; no dependent retrieval.
- Four cases: each contrast postfix with an exact positive typed read. The real registry receives diagnosis first, then a benefits query; the gate trace confirms authorization.
- Four cases: each contrast postfix with an additional unrecognized predicate. No registry read executes.
- Two direct status variants: real CustomerSupportAgent/registry execution, exact terminal argument, and successful diagnostic trace; no Knowledge/Web call.
- Eight mixed cases: each EN/PT direct diagnostic followed by each contrast postfix with an unsupported extra condition. Support succeeds; dependent disclosure remains unresolved and no retrieval executes.
- Two inseparable diagnostic-plus-disclosure cases: no registry execution.

Tests keep orchestrator, specialist execution, ToolRegistry validation and dispatch real. The existing recording wrapper forwards to the real `ToolRegistry.execute`; only handlers are synthetic. Typed diagnostics come from the real registry handler result. Retrieval handlers deliberately raise after dispatch, preventing network use. Direct/mixed checks additionally assert the successful real diagnostic trace so a rejected call attempt cannot satisfy the test.

All test commands below used this prefix:

`GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/pytest`

RED, before production changes:

`-q tests/test_conditional_orchestration_v11.py -k 'contrast_postfix or direct_status_question_variants'`

Result: **42 failed, 47 deselected in 1.41s**. The contrast cases actually dispatched `search_getnet_knowledge`/`web_search_general` instead of the diagnostic; both direct questions performed zero Support reads. Log: `/private/tmp/getnet-v11-agent-a2-red.txt`.

GREEN, after the minimal finite-regex change:

Same command: **42 passed, 47 deselected in 0.97s**. Log: `/private/tmp/getnet-v11-agent-a2-green.txt`.

The ten subsequent scope characterization cases required no further production change. Their initial mixed-scope fixture assigned only the first literal unit (`Check ` / `Verifique `) to Support, accidentally leaving the `if`/`se` fragment to Knowledge. Corrected the fixture to assign every unit within the complete diagnostic sentence to Support; this was a test setup defect, not a further implementation fix. Full M08 run after that correction: **99 passed in 1.64s**, log `/private/tmp/getnet-v11-agent-a2-m08.txt`.

## Final verification

After strengthening successful-dispatch trace assertions, reran:

`-q tests/test_conditional_orchestration_v11.py tests/test_conditional_dependencies_v11.py tests/test_pix_crediario_partial_v11.py tests/test_pix_getnet_routing_v11.py tests/test_replan_scope_v8.py tests/test_action_integrity.py tests/test_orchestrator_contracts.py`

Result: **193 passed in 1.89s**. Log: `/private/tmp/getnet-v11-agent-a2-focused.txt`.

`-v`

Result: **1179 passed, 1 warning in 7.25s**. Warning is the pre-existing Starlette/httpx deprecation. Log: `/private/tmp/getnet-v11-agent-a2-full.txt`.

`/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python scripts/check_no_print.py app`: exit 0.

`git diff --check`: exit 0.

## Limits and handoff

Language recognition remains deliberately finite. This is deterministic offline registry evidence, not native Gemini quality or a V4 score. Unknown predicates still fail closed. Root must perform the independent review/new freeze before another native test, and add the operational WORKLOG summary linking this report. The addendum remains root-owned and was not included in this agent commit.
