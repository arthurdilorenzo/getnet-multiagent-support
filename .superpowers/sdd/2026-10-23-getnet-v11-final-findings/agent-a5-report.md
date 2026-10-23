# A5 — M08 diagnostic payload integrity

Date: 2026-10-23. Worktree: `/private/tmp/getnet-v11-worktree`.
Baseline reviewed: `ac01841`; integrated B5 HEAD during verification: `11487c2`.

## Scope and result

Read `SPEC_V11_ADDENDUM_M08_READ_INTEGRITY.md` and `agent-a5.md` before implementation. Initial ownership was the condition evaluator and its test file; the parent explicitly expanded ownership to `app/tools/customer_tools.py` and tool tests after the missing producer identity was identified. No B5 file was edited.

The evaluator previously skipped null/empty/incomplete diagnostic data and other-terminal reads, allowing a separate positive read to authorize the condition. It also supplied a missing `requested_terminal_id` from the expected condition. Every typed diagnostic now fails closed unless technically successful/completed/error-free, with `found is True`, explicit exact requested and observed terminal IDs, a typed version string matching the existing finite version grammar, typed status, and exact firmware/status predicate matches. Both read orders are tested. Multiple complete exact reads still authorize. Existing parser grammar and exact comparison semantics remain intact.

The real diagnostic producer now echoes the canonical selected terminal ID when a terminal ID was supplied. Registry tests use a synthetic local customer fixture and exercise canonical, lowercase and surrounding-space input, then verify the real tool result authorizes the exact parsed predicate. Missing/failing domain results cannot donate identity.

Invalid payload matrix: null/empty payload, failed null read, other-target payload, missing/null/list/empty terminal, false/numeric/string/missing found, missing/mismatched/list requested ID, missing/mismatched/list terminal ID, missing/list/empty firmware, and missing/bool/empty status. Pure evaluation tests cover invalid alone and both mixed orders; real orchestration tests confirm support-only delegation, false gate, no Knowledge/Web, abstention and unresolved coverage. Existing technical-failure/outcome/error-code tests are retained.

## TDD evidence

All commands ran from the worktree. Runtime supplied by the parent:
`/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python`.
No `.env` was read or modified; `GEMINI_API_KEY` was explicitly empty. No API/network calls, push, or subagents.

Initial RED command (consumer-only tests, before production changes):

```sh
GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python -m pytest -q tests/test_conditional_dependencies_v11.py > /private/tmp/getnet-v11-a5-red.txt 2>&1
```

Exit 1; exact summary: `12 failed, 106 passed in 0.87s`. Positive followed by null/empty/incomplete/missing-requested-ID incorrectly returned True, and orchestration incorrectly delegated Knowledge.

Expanded RED after producer tests and other-target/read-failure cases, still before production changes:

```sh
GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python -m pytest -q tests/test_conditional_dependencies_v11.py tests/test_unit_tools_and_security.py -k 'diagnostic or exact' > /private/tmp/getnet-v11-a5-red-expanded.txt 2>&1
```

Exit 1; exact summary: `18 failed, 80 passed, 38 deselected in 1.24s`.
Five consumer payload cases failed pure checks and both orchestration orders (15 failures); three real producer cases failed with `KeyError: 'requested_terminal_id'`.

After the evaluator and producer implementation, identical focused command with output redirected to `/private/tmp/getnet-v11-a5-green-focused.txt`:

```text
........................................................................ [ 73%]
..........................                                               [100%]
98 passed, 38 deselected in 0.75s
```

Exit 0.

## Regression and full verification

```sh
GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python -m pytest -q tests/test_conditional_dependencies_v11.py tests/test_unit_tools_and_security.py tests/test_conditional_orchestration_v11.py > /private/tmp/getnet-v11-a5-regressions.txt 2>&1
```

Exit 0; exact output summary: `380 passed in 5.30s`.

```sh
GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python -m pytest -q > /private/tmp/getnet-v11-a5-full.txt 2>&1
```

Exit 0; exact summary: `1419 passed, 1 warning in 11.51s`.
The sole warning is upstream `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.` No failed tests.

```sh
GEMINI_API_KEY='' /Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python -m app.evals.evaluator > /private/tmp/getnet-v11-a5-evaluator.txt 2>&1
/Users/arthurdilorenzo/Documents/Vagas-Antigravity/pipeline/getnet_challenge/.venv/bin/python scripts/check_no_print.py app
git diff --check
```

All exit 0. Static commands have no output. Evaluator reports routing/recall/precision/safe selection/limited groundedness = 1.0, overall pass = 0.9231; mode `offline_deterministic`, external services `synthetic_external_api_fixtures`. This is limited synthetic/offline evidence, not a Gemini quality score or promotion justification.

## Handoff

All requested local A5 contracts are implemented and verified. Independent audit, final exact-SHA checks/native evaluation, any CI and any promotion remain parent-owned. No gate, label, budget, API model, retry policy, frontend or B5 implementation was altered.
