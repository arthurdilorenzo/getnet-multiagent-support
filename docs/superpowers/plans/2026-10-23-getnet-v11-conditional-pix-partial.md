# Getnet V11 Conditional Pix Partial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the observed conditional-query, Getnet Pix routing, and Brazil/Norway partial-answer failures without weakening tenant, evidence, or tool-scope rules.

**Architecture:** Add a deterministic server-owned conditional-dependency contract consumed by the orchestrator; keep model tool choice native and use only verified local Getnet evidence to correct a Pix route; preserve approved Brazilian spans and unresolved Norway coverage without pointless replanning.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, Google GenAI Function Calling, local hybrid RAG, pytest.

**Spec:** `SPEC_V11.md`

## Global Constraints

- Base every change on candidate SHA `8c283c8455d02327f39ddb15e1d23e9f81c0a1d4`.
- Never publish to `main` unless independent V4 score is at least 95 and exact-SHA CI is green.
- Never assign factual coverage without typed terminal evidence or authorized verified source spans.
- Keep Gemini model, retries, timeout, token budget, tenancy rules, and action approval policy unchanged.
- All tests and the local evaluator run with `GEMINI_API_KEY=''`; native evaluation happens only after code/tests/CI and a new frozen sample.

## Review Focus

- Initial multi-step plan schedules dependent Knowledge before Support: test that orchestration defers it until the predicate observation exists.
- Firmware/status is null, wrong, ambiguous, or tool call fails: test that every dependent Knowledge call remains blocked.
- Pix asks product eligibility vs. external regulation: test local verified route only for product facts, preserving explicit external intent.
- Pix plus unavailable firmware: test exact target, missing value, partial status, and no unrequested fields from the terminal snapshot.
- Brazil plus Norway with local token budget exhausted for replanning: test the Brazilian evidence survives and no unsupported Norwegian promise appears.

---

### Task 1: Conditional dependency contract (Agent A)

**Files:**
- Create: `app/security/conditional_dependencies.py`
- Test: `tests/test_conditional_dependencies_v11.py`

**Interfaces:**
- Produces an immutable parsed condition representation and a pure predicate evaluator for supported terminal firmware/status predicates; unsupported or ambiguous grammar returns no authorization.
- Does not edit `app/agents/orchestrator.py`; root integrates the helper in Task 3.

- [ ] **Step 1: Write failing helper tests** for exact firmware+online true, firmware mismatch, missing/null value, wrong terminal ID, failed/no observation, and unsupported language.
- [ ] **Step 2: Run targeted tests and confirm expected failures** with the failure localized to the absent helper/API.
- [ ] **Step 3: Implement the smallest pure parser/evaluator** using typed successful `diagnose_terminal_issue` results only.
- [ ] **Step 4: Run targeted tests** and verify all predicate cases pass.
- [ ] **Step 5: Review spec compliance and report the patch without touching orchestrator.**

### Task 2: Pix routing and documentary evidence (Agent B)

**Files:**
- Modify: `app/agents/router_agent.py`
- Modify: `app/rag/embeddings.py` only if diagnosis shows query normalization/lexical anchoring is the cause.
- Test: `tests/test_pix_getnet_routing_v11.py`

**Interfaces:**
- Consumes the existing `RouterAgent._prefer_local_documentary_evidence` guard and retriever provenance contract.
- Produces a bounded decision: Getnet product eligibility questions with a verified, sufficiently relevant corpus span become Knowledge assignments; explicit current/external regulatory searches remain Web.
- Does not edit orchestrator or shared product fixtures.

- [ ] **Step 1: Write failing router tests** for model-proposed Web on account categories, explicit external rule lookup, and mixed Pix+private firmware units.
- [ ] **Step 2: Run targeted tests and confirm the current incorrect routing/retrieval behavior.**
- [ ] **Step 3: Implement the narrow Getnet Pix relevance/route correction** without broad keyword preemption or bypassing citation/authorization.
- [ ] **Step 4: Run targeted tests** and ensure unrelated weather/current external search still routes to Web.
- [ ] **Step 5: Self-review only the assigned files and report changed paths.**

### Task 3: Orchestrator integration and partial preservation (Root)

**Files:**
- Modify: `app/agents/orchestrator.py`
- Modify: `app/agents/knowledge_agent.py` to force verified Getnet product-eligibility questions through local search and, as needed, retain independent approved Brazil spans.
- Modify: `app/security/request_units.py` only to split the observed Brazil-vs-Norway Crediário comparison into country-scoped units so the coverage ledger can represent partial completion honestly.
- Test: `tests/test_conditional_orchestration_v11.py`
- Test: `tests/test_pix_crediario_partial_v11.py`

**Interfaces:**
- Consumes Task 1 condition API and Task 2 router behavior.
- Produces serialized, post-observation gating of dependent Knowledge and no dependent lookup after false/unknown predicate.
- Preserves `fulfilled/unresolved` country units, citation, and no-replan behavior when only the independent country remains unsupported.
- Enforces local Getnet product evidence at the Knowledge specialist's tool boundary; explicit external/regulatory requests remain Web.
- Coalesces adjacent equivalent Knowledge assignments for a single evidence pass when they are independent clauses; does not reorder steps or merge across a dependency.

- [ ] **Step 1: Write failing integration tests** for M08 support-first ordering/blocking and P06 Brazil-fulfilled/Norway-unresolved with no redundant replan.
- [ ] **Step 2: Run tests and verify they fail** for the audited regressions.
- [ ] **Step 3: Integrate the condition helper and partial evidence behavior** in the server-owned execution path; preserve original scope and typed evidence.
- [ ] **Step 4: Run targeted integration tests and the existing V10 tests.**
- [ ] **Step 5: Run full pytest and the offline evaluator; record exact exit codes and counts.**
- [ ] **Step 6: Perform final independent review and a new frozen native audit; do not push main unless both audit threshold and CI gate are met.**

## Preflight Conflict Scan

| Pair/task | Shared file/interface | Scan result |
|---|---|---|
| Task 1 ↔ Task 2 | None; separate new policy module vs router/retriever | No file overlap; both consume frozen V11 spec. |
| Task 1 ↔ Task 3 | Task 3 imports Task 1 helper; Task 1 does not edit orchestrator | One-way interface; root waits for helper contract before integration. |
| Task 2 ↔ Task 3 | Task 3 consumes RouterAgent plan; Task 2 edits router/embeddings, Task 3 edits orchestrator/knowledge/tests | No overlapping source file; integration requires Task 2 source citation/routing behavior. |
| Task 1 self-check | Tests/assertions align with pure parser/evaluator; production scope excludes orchestrator by design | Consistent. |
| Task 2 self-check | Tests assert real plan/retriever behavior; external regulation and mixed private scope retained | Consistent. |
| Task 3 self-check | Tests assert serialized call order, grounded spans, partial ledger, and no redundant replan | Consistent. |

**Rulings:**
- `Ruling: keep conditional execution gating server-side and fail closed for unsupported predicates — the observed M08 trace proves prompt-only guidance did not block Knowledge after `qa-12.07` failed `qa-12.08` — cost if wrong: a valid dependent answer may remain pending until the finite grammar is expanded.`
- `Ruling: use a narrow Getnet-product Pix route correction backed by verified corpus evidence, not a general Pix keyword override — Brazilian Pix regulation can require external primary sources — cost if wrong: an external Pix question may retrieve stale/local product material.`
- `Ruling: preserve the supported Brazil unit and mark Norway unsupported/unresolved rather than declaring unsupported equivalence — the audited corpus supports the Brazil terms but has no Norway evidence — cost if wrong: the response can be partial where new evidence later establishes a full comparison.`
