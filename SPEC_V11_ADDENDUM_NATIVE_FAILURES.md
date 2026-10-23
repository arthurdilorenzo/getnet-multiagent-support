# SPEC V11 — Addendum: falhas observadas na amostra nativa

## Baseline and evidence

Baseline: candidate `037f5b6723576d4ee7aa8509a4c01a5d61c57eaa`.
Evidence: `pipeline/getnet_v11_native_candidate_037f5b6_20261023/native-chat-results.json`, eight synthetic cases, one run, no retries. Independent V4 adjudication: 91.00/100; Semantic Quality Gate not satisfied.

## Contracts to repair

1. **Local facts survive an independent external failure.** For Crediário questions, retrieve supported Brazil facts from the local corpus and retain them if a separate Norway web request fails or exhausts its budget. Mark Norway unresolved; never infer its terms from Brazil. The tool outage is not grounds to erase locally grounded facts.
2. **Source instructions are constraints, not standalone information requests.** Phrases such as “Use Getnet's own guidance, please” and `Consulte fontes atuais e deixe claro se não houver confirmação` must remain attached to the nearest subject they qualify. A Web retry must retain the original public topic, never search only the qualifier text.
3. **Context must not replace the requested task.** In “I found an Algar SIM in the reader box. Walk me through safe placement and how to confirm communication,” retrieval/delegation must include safe placement and communication testing; a context-only subquery is insufficient to fulfill the request.
4. **Operational answers honor requested-field minimization.** For a multi-terminal request naming only connection type for one terminal and signal strength for another, return only those exact target-field pairs. Do not reveal firmware/status or add troubleshooting steps unless requested.
5. **Ledger states reflect user needs, not courtesy/source constraints.** If the factual request is answered from permitted evidence and its source preference was honored, do not leave a qualifier-only unit unresolved or downgrade the response to partial solely for that qualifier.

## Acceptance tests

- C03 Pix: source directive remains attached to the factual question; Getnet categories and citation fulfill it without a spurious unresolved unit.
- C04 partial: local corpus query returns the documented Brazil timing/fee/anticipation facts; simulate Web unavailable and verify those facts remain in the answer while Norway is explicitly unresolved.
- C07 English contextual question: the actual retrieval query contains the installation and communication-test task; evidence-backed manual instructions are returned or the answer abstains for a genuine evidence gap, never because only the context sentence was delegated.
- C08 public outage: after a failed external call, the next query retains the outage subject and cannot accept unrelated generic snippets as evidence; explicit uncertainty remains honest.
- C05 multi-target minimization: output and tool-visible final data contain only requested fields paired with exact terminal IDs; firmware, unrelated status, signal for the other target, and unrequested instructions are absent.
- Keep C01/C02 condition gates and C06 tenant-abstention regressions green. Run focused and full offline suite; independent code audit required. A second native run is outside this spec and requires separate explicit authorization.
