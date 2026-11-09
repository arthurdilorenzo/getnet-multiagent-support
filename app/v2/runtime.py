"""One bounded cycle for new requests and semantic continuation after HITL."""
import hashlib
import json
import uuid

from app.agents.guardrails import GuardrailManager
from app.config import settings
from app.observability.tracer import SystemTracer
from app.schemas import ChatResponse
from app.security.auth import SecurityContext
from app.security.pii import PIIRedactor
from app.v2.contracts import (
    AnswerDocument,
    Binding,
    EvidenceSpan,
    ExecutionBudget,
    Fact,
    ObservationSummary,
    PlanProposal,
    ReplanProposal,
    ResumePlan,
    Requirement,
    StepProposal,
    answer_status,
    canonicalize_plan,
)
from app.v2.knowledge import EvidenceError, human_span, tool_span


def _step_signature(step: StepProposal, user_id: str) -> str:
    canonical = {
        "user": user_id,
        "kind": step.kind,
        "tool_name": step.tool_name or "",
        "arguments": step.arguments or {},
        "query": " ".join((step.query or "").strip().lower().split()),
        "requirement_indexes": sorted(step.requirement_indexes or []),
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _validate_steps_against_requirements(steps: list[StepProposal], planned_requirements: list[Requirement]) -> None:
    for step in steps:
        indices = step.requirement_indexes
        if not indices:
            raise ValueError("Etapa sem referência de requisito.")
        if len(indices) != len(set(indices)):
            raise ValueError("Etapa possui índices de requisitos duplicados.")
        if any(index < 0 or index >= len(planned_requirements) for index in indices):
            raise ValueError("Etapa referencia índice de requisito inválido.")


class Runtime:
    def __init__(self, model, knowledge, state, gateway, queue, budget: ExecutionBudget | None = None):
        self.model = model
        self.knowledge = knowledge
        self.state = state
        self.gateway = gateway
        self.queue = queue
        self.budget = budget or ExecutionBudget(max_steps=settings.MAX_ORCHESTRATION_ITERATIONS)

    def process(self, message: str, security_context: SecurityContext, session_id: str | None = None) -> ChatResponse:
        if "chat:write" not in security_context.scopes and "admin:all" not in security_context.scopes:
            raise PermissionError("Escopo de chat ausente.")
        session_id = session_id or "v2_" + uuid.uuid4().hex
        user = security_context.authenticated_user_id
        self.state.store.ensure_session(session_id, user)
        self.state.append(session_id, user, "user", message)
        tracer = SystemTracer(session_id=session_id, user_id=user)
        safe, reason, _ = GuardrailManager.evaluate_input(message)
        if not safe:
            response = self._empty(session_id, user, reason, tracer)
        elif self.state.store.has_pending_human(session_id, user):
            response = self._empty(session_id, user, "Atendimento aguardando resposta humana.", tracer)
            response.escalated = True
        else:
            response = self._drive(PIIRedactor.mask_text(message), security_context, session_id, tracer)
        tracer.finish(
            status="completed" if not response.abstained else "abstained",
            answer_status=response.answer_status,
            metadata={"escalated": response.escalated, "tools_called": response.tools_called},
        )
        self.state.append(session_id, user, "assistant", response.response, response.tools_called, response.sources)
        return response

    def _drive(
        self,
        original: str,
        security_context: SecurityContext,
        session_id: str,
        tracer: SystemTracer,
        human_guidance: str = "",
        prior: dict | None = None,
        prior_ticket=None,
    ) -> ChatResponse:
        user = security_context.authenticated_user_id
        context = self.state.context(session_id, user)
        spans: list[EvidenceSpan] = [EvidenceSpan.model_validate(s) for s in (prior or {}).get("evidence", [])]
        facts: list[Fact] = [Fact.model_validate(f) for f in (prior or {}).get("facts", [])]
        observations: list[dict] = list((prior or {}).get("observations", []))
        action_proposals: list[dict] = []
        tools: list[str] = []
        modes: list[str] = []

        if human_guidance and prior_ticket:
            op_id = getattr(prior_ticket, "assigned_operator", None) or "operator"
            h_span = human_span(prior_ticket.ticket_id, op_id, human_guidance)
            spans.append(h_span)

        try:
            # Resumption uses prior requirements without re-proposing
            if prior and prior.get("requirements"):
                planned_requirements = [Requirement.model_validate(r) for r in prior["requirements"]]
                plan_payload = {
                    "original_request": original,
                    "context": context,
                    "human_guidance": human_guidance,
                    "observations": observations,
                    "available_tools": self.gateway.declarations(),
                    "tool_descriptors": self.gateway.get_tool_descriptors(user_id=user),
                    "planned_requirements": [r.model_dump() for r in planned_requirements],
                    "satisfied_human_requirement_ids": list((prior or {}).get("waiting_requirement_ids", [])) if human_guidance else [],
                    "human_observation_id": h_span.span_id if human_guidance and prior_ticket else None,
                }
                resume_plan = self.model.generate(ResumePlan, plan_payload, phase="routing", tracer=tracer)
                steps = resume_plan.steps
                entities = context["v2"].get("entities", {})
            else:
                plan_payload = {
                    "original_request": original,
                    "context": context,
                    "human_guidance": human_guidance,
                    "observations": observations,
                    "available_tools": self.gateway.declarations(),
                    "tool_descriptors": self.gateway.get_tool_descriptors(user_id=user),
                }
                plan_proposal = self.model.generate(PlanProposal, plan_payload, phase="routing", tracer=tracer)
                planned_requirements, steps = canonicalize_plan(plan_proposal)
                entities = {**context["v2"].get("entities", {}), **plan_proposal.entities.model_dump(exclude_none=True)}

            max_steps = min(self.budget.max_steps, settings.MAX_ORCHESTRATION_ITERATIONS)
            if len(steps) > max_steps:
                raise ValueError("Plano excede orçamento de etapas.")

            # The resumed steps still refer to the original frozen requirements.
            # Validate before dispatch, just as canonicalize_plan does initially.
            _validate_steps_against_requirements(steps, planned_requirements)

            all_bindings = [b for r in planned_requirements for b in r.bindings]

            if not human_guidance:
                has_human_req = any(r.kind == "human" for r in planned_requirements)
                has_human_step = any(s.kind == "human" for s in steps)
                has_ticket_tool = any(s.kind == "tool" and s.tool_name == "check_escalation_ticket" for s in steps)
                if has_human_req and not has_human_step and not has_ticket_tool:
                    human_req_idx = next(i for i, r in enumerate(planned_requirements) if r.kind == "human")
                    steps.append(StepProposal(kind="human", reason="Validação por equipe humana solicitada", requirement_indexes=[human_req_idx]))

            satisfied_human_ids = set((prior or {}).get("waiting_requirement_ids", [])) if human_guidance else set()
            # Older frozen state predates explicit request IDs. Its resolved ticket
            # still satisfies the original human requirement, never a new grant.
            if human_guidance and prior_ticket and not satisfied_human_ids:
                satisfied_human_ids = {r.requirement_id for r in planned_requirements if r.kind == "human"}

            step_queue: list[StepProposal] = list(steps)
            executed_signatures: set[str] = set()
            seen_retrieval_fingerprints: set[str] = set()
            observation_summaries: list[ObservationSummary] = []
            executed_steps = 0
            replans_used = 0

            while step_queue:
                # Enforce hop and cost budgets before dispatching next step
                if executed_steps >= max_steps:
                    tracer.add_step(
                        agent="V2_SUPERVISOR",
                        action="budget_gate_stopped",
                        details={"reason": "hop_budget_exhausted", "executed_steps": executed_steps, "max_steps": max_steps},
                    )
                    break
                if self._budget_exhausted(tracer):
                    break

                step = step_queue.pop(0)

                if step.kind != "human":
                    sig = _step_signature(step, user)
                    if sig in executed_signatures:
                        tracer.add_step(
                            agent="V2_SUPERVISOR",
                            action="loop_detected",
                            details={"signature": sig, "kind": step.kind, "tool_name": step.tool_name, "query": step.query},
                        )
                        continue
                    executed_signatures.add(sig)

                executed_steps += 1

                if step.kind == "human":
                    remaining = [i for i in step.requirement_indexes
                                 if planned_requirements[i].requirement_id not in satisfied_human_ids]
                    if not remaining:
                        continue
                    ticket = self.queue.create_ticket(
                        user_id=user,
                        session_id=session_id,
                        category="v2_support",
                        reason=step.reason or "Verificação humana necessária",
                        diagnostic_snapshot={"entities": entities},
                        idempotency_key="v2:" + tracer.trace_id,
                    )
                    merchant_scopes = [s for s in security_context.scopes if s in {"chat:write", "customer:read_own", "terminal:action_own"}]
                    self.state.save(
                        session_id,
                        user,
                        {
                            "original_request": original,
                            "original_scopes": merchant_scopes,
                            "requirements": [r.model_dump() for r in planned_requirements],
                            "entities": entities,
                            "evidence": [s.model_dump() for s in spans],
                            "facts": [f.model_dump() for f in facts],
                            "observations": observations,
                            "waiting_ticket": ticket.ticket_id,
                            "waiting_requirement_ids": [planned_requirements[i].requirement_id for i in remaining],
                        },
                    )
                    response = self._empty(
                        session_id,
                        user,
                        "Atendimento encaminhado para validação humana. Protocolo: " + ticket.ticket_id,
                        tracer,
                    )
                    response.ticket_id, response.escalated = ticket.ticket_id, True
                    response.tools_called = tools + ["create_escalation_ticket"]
                    return response

                if step.kind == "knowledge":
                    retrieved, mode = self.knowledge.search(step.query or original)
                    spans.extend(retrieved)
                    modes.append(mode)
                    tools.append("search_getnet_knowledge")
                    fp = hashlib.sha256(
                        "|".join(sorted(s.span_id for s in retrieved)).encode("utf-8")
                    ).hexdigest()[:16]
                    if fp in seen_retrieval_fingerprints:
                        tracer.add_step(
                            agent="V2_SUPERVISOR",
                            action="loop_detected",
                            details={
                                "reason": "duplicate_retrieval_evidence",
                                "evidence_fingerprint": fp,
                                "query": step.query or original,
                            },
                        )
                    seen_retrieval_fingerprints.add(fp)
                    conf = 0.9 if len(retrieved) >= 2 else (0.65 if len(retrieved) == 1 else 0.2)
                    obs_summary = ObservationSummary(
                        observation_id=f"obs_{executed_steps:04d}",
                        step_kind="knowledge",
                        source_name="search_getnet_knowledge",
                        outcome_state="completed" if retrieved else "empty",
                        evidence_fingerprint=fp,
                        confidence_score=conf,
                    )
                    observation_summaries.append(obs_summary)
                    tracer.add_step(
                        agent="V2_SUPERVISOR",
                        action="observe",
                        details={
                            "observation_id": obs_summary.observation_id,
                            "step_kind": "knowledge",
                            "spans_retrieved": len(retrieved),
                            "retrieval_mode": mode,
                            "confidence_score": conf,
                            "confidence_gate_passed": conf >= self.budget.confidence_threshold,
                        },
                    )
                    if conf < self.budget.confidence_threshold and replans_used < self.budget.max_replans and executed_steps < max_steps:
                        if not self._budget_exhausted(tracer):
                            replanned, consumed = self._maybe_replan(
                                original=original,
                                context=context,
                                planned_requirements=planned_requirements,
                                step_queue=step_queue,
                                observation_summary=obs_summary,
                                all_observations=observations,
                                executed_signatures=executed_signatures,
                                user=user,
                                tracer=tracer,
                            )
                            if consumed:
                                replans_used += 1
                            if replanned is not None:
                                step_queue = replanned

                else:
                    result = self.gateway.execute(step.tool_name, step.arguments, security_context, session_id, tracer.trace_id)
                    tools.append(step.tool_name)
                    observations.append(result.model_dump())
                    tracer.add_step(
                        agent="V2_TOOL",
                        action="call:" + step.tool_name,
                        status="success" if result.success else "error",
                        details={"success": result.success, "error_code": result.error_code},
                    )
                    if not result.success or result.outcome_state != "completed":
                        response = self._empty(
                            session_id,
                            user,
                            "Operação não concluída: " + (result.error_code or result.outcome_state),
                            tracer,
                        )
                        response.tools_called = tools
                        return response

                    # Resolve only explicit symbolic aliases through authenticated
                    # server data. Concrete target mismatches stay mismatches.
                    for binding in all_bindings:
                        if binding.tool_name == step.tool_name or (
                            step.tool_name == "get_currency_exchange_rate"
                            and binding.tool_name in ("get_currency_exchange_rate", "get_exchange_rate")
                        ):
                            binding.tool_name = step.tool_name
                            binding.resource_id = self.gateway.resolve_resource_id(
                                binding.resource_type, binding.resource_id, user,
                                arguments=step.arguments, result=result,
                            )
                    # Extract verified facts anchored in tool results
                    extracted_facts = self.gateway.extract_facts(step.tool_name, result, all_bindings)
                    facts.extend(extracted_facts)

                    if step.tool_name == "web_search_general":
                        search_data = result.data or {}
                        for item in search_data.get("results", []):
                            snippet = item.get("snippet", "")
                            if snippet:
                                s_clean, _ = GuardrailManager.sanitize_untrusted_content(snippet)
                                s_clean = PIIRedactor.mask_text(s_clean)
                                s_hash = hashlib.sha256(s_clean.encode("utf-8")).hexdigest()
                                spans.append(EvidenceSpan(
                                    span_id="web:" + s_hash,
                                    text=s_clean,
                                    source=item.get("title", search_data.get("source", "web_search")),
                                    source_url=item.get("url", ""),
                                    kind="document",
                                    content_hash=s_hash,
                                ))

                    if step.tool_name == "execute_terminal_remote_action":
                        prop = result.data or {}
                        if isinstance(prop, dict) and prop.get("proposal_id"):
                            action_proposals.append(prop)

                    if step.tool_name == "check_escalation_ticket":
                        ticket_data = (result.data or {}).get("ticket")
                        if (isinstance(ticket_data, dict)
                                and ticket_data.get("user_id") == user
                                and ticket_data.get("ticket_id")
                                and ticket_data.get("assigned_operator")
                                and ticket_data.get("operator_response")):
                            spans.append(human_span(
                                ticket_data["ticket_id"],
                                ticket_data["assigned_operator"],
                                ticket_data["operator_response"],
                            ))

                    span = tool_span(result)
                    if span:
                        spans.append(span)

                    obs_summary = self._summarize_tool_observation(
                        step=step,
                        result=result,
                        extracted_facts=extracted_facts,
                        step_num=executed_steps,
                    )
                    observation_summaries.append(obs_summary)
                    tracer.add_step(
                        agent="V2_SUPERVISOR",
                        action="observe",
                        details={
                            "observation_id": obs_summary.observation_id,
                            "step_kind": "tool",
                            "tool_name": step.tool_name,
                            "available_facts": len(obs_summary.available_fact_ids),
                            "missing_bindings": len(obs_summary.missing_binding_ids),
                            "diagnostic_codes": obs_summary.diagnostic_codes,
                            "confidence_score": obs_summary.confidence_score,
                        },
                    )

                    # Adaptive Observe -> Replan loop (bounded by max_replans and hop_budget)
                    if replans_used < self.budget.max_replans and executed_steps < max_steps:
                        if not self._budget_exhausted(tracer):
                            replanned, consumed = self._maybe_replan(
                                original=original,
                                context=context,
                                planned_requirements=planned_requirements,
                                step_queue=step_queue,
                                observation_summary=obs_summary,
                                all_observations=observations,
                                executed_signatures=executed_signatures,
                                user=user,
                                tracer=tracer,
                            )
                            if consumed:
                                replans_used += 1
                            if replanned is not None:
                                step_queue = replanned

            if self._budget_exhausted(tracer, log_event=False):
                response = self._empty(
                    session_id,
                    user,
                    "Orçamento de execução excedido antes da síntese final.",
                    tracer,
                )
                response.tools_called = tools
                response.action_proposals = action_proposals
                return response

            compose_context = {
                **context,
                "entities": entities,
                "observations": observations,
                "human_guidance": human_guidance,
                "retrieval_modes": modes,
                "action_proposals": action_proposals,
            }
            document, rendered_text = self.knowledge.compose(
                original,
                spans,
                facts,
                compose_context,
                tracer,
                planned_requirements=planned_requirements,
            )

            if action_proposals:
                notices = [
                    "Proposta " + str(p["proposal_id"]) + " para " + str(p.get("terminal_id", "terminal"))
                    + ": aguardando confirmação explícita do titular. A ação ainda não foi executada."
                    for p in action_proposals if p.get("status") == "PENDING_CONFIRMATION"
                ]
                rendered_text = "\n\n".join([rendered_text, *notices])

            merchant_scopes = [s for s in security_context.scopes if s in {"chat:write", "customer:read_own", "terminal:action_own"}]
            state = {
                "original_request": original,
                "original_scopes": merchant_scopes,
                "requirements": [r.model_dump() for r in planned_requirements],
                "entities": entities,
                "observations": observations,
                "evidence": [s.model_dump() for s in spans],
                "facts": [f.model_dump() for f in facts],
            }
            self.state.save(session_id, user, state)

            status = answer_status(document.requirements)
            tracer.add_step(
                agent="V2_SUPERVISOR",
                action="compose",
                details={
                    "answer_status": status,
                    "retrieval_modes": modes,
                    "requirements": len(document.requirements),
                    "validation_scope": "server_bound_facts_and_blocks",
                },
            )

            return ChatResponse(
                response=GuardrailManager.sanitize_output(rendered_text),
                user_id=user,
                session_id=session_id,
                trace_id=tracer.trace_id,
                agent_used="V2_SUPERVISOR",
                semantic_mode="native_semantic",
                answer_status=status,
                abstained=status != "complete",
                tools_called=tools,
                sources=list(dict.fromkeys(s.source for s in spans)),
                # Public Citation requires a measured relevance score; the
                # candidate retriever supplies none. Preserve references below.
                citations=[],
                request_coverage=[r.model_dump() for r in document.requirements],
                evidence=[f.model_dump() for f in facts] + [s.model_dump() for s in spans],
                action_proposals=action_proposals,
                token_usage=tracer.get_token_usage(),
                trace=tracer.get_steps(),
                execution_time_ms=tracer.get_total_execution_time_ms(),
            )

        except EvidenceError as exc:
            code = getattr(exc, "code", "EVIDENCE_VALIDATION_ERROR")
            tracer.add_step(
                agent="V2_SUPERVISOR",
                action="failed",
                status="error",
                details={"type": "EvidenceError", "code": code, "message": str(exc)},
            )
            response = self._empty(session_id, user, "Não foi possível concluir uma resposta verificada.", tracer)
            response.tools_called = tools
            response.action_proposals = action_proposals
            return response

        except Exception as exc:
            tracer.add_step(
                agent="V2_SUPERVISOR",
                action="failed",
                status="error",
                details={"type": type(exc).__name__, "message": str(exc)},
            )
            response = self._empty(session_id, user, "Não foi possível concluir uma resposta verificada.", tracer)
            response.tools_called = tools
            response.action_proposals = action_proposals
            return response

    @staticmethod
    def _summarize_tool_observation(
        *,
        step: StepProposal,
        result,
        extracted_facts: list[Fact],
        step_num: int,
    ) -> ObservationSummary:
        data = result.data or {}
        available = [f.fact_id for f in extracted_facts if f.availability == "available"]
        missing = [f.binding_id for f in extracted_facts if f.availability != "available" and f.binding_id]
        diag_codes: list[str] = []
        if isinstance(data, dict):
            if data.get("error_code"):
                diag_codes.append(str(data["error_code"]))
            if data.get("ultimo_erro_transacao"):
                diag_codes.append(str(data["ultimo_erro_transacao"]))
            if data.get("chip_algar_status") and str(data["chip_algar_status"]).upper() not in {"ACTIVE", "ATIVO", "OK"}:
                diag_codes.append("CHIP:" + str(data["chip_algar_status"]))
            term = data.get("terminal")
            if isinstance(term, dict):
                if term.get("status") and str(term["status"]).upper() != "ONLINE":
                    diag_codes.append("STATUS:" + str(term["status"]))
                if term.get("model"):
                    diag_codes.append("MODEL:" + str(term["model"]))
        found_flag = data.get("found", True) if isinstance(data, dict) else True
        conf = 1.0 if (found_flag and available) else (0.7 if found_flag else 0.25)
        raw_fp = json.dumps({"tool": step.tool_name, "available": available, "diag": diag_codes}, sort_keys=True)
        return ObservationSummary(
            observation_id=f"obs_{step_num:04d}",
            step_kind="tool",
            source_name=step.tool_name,
            outcome_state="completed" if found_flag else "not_found",
            available_fact_ids=available,
            missing_binding_ids=missing,
            diagnostic_codes=diag_codes,
            evidence_fingerprint=hashlib.sha256(raw_fp.encode("utf-8")).hexdigest()[:16],
            confidence_score=conf,
        )

    def _budget_exhausted(self, tracer: SystemTracer, *, log_event: bool = True) -> bool:
        usage = tracer.get_token_usage()
        if usage.total_tokens > self.budget.token_limit or usage.estimated_cost_usd > self.budget.cost_limit_usd:
            if log_event:
                tracer.add_step(
                    agent="V2_SUPERVISOR",
                    action="budget_gate_stopped",
                    details={
                        "reason": "cost_or_token_budget_exhausted",
                        "total_tokens": usage.total_tokens,
                        "estimated_cost_usd": usage.estimated_cost_usd,
                    },
                )
            return True
        return False

    def _maybe_replan(
        self,
        *,
        original: str,
        context: dict,
        planned_requirements: list[Requirement],
        step_queue: list[StepProposal],
        observation_summary: ObservationSummary,
        all_observations: list[dict],
        executed_signatures: set[str],
        user: str,
        tracer: SystemTracer,
    ) -> tuple[list[StepProposal] | None, bool]:
        has_trigger = (
            bool(observation_summary.diagnostic_codes)
            or observation_summary.outcome_state in {"not_found", "empty"}
            or observation_summary.confidence_score < self.budget.confidence_threshold
        )
        explicit_replan = bool(getattr(self.model, "supports_replan", False))
        if not has_trigger and not explicit_replan:
            return None, False

        if explicit_replan:
            replan_payload = {
                "original_request": original,
                "context": context,
                "planned_requirements": [r.model_dump() for r in planned_requirements],
                "remaining_steps": [s.model_dump() for s in step_queue],
                "latest_observation": observation_summary.model_dump(),
                "observations": all_observations,
                "available_tools": self.gateway.declarations(),
            }
            proposal = self.model.generate(ReplanProposal, replan_payload, phase="replan", tracer=tracer)
            if proposal.action == "continue":
                return None, True
            if proposal.action == "compose":
                tracer.add_step(
                    agent="V2_SUPERVISOR",
                    action="adaptive_replan",
                    details={"action": "compose", "reason_code": proposal.reason_code},
                )
                return [], True
            if proposal.action == "human":
                human_indices = [i for i, r in enumerate(planned_requirements) if r.kind == "human"] or [0]
                new_steps = [StepProposal(kind="human", reason=proposal.reason_code or "Escalonamento adaptativo pós-observação", requirement_indexes=human_indices)]
            elif proposal.action == "replace_remaining":
                _validate_steps_against_requirements(proposal.steps, planned_requirements)
                new_steps = list(proposal.steps)
            else:
                _validate_steps_against_requirements(proposal.steps, planned_requirements)
                new_steps = list(step_queue) + list(proposal.steps)
            filtered: list[StepProposal] = []
            for s in new_steps:
                if s.kind == "human":
                    filtered.append(s)
                    continue
                sig = _step_signature(s, user)
                if sig in executed_signatures:
                    tracer.add_step(
                        agent="V2_SUPERVISOR",
                        action="loop_detected",
                        details={"signature": sig, "kind": s.kind, "tool_name": s.tool_name, "query": s.query},
                    )
                else:
                    filtered.append(s)
            tracer.add_step(
                agent="V2_SUPERVISOR",
                action="adaptive_replan",
                details={
                    "action": proposal.action,
                    "reason_code": proposal.reason_code or "model_replan",
                    "trigger_observation_id": observation_summary.observation_id,
                    "queued_steps": len(filtered),
                },
            )
            return filtered, True

        # Deterministic observation-driven enrichment: if terminal diagnosis surfaces
        # concrete error/model codes and a documentary requirement exists without a
        # query covering those observed codes, append a targeted knowledge lookup.
        doc_req_indices = [i for i, r in enumerate(planned_requirements) if r.kind == "documentary"]
        if observation_summary.source_name == "diagnose_terminal_issue" and observation_summary.diagnostic_codes and doc_req_indices:
            diag_terms = " ".join(
                c.split(":", 1)[-1] for c in observation_summary.diagnostic_codes
            )
            existing_queries = " ".join(s.query for s in step_queue if s.kind == "knowledge").lower()
            if diag_terms.lower() not in existing_queries:
                adaptive_step = StepProposal(
                    kind="knowledge",
                    query=f"{original} {diag_terms}".strip(),
                    reason="adaptive_diagnostic_knowledge_enrichment",
                    requirement_indexes=doc_req_indices,
                )
                if _step_signature(adaptive_step, user) not in executed_signatures:
                    tracer.add_step(
                        agent="V2_SUPERVISOR",
                        action="adaptive_replan",
                        details={
                            "action": "append_steps",
                            "reason_code": "observed_diagnostic_codes",
                            "diagnostic_codes": observation_summary.diagnostic_codes,
                            "trigger_observation_id": observation_summary.observation_id,
                        },
                    )
                    return list(step_queue) + [adaptive_step], True
        return None, False

    @staticmethod
    def _empty(session_id: str, user: str, message: str, tracer: SystemTracer) -> ChatResponse:
        return ChatResponse(
            response=message,
            user_id=user,
            session_id=session_id,
            trace_id=tracer.trace_id,
            agent_used="V2_SUPERVISOR",
            semantic_mode="native_semantic",
            answer_status="abstained",
            abstained=True,
            confidence=0,
            trace=tracer.get_steps(),
            token_usage=tracer.get_token_usage(),
            execution_time_ms=tracer.get_total_execution_time_ms(),
        )

    def resume(self, ticket_id: str, operator: SecurityContext) -> ChatResponse:
        if operator.role != "operator" or not ({"hitl:manage", "admin:all"} & set(operator.scopes)):
            raise PermissionError("A retomada exige operador autorizado.")
        ticket = self.queue.get_ticket(ticket_id)
        if ticket is None or ticket.status != "RESOLVED":
            raise ValueError("Protocolo resolvido não encontrado.")

        # Check if other tickets remain open in this session
        with self.state.store._connect() as conn:
            pending_other = conn.execute(
                "SELECT 1 FROM escalation_tickets WHERE session_id=? AND user_id=? AND ticket_id!=? AND status IN ('OPEN','CLAIMED','RESPONDED') LIMIT 1",
                (ticket.session_id, ticket.user_id, ticket_id),
            ).fetchone()
        if pending_other:
            self.state.store.set_session_status(ticket.session_id, "waiting_human", ticket.user_id)
            tracer = SystemTracer(session_id=ticket.session_id, user_id=ticket.user_id)
            resp = self._empty(ticket.session_id, ticket.user_id, "Atendimento aguardando resposta humana de outro chamado aberto.", tracer)
            resp.escalated = True
            return resp

        state = self.state.context(ticket.session_id, ticket.user_id)["v2"]
        if state.get("waiting_ticket") != ticket_id:
            with self.state.store._connect() as conn:
                row = conn.execute(
                    "SELECT response_json FROM hitl_resumptions WHERE ticket_id=? AND session_id=? AND user_id=?",
                    (ticket_id, ticket.session_id, ticket.user_id),
                ).fetchone()
            if row and row["response_json"]:
                return ChatResponse.model_validate_json(row["response_json"])
            raise ValueError("Protocolo não corresponde à continuação v2.")

        claim = self.state.store.reserve_hitl_resumption(ticket.session_id, ticket.user_id, ticket_id)
        if not claim["acquired"]:
            if claim.get("result"):
                return ChatResponse.model_validate(claim["result"])
            return self._empty(
                ticket.session_id,
                ticket.user_id,
                "Retomada pendente ou em andamento; nenhuma nova execução.",
                SystemTracer(session_id=ticket.session_id, user_id=ticket.user_id),
            )

        tracer = SystemTracer(session_id=ticket.session_id, user_id=ticket.user_id)
        try:
            # Recreate merchant context with strictly preserved original scopes (never escalate or add customer:read_own)
            raw_scopes = state.get("original_scopes")
            if raw_scopes is None:
                preserved_scopes = ["chat:write"]
            else:
                preserved_scopes = [s for s in raw_scopes if s in {"chat:write", "customer:read_own", "terminal:action_own"}]
            merchant = SecurityContext(ticket.user_id, scopes=preserved_scopes)
            response = self._drive(
                state["original_request"],
                merchant,
                ticket.session_id,
                tracer,
                human_guidance=ticket.operator_response or "",
                prior=state,
                prior_ticket=ticket,
            )
            response.ticket_id = ticket_id
            tracer.finish(
                status="completed" if response.answer_status == "complete" else "abstained",
                answer_status=response.answer_status,
                metadata={"resumed_ticket_id": ticket_id, "tools_called": response.tools_called},
            )
            self.state.finish_resumption(
                ticket.session_id,
                ticket.user_id,
                ticket_id,
                claim["reservation_id"],
                response,
                "completed" if response.answer_status == "complete" else "failed",
            )
            return response
        except BaseException as exc:
            tracer.finish(status="failed", answer_status="abstained", metadata={"error": type(exc).__name__})
            self.state.store.fail_hitl_resumption(ticket_id, claim["reservation_id"], "unknown")
            raise
