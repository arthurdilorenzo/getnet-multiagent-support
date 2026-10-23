from collections import deque
import re
import hashlib
from typing import Any, Deque, Dict, List, Optional
from google import genai
from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.function_calling import RouterPlanningError
from app.agents.escalation_agent import EscalationAgent
from app.agents.guardrails import GuardrailManager
from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.router_agent import RouterAgent
from app.config import settings
from app.memory.store import state_store
from app.observability.logger import logger
from app.observability.metrics import metrics_collector
from app.observability.tracer import SystemTracer
from app.schemas import ChatRequest, ChatResponse, Citation, PlanStep, SpecialistOutput, RoutingPlan
from app.security.auth import SecurityContext
from app.security.pii import PIIRedactor
from app.security.request_units import RequestCoverageLedger, request_units
from app.security.operational_intent import classify_operational_intent
from app.security.fulfillment import canonical_terminal_read_text


class MultiAgentOrchestrator:
    """
    Supervisor Multi-Agente com Ciclo Plan-Delegate-Observe-Replan:
    1. Carrega estado multi-turno da sessão no SQLite (`state_store`) e verifica retomada pós-HITL.
    2. Aplica Guardrails de Entrada (Prompt Injection / XSS / SQLi) sem vazar regexes internas.
    3. Planeja a execução via `RouterAgent` com saída validada por Pydantic (`RoutingPlan`).
    4. Executa loop de delegação aos especialistas (`KnowledgeAgent`, `WebSearchAgent`,
       `CustomerSupportAgent`, `EscalationAgent`), observando cada `SpecialistOutput` e
       replanejando dinamicamente quando o resultado de um agente exige outro (ex: falha de hardware
       detectada pelo `CustomerSupportAgent` dispara automaticamente `EscalationAgent`).
    5. Aplica Guardrail de Saída (PII Redaction), persiste o turno no SQLite e registra métricas p50/p95.
    """

    @staticmethod
    def _answer_metadata(outputs: List[SpecialistOutput], incomplete: bool = False) -> tuple[str, str]:
        modes = {out.semantic_mode for out in outputs if out.semantic_mode}
        mode = 'limited_offline' if 'limited_offline' in modes else 'native_semantic' if modes else ''
        if not mode:
            return '', ''
        if all(out.abstained and out.answer_status != 'partial' for out in outputs):
            return mode, 'abstained'
        if not incomplete and all(out.answer_status == 'complete' and not out.abstained for out in outputs):
            return mode, 'complete'
        return mode, 'partial'

    @staticmethod
    def _verified_unit_exemptions(output: SpecialistOutput, original: str) -> Optional[Dict[str,str]]:
        mapping=output.structured_data.get('request_unit_exemptions',{})
        if not mapping: return {}
        if not isinstance(mapping,dict) or output.agent_name!='KNOWLEDGE_AGENT' or output.semantic_mode!='native_semantic': return None
        digest=hashlib.sha256(original.encode()).hexdigest()
        proof={}
        for result in output.tool_results:
            if not result.success or result.tool_name!='search_getnet_knowledge': continue
            source=(result.data or {}).get('answerability',{})
            if source.get('method')!='native_two_pass_evidence_authorization' or source.get('request_hash')!=digest: continue
            for row in source.get('coverage',[]):
                if row.get('kind') in {'context','courtesy'} and (row.get('verification') or {}).get('verdict')=='accept':
                    proof[row.get('unit_id')]=row['kind']
        if any(uid not in output.request_unit_ids or kind not in {'context','courtesy'} or proof.get(uid)!=kind for uid,kind in mapping.items()): return None
        return dict(mapping)

    @staticmethod
    def _without_unit_footer(text: str) -> str:
        marker='Partes do pedido sem resposta verificada:'
        if marker not in text: return text
        prefix,tail=text.split(marker,1)
        lines=tail.splitlines()
        # Remove only the generated list; preserve subsequent caveats/errors.
        while lines and (not lines[0].strip() or lines[0].lstrip().startswith('- ')):
            lines.pop(0)
        return prefix.rstrip()+("\n\n"+"\n".join(lines) if lines else '')

    def __init__(self) -> None:
        self.gemini_client = None
        if settings.GEMINI_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY,
                    http_options={"timeout": int(settings.LLM_TIMEOUT_SECONDS * 1000)})
            except Exception as exc:
                logger.warning("gemini_client_init_failed", error_type=type(exc).__name__)

        self.router = RouterAgent(gemini_client=self.gemini_client)
        self.knowledge_agent = KnowledgeAgent(gemini_client=self.gemini_client)
        self.customer_support_agent = CustomerSupportAgent(gemini_client=self.gemini_client)
        self.escalation_agent = EscalationAgent()

    def process(
        self,
        request: ChatRequest,
        security_context: Optional[SecurityContext] = None,
    ) -> ChatResponse:
        sec_ctx = security_context or SecurityContext(authenticated_user_id=request.user_id)
        session_id = state_store.resolve_session_id(
            user_id=sec_ctx.authenticated_user_id,
            provided_session_id=request.session_id,
        )
        tracer = SystemTracer(session_id=session_id, user_id=sec_ctx.authenticated_user_id)

        # 0. Carrega memória multi-turno da sessão e checa retomada de atendimento humano (HITL)
        conversation_context = state_store.get_conversation_context(
            session_id=session_id,
            user_id=sec_ctx.authenticated_user_id,
            limit=settings.MEMORY_WINDOW_MESSAGES,
        )
        recent_msgs = conversation_context.get("recent_messages", [])
        latest_operator_msg: Optional[str] = None
        if recent_msgs and recent_msgs[-1].get("role") == "human_operator":
            latest_operator_msg = recent_msgs[-1].get("content")

        tracer.add_step(
            agent="SYSTEM_INGRESS",
            action="receive_and_load_session",
            details={
                "user_id": sec_ctx.authenticated_user_id,
                "session_id": session_id,
                "auth_method": sec_ctx.auth_method,
                "previous_turns": conversation_context.get("turn_count", 0),
                "hitl_operator_reply_present": bool(latest_operator_msg),
            },
        )

        # 1. Validação de Segurança e Guardrails de Entrada (sem expor regex!)
        is_safe, public_reason, rule_id = GuardrailManager.evaluate_input(request.message)
        if not is_safe:
            tracer.add_step(
                agent="GUARDRAILS_INPUT",
                action="block_unsafe_request",
                status="blocked",
                details={"rule_id": rule_id, "policy_action": "REJECT"},
            )
            safe_refusal = (
                f"Não foi possível processar esta solicitação: {public_reason} "
                f"(Código de auditoria: `{rule_id}`). Sou o assistente virtual da Getnet e estou "
                f"disponível para auxiliar com dúvidas sobre maquininhas, Pix, vendas e suporte técnico."
            )
            token_usage = tracer.get_token_usage()
            total_ms = tracer.get_total_execution_time_ms()
            metrics_collector.record_request(
                agent_used="GUARDRAILS_BLOCK",
                tools_called=["input_guardrail_filter"],
                latency_ms=total_ms,
                token_usage=token_usage,
                blocked=True, semantic_mode="", answer_status="abstained",
            )
            return ChatResponse(
                response=safe_refusal,
                user_id=sec_ctx.authenticated_user_id,
                session_id=session_id,
                trace_id=tracer.trace_id,
                agent_used="GUARDRAILS_BLOCK",
                intent="Blocked by Security Policy",
                confidence=1.0,
                abstained=True,
                tools_called=["input_guardrail_filter"], answer_status="abstained",
                sources=["Getnet AI Security Policy"],
                token_usage=token_usage,
                trace=tracer.get_steps(),
                execution_time_ms=total_ms,
            )

        # Registra a mensagem do usuário na memória persistente SQLite
        state_store.append_message(
            session_id=session_id,
            user_id=sec_ctx.authenticated_user_id,
            role="user",
            content=request.message,
        )

        original_request=PIIRedactor.mask_text(request.message)
        ledger=RequestCoverageLedger(original_request)

        # 2. Planejamento pelo Supervisor / Router Agent (validado via Pydantic RoutingPlan)
        try:
            routing_plan = self.router.plan(
                message=request.message,
                user_id=sec_ctx.authenticated_user_id,
                conversation_context=conversation_context,
                tracer=tracer,
            )
            routing_plan=RouterAgent.bind_request_units(routing_plan,original_request,conversation_context)
        except RouterPlanningError:
            refusal = "Não foi possível planejar o atendimento pelo provedor de IA. Tente novamente em instantes."
            tracer.add_step(agent="ROUTER_AGENT", action="planning_failed", status="abstained",
                            details={"reason": "configured_provider_unavailable_or_invalid"})
            state_store.append_message(
                session_id=session_id, user_id=sec_ctx.authenticated_user_id,
                role="assistant", content=refusal, agent_used="ROUTER_AGENT",
            )
            total_ms = tracer.get_total_execution_time_ms()
            token_usage = tracer.get_token_usage()
            metrics_collector.record_request(
                agent_used="ROUTER_AGENT", tools_called=[], latency_ms=total_ms,
                token_usage=token_usage, abstained=True,semantic_mode="",answer_status="abstained",
            )
            return ChatResponse(
                response=refusal, user_id=sec_ctx.authenticated_user_id,
                session_id=session_id, trace_id=tracer.trace_id,
                agent_used="ROUTER_AGENT", confidence=0.0, abstained=True,answer_status="abstained",request_coverage=ledger.snapshot(),
                token_usage=token_usage, trace=tracer.get_steps(), execution_time_ms=total_ms,
            )

        tracer.add_step(
            agent="ROUTER_AGENT",
            action="plan_and_route",
            details={
                "primary_agent": routing_plan.primary_agent,
                "confidence": routing_plan.confidence,
                "is_multi_step": routing_plan.is_multi_step,
                "planned_steps": [s.target_agent for s in routing_plan.plan_steps],
                "reasoning": routing_plan.reasoning,
            },
        )

        # 3. Loop de Orquestração do Supervisor (Delegate -> Observe -> Dynamic Replan -> Stop)
        for step in routing_plan.plan_steps:
            ledger.assign(step.target_agent,step.request_unit_ids)
        pending_steps: Deque[PlanStep] = deque(routing_plan.plan_steps)
        executed_outputs: List[SpecialistOutput] = []
        accumulated_diagnostics: Dict[str, Any] = {}
        agents_executed: List[str] = []
        all_tools_called: List[str] = []
        all_sources: List[str] = []
        all_citations: List[Citation] = []
        all_evidence: List[Dict[str, Any]] = []
        action_proposals: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []
        replan_failed = False
        escalated = False
        ticket_id: Optional[str] = None
        abstained_final = False

        iteration = 0
        while pending_steps and iteration < settings.MAX_ORCHESTRATION_ITERATIONS:
            iteration += 1
            current_step = pending_steps.popleft()
            target_agent = current_step.target_agent
            try:
                scoped_plan=RouterAgent.bind_request_units(RoutingPlan(primary_agent=target_agent,plan_steps=[current_step],confidence=0.0,reasoning="Server unit binding"),original_request,conversation_context,complete_assignment=False)
                current_step=scoped_plan.plan_steps[0]
                ledger.assign(target_agent,current_step.request_unit_ids)
            except (ValueError,RouterPlanningError):
                replan_failed=True
                break
            rows=ledger.snapshot()
            assigned=list(current_step.request_unit_ids)
            other=[row['unit_id'] for row in rows if row['unit_id'] not in assigned or any(agent!=target_agent for agent in row['assigned_agents'])]
            scope={'assigned_unit_ids':assigned,'other_unit_ids':other}
            step_context = dict(conversation_context, specialist_observations=list(observations))
            sub_query = self._resolve_dependent_query(current_step.sub_query, target_agent, observations)

            spec_output = self._delegate_to_specialist(
                target_agent=target_agent,
                sub_query=sub_query,
                authorization_message=original_request,
                delegation_scope=scope,
                security_context=sec_ctx,
                session_id=session_id,
                conversation_context=step_context,
                accumulated_diagnostics=accumulated_diagnostics,
                rationale=current_step.rationale,
                tracer=tracer,
            )

            exemptions=self._verified_unit_exemptions(spec_output,original_request)
            accepted=exemptions is not None and ledger.report(target_agent,spec_output.request_unit_ids,spec_output.covered_unit_ids,
                                   spec_output.unresolved_unit_ids,evidence_count=len(spec_output.evidence),exemptions=exemptions)
            if not accepted:
                spec_output.abstained=True
                spec_output.answer_status='partial' if spec_output.evidence else 'abstained'
                tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR",action="coverage_report_rejected",status="error",
                                details={"reason":"unknown_or_unassigned_unit_ids"})
            executed_outputs.append(spec_output)
            observations.append(self._observation(spec_output))
            state_store.remember_tool_referents(session_id, sec_ctx.authenticated_user_id,
                                                [r.model_dump() for r in spec_output.tool_results])
            agents_executed.append(spec_output.agent_name)
            for t_name in spec_output.tools_called:
                if t_name not in all_tools_called:
                    all_tools_called.append(t_name)
            for src in spec_output.sources:
                if src not in all_sources:
                    all_sources.append(src)
            for cit in spec_output.citations:
                all_citations.append(cit)
            for evidence in spec_output.evidence:
                masked = PIIRedactor.mask_structure(evidence)
                if masked not in all_evidence:
                    all_evidence.append(masked)
            for result in spec_output.tool_results:
                data = result.data or {}
                if (result.success and result.tool_name == "execute_terminal_remote_action"
                    and data.get("status") == "PENDING_CONFIRMATION" and data.get("proposal_id")):
                    if not any(p.get("proposal_id") == data["proposal_id"] for p in action_proposals):
                        action_proposals.append(PIIRedactor.mask_structure(data))
            if spec_output.structured_data:
                accumulated_diagnostics[spec_output.agent_name] = spec_output.structured_data

            if spec_output.agent_name == "HUMAN_ESCALATION_AGENT" and not spec_output.abstained:
                ticket_id = spec_output.structured_data.get("protocol_number")
                escalated = bool(ticket_id)

            # Observação e Replanejamento Dinâmico pelo Supervisor
            replan_failed = self._observe_and_replan(
                spec_output=spec_output,
                original_message=request.message,
                pending_steps=pending_steps,
                agents_executed=agents_executed,
                user_id=sec_ctx.authenticated_user_id,
                observations=observations,
                conversation_context=conversation_context,
                tracer=tracer,
                request_coverage=ledger.snapshot(),
            ) or replan_failed

        # Condição de parada atingida
        tracer.add_step(
            agent="SUPERVISOR_ORCHESTRATOR",
            action="evaluate_stopping_condition",
            details={
                "iterations_used": iteration,
                "agents_executed": agents_executed,
                "stopping_reason": "iteration_limit" if pending_steps else "all_planned_and_replanned_steps_completed",
                "pending_steps": len(pending_steps),
            },
        )

        # 4. Composição da Resposta Final + Retomada de Operador Humano (se aplicável)
        response_segments: List[str] = []
        if latest_operator_msg:
            response_segments.append(
                f"**Atualização do Especialista Humano (Retomada de Sessão):**\n> {latest_operator_msg}"
            )

        for out in executed_outputs:
            # Preserve partial failures: success in one intent cannot hide a denied
            # operation or a failed human handoff in another intent.
            response_segments.append(self._without_unit_footer(out.response))

        if not response_segments and executed_outputs:
            response_segments.append(executed_outputs[-1].response)
        if pending_steps:
            response_segments.append("O atendimento atingiu o limite de etapas. Parte do pedido ficou pendente; envie uma nova mensagem para continuar.")
        if replan_failed:
            response_segments.append("O replanejamento ficou indisponível. As consultas concluídas estão preservadas; parte do pedido pode estar pendente.")

        abstained_final = bool(pending_steps or replan_failed or any(o.abstained for o in executed_outputs))
        semantic_mode, answer_status = self._answer_metadata(
            executed_outputs, incomplete=bool(pending_steps or replan_failed))
        request_coverage=ledger.snapshot()
        unresolved=[row for row in request_coverage if row['state'] not in {'fulfilled','exempt'}]
        if unresolved:
            abstained_final=True
            answer_status='partial' if any(row['state']=='fulfilled' for row in request_coverage) or any(out.evidence and out.structured_data for out in executed_outputs) else 'abstained'
        elif request_coverage and not pending_steps and not replan_failed and not abstained_final:
            answer_status='complete'
        raw_final_response = "\n\n---\n\n".join(segment for segment in response_segments if segment)
        if unresolved:
            raw_final_response+="\n\nPartes do pedido sem resposta verificada:\n"+"\n".join("- "+row['text'] for row in unresolved)

        # 5. Guardrail de Saída (Redaction de PII e prevenção de vazamentos)
        final_response = GuardrailManager.sanitize_output(raw_final_response)

        final_agent_label = (
            f"CHAIN: {' -> '.join(agents_executed)}"
            if len(agents_executed) > 1
            else (agents_executed[0] if agents_executed else routing_plan.primary_agent)
        )

        # Persiste resposta do assistente na memória SQLite da sessão
        state_store.append_message(
            session_id=session_id,
            user_id=sec_ctx.authenticated_user_id,
            role="assistant",
            content=final_response,
            agent_used=final_agent_label,
            tools_called=all_tools_called,
            sources=all_sources,
            active_topic=routing_plan.primary_agent,
        )

        tracer.add_step(
            agent="SYSTEM_EGRESS",
            action="sanitize_and_persist_response",
            details={
                "response_char_count": len(final_response),
                "session_id": session_id,
                "abstained": abstained_final,
                "semantic_mode": semantic_mode,
                "answer_status": answer_status,
                "escalated": escalated,
            },
        )

        token_usage = tracer.get_token_usage()
        total_ms = tracer.get_total_execution_time_ms()
        avg_conf = (
            round(sum(o.confidence for o in executed_outputs) / len(executed_outputs), 4)
            if executed_outputs
            else routing_plan.confidence
        )

        metrics_collector.record_request(
            agent_used=final_agent_label,
            tools_called=all_tools_called,
            latency_ms=total_ms,
            token_usage=token_usage,
            blocked=False,
            escalated=escalated,
            abstained=abstained_final,semantic_mode=semantic_mode,answer_status=answer_status,
        )

        return ChatResponse(
            response=final_response,
            user_id=sec_ctx.authenticated_user_id,
            session_id=session_id,
            trace_id=tracer.trace_id,
            agent_used=final_agent_label,
            intent=routing_plan.reasoning,
            confidence=avg_conf,
            abstained=abstained_final,
            escalated=escalated,
            ticket_id=ticket_id,
            tools_called=all_tools_called,
            sources=all_sources,
            citations=all_citations,
            token_usage=token_usage,
            evidence=all_evidence,
            action_proposals=action_proposals,
            semantic_mode=semantic_mode,
            answer_status=answer_status,
            request_coverage=request_coverage,
            trace=tracer.get_steps(),
            execution_time_ms=total_ms,
        )

    def _delegate_to_specialist(
        self,
        target_agent: str,
        sub_query: str,
        authorization_message: str,
        security_context: SecurityContext,
        session_id: str,
        conversation_context: Dict[str, Any],
        accumulated_diagnostics: Dict[str, Any],
        rationale: str,
        tracer: SystemTracer,
        delegation_scope: Optional[Dict[str,Any]] = None,
    ) -> SpecialistOutput:
        if target_agent == "CUSTOMER_SUPPORT_AGENT":
            out = self.customer_support_agent.run(
                message=sub_query,
                authorization_message=authorization_message,
                delegation_scope=delegation_scope,
                security_context=security_context,
                session_id=session_id,
                conversation_context=conversation_context,
                tracer=tracer,
            )
            tracer.add_step(
                agent="CUSTOMER_SUPPORT_AGENT",
                action="complete_support_cycle",
                details={
                    "tools_invoked": out.tools_called,
                    "requires_escalation": out.requires_escalation,
                    "confidence": out.confidence,
                },
            )
            return out

        if target_agent == "WEB_SEARCH_AGENT":
            out = self.knowledge_agent.run(
                message=sub_query,
                request_message=authorization_message,
                delegation_scope=delegation_scope,
                security_context=security_context,
                is_web_search=True,
                session_id=session_id,
                conversation_context=conversation_context,
                tracer=tracer,
            )
            tracer.add_step(
                agent="WEB_SEARCH_AGENT",
                action="complete_external_search",
                status="success" if not out.abstained else "abstained",
                details={
                    "tools_invoked": out.tools_called,
                    "sources_consulted": out.sources,
                    "abstained": out.abstained,
                },
            )
            return out

        if target_agent == "HUMAN_ESCALATION_AGENT":
            diag_reason = rationale
            if accumulated_diagnostics:
                diag_reason = (
                    f"{rationale} | Diagnóstico prévio anexado: "
                    f"{', '.join(accumulated_diagnostics.keys())}"
                )
            out = self.escalation_agent.run(
                message=sub_query,
                security_context=security_context,
                session_id=session_id,
                reason=diag_reason,
                category="chained_support_escalation" if accumulated_diagnostics else "human_handoff",
                priority="critical" if accumulated_diagnostics else "high",
                diagnostic_snapshot=accumulated_diagnostics,
                tracer=tracer,
            )
            tracer.add_step(
                agent="HUMAN_ESCALATION_AGENT",
                action="enqueue_hitl_ticket",
                status="abstained" if out.abstained else "success",
                details={
                    "protocol": out.structured_data.get("protocol_number"),
                    "has_diagnostic_snapshot": bool(accumulated_diagnostics),
                },
            )
            return out

        # Default: KNOWLEDGE_AGENT (Hybrid RAG)
        out = self.knowledge_agent.run(
            message=sub_query,
            request_message=authorization_message,
            delegation_scope=delegation_scope,
            security_context=security_context,
            is_web_search=False,
            session_id=session_id,
            conversation_context=conversation_context,
            tracer=tracer,
        )
        tracer.add_step(
            agent="KNOWLEDGE_AGENT",
            action="retrieve_and_synthesize_hybrid_rag",
            status="success" if not out.abstained else "abstained",
            details={
                "tools_invoked": out.tools_called,
                "citations_count": len(out.citations),
                "confidence": out.confidence,
                "abstained": out.abstained,
            },
        )
        return out

    def _observe_and_replan(
        self,
        spec_output: SpecialistOutput,
        original_message: str,
        pending_steps: Deque[PlanStep],
        agents_executed: List[str],
        user_id: str,
        observations: List[Dict[str, Any]],
        conversation_context: Dict[str, Any],
        tracer: SystemTracer,
        request_coverage: Optional[List[Dict[str,Any]]] = None,
    ) -> bool:
        """Bounded feedback planning; failures preserve every completed/partial result."""
        if spec_output.agent_name == "HUMAN_ESCALATION_AGENT":
            return False
        if (request_coverage and not spec_output.requires_escalation
                and all(row['state'] in {'fulfilled','exempt'} for row in request_coverage)):
            tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR",action="replan_not_needed",status="skipped",
                            details={"reason":"all_request_units_covered","unit_count":len(request_coverage)})
            return False
        failed_tools = any(not r.success for r in spec_output.tool_results)
        if request_coverage and not (spec_output.requires_escalation or spec_output.abstained or failed_tools):
            remaining={row['unit_id'] for row in request_coverage if row['state'] not in {'fulfilled','exempt'}}
            scheduled_ids={uid for step in pending_steps for uid in step.request_unit_ids}
            if remaining and remaining<=scheduled_ids:
                tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR",action="replan_not_needed",status="skipped",
                                details={"reason":"remaining_request_units_scheduled","unit_count":len(remaining)})
                return False
        # A provider/protocol rejection supplies no new actionable fact. Do not
        # spend another provider call inventing a private recovery for a public
        # request; an explicitly requested private unit can still need work.
        canonical_units = request_units(PIIRedactor.mask_text(original_message))
        known_ids = {unit['unit_id'] for unit in canonical_units}
        remaining_ids = ({row['unit_id'] for row in request_coverage
                          if row['state'] not in {'fulfilled', 'exempt'}}
                         if request_coverage is not None else known_ids)
        unit_intents = {unit['unit_id']: classify_operational_intent(
            canonical_terminal_read_text(unit['text'].lstrip(' ;,.'),
                PIIRedactor.mask_text(original_message), conversation_context), conversation_context)
            for unit in canonical_units}
        private_ids = {uid for uid, intent in unit_intents.items()
                       if intent.private_scope and not intent.documentary_scope}
        semantic_error = (spec_output.structured_data or {}).get('semantic_error') or {}
        if (spec_output.agent_name in {'KNOWLEDGE_AGENT', 'WEB_SEARCH_AGENT'}
                and isinstance(semantic_error, dict) and semantic_error.get('code')
                and not (remaining_ids & private_ids) and not spec_output.requires_escalation):
            tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR", action="replan_not_needed", status="skipped",
                            details={"reason": "no_actionable_observation"})
            return False
        # A completed support result can reveal a dependent query even without a failure.
        if not (spec_output.requires_escalation or spec_output.abstained or failed_tools
                or spec_output.agent_name == "CUSTOMER_SUPPORT_AGENT"):
            return False
        try:
            plan = self.router.replan(original_message=original_message, user_id=user_id,
                observations=observations, conversation_context=conversation_context, tracer=tracer)
        except RouterPlanningError:
            tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR", action="replanning_failed", status="abstained",
                            details={"reason": "configured_provider_unavailable_or_invalid"})
            return True
        scheduled = {s.target_agent for s in pending_steps} | set(agents_executed)
        for step in plan.plan_steps:
            if step.target_agent in scheduled:
                tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR", action="replan_no_progress", status="skipped",
                                details={"target_agent": step.target_agent, "reason": "already_executed_or_pending"})
                continue
            # Native delegation proposes work; canonical pending units authorize
            # its scope. Rationale/sub_query cannot create a private need.
            try:
                rebound = RouterAgent.bind_request_units(
                    RoutingPlan(primary_agent=step.target_agent, plan_steps=[step],
                                confidence=0.0, reasoning="Server replan scope validation"),
                    PIIRedactor.mask_text(original_message), conversation_context,
                    complete_assignment=False)
                step = rebound.plan_steps[0]
                ids = set(step.request_unit_ids)
                if step.target_agent == 'HUMAN_ESCALATION_AGENT':
                    permitted = spec_output.requires_escalation or failed_tools or bool(ids & remaining_ids)
                else:
                    permitted = bool(ids) and ids <= remaining_ids
                    if step.target_agent == 'CUSTOMER_SUPPORT_AGENT':
                        permitted = permitted and ids <= private_ids
            except (ValueError, RouterPlanningError):
                permitted = False
            if not permitted:
                tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR", action="replan_scope_rejected", status="skipped",
                                details={"target_agent": step.target_agent,
                                         "reason": "no_authorized_pending_units"})
                continue
            if len(agents_executed) + len(pending_steps) >= settings.MAX_ORCHESTRATION_ITERATIONS:
                # Keep one pending step so the final answer explicitly reports the limit.
                if not pending_steps:
                    pending_steps.append(step)
                break
            pending_steps.append(step)
            scheduled.add(step.target_agent)
            trigger = ("support_tool_detected_critical_escalation" if spec_output.requires_escalation
                       else "operational_tool_failure" if failed_tools
                       else "insufficient_evidence" if spec_output.abstained else "dependent_observation")
            tracer.add_step(agent="SUPERVISOR_ORCHESTRATOR", action="dynamic_replan", details={
                "trigger": trigger, "added_agent": step.target_agent, "reason": step.rationale,
                "policy": "native" if settings.GEMINI_API_KEY else "explicit_offline_feedback"})
        return False

    @staticmethod
    def _observation(output: SpecialistOutput) -> Dict[str, Any]:
        return PIIRedactor.mask_structure({
            "agent_name": output.agent_name, "completed": True,
            "response": output.response[:2000], "abstained": output.abstained,
            "abstention_reason": output.abstention_reason,
            "requires_escalation": output.requires_escalation, "escalation_reason": output.escalation_reason,
            "structured_data": output.structured_data,
            "tool_results": [r.model_dump() for r in output.tool_results], "evidence": output.evidence,
        })

    @staticmethod
    def _resolve_dependent_query(sub_query: str, target_agent: str, observations: List[Dict[str, Any]]) -> str:
        if target_agent != "KNOWLEDGE_AGENT":
            return sub_query
        norm = sub_query.lower()
        if re.search(r"\b(?:código|codigo|code)\s*\d+", norm):
            return sub_query
        if re.search(r"\b(?:código|codigo|code)\b", norm):
            code = RouterAgent.observed_decline_code(observations)
            if code:
                return f"{sub_query}. Código observado no diagnóstico: {code}."
        return sub_query


orchestrator = MultiAgentOrchestrator()
