from collections import deque
from typing import Any, Deque, Dict, List, Optional
from google import genai
from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.escalation_agent import EscalationAgent
from app.agents.guardrails import GuardrailManager
from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.router_agent import RouterAgent
from app.config import settings
from app.memory.store import state_store
from app.observability.logger import logger
from app.observability.metrics import metrics_collector
from app.observability.tracer import SystemTracer
from app.schemas import ChatRequest, ChatResponse, Citation, PlanStep, SpecialistOutput
from app.security.auth import SecurityContext


class MultiAgentOrchestrator:
    """
    Supervisor Multi-Agente com Ciclo Plan-Delegate-Observe-Replan:
    1. Carrega estado multi-turno da sessão no SQLite (`state_store`) e verifica retomada pós-HITL.
    2. Aplica Guardrails de Entrada (Prompt Injection / XSS / SQLi) sem vazar regexes internas.
    3. Planeja a execução via `RouterAgent` com saída validada por Pydantic (`RoutingPlan`).
    4. Executa loop de delegação aos especialistas (`KnowledgeAgent`, `WebSearchAgent`,
       `CustomerSupportAgent`, `EscalationAgent`), observando cada `SpecialistOutput` e
       replanejando dinamicamente quando o resultado de um agente exige outro (ex: falha de hardware
       detectada pelo `CustomerSupportAgent` dispara automaticamente `EscalationAgent`, ou abstention
       do `KnowledgeAgent` redireciona o fluxo).
    5. Aplica Guardrail de Saída (PII Redaction), persiste o turno no SQLite e registra métricas p50/p95.
    """

    def __init__(self) -> None:
        self.gemini_client = None
        if settings.GEMINI_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as exc:
                logger.warning("gemini_client_init_failed", error=str(exc))

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
                blocked=True,
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
                tools_called=["input_guardrail_filter"],
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

        # 2. Planejamento pelo Supervisor / Router Agent (validado via Pydantic RoutingPlan)
        routing_plan = self.router.plan(
            message=request.message,
            user_id=sec_ctx.authenticated_user_id,
            conversation_context=conversation_context,
            tracer=tracer,
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
        pending_steps: Deque[PlanStep] = deque(routing_plan.plan_steps)
        executed_outputs: List[SpecialistOutput] = []
        accumulated_diagnostics: Dict[str, Any] = {}
        agents_executed: List[str] = []
        all_tools_called: List[str] = []
        all_sources: List[str] = []
        all_citations: List[Citation] = []
        escalated = False
        ticket_id: Optional[str] = None
        abstained_final = False

        iteration = 0
        while pending_steps and iteration < settings.MAX_ORCHESTRATION_ITERATIONS:
            iteration += 1
            current_step = pending_steps.popleft()
            target_agent = current_step.target_agent

            spec_output = self._delegate_to_specialist(
                target_agent=target_agent,
                sub_query=current_step.sub_query,
                security_context=sec_ctx,
                session_id=session_id,
                conversation_context=conversation_context,
                accumulated_diagnostics=accumulated_diagnostics,
                rationale=current_step.rationale,
                tracer=tracer,
            )

            executed_outputs.append(spec_output)
            agents_executed.append(spec_output.agent_name)
            for t_name in spec_output.tools_called:
                if t_name not in all_tools_called:
                    all_tools_called.append(t_name)
            for src in spec_output.sources:
                if src not in all_sources:
                    all_sources.append(src)
            for cit in spec_output.citations:
                all_citations.append(cit)
            if spec_output.structured_data:
                accumulated_diagnostics[spec_output.agent_name] = spec_output.structured_data

            if spec_output.agent_name == "HUMAN_ESCALATION_AGENT":
                escalated = True
                ticket_id = spec_output.structured_data.get("protocol_number")

            # Observação e Replanejamento Dinâmico pelo Supervisor
            self._observe_and_replan(
                spec_output=spec_output,
                original_message=request.message,
                pending_steps=pending_steps,
                agents_executed=agents_executed,
                tracer=tracer,
            )

        # Condição de parada atingida
        tracer.add_step(
            agent="SUPERVISOR_ORCHESTRATOR",
            action="evaluate_stopping_condition",
            details={
                "iterations_used": iteration,
                "agents_executed": agents_executed,
                "stopping_reason": "all_planned_and_replanned_steps_completed",
            },
        )

        # 4. Composição da Resposta Final + Retomada de Operador Humano (se aplicável)
        response_segments: List[str] = []
        if latest_operator_msg:
            response_segments.append(
                f"**Atualização do Especialista Humano (Retomada de Sessão):**\n> {latest_operator_msg}"
            )

        for out in executed_outputs:
            # Se um agente absteve mas houve replanejamento para outro agente que respondeu com sucesso, omite o aviso intermediário de abstention
            if out.abstained and len(executed_outputs) > 1 and not all(o.abstained for o in executed_outputs):
                continue
            response_segments.append(out.response)

        if not response_segments and executed_outputs:
            response_segments.append(executed_outputs[-1].response)

        abstained_final = bool(executed_outputs and all(o.abstained for o in executed_outputs))
        raw_final_response = "\n\n---\n\n".join(response_segments)

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
            abstained=abstained_final,
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
            trace=tracer.get_steps(),
            execution_time_ms=total_ms,
        )

    def _delegate_to_specialist(
        self,
        target_agent: str,
        sub_query: str,
        security_context: SecurityContext,
        session_id: str,
        conversation_context: Dict[str, Any],
        accumulated_diagnostics: Dict[str, Any],
        rationale: str,
        tracer: SystemTracer,
    ) -> SpecialistOutput:
        if target_agent == "CUSTOMER_SUPPORT_AGENT":
            out = self.customer_support_agent.run(
                message=sub_query,
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
                details={
                    "protocol": out.structured_data.get("protocol_number"),
                    "has_diagnostic_snapshot": bool(accumulated_diagnostics),
                },
            )
            return out

        # Default: KNOWLEDGE_AGENT (Hybrid RAG)
        out = self.knowledge_agent.run(
            message=sub_query,
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

    @staticmethod
    def _observe_and_replan(
        spec_output: SpecialistOutput,
        original_message: str,
        pending_steps: Deque[PlanStep],
        agents_executed: List[str],
        tracer: SystemTracer,
    ) -> None:
        """
        Inspeciona o resultado real do agente especialista e altera dinamicamente o plano
        caso o diagnóstico revele falha crítica de hardware ou necessidade de transbordo.
        """
        scheduled_agents = {s.target_agent for s in pending_steps} | set(agents_executed)

        # 1. Se o CustomerSupportAgent detectou falha crítica no resultado da tool (ex: hardware fault),
        # o Supervisor replaneja dinamicamente adicionando o HumanEscalationAgent!
        if (
            spec_output.agent_name == "CUSTOMER_SUPPORT_AGENT"
            and spec_output.requires_escalation
            and "HUMAN_ESCALATION_AGENT" not in scheduled_agents
        ):
            reason = spec_output.escalation_reason or "Escalonamento dinâmico disparado pelo diagnóstico técnico."
            pending_steps.append(
                PlanStep(
                    step_order=len(agents_executed) + 1,
                    target_agent="HUMAN_ESCALATION_AGENT",
                    sub_query=original_message,
                    rationale=reason,
                )
            )
            tracer.add_step(
                agent="SUPERVISOR_ORCHESTRATOR",
                action="dynamic_replan",
                details={
                    "trigger": "support_tool_detected_critical_escalation",
                    "added_agent": "HUMAN_ESCALATION_AGENT",
                    "reason": reason,
                },
            )


orchestrator = MultiAgentOrchestrator()
