from typing import Optional, Any
from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.observability.tracer import SystemTracer
from app.agents.guardrails import GuardrailManager
from app.agents.router_agent import RouterAgent
from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.escalation_agent import EscalationAgent

class MultiAgentOrchestrator:
    """
    Orquestrador Central do Sistema Multi-Agente Getnet.
    Coordena a recepção, guardrails de segurança, roteamento semântico,
    execução de agentes especializados e observabilidade completa dos traces.
    """
    def __init__(self):
        self.gemini_client: Optional[Any] = None
        self._init_gemini()
        
        self.router = RouterAgent(gemini_client=self.gemini_client)
        self.knowledge_agent = KnowledgeAgent(gemini_client=self.gemini_client)
        self.customer_support_agent = CustomerSupportAgent(gemini_client=self.gemini_client)
        self.escalation_agent = EscalationAgent()

    def _init_gemini(self):
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                print(f"[Orchestrator] Cliente Google Gemini ({settings.GEMINI_MODEL}) inicializado com sucesso.")
            except Exception as e:
                print(f"[Orchestrator] Aviso: Falha ao inicializar SDK Google GenAI: {e}. Modo autônomo ativado.")

    def process(self, request: ChatRequest) -> ChatResponse:
        tracer = SystemTracer()
        tracer.add_step(
            agent="SYSTEM_INGRESS",
            action="receive_request",
            details={"user_id": request.user_id, "message_preview": request.message[:80]}
        )

        # 1. Guardrail de Segurança (Detecção de injeção e conformidade)
        is_safe, guardrail_reason = GuardrailManager.evaluate_input(request.message)
        if not is_safe:
            tracer.add_step(
                agent="GUARDRAIL_AGENT",
                action="block_unsafe_input",
                details={"reason": guardrail_reason},
                status="blocked"
            )
            return ChatResponse(
                response=f"Solicitação não processada por questões de segurança e integridade: {guardrail_reason}",
                user_id=request.user_id,
                agent_used="GUARDRAIL_AGENT",
                intent="SECURITY_BLOCK",
                tools_called=["guardrail_evaluator"],
                sources=[],
                trace=tracer.get_steps(),
                execution_time_ms=tracer.get_total_execution_time_ms()
            )

        # 2. Triagem e Roteamento Semântico (Router Agent)
        routing_decision = self.router.route(message=request.message, user_id=request.user_id)
        target_agent = routing_decision.get("target_agent", "KNOWLEDGE_AGENT")
        confidence = routing_decision.get("confidence", 0.9)
        reasoning = routing_decision.get("reasoning", "")

        tracer.add_step(
            agent="ROUTER_AGENT",
            action="classify_intent_and_route",
            details={
                "target_agent": target_agent,
                "confidence": confidence,
                "reasoning": reasoning
            }
        )

        # 3. Despacho para Agente(s) Especialista(s) - Suporta Single-Agent ou Multi-Agent Chaining
        response_text = ""
        tools_called = []
        sources = []
        final_agent_label = target_agent

        # Caso A: Cadeia Encadeada de Agentes (Multi-Agent Chaining)
        if routing_decision.get("is_chained"):
            next_agent = routing_decision.get("next_agent", "HUMAN_ESCALATION_AGENT")
            final_agent_label = f"CHAIN: {target_agent} -> {next_agent}"

            # Passo 1 da cadeia: Suporte Operacional / Diagnóstico Preliminar
            support_result = self.customer_support_agent.handle(request.message, request.user_id)
            diag_resp = support_result.get("response", "")
            tools_called.extend(support_result.get("tools_called", []))
            sources.extend(["Getnet Internal Customer DB", "POS Terminal Telemetry"])

            tracer.add_step(
                agent="CUSTOMER_SUPPORT_AGENT",
                action="execute_support_diagnosis",
                details={"tools_invoked": support_result.get("tools_called", []), "user_id": request.user_id}
            )

            # Passo 2 da cadeia: Escalação Humana com contexto enriquecido
            chain_context = f"Diagnóstico automatizado prévio executado ({', '.join(tools_called)}). Solicitação do cliente: {request.message}"
            escalation_result = self.escalation_agent.handle(
                message=request.message,
                user_id=request.user_id,
                reason=chain_context
            )
            esc_resp = escalation_result.get("response", "")
            tools_called.extend(escalation_result.get("tools_called", []))
            sources.append("Getnet Helpdesk Ticketing API")

            tracer.add_step(
                agent="HUMAN_ESCALATION_AGENT",
                action="dispatch_human_ticket_with_context",
                details={
                    "protocol": escalation_result.get("protocol_number"),
                    "enriched_context": f"Tools: {tools_called}"
                }
            )

            response_text = f"{diag_resp}\n\n---\n\n{esc_resp}"

        # Caso B: Agente Único Especialista
        elif target_agent == "CUSTOMER_SUPPORT_AGENT":
            support_result = self.customer_support_agent.handle(request.message, request.user_id)
            response_text = support_result.get("response", "")
            tools_called = support_result.get("tools_called", [])
            sources = ["Getnet Internal Customer DB", "POS Terminal Telemetry"]
            
            tracer.add_step(
                agent="CUSTOMER_SUPPORT_AGENT",
                action="execute_support_tools",
                details={"tools_invoked": tools_called, "user_id": request.user_id}
            )

        elif target_agent == "WEB_SEARCH_AGENT":
            knowledge_result = self.knowledge_agent.handle(request.message, is_web_search=True)
            response_text = knowledge_result.get("response", "")
            tools_called = knowledge_result.get("tools_called", [])
            sources = knowledge_result.get("sources", ["Web Search Engine"])

            tracer.add_step(
                agent="KNOWLEDGE_AGENT",
                action="execute_web_search",
                details={"sources_consulted": sources, "tools_invoked": tools_called}
            )

        elif target_agent == "HUMAN_ESCALATION_AGENT":
            escalation_result = self.escalation_agent.handle(
                message=request.message,
                user_id=request.user_id,
                reason=reasoning or "Solicitação direta de transbordo humano"
            )
            response_text = escalation_result.get("response", "")
            tools_called = escalation_result.get("tools_called", [])
            sources = ["Getnet Helpdesk Ticketing API"]

            tracer.add_step(
                agent="HUMAN_ESCALATION_AGENT",
                action="dispatch_human_ticket",
                details={"protocol": escalation_result.get("protocol_number")}
            )

        else: # Padrão: KNOWLEDGE_AGENT com RAG
            knowledge_result = self.knowledge_agent.handle(request.message, is_web_search=False)
            response_text = knowledge_result.get("response", "")
            tools_called = knowledge_result.get("tools_called", [])
            sources = knowledge_result.get("sources", ["getnet.net official knowledge"])

            tracer.add_step(
                agent="KNOWLEDGE_AGENT",
                action="retrieve_and_synthesize_rag",
                details={"sources_consulted": sources, "tools_invoked": tools_called}
            )

        tracer.add_step(
            agent="SYSTEM_EGRESS",
            action="finalize_response",
            details={"response_char_count": len(response_text)}
        )

        return ChatResponse(
            response=response_text,
            user_id=request.user_id,
            agent_used=final_agent_label,
            intent=reasoning,
            tools_called=tools_called,
            sources=sources,
            trace=tracer.get_steps(),
            execution_time_ms=tracer.get_total_execution_time_ms()
        )

# Instância singleton
orchestrator = MultiAgentOrchestrator()
