import json
import re
from typing import Any, Dict, List, Optional
from app.config import settings
from app.observability.logger import logger
from app.observability.resilience import call_with_resilience
from app.observability.tracer import SystemTracer
from app.rag.embeddings import BilingualDenseEmbedder
from app.schemas import AgentTypeLiteral, PlanStep, RoutingPlan

ROUTER_SYSTEM_PROMPT = """Você é o Supervisor & Router Agent do sistema multi-agente da Getnet.
Sua responsabilidade é analisar a mensagem do usuário, considerar o histórico da sessão e construir um plano de execução estruturado (`RoutingPlan`).

Agentes especialistas disponíveis (como tools de delegação):
1. "KNOWLEDGE_AGENT": Dúvidas sobre produtos, maquininhas (Get Clássica, Get Smart, Get Tap), Pix, Get Conta, Antecipação de Recebíveis, Crediário, Link de Pagamento, taxas e manuais técnicos Getnet.
2. "WEB_SEARCH_AGENT": Consultas externas que dependem de dados públicos em tempo real na web (previsão do tempo/clima em cidades, cotação de moedas como Euro/Dólar hoje, fatos públicos gerais).
3. "CUSTOMER_SUPPORT_AGENT": Consultas operacionais autenticadas sobre a conta específica do cliente ou telemetria de suas maquininhas (quando cai o dinheiro das vendas de ontem, maquininha sem internet/sinal, erro de transação recusada no visor, perfil cadastral, reinício remoto de terminal).
4. "HUMAN_ESCALATION_AGENT": Solicitações explícitas de atendimento por especialista humano, ouvidoria, abertura de chamado Nível 2 ou contestação crítica.

Se a mensagem contiver múltiplas intenções (ex: consultar problema na máquina E pedir atendente humano, ou perguntar cotação do euro E status das vendas de ontem), decomponha em múltiplos passos ordenados em `plan_steps`.

Responda ESTRITAMENTE em JSON válido seguindo o schema `RoutingPlan`:
{
  "primary_agent": "KNOWLEDGE_AGENT | WEB_SEARCH_AGENT | CUSTOMER_SUPPORT_AGENT | HUMAN_ESCALATION_AGENT",
  "confidence": 0.95,
  "reasoning": "Justificativa técnica do plano de roteamento",
  "plan_steps": [
    {
      "step_order": 1,
      "target_agent": "CUSTOMER_SUPPORT_AGENT",
      "sub_query": "Sub-consulta para o agente",
      "rationale": "Motivo deste passo"
    }
  ],
  "requires_user_context": true,
  "is_multi_step": false
}
"""


class RouterAgent:
    """
    Agente 1: Supervisor Planner & Router Agent.
    - Quando `GEMINI_API_KEY` está configurada, utiliza o LLM primeiro (sem pré-empção por palavras-chave)
      e valida estritamente o JSON de saída com Pydantic (`RoutingPlan.model_validate`).
    - Possui decompositor multi-intenção determinístico para operação resiliente em CI/offline.
    """

    VALID_AGENTS = {
        "KNOWLEDGE_AGENT",
        "WEB_SEARCH_AGENT",
        "CUSTOMER_SUPPORT_AGENT",
        "HUMAN_ESCALATION_AGENT",
    }

    def __init__(self, gemini_client: Optional[Any] = None) -> None:
        self.client = gemini_client

    def plan(
        self,
        message: str,
        user_id: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        tracer: Optional[SystemTracer] = None,
    ) -> RoutingPlan:
        """
        Gera um `RoutingPlan` validado via Pydantic.
        """
        history_summary = ""
        if conversation_context:
            recent = conversation_context.get("recent_messages", [])
            if recent:
                last_turns = [f"{m['role']}: {m['content'][:120]}" for m in recent[-4:]]
                history_summary = "\n".join(last_turns)

        # 1. Classificação e Planejamento primário via LLM (quando configurado)
        if self.client and settings.GEMINI_API_KEY:
            try:
                prompt = (
                    f"Usuário autenticado: {user_id}\n"
                    f"Histórico recente da sessão:\n{history_summary or 'Início de sessão'}\n\n"
                    f"Mensagem atual do usuário:\n\"{message}\"\n\n"
                    f"Elabore o RoutingPlan em JSON."
                )

                def _call_llm() -> Any:
                    return self.client.models.generate_content(
                        model=settings.GEMINI_MODEL,
                        contents=prompt,
                        config={
                            "system_instruction": ROUTER_SYSTEM_PROMPT,
                            "response_mime_type": "application/json",
                            "temperature": 0.0,
                        },
                    )

                response = call_with_resilience("gemini_api", _call_llm, max_retries=1)
                if response and response.text:
                    if tracer:
                        tracer.record_tokens(prompt_text=prompt, completion_text=response.text)
                    raw_data = json.loads(response.text)
                    validated_plan = RoutingPlan.model_validate(raw_data)
                    if not validated_plan.plan_steps:
                        validated_plan.plan_steps = [
                            PlanStep(
                                step_order=1,
                                target_agent=validated_plan.primary_agent,
                                sub_query=message,
                                rationale=validated_plan.reasoning,
                            )
                        ]
                    return validated_plan
            except Exception as exc:
                logger.warning(
                    "router_llm_validation_or_call_failed",
                    trace_id=tracer.trace_id if tracer else "",
                    error=str(exc),
                )

        # 2. Planejador Semântico Multi-Intenção Determinístico (Validado por Pydantic)
        plan = self._semantic_decompose_and_plan(message, user_id, conversation_context)
        if tracer:
            tracer.record_tokens(prompt_text=message, completion_text=plan.model_dump_json())
        return plan

    def route(self, message: str, user_id: str) -> Dict[str, Any]:
        """
        Método de compatibilidade que retorna o dicionário validado do `RoutingPlan`.
        """
        plan = self.plan(message=message, user_id=user_id)
        return {
            "target_agent": plan.primary_agent,
            "confidence": plan.confidence,
            "reasoning": plan.reasoning,
            "requires_user_context": plan.requires_user_context,
            "is_chained": plan.is_multi_step,
            "plan_steps": [s.model_dump() for s in plan.plan_steps],
        }

    def _semantic_decompose_and_plan(
        self,
        message: str,
        user_id: str,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> RoutingPlan:
        norm = BilingualDenseEmbedder.normalize(message)
        lower = message.lower()

        # Detecta sub-intenções presentes na mensagem (suporta perguntas simples e compostas)
        detected_steps: List[PlanStep] = []

        # Sinais de Suporte Operacional / Conta / Telemetria POS
        support_signals = [
            "vendas de ontem", "yesterday s sales", "yesterday sales", "meu dinheiro", "deposited",
            "cair o dinheiro", "quando cai", "liquidacao", "meu extrato", "meu saldo",
            "minha maquininha", "minha maquina", "my card machine", "nao conecta", "won t connect",
            "wont connect", "sem internet", "sem sinal", "transacao recusada", "transaction decline",
            "decline error", "erro de transacao", "codigo 05", "codigo 51", "codigo 12", "codigo 91",
            "meu terminal", "meu cadastro", "meu perfil", "reiniciar maquina",
        ]
        # Nota: se o usuário diz "Minha maquininha..." ou "My card machine..." é suporte operacional da máquina dele;
        # mas se pergunta "Qual a diferença entre a Get Clássica e a Get Smart?", é catálogo/RAG.
        is_catalog_comparison = ("diferenca" in norm or "difference" in norm) and ("classica" in norm or "smart" in norm)
        has_support = (not is_catalog_comparison) and any(s in norm or s in lower for s in support_signals)

        # Sinais de Web Search Externo (Clima, Câmbio, Fatos Gerais fora da Getnet)
        web_signals = [
            "weather", "previsao do tempo", "clima", "temperatura", "porto alegre", "chover",
            "euro", "exchange rate", "cotacao", "dolar", "usd", "brl", "moeda hoje",
            "quem descobriu", "capital da", "wikipedia",
        ]
        has_web = any(s in norm or s in lower for s in web_signals)

        # Sinais de Escalação Humana / HITL
        human_signals = [
            "falar com humano", "atendente humano", "falar com pessoa", "falar com atendente",
            "quero um atendente", "transferir para humano", "especialista humano", "ouvidoria",
            "abrir chamado humano", "human agent", "talk to a human", "human operator",
        ]
        has_human = any(s in norm or s in lower for s in human_signals)
        # Caso curto explícito ("quero falar com humano" / "atendente")
        if not has_human and any(w in norm.split() for w in ["humano", "atendente", "ouvidoria"]):
            has_human = True

        # Sinais de Conhecimento Getnet (RAG)
        rag_signals = [
            "get classica", "get smart", "get tap", "pix", "conta bancaria", "bank account",
            "get conta", "antecipacao", "recebiveis", "receivables", "advance", "crediario",
            "parcela", "installments", "whatsapp", "link de pagamento", "payment link",
            "taxa", "bandeiras", "aluguel", "gravame", "receba ja",
        ]
        has_rag = any(s in norm or s in lower for s in rag_signals)

        # Resolução de co-referência multi-turno (se a pergunta for curta de continuação, ex: "E qual o prazo disso?")
        if not (has_support or has_web or has_human or has_rag) and conversation_context:
            last_agent = conversation_context.get("last_agent", "")
            if "CUSTOMER_SUPPORT" in last_agent and any(w in norm for w in ["ela", "maquina", "venda", "conta", "terminal", "codigo"]):
                has_support = True
            elif "WEB_SEARCH" in last_agent and any(w in norm for w in ["amanha", "hoje", "dolar", "euro", "clima", "tempo"]):
                has_web = True

        step_order = 1
        if has_web:
            detected_steps.append(
                PlanStep(
                    step_order=step_order,
                    target_agent="WEB_SEARCH_AGENT",
                    sub_query=message,
                    rationale="Consulta requer busca externa em tempo real (clima, câmbio ou informações públicas).",
                )
            )
            step_order += 1

        if has_support:
            detected_steps.append(
                PlanStep(
                    step_order=step_order,
                    target_agent="CUSTOMER_SUPPORT_AGENT",
                    sub_query=message,
                    rationale="Consulta operacional requer inspeção de dados autenticados de vendas ou telemetria de terminal POS.",
                )
            )
            step_order += 1

        if has_rag and not (has_support and not is_catalog_comparison):
            detected_steps.append(
                PlanStep(
                    step_order=step_order,
                    target_agent="KNOWLEDGE_AGENT",
                    sub_query=message,
                    rationale="Consulta sobre produtos, regras de negócio e documentação oficial da Getnet via RAG Híbrido.",
                )
            )
            step_order += 1

        if has_human:
            detected_steps.append(
                PlanStep(
                    step_order=step_order,
                    target_agent="HUMAN_ESCALATION_AGENT",
                    sub_query=message,
                    rationale="Solicitação de transbordo para especialista humano na fila HITL com contexto da sessão.",
                )
            )
            step_order += 1

        # Se nenhum sinal específico disparou, encaminha para o Knowledge Agent (que aplicará threshold de abstention)
        if not detected_steps:
            detected_steps.append(
                PlanStep(
                    step_order=1,
                    target_agent="KNOWLEDGE_AGENT",
                    sub_query=message,
                    rationale="Roteamento padrão para verificação de cobertura na base oficial de conhecimento Getnet.",
                )
            )

        primary: AgentTypeLiteral = detected_steps[0].target_agent
        is_multi = len(detected_steps) > 1
        requires_ctx = any(
            s.target_agent in ("CUSTOMER_SUPPORT_AGENT", "HUMAN_ESCALATION_AGENT") for s in detected_steps
        )

        reasoning = (
            "Plano multi-agente composto: " + " -> ".join(s.target_agent for s in detected_steps)
            if is_multi
            else detected_steps[0].rationale
        )

        return RoutingPlan(
            primary_agent=primary,
            confidence=0.96 if (has_support or has_web or has_rag or has_human) else 0.72,
            reasoning=reasoning,
            plan_steps=detected_steps,
            requires_user_context=requires_ctx,
            is_multi_step=is_multi,
        )
