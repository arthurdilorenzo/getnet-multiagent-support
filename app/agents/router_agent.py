import json
import re
from typing import Dict, Any, Optional
from app.config import settings

ROUTER_SYSTEM_PROMPT = """Você é o Router Agent (Agente de Roteamento) do sistema multi-agente de suporte da Getnet.
Sua função única é analisar a mensagem do usuário e determinar qual agente especialista deve assumir a requisição.

Os agentes disponíveis são:
1. "KNOWLEDGE_AGENT" -> Para dúvidas conceituais, comerciais e informativas sobre produtos e serviços da Getnet (Get Clássica vs Get Smart, Pix, Antecipação de Recebíveis, Crediário, Link de Pagamento, taxas, etc.).
2. "WEB_SEARCH_AGENT" -> Para perguntas gerais que não são sobre a Getnet e necessitam de pesquisa externa (ex: previsão do tempo, cotação de moedas/euro/dólar, notícias).
3. "CUSTOMER_SUPPORT_AGENT" -> Para solicitações operacionais ligadas à conta do cliente ou aos seus equipamentos específicos (ex: quando cai o dinheiro das vendas de ontem, maquininha sem internet/problema de conexão, erro de transação recusada na máquina, extrato da conta).
4. "HUMAN_ESCALATION_AGENT" -> Para situações onde o usuário pede expressamente para falar com um humano, atendente, ou manifesta insatisfação crítica com a IA.

Responda ESTRITAMENTE em formato JSON com o seguinte schema:
{
  "target_agent": "KNOWLEDGE_AGENT | WEB_SEARCH_AGENT | CUSTOMER_SUPPORT_AGENT | HUMAN_ESCALATION_AGENT",
  "confidence": 0.95,
  "reasoning": "Breve justificativa da classificação em 1 linha",
  "requires_user_context": true/false
}
"""

class RouterAgent:
    """
    Agente 1: Ponto de entrada do sistema multi-agente.
    Analisa a mensagem recebida e decide o agente especialista de destino.
    """
    def __init__(self, gemini_client: Optional[Any] = None):
        self.client = gemini_client

    def route(self, message: str, user_id: str) -> Dict[str, Any]:
        """
        Executa a análise de intenção e retorna a decisão de roteamento.
        Suporta tanto agentes individuais quanto sequências encadeadas (chaining).
        """
        lower_msg = message.lower()

        # 1. Checagem de intenção composta / encadeamento (Multi-Agent Chaining)
        has_human_intent = any(w in lower_msg for w in ["falar com humano", "atendente humano", "falar com pessoa", "humano", "atendente", "ouvidoria"])
        customer_support_triggers = [
            "vendas de ontem", "yesterday's sales", "meu dinheiro", "deposited", "cair o dinheiro",
            "não conecta", "won't connect", "sem internet", "conexão", "minha maquininha", "minha máquina",
            "transação recusada", "transaction decline", "erro de transação", "decline error", "saldo", "extrato"
        ]
        has_support_intent = any(t in lower_msg for t in customer_support_triggers)

        if has_human_intent and has_support_intent:
            return {
                "target_agent": "CUSTOMER_SUPPORT_AGENT",
                "next_agent": "HUMAN_ESCALATION_AGENT",
                "is_chained": True,
                "confidence": 0.98,
                "reasoning": "Intenção composta identificada: execução de diagnóstico operacional preliminar seguido de escalação humana com contexto enriquecido.",
                "requires_user_context": True
            }

        # 2. Checagem de transbordo humano explícito e puro
        if has_human_intent:
            return {
                "target_agent": "HUMAN_ESCALATION_AGENT",
                "confidence": 0.99,
                "reasoning": "Usuário solicitou expressamente atendimento por operador humano.",
                "requires_user_context": True
            }

        # 3. Tenta classificação com Gemini se disponível
        if self.client and settings.GEMINI_API_KEY:
            try:
                response = self.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=f"Analise a seguinte mensagem do usuário (ID: {user_id}):\n\"{message}\"\n\nClassifique conforme suas instruções.",
                    config={"system_instruction": ROUTER_SYSTEM_PROMPT, "response_mime_type": "application/json"}
                )
                if response and response.text:
                    parsed = json.loads(response.text)
                    return parsed
            except Exception as e:
                print(f"[RouterAgent] Fallback semântico ativado devido a erro na API Gemini: {e}")

        # 3. Classificação Semântica Heurística (Robustez extrema)
        return self._heuristic_route(message, user_id)

    def _heuristic_route(self, message: str, user_id: str) -> Dict[str, Any]:
        lower = message.lower()

        # Cenários de Suporte ao Cliente (dados do cliente ou problemas técnicos na máquina)
        customer_support_triggers = [
            "vendas de ontem", "yesterday's sales", "meu dinheiro", "deposited", "cair o dinheiro",
            "não conecta", "won't connect", "sem internet", "conexão", "minha maquininha", "minha máquina",
            "transação recusada", "transaction decline", "erro de transação", "decline error", "saldo", "extrato"
        ]
        if any(t in lower for t in customer_support_triggers):
            return {
                "target_agent": "CUSTOMER_SUPPORT_AGENT",
                "confidence": 0.96,
                "reasoning": "Consulta operacional do cliente envolvendo liquidação de vendas ou diagnóstico de terminal/transação.",
                "requires_user_context": True
            }

        # Cenários de Web Search (perguntas externas)
        web_search_triggers = [
            "weather", "previsão do tempo", "clima", "porto alegre", "tomorrow",
            "euro", "exchange rate", "cotação", "dólar", "moeda hoje"
        ]
        if any(t in lower for t in web_search_triggers):
            return {
                "target_agent": "WEB_SEARCH_AGENT",
                "confidence": 0.95,
                "reasoning": "Consulta informativa externa geral (clima/cotação de câmbio) necessitando de Web Search.",
                "requires_user_context": False
            }

        # Cenários de Base de Conhecimento Getnet (RAG)
        return {
            "target_agent": "KNOWLEDGE_AGENT",
            "confidence": 0.94,
            "reasoning": "Consulta informacional e institucional sobre produtos, serviços e regras de negócio Getnet.",
            "requires_user_context": False
        }
