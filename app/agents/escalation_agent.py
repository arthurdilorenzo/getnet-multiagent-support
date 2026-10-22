import time
from typing import Dict, Any, List

class EscalationAgent:
    """
    Agente 4 (Bônus): Human Escalation & Handoff Agent.
    Acionado quando o usuário solicita atendimento humano, quando os guardrails detectam
    situações atípicas ou quando a IA reconhece que não possui alçada para resolver a questão.
    Gera protocolo oficial de atendimento e simula encaminhamento para a equipe de FDE / Suporte Nível 2.
    """
    def handle(self, message: str, user_id: str, reason: str = "Solicitação explícita do cliente") -> Dict[str, Any]:
        protocol_number = f"GET-ESC-{int(time.time())}"
        
        response_text = (
            f"Compreendido. Estamos transferindo o seu atendimento para um especialista humano da Getnet.\n\n"
            f"• **Protocolo de Atendimento:** `{protocol_number}`\n"
            f"• **Motivo do Transbordo:** {reason}\n"
            f"• **Identificador da Conta:** `{user_id}`\n\n"
            f"Um de nossos analistas sênior de suporte operacional assumirá esta conversa nos próximos instantes. "
            f"O histórico completo desta interação e os diagnósticos prévios dos equipamentos já foram anexados ao chamado."
        )

        return {
            "response": response_text,
            "escalated_to_human": True,
            "protocol_number": protocol_number,
            "tools_called": ["ticket_dispatch_tool", "crm_human_handoff"],
            "queue": "FDE Operations & Tier-2 Support"
        }
