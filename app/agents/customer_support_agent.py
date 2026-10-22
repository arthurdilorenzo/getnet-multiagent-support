import json
from typing import Dict, Any, List, Optional
from app.config import settings
from app.tools.customer_tools import get_sales_settlement, diagnose_terminal_issue, get_customer_profile

SUPPORT_SYSTEM_PROMPT = """Você é o Customer Support Agent da Getnet, prestando atendimento técnico e operacional a clientes credenciados.
Sua missão:
1. Usar as ferramentas de sistema para buscar os dados reais da conta do cliente e o status de seus equipamentos.
2. Ser empático, objetivo, prestativo e tranquilizador.
3. Explicar valores monetários, datas de depósito ou instruções passo a passo para resolução de falhas técnicas.
4. NUNCA invente números ou status que não constem nos dados retornados pelas ferramentas.
"""

class CustomerSupportAgent:
    """
    Agente 3: Customer Support Agent.
    Atendimento autenticado com acesso a ferramentas operacionais (Tools) para consultar
    liquidação de vendas e diagnóstico de terminais POS.
    """
    def __init__(self, gemini_client: Optional[Any] = None):
        self.client = gemini_client

    def handle(self, message: str, user_id: str) -> Dict[str, Any]:
        tools_called: List[str] = []
        lower = message.lower()
        context_data: Dict[str, Any] = {}

        # 1. Decisão de Tools a acionar
        # Se for pergunta sobre vendas, depósito, dinheiro de ontem, saldo
        if any(w in lower for w in ["vendas", "ontem", "yesterday", "dinheiro", "deposit", "cai", "pagamento"]):
            tools_called.append("get_sales_settlement")
            settlement_res = get_sales_settlement(user_id)
            context_data["sales_settlement"] = settlement_res

        # Se for problema de conectividade, máquina sem internet ou recusa de transação
        if any(w in lower for w in ["não conecta", "won't connect", "internet", "sinal", "decline", "recusa", "erro", "máquina", "terminal"]):
            tools_called.append("diagnose_terminal_issue")
            diagnosis_res = diagnose_terminal_issue(user_id, error_context=message)
            context_data["terminal_diagnosis"] = diagnosis_res

        # Se nenhuma tool foi chamada por padrão, recupera o perfil
        if not tools_called:
            tools_called.append("get_customer_profile")
            profile_res = get_customer_profile(user_id)
            context_data["customer_profile"] = profile_res

        # 2. Síntese via Gemini se disponível
        if self.client and settings.GEMINI_API_KEY:
            try:
                prompt = (
                    f"Cliente ID: {user_id}\n"
                    f"Mensagem do cliente: \"{message}\"\n\n"
                    f"Dados extraídos pelas ferramentas operacionais:\n"
                    f"{json.dumps(context_data, ensure_ascii=False, indent=2)}\n\n"
                    f"Responda ao cliente com base nestes dados precisos."
                )
                response = self.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config={"system_instruction": SUPPORT_SYSTEM_PROMPT}
                )
                if response and response.text:
                    return {
                        "response": response.text.strip(),
                        "tools_called": tools_called,
                        "data_used": context_data
                    }
            except Exception as e:
                print(f"[CustomerSupportAgent] Síntese Gemini indisponível, ativando síntese local: {e}")

        # 3. Síntese determinística de alta fidelidade
        return self._synthesize_local(message, user_id, context_data, tools_called)

    def _synthesize_local(self, message: str, user_id: str, data: Dict[str, Any], tools_called: List[str]) -> Dict[str, Any]:
        lower = message.lower()

        # Cenário 3: Quando cai o dinheiro das vendas de ontem?
        if "sales_settlement" in data:
            st = data["sales_settlement"]
            if not st.get("found"):
                resp = f"Olá! Não localizamos cadastro ativo para o identificador '{user_id}'. Por favor, verifique seus dados de acesso."
            else:
                sales = st.get("sales_data", {})
                company = st.get("company_name", "Cliente")
                resp = (
                    f"Olá, {company}!\n\n"
                    f"As suas vendas realizadas ontem ({sales.get('date')}) totalizaram R$ {sales.get('total_gross_brl', 0):,.2f}.\n\n"
                    f"• **Status do Depósito:** {sales.get('payout_status')}.\n"
                    f"• **Valor Líquido a Receber:** R$ {sales.get('net_payout_scheduled_today_brl', 0):,.2f} "
                    f"(referente às vendas de débito e crédito com antecipação automática ativada).\n"
                    f"• **Conta de Destino:** {sales.get('target_bank')}.\n"
                    f"• **Pix:** {sales.get('pix_note')}"
                )
            return {"response": resp, "tools_called": tools_called, "data_used": data}

        # Cenário 5: Maquininha não conecta na internet
        if any(w in lower for w in ["não conecta", "won't connect", "internet", "sinal"]):
            diag = data.get("terminal_diagnosis", {})
            term = diag.get("terminal", {})
            steps = diag.get("recommended_troubleshooting", {}).get("connectivity", [])
            term_model = term.get("model", "Getnet")
            term_id = term.get("terminal_id", "")
            signal = term.get("signal_strength", "Instável")
            steps_formatted = "\n".join(steps)

            resp = (
                f"Identificamos a sua máquina {term_model} (ID: {term_id}). "
                f"No momento, o diagnóstico acusa sinal de rede {signal}.\n\n"
                f"Para restabelecer a conexão agora, siga estes passos rápidos:\n"
                f"{steps_formatted}\n\n"
                f"Caso após o teste de comunicação o sinal não retorne, nossa equipe técnica pode solicitar a troca remota de operadora do chip."
            )
            return {"response": resp, "tools_called": tools_called, "data_used": data}

        # Cenário 8: Erro de transação recusada (Transaction decline error)
        if any(w in lower for w in ["decline", "recusa", "recusada", "erro"]):
            diag = data.get("terminal_diagnosis", {})
            last_txn = diag.get("last_declined_transaction")
            if last_txn:
                resp = (
                    f"Analisamos o registro recente do seu terminal e identificamos uma tentativa de venda recusada às {last_txn.get('timestamp')}:\n\n"
                    f"• **Valor:** R$ {last_txn.get('amount_brl', 0):,.2f} no {last_txn.get('card_brand')}\n"
                    f"• **Código do Erro:** Código {last_txn.get('decline_code')}\n"
                    f"• **Motivo:** {last_txn.get('decline_reason')}\n\n"
                    f"**Orientação para o cliente:** {last_txn.get('action_required')} "
                    f"Esse tipo de recusa é emitido diretamente pelo banco emissor do comprador por segurança, não sendo uma falha da sua maquininha Getnet."
                )
            else:
                resp = (
                    "Quando ocorre uma recusa de transação, verifique o código numérico no comprovante impresso ou no visor da máquina:\n"
                    "• **Código 51:** Saldo ou limite insuficiente no cartão do comprador.\n"
                    "• **Código 05:** Bloqueio preventivo pelo banco emissor do cartão.\n"
                    "• **Código 12:** Transação inválida (oriente o cliente a inserir o chip e digitar a senha).\n"
                    "• **Código 91/96:** Instabilidade temporária de comunicação com a bandeira."
                )
            return {"response": resp, "tools_called": tools_called, "data_used": data}

        # Resposta padrão
        resp = f"Atendimento Getnet localizado para o cliente {user_id}. Como podemos ajudar com seus terminais ou vendas hoje?"
        return {"response": resp, "tools_called": tools_called, "data_used": data}
