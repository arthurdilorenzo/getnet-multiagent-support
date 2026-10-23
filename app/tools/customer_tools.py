import hashlib
import json
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from app.config import settings
from app.hitl.queue import hitl_queue
from app.memory.store import state_store
from app.observability.logger import logger
from app.security.pii import PIIRedactor
from app.tools.registry import tool_registry


# ==============================================================================
# Pydantic Tool Input Schemas
# ==============================================================================
class GetSalesSettlementInput(BaseModel):
    user_id: str = Field(..., description="ID autenticado do cliente")
    payment_modality: Optional[Literal["all", "debit", "credit", "pix"]] = Field(
        default="all", description="Filtro opcional de modalidade (all, debit, credit, pix)"
    )


class DiagnoseTerminalIssueInput(BaseModel):
    user_id: str = Field(..., description="ID autenticado do cliente")
    terminal_id: Optional[str] = Field(
        default=None, description="ID específico do terminal POS (ex: GET-SMART-88219)"
    )
    terminal_model: Optional[str] = Field(
        default=None, description="Modelo do terminal caso especificado (ex: Get Smart, Get Clássica)"
    )
    symptom: Literal["connectivity", "decline", "hardware", "general"] = Field(
        default="general", description="Categoria do problema relatado (connectivity, decline, hardware, general)"
    )
    decline_code: Optional[str] = Field(
        default=None, description="Código numérico ISO-8583 de recusa se mencionado (ex: 05, 51, 12, 91)"
    )
    error_context: str = Field(
        default="", description="Descrição textual original do problema relatado pelo cliente"
    )


class GetCustomerProfileInput(BaseModel):
    user_id: str = Field(..., description="ID autenticado do cliente")


class ExecuteTerminalRemoteActionInput(BaseModel):
    user_id: str = Field(..., description="ID autenticado do cliente")
    terminal_id: str = Field(..., description="ID do terminal alvo da ação remota")
    action_type: Literal["remote_reboot", "sim_carrier_switch", "parameter_table_update"] = Field(
        ..., description="Tipo de comando remoto a enviar ao terminal"
    )
    reason: str = Field(default="Diagnóstico automatizado de suporte", description="Justificativa técnica")
    idempotency_key: str = Field(
        default="", description="Chave única de idempotência para evitar execução duplicada"
    )


class CreateEscalationTicketInput(BaseModel):
    user_id: str = Field(..., description="ID autenticado do cliente")
    session_id: str = Field(default="", description="ID da sessão ativa")
    category: str = Field(
        default="operational_support", description="Categoria do chamado (hardware_fault, risk_release, human_request, etc.)"
    )
    reason: str = Field(..., description="Resumo técnico do motivo de escalonamento e contexto")
    priority: Literal["low", "medium", "high", "critical"] = Field(
        default="high", description="Prioridade operacional do chamado"
    )
    diagnostic_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Snapshot estruturado do diagnóstico prévio"
    )
    idempotency_key: str = Field(
        default="", description="Chave de idempotência para evitar abertura duplicada de chamado"
    )


class CheckEscalationTicketInput(BaseModel):
    user_id: str = Field(..., description="ID autenticado do cliente")
    ticket_id: Optional[str] = Field(default=None, description="ID do protocolo GET-ESC-... se conhecido")


# ==============================================================================
# Helpers & Database Loader
# ==============================================================================
ISO8583_DECLINE_CATALOG: Dict[str, Dict[str, str]] = {
    "05": {
        "code": "05",
        "name": "Não Autorizada pelo Emissor (Do Not Honor)",
        "cause": "Bloqueio preventivo de segurança pelo banco emissor do cartão do comprador.",
        "action": "Orientar o comprador a liberar a compra no aplicativo do banco emissor ou utilizar outro cartão / Pix.",
    },
    "51": {
        "code": "51",
        "name": "Saldo ou Limite Insuficiente (Insufficient Funds)",
        "cause": "O cartão utilizado pelo comprador não possui saldo ou limite disponível suficiente.",
        "action": "Solicitar outro cartão ao comprador, dividir o pagamento ou cobrar via QR Code Pix.",
    },
    "12": {
        "code": "12",
        "name": "Transação Inválida (Invalid Transaction / NFC Limit)",
        "cause": "Tentativa de pagamento por aproximação (NFC) acima do limite sem senha ou modalidade incompatível.",
        "action": "Solicitar que o cliente insira o cartão físico no leitor de chip e digite a senha.",
    },
    "54": {
        "code": "54",
        "name": "Cartão Vencido (Expired Card)",
        "cause": "O cartão do comprador está com a data de validade expirada.",
        "action": "Solicitar outro cartão dentro do prazo de validade.",
    },
    "57": {
        "code": "57",
        "name": "Transação Não Permitida para o Cartão",
        "cause": "Cartão bloqueado para a função selecionada (ex: crédito em cartão puramente débito/voucher).",
        "action": "Conferir a função correta do cartão (Débito, Crédito ou Voucher).",
    },
    "91": {
        "code": "91",
        "name": "Emissor Fora do Ar (Issuer or Switch Inoperative)",
        "cause": "Instabilidade momentânea no sistema autorizador do banco emissor do comprador.",
        "action": "Aguardar 2 minutos e repetir a transação ou utilizar outra bandeira / Pix.",
    },
}


def _load_customers_db() -> Dict[str, Any]:
    file_path = settings.MOCK_CUSTOMERS_PATH
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("customers", {})
    except Exception as exc:
        logger.error("customer_db_load_failed", error=str(exc))
        return {}


# ==============================================================================
# Tool Implementations
# ==============================================================================
def get_sales_settlement(
    user_id: str,
    payment_modality: Optional[str] = "all",
) -> Dict[str, Any]:
    """
    Tool 1 (Read): Consulta status de liquidação financeira, repasses D+1/D+0 e vendas de ontem.
    """
    customers = _load_customers_db()
    customer = customers.get(user_id)
    if not customer:
        return {
            "found": False,
            "user_id": user_id,
            "error": f"Cliente com identificador '{user_id}' não localizado na base cadastral Getnet.",
        }

    settlement = customer.get("sales_settlement", {}).get("yesterday_sales", {})
    bank = customer.get("domicilio_bancario", {})
    masked_bank = PIIRedactor.mask_bank_account(
        str(bank.get("agencia", "")), str(bank.get("conta_corrente", ""))
    )
    target_bank_masked = (
        f"{bank.get('banco', 'Domicílio Cadastrado')} "
        f"Agência {masked_bank['agencia']} Conta {masked_bank['conta_corrente']}"
    )

    return {
        "found": True,
        "user_id": user_id,
        "company_name": customer.get("company_name"),
        "account_plan": customer.get("account_plan"),
        "antecipacao_automatica": customer.get("antecipacao_automatica"),
        "payment_modality_filter": payment_modality or "all",
        "sales_data": {
            "date": settlement.get("date"),
            "settlement_batch_id": settlement.get("settlement_batch_id", "LOTE-PADRAO"),
            "total_gross_brl": settlement.get("total_gross_brl"),
            "debit_brl": settlement.get("debit_brl"),
            "credit_brl": settlement.get("credit_brl"),
            "pix_brl": settlement.get("pix_brl"),
            "net_payout_scheduled_today_brl": settlement.get("net_payout_scheduled_today_brl"),
            "payout_status": settlement.get("payout_status"),
            "target_bank": target_bank_masked,
            "pix_note": settlement.get("pix_note"),
        },
    }


def diagnose_terminal_issue(
    user_id: str,
    terminal_id: Optional[str] = None,
    terminal_model: Optional[str] = None,
    symptom: str = "general",
    decline_code: Optional[str] = None,
    error_context: str = "",
) -> Dict[str, Any]:
    """
    Tool 2 (Read): Diagnóstico técnico multi-terminal POS (conectividade, hardware e códigos ISO-8583).
    Seleciona dinamicamente o terminal específico por ID, modelo ou sintoma, retornando toda a frota.
    """
    customers = _load_customers_db()
    customer = customers.get(user_id)
    if not customer:
        return {
            "found": False,
            "user_id": user_id,
            "error": f"Cliente '{user_id}' não localizado no sistema de telemetria de terminais.",
        }

    terminals: List[Dict[str, Any]] = customer.get("terminals", [])
    recent_transactions: List[Dict[str, Any]] = customer.get("recent_transactions", [])
    ctx_lower = f"{error_context} {terminal_model or ''} {terminal_id or ''} {decline_code or ''}".lower()

    # Inferência automática do sintoma caso symptom == 'general'
    effective_symptom = symptom
    if effective_symptom == "general":
        if any(w in ctx_lower for w in ["decline", "recusa", "recusada", "transação", "transaction", "código", "code", "05", "51", "12", "91"]):
            effective_symptom = "decline"
        elif any(w in ctx_lower for w in ["conecta", "connect", "internet", "sinal", "wifi", "wi-fi", "chip", "rede", "offline"]):
            effective_symptom = "connectivity"
        elif any(w in ctx_lower for w in ["leitor", "defeito", "quebr", "hardware", "troca"]):
            effective_symptom = "hardware"

    # Filtra transações recusadas relevantes
    declined_txns = [t for t in recent_transactions if t.get("status") == "RECUSADA"]
    selected_declined_txn: Optional[Dict[str, Any]] = None

    # Se um código específico foi mencionado (ex: 51, 05, 12, 91), prioriza a transação com esse código
    extracted_code = decline_code
    if not extracted_code:
        for code_candidate in ("05", "51", "12", "54", "57", "91"):
            if f" {code_candidate}" in f" {ctx_lower}" or f"código {code_candidate}" in ctx_lower or f"code {code_candidate}" in ctx_lower:
                extracted_code = code_candidate
                break

    if extracted_code:
        for t in declined_txns:
            if t.get("decline_code") == extracted_code:
                selected_declined_txn = t
                break

    # 1. Seleção inteligente do terminal alvo (não apenas terminals[0] fixo!)
    selected_terminal: Optional[Dict[str, Any]] = None
    if terminal_id:
        for term in terminals:
            if term.get("terminal_id", "").lower() == terminal_id.strip().lower():
                selected_terminal = term
                break

    if not selected_terminal and (terminal_model or "clássica" in ctx_lower or "classica" in ctx_lower or "smart" in ctx_lower):
        for term in terminals:
            model_low = term.get("model", "").lower()
            if ("clássica" in ctx_lower or "classica" in ctx_lower) and ("clássica" in model_low or "classica" in model_low):
                selected_terminal = term
                break
            if "smart" in ctx_lower and "smart" in model_low:
                selected_terminal = term
                break

    # Se houve busca por transação recusada com terminal_id vinculado, seleciona o terminal dessa transação
    if not selected_terminal and selected_declined_txn and selected_declined_txn.get("terminal_id"):
        for term in terminals:
            if term.get("terminal_id") == selected_declined_txn.get("terminal_id"):
                selected_terminal = term
                break

    # Se o sintoma é conectividade ou hardware, prioriza o terminal que apresenta anomalia na telemetria
    if not selected_terminal and terminals:
        for term in terminals:
            status_low = f"{term.get('status', '')} {term.get('signal_strength', '')}".lower()
            if any(flag in status_low for flag in ("instável", "sem sinal", "offline", "falha")) or term.get("requires_replacement"):
                selected_terminal = term
                break
        if not selected_terminal:
            selected_terminal = terminals[0]

    if not selected_declined_txn and declined_txns:
        # Prioriza recusa do terminal selecionado, ou a mais recente
        if selected_terminal:
            for t in declined_txns:
                if t.get("terminal_id") == selected_terminal.get("terminal_id"):
                    selected_declined_txn = t
                    break
        if not selected_declined_txn:
            selected_declined_txn = declined_txns[0]

    catalog_entry = ISO8583_DECLINE_CATALOG.get(
        extracted_code or (selected_declined_txn.get("decline_code") if selected_declined_txn else ""),
        {},
    )

    requires_replacement = bool(selected_terminal and selected_terminal.get("requires_replacement"))

    return {
        "found": True,
        "user_id": user_id,
        "company_name": customer.get("company_name"),
        "symptom_classified": effective_symptom,
        "terminals_count": len(terminals),
        "terminal": selected_terminal,
        "all_terminals": terminals,
        "last_declined_transaction": selected_declined_txn,
        "all_declined_transactions": declined_txns,
        "iso8583_catalog_reference": catalog_entry,
        "requires_hardware_replacement": requires_replacement,
        "recommended_troubleshooting": {
            "connectivity": [
                "1. Reinicie a maquininha segurando o botão vermelho (Desligar/Cancelar) por 5 segundos.",
                "2. Com a máquina desligada, verifique na tampa traseira se o Chip multi-operadora (Algar/Claro/Vivo) está bem encaixado no slot SIM1.",
                "3. Acesse Menu > Configurações > Conectividade e alterne entre rede móvel (3G/4G) e uma rede Wi-Fi local estável.",
                "4. Execute o Teste de Comunicação em Menu > Funções / Suporte > Teste de Comunicação para sincronizar com a Getnet.",
            ],
            "decline_resolution": (
                f"Código {selected_declined_txn.get('decline_code')}: {selected_declined_txn.get('decline_reason')} "
                f"Ação recomendada: {selected_declined_txn.get('action_required')}"
                if selected_declined_txn
                else (
                    f"Código {catalog_entry.get('code')} ({catalog_entry.get('name')}): {catalog_entry.get('action')}"
                    if catalog_entry
                    else "Verifique o código numérico ISO-8583 impresso no comprovante (ex: 05 Não Autorizada pelo Emissor, 51 Saldo Insuficiente, 12 Transação Inválida, 91 Emissor Indisponível)."
                )
            ),
        },
    }


def get_customer_profile(user_id: str) -> Dict[str, Any]:
    """
    Tool 3 (Read): Retorna o perfil cadastral do estabelecimento com PII mascarado (CNPJ e Conta).
    """
    customers = _load_customers_db()
    customer = customers.get(user_id)
    if not customer:
        return {"found": False, "user_id": user_id, "error": f"Cliente '{user_id}' não encontrado."}

    bank = customer.get("domicilio_bancario", {})
    masked_bank = PIIRedactor.mask_bank_account(
        str(bank.get("agencia", "")), str(bank.get("conta_corrente", ""))
    )

    return {
        "found": True,
        "user_id": user_id,
        "company_name": customer.get("company_name"),
        "cnpj": PIIRedactor.mask_cnpj(str(customer.get("cnpj", ""))),
        "contact_person": customer.get("contact_person"),
        "account_plan": customer.get("account_plan"),
        "antecipacao_automatica": customer.get("antecipacao_automatica"),
        "crediario_enabled": customer.get("crediario_enabled", True),
        "payment_link_enabled": customer.get("payment_link_enabled", True),
        "terminals_count": len(customer.get("terminals", [])),
        "terminals_summary": [
            {"terminal_id": t.get("terminal_id"), "model": t.get("model"), "status": t.get("status")}
            for t in customer.get("terminals", [])
        ],
        "domicilio_bancario": {
            "banco": bank.get("banco"),
            "agencia": masked_bank["agencia"],
            "conta_corrente": masked_bank["conta_corrente"],
            "tipo_chave_pix": bank.get("tipo_chave_pix"),
        },
    }


def execute_terminal_remote_action(
    user_id: str,
    terminal_id: str,
    action_type: str,
    reason: str = "Diagnóstico automatizado de suporte",
    idempotency_key: str = "",
) -> Dict[str, Any]:
    """
    Tool 4 (Write / Idempotent): Executa comando remoto de escrita no terminal POS do cliente
    (remote_reboot, sim_carrier_switch, parameter_table_update) com persistência SQLite e idempotência.
    """
    customers = _load_customers_db()
    customer = customers.get(user_id)
    if not customer:
        return {"success": False, "error": f"Cliente '{user_id}' não encontrado."}

    terminals = customer.get("terminals", [])
    target_term = next(
        (t for t in terminals if t.get("terminal_id", "").lower() == terminal_id.strip().lower()),
        terminals[0] if terminals else None,
    )
    if not target_term:
        return {"success": False, "error": f"Nenhum terminal localizado para o cliente '{user_id}'."}

    resolved_term_id = target_term.get("terminal_id", terminal_id)
    idem_key = idempotency_key.strip() or hashlib.sha256(
        f"{user_id}:{resolved_term_id}:{action_type}".encode("utf-8")
    ).hexdigest()[:20]

    # Verifica idempotência no SQLite
    existing = state_store.get_idempotent_result(idem_key, user_id)
    if existing:
        replay = dict(existing)
        replay["idempotent_replay"] = True
        return replay

    action_hash = hashlib.sha256(f"{idem_key}:{user_id}".encode("utf-8")).hexdigest()[:8].upper()
    action_id = f"ACT-{action_hash}"
    details = {
        "terminal_model": target_term.get("model"),
        "reason": reason,
        "command_dispatched": action_type,
    }
    record = state_store.record_terminal_action(
        action_id=action_id,
        user_id=user_id,
        terminal_id=resolved_term_id,
        action_type=action_type,
        status="SCHEDULED_ON_TERMINAL",
        idempotency_key=idem_key,
        details=details,
    )
    record["idempotent_replay"] = False
    state_store.save_idempotent_result(
        idempotency_key=idem_key,
        tool_name="execute_terminal_remote_action",
        user_id=user_id,
        result=record,
    )
    return record


def create_escalation_ticket(
    user_id: str,
    session_id: str = "",
    category: str = "operational_support",
    reason: str = "Solicitação de atendimento especializado",
    priority: str = "high",
    diagnostic_snapshot: Optional[Dict[str, Any]] = None,
    idempotency_key: str = "",
) -> Dict[str, Any]:
    """
    Tool 5 (Write / HITL): Abre protocolo real persistido na fila SQLite de Human-in-the-Loop
    e aciona notificação externa (Telegram / Webhook) quando configurada.
    """
    eff_session = session_id or f"sess_{user_id}"
    ticket = hitl_queue.create_ticket(
        user_id=user_id,
        session_id=eff_session,
        category=category,
        reason=reason,
        priority=priority,
        diagnostic_snapshot=diagnostic_snapshot or {},
        idempotency_key=idempotency_key or None,
    )
    return ticket.model_dump()


def check_escalation_ticket(
    user_id: str,
    ticket_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool 6 (Read / HITL): Consulta o status real e respostas de operadores humanos na fila SQLite.
    """
    if ticket_id:
        t = hitl_queue.get_ticket(ticket_id.strip())
        if not t or t.user_id != user_id:
            return {"found": False, "error": f"Chamado '{ticket_id}' não encontrado para o cliente '{user_id}'."}
        return {"found": True, "ticket": t.model_dump()}

    tickets = hitl_queue.list_tickets(user_id=user_id)
    return {
        "found": len(tickets) > 0,
        "tickets_count": len(tickets),
        "tickets": [t.model_dump() for t in tickets[:5]],
    }


# ==============================================================================
# Register Customer & HITL Tools in Global ToolRegistry
# ==============================================================================
def register_customer_and_hitl_tools() -> None:
    tool_registry.register(
        name="get_sales_settlement",
        description="Consulta liquidação de vendas de ontem, valores brutos/líquidos de débito, crédito e Pix e status de depósito bancário do lojista.",
        args_schema=GetSalesSettlementInput,
        handler=get_sales_settlement,
        requires_tenant_context=True,
        is_write_action=False,
    )
    tool_registry.register(
        name="diagnose_terminal_issue",
        description="Executa diagnóstico técnico de terminais POS (Get Smart / Get Clássica) para problemas de conexão com internet, sinal de chip/Wi-Fi, defeitos de hardware ou códigos de erro de transação recusada (05, 51, 12, 91).",
        args_schema=DiagnoseTerminalIssueInput,
        handler=diagnose_terminal_issue,
        requires_tenant_context=True,
        is_write_action=False,
    )
    tool_registry.register(
        name="get_customer_profile",
        description="Consulta dados cadastrais do lojista, plano contratado, status de antecipação automática, quantidade de máquinas e domicílio bancário.",
        args_schema=GetCustomerProfileInput,
        handler=get_customer_profile,
        requires_tenant_context=True,
        is_write_action=False,
    )
    tool_registry.register(
        name="execute_terminal_remote_action",
        description="Envia comando remoto idempotente para o terminal POS do cliente (remote_reboot, sim_carrier_switch ou parameter_table_update) e persiste a ação no banco.",
        args_schema=ExecuteTerminalRemoteActionInput,
        handler=execute_terminal_remote_action,
        requires_tenant_context=True,
        is_write_action=True,
    )
    tool_registry.register(
        name="create_escalation_ticket",
        description="Cria um protocolo persistido na fila Human-in-the-Loop (HITL) com snapshot de diagnóstico e histórico da sessão para atendimento por especialista humano.",
        args_schema=CreateEscalationTicketInput,
        handler=create_escalation_ticket,
        requires_tenant_context=True,
        is_write_action=True,
    )
    tool_registry.register(
        name="check_escalation_ticket",
        description="Consulta o status atual e eventuais respostas enviadas por operadores humanos nos chamados do cliente.",
        args_schema=CheckEscalationTicketInput,
        handler=check_escalation_ticket,
        requires_tenant_context=True,
        is_write_action=False,
    )


register_customer_and_hitl_tools()
