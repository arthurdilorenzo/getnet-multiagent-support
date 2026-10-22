import json
from pathlib import Path
from typing import Dict, Any, Optional
from app.config import settings

def _load_customers_db() -> Dict[str, Any]:
    file_path = settings.MOCK_CUSTOMERS_PATH
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("customers", {})
    except Exception as e:
        print(f"[CustomerTools] Erro ao carregar base de clientes: {e}")
        return {}

def get_sales_settlement(user_id: str) -> Dict[str, Any]:
    """
    Tool 1: Consulta o status de liquidação e depósito das vendas de ontem e Pix para o cliente.
    """
    customers = _load_customers_db()
    customer = customers.get(user_id)
    if not customer:
        return {
            "found": False,
            "error": f"Cliente com identificador '{user_id}' não localizado na base Getnet."
        }
    
    settlement = customer.get("sales_settlement", {}).get("yesterday_sales", {})
    bank = customer.get("domicilio_bancario", {})
    
    return {
        "found": True,
        "user_id": user_id,
        "company_name": customer.get("company_name"),
        "antecipacao_automatica": customer.get("antecipacao_automatica"),
        "sales_data": {
            "date": settlement.get("date"),
            "total_gross_brl": settlement.get("total_gross_brl"),
            "debit_brl": settlement.get("debit_brl"),
            "credit_brl": settlement.get("credit_brl"),
            "pix_brl": settlement.get("pix_brl"),
            "net_payout_scheduled_today_brl": settlement.get("net_payout_scheduled_today_brl"),
            "payout_status": settlement.get("payout_status"),
            "target_bank": settlement.get("target_bank") or f"{bank.get('banco')} Ag {bank.get('agencia')} CC {bank.get('conta_corrente')}",
            "pix_note": settlement.get("pix_note")
        }
    }

def diagnose_terminal_issue(user_id: str, error_context: str = "") -> Dict[str, Any]:
    """
    Tool 2: Diagnóstico técnico do terminal POS (maquininha) do cliente ou de erros de transação recusada.
    """
    customers = _load_customers_db()
    customer = customers.get(user_id)
    if not customer:
        return {
            "found": False,
            "error": f"Cliente '{user_id}' não localizado no sistema de monitoramento de terminais."
        }
    
    terminals = customer.get("terminals", [])
    primary_terminal = terminals[0] if terminals else None
    recent_transactions = customer.get("recent_transactions", [])
    
    # Verifica se há menção a declínio/recusa de transação
    is_decline_query = any(w in error_context.lower() for w in ["decline", "recusa", "recusada", "transação", "erro"])
    last_declined_txn = None
    if is_decline_query and recent_transactions:
        for txn in recent_transactions:
            if txn.get("status") == "RECUSADA":
                last_declined_txn = txn
                break

    return {
        "found": True,
        "user_id": user_id,
        "company_name": customer.get("company_name"),
        "terminal": primary_terminal,
        "last_declined_transaction": last_declined_txn,
        "recommended_troubleshooting": {
            "connectivity": [
                "1. Reinicie a maquininha segurando o botão vermelho por 5 segundos.",
                "2. Acesse Menu > Configurações > Conectividade e alterne entre Chip Móvel e Wi-Fi.",
                "3. Execute o Teste de Comunicação em Menu > Funções > Teste de Comunicação.",
                "4. Se o sinal continuar fraco no chip Claro/Vivo, reposicione a máquina ou conecte ao Wi-Fi local."
            ],
            "decline_resolution": (
                f"Código {last_declined_txn.get('decline_code')}: {last_declined_txn.get('decline_reason')}. "
                f"Ação: {last_declined_txn.get('action_required')}"
                if last_declined_txn else "Consulte o código de erro no comprovante da maquininha."
            )
        }
    }

def get_customer_profile(user_id: str) -> Dict[str, Any]:
    """
    Tool 3: Retorna o perfil cadastral do cliente na Getnet.
    """
    customers = _load_customers_db()
    customer = customers.get(user_id)
    if not customer:
        return {"found": False, "error": f"Cliente '{user_id}' não encontrado."}
    
    return {
        "found": True,
        "user_id": user_id,
        "company_name": customer.get("company_name"),
        "cnpj": customer.get("cnpj"),
        "contact_person": customer.get("contact_person"),
        "account_plan": customer.get("account_plan"),
        "antecipacao_automatica": customer.get("antecipacao_automatica"),
        "terminals_count": len(customer.get("terminals", [])),
        "domicilio_bancario": customer.get("domicilio_bancario")
    }
