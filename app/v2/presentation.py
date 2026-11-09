"""Server-side semantic presentation for verified operational facts.

Preserves 100% deterministic server-side rendering (the LLM never writes or formats
operational numbers, dates, currencies, statuses, or IDs) while presenting natural,
clear Portuguese sentences alongside the canonical audit reference.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.v2.contracts import Fact, OperationalBlock


@dataclass(frozen=True)
class FieldPresentation:
    label_pt: str
    available_template: str
    unavailable_template: str
    value_translations: Dict[str, str]


FIELD_REGISTRY: Dict[Tuple[str, str], FieldPresentation] = {
    ("customer", "/profile/status"): FieldPresentation(
        label_pt="Status cadastral do lojista",
        available_template="O cadastro do lojista {resource_id} está com status {human_value}",
        unavailable_template="Não foi possível confirmar o status cadastral do lojista {resource_id}",
        value_translations={"ACTIVE": "ativo", "ATIVO": "ativo", "BLOCKED": "bloqueado", "PENDING": "pendente"},
    ),
    ("customer", "/profile/segment"): FieldPresentation(
        label_pt="Segmento comercial",
        available_template="O lojista {resource_id} está classificado no segmento {human_value}",
        unavailable_template="O segmento comercial do lojista {resource_id} não consta no cadastro consultado",
        value_translations={},
    ),
    ("customer", "/sales_data/saldo_disponivel_brl"): FieldPresentation(
        label_pt="Saldo disponível para liquidação",
        available_template="O saldo disponível para liquidação da conta {resource_id} é de {money_brl}",
        unavailable_template="O saldo disponível para liquidação da conta {resource_id} não está disponível neste registro",
        value_translations={},
    ),
    ("customer", "/sales_data/net_payout_scheduled_today_brl"): FieldPresentation(
        label_pt="Valor líquido programado",
        available_template="O valor líquido programado para a conta {resource_id} é de {money_brl}",
        unavailable_template="Não há valor líquido programado disponível para a conta {resource_id}",
        value_translations={},
    ),
    ("customer", "/sales_data/net_amount_brl"): FieldPresentation(
        label_pt="Valor líquido apurado",
        available_template="O valor líquido apurado de vendas da conta {resource_id} é de {money_brl}",
        unavailable_template="O valor líquido apurado de vendas da conta {resource_id} não está disponível",
        value_translations={},
    ),
    ("customer", "/sales_data/gross_amount_brl"): FieldPresentation(
        label_pt="Volume bruto de vendas",
        available_template="O volume bruto de vendas apurado para a conta {resource_id} é de {money_brl}",
        unavailable_template="O volume bruto de vendas da conta {resource_id} não consta neste registro",
        value_translations={},
    ),
    ("customer", "/sales_data/date"): FieldPresentation(
        label_pt="Data de referência do registro de vendas",
        available_template="A data de referência do snapshot de vendas da conta {resource_id} é {human_value}",
        unavailable_template="A data de referência das vendas da conta {resource_id} não está disponível",
        value_translations={},
    ),
    ("terminal", "/terminal/status"): FieldPresentation(
        label_pt="Status operacional do terminal",
        available_template="A consulta técnica indica que o terminal {resource_id} encontra-se {human_value}",
        unavailable_template="Não há confirmação do status operacional atual do terminal {resource_id}",
        value_translations={
            "ONLINE": "online",
            "OFFLINE": "offline",
            "DEGRADED": "com conectividade degradada",
            "MAINTENANCE": "em manutenção",
        },
    ),
    ("terminal", "/terminal/connectivity"): FieldPresentation(
        label_pt="Conectividade do terminal",
        available_template="A conectividade configurada no terminal {resource_id} é {human_value}",
        unavailable_template="O tipo de conectividade do terminal {resource_id} não está disponível",
        value_translations={},
    ),
    ("terminal", "/terminal/model"): FieldPresentation(
        label_pt="Modelo do terminal",
        available_template="O modelo identificado para o terminal {resource_id} é {human_value}",
        unavailable_template="O modelo do terminal {resource_id} não consta no diagnóstico",
        value_translations={},
    ),
    ("terminal", "/ultimo_erro_transacao"): FieldPresentation(
        label_pt="Último código de erro de transação",
        available_template="O último erro de transação registrado no terminal {resource_id} foi {human_value}",
        unavailable_template="Não consta registro disponível para o último erro de transação do terminal {resource_id}",
        value_translations={},
    ),
    ("terminal", "/chip_algar_status"): FieldPresentation(
        label_pt="Status do chip da operadora",
        available_template="O status do chip de dados no terminal {resource_id} é {human_value}",
        unavailable_template="O status do chip de dados do terminal {resource_id} não está disponível",
        value_translations={},
    ),
    ("terminal", "/observacao_sistema"): FieldPresentation(
        label_pt="Observação técnica do sistema",
        available_template="Observação técnica registrada para o terminal {resource_id}: {human_value}",
        unavailable_template="Não há observação técnica registrada para o terminal {resource_id}",
        value_translations={},
    ),
    ("terminal", "/compartimento_detectado"): FieldPresentation(
        label_pt="Compartimento físico detectado",
        available_template="Diagnóstico físico do terminal {resource_id}: {human_value}",
        unavailable_template="Não há informação sobre compartimento físico para o terminal {resource_id}",
        value_translations={},
    ),
    ("terminal", "/proposal_id"): FieldPresentation(
        label_pt="Protocolo da proposta de ação remota",
        available_template="Foi gerada a proposta de ação remota {human_value} para o terminal {resource_id}",
        unavailable_template="Nenhum protocolo de proposta foi gerado para o terminal {resource_id}",
        value_translations={},
    ),
    ("terminal", "/status"): FieldPresentation(
        label_pt="Estado da proposta de ação remota",
        available_template="O estado atual da proposta para o terminal {resource_id} é {human_value}",
        unavailable_template="O estado da proposta para o terminal {resource_id} não está disponível",
        value_translations={
            "PENDING_CONFIRMATION": "aguardando confirmação explícita do titular",
        },
    ),
    ("terminal", "/action_type"): FieldPresentation(
        label_pt="Tipo de ação remota",
        available_template="A ação remota solicitada para o terminal {resource_id} é {human_value}",
        unavailable_template="O tipo de ação remota para o terminal {resource_id} não está disponível",
        value_translations={},
    ),
    ("ticket", "/ticket/status"): FieldPresentation(
        label_pt="Status do protocolo de atendimento humano",
        available_template="O protocolo de atendimento {resource_id} está com status {human_value}",
        unavailable_template="O status do protocolo {resource_id} não está disponível",
        value_translations={
            "OPEN": "aberto na fila de atendimento especializado",
            "CLAIMED": "em análise por um especialista humano",
            "RESPONDED": "respondido pelo especialista humano",
            "RESOLVED": "resolvido pelo especialista humano",
        },
    ),
    ("ticket", "/ticket/operator_response"): FieldPresentation(
        label_pt="Parecer do operador humano",
        available_template="Parecer registrado no protocolo {resource_id}: {human_value}",
        unavailable_template="Ainda não há parecer humano registrado no protocolo {resource_id}",
        value_translations={},
    ),
    ("weather", "/condition"): FieldPresentation(
        label_pt="Condição meteorológica prevista",
        available_template="A condição meteorológica prevista para {resource_id} é de {human_value}",
        unavailable_template="A condição meteorológica para {resource_id} não está disponível",
        value_translations={},
    ),
    ("weather", "/max_temp_celsius"): FieldPresentation(
        label_pt="Temperatura máxima prevista",
        available_template="A temperatura máxima prevista em {resource_id} é de {human_value}",
        unavailable_template="A temperatura máxima para {resource_id} não está disponível",
        value_translations={},
    ),
    ("weather", "/min_temp_celsius"): FieldPresentation(
        label_pt="Temperatura mínima prevista",
        available_template="A temperatura mínima prevista em {resource_id} é de {human_value}",
        unavailable_template="A temperatura mínima para {resource_id} não está disponível",
        value_translations={},
    ),
    ("weather", "/rain_probability_percent"): FieldPresentation(
        label_pt="Probabilidade de chuva",
        available_template="A probabilidade de chuva em {resource_id} é de {human_value}",
        unavailable_template="A probabilidade de chuva para {resource_id} não está disponível",
        value_translations={},
    ),
    ("currency", "/rate"): FieldPresentation(
        label_pt="Cotação verificada",
        available_template="A cotação consultada para {resource_id} é {human_value}",
        unavailable_template="A cotação para {resource_id} não está disponível no momento",
        value_translations={},
    ),
    ("currency", "/rate_type"): FieldPresentation(
        label_pt="Natureza da cotação",
        available_template="Natureza da cotação de {resource_id}: {human_value}",
        unavailable_template="A natureza da cotação de {resource_id} não está disponível",
        value_translations={},
    ),
}


def format_canonical_fact_value(fact: Fact) -> str:
    """Canonical server-verified representation preserving exact audit tokens."""
    field_name = fact.field_pointer.strip("/").split("/")[-1]
    if fact.availability != "available":
        return f"{field_name} ({fact.resource_id}): Informação indisponível."

    if fact.resource_type == "currency":
        return f"{field_name} ({fact.resource_id}): {fact.value} {fact.unit or ''}".rstrip()

    if fact.unit == "BRL":
        try:
            val_num = float(fact.value)
            formatted = f"{val_num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return f"{field_name} ({fact.resource_id}): BRL {formatted}"
        except (ValueError, TypeError):
            return f"{field_name} ({fact.resource_id}): BRL {fact.value}"

    if fact.unit:
        return f"{field_name} ({fact.resource_id}): {fact.value} {fact.unit}"

    return f"{field_name} ({fact.resource_id}): {fact.value}"


def _format_money_brl(value: object) -> str:
    try:
        val_num = float(value)
        formatted = f"{val_num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {formatted}"
    except (ValueError, TypeError):
        return f"R$ {value}"


def format_semantic_fact(fact: Fact) -> str:
    """Renders a natural Portuguese sentence paired with its canonical server-bound audit token."""
    canonical = format_canonical_fact_value(fact)
    field_name = fact.field_pointer.strip("/").split("/")[-1]
    human_label = field_name.replace("_", " ")
    spec = FIELD_REGISTRY.get((fact.resource_type, fact.field_pointer))

    if fact.availability != "available":
        if spec:
            sentence = spec.unavailable_template.format(resource_id=fact.resource_id)
        else:
            sentence = f"Não há dado disponível para {human_label} no recurso {fact.resource_id}"
        base = f"{sentence} [{canonical}]"
    else:
        raw_str = str(fact.value)
        if spec:
            translated = spec.value_translations.get(raw_str.upper(), raw_str)
            if fact.unit and fact.unit != "BRL" and not translated.endswith(str(fact.unit)):
                translated = f"{translated} {fact.unit}"
            sentence = spec.available_template.format(
                resource_id=fact.resource_id,
                human_value=translated,
                money_brl=_format_money_brl(fact.value),
            )
        else:
            if fact.unit == "BRL":
                val_display = _format_money_brl(fact.value)
            elif fact.unit:
                val_display = f"{fact.value} {fact.unit}"
            else:
                val_display = raw_str
            sentence = f"Registro verificado de {human_label} para {fact.resource_id}: {val_display}"
        base = f"{sentence}. [{canonical}]"

    if fact.scope_notice:
        return f"{base}\nObservação de escopo: {fact.scope_notice}"
    return base


def render_operational_block(block: OperationalBlock, facts_by_id: Dict[str, Fact]) -> Optional[str]:
    """Groups verified facts of an OperationalBlock into a coherent, natural server-rendered section."""
    presented_lines: List[str] = []
    seen_notices: List[str] = []

    for fid in block.fact_ids:
        fact = facts_by_id.get(fid)
        if not fact:
            continue
        line = format_semantic_fact(fact)
        if fact.scope_notice:
            base_line = line.split("\nObservação de escopo:", 1)[0]
            presented_lines.append(base_line)
            if fact.scope_notice not in seen_notices:
                seen_notices.append(fact.scope_notice)
        else:
            presented_lines.append(line)

    if not presented_lines:
        return None

    body = "\n".join(presented_lines)
    if seen_notices:
        notices_text = " | ".join(seen_notices)
        return f"{body}\nNota de escopo: {notices_text}"
    return body
