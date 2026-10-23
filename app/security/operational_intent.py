"""Bounded offline operational slots shared by routing and tool planning.

This classifier recognizes operational shapes, not unrestricted language. Unknown
requests stay unknown; callers must not turn them into a private profile query.
"""
from dataclasses import dataclass
import re
from app.security.request_constraints import extract_constraints, _normalize


@dataclass(frozen=True)
class OperationalIntent:
    intents: tuple[str, ...]
    private_scope: bool
    documentary_scope: bool
    ambiguous: bool


def _classify_clause(message: str, conversation_context: dict | None = None) -> OperationalIntent:
    norm = _normalize(message)
    constraints = extract_constraints(message, conversation_context)
    def has(pattern: str) -> bool:
        return bool(re.search(pattern, norm))
    personal = has(r'\b(?:my|our|meu|minha|meus|minhas|nosso|nossa)\b')
    temporal = bool(constraints.sales_dates or constraints.ambiguous_date) or has(r'\b(?:today|yesterday|ontem|hoje|last|previous|semana|week|mes|month)\b')
    documentary = has(r'\b(?:manuals?|user guide|documentation|catalogo?|diferenca|difference|compare|comparar)\b')
    commercial = has(r'\b(?:como funciona\w*|how (?:does|do).{0,80}work|what is|o que e|quais (?:as|os)|qual a taxa|quanto custa|how much does.{0,40}cost)\b')
    requirement = has(r'\b(?:preciso|necessario|obrigatorio|need|require\w*|mandatory)\b.{0,80}\b(?:conta|account|bank)\b')
    documentary = documentary or requirement or (commercial and not temporal)
    settlement_subject = has(r'\b(?:sales?|takings?|settlements?|payouts?|proceeds|vendas?|recebimentos?|repasses?|liquidacao|extrato|saldo|deposit\w*|dinheiro)\b|\b(?:valor liquido|net (?:amount|payout|proceeds))\b')
    diagnosis_subject = bool(constraints.terminal_ids or constraints.transaction_ids) or has(r'\b(?:terminal|maquininha|maquina|machine|pos|transacao|transaction|firmware|telemetria|telemetry)\b')
    diagnostic = has(r'\b(?:diagnos\w*|check|cheque|inspect\w*|verifi\w*|inspec\w*|analis\w*|analy\w*|investig\w*|status|estado|situacao|signal|sinal|connect\w*|conex\w*|telemetr\w*|declin\w*|recus\w*|refus\w*|error|erro|code|codigo|firmware|hardware|offline|broken|defeito|leitor|wifi|internet|rede|chip)\b')
    action = has(r'\b(?:reinicie|reinicia|reiniciar|restart|reboot|atualize|update|troque|switch)\b')
    known_error = has(r'\b(?:codigo|code|erro|error)\s*[:#]?\s*\d{2,3}\b')
    intents = []
    if settlement_subject and (personal or temporal) and not documentary:
        intents.append('settlement')
    if has(r'\b(?:perfil|cadastro|profile|account details|account plan|plano cadastrado)\b') and personal and not documentary:
        intents.append('profile')
    if not documentary and ((diagnosis_subject and (personal or diagnostic or action or bool(constraints.terminal_ids or constraints.transaction_ids))) or known_error):
        intents.append('diagnosis')
    if action and not documentary and (diagnosis_subject or personal):
        if 'diagnosis' not in intents:
            intents.append('diagnosis')
        intents.append('action')
    # Recognize uncertainty instead of authorizing arbitrary profile fallback.
    ambiguous = bool((settlement_subject or diagnosis_subject) and not intents and not documentary)
    return OperationalIntent(tuple(intents), bool(intents), documentary, ambiguous)


def classify_operational_intent(message: str, conversation_context: dict | None = None) -> OperationalIntent:
    # Boundaries are explicit punctuation or conjunction followed by a new
    # request verb. Do not spread one clause's date/private scope to another.
    clauses = re.split(r"[;?!]\s*|\.\s+|\b(?:e|and)\s+(?=(?:consulte|verifique|check|inspect|explique|explain)\b)", message, flags=re.I)
    classified = [_classify_clause(clause, conversation_context) for clause in clauses if clause.strip()]
    intents = tuple(dict.fromkeys(intent for item in classified for intent in item.intents))
    return OperationalIntent(intents, bool(intents),
        any(item.documentary_scope for item in classified), any(item.ambiguous for item in classified))
