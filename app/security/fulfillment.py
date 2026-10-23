"""Bounded operational fulfillment from server-owned typed observations.

This is a deliberately finite property-read contract, not open-world entailment.
A successful observation is only eligible evidence; relation, useful field values,
requested target, historical date and scope must all agree. Unknown predicates and
conditional/negative scopes remain pending. No model-written text grants credit.
"""
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Iterable

from app.schemas import ToolExecutionResult
from app.security.operational_intent import classify_operational_intent
from app.security.request_constraints import _normalize, extract_constraints, TERMINAL_PATTERN, TRANSACTION_PATTERN


class FulfillmentReason(str, Enum):
    VERIFIED = 'verified'
    UNKNOWN_RELATION = 'unknown_relation'
    CONDITIONAL_SCOPE = 'conditional_scope'
    ACTION_PENDING = 'action_pending'
    MISSING_OBSERVATION = 'missing_observation'
    MISSING_FACT = 'missing_fact'
    TARGET_MISMATCH = 'target_mismatch'
    DATE_MISMATCH = 'date_mismatch'
    SCOPE_MISMATCH = 'scope_mismatch'


@dataclass(frozen=True)
class FulfillmentDecision:
    fulfilled: bool
    reason: FulfillmentReason
    required_fact_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationalObservation:
    tool_name: str
    data: dict[str, Any]

    @classmethod
    def from_result(cls, result: ToolExecutionResult):
        if (result.success and result.outcome_state == 'completed'
                and isinstance(result.data, dict) and result.data.get('found') is True):
            return cls(result.tool_name, result.data)
        return None


@dataclass(frozen=True)
class PropertyRead:
    tool_name: str
    paths: tuple[str, ...]
    relation: str = 'property_read'


def _at(data: dict, path: str) -> Any:
    value: Any = data
    for key in path.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _useful(value: Any, path: str) -> bool:
    if path == 'recommended_troubleshooting.connectivity':
        return isinstance(value, list) and bool(value) and all(isinstance(item, str) and bool(item.strip()) for item in value)
    if path.endswith('_brl') or path == 'terminals_count':
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if path in {'antecipacao_automatica', 'requires_hardware_replacement'}:
        return isinstance(value, bool)
    if isinstance(value, str):
        return bool(value.strip()) and _normalize(value.strip()) not in {
            'nao informado', 'not informed', 'unknown', 'unavailable', 'n/a', 'none', 'null'}
    if isinstance(value, list):
        return bool(value) and all(isinstance(item, dict) and _useful(item.get('terminal_id'), 'terminal_id')
                                   for item in value)
    return False


# Conditions apply to the full original, because the literal unit splitter can
# isolate "if ..." from the proposition it governs. Conservative across clauses.
SCOPE_MODIFIER = re.compile(r'\b(?:se|if|unless|exceto|except|desde que|provided that|only when|somente quando|apenas quando|garant\w*|guarantee\w*|nao|nunca|not|never)\b')
# A property occurrence is not proof of a relation about that property (e.g.
# firmware security, payment equivalence or public eligibility).
UNSUPPORTED_PREDICATE = re.compile(r'\b(?:equival\w*|igual|same as|become\w*|torna\w*|significa|means?|every|toda\w*|secure|segur\w*|encrypt\w*|criptograf\w*|certific\w*|compativ\w*|compatib\w*|eligible|elegiv\w*|permit\w*|permite\w*|suporta\w*|supports?|pode|can|necessari\w*|obrigatori\w*|mandatory|requires?|requirement|juros|interest|taxa|fees?|battery|bateria|capacity|capacidade)\b')


CONNECTIVITY_SYMPTOM = re.compile(r"\b(?:nao conect\w*|sem (?:sinal|internet|conexao)|(?:won[’']t|will not|does not|do not|cannot|can[’']t) connect(?:ing)?)\b")
ADVICE_REQUEST = re.compile(r'\b(?:o que (?:eu )?faco|como (?:eu )?resolvo|what should i do|what can i do|how can i fix (?:it|this))\b')


def _connectivity_advice(norm: str) -> bool:
    return bool(CONNECTIVITY_SYMPTOM.search(norm) and ADVICE_REQUEST.search(norm))


def _advice_shape(norm: str) -> str:
    """Only reported connectivity negation is consumed, never command negation."""
    return ADVICE_REQUEST.sub(' check ', CONNECTIVITY_SYMPTOM.sub(' connectivity ', norm))


READ_VERBS = frozenset({
    'check', 'inspect', 'verify', 'confirm', 'consult', 'examine', 'diagnose',
    'verifique', 'verificar', 'verifiquem', 'confira', 'conferir', 'cheque',
    'checar', 'consulte', 'consultar', 'confirme', 'confirmar', 'examine',
    'examinar', 'diagnostique', 'diagnosticar', 'inspecione', 'inspecionar',
})
_READ_VERB_PATTERN = '|'.join(sorted(READ_VERBS, key=lambda word: (-len(word), word)))


def _decline_incident(text: str) -> bool:
    """Finite device/report/event grammar, distinct from a property read.

    This relation is a diagnosis plus advice, not proof that a code alone
    explains the event. Identifiers remain opaque request-authority slots.
    """
    norm = _normalize(re.sub(TRANSACTION_PATTERN, ' transactiontarget ',
        re.sub(TERMINAL_PATTERN, ' target ', text, flags=re.I), flags=re.I)).strip()
    if not re.search(r'\b(?:decline|declined|recusa|recusada)\b', norm):
        return False
    subject = r'(?:(?:my|our|meu|minha|nosso|nossa|o|a|the)\s+)?(?:card machine|terminal|pos|maquininha|maquina)(?:\s+target)?'
    report = r'(?:is showing|shows|displays|reports|presents|esta mostrando|esta exibindo|mostra|exibe|apresenta|retorna)'
    event = r'(?:a|an|um|uma|erro|error|de|da|do|transaction|transacao|payment|pagamento|decline|declined|recusa|recusada|code|codigo|for|para|on|at|transactiontarget|[0-9]{2,3})'
    return bool(re.fullmatch(subject + r'\s+' + report + r'\s+' + event + r'(?:\s+' + event + r')*[.!?\s]*', norm))


def _read_prefix(norm: str) -> str:
    norm = norm.strip()
    # Consume a finite leading request prefix only. These forms ask to show a
    # property; residue still has to satisfy the same bounded field grammar.
    norm = re.sub(r'^(?:(?:por favor|please)[,\s]+)?(?:(?:voce pode|pode|can you|could you)\s+)?'
                  r'(?:(?:me\s+)?(?:diga|informe|mostre)|(?:give|tell|show)\s+me)\b[,\s]*', 'check ', norm)
    return re.sub(r'^(?:pode|voce pode|can you|could you)\s+(?=(?:' + _READ_VERB_PATTERN + r')\b)', '', norm)


def _property_read_shape(text: str, domain: str) -> bool:
    """Consume a finite read grammar; unknown residue never grants credit.

    IDs and dates are opaque slots. Remaining words can name a read, known
    properties and their subject/reference, but cannot add a new relation.
    """
    text = re.sub(TERMINAL_PATTERN, ' terminal ', text, flags=re.I)
    text = re.sub(TRANSACTION_PATTERN, ' transaction ', text, flags=re.I)
    norm = _read_prefix(_normalize(text).strip(' ;,.'))
    if domain == 'terminal' and _connectivity_advice(norm):
        norm = _advice_shape(norm)
    # Copulas are permitted in a leading property question, not as an added
    # predicate such as "check firmware ... is current".
    norm = re.sub(r'^(?:what (?:is|are)|qual e|quais sao)\b', '', norm)
    if re.search(r'\b(?:is|are|has|tem)\b', norm):
        return False
    common = {
        'a', 'as', 'o', 'os', 'de', 'do', 'da', 'dos', 'das', 'no', 'na', 'com',
        'em', 'e', 'the', 'of', 'on', 'for', 'to', 'and', 'its', 'it', 'that',
        'this', 'same', 'my', 'our', 'meu', 'minha', 'meus', 'minhas', 'nosso',
        'nossa', 'esse', 'essa', 'aquele', 'aquela', 'mesmo', 'mesma', 'dele',
        'dela', 'ele', 'ela', 'please', 'por', 'favor', 'what', 'which', 'qual', 'quais', 'sua', 'seu', 'atual',
        'plus', 'mais', 'alem', 'well', 'current', 'que', 'citei', 'anteriormente', 'mencionado', 'mencionada',
        'referido', 'referida', 'acima', 'mentioned', 'earlier', 'above', 'previously',
    }
    fields = {
        'terminal': {
            'get', 'smart', 'classica', 'terminal', 'pos', 'device', 'equipment',
            'aparelho', 'equipamento', 'maquina', 'maquininha', 'machine', 'card',
            'firmware', 'versao', 'version', 'status', 'estado', 'situacao',
            'sinal', 'signal', 'strength', 'tipo', 'type', 'conexao',
            'conectividade', 'connection', 'connectivity', 'telemetria',
            'telemetry', 'hardware', 'defeito', 'falha', 'leitor', 'reader',
            'fault', 'broken', 'chip', 'internet', 'rede', 'wifi', 'offline',
            'transacao', 'transaction', 'recusa', 'recusada', 'decline',
            'declined', 'codigo', 'code', 'erro', 'error', 'esta', 'estao',
            'apresenta', 'acontecendo', 'why', 'porque', 'motivo', 'reason', 'was', 'causa',
        },
        'profile': {
            'perfil', 'profile', 'cadastro', 'cadastral', 'cadastrado', 'registered',
            'completo', 'complete', 'details', 'account', 'plan', 'plano', 'ativo',
            'automatic', 'advance', 'antecipacao', 'automatica', 'domicilio',
            'bancario', 'bank', 'conta', 'cnpj',
        },
        'settlement': {
            'vendas', 'venda', 'sales', 'sale', 'settlement', 'settlements',
            'payout', 'payouts', 'proceeds', 'recebimentos', 'recebimento',
            'repasse', 'repasses', 'liquidacao', 'extrato', 'valor', 'amount',
            'bruto', 'gross', 'total', 'volume', 'liquido', 'net', 'takings',
            'debit', 'debito', 'credit', 'credito', 'pix', 'status', 'estado',
            'situacao', 'lote', 'batch', 'data', 'date', 'how', 'much', 'were',
            'janeiro', 'january', 'fevereiro', 'february', 'marco', 'march',
            'abril', 'april', 'maio', 'may', 'junho', 'june', 'julho', 'july',
            'agosto', 'august', 'setembro', 'september', 'outubro', 'october',
            'novembro', 'november', 'dezembro', 'december',
        },
    }
    if domain == 'terminal':
        norm = re.sub(r'\b(code|codigo|erro|error)\s*[:#]?\s*\d{2,3}\b', r'\1', norm)
    words = re.findall(r"[a-z0-9]+", norm)
    vocabulary = common | fields[domain]
    return all(word in vocabulary or word in READ_VERBS or (domain == 'settlement' and word.isdigit()) for word in words)


def _property_read(text: str, context: dict | None) -> PropertyRead | None:
    norm = _read_prefix(_normalize(text))
    has = lambda pattern: bool(re.search(pattern, norm))
    if _decline_incident(text):
        return PropertyRead('diagnose_terminal_issue', ('last_declined_transaction.decline_code',
            'last_declined_transaction.decline_reason', 'last_declined_transaction.action_required'),
            relation='diagnose_decline_and_recommend')
    if _connectivity_advice(norm) and _property_read_shape(text, 'terminal'):
        return PropertyRead('diagnose_terminal_issue', ('terminal.status', 'terminal.connection_type',
            'terminal.signal_strength', 'recommended_troubleshooting.connectivity'))
    if UNSUPPORTED_PREDICATE.search(norm):
        return None
    intent = classify_operational_intent(text, context)
    if intent.documentary_scope:
        return None
    # Explicit profile properties can be requested without the word "profile".
    personal = has(r'\b(?:my|our|meu|minha|nosso|nossa)\b')
    profile_fields = (
        (r'\b(?:automatic advance|antecipacao automatica)\b', ('antecipacao_automatica',)),
        (r'\b(?:account plan|plano)\b', ('account_plan',)),
        (r'\b(?:domicilio bancario|bank account|bank details)\b', ('domicilio_bancario.banco', 'domicilio_bancario.agencia', 'domicilio_bancario.conta_corrente')),
        (r'\b(?:cnpj)\b', ('cnpj',)),
    )
    profile_shape = _property_read_shape(text, 'profile')
    if personal and profile_shape:
        paths = [path for pattern, fields in profile_fields if has(pattern) for path in fields]
        if has(r'\b(?:completo|complete)\b') and has(r'\b(?:perfil|profile)\b'):
            paths.extend(('company_name', 'cnpj', 'account_plan', 'antecipacao_automatica',
                'domicilio_bancario.banco', 'domicilio_bancario.agencia', 'domicilio_bancario.conta_corrente',
                'terminals_count', 'terminals_summary'))
        if paths:
            return PropertyRead('get_customer_profile', tuple(dict.fromkeys(paths)))
    if 'profile' in intent.intents and profile_shape:
        # An unqualified profile request has a bounded summary contract; an
        # explicit full profile above requires every rendered profile category.
        return PropertyRead('get_customer_profile', ('company_name', 'account_plan', 'terminals_summary'))
    if 'settlement' in intent.intents and _property_read_shape(text, 'settlement'):
        paths = []
        for pattern, field in (
            (r'\b(?:liquido|net|proceeds)\b', 'net_payout_scheduled_today_brl'),
            (r'\b(?:bruto|gross|volume|vendas|sales|takings)\b', 'total_gross_brl'),
            (r'\b(?:status|estado|situacao)\b', 'payout_status'),
            (r'\b(?:lote|batch)\b', 'settlement_batch_id'),
            (r'\b(?:debit\w*)\b', 'debit_brl'),
            (r'\b(?:credit\w*)\b', 'credit_brl'),
            (r'\bpix\b', 'pix_brl'),
        ):
            if has(pattern): paths.append('sales_data.' + field)
        if not paths:
            paths = ['sales_data.net_payout_scheduled_today_brl', 'sales_data.payout_status']
        return PropertyRead('get_sales_settlement', tuple(paths))
    property_inspection = bool(set(re.findall(r'[a-z]+', norm)) & READ_VERBS) and has(r'\b(?:firmware|hardware|signal|sinal|telemetr\w*)\b')
    if 'diagnosis' not in intent.intents and not property_inspection:
        return None
    if not _property_read_shape(text, 'terminal'):
        return None
    cause = has(r'\b(?:why|porque|por que|motivo|reason|causa)\b')
    decline = has(r'\b(?:recus\w*|declin\w*|refus\w*)\b')
    if cause and not decline:
        return None
    paths = ['last_declined_transaction.decline_reason'] if cause else []
    for pattern, path in (
        (r'\b(?:firmware|versao|version)\b', 'terminal.firmware_version'),
        (r'\b(?:status|estado|situacao)\b', 'terminal.status'),
        (r'\b(?:signal|sinal)\b', 'terminal.signal_strength'),
        (r'\b(?:connect\w*|conect\w*|conex\w*|wifi|internet|rede|offline)\b', 'terminal.connection_type'),
        (r'\b(?:hardware|defeit\w*|falha|leitor|reader|fault|broken)\b', 'requires_hardware_replacement'),
        (r'\b(?:recus\w*|declin\w*|refus\w*|codigo|code)\b', 'last_declined_transaction.decline_code'),
    ):
        if has(pattern): paths.append(path)
    if has(r'\b(?:telemetr\w*)\b'):
        paths.extend(('terminal.status', 'terminal.connection_type', 'terminal.signal_strength', 'terminal.firmware_version'))
    if not paths and re.fullmatch(r'(?:check|inspect|verifique|cheque|diagnostique)\s+(?:(?:o|the|terminal)\s+)*get-[a-z0-9-]+[.!?\s]*', norm.strip()):
        paths.extend(('terminal.status', 'terminal.connection_type'))
    return PropertyRead('diagnose_terminal_issue', tuple(dict.fromkeys(paths))) if paths else None


def canonical_terminal_read_text(text: str, original: str, context: dict | None = None) -> str:
    """Resolve an elliptical property unit only within a pure, single-subject read.

    The unit must be a literal server-owned span. The entire original must fit
    the finite terminal property grammar; arbitrary predicates or other subjects
    cannot donate a target or a read relation to an isolated keyword.
    """
    from app.security.request_units import request_units
    literal = text.lstrip(' ;,.').strip()
    if not any(unit['text'].lstrip(' ;,.').strip() == literal for unit in request_units(original)):
        return text
    from app.security.request_units import operational_unit_text
    text = operational_unit_text(text, original)
    authority = extract_constraints(original, context)
    local = extract_constraints(text, context)
    if local.terminal_ids or len(authority.terminal_ids) != 1 or authority.transaction_ids:
        return text
    if (SCOPE_MODIFIER.search(_normalize(original))
            or not _property_read_shape(original, 'terminal')
            or _property_read(original, context) is None):
        return text
    return f"{text.rstrip(' .!?')} of {authority.terminal_ids[0]}"


def _independent_action_veto(text: str) -> bool:
    """A fully consumed veto of a supported action, with an explicit target."""
    norm = _normalize(re.sub(TERMINAL_PATTERN, ' target ', text, flags=re.I)).strip(' ;,.!?')
    prefix = r'(?:(?:please|por favor)[,\s]+)?(?:do not|don[’\']t|never|nao|nunca)\s+'
    subject = r'(?:(?:the|my|our|o|a|meu|minha|nosso|nossa|terminal|pos|machine|maquina|maquininha|device)\s+)*target'
    reboot = r'(?:reboot|restart|reinicie|reinicia|reiniciar)\s+' + subject
    table = r'(?:update|atualize|atualizar)\s+(?:(?:the|a)\s+)?(?:parameter table|tabela(?: de parametros)?)\s+(?:(?:of|de|do|da)\s+)?' + subject
    carrier = r'(?:switch|troque|mude|trocar)\s+(?:(?:the|a)\s+)?(?:carrier|operadora)\s+(?:(?:of|de|do|da)\s+)?' + subject
    return bool(re.fullmatch(prefix + '(?:' + '|'.join((reboot, table, carrier)) + ')', norm))


def _unit_scope_text(text: str, original: str) -> str:
    """Exclude only fully known independent action vetoes from another unit.

    Standalone/attached conditions retain global scope. Conjunctions are never
    independence boundaries, and the negative unit itself is never exempted.
    """
    from app.security.request_units import request_units
    literal = text.lstrip(' ;,.').strip()
    units = [unit for unit in request_units(original) if unit['text'].lstrip(' ;,.').strip() == literal]
    if len(units) != 1:
        return original
    position = units[0]['start'] + units[0]['text'].find(literal)
    spans = []
    start = 0
    for boundary in re.finditer(r';|(?<=[.!?])\s+(?=\S)', original):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(original)))
    return '; '.join(original[a:b] for a, b in spans
                     if a <= position < b or not _independent_action_veto(original[a:b]))


def evaluate_operational_unit(text: str, results: Iterable[ToolExecutionResult],
                              context: dict | None = None, *, original: str | None = None,
                              authenticated_user_id: str | None = None) -> FulfillmentDecision:
    """Return credit only for one complete, matching typed observation.

    Different observations cannot donate unrelated fields to fabricate a complete
    row. Multiple targets/dates require separate unequivocal units or clarification.
    """
    scope_text = _unit_scope_text(text, original) if original is not None else text
    if original is not None:
        text = canonical_terminal_read_text(text, original, context)
    norm = _normalize(text)
    scope_norm = _normalize(scope_text)
    if _connectivity_advice(scope_norm):
        scope_norm = _advice_shape(scope_norm)
    if SCOPE_MODIFIER.search(scope_norm):
        return FulfillmentDecision(False, FulfillmentReason.CONDITIONAL_SCOPE)
    if 'action' in classify_operational_intent(text, context).intents:
        return FulfillmentDecision(False, FulfillmentReason.ACTION_PENDING)
    read = _property_read(text, context)
    if read is None:
        return FulfillmentDecision(False, FulfillmentReason.UNKNOWN_RELATION)
    failure = FulfillmentReason.MISSING_OBSERVATION
    constraints = extract_constraints(text, context)
    original_constraints = extract_constraints(original, context) if original is not None else constraints
    for result in results:
        observation = OperationalObservation.from_result(result)
        if observation is None or observation.tool_name != read.tool_name:
            continue
        data = observation.data
        required_paths = read.paths
        if read.relation == 'diagnose_decline_and_recommend' and not data.get('last_declined_transaction'):
            required_paths = ('recommended_troubleshooting.decline_resolution', 'iso8583_catalog_reference.code',
                              'iso8583_catalog_reference.name', 'iso8583_catalog_reference.action')
            if data.get('symptom_classified') != 'decline':
                failure = FulfillmentReason.SCOPE_MISMATCH
                continue
            catalog = data.get('iso8583_catalog_reference') or {}
            canonical = f"Código {catalog.get('code')} ({catalog.get('name')}): {catalog.get('action')}"
            if _normalize(str(_at(data, 'recommended_troubleshooting.decline_resolution'))) != _normalize(canonical):
                failure = FulfillmentReason.SCOPE_MISMATCH
                continue
        if authenticated_user_id is not None and data.get('user_id') != authenticated_user_id:
            failure = FulfillmentReason.SCOPE_MISMATCH
            continue
        if read.tool_name == 'diagnose_terminal_issue':
            terminal_ids = constraints.terminal_ids or original_constraints.terminal_ids
            transaction_ids = constraints.transaction_ids or original_constraints.transaction_ids
            if terminal_ids and (len(terminal_ids) != 1 or str(_at(data, 'terminal.terminal_id')).upper() != terminal_ids[0]):
                failure = FulfillmentReason.TARGET_MISMATCH
                continue
            if transaction_ids and (len(transaction_ids) != 1 or str(_at(data, 'last_declined_transaction.transaction_id')).upper() != transaction_ids[0]):
                failure = FulfillmentReason.TARGET_MISMATCH
                continue
            if not re.fullmatch(TERMINAL_PATTERN, str(_at(data, 'terminal.terminal_id')).upper()):
                failure = FulfillmentReason.TARGET_MISMATCH
                continue
            requested_code = re.search(r'\b(?:code|codigo|erro|error)\s*[:#]?\s*(\d{2,3})\b', norm)
            if requested_code and str(_at(data, 'last_declined_transaction.decline_code') or _at(data, 'iso8583_catalog_reference.code')) != requested_code.group(1):
                failure = FulfillmentReason.TARGET_MISMATCH
                continue
            model = re.search(r'\bget (smart|classica)\b', norm)
            if model and model.group(1) not in _normalize(str(_at(data, 'terminal.model'))):
                failure = FulfillmentReason.TARGET_MISMATCH
                continue
        if read.tool_name == 'get_sales_settlement':
            if (constraints.ambiguous_date or len(constraints.sales_dates) != 1
                    or data.get('requested_sales_date') != constraints.sales_dates[0]
                    or _at(data, 'sales_data.date') != constraints.sales_dates[0]):
                failure = FulfillmentReason.DATE_MISMATCH
                continue
            if data.get('data_scope') != 'synthetic_historical_snapshot':
                failure = FulfillmentReason.SCOPE_MISMATCH
                continue
        if any(path.startswith('last_declined_transaction.') for path in required_paths) and data.get('symptom_classified') != 'decline':
            failure = FulfillmentReason.SCOPE_MISMATCH
            continue
        if not all(_useful(_at(data, path), path) for path in required_paths):
            failure = FulfillmentReason.MISSING_FACT
            continue
        return FulfillmentDecision(True, FulfillmentReason.VERIFIED, required_paths)
    return FulfillmentDecision(False, failure, read.paths)
