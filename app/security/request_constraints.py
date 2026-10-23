"""Server-owned request authority. Parsing is deliberately bounded, never model-owned."""
from dataclasses import dataclass
from datetime import date
import re
import unicodedata
from typing import Any

from app.security.pii import PIIRedactor

TERMINAL_PATTERN = r'\bGET-(?!ESC-)[A-Z0-9]+(?:-[A-Z0-9]+)+\b'
TRANSACTION_PATTERN = r'\bTXN-[A-Z0-9]+(?:-[A-Z0-9]+)*\b'


@dataclass(frozen=True)
class RequestConstraints:
    original_message: str
    terminal_ids: tuple[str, ...]
    transaction_ids: tuple[str, ...]
    sales_dates: tuple[str, ...]
    ambiguous_date: bool


class RequestConstraintError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _normalize(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))


def diagnostic_reference_intent(message: str) -> bool:
    """Recognized inspection + device/property slots, independent of deixis.

    A singular persisted POS can fill the missing subject of these diagnostic
    reads. This does not infer arbitrary document attributes or action consent.
    """
    norm=_normalize(message)
    inspection=re.search(r'\b(?:check|inspect\w*|verifi\w*|confir\w*|consult\w*|examin\w*|cheque|diagnos\w*)\b',norm)
    property_slot=re.search(r'\b(?:versao|version|firmware|status|estado|situacao|telemetr\w*|connect\w*|conect\w*|conex\w*|sinal|signal|hardware)\b',norm)
    device_slot=re.search(r'\b(?:terminal|pos|device|equipment|aparelho|equipamento|maquina|maquininha|machine)\b',norm)
    specific_property=re.search(r'\b(?:firmware|telemetr\w*|connect\w*|conect\w*|conex\w*|sinal|signal|hardware)\b',norm)
    return bool((inspection and (specific_property or device_slot)) or (property_slot and device_slot))


def _new_device_target(norm: str) -> bool:
    return bool(re.search(r'\b(?:outro|outra|novo|nova|another|different|new)\s+(?:terminal|pos|device|equipment|aparelho|equipamento|maquina|maquininha|machine)\b',norm))


def extract_constraints(message: str, conversation_context: dict | None = None) -> RequestConstraints:
    """Extract explicit IDs, ISO/PT/EN dates and unambiguous persisted POS references.

    Numeric dates with two possible day/month interpretations, invalid calendar
    dates, and written dates without a year require clarification. Relative dates
    are outside this parser; existing historical-snapshot notices still apply.
    """
    clean = PIIRedactor.mask_text(message)
    terminals = tuple(dict.fromkeys(re.findall(TERMINAL_PATTERN, clean.upper())))
    transactions = tuple(dict.fromkeys(re.findall(TRANSACTION_PATTERN, clean.upper())))
    norm = _normalize(clean)
    if not terminals and re.search(r'\b(?:mesm[oa]|same|ela|ele|dela|dele|esse|essa|aquele|aquela|it|its|that)\b', norm):
        entity = ((conversation_context or {}).get('entities') or {}).get('terminal_id')
        if isinstance(entity, str) and re.fullmatch(TERMINAL_PATTERN, entity.upper()):
            terminals = (entity.upper(),)
    # A recognized diagnostic continuation can use a singular persisted POS
    # without deictic words. Explicit new models/topics and documentary scope
    # stop inheritance; these slots are reference data, never permission.
    diagnostic = diagnostic_reference_intent(clean)
    new_topic = re.search(r'\b(?:get (?:smart|classica|tap)|manual\w*|document\w*|catalog\w*|como funciona|how does|perfil|profile|vendas|sales|payout|pix|link de pagamento|software|application|aplicativo|browser|navegador|dashboard)\b',norm)
    if not terminals and not transactions and diagnostic and not new_topic and not _new_device_target(norm):
        entity = ((conversation_context or {}).get('entities') or {}).get('terminal_id')
        if isinstance(entity,str) and re.fullmatch(TERMINAL_PATTERN,entity.upper()):
            terminals = (entity.upper(),)
    # Even an explicit deictic does not override a newly named model/topic.
    if (new_topic or _new_device_target(norm)) and not re.search(TERMINAL_PATTERN,clean,re.I):
        terminals = ()
    dates: list[str] = []
    ambiguous = False

    def add(year: str, month: str | int, day: str):
        nonlocal ambiguous
        try:
            value = date(int(year), int(month), int(day)).isoformat()
        except (ValueError, TypeError):
            ambiguous = True
        else:
            if value not in dates:
                dates.append(value)

    for match in re.finditer(r'\b(\d{4})-(\d{2})-(\d{2})\b', norm):
        add(*match.groups())
    for match in re.finditer(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', norm):
        first, second, year = match.groups()
        a, b = int(first), int(second)
        if a <= 12 and b <= 12 and a != b:
            ambiguous = True
        elif a > 12 or a == b:
            add(year, second, first)
        else:
            add(year, first, second)
    months = {
        'janeiro': 1, 'january': 1, 'fevereiro': 2, 'february': 2,
        'marco': 3, 'march': 3, 'abril': 4, 'april': 4, 'maio': 5, 'may': 5,
        'junho': 6, 'june': 6, 'julho': 7, 'july': 7, 'agosto': 8, 'august': 8,
        'setembro': 9, 'september': 9, 'outubro': 10, 'october': 10,
        'novembro': 11, 'november': 11, 'dezembro': 12, 'december': 12,
    }
    month_names = '|'.join(months)
    before = rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:(?:de|of)\s+)?({month_names})(?:,)?(?:\s+de)?\s+(\d{{4}})\b'
    after = rf'\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b'
    for match in re.finditer(before, norm):
        day, month, year = match.groups()
        add(year, months[month], day)
    for match in re.finditer(after, norm):
        month, day, year = match.groups()
        add(year, months[month], day)
    # A partial written date must not silently become the historical snapshot.
    written = re.search(rf'\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:(?:de|of)\s+)?(?:{month_names})|(?:{month_names})\s+\d{{1,2}})\b', norm)
    if written and not dates:
        ambiguous = True
    return RequestConstraints(clean, terminals, transactions, tuple(dates), ambiguous)


def declared_foreign_terminal(message: str, terminal_id: str) -> bool:
    """Explicit ownership is a denial before dispatch, not a tenant lookup hint."""
    norm = _normalize(message)
    ownership = re.compile(
        r"\b(?:pertenc\w*|belongs?\s+to|owned\s+by)\s+(?:(?:a|ao|aos|o|um|uma|meu|minha|our|my|the|a|an)\s+){0,3}"
        r"(?:fornecedor\w*|terceir\w*|outr[oa]\s+(?:cliente|empresa|conta)|supplier\w*|vendor\w*|third.party|another\s+(?:customer|company|account))\b"
        r"|\b(?:do|da|de um|de uma)\s+(?:fornecedor\w*|terceir\w*|outro cliente|outra empresa)\b"
        r"|\b(?:supplier|vendor|third.party|another customer)(?:'s|’s)?\s+(?:terminal|device|pos)\b")
    all_ids = extract_constraints(message).terminal_ids
    for sentence in re.split(r"[.!?;\n]+", message):
        sentence_norm = _normalize(sentence)
        local_ids = list(re.finditer(TERMINAL_PATTERN, sentence_norm, re.I))
        for declaration in ownership.finditer(sentence_norm):
            prefix = sentence_norm[max(0, declaration.start()-35):declaration.start()]
            if re.search(r"\b(?:nao|not|never)\s+(?:(?:does|do|actually|really)\s+)?$", prefix):
                continue
            if local_ids:
                before = [item for item in local_ids if item.end() <= declaration.start()]
                subject = before[-1] if before else local_ids[0]
                if subject.group().upper() == terminal_id.upper():
                    return True
            elif all_ids == (terminal_id.upper(),):
                return True
            elif terminal_id.upper() in all_ids:
                # An explicit plural subject refers to the complete named set;
                # merely mentioning a supplier supplies no ownership authority.
                subject = sentence_norm[:declaration.start()]
                plural_pair = len(all_ids) == 2 and re.search(r'\b(?:both|ambos|ambas|os dois|as duas)\b', subject)
                plural_all = len(all_ids) > 1 and re.search(r'\b(?:all (?:the )?(?:terminals|devices)|todos os (?:terminais|equipamentos)|todas as (?:maquinas|maquininhas))\b', subject)
                if plural_pair or plural_all:
                    return True
    return False


def constrain_call(tool_name: str, arguments: dict, constraints: RequestConstraints,
                   *, assigned_unit_ids: list[str] | None = None) -> dict:
    """Return a copy, bind a single omitted value, reject divergent/ambiguous calls."""
    result: dict[str, Any] = dict(arguments)
    if tool_name in {'diagnose_terminal_issue', 'execute_terminal_remote_action', 'get_customer_profile'}:
        # Reject even profile fallback when the sole declared subject is foreign.
        target = str(result.get('terminal_id') or '').upper()
        targets = (target,) if target else constraints.terminal_ids
        if any(declared_foreign_terminal(constraints.original_message, item) for item in targets):
            raise RequestConstraintError('FOREIGN_TERMINAL_NOT_AUTHORIZED',
                'O terminal foi declarado de terceiro; consulta ou proposta não autorizada.')

    def bind(field: str, requested: tuple[str, ...], code: str):
        value = result.get(field)
        normalized = str(value).upper() if field.endswith('_id') and value is not None else value
        if requested:
            if value in (None, ''):
                if len(requested) != 1:
                    raise RequestConstraintError('REQUEST_NEEDS_CLARIFICATION', 'Informe um único alvo ou data para esta consulta.')
                result[field] = requested[0]
            elif normalized not in requested:
                raise RequestConstraintError(code, 'A chamada diverge do alvo ou data da mensagem original; confirme os dados solicitados.')
            else:
                result[field] = normalized

    # Tool choice belongs to the model; access purpose belongs to the original
    # assigned literal units. A rewritten "full profile" subquery is not consent
    # to inspect registration when only a firmware read was requested.
    purpose = {'get_customer_profile': 'profile', 'get_sales_settlement': 'settlement',
               'diagnose_terminal_issue': 'diagnosis'}.get(tool_name)
    if purpose is not None:
        from app.security.request_units import request_units
        from app.security.operational_intent import classify_operational_intent
        from app.security.fulfillment import canonical_terminal_read_text, _property_read, _independent_action_veto
        authorized = False
        for unit in request_units(constraints.original_message):
            if assigned_unit_ids is not None and unit['unit_id'] not in assigned_unit_ids:
                continue
            if _independent_action_veto(unit['text']):
                continue
            canonical = canonical_terminal_read_text(unit['text'].lstrip(' ;,.'), constraints.original_message)
            # Persisted subject slots were already resolved by extract_constraints
            # with the trusted conversation context, never by the subquery.
            reference_context = ({'entities': {'terminal_id': constraints.terminal_ids[0]}}
                                 if len(constraints.terminal_ids) == 1 and
                                 re.search(r'\b(?:diagnostico|diagnosis|firmware|hardware|status|estado|situacao|signal|sinal|connect\w*|conect\w*|conex\w*|telemetr\w*|declin\w*|recus\w*|codigo|code)\b',
                                           _normalize(canonical)) else None)
            intent = classify_operational_intent(canonical, reference_context)
            read = _property_read(canonical, reference_context)
            if purpose in intent.intents or (read is not None and read.tool_name == tool_name):
                authorized = True
                break
        if not authorized:
            raise RequestConstraintError('REQUEST_PURPOSE_MISMATCH',
                'A ferramenta diverge do propósito da unidade original atribuída; confirme a consulta desejada.')

    # Read authority is a canonical original unit, never a model-written subquery.
    # Explicit independent targets can be inspected separately. Transaction pairs
    # remain local to that target's unit; an ambiguous list is not a Cartesian product.
    read_constraints = constraints
    if (tool_name == 'diagnose_terminal_issue' and constraints.terminal_ids
            and (assigned_unit_ids is not None or len(constraints.terminal_ids) > 1)
            and re.search(TERMINAL_PATTERN, constraints.original_message, re.I)):
        from app.security.request_units import request_units
        from app.security.fulfillment import canonical_terminal_read_text
        candidates = []
        for unit in request_units(constraints.original_message):
            if assigned_unit_ids is not None and unit['unit_id'] not in assigned_unit_ids:
                continue
            if _independent_action_veto(unit['text']):
                continue
            canonical = canonical_terminal_read_text(unit['text'].lstrip(' ;,.'), constraints.original_message)
            local = extract_constraints(canonical)
            if len(local.terminal_ids) == 1:
                candidates.append(local)
        requested_target = str(result.get('terminal_id') or '').upper()
        if not requested_target:
            targets = tuple(dict.fromkeys(target for item in candidates for target in item.terminal_ids))
            if len(targets) != 1:
                raise RequestConstraintError('REQUEST_NEEDS_CLARIFICATION', 'Informe um alvo inequívoco para esta consulta.')
            requested_target = targets[0]
        matching = [item for item in candidates if requested_target in item.terminal_ids]
        if not matching:
            raise RequestConstraintError('REQUEST_TARGET_MISMATCH', 'O alvo não pertence a uma unidade original atribuída inequívoca.')
        transaction = str(result.get('transaction_id') or '').upper()
        if transaction:
            matching = [item for item in matching if transaction in item.transaction_ids]
            if not matching:
                raise RequestConstraintError('REQUEST_TRANSACTION_MISMATCH', 'A transação não pertence à unidade original deste terminal.')
        transaction_sets = {item.transaction_ids for item in matching}
        if len(transaction_sets) != 1:
            raise RequestConstraintError('REQUEST_NEEDS_CLARIFICATION', 'Confirme a transação para este terminal.')
        transactions = next(iter(transaction_sets))
        if len(transactions) > 1:
            raise RequestConstraintError('REQUEST_NEEDS_CLARIFICATION', 'Informe uma única transação para esta consulta.')
        # An explicitly supplied unrelated transaction must not pass when the
        # assigned unit is merely a firmware/status read.
        read_constraints = RequestConstraints(constraints.original_message, (requested_target,), transactions,
            constraints.sales_dates, constraints.ambiguous_date)
        result['terminal_id'] = requested_target

    if tool_name in ('diagnose_terminal_issue', 'execute_terminal_remote_action'):
        if not constraints.terminal_ids and _new_device_target(_normalize(constraints.original_message)):
            raise RequestConstraintError('REQUEST_NEEDS_CLARIFICATION','Informe o ID do novo terminal; o terminal anterior não será substituído silenciosamente.')
        if not constraints.terminal_ids and re.search(r'\b(?:mesm[oa]|same)\s+(?:terminal|maquina|maquininha|machine)\b', _normalize(constraints.original_message)):
            raise RequestConstraintError('REQUEST_NEEDS_CLARIFICATION', 'Não há um terminal de referência inequívoco. Informe o ID do terminal.')
        bind('terminal_id', read_constraints.terminal_ids, 'REQUEST_TARGET_MISMATCH')
    if tool_name == 'diagnose_terminal_issue':
        bind('transaction_id', read_constraints.transaction_ids, 'REQUEST_TRANSACTION_MISMATCH')
    if tool_name == 'get_sales_settlement':
        if constraints.ambiguous_date:
            raise RequestConstraintError('REQUEST_DATE_AMBIGUOUS', 'A data é ambígua ou inválida. Informe dia, mês e ano por extenso ou em ISO (AAAA-MM-DD).')
        bind('sales_date', constraints.sales_dates, 'REQUEST_DATE_MISMATCH')
    return result
