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


def constrain_call(tool_name: str, arguments: dict, constraints: RequestConstraints) -> dict:
    """Return a copy, bind a single omitted value, reject divergent/ambiguous calls."""
    result: dict[str, Any] = dict(arguments)

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

    if tool_name in ('diagnose_terminal_issue', 'execute_terminal_remote_action'):
        if not constraints.terminal_ids and re.search(r'\b(?:mesm[oa]|same)\s+(?:terminal|maquina|maquininha|machine)\b', _normalize(constraints.original_message)):
            raise RequestConstraintError('REQUEST_NEEDS_CLARIFICATION', 'Não há um terminal de referência inequívoco. Informe o ID do terminal.')
        bind('terminal_id', constraints.terminal_ids, 'REQUEST_TARGET_MISMATCH')
    if tool_name == 'diagnose_terminal_issue':
        bind('transaction_id', constraints.transaction_ids, 'REQUEST_TRANSACTION_MISMATCH')
    if tool_name == 'get_sales_settlement':
        if constraints.ambiguous_date:
            raise RequestConstraintError('REQUEST_DATE_AMBIGUOUS', 'A data é ambígua ou inválida. Informe dia, mês e ano por extenso ou em ISO (AAAA-MM-DD).')
        bind('sales_date', constraints.sales_dates, 'REQUEST_DATE_MISMATCH')
    return result
