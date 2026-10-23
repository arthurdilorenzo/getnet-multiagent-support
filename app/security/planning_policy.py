"""Conservative work eligibility; this never grants fulfillment or consent."""
import re
from app.security.fulfillment import _independent_action_veto
from app.security.request_units import request_units


def independent_veto_unit_ids(message: str) -> set[str]:
    """Recognize complete independent clauses, retaining literal canonical units.

    Conjunction/condition boundaries in request_units are deliberately NOT
    independence boundaries: consuming only the first half would hide a request.
    """
    units = request_units(message)
    spans = []
    start = 0
    for boundary in re.finditer(r';|(?<=[.!?])\s+(?=\S)', message):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(message)))
    result = set()
    for start, end in spans:
        if not _independent_action_veto(message[start:end]):
            continue
        for unit in units:
            # Semicolons/whitespace can belong to the next canonical unit.
            left = max(start, unit['start'])
            right = min(end, unit['end'])
            if left < right and message[left:right].strip(' ;,.!?'):
                result.add(unit['unit_id'])
    return result
