"""Shared boundary for explicit external evidence versus Getnet product facts."""
import re


def asks_explicit_external_evidence(text: str) -> bool:
    """Current product rules alone do not request an external/regulatory source."""
    return bool(re.search(
        r'\b(?:banco central|bcb|central bank|regulator|regulador|'
        r'external|extern[ao]s?|regula\w*|norma\w*|law\w*|lei|legisla\w*)\b',
        text, re.I,
    ))


def documentary_clause_span(message: str, offset: int) -> tuple[int, int]:
    """Keep conjoined qualifiers together; independent sentences/clauses may differ."""
    separators = list(re.finditer(r';|(?<=[.!?])\s+(?=\S)', message))
    start = max([0] + [match.end() for match in separators if match.end() <= offset])
    end = min([len(message)] + [match.start() for match in separators if match.start() > offset])
    return start, end
