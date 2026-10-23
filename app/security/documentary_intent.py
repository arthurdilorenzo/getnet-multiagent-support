"""Shared boundary for explicit external evidence versus Getnet product facts."""
import re


_SOURCE_QUALIFIER = re.compile(
    r"(?i)^\s*(?:use\s+getnet['’]s\s+own\s+guidance(?:,\s*please)?|"
    r"use\s+a\s+orienta[cç][aã]o\s+da\s+pr[oó]pria\s+getnet(?:,\s*por\s+favor)?|"
    r"consulte\s+fontes\s+atuais(?:\s+e\s+deixe\s+claro\s+"
    r"se\s+n[aã]o\s+houver\s+confirma[cç][aã]o)?)\s*[.!?]?\s*$"
)


def is_source_qualifier_only(text: str) -> bool:
    """Whether a query is only one of the supported source preference clauses."""
    return bool(_SOURCE_QUALIFIER.fullmatch(text))


_SOURCE_QUALIFIER_ANYWHERE = re.compile(
    r"(?i)(?:use\s+getnet['’]s\s+own\s+guidance|"
    r"use\s+a\s+orienta[cç][aã]o\s+da\s+pr[oó]pria\s+getnet|"
    r"consulte\s+fontes\s+atuais)"
)


def _source_qualifier_sentence_spans(message: str) -> list[tuple[int, int]]:
    """Find narrow standalone source constraints that qualify the prior request."""
    separators = list(re.finditer(r"(?<=[.!?])\s+(?=\S)", message))
    spans = []
    for index, boundary in enumerate(separators):
        end = separators[index + 1].start() if index + 1 < len(separators) else len(message)
        if _SOURCE_QUALIFIER.fullmatch(message[boundary.end():end].strip()):
            spans.append((boundary.start(), end))
    return spans


def asks_explicit_external_evidence(text: str) -> bool:
    """Current product rules alone do not request an external/regulatory source."""
    return bool(re.search(
        r'\b(?:banco central|bcb|central bank|regulator|regulador|'
        r'external|extern[ao]s?|regula\w*|norma\w*|law\w*|lei|legisla\w*|'
        r'fontes?\s+atuais|current\s+sources)\b',
        text, re.I,
    ))


def documentary_clause_span(message: str, offset: int) -> tuple[int, int]:
    """Keep conjoined qualifiers together; independent sentences/clauses may differ."""
    attached = _source_qualifier_sentence_spans(message)
    separators = [match for match in re.finditer(r';|(?<=[.!?])\s+(?=\S)', message)
                  if not any(start <= match.start() < end for start, end in attached)]
    start = max([0] + [match.end() for match in separators if match.end() <= offset])
    end = min([len(message)] + [match.start() for match in separators if match.start() > offset])
    return start, end
