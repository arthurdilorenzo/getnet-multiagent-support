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
        if is_documentary_qualifier(message[boundary.end():end].strip()):
            spans.append((boundary.start(), end))
    for boundary in separators:
        prior = message[:boundary.start()].rsplit('. ', 1)[-1]
        following = message[boundary.end():]
        if (_public_context(prior) and re.match(
                r'(?i)(?:walk|explain|show|how|me\s+explique|explique|mostre|como)\b', following)):
            next_boundary = next((item.start() for item in separators if item.start() > boundary.start()), len(message))
            spans.append((boundary.start(), next_boundary))
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


# A source/format directive qualifies the preceding need. Restrict the head to
# metalinguistic instructions; an additional factual question is never swallowed.
_SOURCE_DIRECTIVE = re.compile(
    r"(?is)^\s*(?:(?:answer|respond|responda)\s+(?:using|from|with|pelo|pela|com|em|"
    r"com base n[oa])\s+.*(?:getnet|material|source|fonte|documenta|internal|interno).*|"
    r"(?:consult|consulte|use|utilize)\s+(?:only\s+|apenas\s+)?(?:current\s+sources|"
    r"fontes\s+atuais|Getnet.{0,25}(?:guidance|material)|(?:o\s+)?material\s+interno).*)[.!?]?\s*$"
)


def is_documentary_qualifier(text: str) -> bool:
    if re.fullmatch(r'(?i)\s*(?:answer|respond|responda)\s+(?:in\s+(?:bullet\s+points|a\s+table|short\s+paragraphs)|em\s+(?:t[oó]picos|uma?\s+tabela|par[aá]grafos\s+curtos))[.!]?\s*', text):
        return True
    substantive = re.sub(r'(?i)\be\s+informe\s+se\s+n[aã]o\s+houver\s+confirma[cç][aã]o', '', text)
    # A qualifier cannot contain a new interrogative/imperative clause. A final
    # question mark alone is not enough to locate the factual clause.
    if re.search(r'(?i)(?:[:;]\s*|\b(?:and|e)\s+)(?:what|which|how|why|when|who|qual|quais|como|quanto|por\s+que|explain|tell|show|find|check|list|give|informe|explique|mostre|busque|consulte|liste|verifique)\b', substantive):
        return False
    if re.search(r'(?i)[?].*\S|\b(?:and|e|also|tamb[eé]m)\s+(?:explain|tell|show|find|check|list|give|informe|explique|mostre|busque|consulte|liste|verifique)\b', substantive):
        return False
    return is_source_qualifier_only(text) or bool(_SOURCE_DIRECTIVE.fullmatch(text))


def _public_context(text: str) -> bool:
    """A bounded declarative context can accompany a public manual task.

    Terminal identifiers and account/telemetry state are never inherited.
    """
    return (bool(re.match(r"(?i)\s*(?:I\s+(?:found|have|see)|there\s+is|there's|"
                         r"encontrei|achei|tenho|h[aá]\s+um)\b", text))
            and not re.search(r'(?i)\b(?:GET-[A-Z0-9-]+|firmware|status|saldo|balance|'
                              r'perfil|profile|transactions?|transa[cç][aã]o|account|conta)\b', text))


def canonical_documentary_text(original: str, start: int, end: int, text: str) -> str:
    """Keep referential country restrictions attached to their product identity.

    Inheritance is limited to literal public product names, never old terms,
    geography, private identifiers, or a separate unassigned factual request.
    """
    entities = r"getnet|credi[aá]rio|pix|get\s+conta|payment\s+link|link\s+de\s+pagamento|get\s+tap"
    if (re.search(r'(?i)\b(?:applicability|aplicabilidade|same\s+terms|mesmos\s+termos)\b', text)
            and not re.search(r'(?i)\b(?:' + entities + r')\b', text)):
        previous = original[:start]
        # Inherit the nearest explicit public topic, never its prices, locale,
        # or private state. Public entity vocabulary is shared across countries.
        products = list(re.finditer(r'(?i)\b(?:' + entities + r')\b', previous))
        if products:
            names = [match.group() for match in products[-2:]]
            if names[-1].lower() == 'getnet' and len(names) > 1:
                product = ' '.join(names)
            else:
                product = names[-1]
                if product.lower() != 'getnet' and re.search(r'(?i)\bgetnet\b', previous):
                    product = 'Getnet ' + product
            return product + ': ' + text
    return text


def factual_documentary_text(text: str) -> str:
    """Separate retrieval terms from attached source/format directives.

    The literal canonical unit remains the coverage identity. Source policy is
    evaluated before this function, and external constraints are never removed.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=\S)', text)
    return ' '.join(sentence for sentence in sentences
                    if not is_documentary_qualifier(sentence)
                    or asks_explicit_external_evidence(sentence)).strip()


def is_independent_private_state(text: str) -> bool:
    """Exclude account/telemetry reads from public documentary inheritance."""
    from app.security.operational_intent import classify_operational_intent
    intent = classify_operational_intent(text)
    # A bound predicate is documentary context, rather than an independent
    # private read. The orchestrator still withholds its entire dependent scope
    # until typed proof; this classification grants no predicate authorization.
    if (re.search(r'(?i)\b(?:if|se|when|quando|provided\s+that|desde\s+que|as\s+long\s+as|since)\b', text)
            and re.search(r'(?i)\b(?:firmware|status|terminal|conex[aã]o)\b', text)
            and not re.match(r'(?i)\s*(?:please\s+)?(?:check|inspect|verify|confirm|consulte|verifique|inspecione|confirme)\b', text)):
        return False
    return ((intent.private_scope and not intent.documentary_scope)
            or bool(re.search(r'(?i)\b(?:my\s+(?:account|balance|profile|transactions?)|'
                              r'meu\s+(?:saldo|perfil|cadastro)|minha\s+(?:conta|transa[cç][aã]o))\b', text)))
