"""Conservative, inspectable answerability over body evidence.

This is a bilingual domain ontology and constraint check, not a universal semantic
judge. Similarity proposes candidates; product/operation/facet coverage decides
whether their bodies contain usable evidence. No questions or canned answers live
here. Unknown essential constraints produce abstention.
"""
import re
from typing import Any, Dict, List, Set
from app.rag.embeddings import BilingualDenseEmbedder as Text

# Features describe capabilities, not the ten demonstration questions.
CONCEPTS = {
    'pix': ('pix',),
    'crediario': ('crediario',),
    'classica': ('get classica', 'classic machine'),
    'smart': ('get smart', 'smart pos'),
    'tap': ('get tap',),
    'link': ('link de pagamento', 'payment link', 'get pay', 'whatsapp', 'redes sociais', 'social media'),
    'account': ('get conta', 'conta bancaria', 'conta digital', 'bank account', 'traditional bank'),
    'advance': ('antecip', 'recebiveis', 'adiant', 'receba ja', 'receivables', 'advance', 'unsettled', 'before maturity', 'before their due date', 'before the due date', 'nao vencidas', 'a vencer'),
    'remote_payment': ('a distancia', 'longe', 'remoto', 'remota', 'nao esta presente', 'nao esta na loja', 'not physically', 'not at my store', 'not present', 'without being', 'remotely', 'remote customer', 'distance', 'online', 'whatsapp', 'redes sociais'),
    'connectivity': ('conect', 'internet', 'wi fi', 'wifi', 'chip', 'sinal', 'network', 'connection'),
    'installments': ('parcel', 'installment', 'split a sale', 'divide a sale', 'dividir'),
}
# Query wording and evidence wording may differ. Each requested operation must
# occur in the BODY for the same product. A password mention in a card manual
# cannot prove how to reset a Pix password.
FACETS = {
    'password': (('senha', 'password', 'passcode', 'pin'), ('senha', 'password', 'passcode', 'pin')),
    'reset': (('redefin', 'reset', 'recuperar', 'esqueci'), ('redefin', 'reset', 'recuper', 'esqueci')),
    'cancellation': (('cancel', 'estorn', 'refund', 'devolver'), ('cancel', 'estorn', 'refund', 'devol')),
    'dispute': (('chargeback', 'contesta', 'disput'), ('chargeback', 'contesta', 'disput')),
    'expiry': (('validade', 'expira', 'expiration', 'expiry', 'expire', 'expires', 'vencimento do link'), ('validade', 'expira', 'expiration', 'expiry', 'expire', 'vence')),
    'fees': (('taxa', 'tarifa', 'juros', 'fee', 'interest', 'custo', 'cost'), ('taxa', 'tarifa', 'juros', 'fee', 'interest', 'custo', 'cost', 'cobranca')),
    'settlement_time': (('prazo', 'quando recebo', 'quando cai', 'how long', 'when will', 'when do', 'settlement time'), ('dia', 'prazo', 'd+2', 'd+1', 'hours', 'days', 'recebimento')),
    'enablement': (('habilit', 'contrat', 'ativar', 'enable', 'activate', 'sign up'), ('habilit', 'contrat', 'ativ', 'analise', 'cadastro', 'enable')),
    'creation': (('criar', 'gerar', 'create', 'generate'), ('criar', 'gerar', 'cadastro', 'cadastrar', 'compartilha', 'create', 'generate')),
    'limit': (('quantas', 'maxim', 'limite', 'how many', 'maximum'), ('ate', 'limite', 'maxim', 'emissor', 'up to')),
}
COUNTRIES = ('argentina','chile','mexico','mexican','colombia','peru','uruguai','uruguay','paraguai','paraguay','portugal','espanha','spain','estados unidos','united states','brasil','brazil')
GEOGRAPHY={
    'argentina':('argentina','argentino','argentine','argentinian'), 'chile':('chile','chileno','chilean'),
    'mexico':('mexico','mexicano','mexican'), 'colombia':('colombia','colombiano','colombian'),
    'uruguai':('uruguai','uruguaio','uruguay','uruguayan'), 'paraguai':('paraguai','paraguaio','paraguayan'),
    'peru':('peru','peruano','peruvian'), 'portugal':('portugal','portugues','portuguesa','portuguese'),
    'espanha':('espanha','espanhol','espanhola','spain','spanish'),
    'estados unidos':('estados unidos','united states','americano','americana','american','estadunidense'),
    'reino unido':('reino unido','united kingdom','britanico','britanica','british'),
    'alemanha':('alemanha','alemao','alema','germany','german'),
    'franca':('franca','frances','francesa','france','french'),
    'brasil':('brasil','brazil','brasileiro','brasileira','brazilian'),
}


def _has(text: str, phrase: str) -> bool:
    # Stems are intentional; boundaries prevent "pin" matching "opinion".
    stems={'antecip','adiant','conect','parcel','redefin','recuper','cancel','estorn','devol','contesta','disput','expira','habilit','contrat','ativ','maxim'}
    ending='' if phrase in stems else r'(?:s)?(?!\w)'
    return bool(re.search(r'(?<!\w)' + re.escape(phrase) + ending, text))


def concepts(text: str) -> Set[str]:
    norm = Text.normalize(text)
    return {key for key, words in CONCEPTS.items() if any(_has(norm, word) for word in words)}


def body_snippets(content: str) -> List[str]:
    """Headers, URL lists and page furniture cannot satisfy answerability."""
    result=[]
    for line in content.splitlines():
        line=line.strip()
        if not line or line.startswith('#') or re.match(r'(?i)^(?:fonte|source|https?://|evidencia|capturad[oa])', line):
            continue
        plain=re.sub(r'https?://\S+', '', line)
        if len(Text.tokenize(plain)) < 4 or plain.rstrip().endswith('?'):
            continue
        # Preserve the exact local evidence text for auditable extraction.
        result.append(line)
    return result


def assess(query: str, content: str, title: str = '') -> Dict[str, Any]:
    snippets=body_snippets(content)
    q=Text.normalize(query)
    body=Text.normalize(' '.join(snippets))
    requested=concepts(query)
    present=concepts(body)
    missing=[]
    # A product must be in the actual body, not just navigation/title/URL.
    explicit=requested & {'pix','crediario','classica','smart','tap','link','advance'}
    for product in explicit:
        if product not in present:
            missing.append('product:'+product)
    if 'remote_payment' in requested and not (present & {'remote_payment','link'}):
        missing.append('capability:remote_payment')
    if 'installments' in requested and 'installments' not in present:
        missing.append('capability:installments')
    if 'account' in requested and 'account' not in present:
        missing.append('product:account')
    if 'connectivity' in requested and 'connectivity' not in present:
        missing.append('capability:connectivity')
    for facet,(query_words,evidence_words) in FACETS.items():
        if any(_has(q,word) for word in query_words) and not any(_has(body,word) for word in evidence_words):
            missing.append('facet:'+facet)
    for country,aliases in GEOGRAPHY.items():
        if any(_has(q,alias) for alias in aliases) and not any(_has(body,alias) for alias in aliases):
            missing.append('country:'+country)
    if any(_has(q,term) for term in ('estrangeiro','estrangeira','foreign','abroad','non resident','nao residente','fora do brasil','exterior','internacional')):
        if not any(_has(body,term) for term in ('estrangeiro','estrangeira','foreign','abroad','non resident','nao residente','fora do brasil','exterior','internacional')):
            missing.append('eligibility:foreign_or_nonresident')
    # Installment limits and their decision-maker need actual cardinality or
    # authority evidence. A fee/simulation paragraph alone proves neither.
    limit_requested=any(_has(q,term) for term in ('quantas','limite','maxim','how many','maximum','quem define','quem determina','who sets','who determines'))
    if 'installments' in requested and limit_requested:
        cardinality=bool(re.search(r'\b(?:ate|up to)\s+\d+\s*(?:x|vezes|parcelas|installments)?\b',body))
        authority=bool(re.search(r'\b(?:limite|maximo|maximum|parcelas)\b.{0,100}\b(?:emissor|issuer|banco|bank)\b|\b(?:emissor|issuer)\b.{0,100}\b(?:limite|maximum|parcelas)\b',body))
        if not (cardinality or authority): missing.append('facet:installment_limit')
        asks_authority=any(_has(q,term) for term in ('quem define','quem determina','who sets','who determines','who decides'))
        if asks_authority and not authority: missing.append('facet:installment_authority')
    # Named dates and explicit percentages constrain the claim. The body must
    # mention them; shared topical vocabulary cannot establish their truth.
    for token in re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:[.,]\d+)?\s*%', query):
        if Text.normalize(token) not in body:
            missing.append('constraint:'+token)
    if not snippets:
        missing.append('substantive_body')
    matched=Text.tokenize(body)
    query_tokens=set(Text.tokenize(query))-{'getnet','funciona','funcionar','posso','pode','possible','would','please','preciso','quero','gostaria','need','get','customer','cliente','store','loja'}
    overlaps=query_tokens & set(matched)
    meaningful = bool(requested & present) or bool(overlaps)
    if not meaningful:
        missing.append('topic_evidence')
    # Unknown properties must not be answered by a generic product paragraph.
    # Consume the ontology's vocabulary and ordinary question scaffolding,
    # then require the remaining content words in the actual product evidence.
    semantic_words=set()
    for country,aliases in GEOGRAPHY.items():
        if any(_has(q,alias) for alias in aliases):
            for alias in aliases:
                semantic_words.update(Text.tokenize(alias))
    for key in requested:
        for phrase in CONCEPTS[key]:
            semantic_words.update(Text.tokenize(phrase))
    for words,_ in FACETS.values():
        for phrase in words:
            if _has(q,phrase): semantic_words.update(Text.tokenize(phrase))
    scaffolding=set(('getnet funciona funcionar funcionamento diferenca diferencas diferenca entre comparativo comparar compare difference differences between '
        'preciso quero gostaria saber entender explique explicar sobre informacoes informacao posso pode podem tem ter possui usa usar uso quem define determina numero '
        'vender venda vendas receber recebimento dinheiro cobrar cliente clientes comprador loja estabelecimento atraves usando necessito cartao ainda via '
        'possible want need know tell explain works work information support supports use have has take collect request customer customers sell through using merchant open '
        'buyer payments payment sales sale card cash takings money not physically present store without opening checking account who defines sets determines receive receiving '
        'before their contractual due date from at who with get is does it e acompanhamento pergunta contexto produto atributo qual quais').split())
    residual=set(Text.tokenize(query))-semantic_words-scaffolding
    residual={term for term in residual if not any(len(word)>=4 and term.startswith(word) for word in semantic_words)}
    def covered(term,text):
        return any(term==word or (len(term)>=5 and len(word)>=5 and term[:5]==word[:5]) for word in Text.tokenize(text))
    for term in sorted(residual):
        if not covered(term,body):
            missing.append('property:'+term)
        elif explicit and term not in {'hoje','today'}:
            sentences=re.split(r'(?<=[.!?])\s+', ' '.join(snippets))
            if not any(covered(term,sentence) and explicit <= concepts(sentence) for sentence in sentences):
                # Do not assign Clássica's keyboard to Smart merely because both
                # products occur somewhere in a comparative document.
                missing.append('property_subject:'+term)
    return {'answerable':not missing,'missing':missing,'requested_concepts':sorted(requested),'body_concepts':sorted(present),'answer_snippets':snippets}


def expanded_query(query: str) -> str:
    """Translate capability vocabulary before similarity scoring."""
    props=concepts(query)
    expansions={
        'remote_payment':'link de pagamento whatsapp vendas online distancia',
        'advance':'antecipacao recebiveis adiantar vendas credito',
        'installments':'parcelamento parcelas crediario',
        'account':'conta bancaria get conta digital',
        'connectivity':'conexao rede chip internet',
    }
    return query+' '+' '.join(value for key,value in expansions.items() if key in props)
