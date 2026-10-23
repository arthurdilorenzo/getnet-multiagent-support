"""Conservative, inspectable answerability over body evidence.

This is a bilingual domain ontology and constraint check, not a universal semantic
judge. Similarity proposes candidates; product/operation/facet coverage decides
whether their bodies contain usable evidence. No questions or canned answers live
here. Unknown essential constraints produce abstention.
"""
import re
import unicodedata
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


def parse_requirements(query: str) -> Dict[str, Any]:
    """Bounded PT/EN request grammar. Incidental tokens are not fact requirements.

    Only predicate complements, subjects, values and explicit scope constrain the
    evidence. Unrecognized property/operation constructions remain unknown.
    """
    from app.rag.facts import SUBJECTS, FEATURES, PAYMENT_METHODS, mentioned_subjects
    q = Text.normalize(query)
    subjects = mentioned_subjects(query)
    # Generic hardware words identify a terminal, not an additional POS model.
    if len(subjects) > 1 and 'terminal' in subjects:
        subjects.remove('terminal')
    requested = concepts(query)
    if not subjects:
        if 'remote_payment' in requested: subjects = ['link']
        elif 'advance' in requested: subjects = ['advance']
        elif 'installments' in requested: subjects = ['crediario']
        elif 'connectivity' in requested: subjects = ['terminal']
        elif 'account' in requested: subjects = ['account']
    # A named payment after an acceptance verb is its object, not its subject.
    acceptance = re.search(r'\b(?:aceit\w*|accept\w*|pagar|pay|usando|using)\b', q)
    if acceptance and len(subjects) > 1:
        leading = mentioned_subjects(q[:acceptance.start()])
        if leading: subjects = leading
    countries = [country for country, aliases in GEOGRAPHY.items() if any(_has(q,a) for a in aliases)]
    # Unknown named territories are preserved without expanding a country list.
    for scope in re.findall(r'\b(?:no|na|em|in)\s+([A-ZÀ-Ý][\wÀ-ÿ-]*(?:\s+[A-ZÀ-Ý][\wÀ-ÿ-]*)*)',query):
        normalized = Text.normalize(scope).strip()
        if any(any(alias.startswith(normalized) or normalized.startswith(alias) for alias in aliases) for aliases in SUBJECTS.values()): continue
        if not any(normalized in aliases for aliases in GEOGRAPHY.values()): countries.append(normalized)
    foreign = bool(re.search(r'\b(?:estrangeir\w*|foreign|abroad|non resident|nao residente|exterior|internacional)\b',q))
    requirements = []
    def add(relation, value='', unit=None, targets=None):
        for subject in targets if targets is not None else subjects or ['unknown']:
            item = {'subject':subject,'relation':relation,'value':value,'unit':unit,
                    'countries':countries,'foreign_eligibility':foreign,
                    'query_span':[0,len(query)]}
            if item not in requirements: requirements.append(item)
    # Operations cannot be proven by a general product overview.
    operation_patterns = {
        'password_reset':r'redefin|reset|recuperar|esqueci',
        'password':r'\b(?:senha|password|passcode|pin)\b',
        'cancellation':r'cancel|estorn|refund|devolv',
        'dispute':r'chargeback|contesta|disput',
        'expiry':r'validade|expira|expiration|expiry|vencimento do link',
        'signup':r'habilit|contrat|ativar|enable|activate|sign up|\bopen\b|cadastro de estabelecimento',
    }
    for relation, pattern in operation_patterns.items():
        if re.search(pattern,q):
            if relation == 'password' and re.search(operation_patterns['password_reset'],q): continue
            target = ['app_access'] if relation == 'password_reset' and not subjects and re.search(r'aplicativo|portal|app|access|acesso',q) else None
            add(relation,targets=target)
    if re.search(r'cadastr\w* chave|criar\w* chave|register.*key|create.*key',q): add('key_creation')
    elif re.search(r'\b(?:criar|gerar|create|generate)\b',q): add('creation')
    if re.search(r'\b(?:taxas?|tarifas?|juros|fees?|interest|custos?|cost)\b',q): add('fees')
    installment = 'installments' in requested or ('crediario' in subjects and re.search(r'parcel|limit|maximum|maximo',q))
    if installment:
        if re.search(r'quem.*(?:defin|determin)|who.*(?:sets|determines|decides)|emissor|issuer',q): add('installment_authority')
        elif re.search(r'quantas|limite|maxim|how many|up to|ate \d|\d+\s*x',q): add('installment_limit')
        else: add('installments')
    time_requested = bool(re.search(r'prazo|quando.*(?:receb|cai)|how long|when (?:will|do)|settlement time|\d+\s+(?:dias?|days?|horas?|hours?)',q))
    if time_requested and not re.search(r'validade|expira|vencimento do link',q): add('settlement_time')
    # Values retain units and are checked on the same subject/relation fact.
    numbers = []
    numeric_q = ''.join(c for c in unicodedata.normalize('NFKD',query.lower()) if not unicodedata.combining(c))
    for match in re.finditer(r'\b(\d+(?:[.,]\d+)?)\s*(x|vezes|parcelas|installments|dias uteis|business days|dias?|days?|horas?|hours?|%)',numeric_q):
        amount, unit = match.groups()
        canonical = 'installments' if unit in {'x','vezes','parcelas','installments'} else 'business_days' if unit in {'dias uteis','business days'} else 'hours' if unit in {'hora','horas','hour','hours'} else 'percent' if unit == '%' else 'days'
        relation = 'installment_limit' if canonical == 'installments' else 'fee_value' if canonical == 'percent' else 'settlement_time'
        numbers.append(relation)
        add(relation,amount.replace(',','.'),canonical)
    if numbers:
        requirements = [r for r in requirements if r['value'] or r['relation'] not in numbers]
    for date in re.findall(r'\b\d{4}-\d{2}-\d{2}\b',query): add('effective_date',date)
    # A requested tender is a value even when phrased without an acceptance verb.
    tender_context = bool(re.search(r'aceit|accept|pagar|pay with|cobrar.*(?:usando|com)|collect.*(?:using|with)',q))
    if tender_context:
        object_text = q[acceptance.end():] if acceptance else q
        methods = [key for key,aliases in PAYMENT_METHODS.items() if any(_has(object_text,a) for a in aliases)]
        for method in methods: add('payment_method',method)
        if not methods and re.search(r'aceit|accept',q): add('payment_method','unknown')
    # A qualifier attached with com/with is a request constraint, unlike a
    # courtesy phrase or a shop/customer introduction before the question.
    for qualifier in re.findall(r'\b(?:com|with)\s+(.+?)(?=[?.;]|$)',query,flags=re.I):
        value = Text.normalize(qualifier).strip()
        if mentioned_subjects(value) or value == 'getnet': continue
        if any(any(_has(value,a) for a in aliases) for aliases in PAYMENT_METHODS.values()):
            for method,aliases in PAYMENT_METHODS.items():
                if any(_has(value,a) for a in aliases): add('payment_method',method)
        elif not any(any(_has(value,a) for a in aliases) for aliases in FEATURES.values()):
            add('property',' '.join(Text.tokenize(value)))
    feature_requested = False
    if any(subject in {'smart','classica','tap'} for subject in subjects):
        for feature, aliases in FEATURES.items():
            if any(_has(q,a) for a in aliases):
                # An attribute such as battery autonomy is not the existence of a battery.
                attr = 'battery_autonomy' if feature == 'battery' and re.search(r'autonomia|duration|how long|life',q) else feature
                add('hardware_feature',attr); feature_requested = True
        predicate = re.search(r'\b(?:possui|tem|has|have|support\w*|include\w*)\s+(.+)',q)
        if predicate and not feature_requested:
            add('hardware_feature',' '.join(Text.tokenize(predicate[1]))); feature_requested = True
        elif not feature_requested and re.search(r'resistent|waterproof|biometri|autonomia|compativel|compatible|\b(?:faz|does)\b',q):
            add('hardware_feature','unknown'); feature_requested = True
    if re.search(r'simulacao|simular|simulation|simulate',q): add('simulation')
    if re.search(r'frequencia|frequency',q): add('frequency')
    if 'remote_payment' in requested and not tender_context: add('remote_collection',targets=['link'] if not subjects else None)
    if 'advance' in requested and not time_requested and not any(r['relation'] in {'fees','signup','cancellation'} for r in requirements): add('advance_receivables')
    if 'connectivity' in requested: add('connectivity')
    if re.search(r'conta (?:bancaria|corrente)|bank account|banco tradicional|traditional bank',q):
        add('account_requirement',targets=[s for s in subjects if s in {'pix','account'}] or ['account'])
    if foreign: add('eligibility')
    # Overview has a narrow grammar; unknown substantive operations do not fall through.
    overview = bool(re.search(r'funciona|funcionam|funcionamento|diferenc|compar|difference|what is|what are|works|work|expli|explain|tell me|sobre|about|como.*permite|\bo que\b',q))
    if not requirements:
        add('overview' if overview else 'unknown')
    return {'requirements':requirements,'parser':'bounded_local_relations_v4',
            'parser_limit':'Gramática local PT/EN delimitada; não é entailment semântico universal.'}


def assess(query: str, content: str, title: str = '', *, facts=None, chunk_id: str = 'inline') -> Dict[str, Any]:
    from app.rag.facts import extract_facts
    parsed = parse_requirements(query)
    valid = [f for f in (extract_facts(chunk_id,content) if facts is None else facts) if f.valid_for(chunk_id,content)]
    coverage = []
    selected = []
    for requirement in parsed['requirements']:
        matches = []
        for fact in valid:
            if fact.subject != requirement['subject'] or fact.relation != requirement['relation']: continue
            value = requirement['value']
            if value and value != fact.value and not (value == 'card' and fact.value in {'credit','debit'}): continue
            if requirement['unit'] and requirement['unit'] != fact.unit: continue
            if requirement['countries'] and not all(country == fact.geographic_scope for country in requirement['countries']): continue
            if requirement['foreign_eligibility'] and fact.relation != 'eligibility': continue
            matches.append(fact)
        coverage.append({'requirement':requirement,'status':'supported' if matches else 'unknown','facts':[f.to_dict() for f in matches]})
        selected.extend(f.excerpt for f in matches)
    missing = [r['requirement']['subject']+':'+r['requirement']['relation']+(':'+r['requirement']['value'] if r['requirement']['value'] else '') for r in coverage if r['status']=='unknown']
    return {'answerable':bool(coverage) and not missing,'missing':missing,'coverage':coverage,
            'requested_concepts':sorted(concepts(query)),'body_concepts':sorted(concepts(content)),
            'answer_snippets':list(dict.fromkeys(selected)),
            'parser':parsed['parser'],'limit':parsed['parser_limit']}

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
