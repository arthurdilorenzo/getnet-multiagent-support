"""Bounded relational evidence, with exact paragraph spans and content identity.

This is an inspectable local grammar, not semantic entailment or neural NLP.
Catalog entries are facts about subjects, never questions or canned answers.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from app.rag.embeddings import BilingualDenseEmbedder as Text

SUBJECTS = {
    'classica': ('get classica', 'classic machine'),
    'smart': ('get smart', 'smart pos'),
    'tap': ('get tap',),
    'link': ('link de pagamento', 'payment link', 'get pay'),
    'account': ('get conta', 'conta digital', 'digital account'),
    'advance': ('antecipacao', 'antecipacao de recebiveis', 'receba ja', 'receivables advance'),
    'crediario': ('crediario',),
    'pix': ('pix',),
    'terminal': ('maquininha', 'maquina', 'terminal', 'chip algar'),
}
FEATURES = {
    'physical_keyboard': ('teclado fisico', 'physical keyboard'),
    'printer': ('impressora', 'impressao fiscal', 'printer', 'printing'),
    'camera': ('camera',),
    'app_store': ('get store', 'loja de aplicativos', 'app store'),
    'android': ('android',),
    'barcode_reader': ('leitor de codigos', 'barcode reader'),
    'battery': ('bateria', 'battery'),
    'touchscreen': ('touchscreen', 'tela sensivel'),
}
PAYMENT_METHODS = {
    'credit': ('credito', 'credit'), 'debit': ('debito', 'debit'),
    'card': ('cartao', 'card'), 'pix': ('pix',), 'wallet': ('wallet',),
    'boleto': ('boleto',), 'cash': ('dinheiro', 'cash', 'especie'),
}


def has(text: str, phrase: str) -> bool:
    return bool(re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text))


def mentioned_subjects(text: str) -> List[str]:
    norm = Text.normalize(text)
    found = [(match.start(), subject) for subject, aliases in SUBJECTS.items()
             for alias in aliases for match in re.finditer(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', norm)]
    found.extend((m.start(), 'custom:' + ' '.join(Text.tokenize(m[0]))) for m in re.finditer(r'\b(?:modulo|produto|module|product) (?:getnet )?[a-z]+[0-9]+\b', norm))
    return list(dict.fromkeys(subject for _, subject in sorted(found)))


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    subject: str
    relation: str
    value: str
    chunk_id: str
    content_hash: str
    start: int
    end: int
    excerpt: str
    unit: Optional[str] = None
    polarity: bool = True
    conditions: str = ''
    geographic_scope: str = ''
    source_kind: str = 'curated_local_summary'

    def valid_for(self, chunk_id: str, content: str) -> bool:
        return (self.chunk_id == chunk_id and self.content_hash == hashlib.sha256(content.encode()).hexdigest()
                and 0 <= self.start < self.end <= len(content)
                and content[self.start:self.end] == self.excerpt
                and not self.excerpt.lstrip().startswith('#'))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FactCatalog:
    def __init__(self, facts: Iterable[EvidenceFact] = ()) -> None:
        self.facts = list(facts)

    @classmethod
    def load(cls, path: Path) -> 'FactCatalog':
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding='utf-8'))
        return cls(EvidenceFact(**item) for item in payload.get('facts', []))

    def for_chunk(self, chunk_id: str, content: str) -> List[EvidenceFact]:
        return [fact for fact in self.facts if fact.valid_for(chunk_id, content)]


def extract_facts(chunk_id: str, content: str) -> List[EvidenceFact]:
    """Extract only explicit local sentence patterns; unknown language yields no fact.

    Full paragraph spans retain negation, exceptions and contractual qualifications.
    No subject is inferred from a heading or from an object elsewhere in a chunk.
    """
    digest = hashlib.sha256(content.encode()).hexdigest()
    facts: List[EvidenceFact] = []
    for match in re.finditer(r'[^\n]+', content):
        paragraph = match[0].strip()
        if not paragraph or paragraph.startswith('#') or re.match(r'(?i)^(?:fonte|source|https?://)', paragraph):
            continue
        if paragraph.endswith('?') or len(Text.tokenize(paragraph)) < 4:
            continue
        start = match.start() + len(match[0]) - len(match[0].lstrip())
        end = start + len(paragraph)
        current = None
        clauses = re.split(r'(?<=[.!?])\s+|(?=\b(?:[AaOo]|[Tt]he)\s+(?:Get |Link de |Pix\b|Crediário\b))|,?\s+(?:mas|but|e|and)\s+(?=(?:possui|tem|has|supports|accepts)\b)', paragraph)
        for sentence in clauses:
            norm = Text.normalize(sentence)
            # Subject must precede a predicate. Mentions in lists remain objects.
            predicate = re.search(r'\b(?:possui|tem|reune|permite|aceita|pode|sao|e |adianta|recebe|utiliza|inclui|apresenta|variam|depende|has|supports|accepts|allows)\b', norm)
            prefix = norm[:predicate.start()] if predicate else norm[:90]
            explicit = mentioned_subjects(prefix)
            if explicit:
                current = explicit[0]
            if not current:
                # Procedures may explicitly name their subject after their verb.
                if 'chip' in norm and ('sim1' in norm or 'multioperadora' in norm): current = 'terminal'
                elif ('redefinir a senha de acesso' in norm or 'esqueci a senha' in norm): current = 'app_access'
                else: continue
            subject = current
            negative = bool(re.search(r'\b(?:nao|not|never|nunca)\b', norm[:max(0, norm.find(';'))] if ';' in norm else norm))
            def add(relation: str, value: str = '', unit: Optional[str] = None, polarity: Optional[bool] = None):
                key = (subject, relation, value, unit, start)
                if any((f.subject,f.relation,f.value,f.unit,f.start) == key for f in facts): return
                facts.append(EvidenceFact(f'{chunk_id}:{len(facts)+1}',subject,relation,value,chunk_id,digest,start,end,paragraph,
                                          unit,not negative if polarity is None else polarity,
                                          paragraph if re.search(r'depend|conforme|contrat|elegiv|varia|limite efetivo',Text.normalize(paragraph)) else ''))
            add('overview')
            if subject in {'smart','classica','tap'}:
                for feature, aliases in FEATURES.items():
                    if any(has(norm, alias) for alias in aliases): add('hardware_feature',feature)
                # A new explicitly possessed attribute can be indexed without a new question rule.
                owned = re.search(r'\b(?:possui|tem|has)\s+(.+?)(?:[;.]|$)',norm)
                if owned and not any(any(has(owned[1], a) for a in aliases) for aliases in FEATURES.values()):
                    add('hardware_feature',' '.join(Text.tokenize(owned[1])))
            if subject == 'link' and re.search(r'\b(?:distancia|whatsapp|redes sociais|remotely)\b',norm):
                add('remote_collection')
            if re.search(r'\b(?:aceita|escolher entre|accepts|pagar com)\b', norm):
                for method, aliases in PAYMENT_METHODS.items():
                    if any(has(norm, alias) for alias in aliases): add('payment_method',method)
            if subject == 'advance' and re.search(r'antecip|adianta',norm): add('advance_receivables')
            if subject == 'crediario' and re.search(r'parcel|installment',norm):
                add('installments')
                for number in re.findall(r'\b(?:ate|up to)\s+(\d+)\s*(?:x|vezes|parcelas|installments)\b',norm): add('installment_limit',number,'installments')
            if subject == 'crediario' and re.search(r'limite|juros|parcelas',norm) and re.search(r'\b(?:emissor|issuer)\b',norm):
                add('installment_authority','issuer')
            if re.search(r'receb|recebe|antecip|settle',norm):
                for number, unit in re.findall(r'\b(?:ate|em|up to|within)\s+(\d+)\s+(dias? uteis|dias?|business days?|days?|hours?|horas?)\b',norm):
                    canonical = 'business_days' if unit in {'dia util','dias uteis','business day','business days'} else 'days' if unit in {'dia','dias','day','days'} else 'hours'
                    add('settlement_time',number,canonical)
            if re.search(r'\b(?:taxa|tarifa|juros|encargos|fee|interest)\b',norm):
                add('fees')
                for amount in re.findall(r'\b(\d+(?:[.,]\d+)?)\s*%',sentence): add('fee_value',amount.replace(',','.'),'percent')
            if subject == 'pix' and re.search(r'conta corrente|poupanca|conta de pagamento',norm): add('account_requirement')
            if subject == 'account' and re.search(r'banco tradicional|conta digital gratuita',norm): add('account_requirement')
            if subject == 'pix' and 'cadastrar chave' in norm: add('key_creation')
            if subject == 'app_access' and ('redefinir' in norm or 'esqueci' in norm): add('password_reset')
            if subject == 'terminal' and re.search(r'chip|conect|comunicacao',norm): add('connectivity')
            if 'simulacao' in norm: add('simulation')
            if re.search(r'abertura|abrir|open|sign up',norm): add('signup')
            if re.search(r'frequencia|frequency',norm):
                for number, unit in re.findall(r'(\d+(?:[.,]\d+)?)\s+(terahertz|gigahertz|megahertz|hertz|hz)\b',norm): add('frequency',number.replace(',','.'),unit)
            # Geographic scope must occur explicitly in the same source sentence.
            from app.rag.answerability import GEOGRAPHY
            geographic = [country for country, aliases in GEOGRAPHY.items() if any(has(norm,a) for a in aliases)]
            if len(geographic) == 1:
                facts[:] = [replace(f, geographic_scope=geographic[0]) if f.start == start and f.subject == subject else f for f in facts]
    return facts
