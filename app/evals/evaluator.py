import json
import re
import math
import unicodedata
import uuid
from contextlib import nullcontext
from app.evals.offline_transport import synthetic_external_transport, external_services_mode
from decimal import Decimal, InvalidOperation
from app.rag.embeddings import BilingualDenseEmbedder as Text
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.agents.orchestrator import orchestrator
from app.config import settings
from app.schemas import ChatRequest, ChatResponse
from app.security.pii import PIIRedactor


class GroundednessJudge:
    """Limited evidence-consistency validator, not a universal semantic judge.

    Checks numeric support, explicit polarity conflicts, forbidden claims,
    verbatim source extracts, tool-field consistency and safe abstention. The
    independent audit must still label answerability and factual entailment.
    A URL and lexical terms alone never prove correctness.
    """
    FORBIDDEN_HALLUCINATION_STRINGS = [
        'Serviço Meteorológico Regional (Cache)',
        'Banco Central / Mercado Financeiro (Referência)',
        'O tema foi verificado com sucesso nas fontes externas públicas',
    ]
    NEGATION=re.compile(r'\b(?:nao|not|never|nunca|nenhum|impossivel|doesn t|cannot|can t)\b')

    @staticmethod
    def _strip_links(text):
        return re.sub(r'https?://[^\s)\]]+', '', text)

    @classmethod
    def _numbers(cls,text):
        values=set()
        for number in re.findall(r'(?<![\w])[-+]?\d+(?:[.,]\d+)*(?:x)?(?![\w])',cls._strip_links(text)):
            number=number.rstrip('x')
            if ',' in number and '.' in number:
                number=number.replace('.','').replace(',','.') if number.rfind(',')>number.rfind('.') else number.replace(',','')
            else:
                number=number.replace(',','.')
            try:
                values.add(str(Decimal(number).normalize()))
            except InvalidOperation:
                pass
        return values

    @staticmethod
    def _sentences(text):
        # Decimal points do not terminate sentences.
        return [part.strip() for part in re.split(r'(?<!\d)[.!?](?:\s|$)|\n+',text) if part.strip()]

    @classmethod
    def _reference_claim(cls, text):
        """Normalize only complete known source-attribution formatter suffixes.

        The factual predicate remains literal. Unknown tails and added conditions
        stay in the claim; this does not authorize unrestricted paraphrases.
        """
        clean = cls._strip_links(text).strip().rstrip('.')
        canonical = re.sub(
            r', (?:conforme o manual de uso(?: dispon[ií]vel no [ií]ndice p[úu]blico da fonte prim[áa]ria)?|'
            r'como mostra seu manual de uso)$', '', clean, flags=re.I)
        return Text.normalize(canonical).strip()

    @classmethod
    def _abstention_is_clean(cls,text):
        stripped=cls._strip_links(text)
        for sentence in cls._sentences(stripped):
            norm=Text.normalize(sentence)
            if norm.strip()==Text.normalize('Sou o assistente virtual da Getnet e estou disponível para auxiliar com dúvidas sobre maquininhas, Pix, vendas e suporte técnico').strip():
                continue
            safe=any(marker in norm for marker in (
                'nao encontrei','nao foi possivel','nao consigo','nao posso confirmar',
                'nao tenho','nao ha evidencia','sem evidencia','sem informac',
                'insuficiente','indisponivel','nao informad','nao verificad',
                'nao exibimos','nao foi executad','nao execut','nao foi realizad',
                'nao realiz','nao confirmou','nao confim','nao forneca',
                'tente novamente','entre em contato','documentacao','com seguranca',
                'nao efetuei','nenhuma acao','nenhum comando','aguardando confirmacao',
                'confirmacao pendente','pendente de confirmacao','proposta','consulte',
                'i cannot','i could not','insufficient','unavailable','no evidence',
                'solicitacao bloqueada','codigo de auditoria'))
            # Even a refusal with "documentação" is not permission to append
            # an unsupported fee, deadline, payment, or completed action.
            asserts=bool(re.search(r'\b(?:a taxa e|taxa de|pagamento ocorre|paga em|permite|aceita|reiniciei|executei|depositei|rebooted|fee is|pays in)\b',norm))
            if asserts and not any(x in norm for x in ('nao posso confirmar','nao consigo confirmar','nao ha evidencia','nao foi executad','nao executei','nao reiniciei')):
                return False
            if cls._numbers(sentence) and not safe:
                return False
            if not safe and len(Text.tokenize(sentence))>=5:
                return False
        return True

    @classmethod
    def _score_partial(cls, case, response):
        """Check factual extracts separately from explicit unmet request units.

        A supported subset is neither a hallucination nor complete fulfillment.
        Do not treat a quoted unanswered question as an asserted source fact.
        """
        def invalid(reason):
            return {'groundedness_score':0.0,'passed':False,'fulfillment_passed':False,
                    'judge_scope':'limited_partial_evidence_consistency','reason':reason,
                    'semantic_review_required':True}
        if not response.abstained:
            return invalid('Status parcial inconsistente com a declaração de incompletude.')
        marker='\n\nPartes do pedido sem resposta verificada:\n'
        unavailable='\n\nNão foi possível verificar parte das consultas solicitadas; os resultados acima cobrem somente as consultas com evidência disponível.'
        text=response.response
        has_unavailable=text.endswith(unavailable)
        if has_unavailable:
            text=text[:-len(unavailable)]
        unresolved=[]
        if marker in text:
            claims, quoted=text.split(marker,1)
            original=PIIRedactor.mask_text(str(case.get('message','')))
            for line in quoted.splitlines():
                if not line.startswith('- ') or not line[2:].strip():
                    return invalid('Unidade não respondida sem delimitação verificável.')
                unit=line[2:].strip()
                if not original or ' '.join(unit.split()) not in ' '.join(original.split()):
                    return invalid('Parte não respondida não corresponde literalmente ao pedido original.')
                unresolved.append(unit)
        elif has_unavailable:
            claims=text
        else:
            return invalid('Resposta parcial precisa explicitar a lacuna ou indisponibilidade.')
        if not claims.strip():
            return invalid('Resposta parcial sem trecho factual verificável.')
        factual_case={**case,'expect_abstained':False,'required_grounding_terms':[]}
        factual=response.model_copy(update={'response':claims,'abstained':False,'answer_status':'complete'})
        checked=cls.score_response(factual_case,factual)
        consistent=bool(checked['passed'])
        fulfilled=consistent and case.get('expected_answer_status')=='partial'
        return {**checked,'passed':fulfilled,'fulfillment_passed':fulfilled,
                'partial_evidence_consistent':consistent,'unresolved_units':unresolved,
                'judge_scope':'limited_partial_evidence_consistency',
                'reason':checked.get('reason') if not consistent else
                         'Extratos sustentados; cobertura parcial declarada, sem aprovação implícita do pedido inteiro.',
                'semantic_review_required':True}

    @classmethod
    def score_response(cls,case:Dict[str,Any],response:ChatResponse)->Dict[str,Any]:
        if response.answer_status=='partial':
            return cls._score_partial(case,response)
        text=response.response
        # A literal unanswered question is not an asserted fact. Validate its
        # boundary before excluding it, preserving every preceding claim.
        marker='\n\nPartes do pedido sem resposta verificada:\n'
        if response.abstained and marker in text:
            claims, quoted=text.split(marker,1)
            original=' '.join(PIIRedactor.mask_text(str(case.get('message',''))).split())
            lines=quoted.splitlines()
            valid=bool(claims.strip() and original and lines) and all(
                line.startswith('- ') and line[2:].strip() and
                ' '.join(line[2:].split()) in original for line in lines)
            if not valid:
                return {'groundedness_score':0.0,'passed':False,
                        'reason':'Pendência sem correspondência literal verificável no pedido original.',
                        'judge_scope':'limited_abstention_claim_check'}
            text=claims
        norm=Text.normalize(text)
        checks=[]
        def fail(reason, **details):
            return {'groundedness_score':0.0,'passed':False,'reason':reason,
                    'judge_scope':'limited_evidence_consistency','claim_checks':checks,**details}
        for forbidden in cls.FORBIDDEN_HALLUCINATION_STRINGS:
            if forbidden.lower() in text.lower():
                return fail('Fallback não verificável detectado.')
        for forbidden in case.get('forbidden_claims',[])+case.get('forbidden_response_terms',[]):
            if Text.normalize(str(forbidden)) in norm:
                return fail('Alegação explicitamente proibida pelo caso: '+str(forbidden))
        abstention_expected=case.get('expect_abstained') is True
        if response.abstained:
            clean=cls._abstention_is_clean(text)
            if not clean:
                return fail('Flag de abstenção acompanhada de alegação factual sem suporte.')
            if abstention_expected or case.get('category')=='web_search':
                return {'groundedness_score':1.0,'passed':abstention_expected,
                        'reason':'Abstenção limpa; atendimento não comprovado.' if not abstention_expected else 'Abstenção sem alegações materiais não suportadas.',
                        'judge_scope':'limited_abstention_claim_check','availability_passed':False if case.get('category')=='web_search' else None,
                        'fulfillment_passed':abstention_expected}
            return fail('Absteve quando o caso exigia uma resposta coberta.')
        if abstention_expected:
            return fail('Falhou em abster-se diante de evidência insuficiente.')

        evidence=list(getattr(response,'evidence',[]) or [])
        references=[]
        for key in ('reference_facts','reference_answer','reference_evidence','expected_facts','supporting_excerpts'):
            value=case.get(key,[])
            if isinstance(value,str): references.append(value)
            elif isinstance(value,dict): references.append(json.dumps(value,ensure_ascii=False))
            elif isinstance(value,list):
                references.extend(str(v.get('excerpt',v)) if isinstance(v,dict) else str(v) for v in value)
        attached=[]
        for item in evidence:
            if item.get('kind')=='source_excerpt' and item.get('excerpt'):
                attached.append(str(item['excerpt']))
            elif item.get('kind')=='tool_fact' and item.get('value') is not None:
                attached.append(str(item['fact_path'])+': '+json.dumps(item['value'],ensure_ascii=False))
            for key in ('observed_at','verified_at'):
                if item.get(key): attached.append(str(item[key]))
        if not references and not attached:
            return fail('Sem evidência de conteúdo: termos e URL não estabelecem entailment.',evidence_available=False)
        # A frozen reference takes precedence over facts asserted by the candidate.
        support='\n'.join(references or attached)
        unsupported_numbers=cls._numbers(text)-cls._numbers(support)
        if unsupported_numbers:
            return fail('Número sem suporte nas evidências.',unsupported_numbers=sorted(unsupported_numbers))
        required=case.get('required_grounding_terms',[])
        def term_present(term):
            if re.fullmatch(r'\d+(?:[.,]\d+)?',str(term)):
                return bool(cls._numbers(str(term)) & cls._numbers(text))
            return bool(re.search(r'(?<!\w)'+re.escape(Text.normalize(term).strip())+r'(?!\w)',norm))
        matched=sum(1 for term in required if term_present(term))
        if required and matched!=len(required):
            return fail('Faltam fatos obrigatórios do caso.',matched_terms=matched,total_required_terms=len(required))
        anchors=required or [str(x) for x in cls._numbers(support)]
        response_sentences=cls._sentences(text)
        support_sentences=cls._sentences(support)
        for anchor in anchors:
            a=Text.normalize(anchor)
            r_sents=[Text.normalize(x) for x in response_sentences if a in Text.normalize(x)]
            s_sents=[Text.normalize(x) for x in support_sentences if a in Text.normalize(x)]
            if not s_sents: continue
            for sentence in r_sents:
                r_negative=bool(cls.NEGATION.search(sentence))
                if all(bool(cls.NEGATION.search(ref)) != r_negative for ref in s_sents):
                    return fail('Contradição explícita de polaridade em torno de: '+str(anchor))
        # Matching words or numbers cannot establish WHO controls a condition.
        # This checker has no NLI model: certify independent reference extracts
        # only; unrestricted paraphrases require independent semantic review.
        if references:
            reference_text = Text.normalize(cls._strip_links(support))
            canonical_references = {cls._reference_claim(sentence) for sentence in support_sentences}
            for sentence in response_sentences:
                if sentence.startswith('['):
                    continue
                claim = Text.normalize(cls._strip_links(sentence)).strip()
                if claim and claim not in reference_text and cls._reference_claim(sentence) not in canonical_references:
                    return fail('Alegação não extrativa exige revisão semântica independente.',
                                unsupported_claim=sentence, semantic_review_required=True)
        # For attached extracts, substantive non-source text must be verbatim.
        # Typed facts use deterministic renderers; validate the numeric field
        # association as well as the global numeric allowlist.
        if not references and evidence:
            excerpts='\n'.join(str(e.get('excerpt','')) for e in evidence if e.get('kind')=='source_excerpt')
            if excerpts and not any(e.get('kind')=='tool_fact' for e in evidence):
                for sentence in response_sentences:
                    if sentence.startswith('['): continue
                    clean=Text.normalize(cls._strip_links(sentence)).strip()
                    if clean and clean not in Text.normalize(excerpts):
                        return fail('Prosa não extrativa sem referência factual independente.',unsupported_claim=sentence)
            field_labels={
                'max_temp_celsius':('maxima','maximum'), 'min_temp_celsius':('minima','minimum'),
                'rain_probability_percent':('precipitacao','rain'),
                'high_today_brl':('maxima informada',), 'low_today_brl':('minima informada',),
                'current_rate_brl':('cotacao informada','taxa de referencia'),
            }
            for e in evidence:
                field=e.get('fact_path')
                if field not in field_labels or not isinstance(e.get('value'),(int,float)): continue
                supported=cls._numbers(str(e['value']))
                for sentence in response_sentences:
                    if any(label in Text.normalize(sentence) for label in field_labels[field]):
                        # A temperature line contains two labeled values; split
                        # each label span before comparing its nearest number.
                        for label in field_labels[field]:
                            folded=''.join(c for c in unicodedata.normalize('NFKD',sentence.lower()) if not unicodedata.combining(c))
                            tail=folded.split(label,1)
                            if len(tail)>1:
                                found=re.search(r'[-+]?\d+(?:[.,]\d+)?',tail[1])
                                if found and not cls._numbers(found[0]) <= supported:
                                    return fail('Valor atribuído ao campo errado: '+field)
        expected_url=case.get('expected_url_substring')
        url_ok=not expected_url or expected_url.lower() in (' '.join(response.sources)+' '+text).lower()
        if not url_ok: return fail('Fonte esperada ausente; URL não é substituto de suporte factual.')
        checks.append({'numeric_support':True,'explicit_polarity_conflicts':False,'reference_kind':'independent_case' if references else 'attached_runtime_evidence'})
        return {'groundedness_score':1.0,'passed':True,'judge_scope':'limited_evidence_consistency',
                'claim_checks':checks,'matched_terms':matched,'total_required_terms':len(required),
                'url_citation_verified':bool(url_ok),'semantic_review_required':True}


class SystemEvaluator:
    """
    Executor da Suíte de Avaliação de Qualidade (Evals) do Sistema Multi-Agente.
    Calcula:
    - Routing Accuracy
    - Tool Selection Recall
    - Mean Groundedness Score
    - Abstention & Guardrail Precision
    - Latências p50 e p95
    """

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.dataset_path = Path(dataset_path or settings.EVAL_DATASET_PATH)

    @staticmethod
    def _route_matches(expected: str, actual: str) -> bool:
        def tokens(value):
            return [part.strip() for part in re.sub(r'^CHAIN:\s*','',value).split(' -> ')]
        expected_parts,actual_parts=tokens(expected),tokens(actual)
        if expected.startswith('CHAIN:'):
            return expected_parts==actual_parts
        return len(expected_parts)==1 and expected_parts[0] in actual_parts

    @classmethod
    def run_retrieval_benchmark(cls, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        from app.rag.retriever import HybridKnowledgeRetriever
        retriever_instance = HybridKnowledgeRetriever()
        hits_at_1: List[float] = []
        hits_at_3: List[float] = []
        hits_at_5: List[float] = []
        reciprocal_ranks: List[float] = []

        rag_cases = [
            c for c in cases
            if ("search_getnet_knowledge" in c.get("expected_tools", []) or str(c.get("category", "")).startswith("rag_"))
            and not c.get("expect_abstained", False)
            and (c.get("expected_url_substring") or c.get("expected_chunk_ids"))
        ]
        if not rag_cases:
            return {
                "hit_at_1": None,
                "hit_at_3": None,
                "hit_at_5": None,
                "mrr_at_5": None,
                "cases_evaluated": 0,
                # Aliases legados descontinuados mantidos apenas para compatibilidade de schema com metadados
                "recall_at_1": None,
                "recall_at_3": None,
                "recall_at_5": None,
                "mrr": None,
                "deprecations": {
                    "recall_at_1": {"status": "deprecated", "replacement": "hit_at_1", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                    "recall_at_3": {"status": "deprecated", "replacement": "hit_at_3", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                    "recall_at_5": {"status": "deprecated", "replacement": "hit_at_5", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                    "mrr": {"status": "deprecated", "replacement": "mrr_at_5", "note": "MRR truncado em top-5."},
                },
            }

        for c in rag_cases:
            query = c["message"]
            expected_url = c.get("expected_url_substring", "").lower()
            expected_cids = set(c.get("expected_chunk_ids", []))

            outcome = retriever_instance.retrieve_hybrid(query, top_k=5)
            chunks = outcome.get("results", [])

            rank_found = 0
            for i, chunk in enumerate(chunks, start=1):
                chunk_url = (chunk.get("source_url", "") if isinstance(chunk, dict) else getattr(chunk, "source_url", "")).lower()
                chunk_cid = chunk.get("chunk_id", "") if isinstance(chunk, dict) else getattr(chunk, "chunk_id", "")
                url_match = bool(expected_url and expected_url in chunk_url)
                cid_match = bool(expected_cids and chunk_cid in expected_cids)
                if url_match or cid_match:
                    rank_found = i
                    break

            hits_at_1.append(1.0 if rank_found == 1 else 0.0)
            hits_at_3.append(1.0 if (rank_found and rank_found <= 3) else 0.0)
            hits_at_5.append(1.0 if (rank_found and rank_found <= 5) else 0.0)
            reciprocal_ranks.append(1.0 / rank_found if rank_found > 0 else 0.0)

        n_rag = len(rag_cases)
        h1 = round(sum(hits_at_1) / n_rag, 4)
        h3 = round(sum(hits_at_3) / n_rag, 4)
        h5 = round(sum(hits_at_5) / n_rag, 4)
        mrr = round(sum(reciprocal_ranks) / n_rag, 4)
        return {
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_5": h5,
            "mrr_at_5": mrr,
            "cases_evaluated": n_rag,
            # Aliases legados mantidos para clientes existentes com metadados de depreciação
            "recall_at_1": h1,
            "recall_at_3": h3,
            "recall_at_5": h5,
            "mrr": mrr,
            "deprecations": {
                "recall_at_1": {"status": "deprecated", "replacement": "hit_at_1", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                "recall_at_3": {"status": "deprecated", "replacement": "hit_at_3", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                "recall_at_5": {"status": "deprecated", "replacement": "hit_at_5", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                "mrr": {"status": "deprecated", "replacement": "mrr_at_5", "note": "MRR truncado em top-5."},
            },
        }

    def run_suite(self) -> Dict[str, Any]:
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        cases: List[Dict[str, Any]] = raw.get("cases", [])
        run_id=uuid.uuid4().hex

        case_results: List[Dict[str, Any]] = []
        routing_hits = 0
        tool_recall_scores: List[float] = []
        tool_precision_scores: List[float] = []
        groundedness_scores: List[float] = []
        latencies_ms: List[float] = []

        for idx, case in enumerate(cases, start=1):
            req = ChatRequest(
                message=case["message"],
                user_id=case["user_id"],
                session_id=f"eval_{run_id}_{idx}",
            )
            resp = orchestrator.process(req)
            latencies_ms.append(resp.execution_time_ms)

            # 1. Routing check
            expected_agent = case["expected_agent"]
            route_ok = self._route_matches(expected_agent,resp.agent_used)
            if route_ok:
                routing_hits += 1

            # 2. Tool Recall check
            exp_tools: List[str] = case.get("expected_tools", [])
            if exp_tools:
                called_set = set(resp.tools_called)
                t_recall = sum(1 for t in exp_tools if t in called_set) / len(exp_tools)
            else:
                t_recall = 1.0
            called_set=set(resp.tools_called)
            allowed_set=set(case.get('allowed_tools',exp_tools))
            extras=sorted(called_set-allowed_set)
            t_precision=len(called_set & allowed_set)/len(called_set) if called_set else 1.0
            forbidden=set(case.get('forbidden_tools',[]))
            # Any unexpected mutation or escalation is a hard failure, even
            # when recall is perfect. Read-only extras are still imprecision.
            dangerous_extras=[name for name in extras if name in {'execute_terminal_remote_action','create_escalation_ticket'} or name in forbidden]
            tools_safe=not dangerous_extras and not (called_set & forbidden)
            tool_recall_scores.append(t_recall)
            tool_precision_scores.append(t_precision)

            # 3. Groundedness Judge
            judge_res = GroundednessJudge.score_response(case, resp)
            groundedness_scores.append(judge_res["groundedness_score"])

            case_results.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "expected_agent": expected_agent,
                    "actual_agent": resp.agent_used,
                    "routing_correct": route_ok,
                    "expected_tools": exp_tools,
                    "actual_tools": resp.tools_called,
                    "tool_recall": round(t_recall, 4),
                    "tool_precision": round(t_precision,4),
                    "unexpected_tools": extras,
                    "dangerous_extra_tools": dangerous_extras,
                    "tool_selection_safe": tools_safe,
                    "abstained": resp.abstained,
                    "semantic_mode": resp.semantic_mode,
                    "answer_status": resp.answer_status,
                    "groundedness": judge_res,
                    "latency_ms": resp.execution_time_ms,
                }
            )

        n = max(1, len(cases))
        sorted_lat = sorted(latencies_ms)
        p50 = sorted_lat[int(0.50 * (len(sorted_lat) - 1))] if sorted_lat else 0.0
        p95 = sorted_lat[int(0.95 * (len(sorted_lat) - 1))] if sorted_lat else 0.0
        modes = {}
        for mode in sorted({c['semantic_mode'] or 'unclassified' for c in case_results}):
            group = [c for c in case_results if (c['semantic_mode'] or 'unclassified') == mode]
            count = len(group)
            modes[mode] = {
                'case_count': count,
                'non_abstention_rate': round(sum(not c['abstained'] for c in group) / count, 4),
                'declared_complete_rate': round(sum(c['answer_status'] == 'complete' for c in group) / count, 4),
                'limited_consistency_pass_rate': round(sum(c['groundedness']['passed'] for c in group) / count, 4),
            }

        retrieval_benchmark = self.run_retrieval_benchmark(cases)

        summary = {
            "evaluation_run_id": run_id,
            "dataset_version": raw.get("dataset_version", "2.0.0"),
            "execution_mode": "gemini" if settings.GEMINI_API_KEY else ("offline_deterministic" if external_services_mode() == "synthetic_external_api_fixtures" else "offline_llm_live_external_services"),
            "external_services_mode": external_services_mode(),
            "judge_type": "limited_evidence_numeric_polarity_and_extract_consistency",
            "judge_limitations": "Não verifica entailment universal; revisão independente por alegação continua necessária.",
            "total_cases": len(cases),
            "metrics_by_mode": modes,
            "mode_metrics_limitations": "Metadados e checks limitados não medem utilidade, falsa suficiência ou entailment; exigem adjudicação independente por necessidade.",
            "retrieval_benchmark": retrieval_benchmark,
            "metrics": {
                "routing_accuracy": round(routing_hits / n, 4),
                "mean_tool_recall": round(sum(tool_recall_scores) / n, 4),
                "mean_tool_precision": round(sum(tool_precision_scores)/n,4),
                "safe_tool_selection_rate": round(sum(c["tool_selection_safe"] for c in case_results)/n,4),
                "mean_groundedness_score": round(sum(groundedness_scores) / n, 4),
                "hit_at_3": retrieval_benchmark["hit_at_3"],
                "hit_at_5": retrieval_benchmark["hit_at_5"],
                "mrr_at_5": retrieval_benchmark["mrr_at_5"],
                "retrieval_cases_evaluated": retrieval_benchmark["cases_evaluated"],
                # Aliases legados descontinuados
                "retrieval_recall_at_1": retrieval_benchmark["recall_at_1"],
                "retrieval_recall_at_3": retrieval_benchmark["recall_at_3"],
                "retrieval_recall_at_5": retrieval_benchmark["recall_at_5"],
                "retrieval_mrr": retrieval_benchmark["mrr"],
                "deprecations": {
                    "retrieval_recall_at_1": {"status": "deprecated", "replacement": "hit_at_1", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                    "retrieval_recall_at_3": {"status": "deprecated", "replacement": "hit_at_3", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                    "retrieval_recall_at_5": {"status": "deprecated", "replacement": "hit_at_5", "note": "Hit@K mede presença de chunk esperado no top-K, não recall multi-documento."},
                    "retrieval_mrr": {"status": "deprecated", "replacement": "mrr_at_5", "note": "MRR truncado em top-5."},
                },
                "overall_pass_rate": round(
                    sum(1 for c in case_results if c["routing_correct"] and c["tool_recall"] == 1.0 and c["tool_precision"] == 1.0 and c["tool_selection_safe"] and c["groundedness"]["passed"]) / n,
                    4,
                ),
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
            },
            "cases": case_results,
        }
        return summary


evaluator = SystemEvaluator()


def validate_evaluation_report(report: Dict[str, Any]) -> List[str]:
    """
    Função pura de aceitação de gates do CI/harness de avaliação.
    Retorna lista de erros encontrados. Lista vazia indica aprovação total.
    """
    failures = []
    total_cases = report.get("total_cases", 0)
    if isinstance(total_cases, bool) or not isinstance(total_cases, int) or total_cases <= 0:
        failures.append("total_cases must be a positive integer")

    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        return ["metrics dictionary missing from report"]

    # F4: Gate estrito de retrieval (exige contagem positiva e pisos de regressão >= 0.90)
    retrieval_cases = metrics.get("retrieval_cases_evaluated")
    if isinstance(retrieval_cases, bool) or not isinstance(retrieval_cases, int) or retrieval_cases <= 0:
        failures.append(f"retrieval_cases_evaluated: expected positive integer, got {retrieval_cases}")
    else:
        for metric, minimum in (
            ("hit_at_3", 0.90),
            ("hit_at_5", 0.90),
            ("mrr_at_5", 0.90),
        ):
            actual = metrics.get(metric)
            if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual) or not (minimum <= actual <= 1.0):
                failures.append(f"{metric}: expected {minimum}..1.0, got {actual}")

    for metric, minimum in (
        ("routing_accuracy", 1.0),
        ("mean_tool_recall", 1.0),
        ("mean_groundedness_score", 0.90),
    ):
        actual = metrics.get(metric)
        if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual) or not (minimum <= actual <= 1.0):
            failures.append(f"{metric}: expected {minimum}..1.0, got {actual}")

    for metric in ("mean_tool_precision", "safe_tool_selection_rate"):
        if metric in metrics:
            actual = metrics[metric]
            if isinstance(actual, bool) or not isinstance(actual, (float, int)) or actual != 1.0:
                failures.append(f"{metric}: expected 1.0, got {actual}")

    if "overall_pass_rate" in metrics:
        actual = metrics["overall_pass_rate"]
        if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual) or not (0.90 <= actual <= 1.0):
            failures.append(f"overall_pass_rate: expected 0.90..1.0, got {actual}")

    return failures


def main() -> int:
    """Emit metrics and fail CI when any SPEC acceptance threshold regresses."""
    import sys

    mode = 'live_external_services' if settings.GEMINI_API_KEY else 'synthetic_external_api_fixtures'
    with (nullcontext() if settings.GEMINI_API_KEY else synthetic_external_transport()):
        report = evaluator.run_suite()
    cli_metrics = {
        'total_cases': report.get('total_cases', 0),
        'retrieval_cases_evaluated': report.get('metrics', {}).get('retrieval_cases_evaluated'),
        **report.get('metrics', {}),
        'external_services_mode': mode,
        'execution_mode': report.get('execution_mode'),
        'metrics_by_mode': report.get('metrics_by_mode', {}),
        'mode_metrics_limitations': report.get('mode_metrics_limitations', ''),
    }
    sys.stdout.write(json.dumps(cli_metrics, indent=2, ensure_ascii=False) + "\n")
    failures = validate_evaluation_report(report)
    if failures:
        sys.stderr.write("Evaluation acceptance failed: " + "; ".join(failures) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
