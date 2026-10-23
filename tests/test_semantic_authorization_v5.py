"""Semantic protocol contracts over fresh synthetic source text, never heldout cases."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from google.genai import types


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def candidate():
    body = '## Serviço Aurora\nO Serviço Aurora permite cobrança remota em euros, somente para empresas residentes na região documentada. A liquidação não é imediata.'
    return {'chunk_id':'synthetic-aurora','content':body,'content_hash':digest(body),'title':'Serviço Aurora','source':'synthetic.md','source_url':'https://example.test/source','provenance':{'review_method':'synthetic_fixture'}}


def selection():
    chunk=candidate();start=chunk['content'].index('\n')+1
    return {'span_id':'synthetic-aurora:p0','chunk_id':chunk['chunk_id'],'content_hash':chunk['content_hash'],'start':start,'end':len(chunk['content']),'quote':chunk['content'][start:]}


def analysis(message, *, units=None):
    # V6 transports decisions by IDs; literal bodies/offsets remain canonical.
    canonical = units or [unit('u1',message,0,len(message),selections=[selection()])]
    decisions = {item['unit_id']: {key: value for key, value in item.items()
                 if key not in {'unit_id', 'start', 'end', 'text', 'selections'}}
                 for item in canonical}
    for item in canonical:
        decisions[item['unit_id']]['span_ids'] = [s['span_id'] for s in item['selections']]
    return {'request_hash':digest(message),'units':decisions}


def unit(uid,message,start,end,*,kind='requirement',decision='supported',selections=None):
    return {'unit_id':uid,'start':start,'end':end,'text':message[start:end],'kind':kind,
            'subject':'aurora' if kind=='requirement' else '', 'relation':'remote_collection' if kind=='requirement' else '',
            'value':'','unit':'','qualifiers':[],'decision':decision,'selections':selections or []}


def verification(message, ids=('u1',), *, rejected=(), complete=True, issue='unsupported_relation'):
    return {'request_hash':digest(message),'checks':{uid:{'verdict':'reject' if uid in rejected else 'accept','issues':[issue] if uid in rejected else []} for uid in ids},'complete':complete}


def provider(replies, requests):
    iterator=iter(replies)
    def generate_content(**kwargs):
        requests.append(kwargs)
        item=next(iterator)
        if isinstance(item,Exception): raise item
        payload=item if isinstance(item,str) else json.dumps(item,ensure_ascii=False)
        return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP,content=types.Content(role='model',parts=[types.Part.from_text(text=payload)]))])
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def authorize(message, first=None, second=None, requests=None):
    from app.rag.semantic import SemanticEvidenceAuthorizer
    output=requests if requests is not None else []
    return SemanticEvidenceAuthorizer(provider([first or analysis(message),second or verification(message)],output)).authorize(message,[candidate()])


def test_raw_candidates_bypass_the_local_answerability_gate(tmp_path,monkeypatch):
    from app.rag.retriever import HybridKnowledgeRetriever
    import app.rag.retriever as module
    (tmp_path/'new.md').write_text(candidate()['content'],encoding='utf-8')
    retriever=HybridKnowledgeRetriever(data_dir=tmp_path,corpus_path=tmp_path/'c.json',vector_index_path=tmp_path/'v.json')
    monkeypatch.setattr(module,'assess',lambda *args,**kwargs:pytest.fail('Legacy semantic gate ran during candidate retrieval.'))
    outcome=retriever.retrieve_candidates('Uma necessidade comercial formulada sem os aliases locais')
    assert outcome['candidate_chunks']
    assert 'answerability' not in outcome['candidate_chunks'][0]
    assert outcome['candidate_chunks'][0]['content_hash']==digest(outcome['candidate_chunks'][0]['content'])


def test_native_two_passes_publish_only_enumerated_whole_paragraphs():
    message='Consigo cobrar remotamente usando o Serviço Aurora?'
    requests=[]
    out=authorize(message,requests=requests)
    assert out.status=='complete'
    assert out.approved_chunks[0]['approved_spans'][0]['quote']==selection()['quote']
    assert len(requests)==2
    assert all(request['config'].response_mime_type=='application/json' for request in requests)


@pytest.mark.parametrize('field,value',[
    ('span_id','invented'),('chunk_id','invented'),('content_hash','bad'),
    ('quote','A liquidação é imediata.'),('start',0),('end',22),
])
def test_model_cannot_override_server_owned_selection_fields(field,value):
    message='O Serviço Aurora é adequado?'
    first=analysis(message);first['units']['u1'][field]=value
    out=authorize(message,first=first)
    assert out.status=='abstained' and not out.approved_chunks


def test_unit_gaps_cannot_hide_a_second_clause():
    message='Explique o Serviço Aurora; também preciso cancelar a operação.'
    end=message.index(';')
    first=analysis(message,units=[unit('u1',message,0,end,selections=[selection()])])
    out=authorize(message,first=first)
    assert out.status=='abstained'


def test_rejected_condition_classified_as_context_prevents_complete():
    message='Explique o Serviço Aurora; para uma empresa não residente.'
    split=message.index(';')
    units=[unit('u1',message,0,split,selections=[selection()]),unit('u2',message,split,len(message),kind='context',decision='not_applicable')]
    out=authorize(message,first=analysis(message,units=units),second=verification(message,('u1','u2'),rejected=('u2',),complete=False,issue='omitted_condition'))
    assert out.status=='partial'
    assert out.unresolved_units==[message[split:]]
    assert len(out.approved_chunks)==1


@pytest.mark.parametrize('issue',['subject_mismatch','unit_mismatch','scope_mismatch','polarity_mismatch','omitted_exception'])
def test_verifier_rejection_cannot_be_reported_complete(issue):
    message='O Serviço Aurora liquida imediatamente em dólares?'
    out=authorize(message,second=verification(message,rejected=('u1',),complete=False,issue=issue))
    assert out.status=='abstained' and not out.approved_chunks


@pytest.mark.parametrize('second',[RuntimeError('provider unavailable'),'{"request_hash":',{'request_hash':'wrong','checks':[],'complete':True}])
def test_verification_failure_never_falls_back_to_local_approval(second):
    message='Explique o Serviço Aurora.'
    out=authorize(message,second=second)
    assert out.status=='abstained' and not out.approved_chunks


def test_extra_free_prose_is_rejected_by_the_strict_schema():
    message='Explique o Serviço Aurora.'
    first=analysis(message);first['answer']='Taxa inventada de 7%.'
    assert authorize(message,first=first).status=='abstained'


def test_missing_verification_unit_is_protocol_failure():
    message='Explique Aurora; cancele a cobrança.'
    split=message.index(';')
    units=[unit('u1',message,0,split,selections=[selection()]),unit('u2',message,split,len(message),decision='unknown')]
    out=authorize(message,first=analysis(message,units=units),second=verification(message,('u1',)))
    assert out.status=='abstained' and not out.approved_chunks


def tool_result():
    from app.schemas import ToolExecutionResult
    return ToolExecutionResult(tool_name='search_getnet_knowledge',success=True,data={
        'abstained':True,'abstention_reason':'Legacy parser does not know Aurora',
        'chunks':[],'citations':[],'candidate_chunks':[candidate()],'max_relevance_score':0.1})


def integrated_agent(monkeypatch,replies):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.config import settings
    from app.schemas import ToolCallSpec
    from app.tools.registry import tool_registry
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    requests=[]
    agent=KnowledgeAgent(provider(replies,requests))
    monkeypatch.setattr(agent,'_plan_tool_calls',lambda **kwargs:[ToolCallSpec(tool_name='search_getnet_knowledge',arguments={'query':'rewritten'})])
    monkeypatch.setattr(tool_registry,'execute',lambda **kwargs:tool_result())
    return agent,requests


def test_knowledge_native_uses_candidates_despite_legacy_abstention(monkeypatch):
    from app.security.auth import SecurityContext
    message='Consigo cobrar remotamente usando o Serviço Aurora?'
    agent,requests=integrated_agent(monkeypatch,[analysis(message),verification(message)])
    out=agent.run('rewritten',SecurityContext(authenticated_user_id='synthetic'),request_message=message)
    assert not out.abstained and out.semantic_mode=='native_semantic' and out.answer_status=='complete'
    assert selection()['quote'] in out.response
    assert len(requests)==2  # no redundant single-chunk generation
    assert out.evidence and 'não é imediata' in out.evidence[0]['excerpt']


def test_knowledge_partial_reports_the_literal_unanswered_unit(monkeypatch):
    from app.security.auth import SecurityContext
    message='Explique Aurora; cancele meu contrato.'
    split=message.index(';')
    units=[unit('u1',message,0,split,selections=[selection()]),unit('u2',message,split,len(message),decision='unknown')]
    agent,_=integrated_agent(monkeypatch,[analysis(message,units=units),verification(message,('u1','u2'),complete=False)])
    out=agent.run(message,SecurityContext(authenticated_user_id='synthetic'))
    assert out.abstained and out.answer_status=='partial'
    assert message[split:] in out.response
    assert selection()['quote'] in out.response


def test_knowledge_provider_failure_does_not_publish_legacy_evidence(monkeypatch):
    from app.security.auth import SecurityContext
    message='Explique Aurora.'
    agent,_=integrated_agent(monkeypatch,[RuntimeError('503')])
    out=agent.run(message,SecurityContext(authenticated_user_id='synthetic'))
    assert out.abstained and out.answer_status=='abstained' and out.semantic_mode=='native_semantic'
    assert selection()['quote'] not in out.response and not out.evidence


def test_native_document_tool_does_not_run_the_local_authorization_gate(monkeypatch):
    from app.config import settings
    from app.tools import search_tools
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    monkeypatch.setattr(search_tools.retriever,'retrieve_hybrid',lambda **kwargs:pytest.fail('Native tool invoked local authorization.'))
    monkeypatch.setattr(search_tools.retriever,'retrieve_candidates',lambda **kwargs:{'candidate_chunks':[candidate()],'candidate_count':1})
    result=search_tools.search_getnet_knowledge('Request with unfamiliar clauses')
    assert result['candidate_chunks']==[candidate()]
    assert not result['chunks']


def test_offline_unknown_second_clause_cannot_receive_complete_status(monkeypatch):
    from app.config import settings
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings,'GEMINI_API_KEY','')
    out=KnowledgeAgent().run('Como funciona o Pix? Também preciso de uma condição inédita.',SecurityContext(authenticated_user_id='synthetic'))
    assert out.abstained and out.answer_status!='complete' and out.semantic_mode=='limited_offline'


def test_request_units_and_sources_are_enumerated_for_copying_not_counting():
    message='Explique Aurora; preciso de uma exceção contratual.'
    split=message.index(';')
    units=[unit('u1',message,0,split,selections=[selection()]),unit('u2',message,split,len(message),decision='unknown')]
    requests=[]
    authorize(message,first=analysis(message,units=units),second=verification(message,('u1','u2'),complete=False),requests=requests)
    payload=json.loads(requests[0]['contents'][0].parts[0].text)
    assert payload['original_request']==message
    assert ''.join(item['text'] for item in payload['request_units'])==message
    assert payload['candidates'][0]['spans'][0]==selection()


def test_unknown_requirement_cannot_be_promoted_by_verifier_complete_true():
    message='Explique Aurora; cancele a operação.'
    split=message.index(';')
    units=[unit('u1',message,0,split,selections=[selection()]),unit('u2',message,split,len(message),decision='unknown')]
    out=authorize(message,first=analysis(message,units=units),second=verification(message,('u1','u2'),complete=True))
    assert out.status=='partial' and message[split:] in out.unresolved_units


def test_subparagraph_quote_cannot_remove_a_contractual_condition():
    message='Explique Aurora.'
    first=analysis(message)
    # Compact transport may select a whole span ID, never supply edited text.
    quote=selection()
    first['units']['u1']['quote']=quote['quote'].split(',')[0]
    first['units']['u1']['end']=quote['start']+len(first['units']['u1']['quote'])
    assert authorize(message,first=first).status=='abstained'


def test_duplicate_documental_calls_still_use_at_most_two_semantic_generations(monkeypatch):
    from app.security.auth import SecurityContext
    from app.schemas import ToolCallSpec
    message='Explique Aurora.'
    agent,requests=integrated_agent(monkeypatch,[analysis(message),verification(message)])
    monkeypatch.setattr(agent,'_plan_tool_calls',lambda **kwargs:[
        ToolCallSpec(tool_name='search_getnet_knowledge',arguments={'query':'first'}),
        ToolCallSpec(tool_name='search_getnet_knowledge',arguments={'query':'second'})])
    out=agent.run(message,SecurityContext(authenticated_user_id='synthetic'))
    assert len(requests)==2
    assert out.answer_status=='complete'


def test_offline_unsupported_condition_cannot_hide_in_a_single_overview(monkeypatch):
    from app.config import settings
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings,'GEMINI_API_KEY','')
    message='Como funciona o Pix para uma empresa com autorização temporária?'
    out=KnowledgeAgent().run(message,SecurityContext(authenticated_user_id='synthetic'))
    assert out.abstained and out.answer_status!='complete'


@pytest.mark.parametrize('message',[
    'What is the difference between Get Smart and Get Clássica?',
    'Como funciona Receba Já com a Getnet?',
    'Do I need a bank account to receive payments via Pix?',
    'Em quantas parcelas podemos dividir uma venda no Crediário?',
    'Posso vender via WhatsApp usando Link de Pagamento?',
])
def test_offline_compositional_domain_roles_preserve_known_capabilities(message):
    from app.rag.answerability import offline_request_coverage
    assert offline_request_coverage(message)['complete_parse']


@pytest.mark.parametrize('suffix',[
    ' para uma empresa com autorização temporária',
    ' e também quero rescindir um contrato',
    ' desde que o comprador resida em outra jurisdição',
])
def test_offline_domain_grammar_never_discards_an_unconsumed_suffix(suffix):
    from app.rag.answerability import offline_request_coverage
    result=offline_request_coverage('Como funciona Receba Já com a Getnet'+suffix+'?')
    assert not result['complete_parse'] and result['unresolved_units']


def test_server_enumerated_unit_ids_cannot_be_merged_or_invented():
    message='Explique Aurora; cancele a operação.'
    # Even a valid exact whole-request quote may not erase enumerated clauses.
    out=authorize(message,first=analysis(message),second=verification(message))
    assert out.status=='abstained' and not out.approved_chunks


def test_server_enumerates_a_conjoined_substantive_clause():
    from app.rag.semantic import request_units
    message='Explique Aurora e cancele uma operação.'
    parts=request_units(message)
    assert len(parts)>=2
    assert ''.join(part['text'] for part in parts)==message
    assert authorize(message).status=='abstained'


def test_semantic_provider_boundary_masks_original_and_reference_context():
    from app.rag.semantic import SemanticEvidenceAuthorizer
    from app.security.pii import PIIRedactor
    original='Explique Aurora para CPF 123.456.789-01.'
    message=PIIRedactor.mask_text(original)
    requests=[]
    out=SemanticEvidenceAuthorizer(provider([analysis(message),verification(message)],requests)).authorize(
        original,[candidate()],{'summary':'senha: privatePass123'})
    encoded=' '.join(request['contents'][0].parts[0].text for request in requests)
    assert '123.456.789-01' not in encoded and 'privatePass123' not in encoded
    assert out.status=='complete'
