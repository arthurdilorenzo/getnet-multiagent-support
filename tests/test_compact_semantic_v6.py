"""Fresh synthetic compact-protocol tests; no reserved evaluation data."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from app.rag.semantic import SemanticEvidenceAuthorizer, request_units, valid_selection


def digest(text): return hashlib.sha256(text.encode()).hexdigest()


def source():
    body='## Serviço Lumen\nO Serviço Lumen aceita cobrança em euros somente para residentes elegíveis. A liquidação não é imediata.'
    return {'chunk_id':'synthetic-lumen','content':body,'content_hash':digest(body),'source':'synthetic.md','source_url':'https://example.test/lumen','title':'Lumen'}


def decision(kind='requirement',status='supported',spans=None):
    substantive=kind=='requirement'
    return {'kind':kind,'decision':status,'subject':'lumen' if substantive else '',
            'relation':'collection' if substantive else '', 'value':'','unit':'','qualifiers':[],
            'span_ids':(['synthetic-lumen:p0'] if substantive and status in {'supported','contradicted'} else []) if spans is None else spans}


def analysis(message):
    return {'request_hash':digest(message),'units':{u['unit_id']:decision() for u in request_units(message)}}


def verification(message):
    return {'request_hash':digest(message),'checks':{u['unit_id']:{'verdict':'accept','issues':[]} for u in request_units(message)},'complete':True}


def provider(replies,requests):
    pending=iter(replies)
    def generate_content(**kwargs):
        requests.append(kwargs);item=next(pending)
        if isinstance(item,Exception):raise item
        text=item if isinstance(item,str) else json.dumps(item,ensure_ascii=False)
        return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP,content=types.Content(role='model',parts=[types.Part.from_text(text=text)]))])
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def run(message='Explique Lumen.',first=None,second=None,scope=None,tracer=None,chunks=None):
    requests=[]
    out=SemanticEvidenceAuthorizer(provider([first if first is not None else analysis(message),second if second is not None else verification(message)],requests)).authorize(
        message,chunks if chunks is not None else [source()],tracer=tracer,delegation_scope=scope)
    return out,requests


def test_compact_protocol_reconstructs_exact_canonical_evidence_and_dynamic_schema():
    out,requests=run()
    assert out.status=='complete'
    span=out.approved_chunks[0]['approved_spans'][0]
    assert span['quote']==source()['content'].split('\n')[1]
    assert valid_selection(span,source())
    assert len(requests)==2
    schema=requests[0]['config'].response_json_schema
    assert schema['properties']['request_hash']['enum']==[digest('Explique Lumen.')]
    units=schema['properties']['units']
    assert units['required']==['u1'] and units['additionalProperties'] is False
    item=units['properties']['u1']
    assert set(item)=={'anyOf'}
    assert len(item['anyOf'])==4
    for branch in item['anyOf']:
        assert branch['additionalProperties'] is False
        assert set(branch['required'])==set(branch['properties'])
        assert not ({'text','start','end','quote','content_hash'} & set(branch['properties']))
    supported=item['anyOf'][0]
    assert supported['properties']['span_ids']['items']['enum']==['synthetic-lumen:p0']
    assert 'maxItems' not in supported['properties']['span_ids']
    assert out.covered_unit_ids==['u1']


def test_object_order_does_not_change_unit_identity():
    message='Explique Lumen; descreva as condições.'
    first=analysis(message);first['units']=dict(reversed(list(first['units'].items())))
    second=verification(message);second['checks']=dict(reversed(list(second['checks'].items())))
    out,_=run(message,first,second)
    assert out.status=='complete' and out.covered_unit_ids==['u1','u2']


@pytest.mark.parametrize('mutation,code',[
    ('hash','REQUEST_IDENTITY'),('missing_unit','UNIT_KEYSET'),('extra_unit','UNIT_KEYSET'),
    ('unknown_span','SPAN_UNKNOWN'),('duplicate_span','SPAN_DUPLICATE'),('no_span','UNIT_STATE'),
])
def test_local_validation_rejects_schema_or_identity_lies(mutation,code):
    first=analysis('Explique Lumen.')
    if mutation=='hash':first['request_hash']='wrong'
    elif mutation=='missing_unit':first['units']={}
    elif mutation=='extra_unit':first['units']['outside']=decision()
    elif mutation=='unknown_span':first['units']['u1']['span_ids']=['foreign:p0']
    elif mutation=='duplicate_span':first['units']['u1']['span_ids']*=2
    elif mutation=='no_span':first['units']['u1']['span_ids']=[]
    out,requests=run(first=first)
    assert out.status=='abstained' and not out.approved_chunks
    assert out.error_code==code and out.error_stage=='semantic_analysis'
    assert len(requests)==1


def test_duplicate_json_keys_are_rejected_before_schema_validation():
    first=json.dumps(analysis('Explique Lumen.'))
    first=first.replace('"units": {','"units": {"u1": '+json.dumps(decision())+',',1)
    out,_=run(first=first)
    assert out.error_code=='JSON_DUPLICATE_KEY'


def test_safe_diagnostics_never_store_rejected_input_or_exception_text():
    from app.observability.tracer import SystemTracer
    secret='private-field-123.456.789-01'
    first=analysis('Explique Lumen.');first['units']['u1']['quote']=secret
    tracer=SystemTracer()
    out,_=run(first=first,tracer=tracer)
    assert out.error_code=='SCHEMA_VALIDATION'
    assert secret not in repr(out) and secret not in repr(tracer.get_steps())
    assert out.diagnostics['error_types']==['extra_forbidden']


def test_changed_source_identity_has_a_distinct_safe_code():
    chunk=source();chunk['content_hash']='incorrect'
    out,requests=run(chunks=[chunk])
    assert out.error_code=='SOURCE_IDENTITY' and not requests


def test_verification_keys_and_verdict_conflict_have_distinct_codes():
    second=verification('Explique Lumen.');second['checks']={}
    assert run(second=second)[0].error_code=='VERIFICATION_KEYSET'
    second=verification('Explique Lumen.');second['checks']['u1']['issues']=['scope_mismatch']
    assert run(second=second)[0].error_code=='VERIFICATION_STATE'


@pytest.mark.parametrize('issue',['omitted_condition','subject_mismatch','unit_mismatch','polarity_mismatch','omitted_exception'])
def test_semantic_verifier_veto_is_preserved_without_free_prose(issue):
    second=verification('Explique Lumen.')
    second.update(complete=False);second['checks']['u1']={'verdict':'reject','issues':[issue]}
    out,_=run(second=second)
    assert out.status=='abstained' and out.unresolved_unit_ids==['u1']


def test_delegated_unit_has_no_factual_credit_and_does_not_hide_a_condition():
    message='Explique Lumen; consulte o terminal.'
    first=analysis(message);first['units']['u2']=decision('delegated','not_applicable')
    scope={'assigned_unit_ids':['u1'],'other_unit_ids':['u2']}
    out,requests=run(message,first=first,scope=scope)
    assert out.status=='complete'  # local completion only
    assert out.request_unit_ids==['u1'] and out.covered_unit_ids==['u1'] and out.delegated_unit_ids==['u2']
    assert not out.coverage[1]['answered']
    payload=json.loads(requests[1]['contents'][0].parts[0].text)
    assert payload['original_request']==message and payload['delegation_scope']==scope
    second=verification(message);second.update(complete=False);second['checks']['u2']={'verdict':'reject','issues':['omitted_condition']}
    rejected,_=run(message,first=first,second=second,scope=scope)
    assert rejected.status=='abstained' and not rejected.approved_chunks and 'u2' in rejected.unresolved_unit_ids
    assert not rejected.covered_unit_ids  # discarded condition cannot release local facts


@pytest.mark.parametrize('scope',[None,{'assigned_unit_ids':['u1'],'other_unit_ids':['u1']}])
def test_assigned_or_shared_unit_cannot_be_delegated(scope):
    first=analysis('Explique Lumen.');first['units']['u1']=decision('delegated','not_applicable')
    assert run(first=first,scope=scope)[0].error_code=='DELEGATION_STATE'


def test_scope_cannot_introduce_an_unknown_request_id():
    out,requests=run(scope={'assigned_unit_ids':['missing'],'other_unit_ids':['u1']})
    assert out.error_code=='DELEGATION_SCOPE' and not requests


def test_provider_and_json_failures_receive_safe_stage_codes():
    out,_=run(first=RuntimeError('payload with password secret'))
    assert out.error_code=='PROVIDER_ERROR' and out.error_stage=='semantic_analysis'
    assert 'secret' not in repr(out)
    assert run(second='{"request_hash":')[0].error_code=='JSON_PARSE'


def integrated_agent(monkeypatch,replies):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.config import settings
    from app.schemas import ToolCallSpec,ToolExecutionResult
    from app.tools.registry import tool_registry
    monkeypatch.setattr(settings,'GEMINI_API_KEY','fake-never-sent')
    requests=[];agent=KnowledgeAgent(provider(replies,requests))
    monkeypatch.setattr(agent,'_plan_tool_calls',lambda **kwargs:[ToolCallSpec(tool_name='search_getnet_knowledge',arguments={'query':'rewrite'})])
    monkeypatch.setattr(tool_registry,'execute',lambda **kwargs:ToolExecutionResult(tool_name='search_getnet_knowledge',success=True,
        data={'abstained':True,'chunks':[],'citations':[],'candidate_chunks':[source()],'max_relevance_score':0.1}))
    return agent,requests


def test_knowledge_passes_scope_and_reports_only_its_actual_factual_coverage(monkeypatch):
    from app.security.auth import SecurityContext
    message='Explique Lumen; consulte o terminal.'
    first=analysis(message);first['units']['u2']=decision('delegated','not_applicable')
    agent,requests=integrated_agent(monkeypatch,[first,verification(message)])
    out=agent.run('rewrite',SecurityContext(authenticated_user_id='synthetic'),request_message=message,
                  delegation_scope={'assigned_unit_ids':['u1'],'other_unit_ids':['u2']})
    assert out.answer_status=='complete'
    assert out.request_unit_ids==['u1'] and out.covered_unit_ids==['u1'] and not out.unresolved_unit_ids
    assert out.structured_data['answerability']['coverage'][1]['kind']=='delegated'


def test_bad_scope_is_rejected_before_any_tool(monkeypatch):
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.security.auth import SecurityContext
    from app.tools.registry import tool_registry
    monkeypatch.setattr(tool_registry,'execute',lambda **kwargs:pytest.fail('Invalid scope reached a tool.'))
    out=KnowledgeAgent().run('Como funciona o Pix?',SecurityContext(authenticated_user_id='synthetic'),
                            delegation_scope={'assigned_unit_ids':['invented'],'other_unit_ids':[]})
    assert out.abstained and not out.tools_called


def test_offline_coverage_notice_does_not_claim_the_answered_question_is_unanswered(monkeypatch):
    from app.config import settings
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings,'GEMINI_API_KEY','')
    message='Mostre como funciona o Pix?'
    out=KnowledgeAgent().run(message,SecurityContext(authenticated_user_id='synthetic'))
    assert out.answer_status=='partial' and out.abstained
    assert 'Partes do pedido sem resposta verificada' not in out.response
    assert 'Cobertura no modo local' in out.response
    assert message not in out.response


def test_offline_metadata_separates_supported_and_unknown_literal_clauses(monkeypatch):
    from app.config import settings
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.security.auth import SecurityContext
    monkeypatch.setattr(settings,'GEMINI_API_KEY','')
    message='Como funciona o Pix? Preciso de uma regra ainda não documentada.'
    out=KnowledgeAgent().run(message,SecurityContext(authenticated_user_id='synthetic'))
    assert out.request_unit_ids==['u1','u2']
    assert out.covered_unit_ids==['u1'] and out.unresolved_unit_ids==['u2']
    assert 'Preciso de uma regra ainda não documentada' in out.response


@pytest.mark.parametrize('rate,covered',[(6.15,['u1']),(None,[])])
def test_typed_currency_coverage_requires_an_observed_rate(monkeypatch,rate,covered):
    from app.config import settings
    from app.tools.search_tools import _today_local
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.schemas import ToolExecutionResult
    from app.security.auth import SecurityContext
    from app.tools.registry import tool_registry
    monkeypatch.setattr(settings,'GEMINI_API_KEY','')
    monkeypatch.setattr(tool_registry,'execute',lambda tool_name,**kwargs:ToolExecutionResult(tool_name=tool_name,success=True,
        data={'currency_pair':'EUR/BRL','current_rate_brl':rate,'rate_type':'market_quote','live':True,
              'updated_at':_today_local().isoformat()}))
    out=KnowledgeAgent().run('Qual a cotação do euro hoje?',SecurityContext(authenticated_user_id='synthetic'),is_web_search=True)
    assert out.request_unit_ids==['u1'] and out.covered_unit_ids==covered
    assert out.unresolved_unit_ids==([] if covered else ['u1'])


def test_verified_context_exemption_is_reported_without_factual_credit(monkeypatch):
    from app.security.auth import SecurityContext
    message='Explique Lumen; por gentileza.'
    first=analysis(message);first['units']['u2']=decision('courtesy','not_applicable')
    agent,_=integrated_agent(monkeypatch,[first,verification(message)])
    out=agent.run(message,SecurityContext(authenticated_user_id='synthetic'))
    assert out.covered_unit_ids==['u1'] and out.structured_data['request_unit_exemptions']=={'u2':'courtesy'}
    assert 'u2' not in out.unresolved_unit_ids


def test_uncertain_global_verification_does_not_report_overlapping_credit():
    message='Explique Lumen.'
    second=verification(message);second['complete']=False
    out,_=run(message,second=second)
    assert out.status=='abstained' and not out.approved_chunks and not out.covered_unit_ids
    assert out.unresolved_unit_ids==['u1']
