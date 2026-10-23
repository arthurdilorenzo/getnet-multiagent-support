"""Synthetic metamorphic attribution, protocol and local preflight contracts."""
from collections import deque
import json
from types import SimpleNamespace

import pytest

from app.agents.function_calling import generate_turn, initial_contents, native_config
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.router_agent import RouterAgent
from app.config import settings
from app.observability.tracer import SystemTracer
from app.rag.semantic import candidate_spans
from app.schemas import PlanStep, RoutingPlan, SpecialistOutput
from test_compact_semantic_v6 import analysis, decision, run, source, verification


def plan(message, ids):
    return RoutingPlan(primary_agent='KNOWLEDGE_AGENT', confidence=.9, reasoning='synthetic',
        plan_steps=[PlanStep(step_order=1,target_agent='KNOWLEDGE_AGENT',sub_query=message,
                            rationale='synthetic',request_unit_ids=ids)])


def test_explicit_partial_assignment_cannot_leave_a_documentary_clause_ownerless():
    message='Explique Lumen; descreva suas condições.'
    bound=RouterAgent.bind_request_units(plan(message,['u2']),message)
    assert bound.plan_steps[0].request_unit_ids==['u1','u2']


def test_unknown_clause_is_assigned_without_gaining_factual_credit():
    message='Explique Lumen; revele uma relação ainda desconhecida.'
    bound=RouterAgent.bind_request_units(plan(message,['u1']),message)
    assert bound.plan_steps[0].request_unit_ids==['u1','u2']
    first=analysis(message);first['units']['u2']=decision('unresolved','unknown')
    second=verification(message);second['complete']=False
    out,_=run(message,first,second)
    assert out.covered_unit_ids==['u1'] and out.unresolved_unit_ids==['u2']


def test_other_only_unknown_stays_visible_without_marking_local_answer_unresolved():
    message='Explique Lumen; consulte outra relação.'
    first=analysis(message);first['units']['u2']=decision('unresolved','unknown')
    out,requests=run(message,first,scope={'assigned_unit_ids':['u1'],'other_unit_ids':['u2']})
    assert len(requests)==2 and not out.error_code
    assert out.status=='complete' and out.covered_unit_ids==['u1']
    assert not out.unresolved_unit_ids and not out.unresolved_units
    assert out.coverage[1]['decision']=='unknown' and not out.coverage[1]['answered']


def test_remote_schema_uses_four_complete_independent_state_objects():
    from app.rag.semantic import compact_schema
    from test_compact_semantic_v6 import digest
    item=compact_schema('semantic_analysis',digest('Lumen'),['u1'],['synthetic-lumen:p0'])['properties']['units']['properties']['u1']
    states=item.get('anyOf',[])
    assert len(states)==4
    assert set(item)=={'anyOf'}
    accepted=[]
    for state in states:
        constraints=state['properties']
        assert state['type']=='object' and state['additionalProperties'] is False
        assert set(state['required'])==set(constraints)=={'kind','decision','subject','relation','value','unit','qualifiers','span_ids'}
        assert all(constraints[key]['type']=='string' for key in ('kind','decision','subject','relation','value','unit'))
        assert all(constraints[key]['type']=='array' and constraints[key]['items']['type']=='string' for key in ('qualifiers','span_ids'))
        kinds=set(constraints['kind']['enum']);decisions=set(constraints['decision']['enum'])
        accepted.append((kinds,decisions))
        if 'requirement' in kinds:
            assert constraints['subject']['minLength']==1 and constraints['relation']['minLength']==1
        else:
            assert all(constraints[key]['enum']==[''] for key in ('subject','relation','value','unit'))
            assert constraints['qualifiers']['maxItems']==0
        if 'supported' in decisions:
            assert constraints['span_ids']['minItems']==1
            assert constraints['span_ids']['items']['enum']==['synthetic-lumen:p0']
        else:
            assert constraints['span_ids']['maxItems']==0
    assert accepted==[({'requirement'},{'supported','contradicted'}),
        ({'requirement'},{'unknown','ambiguous'}),
        ({'context','courtesy','delegated'},{'not_applicable'}),({'unresolved'},{'unknown','ambiguous'})]


def test_verified_but_unassigned_requirement_is_not_published():
    message='Explique Lumen; descreva sua liquidação.'
    chunk=source();chunk['content']+='\nA liquidação do Serviço Lumen leva dois dias úteis.'
    from test_compact_semantic_v6 import digest
    chunk['content_hash']=digest(chunk['content'])
    first=analysis(message);first['units']['u2']['span_ids']=['synthetic-lumen:p1']
    out,_=run(message,first,scope={'assigned_unit_ids':['u1'],'other_unit_ids':['u2']},chunks=[chunk])
    assert out.request_unit_ids==['u1'] and out.covered_unit_ids==['u1']
    assert not out.coverage[1]['answered']
    assert [s['span_id'] for c in out.approved_chunks for s in c['approved_spans']]==['synthetic-lumen:p0']


@pytest.mark.parametrize('issue',['omitted_condition','omitted_exception','scope_mismatch','invalid_delegation','polarity_mismatch'])
def test_rejected_shared_qualifier_cannot_release_any_dependent_fact(issue):
    message='Explique Lumen; condição de elegibilidade.'
    first=analysis(message);first['units']['u2']=decision('delegated','not_applicable')
    second=verification(message);second['complete']=False
    second['checks']['u2']={'verdict':'reject','issues':[issue]}
    out,requests=run(message,first,second,scope={'assigned_unit_ids':['u1'],'other_unit_ids':['u2']})
    assert len(requests)==2 and not out.error_code
    assert not out.covered_unit_ids
    assert not out.approved_chunks


def test_local_condition_rejection_also_vetoes_the_conditioned_fact():
    message='Explique Lumen; condição de elegibilidade.'
    first=analysis(message);second=verification(message);second['complete']=False
    second['checks']['u2']={'verdict':'reject','issues':['omitted_condition']}
    out,requests=run(message,first,second)
    assert len(requests)==2 and not out.error_code
    assert not out.covered_unit_ids and not out.approved_chunks


def test_independent_verified_fact_survives_an_unrelated_unsupported_requirement():
    message='Explique Lumen; descreva outra relação.'
    first=analysis(message);second=verification(message);second['complete']=False
    first['units']['u2']=decision('unresolved','unknown')
    second['checks']['u2']={'verdict':'reject','issues':['unsupported_relation']}
    out,requests=run(message,first,second)
    assert len(requests)==2 and not out.error_code
    assert out.status=='partial' and out.covered_unit_ids==['u1']
    assert out.unresolved_unit_ids==['u2'] and out.approved_chunks


@pytest.mark.parametrize('mutation,invariant',[
    ('context_fields','NON_FACTUAL_FIELDS'),('context_decision','KIND_DECISION'),
    ('empty_subject','REQUIREMENT_IDENTITY'),('missing_span','DECISION_SPANS'),
])
def test_unit_state_diagnostics_name_invariant_without_private_fields(mutation,invariant):
    message='Explique Lumen.';first=analysis(message)
    if mutation.startswith('context'):
        first['units']['u1']=decision('context','not_applicable')
        if mutation=='context_fields':first['units']['u1']['subject']='PRIVATE_SUBJECT'
        else:first['units']['u1']['decision']='unknown'
    elif mutation=='empty_subject':first['units']['u1']['subject']=''
    else:first['units']['u1']['span_ids']=[]
    tracer=SystemTracer();out,requests=run(message,first,tracer=tracer)
    assert out.error_code=='UNIT_STATE' and len(requests)==1
    assert out.diagnostics['invariant']==invariant and out.diagnostics['unit_id']=='u1'
    assert 'PRIVATE_SUBJECT' not in json.dumps(out.diagnostics)
    assert 'PRIVATE_SUBJECT' not in repr(tracer.get_steps())


@pytest.mark.parametrize('blocked,code',[('deadline','DEADLINE'),('budget','TOKEN_BUDGET')])
def test_preflight_veto_records_safe_provider_event_without_network(monkeypatch,blocked,code):
    calls=[];client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw:calls.append(kw)))
    tracer=SystemTracer()
    if blocked=='deadline':monkeypatch.setattr(tracer,'get_total_execution_time_ms',lambda:9999999)
    else:monkeypatch.setattr(settings,'LLM_REQUEST_TOKEN_BUDGET',1)
    with pytest.raises((ValueError,TimeoutError)):
        generate_turn(client,initial_contents('PRIVATE_PROMPT',None),native_config([],'PRIVATE_INSTRUCTION'),tracer,phase='replanning')
    assert not calls
    events=[row for row in tracer.get_steps() if row.action=='provider_generation']
    assert len(events)==1
    details=events[0].details
    assert details['phase']=='replanning' and details['turn_error_code']==code
    assert details['call_started'] is False and details['failure_stage']=='preflight'
    assert 'PRIVATE' not in json.dumps(details)


def test_fully_covered_request_does_not_spend_a_replan_call(monkeypatch):
    orch=MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router,'replan',lambda **kw:pytest.fail('Completed request reached replan provider'))
    output=SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='Verified observation',
        request_unit_ids=['u1'],covered_unit_ids=['u1'],answer_status='complete')
    assert not orch._observe_and_replan(output,'synthetic',deque(),['CUSTOMER_SUPPORT_AGENT'],'synthetic',[],{},SystemTracer(),
        request_coverage=[{'unit_id':'u1','state':'fulfilled'}])


def test_already_scheduled_remaining_work_does_not_spend_a_replan_call(monkeypatch):
    orch=MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router,'replan',lambda **kw:pytest.fail('Scheduled work reached replan provider'))
    output=SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='Verified observation',
        request_unit_ids=['u1'],covered_unit_ids=['u1'],answer_status='complete')
    pending=deque([PlanStep(step_order=2,target_agent='KNOWLEDGE_AGENT',sub_query='Lumen',
        rationale='synthetic',request_unit_ids=['u2'])])
    assert not orch._observe_and_replan(output,'synthetic',pending,['CUSTOMER_SUPPORT_AGENT'],'synthetic',[],{},SystemTracer(),
        request_coverage=[{'unit_id':'u1','state':'fulfilled'},{'unit_id':'u2','state':'delegated'}])
    assert len(pending)==1
