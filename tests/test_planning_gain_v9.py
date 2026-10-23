"""Planning gain is permission from canonical pending units, not model rationale."""
from collections import deque
import pytest
from app.agents.router_agent import RouterAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.observability.tracer import SystemTracer
from app.schemas import PlanStep, RoutingPlan, SpecialistOutput, ToolExecutionResult
from app.security.request_units import request_units


def step(agent='KNOWLEDGE_AGENT', ids=None):
    return PlanStep(step_order=1,target_agent=agent,request_unit_ids=ids or [],
                    sub_query='model proposed work',rationale='model rationale is not gain')

def plan(steps):
    return RoutingPlan(primary_agent=steps[0].target_agent if steps else 'KNOWLEDGE_AGENT',
                       plan_steps=steps,confidence=.9,reasoning='synthetic')

@pytest.mark.parametrize('veto',[
    'Never reboot GET-SYNTH-OTHER-991.',
    'Please, do not restart the terminal GET-SYNTH-OTHER-991.',
    'Não reinicie GET-SYNTH-OTHER-991.',
    'Por favor, nunca reiniciar o terminal GET-SYNTH-OTHER-991.',
    'Do not update the parameter table of GET-SYNTH-OTHER-991.',
    'Não troque a operadora de GET-SYNTH-OTHER-991.',
])
@pytest.mark.parametrize('first',[False,True])
def test_initial_binding_does_not_delegate_independent_veto(veto,first):
    read='Check the status of GET-SYNTH-READ-772.'
    message=(veto+' '+read) if first else (read+'; '+veto)
    units=request_units(message)
    veto_id=next(u['unit_id'] for u in units if 'OTHER' in u['text'])
    read_id=next(u['unit_id'] for u in units if 'READ' in u['text'])
    bound=RouterAgent.bind_request_units(plan([step('CUSTOMER_SUPPORT_AGENT',[read_id]),step(ids=[veto_id])]),message)
    assert len(bound.plan_steps)==1
    assert bound.plan_steps[0].request_unit_ids==[read_id]
    assert request_units(message)==units
    # A second automatic binding must not put the ownerless veto back.
    assert RouterAgent.bind_request_units(bound,message).plan_steps[0].request_unit_ids==[read_id]

@pytest.mark.parametrize('message',[
    'Do not reboot GET-SYNTH-OTHER-991 unless it is offline.',
    'Do not reboot GET-SYNTH-OTHER-991 and check its status.',
    'Do not tell me to reboot GET-SYNTH-OTHER-991.',
    'Never erase GET-SYNTH-OTHER-991.',
    'Não reinicie o terminal.',
])
def test_uncertain_or_positive_attached_clause_is_not_filtered(message):
    original=request_units(message)
    result=RouterAgent.bind_request_units(plan([step()]),message)
    assert result.plan_steps and result.plan_steps[0].request_unit_ids==[u['unit_id'] for u in original]


def test_veto_only_pending_skips_provider_replan_even_after_error(monkeypatch):
    message='Check status of GET-SYNTH-READ-772; never reboot GET-SYNTH-OTHER-991.'
    orch=MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router,'replan',lambda **kw:pytest.fail('veto is not provider work'))
    out=SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='Offline',abstained=True,
        structured_data={'semantic_error':{'code':'PROVIDER_503'}})
    pending=deque()
    failed=orch._observe_and_replan(out,message,pending,['CUSTOMER_SUPPORT_AGENT'],'synthetic',[],{},SystemTracer(),[
        {'unit_id':'u1','state':'fulfilled'},{'unit_id':'u2','state':'unresolved'}])
    assert not failed and not pending


def test_replan_does_not_duplicate_scheduled_units_across_agents(monkeypatch):
    message='Explain Pix; check status of GET-SYNTH-READ-772.'
    orch=MultiAgentOrchestrator()
    monkeypatch.setattr(orch.router,'replan',lambda **kw:plan([step('KNOWLEDGE_AGENT',['u1'])]))
    pending=deque([step('WEB_SEARCH_AGENT',['u1'])])
    out=SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='partial',abstained=True)
    orch._observe_and_replan(out,message,pending,['CUSTOMER_SUPPORT_AGENT'],'synthetic',[],{},SystemTracer(),[
        {'unit_id':'u1','state':'unresolved'},{'unit_id':'u2','state':'fulfilled'}])
    assert len(pending)==1


def native_provider(*delegations):
    from google.genai import types
    class Provider:
        models=None
        calls=0
        def __init__(self): self.models=self
        def generate_content(self,**kwargs):
            self.calls+=1
            return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',
                content=types.Content(role='model',parts=[types.Part(function_call=types.FunctionCall(
                    name=s.target_agent,args={'request_unit_ids':s.request_unit_ids,
                                             'sub_query':s.sub_query,'rationale':s.rationale}))
                    for s in delegations]))])
    return Provider()


def test_native_sdk_initial_veto_filtered_and_no_replan_after_read(monkeypatch,tmp_path):
    import importlib
    from app.config import settings
    from app.memory.store import StateStore
    from app.schemas import ChatRequest
    from app.security.auth import SecurityContext
    from app.observability.resilience import breaker_registry
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    monkeypatch.setattr(importlib.import_module('app.agents.orchestrator'),'state_store',StateStore(tmp_path/'state.db'))
    breaker_registry.reset_all()
    message='Check status of GET-SYNTH-READ-772; never reboot GET-SYNTH-OTHER-991.'
    sdk=native_provider(step('CUSTOMER_SUPPORT_AGENT',['u1']),step('KNOWLEDGE_AGENT',['u2']))
    orch=MultiAgentOrchestrator();orch.router=RouterAgent(sdk)
    executed=[]
    def delegate(**kwargs):
        executed.append(kwargs)
        assert kwargs['target_agent']=='CUSTOMER_SUPPORT_AGENT'
        assert kwargs['delegation_scope']['assigned_unit_ids']==['u1']
        return SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='Status: Offline',
            request_unit_ids=['u1'],covered_unit_ids=['u1'],evidence=[{'source':'synthetic observation'}],
            tool_results=[ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data={
                'found': True, 'user_id': 'synthetic', 'requested_terminal_id': 'GET-SYNTH-READ-772',
                'terminal': {'terminal_id': 'GET-SYNTH-READ-772', 'status': 'Offline'}})])
    monkeypatch.setattr(orch,'_delegate_to_specialist',delegate)
    result=orch.process(ChatRequest(message=message,user_id='synthetic',session_id='planning-v9'),SecurityContext('synthetic'))
    assert sdk.calls==1 and len(executed)==1
    assert result.answer_status=='partial' and not result.action_proposals and not result.tools_called
    assert result.request_coverage[0]['state']=='fulfilled'
    assert result.request_coverage[1]['state']=='unresolved'
    assert result.request_coverage[1]['assigned_agents']==[]
    assert 'never reboot GET-SYNTH-OTHER-991' in result.response
    breaker_registry.reset_all()


def test_native_sdk_replan_proposal_cannot_reintroduce_veto_or_covered_read(monkeypatch):
    from app.config import settings
    from app.observability.resilience import breaker_registry
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent');breaker_registry.reset_all()
    message='Check status of GET-SYNTH-READ-772; never reboot GET-SYNTH-OTHER-991; Explain Pix.'
    sdk=native_provider(step('KNOWLEDGE_AGENT',['u2']),step('CUSTOMER_SUPPORT_AGENT',['u1']))
    orch=MultiAgentOrchestrator();orch.router=RouterAgent(sdk)
    pending=deque()
    out=SpecialistOutput(agent_name='WEB_SEARCH_AGENT',response='No source',abstained=True)
    failed=orch._observe_and_replan(out,message,pending,['WEB_SEARCH_AGENT'],'synthetic',[],{},SystemTracer(),[
        {'unit_id':'u1','state':'fulfilled'},{'unit_id':'u2','state':'unresolved'},{'unit_id':'u3','state':'unresolved'}])
    assert sdk.calls==1 and not failed and not pending
    breaker_registry.reset_all()


def test_explicit_veto_delegation_cannot_be_reassigned_omitted_positive_or_handoff():
    message='Check status of GET-SYNTH-READ-772; never reboot GET-SYNTH-OTHER-991.'
    for agent in ['KNOWLEDGE_AGENT','CUSTOMER_SUPPORT_AGENT','HUMAN_ESCALATION_AGENT']:
        bound=RouterAgent.bind_request_units(plan([step(agent,['u2'])]),message)
        assert not bound.plan_steps


def test_veto_only_has_no_delegation_even_if_provider_proposes_human():
    message='Never reboot GET-SYNTH-OTHER-991.'
    assert not RouterAgent.bind_request_units(plan([step('HUMAN_ESCALATION_AGENT')]),message).plan_steps
