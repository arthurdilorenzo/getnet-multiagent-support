"""A verified local gap cannot invent another source or repeat the same lookup."""
from collections import deque
import importlib
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from app.agents.knowledge_agent import KnowledgeAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.config import settings
from app.memory.store import StateStore
from app.observability.resilience import breaker_registry
from app.observability.tracer import SystemTracer
from app.schemas import ChatRequest, PlanStep, RoutingPlan, ToolExecutionResult
from app.security.auth import SecurityContext
from app.security.request_units import request_units, validate_scope


class GapProvider:
    """Only external model decisions are synthetic; tools/guards/semantics are real."""
    def __init__(self, message):
        self.models=self
        self.message=message
        self.calls=[]

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        config=kwargs['config']
        if config.tools:
            assert len(self.calls)==1, 'A verified gap must not trigger a second tool planner'
            parts=[types.Part(function_call=types.FunctionCall(name='KNOWLEDGE_AGENT',args={
                'sub_query':self.message,'rationale':'local product documentation','request_unit_ids':['u1']}))]
        else:
            payload=json.loads(kwargs['contents'][0].parts[0].text)
            ids=list(config.response_json_schema['properties']['checks' if 'analysis' in payload else 'units']['properties'])
            if 'analysis' in payload:
                result={'request_hash':payload['request_hash'],'complete':False,'checks':{
                    uid:{'verdict':'accept','issues':[],'affected_unit_ids':[]} for uid in ids}}
            else:
                result={'request_hash':payload['request_hash'],'units':{uid:{
                    'kind':'requirement','decision':'unknown','subject':'Getnet Crediário',
                    'relation':'settlement timing','value':'','unit':'','qualifiers':[], 'span_ids':[]} for uid in ids}}
            parts=[types.Part.from_text(text=json.dumps(result))]
        return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',
            content=types.Content(role='model',parts=parts))])


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-never-sent')
    monkeypatch.setattr(settings,'GETNET_LIVE_EMBEDDINGS',False)
    breaker_registry.reset_all()
    yield
    breaker_registry.reset_all()


@pytest.mark.parametrize('message',[
    'Em Portugal, o Crediário Getnet paga em dois dias úteis como no Brasil?',
    'Does Getnet Crediário settle within two business days in Chile?',
])
def test_native_public_documentary_gap_stops_after_first_verified_lookup(monkeypatch,tmp_path,message):
    module=importlib.import_module('app.agents.orchestrator')
    monkeypatch.setattr(module,'state_store',StateStore(tmp_path/'state.db'))
    provider=GapProvider(message)
    monkeypatch.setattr(module.genai,'Client',lambda **kw:provider)
    orch=MultiAgentOrchestrator()
    out=orch.process(ChatRequest(message=message,user_id='synthetic',session_id='gap'),SecurityContext('synthetic'))
    assert out.abstained and out.answer_status=='abstained'
    assert not out.citations and not out.evidence and not out.action_proposals
    assert out.request_coverage[0]['state']=='unresolved'
    assert out.tools_called==['search_getnet_knowledge']
    assert len(provider.calls)==3
    assert [s.details['phase'] for s in out.trace if s.action=='provider_generation']==[
        'routing','semantic_analysis','semantic_verification']
    assert len([s for s in out.trace if s.action=='call:search_getnet_knowledge'])==1
    assert any(s.action=='replan_not_needed' and s.details.get('reason')=='verified_local_documentation_exhausted' for s in out.trace)


def verified_gap(message):
    provider=GapProvider(message)
    # A routed documentary read starts directly with the two real semantic passes.
    out=KnowledgeAgent(provider).run(message,SecurityContext('synthetic'),
        request_message=message,delegation_scope=validate_scope(message,{'assigned_unit_ids':['u1'],
            'other_unit_ids':[u['unit_id'] for u in request_units(message) if u['unit_id']!='u1']}),documentary_dispatch=True)
    return out


@pytest.mark.parametrize('exception', ['explicit_web','private','explicit_human','unattempted_unit',
    'human','tool_failure','unverified_metadata','empty_coverage'])
def test_verified_gap_keeps_independent_actionable_work(monkeypatch,exception):
    base='Explique o Crediário Getnet em Portugal.'
    message=(base+' Pesquise na web as condições atuais.' if exception=='explicit_web' else
             base+' Consulte meu perfil.' if exception=='private' else
             base+' Quero falar com atendente humano.' if exception=='explicit_human' else
             base+' Explique também o Pix.' if exception=='unattempted_unit' else base)
    # Keep one literal documentary unit for the provider; tests below use the
    # server-produced ToolExecutionResult rather than forged memory metadata.
    out=verified_gap(base)
    if exception=='human':out.requires_escalation=True
    if exception=='tool_failure':
        out.tool_results.append(ToolExecutionResult(tool_name='web_search_general',success=False,
            error_code='EXTERNAL_SERVICE_UNAVAILABLE'))
    if exception=='unverified_metadata':out.tool_results[0].data['answerability'].pop('method')
    if exception=='empty_coverage':out.tool_results[0].data['answerability']['coverage']=[]
    target='CUSTOMER_SUPPORT_AGENT' if exception=='private' else 'HUMAN_ESCALATION_AGENT' if exception in {'human','tool_failure','explicit_human'} else 'WEB_SEARCH_AGENT'
    ids=['u2'] if exception in {'private','explicit_human','unattempted_unit'} else [u['unit_id'] for u in request_units(message)]
    orch=MultiAgentOrchestrator();pending=deque();calls=[]
    def replan(**kwargs):
        calls.append(kwargs)
        return RoutingPlan(primary_agent=target,confidence=.9,reasoning='actionable',plan_steps=[
            PlanStep(step_order=1,target_agent=target,sub_query=message,rationale='actionable',request_unit_ids=ids)])
    monkeypatch.setattr(orch.router,'replan',replan)
    coverage=[{'unit_id':u['unit_id'],'text':u['text'],'state':'unresolved'} for u in request_units(message)]
    assert not orch._observe_and_replan(out,message,pending,['KNOWLEDGE_AGENT'],'synthetic',[],{},SystemTracer(),request_coverage=coverage)
    assert len(calls)==1 and len(pending)==1
    assert pending[0].target_agent==target


def test_verified_gap_preserves_already_scheduled_private_step(monkeypatch):
    base='Explique o Crediário Getnet em Portugal.';message=base+' Consulte meu perfil.'
    out=verified_gap(base);orch=MultiAgentOrchestrator()
    pending=deque([PlanStep(step_order=2,target_agent='CUSTOMER_SUPPORT_AGENT',sub_query='Consulte meu perfil.',rationale='private',request_unit_ids=['u2'])])
    monkeypatch.setattr(orch.router,'replan',lambda **kw:pytest.fail('Already scheduled work must remain, without another plan'))
    coverage=[{'unit_id':u['unit_id'],'text':u['text'],'state':'unresolved'} for u in request_units(message)]
    assert not orch._observe_and_replan(out,message,pending,['KNOWLEDGE_AGENT'],'synthetic',[],{},SystemTracer(),request_coverage=coverage)
    assert len(pending)==1 and pending[0].request_unit_ids==['u2']
