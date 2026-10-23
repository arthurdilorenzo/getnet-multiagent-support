"""Replanning cannot turn a public evidence gap into private account access."""
from collections import deque

import pytest

from app.agents.orchestrator import MultiAgentOrchestrator
from app.schemas import PlanStep, RoutingPlan, SpecialistOutput
from app.observability.tracer import SystemTracer


def output(**changes):
    values=dict(agent_name='KNOWLEDGE_AGENT',response='Unsupported public relation',
                abstained=True,answer_status='abstained',request_unit_ids=['u1'],unresolved_unit_ids=['u1'])
    values.update(changes)
    return SpecialistOutput(**values)


def step(target='CUSTOMER_SUPPORT_AGENT',ids=None):
    return PlanStep(step_order=1,target_agent=target,sub_query='model invented private profile',
                    rationale='model rationale is not authority',request_unit_ids=ids or [])


def run(monkeypatch,message,steps,out=None,rows=None):
    orch=MultiAgentOrchestrator();pending=deque();tracer=SystemTracer();calls=[]
    def replan(**kwargs):
        calls.append(kwargs)
        return RoutingPlan(primary_agent=steps[0].target_agent if steps else 'KNOWLEDGE_AGENT',
            plan_steps=steps,confidence=.9,reasoning='synthetic proposal')
    monkeypatch.setattr(orch.router,'replan',replan)
    out=out or output()
    failed=orch._observe_and_replan(out,message,pending,[out.agent_name],'synthetic',[],{},tracer,
        request_coverage=rows or [{'unit_id':'u1','text':message,'state':'unresolved'}])
    return pending,tracer,calls,failed


@pytest.mark.parametrize('message',[
    'Explique as condições de antecipação automática.',
    'Explain the eligibility conditions of the payment service.',
    'What is the public settlement schedule?',
])
def test_public_gap_cannot_schedule_private_profile(monkeypatch,message):
    pending,tracer,calls,failed=run(monkeypatch,message,[step()])
    assert calls and not failed
    assert not pending
    assert any(s.action=='replan_scope_rejected' for s in tracer.get_steps())


def test_semantic_provider_failure_without_new_work_does_not_replan(monkeypatch):
    out=output(structured_data={'semantic_error':{'stage':'semantic_analysis','code':'PROVIDER_500'}})
    pending,tracer,calls,failed=run(monkeypatch,'Explain the public service.',[step()],out)
    assert not calls and not pending and not failed
    assert any(s.action=='replan_not_needed' and s.details['reason']=='no_actionable_observation'
               for s in tracer.get_steps())


def test_explicit_private_remaining_unit_is_allowed(monkeypatch):
    message='Explique o serviço; consulte meu perfil.'
    pending,_,calls,failed=run(monkeypatch,message,[step(ids=['u2'])],rows=[
        {'unit_id':'u1','state':'unresolved'},{'unit_id':'u2','state':'unresolved'}])
    assert calls and not failed and len(pending)==1
    assert pending[0].request_unit_ids==['u2']


def test_private_step_cannot_borrow_a_public_unit_in_mixed_request(monkeypatch):
    message='Explique o serviço; consulte meu perfil.'
    pending,_,_,failed=run(monkeypatch,message,[step(ids=['u1'])],rows=[
        {'unit_id':'u1','state':'unresolved'},{'unit_id':'u2','state':'unresolved'}])
    assert not failed and not pending


def test_already_covered_private_unit_is_not_new_work(monkeypatch):
    message='Explique o serviço; consulte meu perfil.'
    pending,_,_,failed=run(monkeypatch,message,[step(ids=['u2'])],rows=[
        {'unit_id':'u1','state':'unresolved'},{'unit_id':'u2','state':'fulfilled'}])
    assert not failed and not pending


def test_unknown_replan_unit_cannot_be_scheduled(monkeypatch):
    pending,_,_,failed=run(monkeypatch,'Explain the service.',[step(ids=['u999'])])
    assert not failed and not pending


def test_critical_handoff_is_not_blocked_by_covered_ledger(monkeypatch):
    out=output(agent_name='CUSTOMER_SUPPORT_AGENT',requires_escalation=True)
    pending,_,calls,failed=run(monkeypatch,'Check my terminal.',[step('HUMAN_ESCALATION_AGENT')],out,
        rows=[{'unit_id':'u1','state':'fulfilled'}])
    assert calls and not failed and len(pending)==1
    assert pending[0].target_agent=='HUMAN_ESCALATION_AGENT'


def test_public_provider_failure_preserves_pending_and_never_reads_profile(monkeypatch,tmp_path):
    import importlib
    from app.memory.store import StateStore
    from app.schemas import ChatRequest
    from app.security.auth import SecurityContext
    store=StateStore(tmp_path/'state.db')
    monkeypatch.setattr(importlib.import_module('app.agents.orchestrator'),'state_store',store)
    orch=MultiAgentOrchestrator()
    message='Explain the public eligibility conditions.'
    monkeypatch.setattr(orch.router,'plan',lambda **kw:RoutingPlan(primary_agent='KNOWLEDGE_AGENT',
        confidence=.9,reasoning='public',plan_steps=[PlanStep(step_order=1,target_agent='KNOWLEDGE_AGENT',
            sub_query=message,rationale='public',request_unit_ids=['u1'])]))
    monkeypatch.setattr(orch.knowledge_agent,'run',lambda **kw:output(
        structured_data={'semantic_error':{'stage':'semantic_verification','code':'PROVIDER_503'}}))
    monkeypatch.setattr(orch.router,'replan',lambda **kw:pytest.fail('No new observation justifies provider replan'))
    monkeypatch.setattr(orch.customer_support_agent,'run',lambda **kw:pytest.fail('Public query read private profile'))
    result=orch.process(ChatRequest(message=message,user_id='synthetic',session_id='v8-public'),
                        SecurityContext('synthetic'))
    assert result.abstained and result.answer_status=='abstained'
    assert result.request_coverage[0]['state']=='unresolved'
    assert message in result.response
    assert not result.tools_called
    assert 'CNPJ' not in result.response


@pytest.mark.parametrize('message',[
    'Consulte o status e a conexão de GET-SYNTH-VISTA-554.',
    'Check status and connection of GET-SYNTH-VISTA-554.',
])
def test_shared_original_read_subject_can_authorize_pending_private_unit(monkeypatch,message):
    pending,_,_,failed=run(monkeypatch,message,[step(ids=['u1'])],rows=[
        {'unit_id':'u1','state':'unresolved'},{'unit_id':'u2','state':'fulfilled'}])
    assert not failed and len(pending)==1
    assert pending[0].request_unit_ids==['u1']
