"""Compositional server-owned units and conservative specialist coverage."""
import pytest
from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.router_agent import RouterAgent
from app.memory.store import StateStore
from app.security.auth import SecurityContext
from app.config import settings

CTX=SecurityContext(authenticated_user_id='cliente1988')


def test_shared_units_preserve_every_character_and_public_alias():
    from app.security.request_units import request_units
    from app.rag.semantic import request_units as compatibility_alias
    text='Explain the limit; check my payout if a foreign issuer qualifies.'
    units=request_units(text)
    assert ''.join(unit['text'] for unit in units)==text
    assert compatibility_alias(text)==units
    assert all(text[u['start']:u['end']]==u['text'] for u in units)


def test_ledger_never_credits_unknown_or_unassigned_ids():
    from app.security.request_units import RequestCoverageLedger
    ledger=RequestCoverageLedger('Check my profile; explain the fee.')
    ledger.assign('CUSTOMER_SUPPORT_AGENT',['u1'])
    assert not ledger.report('CUSTOMER_SUPPORT_AGENT',['u1','invented'],['u1'],[],evidence_count=1)
    assert ledger.snapshot()[0]['state']!='fulfilled'
    assert not ledger.report('CUSTOMER_SUPPORT_AGENT',['u2'],['u2'],[],evidence_count=1)
    assert all(row['state']!='fulfilled' for row in ledger.snapshot())
    assert ledger.report('CUSTOMER_SUPPORT_AGENT',['u1'],['u1'],[],evidence_count=1)
    assert ledger.snapshot()[0]['state']=='fulfilled'
    assert ledger.snapshot()[1]['state']=='unresolved'


def test_diagnostic_followup_without_deictic_uses_singular_entity():
    context={'entities':{'terminal_id':'GET-CLASSICA-10442'},'last_agent':'KNOWLEDGE_AGENT'}
    message='Verifique a versão do firmware.'
    assert RouterAgent().plan(message,'cliente1988',context).primary_agent=='CUSTOMER_SUPPORT_AGENT'
    out=CustomerSupportAgent().run(message,CTX,conversation_context=context)
    assert out.structured_data['diagnose_terminal_issue']['terminal']['terminal_id']=='GET-CLASSICA-10442'


def test_explicit_model_or_documental_topic_does_not_inherit_previous_terminal():
    from app.security.request_constraints import extract_constraints
    context={'entities':{'terminal_id':'GET-CLASSICA-10442'}}
    assert not extract_constraints('Check Get Smart firmware.',context).terminal_ids
    assert not extract_constraints('Explique o manual do firmware.',context).terminal_ids


def test_support_reports_only_its_assigned_typed_unit():
    original='Consulte meu perfil; explique a tarifa de exportação.'
    out=CustomerSupportAgent().run('Consulte meu perfil',CTX,authorization_message=original,
        delegation_scope={'assigned_unit_ids':['u1'],'other_unit_ids':['u2']})
    assert out.request_unit_ids==['u1']
    assert out.covered_unit_ids==['u1']
    assert not out.unresolved_unit_ids
    assert 'u2' not in out.covered_unit_ids


def test_support_condition_is_never_covered_by_unconditional_snapshot():
    out=CustomerSupportAgent().run('Consulte minhas vendas de 2026-10-21 se o emissor liberar.',CTX)
    assert out.unresolved_unit_ids
    assert out.answer_status=='partial'


def test_offline_router_assigns_both_documental_and_operational_units():
    message='Consulte meu perfil; explique como funciona o repasse.'
    steps=RouterAgent().plan(message,'cliente1988').plan_steps
    by_agent={s.target_agent:s.request_unit_ids for s in steps}
    assert by_agent['CUSTOMER_SUPPORT_AGENT']==['u1']
    assert by_agent['KNOWLEDGE_AGENT']==['u2']


def test_native_router_rejects_invented_unit_ids(monkeypatch):
    from google.genai import types
    class Provider:
        def __init__(self): self.models=self
        def generate_content(self,**kwargs):
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model',parts=[types.Part(function_call=types.FunctionCall(name='CUSTOMER_SUPPORT_AGENT',args={'sub_query':'Check profile','rationale':'Private query','request_unit_ids':['u99']}))]),finish_reason='STOP')])
    monkeypatch.setattr(settings,'GEMINI_API_KEY','synthetic-not-sent')
    from app.agents.function_calling import RouterPlanningError
    with pytest.raises(RouterPlanningError):
        RouterAgent(Provider()).plan('Check my profile.','cliente1988')


def test_global_ledger_exposes_omitted_documental_clause(monkeypatch,tmp_path):
    import app.agents.orchestrator as module
    from app.schemas import ChatRequest,SpecialistOutput
    orchestrator=module.MultiAgentOrchestrator()
    monkeypatch.setattr(module,'state_store',StateStore(tmp_path/'ledger.sqlite'))
    monkeypatch.setattr(orchestrator,'_observe_and_replan',lambda **kwargs:False)
    observed=[]
    def knowledge(**kwargs):
        observed.append(kwargs)
        return SpecialistOutput(agent_name='KNOWLEDGE_AGENT',response='Documentação insuficiente.',abstained=True,
          semantic_mode='limited_offline',answer_status='abstained',request_unit_ids=['u2'],unresolved_unit_ids=['u2'])
    monkeypatch.setattr(orchestrator.knowledge_agent,'run',knowledge)
    original='Consulte meu perfil; explique como funciona a tarifa de exportação.'
    out=orchestrator.process(ChatRequest(user_id='cliente1988',session_id='ledger',message=original),CTX)
    assert observed[0]['request_message']==original
    assert observed[0]['delegation_scope']['assigned_unit_ids']==['u2']
    rows={row['unit_id']:row for row in out.request_coverage}
    assert rows['u1']['state']=='fulfilled'
    assert rows['u2']['state']=='unresolved'
    assert rows['u2']['text'] in out.response
    assert out.answer_status=='partial'
    assert out.abstained


def test_shared_unit_requires_each_assigned_specialists_evidence():
    from app.security.request_units import RequestCoverageLedger
    ledger=RequestCoverageLedger('Inspect eligibility and execute only if permitted')
    ledger.assign('CUSTOMER_SUPPORT_AGENT',['u1'])
    ledger.assign('KNOWLEDGE_AGENT',['u1'])
    ledger.report('CUSTOMER_SUPPORT_AGENT',['u1'],['u1'],[],evidence_count=1)
    assert ledger.snapshot()[0]['state']!='fulfilled'
    ledger.report('KNOWLEDGE_AGENT',['u1'],[],['u1'],evidence_count=0)
    assert ledger.snapshot()[0]['state']=='unresolved'


@pytest.mark.parametrize('original',['Reinicie GET-SMART-88219 se o fabricante autorizar.','Reinicie meu terminal com falha de hardware.'])
def test_pending_action_proposal_cannot_cover_execution_even_after_diagnosis(original):
    from app.schemas import SpecialistOutput,ToolExecutionResult
    output=SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT',response='Pending proposal',structured_data={'proposal':'pending'},tool_results=[
      ToolExecutionResult(tool_name='diagnose_terminal_issue',success=True,data={'found':True,'terminal':{'terminal_id':'GET-SMART-88219'}}),
      ToolExecutionResult(tool_name='execute_terminal_remote_action',success=True,data={'proposal_id':'PROP-SYNTHETIC','status':'PENDING_CONFIRMATION'})])
    from app.security.request_units import validate_scope
    covered=CustomerSupportAgent._with_unit_coverage(output,original,validate_scope(original),None)
    assert covered.covered_unit_ids==[]
    assert covered.unresolved_unit_ids
    assert covered.answer_status=='partial'


def test_unit_ledger_uses_only_redacted_original_in_chat(monkeypatch,tmp_path):
    import app.agents.orchestrator as module
    from app.schemas import ChatRequest
    orchestrator=module.MultiAgentOrchestrator()
    monkeypatch.setattr(module,'state_store',StateStore(tmp_path/'pii.sqlite'))
    monkeypatch.setattr(orchestrator,'_observe_and_replan',lambda **kwargs:False)
    out=orchestrator.process(ChatRequest(user_id='cliente1988',session_id='pii',message='Consulte meu perfil. Meu CPF é 123.456.789-09.'),CTX)
    assert '123.456.789-09' not in str(out.request_coverage)
    assert '123.456.789-09' not in out.response


def test_factual_coverage_does_not_erase_a_specialist_partial_failure(monkeypatch,tmp_path):
    import app.agents.orchestrator as module
    from app.schemas import ChatRequest,SpecialistOutput
    orchestrator=module.MultiAgentOrchestrator()
    monkeypatch.setattr(module,'state_store',StateStore(tmp_path/'partial.sqlite'))
    monkeypatch.setattr(orchestrator,'_observe_and_replan',lambda **kwargs:False)
    monkeypatch.setattr(orchestrator.knowledge_agent,'run',lambda **kwargs:SpecialistOutput(
      agent_name='KNOWLEDGE_AGENT',response='Extrato verificado; outra etapa falhou.',abstained=True,
      semantic_mode='native_semantic',answer_status='partial',request_unit_ids=['u1'],covered_unit_ids=['u1'],evidence=[{'kind':'verified_span'}]))
    out=orchestrator.process(ChatRequest(user_id='cliente1988',session_id='partial',message='Explique o produto.'),CTX)
    assert out.abstained
    assert out.answer_status!='complete'


def test_verified_nonpertinent_unit_is_exempt_without_factual_credit():
    from app.security.request_units import RequestCoverageLedger
    ledger=RequestCoverageLedger('Please; explain the product.')
    ledger.assign('KNOWLEDGE_AGENT',['u1','u2'])
    assert ledger.report('KNOWLEDGE_AGENT',['u1','u2'],['u2'],[],evidence_count=1,exemptions={'u1':'courtesy'})
    first=ledger.snapshot()[0]
    assert first['state']=='exempt' and not first['pertinent']
    assert not first['fulfilled_by'] and not first['evidence_reported']
    assert not ledger.report('CUSTOMER_SUPPORT_AGENT',['u1'],[],[],exemptions={'u1':'courtesy'})


def test_orchestrator_consolidates_one_literal_footer_without_separator(monkeypatch,tmp_path):
    import app.agents.orchestrator as module
    from app.schemas import ChatRequest,SpecialistOutput
    orchestrator=module.MultiAgentOrchestrator()
    monkeypatch.setattr(module,'state_store',StateStore(tmp_path/'footer.sqlite'))
    monkeypatch.setattr(orchestrator,'_observe_and_replan',lambda **kwargs:False)
    monkeypatch.setattr(orchestrator.knowledge_agent,'run',lambda **kwargs:SpecialistOutput(
      agent_name='KNOWLEDGE_AGENT',response='Extrato verificado.\n\nPartes do pedido sem resposta verificada:\n- and unknown condition.',
      abstained=True,semantic_mode='limited_offline',answer_status='partial',request_unit_ids=['u1','u2'],covered_unit_ids=['u1'],unresolved_unit_ids=['u2'],evidence=[{'kind':'verified_span'}]))
    out=orchestrator.process(ChatRequest(user_id='cliente1988',session_id='footer',message='Explain the product and unknown condition.'),CTX)
    marker='Partes do pedido sem resposta verificada:'
    assert out.response.count(marker)==1
    assert '---' not in out.response.split(marker)[0][-10:]
    assert out.request_coverage[1]['text'] in out.response


@pytest.mark.parametrize('mutation',[None,'hash','agent','verdict'])
def test_exemptions_require_native_canonical_request_proof(mutation):
    import hashlib
    from app.agents.orchestrator import MultiAgentOrchestrator
    from app.schemas import SpecialistOutput,ToolExecutionResult
    original='Please; explain the product.'
    proof={'method':'native_two_pass_evidence_authorization','request_hash':hashlib.sha256(original.encode()).hexdigest(),
      'coverage':[{'unit_id':'u1','kind':'courtesy','verification':{'verdict':'accept'}}]}
    out=SpecialistOutput(agent_name='KNOWLEDGE_AGENT',response='Verified extract',semantic_mode='native_semantic',
      request_unit_ids=['u1','u2'],structured_data={'request_unit_exemptions':{'u1':'courtesy'}},
      tool_results=[ToolExecutionResult(tool_name='search_getnet_knowledge',success=True,data={'answerability':proof})])
    if mutation=='hash': out.tool_results[0].data['answerability']['request_hash']='invented'
    if mutation=='agent': out.agent_name='CUSTOMER_SUPPORT_AGENT'
    if mutation=='verdict': out.tool_results[0].data['answerability']['coverage'][0]['verification']['verdict']='reject'
    result=MultiAgentOrchestrator._verified_unit_exemptions(out,original)
    assert result == ({'u1':'courtesy'} if mutation is None else None)


@pytest.mark.parametrize('message',[
    'Pode verificar a versão do aparelho que citei anteriormente?',
    'Confira a versão do equipamento mencionado acima.',
    'Please inspect the version of the device mentioned earlier.',
    'Check the equipment version.',
    'Consulte o estado do aparelho referido.',
])
def test_inspection_and_persisted_device_scope_resolve_reference_family(message):
    context={'entities':{'terminal_id':'GET-CLASSICA-10442'},'last_agent':'KNOWLEDGE_AGENT'}
    assert RouterAgent().plan(message,'cliente1988',context).primary_agent=='CUSTOMER_SUPPORT_AGENT'
    out=CustomerSupportAgent().run(message,CTX,conversation_context=context)
    assert out.structured_data['diagnose_terminal_issue']['terminal']['terminal_id']=='GET-CLASSICA-10442'


@pytest.mark.parametrize('message',[
    'Check a different terminal version.',
    'Verifique a versão de outro aparelho.',
])
def test_new_unspecified_device_requires_clarification_instead_of_old_target(message):
    context={'entities':{'terminal_id':'GET-CLASSICA-10442'}}
    out=CustomerSupportAgent().run(message,CTX,conversation_context=context)
    assert not out.structured_data
    assert out.abstained


def test_proactive_human_meta_step_does_not_claim_original_diagnosis_unit():
    from app.schemas import PlanStep,RoutingPlan
    plan=RoutingPlan(primary_agent='HUMAN_ESCALATION_AGENT',confidence=1.0,reasoning='Hardware follow-up',plan_steps=[
      PlanStep(step_order=1,target_agent='HUMAN_ESCALATION_AGENT',sub_query='Open hardware review',rationale='Diagnostic observation')])
    bound=RouterAgent.bind_request_units(plan,'Inspecione o hardware do meu terminal.')
    assert bound.plan_steps[0].request_unit_ids==[]


@pytest.mark.parametrize('message',[
    'Minha maquininha está com defeito no leitor de chip, o que está acontecendo?',
    'Meu terminal apresenta falha no leitor; verifique o hardware.',
])
def test_typed_hardware_fault_sustains_recognized_diagnostic_relation(message):
    out=CustomerSupportAgent().run(message,SecurityContext(authenticated_user_id="cliente3012"))
    assert out.structured_data['diagnose_terminal_issue']['requires_hardware_replacement']
    assert out.covered_unit_ids
    assert not out.unresolved_unit_ids
