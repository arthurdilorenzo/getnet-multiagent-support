import pytest
from pydantic import ValidationError
from app.v2.contracts import *
from app.v2.knowledge import validate_answer, EvidenceError
from tests.v2.test_consolidation_acceptance import make_customers_json
from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager
from app.v2.adapters import ToolGateway
from app.security.auth import SecurityContext


def fixture_req():
 b=Binding(binding_id='r0001:b0001',tool_name='diagnose_terminal_issue',resource_type='terminal',resource_id='POS-9999',field_pointer='/ultimo_erro_transacao')
 r=Requirement(requirement_id='r0001',kind='operational',summary='Resultado da tentativa',bindings=[b])
 f=Fact(fact_id='f1',binding_id=b.binding_id,operation_id='op1',resource_type='terminal',resource_id=b.resource_id,field_pointer=b.field_pointer,availability='available',value='E99')
 return r,f


def test_a3_mixed_block_cannot_bypass_operational_renderer():
 r,f=fixture_req()
 d=AnswerDocument(requirements=[RequirementDecision(requirement_id=r.requirement_id,status='supported')],final_answer=[OperationalBlock(requirement_id=r.requirement_id,fact_ids=['f1']),DocumentaryBlock(requirement_id=r.requirement_id,text='Saldo inventado BRL 350',span_ids=['invented'])])
 with pytest.raises(EvidenceError): validate_answer(d,[],[f],[r])


def test_a3_missing_fact_cannot_prove_contradiction():
 r,f=fixture_req();f.availability='unavailable';f.value=None
 d=AnswerDocument(requirements=[RequirementDecision(requirement_id=r.requirement_id,status='contradicted')],final_answer=[OperationalBlock(requirement_id=r.requirement_id,fact_ids=['f1'])])
 with pytest.raises(EvidenceError): validate_answer(d,[],[f],[r])


@pytest.mark.parametrize('unit',[None,'BRL'])
def test_a3_actual_gateway_never_fabricates_target_or_unit(tmp_path,unit):
 store=StateStore(tmp_path/'state.db');store.ensure_session('s1','u1')
 path=make_customers_json(tmp_path,{'u1':{'user_id':'u1','terminals':[{'terminal_id':'POS-1111','battery':90}], 'recent_transactions':[{'transaction_id':'T1'}],'ultimo_erro_transacao':99}})
 g=ToolGateway(path,store=store,queue=HITLQueueManager(store))
 r,_=fixture_req();b=r.bindings[0];b.unit=unit
 result=g.execute('diagnose_terminal_issue',{'terminal_id':'POS-9999'},SecurityContext('u1',scopes=['chat:write','customer:read_own']),'s1')
 facts=g.extract_facts('diagnose_terminal_issue',result,[b])
 assert not any(f.availability=='available' for f in facts)
 b.resource_id='POS-1111';b.field_pointer='/terminal/battery';b.unit='BRL'
 result=g.execute('diagnose_terminal_issue',{'terminal_id':'POS-1111'},SecurityContext('u1',scopes=['chat:write','customer:read_own']),'s1')
 assert not any(f.availability=='available' for f in g.extract_facts('diagnose_terminal_issue',result,[b]))


def test_a4_dto_rejects_coercion():
 with pytest.raises(ValidationError): StepProposal(kind='tool',requirement_indexes=['0'])

@pytest.mark.parametrize('block',[HumanBlock(requirement_id='r0001',observation_id='fabricated'),ContextBlock(requirement_id='r0001',context_ref='Invented merchant statement')])
def test_a3_attribution_requires_existing_reference(block):
 r,_=fixture_req()
 d=AnswerDocument(requirements=[RequirementDecision(requirement_id=r.requirement_id,status='unanswerable')],final_answer=[block])
 with pytest.raises(EvidenceError): validate_answer(d,[],[],[r])


def test_a3_fact_is_scalar():
 with pytest.raises(ValidationError): Fact(fact_id='f',operation_id='op',resource_type='customer',resource_id='u',field_pointer='/x',availability='available',value={'injected':'value'})


def test_a3_exact_binding_identity():
 r,f=fixture_req();f.binding_id='other-binding'
 d=AnswerDocument(requirements=[RequirementDecision(requirement_id=r.requirement_id,status='supported')],final_answer=[OperationalBlock(requirement_id=r.requirement_id,fact_ids=['f1'])])
 with pytest.raises(EvidenceError): validate_answer(d,[],[f],[r])


def test_a3_numeric_field_rejects_wrong_source_type(tmp_path):
 store=StateStore(tmp_path/'state.db');store.ensure_session('s1','u1')
 path=make_customers_json(tmp_path,{'u1':{'user_id':'u1','sales_data':{'gross_amount_brl':'abc'}}})
 g=ToolGateway(path,store=store,queue=HITLQueueManager(store))
 # Real extraction frontier, completed server result with malformed source data.
 from app.schemas import ToolExecutionResult
 result=ToolExecutionResult(tool_name='get_sales_settlement',success=True,operation_id='op',outcome_state='completed',data={'user_id':'u1','sales_data':{'gross_amount_brl':'abc'}})
 b=Binding(binding_id='r:b',tool_name='get_sales_settlement',resource_type='customer',resource_id='u1',field_pointer='/sales_data/gross_amount_brl',unit='BRL')
 assert not any(f.availability=='available' for f in g.extract_facts('get_sales_settlement',result,[b]))


def test_a3_snapshot_scope_survives_operational_rendering(tmp_path):
 from app.v2.knowledge import format_fact
 store=StateStore(tmp_path/'state.db');store.ensure_session('s1','u1')
 path=make_customers_json(tmp_path,{'u1':{'user_id':'u1','sales_settlement':{'yesterday_sales':{'date':'2026-10-20','gross_amount_brl':125.0}}}})
 g=ToolGateway(path,store=store,queue=HITLQueueManager(store))
 result=g.execute('get_sales_settlement',{},SecurityContext('u1',scopes=['chat:write','customer:read_own']),'s1')
 b=Binding(binding_id='r:b',tool_name='get_sales_settlement',resource_type='customer',resource_id='u1',field_pointer='/sales_data/gross_amount_brl',unit='BRL')
 text=format_fact(g.extract_facts('get_sales_settlement',result,[b])[0])
 assert '2026-10-20' in text and 'Snapshot histórico' in text and '125,00' in text


def test_a4_operational_requirement_needs_binding():
 p=PlanProposal(requirements=[RequirementProposal(kind='operational',summary='Saldo')],steps=[StepProposal(kind='tool',tool_name='get_customer_profile',requirement_indexes=[0])])
 with pytest.raises(ValueError): canonicalize_plan(p)
