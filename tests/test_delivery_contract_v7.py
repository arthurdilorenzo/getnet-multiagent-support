"""Delivery checks distinguish truthful partial facts from fulfilled needs."""
import importlib.util
from pathlib import Path

import pytest

from app.evals.evaluator import GroundednessJudge
from app.schemas import ChatResponse


def smoke_module():
    path=Path(__file__).resolve().parents[1]/'scripts/docker_smoke.py'
    spec=importlib.util.spec_from_file_location('docker_smoke_v7',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_snapshot_reply():
    return {'abstained':True,'answer_status':'partial',
        'tools_called':['get_sales_settlement'],
        'response':'Snapshot histórico simulado de 2025-03-04; não confirma depósito atual.\n\nPartes do pedido sem resposta verificada:\n- Quando caem as vendas de ontem?',
        'request_coverage':[{'unit_id':'u1','state':'unresolved','text':'Quando caem as vendas de ontem?'}],
        'evidence':[{'kind':'tool_fact','tool_name':'get_sales_settlement','fact_path':'data_scope','value':'synthetic_historical_snapshot'},
                    {'kind':'tool_fact','tool_name':'get_sales_settlement','fact_path':'sales_data.date','value':'2025-03-04'}]}


def test_container_chat_accepts_only_explicit_truthful_historical_partial():
    smoke_module().validate_historical_chat(synthetic_snapshot_reply())


def test_real_offline_snapshot_renderer_satisfies_container_contract(monkeypatch, tmp_path):
    import app.agents.orchestrator as module
    from app.memory.store import StateStore
    from app.schemas import ChatRequest
    from app.security.auth import SecurityContext
    monkeypatch.setattr(module, 'state_store', StateStore(tmp_path/'snapshot.sqlite'))
    orchestrator=module.MultiAgentOrchestrator()
    monkeypatch.setattr(orchestrator, '_observe_and_replan', lambda **kwargs: False)
    response=orchestrator.process(ChatRequest(user_id='cliente1988', session_id='snapshot-contract',
        message='Quando cai o dinheiro das vendas de ontem?'),
        SecurityContext(authenticated_user_id='cliente1988'))
    smoke_module().validate_historical_chat(response.model_dump())


@pytest.mark.parametrize('mutation',['missing_evidence','complete','missing_date','hidden_gap'])
def test_container_chat_rejects_http_success_without_truthful_evidence(mutation):
    reply=synthetic_snapshot_reply()
    if mutation=='missing_evidence':reply['evidence']=[]
    elif mutation=='complete':reply.update(abstained=False,answer_status='complete')
    elif mutation=='missing_date':reply['response']=reply['response'].replace('2025-03-04','data desconhecida')
    elif mutation=='hidden_gap':reply['request_coverage']=[]
    with pytest.raises(AssertionError):smoke_module().validate_historical_chat(reply)


def judge(claim, reference):
    return GroundednessJudge.score_response({'message':'Explain the device.', 'reference_facts':[reference]},
        ChatResponse(user_id='synthetic',agent_used='KNOWLEDGE_AGENT',response=claim))


def test_strict_provenance_suffix_does_not_turn_same_fact_into_paraphrase_failure():
    result=judge('O Terminal Aurora possui teclado físico, conforme o manual de uso disponível no índice público da fonte primária.',
                 'O Terminal Aurora possui teclado físico, como mostra seu manual de uso.')
    assert result['passed'] and result['semantic_review_required']


@pytest.mark.parametrize('claim',[
    'O Terminal Aurora possui touchscreen, conforme o manual de uso.',
    'O Terminal Aurora dispensa confirmação do cliente, conforme o manual de uso.',
    'O cliente define a taxa, conforme o manual de uso.',
    'O Terminal Aurora possui teclado físico, conforme o manual de uso e sem exceções.',
    'O Terminal Aurora possui teclado físico, conforme o manual de uso, e aceita toda modalidade.',
])
def test_provenance_normalization_cannot_launder_predicates_conditions_or_roles(claim):
    assert not judge(claim,'O Terminal Aurora possui teclado físico, como mostra seu manual de uso.')['passed']
