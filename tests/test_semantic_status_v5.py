import pytest
from pydantic import ValidationError

from app.agents.orchestrator import MultiAgentOrchestrator
from app.schemas import ChatResponse, SpecialistOutput


def output(status='', mode='', abstained=False):
    return SpecialistOutput(agent_name='KNOWLEDGE_AGENT', response='Extrato sintético.',
                            answer_status=status, semantic_mode=mode, abstained=abstained)


def test_schema_preserves_mode_and_completion_independently_of_prose():
    out = output('partial', 'native_semantic', True)
    assert out.answer_status == 'partial'
    assert out.semantic_mode == 'native_semantic'
    reply = ChatResponse(response=out.response, user_id='synthetic', agent_used=out.agent_name,
                         answer_status=out.answer_status, semantic_mode=out.semantic_mode)
    assert reply.model_dump()['answer_status'] == 'partial'


def test_status_does_not_accept_an_unregistered_semantic_mode():
    with pytest.raises(ValidationError):
        output('complete', 'universal_verified')


@pytest.mark.parametrize('parts,expected', [
    ([output('complete', 'native_semantic')], ('native_semantic', 'complete')),
    ([output('complete', 'native_semantic'), output('partial', 'native_semantic', True)], ('native_semantic', 'partial')),
    ([output('abstained', 'native_semantic', True)], ('native_semantic', 'abstained')),
    ([output('complete', 'native_semantic'), output()], ('native_semantic', 'partial')),
    ([output('complete', 'native_semantic'), output('complete', 'limited_offline')], ('limited_offline', 'complete')),
    ([output()], ('', '')),
])
def test_chain_metadata_never_promotes_incomplete_steps(parts, expected):
    assert MultiAgentOrchestrator._answer_metadata(parts) == expected


def test_evaluation_separates_modes_without_calling_metadata_semantic_truth(monkeypatch, tmp_path):
    import importlib
    import json
    module = importlib.import_module('app.evals.evaluator')
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps({'cases': [
        {'case_id': name, 'category': 'synthetic', 'user_id': 'synthetic',
         'message': 'Dado sintético.', 'expected_agent': 'KNOWLEDGE_AGENT', 'expected_tools': [],
         'reference_facts': ['Dado sintético.']}
        for name in ('native', 'limited')]}))
    responses = iter([
        ChatResponse(response='Dado sintético.', user_id='synthetic', agent_used='KNOWLEDGE_AGENT',
                     semantic_mode='native_semantic', answer_status='complete'),
        ChatResponse(response='Não encontrei informações suficientes.', user_id='synthetic', agent_used='KNOWLEDGE_AGENT',
                     semantic_mode='limited_offline', answer_status='abstained', abstained=True)])
    monkeypatch.setattr(module.orchestrator, 'process', lambda request: next(responses))
    report = module.SystemEvaluator(path).run_suite()
    assert report['metrics_by_mode']['native_semantic']['case_count'] == 1
    assert report['metrics_by_mode']['native_semantic']['limited_consistency_pass_rate'] == 1
    assert report['metrics_by_mode']['limited_offline']['non_abstention_rate'] == 0
    assert report['cases'][1]['answer_status'] == 'abstained'
    assert report['mode_metrics_limitations']
