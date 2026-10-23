import pytest

from app.evals.evaluator import GroundednessJudge
from app.schemas import ChatResponse


@pytest.mark.parametrize('message', [
    'Qual é a taxa de 5% citada nesse contrato?',
    'Will the terminal be rebooted without my confirmation?',
])
def test_literal_pending_question_is_not_a_factual_assertion(message):
    response = ChatResponse(user_id='synthetic', agent_used='KNOWLEDGE_AGENT',
        response='Não encontrei evidência suficiente.'
                 '\n\nPartes do pedido sem resposta verificada:\n- ' + message,
        abstained=True, answer_status='abstained')
    case = {'message': message, 'expect_abstained': True}
    assert GroundednessJudge.score_response(case, response)['passed']
    # An unanswered covered requirement still fails fulfillment.
    assert not GroundednessJudge.score_response({**case, 'expect_abstained': False}, response)['passed']


@pytest.mark.parametrize('claims,pending', [
    ('A taxa é 5%.', 'Qual é a taxa do contrato?'),
    ('Não encontrei evidência suficiente.', 'Taxa inexistente na pergunta original.'),
])
def test_pending_footer_cannot_launder_a_claim_or_forged_question(claims, pending):
    response = ChatResponse(user_id='synthetic', agent_used='KNOWLEDGE_AGENT',
        response=claims + '\n\nPartes do pedido sem resposta verificada:\n- ' + pending,
        abstained=True, answer_status='abstained')
    result = GroundednessJudge.score_response(
        {'message': 'Qual é a taxa do contrato?', 'expect_abstained': True}, response)
    assert not result['passed'] and result['groundedness_score'] == 0


def test_evaluation_runs_do_not_reuse_persistent_conversation_state(monkeypatch, tmp_path):
    import json
    import app.evals.evaluator as module
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps({'cases': [{'case_id': 'synthetic', 'category': 'boundary',
        'user_id': 'synthetic', 'message': 'Pedido sem dados.', 'expected_agent': 'KNOWLEDGE_AGENT',
        'expected_tools': [], 'expect_abstained': True}]}))
    sessions = []
    def process(request):
        sessions.append(request.session_id)
        return ChatResponse(user_id=request.user_id, agent_used='KNOWLEDGE_AGENT',
                            response='Não encontrei evidência suficiente.', abstained=True)
    monkeypatch.setattr(module.orchestrator, 'process', process)
    runner = module.SystemEvaluator(cases)
    first, second = runner.run_suite(), runner.run_suite()
    assert len(set(sessions)) == 2
    assert first['evaluation_run_id'] != second['evaluation_run_id']
