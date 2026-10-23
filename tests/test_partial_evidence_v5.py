from app.evals.evaluator import GroundednessJudge
from app.schemas import ChatResponse


MESSAGE = 'Como recupero a senha? Isso mantém minhas sessões ativas?'
FACT = 'Para recuperar a senha, selecione Esqueci a senha no portal.'
GAP = 'Isso mantém minhas sessões ativas?'


def partial(claim=FACT, gap=GAP):
    return ChatResponse(response=claim+'\n\nPartes do pedido sem resposta verificada:\n- '+gap,
                        user_id='synthetic', agent_used='KNOWLEDGE_AGENT',
                        semantic_mode='native_semantic', answer_status='partial', abstained=True,
                        evidence=[{'kind':'source_excerpt','excerpt':FACT}])


def test_supported_partial_is_not_hallucination_but_is_incomplete():
    result = GroundednessJudge.score_response({'message':MESSAGE, 'expect_abstained':False}, partial())
    assert result['groundedness_score'] == 1
    assert not result['passed']
    assert result['fulfillment_passed'] is False
    assert result['partial_evidence_consistent'] is True


def test_explicitly_expected_partial_can_pass_limited_consistency():
    result = GroundednessJudge.score_response({'message':MESSAGE, 'expected_answer_status':'partial'}, partial())
    assert result['passed']
    assert result['fulfillment_passed']


def test_partial_does_not_excuse_an_unsupported_material_claim():
    result = GroundednessJudge.score_response({'message':MESSAGE}, partial(FACT+' A taxa é 0,75%.'))
    assert result['groundedness_score'] == 0
    assert not result['passed']


def test_fabricated_unresolved_text_cannot_hide_a_claim():
    result = GroundednessJudge.score_response({'message':MESSAGE}, partial(gap='O serviço custa zero reais.'))
    assert not result['passed']
    assert result['groundedness_score'] == 0


def test_partial_must_show_its_unanswered_request_instead_of_only_setting_a_flag():
    out=partial()
    out.response=FACT
    assert not GroundednessJudge.score_response({'message':MESSAGE},out)['passed']


def test_search_snippets_are_not_promoted_to_complete_answers(monkeypatch):
    import importlib
    from app.agents.knowledge_agent import KnowledgeAgent
    from app.config import settings
    from app.schemas import ToolCallSpec, ToolExecutionResult
    from app.security.auth import SecurityContext
    module=importlib.import_module('app.agents.knowledge_agent')
    monkeypatch.setattr(settings,'GEMINI_API_KEY','')
    message='Quem responde atualmente pela empresa Exemplo?'
    monkeypatch.setattr(KnowledgeAgent,'_plan_tool_calls',lambda *args,**kwargs:[
        ToolCallSpec(tool_name='web_search_general',arguments={'query':message})])
    monkeypatch.setattr(module.tool_registry,'execute',lambda **kwargs:ToolExecutionResult(
        tool_name='web_search_general',success=True,data={'query':message,'results':[
            {'title':'Empresa Exemplo','snippet':'A empresa Exemplo desenvolve equipamentos industriais.',
             'url':'https://example.org/company'}]}))
    out=KnowledgeAgent().run(message,SecurityContext(authenticated_user_id='synthetic'),True)
    assert out.answer_status=='partial'
    assert out.abstained
    assert 'equipamentos industriais' in out.response
    assert message in out.response
