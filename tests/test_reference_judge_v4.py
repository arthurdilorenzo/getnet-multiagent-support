"""The local checker cannot certify unrestricted paraphrases as entailment."""
import pytest
from app.evals.evaluator import GroundednessJudge
from app.schemas import ChatResponse


REFERENCE = 'O limite efetivo e os juros variam conforme o emissor do cartão.'


@pytest.mark.parametrize('claim', [
    'O limite efetivo e os juros são definidos exclusivamente pelo lojista, que pode ignorar o emissor do cartão.',
    'O emissor do cartão obedece ao limite efetivo e aos juros impostos pelo lojista.',
    'The merchant alone sets the credit limit and can disregard the issuing bank.',
])
def test_authority_inversion_is_not_approved_by_matching_words(claim):
    out = ChatResponse(response=claim, user_id='synthetic', agent_used='KNOWLEDGE_AGENT')
    result = GroundednessJudge.score_response({'reference_facts': [REFERENCE]}, out)
    assert not result['passed']


def test_intact_independent_reference_can_pass():
    out = ChatResponse(response=REFERENCE, user_id='synthetic', agent_used='KNOWLEDGE_AGENT')
    assert GroundednessJudge.score_response({'reference_facts': [REFERENCE]}, out)['passed']


def test_unverified_but_plausible_paraphrase_requires_review():
    out = ChatResponse(response='As condições dependem da instituição emissora.', user_id='synthetic', agent_used='KNOWLEDGE_AGENT')
    result = GroundednessJudge.score_response({'reference_facts': [REFERENCE]}, out)
    assert not result['passed']
    assert result['semantic_review_required']
