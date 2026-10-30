"""Cross-source documentary synthesis references remain server-bound."""
import pytest
from app.v2.contracts import AnswerDocument, DocumentaryBlock, Requirement, RequirementDecision, canonicalize_answer_block
from app.v2.knowledge import EvidenceError, validate_answer
from tests.v2.test_availability_normalization_proposal import mixed_availability_case


def supported_synthesis():
    doc, spans, facts, reqs = mixed_availability_case()
    # Use the available chip fact only, retaining an independently operational
    # requirement and rendered value separate from recommendation prose.
    reqs[1].bindings = reqs[1].bindings[:1]
    reqs[1].bindings[0].tool_name = 'get_customer_profile'
    reqs[1].bindings[0].resource_type = 'customer'
    reqs[1].bindings[0].resource_id = 'merchant'
    reqs[1].bindings[0].field_pointer = '/profile/necessidade'
    facts = facts[:1]
    facts[0].resource_type = 'customer'
    facts[0].resource_id = 'merchant'
    facts[0].field_pointer = '/profile/necessidade'
    doc.final_answer[1].fact_ids = ['active']
    reqs.append(Requirement(requirement_id='r3', kind='documentary', summary='Recomendação'))
    doc.requirements.append(RequirementDecision(requirement_id='r3', status='supported'))
    raw = doc.model_dump()
    raw['final_answer'].append({'kind': 'documentary', 'requirement_id': 'r3',
        'text': 'Recomendo seguir o procedimento documentado para o chip cadastrado.',
        'span_ids': [spans[0].span_id], 'fact_ids': ['active']})
    return AnswerDocument.model_validate(raw), spans, facts, reqs


def test_documentary_support_fact_refs_preserved_and_validated():
    doc, spans, facts, reqs = supported_synthesis()
    assert doc.final_answer[-1].fact_ids == ['active']
    validate_answer(doc, spans, facts, reqs)


@pytest.mark.parametrize('defect', ['unknown', 'unavailable', 'resource', 'unit', 'pointer', 'unrendered'])
def test_documentary_fact_support_does_not_bypass_boundaries(defect):
    doc, spans, facts, reqs = supported_synthesis()
    if defect == 'unknown':
        doc.final_answer[-1].fact_ids = ['invented']
    elif defect == 'unavailable':
        facts[0].availability = 'unavailable'
    elif defect == 'resource':
        facts[0].resource_id = 'POS-FOREIGN'
    elif defect == 'unit':
        facts[0].unit = 'BRL'
    elif defect == 'pointer':
        facts[0].field_pointer = '/secret'
    else:
        doc.final_answer = [block for block in doc.final_answer if block.kind != 'operational']
    with pytest.raises(EvidenceError):
        validate_answer(doc, spans, facts, reqs)


@pytest.mark.parametrize('refs', [[''], [3], 'active'])
def test_documentary_support_rejects_invalid_ref_types(refs):
    with pytest.raises(ValueError):
        canonicalize_answer_block({'kind': 'documentary', 'requirement_id': 'r',
            'text': 'Procedimento.', 'span_ids': ['doc'], 'fact_ids': refs})


@pytest.mark.parametrize('kind', ['numeric', 'financial', 'terminal', 'action', 'public'])
def test_documentary_support_is_limited_to_qualitative_profile_text(kind):
    doc, spans, facts, reqs = supported_synthesis()
    binding = reqs[1].bindings[0]
    if kind == 'numeric':
        facts[0].value, facts[0].value_type = 500, 'int'
    elif kind == 'financial':
        binding.tool_name = 'get_sales_settlement'
        binding.field_pointer = facts[0].field_pointer = '/sales_data/amount'
    elif kind == 'terminal':
        binding.tool_name = 'diagnose_terminal_issue'
        binding.resource_type = facts[0].resource_type = 'terminal'
    elif kind == 'action':
        binding.tool_name = 'execute_terminal_remote_action'
    else:
        binding.tool_name = 'get_weather_forecast'
    with pytest.raises(EvidenceError, match='INVALID_DOCUMENTARY_FACT_KIND'):
        validate_answer(doc, spans, facts, reqs)
