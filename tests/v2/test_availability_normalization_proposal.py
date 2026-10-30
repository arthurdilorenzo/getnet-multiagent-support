"""Proposed post-batch repair: model status cannot override observed availability."""
import pytest

from app.v2.contracts import (
    AnswerDocument, Binding, DocumentaryBlock, EvidenceSpan, Fact, OperationalBlock,
    Requirement, RequirementDecision, answer_status,
)
from app.v2.knowledge import EvidenceError, KnowledgeService, validate_answer
from app.observability.tracer import SystemTracer
from tests.v2.test_runtime import ScriptedModel


def mixed_availability_case(status='supported'):
    import hashlib
    text = 'Desligue a máquina antes de instalar o chip e depois execute o teste de comunicação.'
    digest = hashlib.sha256(text.encode()).hexdigest()
    span = EvidenceSpan(span_id='doc:' + digest, text=text, source='manual', content_hash=digest)
    bindings = [Binding(binding_id='r2:b' + str(i), tool_name='diagnose_terminal_issue',
        resource_type='terminal', resource_id='POS-9941', field_pointer=pointer)
        for i, pointer in enumerate(['/chip_algar_status', '/ultimo_erro_transacao'])]
    reqs = [Requirement(requirement_id='r1', kind='documentary', summary='Instalação segura'),
        Requirement(requirement_id='r2', kind='operational', summary='Linha e último erro', bindings=bindings)]
    facts = [Fact(fact_id='active', binding_id=bindings[0].binding_id, operation_id='op',
        resource_type='terminal', resource_id='POS-9941', field_pointer=bindings[0].field_pointer,
        availability='available', value='ativo_em_rede_multioperadora'),
        Fact(fact_id='unknown', binding_id=bindings[1].binding_id, operation_id='op',
        resource_type='terminal', resource_id='POS-9941', field_pointer=bindings[1].field_pointer,
        availability='missing', value=None)]
    doc = AnswerDocument(requirements=[RequirementDecision(requirement_id='r1', status='supported'),
        RequirementDecision(requirement_id='r2', status=status)], final_answer=[
        DocumentaryBlock(requirement_id='r1', text=text, span_ids=[span.span_id]),
        OperationalBlock(requirement_id='r2', fact_ids=['active', 'unknown'])])
    return doc, [span], facts, reqs


@pytest.mark.parametrize('status', ['supported', 'contradicted'])
def test_compose_downgrades_unavailable_status_and_keeps_known_answer(status):
    doc, spans, facts, reqs = mixed_availability_case(status)
    # Direct validation stays strict, preserving existing negative contracts.
    with pytest.raises(EvidenceError, match='UNSUPPORTED_OPERATIONAL_BINDING'):
        validate_answer(doc, spans, facts, reqs)
    service = KnowledgeService(ScriptedModel(doc))
    normalized, text = service.compose('Instale e consulte linha e erro.', spans, facts, {},
        SystemTracer(), planned_requirements=reqs)
    assert answer_status(normalized.requirements) == 'partial'
    assert normalized.requirements[1].status == 'unanswerable'
    assert 'Desligue a máquina' in text
    assert 'ativo_em_rede_multioperadora' in text
    assert 'ultimo_erro_transacao (POS-9941): Informação indisponível.' in text
    assert facts[1].value is None
    assert doc.requirements[1].status == status  # retained provider output


def test_compose_never_turns_resource_mismatch_into_partial_success():
    doc, spans, facts, reqs = mixed_availability_case()
    facts[0].resource_id = 'POS-FOREIGN'
    with pytest.raises(EvidenceError, match='OPERATIONAL_RESOURCE_MISMATCH'):
        KnowledgeService(ScriptedModel(doc)).compose('Consulte.', spans, facts, {},
            SystemTracer(), planned_requirements=reqs)
