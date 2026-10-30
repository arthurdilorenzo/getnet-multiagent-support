"""Finite offline runtime closures for frozen A3/A4 contracts."""
import hashlib
import json

import pytest

from app.hitl.queue import HITLQueueManager
from app.memory.store import StateStore
from app.security.auth import SecurityContext
from app.v2.adapters import StateAdapter, ToolGateway
from app.v2.contracts import (AnswerDocument, DocumentaryBlock, HumanBlock,
    PlanProposal, RequirementDecision, RequirementProposal, ResumePlan, StepProposal)
from app.v2.knowledge import KnowledgeService
from app.v2.runtime import Runtime


def environment(tmp_path, model, retrieve=None):
    store = StateStore(tmp_path / 'state.db')
    state, queue = StateAdapter(store), HITLQueueManager(store)
    customers = tmp_path / 'customers.json'
    customers.write_text(json.dumps({'customers': {'merchant': {'segment': 'B2B'}}}))
    gateway = ToolGateway(customers, store=store, queue=queue)
    return Runtime(model, KnowledgeService(model, retrieve=retrieve), state, gateway, queue)


@pytest.mark.parametrize('indices', [[], [-1], [1], [0, 0]], ids=['empty', 'negative', 'out_of_range', 'duplicate'])
def test_invalid_resume_indices_abstain_before_tool(tmp_path, indices):
    class Model:
        def __init__(self): self.calls = []
        def generate(self, schema, payload, **kwargs):
            self.calls.append(schema.__name__)
            if schema is PlanProposal:
                return PlanProposal(requirements=[RequirementProposal(kind='human', summary='Validar')],
                    steps=[StepProposal(kind='human', requirement_indexes=[0])])
            if schema is ResumePlan:
                return ResumePlan(steps=[StepProposal(kind='tool', tool_name='get_customer_profile', requirement_indexes=indices)])
            return AnswerDocument(requirements=[RequirementDecision(requirement_id='r0001', status='supported')],
                final_answer=[HumanBlock(requirement_id='r0001', observation_id=next(s['span_id'] for s in payload['evidence_spans'] if s['kind'] == 'human'))])
    model = Model()
    runtime = environment(tmp_path, model)
    first = runtime.process('Validar informação.', SecurityContext('merchant', scopes=['chat:write', 'customer:read_own']), 'session')
    runtime.queue.claim_ticket(first.ticket_id, 'operator')
    runtime.queue.reply_to_ticket(first.ticket_id, 'operator', 'Verificado.', resolve_ticket=True)
    response = runtime.resume(first.ticket_id, SecurityContext('operator', role='operator', scopes=['hitl:manage']))
    assert response.answer_status == 'abstained'
    assert model.calls == ['PlanProposal', 'ResumePlan']
    with runtime.state.store._connect() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='tool_operations'").fetchone():
            assert conn.execute('SELECT count(*) FROM tool_operations').fetchone()[0] == 0
    calls = list(model.calls)
    assert runtime.resume(first.ticket_id, SecurityContext('operator', role='operator', scopes=['hitl:manage'])).model_dump() == response.model_dump()
    assert model.calls == calls


def test_documentary_references_do_not_invent_relevance(tmp_path):
    class Model:
        def generate(self, schema, payload, **kwargs):
            if schema is PlanProposal:
                return PlanProposal(requirements=[RequirementProposal(kind='documentary', summary='Condições')],
                    steps=[StepProposal(kind='knowledge', requirement_indexes=[0])])
            return AnswerDocument(requirements=[RequirementDecision(requirement_id='r0001', status='supported')],
                final_answer=[DocumentaryBlock(requirement_id='r0001', text='Condição documentada.', span_ids=[payload['evidence_spans'][0]['span_id']])])
    runtime = environment(tmp_path, Model(), lambda query: {'candidate_chunks': [{'content': 'Condição documentada.', 'source': 'manual'}]})
    response = runtime.process('Quais as condições?', SecurityContext('merchant', scopes=['chat:write']), 'session')
    assert response.answer_status == 'complete'
    assert response.citations == []
    assert any(e.get('kind') == 'document' and e.get('content_hash') for e in response.evidence)


@pytest.mark.parametrize('indices', [[], [-1], [1], [0, 0]], ids=['empty', 'negative', 'out_of_range', 'duplicate'])
def test_invalid_initial_indices_abstain_before_tool(tmp_path, indices):
    class Model:
        def __init__(self): self.calls = []
        def generate(self, schema, payload, **kwargs):
            self.calls.append(schema.__name__)
            if schema is PlanProposal:
                return PlanProposal(requirements=[RequirementProposal(kind='human', summary='Verificar')],
                    steps=[StepProposal(kind='tool', tool_name='get_customer_profile', requirement_indexes=indices)])
            raise AssertionError('Invalid initial steps must not reach composition')
    model = Model()
    runtime = environment(tmp_path, model)
    response = runtime.process('Verificar informação.', SecurityContext('merchant', scopes=['chat:write', 'customer:read_own']), 'session')
    assert response.answer_status == 'abstained'
    assert model.calls == ['PlanProposal']
    assert response.tools_called == []


def test_ticket_tool_guidance_is_attributed_human_observation(tmp_path):
    class Model:
        ticket_id = None
        def generate(self, schema, payload, **kwargs):
            if schema is PlanProposal:
                return PlanProposal(requirements=[RequirementProposal(kind='human', summary='Orientação registrada')],
                    steps=[StepProposal(kind='tool', tool_name='check_escalation_ticket', arguments={'ticket_id': self.ticket_id}, requirement_indexes=[0])])
            human = next(s for s in payload['evidence_spans'] if s['kind'] == 'human')
            assert human['source'] == 'hitl:' + self.ticket_id + ':specialist'
            return AnswerDocument(requirements=[RequirementDecision(requirement_id='r0001', status='supported')],
                final_answer=[HumanBlock(requirement_id='r0001', observation_id=human['span_id'])])
    model = Model()
    runtime = environment(tmp_path, model)
    runtime.state.store.ensure_session('source_session', 'merchant')
    ticket = runtime.queue.create_ticket('merchant', 'source_session', 'support', 'Verificar')
    runtime.queue.claim_ticket(ticket.ticket_id, 'specialist')
    runtime.queue.reply_to_ticket(ticket.ticket_id, 'specialist', 'Orientação registrada pelo especialista.', resolve_ticket=True)
    model.ticket_id = ticket.ticket_id
    response = runtime.process('Consultar a orientação.', SecurityContext('merchant', scopes=['chat:write', 'customer:read_own']), 'query_session')
    assert response.answer_status == 'complete'
    assert 'specialist' in response.response and ticket.ticket_id in response.response
    assert 'Orientação registrada pelo especialista.' in response.response
    human = next(e for e in response.evidence if e.get('kind') == 'human')
    assert human['content_hash'] == hashlib.sha256(human['text'].encode()).hexdigest()
    assert not any(e.get('field_pointer') == '/ticket/operator_response' for e in response.evidence)
