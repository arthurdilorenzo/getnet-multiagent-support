"""Final HTTP boundary contract; synthetic specialists deliberately return excess data."""
import importlib

from fastapi.testclient import TestClient
from app.agents.orchestrator import MultiAgentOrchestrator
from app.config import settings
from app.memory.store import StateStore
from app.schemas import Citation, PlanStep, RoutingPlan, SpecialistOutput, ToolExecutionResult
from app.security.auth import AuthManager
from app.security.request_units import request_units


def test_final_projection_covers_full_http_envelope_and_persisted_assistant(monkeypatch, tmp_path):
    main = importlib.import_module('app.main')
    orchestration = importlib.import_module('app.agents.orchestrator')
    store = StateStore(tmp_path / 'boundary.sqlite')
    monkeypatch.setattr(orchestration, 'state_store', store)
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(settings, 'AUTH_ENFORCE_STRICT', True)
    user = 'boundary_synthetic'
    first, second = 'GET-SMART-QA-44031', 'GET-CLASSICA-QA-44032'
    message = (f'Informe somente o sinal de {first}; informe a conexão de {second}. '
               'Identifique cada máquina em uma linha curta.')
    units = request_units(message)
    targets = (first, second)
    orch = MultiAgentOrchestrator()
    plan = RoutingPlan(primary_agent='CUSTOMER_SUPPORT_AGENT', confidence=.9,
        reasoning='SENTINEL_INTENT_FIRMWARE', plan_steps=[PlanStep(step_order=i+1,
            target_agent='CUSTOMER_SUPPORT_AGENT', sub_query=target,
            rationale='SENTINEL_RATIONALE_STATUS',
            request_unit_ids=[next(u['unit_id'] for u in units if target in u['text'])])
            for i, target in enumerate(targets)])
    monkeypatch.setattr(orch.router, 'plan', lambda **kw: plan)
    monkeypatch.setattr(orch, '_observe_and_replan', lambda **kw: False)

    def specialist(**kw):
        ids = kw['delegation_scope']['assigned_unit_ids']
        target = next(t for t in targets if t in kw['message'])
        kw['tracer'].add_step(agent='SYNTHETIC_SPECIALIST', action='rich_internal_observation',
            details={'terminal': {'firmware_version': 'SENTINEL_TRACE_FIRMWARE'},
                     'arbitrary_nested': ['SENTINEL_TRACE_OTHER']})
        result = ToolExecutionResult(tool_name='diagnose_terminal_issue', success=True, data={
            'user_id': user, 'found': True, 'requested_terminal_id': target,
            'terminal': {'terminal_id': target, 'signal_strength': '1/4 bars' if target == first else '3/4 bars',
                'connection_type': 'Wi-Fi' if target == first else '4G Algar',
                'firmware_version': 'SENTINEL_TOOL_FIRMWARE', 'status': 'SENTINEL_TOOL_STATUS'},
            'recommended_troubleshooting': {'connectivity': ['SENTINEL_INSTRUCTION']}})
        return SpecialistOutput(agent_name='CUSTOMER_SUPPORT_AGENT', response='SENTINEL_PROSE',
            request_unit_ids=ids, covered_unit_ids=ids, answer_status='complete',
            tools_called=['diagnose_terminal_issue'], tool_results=[result],
            structured_data={'diagnose_terminal_issue': result.data},
            evidence=[{'tool_name': 'diagnose_terminal_issue', 'field': 'terminal.firmware_version',
                       'value': 'SENTINEL_EVIDENCE'}], sources=['SENTINEL_SOURCE'],
            citations=[Citation(chunk_id='SENTINEL_CITATION', title='SENTINEL_TITLE', source='SENTINEL_CITE_SOURCE',
                source_url='https://example.invalid/SENTINEL_URL', relevance_score=1)])

    monkeypatch.setattr(orch.customer_support_agent, 'run', specialist)
    monkeypatch.setattr(main, 'orchestrator', orch)
    token = AuthManager.create_jwt_token(user, role='merchant')
    with TestClient(main.app) as client:
        result = client.post('/api/chat', json={'message': message, 'user_id': user},
                             headers={'Authorization': f'Bearer {token}'})
    assert result.status_code == 200
    payload = result.json()
    assert 'SENTINEL_' not in result.text
    assert '1/4 bars' in payload['response'] and '4G Algar' in payload['response']
    assert '3/4 bars' not in result.text and 'Wi-Fi' not in result.text
    assert first in payload['response'] and second in payload['response']
    public_assistant = [m['content'] for m in store.get_conversation_context(payload['session_id'], user)['recent_messages']
                        if m['role'] == 'assistant']
    assert public_assistant == [payload['response']]
    assert all('SENTINEL_' not in text for text in public_assistant)
