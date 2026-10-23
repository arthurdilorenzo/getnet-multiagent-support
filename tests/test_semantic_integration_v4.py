import importlib

import pytest

from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.router_agent import RouterAgent
from app.config import settings
from app.memory.store import StateStore
from app.schemas import ChatRequest, PlanStep, RoutingPlan, SpecialistOutput


@pytest.mark.parametrize('message', [
    'Inspecione a conectividade de GET-POS-LAB-0042.',
    'Check GET-DEVICE-TEST-0042; it has lost connectivity.',
    'Por que a operação TXN-LAB-4301 foi negada?',
    'Please investigate why TXN-LAB-4301 was declined.',
])
def test_offline_router_recognizes_structured_operational_identifiers(message, monkeypatch):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    assert RouterAgent().plan(message, 'cliente1988').primary_agent == 'CUSTOMER_SUPPORT_AGENT'


@pytest.mark.parametrize('message', [
    'Chove amanhã no Recife?',
    'For Manaus tomorrow, forecast the rain probability.',
])
def test_offline_router_recognizes_weather_need_without_city_allowlist(message, monkeypatch):
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    assert RouterAgent().plan(message, 'cliente1988').primary_agent == 'WEB_SEARCH_AGENT'


@pytest.mark.parametrize('agent', ['KNOWLEDGE_AGENT', 'WEB_SEARCH_AGENT'])
def test_delegation_keeps_original_request_separate_from_model_subquery(agent, monkeypatch, tmp_path):
    module = importlib.import_module('app.agents.orchestrator')
    monkeypatch.setattr(module, 'state_store', StateStore(tmp_path / 'scope.db'))
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    orchestrator = MultiAgentOrchestrator()
    original = 'Minha empresa é estrangeira; o produto Getnet está disponível para ela?'
    plan = RoutingPlan(primary_agent=agent, confidence=.9, reasoning='teste', plan_steps=[
        PlanStep(step_order=1, target_agent=agent, sub_query='Quais são as características?', rationale='teste')])
    monkeypatch.setattr(orchestrator.router, 'plan', lambda **kwargs: plan)
    captured = {}
    def run(**kwargs):
        captured.update(kwargs)
        return SpecialistOutput(agent_name=agent, response='Não há evidência.', abstained=True)
    monkeypatch.setattr(orchestrator.knowledge_agent, 'run', run)
    orchestrator.process(ChatRequest(message=original, user_id='cliente1988'))
    assert captured['request_message'] == original
    assert 'Minha empresa é estrangeira' in captured['message']
    assert 'produto Getnet está disponível para ela?' in captured['message']
