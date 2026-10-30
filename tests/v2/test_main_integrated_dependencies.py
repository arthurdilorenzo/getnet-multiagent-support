"""Public HTTP boundaries use the injected runtime's state and authorization."""
import importlib
import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager
from app.security.auth import AuthManager, SecurityContext
from app.v2.adapters import StateAdapter, ToolGateway

main = importlib.import_module('app.main')


def test_exchange_ratio_keeps_provider_precision():
    from app.v2.contracts import Fact
    from app.v2.knowledge import format_fact
    fact = Fact(fact_id='fx',operation_id='op',resource_type='currency',resource_id='EUR/BRL',
                field_pointer='/rate',availability='available',value='6.12345',unit='BRL')
    assert '6.12345' in format_fact(fact)


@pytest.fixture
def integrated(tmp_path, monkeypatch):
    store = StateStore(tmp_path/'integrated.db')
    queue = HITLQueueManager(store)
    path = tmp_path/'customers.json'
    path.write_text(json.dumps({'customers': {'http-owner': {
        'user_id': 'http-owner', 'terminals': [{'terminal_id': 'POS-HTTP', 'status': 'online'}]}}}))
    gateway = ToolGateway(path, store=store, queue=queue)
    runtime = SimpleNamespace(state=StateAdapter(store), gateway=gateway, queue=queue)
    monkeypatch.setattr(main, '_integrated_runtime', runtime)
    store.ensure_session('http-session', 'http-owner')
    return runtime, TestClient(main.app)


def headers(user, scopes=None, role='merchant'):
    return {'Authorization': 'Bearer '+AuthManager.create_jwt_token(user, role=role, scopes=scopes)}


def test_public_confirmation_uses_injected_store_owner_hash_and_replay(integrated):
    runtime, client = integrated
    context = SecurityContext('http-owner', scopes=['chat:write','customer:read_own','terminal:action_own'])
    result = runtime.gateway.execute('execute_terminal_remote_action', {
        'terminal_id':'POS-HTTP', 'action_type':'remote_reboot', 'reason':'HTTP contract',
        'idempotency_key':'http-proposal'}, context, 'http-session')
    assert result.success, result.error_code
    proposal = result.data
    url = '/api/actions/'+proposal['proposal_id']+'/confirm'
    body = {'user_id':'http-owner','expected_payload_hash':proposal['payload_hash'],'confirmation':'CONFIRM'}
    with runtime.state.store._connect() as conn:
        assert conn.execute('SELECT count(*) FROM terminal_actions').fetchone()[0] == 0
    assert client.post(url,json=body,headers=headers('http-other')).status_code == 403
    assert client.post(url,json=body,headers=headers('http-owner',['chat:write'])).status_code == 403
    wrong = {**body,'expected_payload_hash':'0'*64}
    assert client.post(url,json=wrong,headers=headers('http-owner')).status_code == 409
    first = client.post(url,json=body,headers=headers('http-owner'))
    assert first.status_code == 200, first.text
    replay = client.post(url,json=body,headers=headers('http-owner'))
    assert replay.status_code == 200
    assert replay.json()['action_id'] == first.json()['action_id']
    assert replay.json()['idempotent_replay'] is True
    with runtime.state.store._connect() as conn:
        assert conn.execute('SELECT count(*) FROM terminal_actions').fetchone()[0] == 1


def test_public_session_and_hitl_use_injected_state(integrated):
    runtime, client = integrated
    runtime.state.append('http-session','http-owner','user','persisted in injected store')
    history = client.get('/api/sessions/http-session',params={'user_id':'http-owner'},headers=headers('http-owner'))
    assert history.status_code == 200
    assert 'persisted in injected store' in history.text
    ticket = runtime.queue.create_ticket(user_id='http-owner',session_id='http-session',category='support',reason='HTTP test')
    op = headers('http-operator',role='operator')
    assert client.get('/api/hitl/tickets/'+ticket.ticket_id,headers=op).status_code == 200
    claimed = client.post('/api/hitl/tickets/'+ticket.ticket_id+'/claim',headers=op)
    assert claimed.status_code == 200, claimed.text


def test_web_result_preserves_target_url_instead_of_search_page():
    from app.tools.search_tools import _DDGResultParser
    parser = _DDGResultParser()
    parser.feed('<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsite.getnet.com.br%2F">Getnet</a><a class="result__snippet" href="https://site.getnet.com.br/">Soluções</a>')
    assert parser.snippets == ['Soluções']
    assert parser.urls == ['https://site.getnet.com.br/']
