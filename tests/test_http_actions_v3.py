import asyncio
import importlib
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.memory.store import StateStore
from app.schemas import ChatResponse
from app.security.auth import AuthManager
from app.security import action_policy

main = importlib.import_module("app.main")
auth = importlib.import_module("app.security.auth")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = Settings(_env_file=None, GEMINI_API_KEY="", JWT_SECRET="", AUTH_API_KEY="")
    store = StateStore(tmp_path / "http.db")
    monkeypatch.setattr(auth, "settings", config)
    monkeypatch.setattr(main, "state_store", store)
    monkeypatch.setattr(action_policy, "state_store", store)
    return config, store, TestClient(main.app)


def proposal():
    return action_policy.propose_action("cliente1988", "s1", "GET-SMART-88219", "remote_reboot")


def test_anonymous_demo_cannot_confirm(runtime):
    _, _, client = runtime
    item = proposal()
    response = client.post(f"/api/actions/{item['proposal_id']}/confirm", json={
        "user_id": "cliente1988", "expected_payload_hash": item["payload_hash"], "confirmation": "CONFIRM"})
    assert response.status_code == 401


def test_confirm_http_binds_owner_scope_hash_and_replay(runtime):
    _, store, client = runtime
    item = proposal()
    url = f"/api/actions/{item['proposal_id']}/confirm"
    body = {"user_id": "cliente1988", "expected_payload_hash": item["payload_hash"], "confirmation": "CONFIRM"}
    limited = AuthManager.create_jwt_token("cliente1988", scopes=["chat:write"])
    assert client.post(url, json=body, headers={"Authorization": f"Bearer {limited}"}).status_code == 403
    other = AuthManager.create_jwt_token("cliente2045")
    assert client.post(url, json=body, headers={"Authorization": f"Bearer {other}"}).status_code == 403
    token = AuthManager.create_jwt_token("cliente1988")
    headers = {"Authorization": f"Bearer {token}"}
    wrong = {**body, "expected_payload_hash": "0" * 64}
    assert client.post(url, json=wrong, headers=headers).status_code == 409
    first = client.post(url, json=body, headers=headers)
    assert first.status_code == 200
    second = client.post(url, json=body, headers=headers)
    assert second.status_code == 200
    assert second.json()["action_id"] == first.json()["action_id"]
    assert second.json()["idempotent_replay"] is True
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM terminal_actions").fetchone()[0] == 1


def test_chat_requires_chat_scope(runtime, monkeypatch):
    _, _, client = runtime
    token = AuthManager.create_jwt_token("cliente1988", scopes=["customer:read_own"])
    monkeypatch.setattr(main.orchestrator, "process", lambda **kw: pytest.fail("unauthorized execution"))
    response = client.post("/api/chat", json={"user_id": "cliente1988", "message": "perfil"},
                           headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_slow_synchronous_chat_does_not_block_health(runtime, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow(**kwargs):
        started.set()
        release.wait(1)
        return ChatResponse(response="done", user_id="cliente1988", agent_used="support")
    monkeypatch.setattr(main.orchestrator, "process", slow)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        pending = asyncio.create_task(client.post("/api/chat", json={"user_id": "cliente1988", "message": "perfil"}))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(.005)
        health = await client.get("/api/health")
        try:
            assert health.status_code == 200
            assert not pending.done(), "blocking chat occupied the event loop"
        finally:
            release.set()
            await pending
