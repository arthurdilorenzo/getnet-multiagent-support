"""Regression tests for authentication, administrative access and persisted session owners."""

import base64
import hashlib
import hmac
import importlib
import json
import sqlite3
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.hitl.queue import HITLQueueManager
from app.memory.store import StateStore
from app.security.auth import AuthManager, CrossTenantAccessError

auth_module = importlib.import_module("app.security.auth")
main_module = importlib.import_module("app.main")


@pytest.fixture
def local_security(tmp_path, monkeypatch):
    config = Settings(_env_file=None, APP_ENV="development", JWT_SECRET="", AUTH_API_KEY="")
    monkeypatch.setattr(auth_module, "settings", config)
    store = StateStore(tmp_path / "boundaries.db")
    queue = HITLQueueManager(store)
    monkeypatch.setattr(main_module, "state_store", store)
    monkeypatch.setattr(main_module, "hitl_queue", queue)
    return config, store, queue, TestClient(main_module.app)


def _signed_token(config, *, header=None, claims=None):
    now = int(time.time())
    payload = {
        "sub": "merchant-a", "role": "merchant", "scopes": ["chat:write"],
        "iss": "getnet-multiagent-auth", "iat": now - 1, "exp": now + 600,
    }
    if claims:
        payload.update(claims)

    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    signing = f"{encode(header or {'alg': 'HS256', 'typ': 'JWT'})}.{encode(payload)}"
    signature = hmac.new(config.JWT_SECRET.encode(), signing.encode(), hashlib.sha256).digest()
    return f"{signing}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


@pytest.mark.parametrize("path", ["/api/hitl/tickets", "/api/hitl/tickets/missing", "/api/metrics", "/api/evals/run"])
def test_administrative_routes_reject_anonymous_requests(local_security, path):
    _, _, _, client = local_security
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", ["/api/hitl/tickets", "/api/hitl/tickets/missing", "/api/metrics", "/api/evals/run"])
def test_administrative_routes_reject_authenticated_merchants(local_security, path):
    _, _, _, client = local_security
    token = AuthManager.create_jwt_token("merchant-a")
    assert client.get(path, headers={"Authorization": f"Bearer {token}"}).status_code == 403


def test_operator_can_read_ticket_list_and_detail_with_filters(local_security):
    _, _, queue, client = local_security
    own = queue.create_ticket("merchant-a", "arbitrary-session", "support", "Need specialist")
    queue.create_ticket("merchant-b", "another-session", "support", "Need specialist")
    token = AuthManager.create_jwt_token("operator-alice", role="operator")
    headers = {"Authorization": f"Bearer {token}"}
    listing = client.get("/api/hitl/tickets?user_id=merchant-a&status_filter=OPEN", headers=headers)
    assert listing.status_code == 200
    assert [t["ticket_id"] for t in listing.json()["tickets"]] == [own.ticket_id]
    detail = client.get(f"/api/hitl/tickets/{own.ticket_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["user_id"] == "merchant-a"


def test_operator_reply_is_bound_to_authenticated_principal(local_security):
    _, store, queue, client = local_security
    ticket = queue.create_ticket("merchant-a", "reply-session", "support", "Need specialist")
    token = AuthManager.create_jwt_token("operator-alice", role="operator")
    response = client.post(
        f"/api/hitl/tickets/{ticket.ticket_id}/reply",
        headers={"Authorization": f"Bearer {token}"},
        json={"operator_id": "forged-bob", "reply_message": "Resolution registered."},
    )
    assert response.status_code == 200
    assert response.json()["assigned_operator"] == "operator-alice"
    messages = store.get_conversation_context("reply-session", "merchant-a")["recent_messages"]
    assert messages[-1]["agent_used"] == "HUMAN_OPERATOR:operator-alice"


def test_invalid_authorization_cannot_be_overridden_by_valid_api_key(local_security):
    config, _, _, client = local_security
    response = client.get("/api/hitl/tickets", headers={"Authorization": "Bearer broken", "X-API-Key": config.AUTH_API_KEY})
    assert response.status_code == 401


@pytest.mark.parametrize("secret", ["", "short", "getnet-hmac-jwt-signing-secret-2026", "a" * 64, "replace-with-a-strong-secret-value-2026"])
def test_production_rejects_missing_or_predictable_secrets(secret):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_ENV="production", JWT_SECRET=secret, AUTH_API_KEY=secret)


def test_production_enforces_strict_auth_even_if_flag_is_disabled():
    config = Settings(
        _env_file=None, APP_ENV="production", AUTH_ENFORCE_STRICT=False,
        JWT_SECRET="V5w7dzAl_F9Sx4h-M2iRQK26PO0JbN8eC3trYukG1Ha",
        AUTH_API_KEY="K6Ujot0gxScNf9lyRM5Wa38vDQLF2HEsZ7rX4bnIPAk",
    )
    assert config.AUTH_ENFORCE_STRICT is True


def test_development_secrets_are_ephemeral_and_demo_is_explicit():
    first = Settings(_env_file=None, APP_ENV="development", JWT_SECRET="", AUTH_API_KEY="")
    second = Settings(_env_file=None, APP_ENV="development", JWT_SECRET="", AUTH_API_KEY="")
    assert first.JWT_SECRET != second.JWT_SECRET
    assert first.AUTH_API_KEY != second.AUTH_API_KEY
    assert len(first.JWT_SECRET) >= 32 and len(first.AUTH_API_KEY) >= 32
    original = auth_module.settings
    try:
        auth_module.settings = first
        assert AuthManager.resolve_chat_security_context("merchant-a").auth_method == "demo_unverified"
    finally:
        auth_module.settings = original


def test_configured_jwt_disables_anonymous_downgrade(monkeypatch):
    config = Settings(_env_file=None, APP_ENV="development", JWT_SECRET="configured-signing-key-8JfrZ3uL9VPb6Cd1", AUTH_API_KEY="")
    monkeypatch.setattr(auth_module, "settings", config)
    with pytest.raises(HTTPException) as exc:
        AuthManager.resolve_chat_security_context("merchant-a")
    assert exc.value.status_code == 401


def test_using_jwt_disables_later_anonymous_downgrade(local_security):
    AuthManager.create_jwt_token("merchant-a")
    with pytest.raises(HTTPException) as exc:
        AuthManager.resolve_chat_security_context("merchant-a")
    assert exc.value.status_code == 401


@pytest.mark.parametrize("header", [{"alg": "none", "typ": "JWT"}, {"alg": "HS512", "typ": "JWT"}, {"alg": "HS256", "typ": "JOSE"}, {"alg": "HS256", "typ": "JWT", "crit": ["unknown"]}])
def test_jwt_rejects_unsupported_headers_even_with_valid_signature(local_security, header):
    config, _, _, _ = local_security
    with pytest.raises(HTTPException) as exc:
        AuthManager.verify_jwt_token(_signed_token(config, header=header))
    assert exc.value.status_code == 401


@pytest.mark.parametrize("claims", [
    {"iss": "untrusted"}, {"sub": ""}, {"sub": "   "}, {"sub": ["merchant-a"]},
    {"role": "admin"}, {"scopes": "admin:all"}, {"scopes": [7]}, {"scopes": ["admin:all"]},
    {"exp": "tomorrow"}, {"exp": None}, {"exp": True}, {"exp": 0},
    {"iat": "yesterday"}, {"iat": 9999999999}, {"nbf": 9999999999},
])
def test_jwt_rejects_invalid_claims_with_401(local_security, claims):
    config, _, _, _ = local_security
    with pytest.raises(HTTPException) as exc:
        AuthManager.verify_jwt_token(_signed_token(config, claims=claims))
    assert exc.value.status_code == 401


@pytest.mark.parametrize("token", ["é.a.b", "a..b", "a.b.###", "", "a.b.c.d"])
def test_malformed_jwt_never_raises_internal_server_error(local_security, token):
    with pytest.raises(HTTPException) as exc:
        AuthManager.verify_jwt_token(token)
    assert exc.value.status_code == 401


def test_operator_context_preserves_token_subject(local_security):
    token = AuthManager.create_jwt_token("operator-alice", role="operator")
    context = AuthManager.resolve_chat_security_context("operator-alice", authorization=f"Bearer {token}")
    assert context.authenticated_user_id == "operator-alice"


@pytest.mark.parametrize("operation", ["read", "append", "ensure"])
def test_store_rejects_session_owner_mismatch(local_security, operation):
    _, store, _, _ = local_security
    store.append_message("sess_merchant-b_misleading", "merchant-a", "user", "Confidential conversation")
    with pytest.raises(CrossTenantAccessError):
        if operation == "read":
            store.get_conversation_context("sess_merchant-b_misleading", "merchant-b")
        elif operation == "append":
            store.append_message("sess_merchant-b_misleading", "merchant-b", "user", "Injected message")
        else:
            store.ensure_session("sess_merchant-b_misleading", "merchant-b")
    assert len(store.get_conversation_context("sess_merchant-b_misleading", "merchant-a")["recent_messages"]) == 1


@pytest.mark.parametrize("method", ["get", "delete"])
def test_session_endpoints_use_persisted_owner_instead_of_prefix(local_security, method):
    _, store, _, client = local_security
    session = "sess_merchant-b_looks-owned"
    store.append_message(session, "merchant-a", "user", "Confidential conversation")
    token = AuthManager.create_jwt_token("merchant-b")
    response = getattr(client, method)(f"/api/sessions/{session}?user_id=merchant-b", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert len(store.get_conversation_context(session, "merchant-a")["recent_messages"]) == 1


def test_merchant_can_clear_arbitrary_session_it_actually_owns(local_security):
    _, store, _, client = local_security
    store.append_message("uuid-session-without-prefix", "merchant-a", "user", "Owned message")
    token = AuthManager.create_jwt_token("merchant-a")
    response = client.delete("/api/sessions/uuid-session-without-prefix?user_id=merchant-a", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert store.get_conversation_context("uuid-session-without-prefix", "merchant-a")["recent_messages"] == []


def test_hitl_cannot_create_ticket_using_another_tenants_session(local_security):
    _, store, queue, _ = local_security
    store.append_message("shared-session", "merchant-a", "user", "Confidential conversation")
    with pytest.raises(CrossTenantAccessError):
        queue.create_ticket("merchant-b", "shared-session", "support", "Attempted takeover")
    assert queue.list_tickets() == []


def test_hitl_reply_rejects_stale_ticket_pointing_to_foreign_session(local_security):
    _, store, queue, client = local_security
    ticket = queue.create_ticket("merchant-a", "stale-session", "support", "Need specialist")
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE sessions SET user_id = ? WHERE session_id = ?", ("merchant-b", "stale-session"))
        conn.commit()
    token = AuthManager.create_jwt_token("operator-alice", role="operator")
    response = client.post(
        f"/api/hitl/tickets/{ticket.ticket_id}/reply", headers={"Authorization": f"Bearer {token}"},
        json={"operator_id": "operator-alice", "reply_message": "Resolution registered."},
    )
    assert response.status_code == 403
    assert queue.get_ticket(ticket.ticket_id).status == "OPEN"
    assert store.get_conversation_context("stale-session", "merchant-b")["recent_messages"] == []


def test_hitl_idempotency_is_scoped_to_tenant_and_session(local_security):
    _, _, queue, _ = local_security
    first = queue.create_ticket("merchant-a", "session-a", "support", "First request", idempotency_key="shared-key")
    second = queue.create_ticket("merchant-b", "session-b", "support", "Second request", idempotency_key="shared-key")
    third = queue.create_ticket("merchant-a", "session-c", "support", "Third request", idempotency_key="shared-key")
    assert len({first.ticket_id, second.ticket_id, third.ticket_id}) == 3
    assert queue.get_ticket(first.ticket_id).user_id == "merchant-a"
    assert queue.get_ticket(second.ticket_id).user_id == "merchant-b"
    replay = queue.create_ticket("merchant-a", "session-a", "support", "First request", idempotency_key="shared-key")
    assert replay.ticket_id == first.ticket_id


def test_hitl_reply_rolls_back_ticket_if_message_persistence_fails(local_security):
    _, store, queue, _ = local_security
    ticket = queue.create_ticket("merchant-a", "atomic-reply", "support", "Need specialist")
    with store._lock, store._connect() as conn:
        conn.executescript("""
            CREATE TRIGGER reject_human_message BEFORE INSERT ON messages
            WHEN NEW.role = 'human_operator'
            BEGIN SELECT RAISE(ABORT, 'storage failed'); END;
        """)
    with pytest.raises(sqlite3.IntegrityError):
        queue.reply_to_ticket(ticket.ticket_id, "operator-alice", "Resolution registered.")
    assert queue.get_ticket(ticket.ticket_id).status == "OPEN"
    context = store.get_conversation_context("atomic-reply", "merchant-a")
    assert context["status"] == "waiting_human"
    assert context["recent_messages"] == []


def test_non_ascii_api_key_is_rejected_without_internal_error(local_security):
    with pytest.raises(HTTPException) as exc:
        AuthManager.resolve_chat_security_context("merchant-a", x_api_key="invalid-☃")
    assert exc.value.status_code == 401


def test_token_issuance_rejects_blank_subject_as_validation_error(local_security):
    config, _, _, client = local_security
    response = client.post("/api/auth/token", headers={"X-API-Key": config.AUTH_API_KEY}, json={"user_id": "   "})
    assert response.status_code == 422


def test_production_requires_independent_auth_and_signing_keys():
    key = "V5w7dzAl_F9Sx4h-M2iRQK26PO0JbN8eC3trYukG1Ha"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, APP_ENV="production", JWT_SECRET=key, AUTH_API_KEY=key)
