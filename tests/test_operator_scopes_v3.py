import importlib
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.security.auth import AuthManager


@pytest.mark.parametrize("method,path,body", [
    ("get", "/api/hitl/tickets", None),
    ("get", "/api/hitl/tickets/nonexistent", None),
    ("post", "/api/hitl/tickets/nonexistent/claim", None),
    ("post", "/api/hitl/tickets/nonexistent/reply", {"operator_id":"forged","reply_message":"resolve"}),
    ("get", "/api/metrics", None),
    ("get", "/api/evals/run", None),
    ("post", "/api/auth/token", {"user_id":"forged-admin","role":"operator"}),
])
def test_operator_role_without_resource_scope_is_denied(monkeypatch, method, path, body):
    auth=importlib.import_module('app.security.auth')
    main=importlib.import_module('app.main')
    config=Settings(_env_file=None,GEMINI_API_KEY='',AUTH_API_KEY='',JWT_SECRET='')
    monkeypatch.setattr(auth,'settings',config)
    token=AuthManager.create_jwt_token('limited-operator',role='operator',scopes=['chat:write'])
    response=getattr(TestClient(main.app),method)(path,headers={'Authorization':'Bearer '+token},**({'json':body} if body else {}))
    assert response.status_code==403


def test_operator_default_scopes_are_explicitly_privileged(monkeypatch):
    auth=importlib.import_module('app.security.auth')
    monkeypatch.setattr(auth,'settings',Settings(_env_file=None,GEMINI_API_KEY='',AUTH_API_KEY='',JWT_SECRET=''))
    token=AuthManager.create_jwt_token('operator',role='operator')
    claims=AuthManager.verify_jwt_token(token)
    assert 'admin:all' in claims['scopes']
