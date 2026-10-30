import pytest
from app.security.auth import AuthManager
from fastapi import HTTPException


def test_token_with_jti_is_valid_and_then_rejected_after_revocation():
    user = "cliente1988"
    token = AuthManager.create_jwt_token(user_id=user)
    
    # 1. Token válido antes da revogação
    payload = AuthManager.verify_jwt_token(token)
    assert payload["sub"] == user
    assert "jti" in payload
    
    # 2. Outro token emitido para o mesmo usuário
    other_token = AuthManager.create_jwt_token(user_id=user)
    
    # 3. Revoga o primeiro token
    success = AuthManager.revoke_token(token)
    assert success is True
    
    # 4. Primeiro token deve ser rejeitado com 401
    with pytest.raises(HTTPException) as exc_info:
        AuthManager.verify_jwt_token(token)
    assert exc_info.value.status_code == 401
    assert "revoked" in str(exc_info.value.detail).lower() or "inválido" in str(exc_info.value.detail).lower()
    
    # 5. Outro token permanece perfeitamente válido
    other_payload = AuthManager.verify_jwt_token(other_token)
    assert other_payload["sub"] == user


def test_revoking_invalid_or_garbage_token_returns_false():
    assert AuthManager.revoke_token("invalid.token.string") is False
    assert AuthManager.revoke_token("") is False
