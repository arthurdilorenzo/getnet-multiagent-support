import base64
import hashlib
import hmac
import json
import re
import time
import uuid
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import List, Optional, Set
from fastapi import Header, HTTPException, status
from app.config import settings
from app.security.revocations import RevocationStore


class CrossTenantAccessError(PermissionError):
    """Exceção disparada quando há tentativa de leitura/escrita entre contas distintas."""


@dataclass(frozen=True)
class SecurityContext:
    authenticated_user_id: str
    role: str = "merchant"
    auth_method: str = "scoped_session"
    scopes: List[str] = field(default_factory=lambda: ["chat:write", "customer:read_own", "terminal:action_own"])

    def assert_tenant_access(self, target_user_id: str) -> str:
        """
        Garante isolamento estrito de tenant (Zero-Trust Identity Binding).
        O LLM nunca escolhe o user_id livremente; toda tool valida contra este contexto.
        """
        normalized_target = (target_user_id or "").strip()
        if self.role == "operator" or "admin:all" in self.scopes:
            return normalized_target or self.authenticated_user_id
        if normalized_target and normalized_target != self.authenticated_user_id:
            raise CrossTenantAccessError(
                f"Violação de isolamento de tenant: sessão autenticada para '{self.authenticated_user_id}' "
                f"não possui permissão para acessar dados de '{normalized_target}'."
            )
        return self.authenticated_user_id


class AuthManager:
    """
    Gerenciador de autenticação e tokens JWT (HMAC-SHA256) com suporte a JTI e revogação.
    """

    REVOKED_JTIS: Set[str] = set()
    REVOCATION_STORE: Optional[RevocationStore] = None
    _STORE_BY_PATH: dict[str, RevocationStore] = {}

    @classmethod
    def _get_revocation_store(cls) -> RevocationStore:
        if cls.REVOCATION_STORE is not None:
            return cls.REVOCATION_STORE
        db_path: Path | str = settings.SQLITE_DB_PATH
        main_mod = sys.modules.get("app.main")
        if main_mod is not None:
            active_fn = getattr(main_mod, "active_store", None)
            if callable(active_fn):
                try:
                    active = active_fn()
                    if getattr(active, "db_path", None) is not None:
                        db_path = active.db_path
                except Exception:
                    pass
        key = str(Path(db_path))
        store = cls._STORE_BY_PATH.get(key)
        if store is None:
            store = RevocationStore(db_path=key)
            cls._STORE_BY_PATH[key] = store
        return store

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64url_decode(data: str) -> bytes:
        if not data or not re.fullmatch(r"[A-Za-z0-9_-]+", data):
            raise ValueError("Invalid base64url segment")
        padding = "=" * (-len(data) % 4)
        return base64.b64decode(data + padding, altchars=b"-_", validate=True)

    @classmethod
    def create_jwt_token(
        cls,
        user_id: str,
        role: str = "merchant",
        scopes: Optional[List[str]] = None,
        ttl_seconds: int = 3600,
        jti: Optional[str] = None,
    ) -> str:
        if not isinstance(user_id, str) or not user_id.strip() or user_id != user_id.strip() or len(user_id) > 128:
            raise ValueError("Identificador do principal inválido.")
        if role not in {"merchant", "operator"} or type(ttl_seconds) is not int or not 0 < ttl_seconds <= 86400:
            raise ValueError("Role ou validade JWT inválida.")
        header = {"alg": "HS256", "typ": "JWT"}
        now = int(time.time())
        token_jti = jti or uuid.uuid4().hex
        payload = {
            "sub": user_id,
            "role": role,
            "scopes": scopes if scopes is not None else (
                ["admin:all", "hitl:manage"] if role == "operator" else
                ["chat:write", "customer:read_own", "terminal:action_own"]),
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": token_jti,
            "iss": "getnet-multiagent-auth",
        }
        header_b64 = cls._b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = cls._b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = hmac.new(
            settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = cls._b64url_encode(signature)
        # Once JWT is issued, this process cannot return to unauthenticated demo access.
        settings.AUTH_ENFORCE_STRICT = True
        return f"{header_b64}.{payload_b64}.{sig_b64}"

    @classmethod
    def revoke_token(cls, token: str) -> bool:
        """Adiciona o JTI do token à denylist de revogação persistida em SQLite e cache em memória."""
        try:
            payload = cls.verify_jwt_token(token, check_revocation=False)
        except Exception:
            return False
        token_jti = payload.get("jti")
        issuer = payload.get("iss", "getnet-multiagent-auth")
        exp = int(payload.get("exp") or (int(time.time()) + 3600))
        if not isinstance(token_jti, str) or not token_jti.strip():
            return False
        try:
            cls._get_revocation_store().revoke(issuer=issuer, jti=token_jti, expires_at=exp)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Serviço de persistência de revogação indisponível.",
            ) from exc
        cls.REVOKED_JTIS.add(token_jti)
        return True

    @classmethod
    def is_jti_revoked(cls, jti: str, issuer: str = "getnet-multiagent-auth", now: Optional[int] = None) -> bool:
        if not jti:
            return False
        if jti in cls.REVOKED_JTIS:
            return True
        try:
            revoked = cls._get_revocation_store().is_revoked(issuer=issuer, jti=jti, now=now)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Serviço de verificação de revogação de credenciais indisponível.",
            ) from exc
        if revoked:
            cls.REVOKED_JTIS.add(jti)
        return revoked

    @classmethod
    def verify_jwt_token(cls, token: str, check_revocation: bool = True) -> dict:
        try:
            if not isinstance(token, str) or len(token) > 8192:
                raise ValueError("Invalid token length")
            header_b64, payload_b64, sig_b64 = token.split(".")
            header = json.loads(cls._b64url_decode(header_b64).decode("utf-8"))
            if not isinstance(header, dict) or header.get("alg") != "HS256" or header.get("typ") != "JWT" or "crit" in header:
                raise ValueError("Unsupported JWT header")
            signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
            expected_sig = hmac.new(settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
            provided_sig = cls._b64url_decode(sig_b64)
            if not hmac.compare_digest(expected_sig, provided_sig):
                raise ValueError("Invalid JWT signature")
            payload = json.loads(cls._b64url_decode(payload_b64).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Invalid JWT payload")
            subject = payload.get("sub")
            role = payload.get("role")
            scopes = payload.get("scopes")
            now = int(time.time())
            if not isinstance(subject, str) or not subject.strip() or subject != subject.strip() or len(subject) > 128 or any(ord(c) < 32 for c in subject):
                raise ValueError("Invalid JWT subject")
            if payload.get("iss") != "getnet-multiagent-auth" or role not in {"merchant", "operator"}:
                raise ValueError("Invalid JWT issuer or role")
            if not isinstance(scopes, list) or any(not isinstance(scope, str) or not scope.strip() for scope in scopes):
                raise ValueError("Invalid JWT scopes")
            if role != "operator" and any(scope.startswith("admin:") or scope.startswith("hitl:") for scope in scopes):
                raise ValueError("Privileged scopes require an operator")
            exp, issued = payload.get("exp"), payload.get("iat")
            if type(exp) is not int or type(issued) is not int or exp <= now or issued > now + 30 or exp <= issued:
                raise ValueError("Invalid JWT lifetime")
            if "nbf" in payload and (type(payload["nbf"]) is not int or payload["nbf"] > now):
                raise ValueError("JWT is not active")
            token_jti = payload.get("jti")
            if not isinstance(token_jti, str) or not token_jti.strip() or len(token_jti) > 256:
                raise ValueError("Missing or invalid JWT JTI")
            if check_revocation and cls.is_jti_revoked(token_jti, issuer=payload.get("iss", "getnet-multiagent-auth"), now=now):
                raise ValueError("Token JTI has been revoked")
        except HTTPException:
            raise
        except (ValueError, TypeError, UnicodeError) as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token JWT inválido ou expirado.") from exc
        settings.AUTH_ENFORCE_STRICT = True
        return payload

    @classmethod
    def resolve_chat_security_context(
        cls,
        request_user_id: str,
        x_api_key: Optional[str] = None,
        authorization: Optional[str] = None,
        allow_operator_tenant_access: bool = False,
    ) -> SecurityContext:
        """
        Valida credenciais da requisição e vincula a identidade autenticada.
        - Se Authorization Bearer for enviado: verifica assinatura JWT e exige que `sub == request_user_id`
          (salvo role == 'operator').
        - Se X-API-Key for enviado: compara em tempo constante contra AUTH_API_KEY.
        - Se AUTH_ENFORCE_STRICT=True e nenhuma credencial foi enviada: rejeita com 401.
        """
        request_user_id = (request_user_id or "").strip()
        if not request_user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Identificador de usuário inválido.")
        if authorization is not None:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Cabeçalho Authorization deve seguir o formato 'Bearer <token>'.",
                )
            claims = cls.verify_jwt_token(token.strip())
            token_sub = claims.get("sub", "")
            role = claims.get("role", "merchant")
            scopes = claims.get("scopes", [])
            if token_sub != request_user_id and not (role == "operator" and allow_operator_tenant_access):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Acesso negado (Cross-Tenant): token pertence a '{token_sub}', mas a requisição solicitou '{request_user_id}'.",
                )
            return SecurityContext(
                authenticated_user_id=token_sub,
                role=role,
                auth_method="jwt_bearer",
                scopes=scopes,
            )

        if x_api_key is not None:
            if not hmac.compare_digest(x_api_key.strip().encode("utf-8"), settings.AUTH_API_KEY.encode("utf-8")):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="X-API-Key inválida.",
                )
            return SecurityContext(
                authenticated_user_id=request_user_id,
                role="merchant",
                auth_method="api_key",
            )

        if settings.AUTH_ENFORCE_STRICT:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Autenticação obrigatória: forneça cabeçalho 'Authorization: Bearer <token>' ou 'X-API-Key'.",
            )

        # Explicitly unauthenticated development demonstration, never proof of merchant identity.
        return SecurityContext(
            authenticated_user_id=request_user_id,
            role="merchant",
            auth_method="demo_unverified",
        )


def require_operator_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> SecurityContext:
    """
    Dependência FastAPI para rotas administrativas e de operadores humanos (HITL).
    """
    if authorization is not None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            claims = AuthManager.verify_jwt_token(token.strip())
            if claims.get("role") == "operator":
                return SecurityContext(
                    authenticated_user_id=claims.get("sub", "operator"),
                    role="operator",
                    auth_method="jwt_bearer",
                    scopes=claims.get("scopes", ["admin:all", "hitl:manage"]),
                )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Privilégios de operador são necessários para este recurso.",
            )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Cabeçalho Authorization inválido.")
    if x_api_key is not None and hmac.compare_digest(x_api_key.strip().encode("utf-8"), settings.AUTH_API_KEY.encode("utf-8")):
        return SecurityContext(
            authenticated_user_id="operator_system",
            role="operator",
            auth_method="api_key",
            scopes=["admin:all", "hitl:manage"],
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais de operador ausentes ou inválidas (X-API-Key ou Bearer JWT de operador requerido).",
    )
