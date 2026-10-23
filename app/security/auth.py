import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional
from fastapi import Header, HTTPException, status
from app.config import settings


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
    Gerenciador de autenticação e tokens JWT (HMAC-SHA256) e API Keys.
    """

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64url_decode(data: str) -> bytes:
        padding = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + padding)

    @classmethod
    def create_jwt_token(
        cls,
        user_id: str,
        role: str = "merchant",
        scopes: Optional[List[str]] = None,
        ttl_seconds: int = 3600,
    ) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        now = int(time.time())
        payload = {
            "sub": user_id,
            "role": role,
            "scopes": scopes or ["chat:write", "customer:read_own", "terminal:action_own"],
            "iat": now,
            "exp": now + ttl_seconds,
            "iss": "getnet-multiagent-auth",
        }
        header_b64 = cls._b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = cls._b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = hmac.new(
            settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = cls._b64url_encode(signature)
        return f"{header_b64}.{payload_b64}.{sig_b64}"

    @classmethod
    def verify_jwt_token(cls, token: str) -> dict:
        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Formato de token JWT inválido.",
            )
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(
            settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        try:
            provided_sig = cls._b64url_decode(sig_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Assinatura JWT malformada.",
            ) from exc

        if not hmac.compare_digest(expected_sig, provided_sig):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Assinatura do token JWT inválida.",
            )

        try:
            payload = json.loads(cls._b64url_decode(payload_b64).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Payload JWT inválido.",
            ) from exc

        exp = payload.get("exp", 0)
        if int(time.time()) > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token JWT expirado.",
            )
        return payload

    @classmethod
    def resolve_chat_security_context(
        cls,
        request_user_id: str,
        x_api_key: Optional[str] = None,
        authorization: Optional[str] = None,
    ) -> SecurityContext:
        """
        Valida credenciais da requisição e vincula a identidade autenticada.
        - Se Authorization Bearer for enviado: verifica assinatura JWT e exige que `sub == request_user_id`
          (salvo role == 'operator').
        - Se X-API-Key for enviado: compara em tempo constante contra AUTH_API_KEY.
        - Se AUTH_ENFORCE_STRICT=True e nenhuma credencial foi enviada: rejeita com 401.
        """
        if authorization:
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
            if role != "operator" and token_sub != request_user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Acesso negado (Cross-Tenant): token pertence a '{token_sub}', mas a requisição solicitou '{request_user_id}'.",
                )
            return SecurityContext(
                authenticated_user_id=request_user_id,
                role=role,
                auth_method="jwt_bearer",
                scopes=scopes,
            )

        if x_api_key is not None:
            if not hmac.compare_digest(x_api_key.strip(), settings.AUTH_API_KEY):
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

        # Modo compatível com o contrato padrão do desafio (POST /api/chat com user_id),
        # porém com isolamento estrito da sessão ao request_user_id autenticado.
        return SecurityContext(
            authenticated_user_id=request_user_id,
            role="merchant",
            auth_method="scoped_session",
        )


def require_operator_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> SecurityContext:
    """
    Dependência FastAPI para rotas administrativas e de operadores humanos (HITL).
    """
    if x_api_key and hmac.compare_digest(x_api_key.strip(), settings.AUTH_API_KEY):
        return SecurityContext(
            authenticated_user_id="operator_system",
            role="operator",
            auth_method="api_key",
            scopes=["admin:all", "hitl:manage"],
        )
    if authorization:
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
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Credenciais de operador ausentes ou inválidas (X-API-Key ou Bearer JWT de operador requerido).",
    )
