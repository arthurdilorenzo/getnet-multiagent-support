"""Separate authenticated experimental app; does not mount or modify app.main."""
from fastapi import Depends, FastAPI, Header, HTTPException
from starlette.concurrency import run_in_threadpool
from app.schemas import ChatRequest, ChatResponse
from app.security.auth import AuthManager, CrossTenantAccessError, SecurityContext, require_operator_auth


def create_app(runtime):
    app = FastAPI(title="Getnet v2 pilot", version="0.1.0")

    def merchant(authorization: str | None = Header(default=None)):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "JWT do lojista obrigatório.")
        try:
            claims = AuthManager.verify_jwt_token(authorization[7:])
        except Exception:
            raise HTTPException(401, "Credencial inválida.") from None
        if claims["role"] != "merchant":
            raise HTTPException(403, "Este endpoint exige credencial de lojista.")
        return SecurityContext(claims["sub"], role=claims["role"], scopes=claims["scopes"], auth_method="jwt")

    @app.post("/api/v2/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, context=Depends(merchant)):
        try:
            context.assert_tenant_access(request.user_id)
            return await run_in_threadpool(runtime.process, request.message, context, request.session_id)
        except (CrossTenantAccessError, PermissionError):
            raise HTTPException(403, "Acesso negado.") from None

    @app.post("/api/v2/hitl/{ticket_id}/resume", response_model=ChatResponse)
    async def resume(ticket_id: str, operator=Depends(require_operator_auth)):
        try:
            return await run_in_threadpool(runtime.resume, ticket_id, operator)
        except PermissionError:
            raise HTTPException(403, "Acesso negado.") from None
        except ValueError:
            raise HTTPException(409, "Protocolo não está pronto para retomada v2.") from None

    return app
