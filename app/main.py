from pathlib import Path
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.agents.orchestrator import orchestrator
from app.config import settings
from app.evals.evaluator import evaluator
from app.hitl.queue import hitl_queue, HITLTransitionError
from app.memory.store import state_store
from app.observability.metrics import metrics_collector
from app.observability.resilience import breaker_registry
from app.rag.retriever import retriever
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HitlOperatorReplyRequest,
    ActionConfirmationRequest,
)
from app.security.auth import AuthManager, CrossTenantAccessError, SecurityContext, require_operator_auth
from app.security.action_policy import ActionPolicyError, confirm_action
from app.security.rate_limit import rate_limiter
from app.tools.registry import tool_registry
from app.v2.adapters import StateAdapter, ToolGateway
from app.v2.knowledge import KnowledgeService
from app.v2.model import GeminiModel
from app.v2.runtime import Runtime

_integrated_runtime: Optional[Runtime] = None


def get_integrated_runtime() -> Runtime:
    global _integrated_runtime
    if _integrated_runtime is not None:
        return _integrated_runtime
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provedor Gemini não configurado (GEMINI_API_KEY ausente). O sistema opera exclusivamente com o runtime semântico integrado.",
        )
    model = GeminiModel()
    knowledge = KnowledgeService(model)
    state = StateAdapter(state_store)
    gateway = ToolGateway(settings.MOCK_CUSTOMERS_PATH, store=state_store, queue=hitl_queue)
    _integrated_runtime = Runtime(model, knowledge, state, gateway, hitl_queue)
    return _integrated_runtime


def set_integrated_runtime(runtime: Optional[Runtime]) -> None:
    global _integrated_runtime
    _integrated_runtime = runtime


def active_store():
    return _integrated_runtime.state.store if _integrated_runtime is not None else state_store


def active_queue():
    return _integrated_runtime.queue if _integrated_runtime is not None else hitl_queue

app = FastAPI(
    title="Getnet Multi-Agent Support System",
    description=(
        "Sistema Multi-Agente de Suporte Getnet com Supervisor ReAct, RAG Híbrido "
        "(Okapi BM25 + Embeddings Neurais 384d + RRF), Memória Multi-Turno SQLite, "
        "Fila HITL, Escrita em Duas Fases e Observabilidade."
    ),
    version="3.0.0",
)

# CORS restrito às origens autorizadas (sem wildcard + allow_credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.exception_handler(CrossTenantAccessError)
async def cross_tenant_access_handler(_request: Request, _exc: CrossTenantAccessError):
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": "Acesso negado à sessão solicitada."})


@app.exception_handler(ActionPolicyError)
async def action_policy_error_handler(_request: Request, exc: ActionPolicyError):
    code = 410 if exc.code == "PROPOSAL_EXPIRED" else 409
    if exc.code in {"INVALID_CONFIRMATION", "INVALID_ACTION_ARGUMENTS", "INVALID_ACTION_TYPE"}:
        code = 422
    return JSONResponse(status_code=code, content={"detail": str(exc), "code": exc.code})


@app.exception_handler(HITLTransitionError)
async def hitl_transition_error_handler(_request: Request, exc: HITLTransitionError):
    return JSONResponse(status_code=409, content={"detail": str(exc), "code": exc.code})


def require_scope(context: SecurityContext, scope: str) -> None:
    if scope not in context.scopes and "admin:all" not in context.scopes:
        raise HTTPException(status_code=403, detail="A credencial não permite esta operação.")


class TokenIssueRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    role: str = Field(default="merchant", pattern="^(merchant|operator)$")
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)

    @field_validator("user_id")
    @classmethod
    def validate_principal(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("Identificador do principal inválido.")
        return value


# ==============================================================================
# Core Chat Endpoint
# ==============================================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    http_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """
    Ponto de entrada principal do sistema Multi-Agente.
    - Aplica Rate Limiting por usuário/IP.
    - Valida contexto de segurança (JWT Bearer / X-API-Key / Sessão Escopada) com isolamento de tenant.
    - Orquestra Supervisor, Guardrails, RAG Híbrido, Tools e Fila HITL.
    """
    client_ip = http_request.client.host if http_request.client else "local"
    rate_limiter.enforce(key=f"chat:{request.user_id}:{client_ip}")

    sec_ctx = AuthManager.resolve_chat_security_context(
        request_user_id=request.user_id,
        x_api_key=x_api_key,
        authorization=authorization,
    )
    if sec_ctx.auth_method == "api_key":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Atendimento de chat exige token JWT vinculado ao titular; X-API-Key administrativa não pode personificar lojistas.",
        )
    if sec_ctx.auth_method == "jwt_bearer" and sec_ctx.role != "merchant":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Este endpoint exige credencial de lojista.",
        )
    require_scope(sec_ctx, "chat:write")
    sec_ctx.assert_tenant_access(request.user_id)

    runtime = get_integrated_runtime()

    try:
        return await run_in_threadpool(
            runtime.process,
            message=request.message,
            security_context=sec_ctx,
            session_id=request.session_id or None,
        )
    except HTTPException:
        raise
    except CrossTenantAccessError:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno de orquestração; consulte o trace da requisição.",
        ) from exc


# ==============================================================================
# V2 Compatibility Endpoints (Shared Integrated Runtime)
# ==============================================================================
@app.post("/api/v2/chat", response_model=ChatResponse)
async def v2_chat_endpoint(
    request: ChatRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """Alias compatível V2 compartilhando o mesmo runtime semântico integrado."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="JWT do lojista obrigatório.")
    try:
        claims = AuthManager.verify_jwt_token(authorization[7:])
    except HTTPException as exc:
        if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            raise
        raise HTTPException(status_code=401, detail="Credencial inválida.") from None
    except Exception:
        raise HTTPException(status_code=401, detail="Credencial inválida.") from None
    if claims.get("role") != "merchant":
        raise HTTPException(status_code=403, detail="Este endpoint exige credencial de lojista.")
    context = SecurityContext(claims["sub"], role=claims["role"], scopes=claims["scopes"], auth_method="jwt")
    try:
        context.assert_tenant_access(request.user_id)
        runtime = get_integrated_runtime()
        return await run_in_threadpool(runtime.process, request.message, context, request.session_id or None)
    except (CrossTenantAccessError, PermissionError):
        raise HTTPException(status_code=403, detail="Acesso negado.")


@app.post("/api/v2/hitl/{ticket_id}/resume", response_model=ChatResponse)
async def v2_resume_endpoint(
    ticket_id: str,
    operator: SecurityContext = Depends(require_operator_auth),
):
    """Alias compatível V2 de retomada HITL compartilhando o mesmo runtime integrado."""
    runtime = get_integrated_runtime()
    try:
        return await run_in_threadpool(runtime.resume, ticket_id, operator)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Acesso negado.")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=f"Protocolo não está pronto para retomada v2: {exc}")


# ==============================================================================
# Security & Token Issuance Endpoint
# ==============================================================================
@app.post("/api/auth/token")
async def issue_jwt_token(
    body: TokenIssueRequest,
    _operator: SecurityContext = Depends(require_operator_auth),
):
    """
    Emite um token JWT assinado (HMAC-SHA256) vinculado a um `user_id` ou `operator`.
    Requer `X-API-Key` válida.
    """
    require_scope(_operator, "auth:issue")
    token = AuthManager.create_jwt_token(
        user_id=body.user_id,
        role=body.role,
        ttl_seconds=body.ttl_seconds,
    )
    return {
        "access_token": token,
        "token_type": "Bearer",
        "user_id": body.user_id,
        "role": body.role,
        "expires_in": body.ttl_seconds,
    }


@app.post("/api/auth/revoke")
async def revoke_jwt_token(
    authorization: Optional[str] = Header(None),
):
    """
    Revoga um token JWT ativo via JTI denylist.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Authorization header com Bearer token é obrigatório.")
    token = authorization.split("Bearer ", 1)[1].strip()
    success = AuthManager.revoke_token(token)
    return {"revoked": success, "status": "revoked" if success else "invalid_or_expired"}


@app.post("/api/actions/{proposal_id}/confirm")
async def confirm_terminal_action(
    proposal_id: str, body: ActionConfirmationRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    context = AuthManager.resolve_chat_security_context(body.user_id, x_api_key, authorization)
    if context.auth_method == "demo_unverified":
        raise HTTPException(status_code=401, detail="Confirmação exige identidade autenticada.")
    if context.auth_method != "jwt_bearer":
        raise HTTPException(status_code=403, detail="Confirmação exige token JWT vinculado ao titular.")
    require_scope(context, "terminal:action_own")
    # Even an operator cannot confirm on behalf of a different principal.
    if context.authenticated_user_id != body.user_id:
        raise HTTPException(status_code=403, detail="Somente o titular pode confirmar esta proposta.")
    kwargs = {}
    if _integrated_runtime is not None:
        kwargs = {"store": active_store(), "customers_path": _integrated_runtime.gateway.path}
    return await run_in_threadpool(confirm_action, proposal_id, context.authenticated_user_id, body.expected_payload_hash, **kwargs)


@app.get("/api/tool-operations/{operation_id}")
async def get_tool_operation(
    operation_id: str, user_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    context = AuthManager.resolve_chat_security_context(user_id, x_api_key, authorization)
    if context.auth_method == "demo_unverified":
        raise HTTPException(status_code=401, detail="Consulta de operação exige identidade autenticada.")
    if context.auth_method != "jwt_bearer":
        raise HTTPException(status_code=403, detail="Consulta exige token JWT vinculado ao titular.")
    try:
        registry = _integrated_runtime.gateway.registry if _integrated_runtime is not None else tool_registry
        return registry.get_operation(operation_id, context)
    except CrossTenantAccessError:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Escopo insuficiente para a operação.") from exc


# ==============================================================================
# Session Memory Endpoints
# ==============================================================================
@app.get("/api/sessions/{session_id}")
async def get_session_history(
    session_id: str,
    user_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    sec_ctx = AuthManager.resolve_chat_security_context(
        request_user_id=user_id,
        x_api_key=x_api_key,
        authorization=authorization,
        allow_operator_tenant_access=True,
    )
    tenant_id = sec_ctx.assert_tenant_access(user_id)
    require_scope(sec_ctx, "customer:read_own")
    ctx = active_store().get_conversation_context(
        session_id=session_id,
        user_id=tenant_id,
        limit=20,
    )
    return ctx


@app.delete("/api/sessions/{session_id}")
async def clear_session_history(
    session_id: str,
    user_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    sec_ctx = AuthManager.resolve_chat_security_context(
        request_user_id=user_id,
        x_api_key=x_api_key,
        authorization=authorization,
        allow_operator_tenant_access=True,
    )
    tenant_id = sec_ctx.assert_tenant_access(user_id)
    require_scope(sec_ctx, "chat:write")
    active_store().clear_session(session_id, user_id=tenant_id)
    return {"status": "cleared", "session_id": session_id}


# ==============================================================================
# Human-in-the-Loop (HITL) Queue Endpoints
# ==============================================================================
@app.get("/api/hitl/tickets")
async def list_hitl_tickets(
    status_filter: Optional[str] = None,
    user_id: Optional[str] = None,
    _operator: SecurityContext = Depends(require_operator_auth),
):
    require_scope(_operator, "hitl:manage")
    tickets = active_queue().list_tickets(status_filter=status_filter, user_id=user_id)
    return {
        "count": len(tickets),
        "tickets": [t.model_dump() for t in tickets],
    }


@app.get("/api/hitl/tickets/{ticket_id}")
async def get_hitl_ticket_detail(ticket_id: str, _operator: SecurityContext = Depends(require_operator_auth)):
    require_scope(_operator, "hitl:manage")
    ticket = active_queue().get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protocolo HITL não encontrado.")
    return ticket.model_dump()


@app.post("/api/hitl/tickets/{ticket_id}/reply")
async def operator_reply_hitl_ticket(
    ticket_id: str,
    body: HitlOperatorReplyRequest,
    _operator: SecurityContext = Depends(require_operator_auth),
):
    """
    Permite que um operador humano (autenticado com X-API-Key ou Bearer JWT de operador)
    responda a um chamado na fila HITL, injetando a resolução na sessão do cliente e retomando o estado.
    Quando resume_agent=True, aciona a retomada ativa da sessão pelo runtime integrado.
    """
    require_scope(_operator, "hitl:manage")
    orchestrator_adapter = None
    if body.resume_agent:
        runtime = get_integrated_runtime()

        class _RuntimeResumeAdapter:
            @staticmethod
            def resume_session_from_hitl(session_id, user_id, ticket_id, operator_reply, operator_id, store):
                return runtime.resume(ticket_id, _operator)

        orchestrator_adapter = _RuntimeResumeAdapter

    updated = active_queue().reply_to_ticket(
        ticket_id=ticket_id,
        operator_id=_operator.authenticated_user_id,
        reply_message=body.reply_message,
        resolve_ticket=body.resolve_ticket,
        resume_agent=body.resume_agent,
        orchestrator=orchestrator_adapter,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protocolo HITL não encontrado.")
    return updated.model_dump()


@app.post("/api/hitl/tickets/{ticket_id}/resume")
async def resume_hitl_ticket(
    ticket_id: str,
    _operator: SecurityContext = Depends(require_operator_auth),
):
    """
    Aciona a retomada ativa da sessão pelo runtime semântico integrado após a resolução de um chamado HITL.
    Aplica a diretriz do operador e persiste a resposta final na sessão para o cliente.
    """
    require_scope(_operator, "hitl:manage")
    ticket = active_queue().get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protocolo HITL não encontrado.")
    if ticket.status != "RESOLVED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Retomada de sessão exige que o protocolo esteja em status RESOLVED.")
    runtime = get_integrated_runtime()
    try:
        resumed = await run_in_threadpool(runtime.resume, ticket_id, _operator)
        return resumed.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@app.post("/api/hitl/tickets/{ticket_id}/claim")
async def claim_hitl_ticket(ticket_id: str, operator: SecurityContext = Depends(require_operator_auth)):
    require_scope(operator, "hitl:manage")
    updated = active_queue().claim_ticket(ticket_id, operator.authenticated_user_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Protocolo HITL não encontrado.")
    return updated.model_dump()


# ==============================================================================
# Observability, Health & Evaluation Endpoints
# ==============================================================================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="Getnet Multi-Agent System",
        version="3.0.0",
        gemini_configured=bool(settings.GEMINI_API_KEY),
        database_status="connected",
        rag_chunks_indexed=len(retriever.chunks),
        circuit_breakers=breaker_registry.states(),
    )


@app.get("/api/metrics")
async def get_operational_metrics(_operator: SecurityContext = Depends(require_operator_auth)):
    require_scope(_operator, "metrics:read")
    return metrics_collector.snapshot()


@app.get("/api/traces")
async def list_persisted_traces(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    _operator: SecurityContext = Depends(require_operator_auth),
):
    from app.observability.trace_store import TraceRepository
    require_scope(_operator, "metrics:read")
    return {"traces": TraceRepository.default().list_traces(user_id=user_id, session_id=session_id, limit=limit)}


@app.get("/api/traces/{trace_id}")
async def get_persisted_trace(
    trace_id: str,
    _operator: SecurityContext = Depends(require_operator_auth),
):
    from app.observability.trace_store import TraceRepository
    require_scope(_operator, "metrics:read")
    trace = TraceRepository.default().get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace não encontrado.")
    return trace


@app.get("/api/evals/run")
async def run_evaluation_suite(_operator: SecurityContext = Depends(require_operator_auth)):
    require_scope(_operator, "evals:run")
    raise HTTPException(status_code=409, detail="A avaliação legada não mede o runtime integrado. Execute a validação explícita offline ou a campanha Free Tier documentada; este endpoint não dispara consumo de modelo.")


@app.get("/api/scenarios")
async def list_scenarios():
    return [
        {"id": 1, "title": "Get Clássica vs Smart", "message": "Qual é a diferença entre a Get Clássica e a Get Smart?", "user_id": "cliente1988", "category": "Produtos (RAG)"},
        {"id": 2, "title": "Previsão do Tempo", "message": "Qual é a previsão do tempo em Porto Alegre para amanhã?", "user_id": "cliente1988", "category": "Web Search"},
        {"id": 3, "title": "Depósito de Vendas", "message": "Quando cai o dinheiro das minhas vendas de ontem?", "user_id": "cliente1988", "category": "Suporte / Vendas"},
        {"id": 4, "title": "Conta Bancária para Pix", "message": "Preciso ter conta bancária para receber minhas vendas via Pix?", "user_id": "cliente1988", "category": "Pix (RAG)"},
        {"id": 5, "title": "Máquina sem Conexão", "message": "Minha maquininha não conecta na internet, o que eu faço?", "user_id": "cliente1988", "category": "Diagnóstico POS"},
        {"id": 6, "title": "Antecipação de Recebíveis", "message": "Como funciona a antecipação de recebíveis com a Getnet?", "user_id": "cliente1988", "category": "Crédito (RAG)"},
        {"id": 7, "title": "Cotação do Euro", "message": "Qual é a cotação do euro hoje?", "user_id": "cliente1988", "category": "Câmbio / Web"},
        {"id": 8, "title": "Transação Recusada", "message": "Minha maquininha está mostrando um erro de transação recusada.", "user_id": "cliente1988", "category": "Diagnóstico POS"},
        {"id": 9, "title": "Parcelamento Crediário", "message": "Em quantas parcelas posso dividir uma venda no crediário?", "user_id": "cliente1988", "category": "Crediário (RAG)"},
        {"id": 10, "title": "Vender via WhatsApp", "message": "Posso vender pelo WhatsApp usando o Link de Pagamento?", "user_id": "cliente1988", "category": "Link de Pagamento"},
    ]


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))
