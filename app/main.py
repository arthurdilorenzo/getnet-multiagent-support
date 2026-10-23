from pathlib import Path
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agents.orchestrator import orchestrator
from app.config import settings
from app.evals.evaluator import evaluator
from app.hitl.queue import hitl_queue
from app.memory.store import state_store
from app.observability.metrics import metrics_collector
from app.observability.resilience import breaker_registry
from app.rag.retriever import retriever
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HitlOperatorReplyRequest,
)
from app.security.auth import AuthManager, SecurityContext, require_operator_auth
from app.security.rate_limit import rate_limiter

app = FastAPI(
    title="Getnet Multi-Agent Support System",
    description=(
        "Sistema Multi-Agente de Suporte Getnet com Supervisor ReAct, RAG Híbrido "
        "(Okapi BM25 + Dense Embeddings + RRF), Memória Multi-Turno SQLite, Fila HITL e Observabilidade."
    ),
    version="2.0.0",
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


class TokenIssueRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    role: str = Field(default="merchant", pattern="^(merchant|operator)$")
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)


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

    try:
        return orchestrator.process(request=request, security_context=sec_ctx)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno de orquestração: {str(exc)}",
        ) from exc


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
    )
    ctx = state_store.get_conversation_context(
        session_id=session_id,
        user_id=sec_ctx.authenticated_user_id,
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
    )
    expected_prefix = f"sess_{sec_ctx.authenticated_user_id}"
    if sec_ctx.role != "operator" and not session_id.startswith(expected_prefix) and session_id != sec_ctx.authenticated_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Acesso negado à sessão solicitada.")
    state_store.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ==============================================================================
# Human-in-the-Loop (HITL) Queue Endpoints
# ==============================================================================
@app.get("/api/hitl/tickets")
async def list_hitl_tickets(
    status_filter: Optional[str] = None,
    user_id: Optional[str] = None,
):
    tickets = hitl_queue.list_tickets(status_filter=status_filter, user_id=user_id)
    return {
        "count": len(tickets),
        "tickets": [t.model_dump() for t in tickets],
    }


@app.get("/api/hitl/tickets/{ticket_id}")
async def get_hitl_ticket_detail(ticket_id: str):
    ticket = hitl_queue.get_ticket(ticket_id)
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
    """
    updated = hitl_queue.reply_to_ticket(
        ticket_id=ticket_id,
        operator_id=body.operator_id,
        reply_message=body.reply_message,
        resolve_ticket=body.resolve_ticket,
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Protocolo HITL não encontrado.")
    return updated.model_dump()


# ==============================================================================
# Observability, Health & Evaluation Endpoints
# ==============================================================================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="Getnet Multi-Agent System",
        version="2.0.0",
        gemini_configured=bool(settings.GEMINI_API_KEY),
        database_status="connected",
        rag_chunks_indexed=len(retriever.chunks),
        circuit_breakers=breaker_registry.states(),
    )


@app.get("/api/metrics")
async def get_operational_metrics():
    return metrics_collector.snapshot()


@app.get("/api/evals/run")
async def run_evaluation_suite():
    return evaluator.run_suite()


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
