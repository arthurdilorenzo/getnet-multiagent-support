from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.agents.orchestrator import orchestrator

app = FastAPI(
    title="Getnet Multi-Agent Support System",
    description="Sistema Multi-Agente inteligente para atendimento, suporte operacional e RAG Getnet.",
    version="1.0.0"
)

# CORS liberado para testes e integrações
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Endpoint principal exigido pelo desafio
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Ponto de entrada único para o sistema Multi-Agente.
    Recebe message e user_id, orquestra triagem, RAG e ferramentas operacionais.
    """
    try:
        response = orchestrator.process(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de orquestração: {str(e)}")

# Endpoint de saúde operacional
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="Getnet Multi-Agent System",
        version="1.0.0",
        gemini_configured=bool(settings.GEMINI_API_KEY)
    )

# Endpoint auxiliar com os 10 cenários do teste oficial (em Português nativo)
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
        {"id": 10, "title": "Vender via WhatsApp", "message": "Posso vender pelo WhatsApp usando o Link de Pagamento?", "user_id": "cliente1988", "category": "Link de Pagamento"}
    ]

# Monta arquivos estáticos da UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))
