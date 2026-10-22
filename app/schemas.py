from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ChatRequest(BaseModel):
    message: str = Field(..., description="A mensagem ou pergunta enviada pelo usuário")
    user_id: str = Field(..., description="Identificador do usuário (ex: cliente1988)")

class TraceStep(BaseModel):
    step_id: int
    agent: str
    action: str
    status: str = "success"
    details: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0

class ChatResponse(BaseModel):
    response: str
    user_id: str
    agent_used: str
    intent: Optional[str] = None
    tools_called: List[str] = []
    sources: List[str] = []
    trace: List[TraceStep] = []
    execution_time_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    gemini_configured: bool
