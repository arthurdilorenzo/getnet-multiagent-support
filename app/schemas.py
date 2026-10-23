from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


AgentTypeLiteral = Literal[
    "KNOWLEDGE_AGENT",
    "CUSTOMER_SUPPORT_AGENT",
    "WEB_SEARCH_AGENT",
    "HUMAN_ESCALATION_AGENT",
]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="Mensagem ou consulta enviada pelo usuário")
    user_id: str = Field(..., min_length=1, max_length=128, description="Identificador da conta do cliente")
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Identificador opcional da sessão para persistência multi-turno",
    )

    @field_validator("user_id")
    @classmethod
    def sanitize_user_id(cls, v: str) -> str:
        return v.strip()


class Citation(BaseModel):
    chunk_id: str
    title: str
    source: str
    source_url: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    excerpt: str = ""


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class TraceStep(BaseModel):
    step_id: int
    agent: str
    action: str
    status: str = "success"
    details: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    step_duration_ms: float = 0.0


class PlanStep(BaseModel):
    step_order: int = Field(ge=1, le=5)
    target_agent: AgentTypeLiteral
    sub_query: str
    rationale: str
    request_unit_ids: List[str] = Field(default_factory=list)


class RoutingPlan(BaseModel):
    primary_agent: AgentTypeLiteral
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    plan_steps: List[PlanStep] = Field(default_factory=list)
    requires_user_context: bool = False
    is_multi_step: bool = False


class ToolCallSpec(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""
    rationale: str = ""


class ToolExecutionResult(BaseModel):
    tool_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
    execution_ms: float = 0.0
    idempotent_replay: bool = False
    operation_id: Optional[str] = None
    outcome_state: Literal["completed", "pending", "unknown", "failed"] = "completed"


    @model_validator(mode="after")
    def failure_state(self):
        # Pending/unknown describe real in-flight operations; a failed call can
        # never inherit the successful default used by older call sites.
        if not self.success and self.outcome_state == "completed":
            self.outcome_state = "failed"
        return self


class SpecialistOutput(BaseModel):
    agent_name: str
    response: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    abstained: bool = False
    semantic_mode: Literal['', 'native_semantic', 'limited_offline'] = ''
    answer_status: Literal['', 'complete', 'partial', 'abstained'] = ''
    request_unit_ids: List[str] = Field(default_factory=list)
    covered_unit_ids: List[str] = Field(default_factory=list)
    unresolved_unit_ids: List[str] = Field(default_factory=list)
    abstention_reason: Optional[str] = None
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None
    tools_called: List[str] = Field(default_factory=list)
    tool_results: List[ToolExecutionResult] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str = ""
    trace_id: str = ""
    agent_used: str
    intent: Optional[str] = None
    confidence: float = 1.0
    abstained: bool = False
    semantic_mode: Literal['', 'native_semantic', 'limited_offline'] = ''
    answer_status: Literal['', 'complete', 'partial', 'abstained'] = ''
    request_coverage: List[Dict[str, Any]] = Field(default_factory=list)
    escalated: bool = False
    ticket_id: Optional[str] = None
    tools_called: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    action_proposals: List[Dict[str, Any]] = Field(default_factory=list)
    trace: List[TraceStep] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    gemini_configured: bool
    database_status: str = "connected"
    rag_chunks_indexed: int = 0
    circuit_breakers: Dict[str, str] = Field(default_factory=dict)


class HitlTicketModel(BaseModel):
    ticket_id: str
    session_id: str
    user_id: str
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    category: str
    reason: str
    status: Literal["OPEN", "CLAIMED", "RESPONDED", "RESOLVED"] = "OPEN"
    assigned_operator: Optional[str] = None
    operator_response: Optional[str] = None
    diagnostic_snapshot: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str
    updated_at: str


class HitlOperatorReplyRequest(BaseModel):
    operator_id: str = Field(..., min_length=2, max_length=64)
    reply_message: str = Field(..., min_length=2, max_length=4000)
    resolve_ticket: bool = True


class ActionConfirmationRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    expected_payload_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    confirmation: Literal["CONFIRM"]
