"""Private DTOs and strict schemas for Getnet Núcleo 2.0 consolidation."""
from typing import Annotated, Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


# --- Planning Phase Proposals (from model) ---

class BindingProposal(Contract):
    tool_name: str
    resource_type: str
    resource_id: str
    field_pointer: str
    unit: Optional[str] = None


class RequirementProposal(Contract):
    kind: Literal["documentary", "operational", "human"] = "documentary"
    summary: str = ""
    requirement_summary: Optional[str] = None
    requirement_id: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)
    user_constraints_and_negations: list[str] = Field(default_factory=list)
    bindings: list[BindingProposal] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_proposal(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "summary" not in data and "requirement_summary" in data:
                data["summary"] = data["requirement_summary"]
            elif "summary" not in data:
                data["summary"] = "Consulta"
            if "kind" not in data:
                data["kind"] = "documentary"
        return data


class StepProposal(Contract):
    kind: Literal["tool", "knowledge", "human"]
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    query: str = ""
    reason: str = ""
    requirement_indexes: list[int] = Field(default_factory=list)


class Entities(Contract):
    terminal_id: Optional[str] = None
    product_discussed: Optional[str] = None
    ticket_id: Optional[str] = None


class PlanProposal(Contract):
    requirements: list[RequirementProposal] = Field(min_length=1)
    steps: list[StepProposal] = Field(min_length=1, max_length=8)
    entities: Entities = Field(default_factory=Entities)


class ResumePlan(Contract):
    steps: list[StepProposal] = Field(default_factory=list, max_length=8)


# --- Canonical Server Structures (assigned/bound by server) ---

class Binding(Contract):
    binding_id: str
    tool_name: str
    resource_type: str
    resource_id: str
    field_pointer: str
    unit: Optional[str] = None


class Requirement(Contract):
    requirement_id: str
    kind: Literal["documentary", "operational", "human"]
    summary: str
    constraints: list[str] = Field(default_factory=list)
    bindings: list[Binding] = Field(default_factory=list)


class Fact(Contract):
    fact_id: str
    binding_id: Optional[str] = None
    operation_id: str
    resource_type: str
    resource_id: str
    field_pointer: str
    availability: Literal["available", "missing", "unavailable"]
    value: str | bool | int | float | None = None
    value_type: str = "str"
    unit: Optional[str] = None
    source_kind: Literal["tool", "human", "system"] = "tool"
    content_hash: str = ""
    scope_notice: str = ""


# --- Composition Blocks and Documents ---

class DocumentaryBlock(Contract):
    kind: Literal["documentary"] = "documentary"
    requirement_id: str
    text: str = Field(min_length=1)
    span_ids: list[str] = Field(min_length=1)
    # Cross-source synthesis support only; values still have their own
    # operational block and are rendered by the server.
    fact_ids: list[str] = Field(default_factory=list)


class OperationalBlock(Contract):
    kind: Literal["operational"] = "operational"
    requirement_id: str
    fact_ids: list[str] = Field(min_length=1)


class ContextBlock(Contract):
    kind: Literal["context"] = "context"
    requirement_id: str
    context_ref: str = Field(min_length=1)


class HumanBlock(Contract):
    kind: Literal["human"] = "human"
    requirement_id: str
    observation_id: str = Field(min_length=1)


class PendingBlock(Contract):
    kind: Literal["pending"] = "pending"
    requirement_id: str


AnswerBlock = Union[DocumentaryBlock, OperationalBlock, ContextBlock, HumanBlock, PendingBlock]


class TransportAnswerBlock(Contract):
    """External flat transport block for model generation.
    kind is required in enum (no default), using a flat structure for compatibility.
    """
    kind: Literal["documentary", "operational", "context", "human", "pending"]
    requirement_id: str
    text: Optional[str] = Field(default=None, description="Texto explicativo para requisitos documentais")
    span_ids: Optional[list[str]] = Field(default=None, description="Lista de span_ids citados")
    fact_ids: Optional[list[str]] = Field(default=None, description="Lista de fact_ids observados pelo servidor")
    context_ref: Optional[str] = Field(default=None, description="Ponteiro de contexto informado pelo lojista")
    observation_id: Optional[str] = Field(default=None, description="ID da observação humana homologada")


def canonicalize_answer_block(data: Any) -> AnswerBlock:
    """Converts transport block dictionary or canonical instance into a strictly verified canonical block.
    Rejects incompatible fields, unknown kinds, or fabricated payload types without silent discarding.
    """
    if isinstance(data, (DocumentaryBlock, OperationalBlock, ContextBlock, HumanBlock, PendingBlock)):
        return data
    if not isinstance(data, dict):
        raise ValueError("Bloco de resposta deve ser um dicionário ou bloco canônico.")

    allowed_fields = {"kind", "requirement_id", "text", "span_ids", "fact_ids", "context_ref", "observation_id"}
    extra = set(data.keys()) - allowed_fields
    if extra:
        raise ValueError(f"Campos desconhecidos no bloco de resposta: {extra}")

    kind = data.get("kind")
    req_id = data.get("requirement_id")
    if not req_id or not isinstance(req_id, str) or not req_id.strip():
        raise ValueError("Bloco de resposta sem requirement_id válido.")

    if kind == "documentary":
        text = data.get("text")
        spans = data.get("span_ids")
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Bloco documental requer texto não vazio.")
        if not spans or not isinstance(spans, list) or not all(isinstance(s, str) and s.strip() for s in spans):
            raise ValueError("Bloco documental requer span_ids não vazio.")
        fact_ids = data.get("fact_ids")
        if fact_ids is None:
            fact_ids = []
        if not isinstance(fact_ids, list) or not all(isinstance(fid, str) and fid.strip() for fid in fact_ids):
            raise ValueError("Referências fact_ids documentais devem ser uma lista de IDs não vazios.")
        if data.get("context_ref") or data.get("observation_id"):
            raise ValueError("Bloco documental contém campos humanos/contexto incompatíveis.")
        return DocumentaryBlock(requirement_id=req_id, text=text, span_ids=spans, fact_ids=fact_ids)

    elif kind == "operational":
        facts = data.get("fact_ids")
        if not facts or not isinstance(facts, list) or not all(isinstance(f, str) and f.strip() for f in facts):
            raise ValueError("Bloco operacional requer fact_ids não vazio.")
        if (data.get("text") and str(data["text"]).strip()) or data.get("span_ids") or data.get("context_ref") or data.get("observation_id"):
            raise ValueError("Bloco operacional contém texto livre ou campos incompatíveis; valores são apresentados exclusivamente pelo servidor.")
        return OperationalBlock(requirement_id=req_id, fact_ids=facts)

    elif kind == "context":
        cref = data.get("context_ref")
        if not cref or not isinstance(cref, str) or not cref.strip():
            raise ValueError("Bloco de contexto requer context_ref não vazio.")
        if (data.get("text") and str(data["text"]).strip()) or data.get("span_ids") or data.get("fact_ids") or data.get("observation_id"):
            raise ValueError("Bloco de contexto contém campos incompatíveis.")
        return ContextBlock(requirement_id=req_id, context_ref=cref)

    elif kind == "human":
        if any(data.get(field) not in (None, "", []) for field in ("text", "span_ids", "fact_ids", "context_ref")):
            raise ValueError("Bloco humano contém campos incompatíveis.")
        obs = data.get("observation_id")
        if not obs or not isinstance(obs, str) or not obs.strip():
            return PendingBlock(requirement_id=req_id)
        return HumanBlock(requirement_id=req_id, observation_id=obs)

    elif kind == "pending":
        if any(data.get(field) not in (None, "", []) for field in ("text", "span_ids", "fact_ids", "context_ref", "observation_id")):
            raise ValueError("Bloco pendente contém campos incompatíveis.")
        return PendingBlock(requirement_id=req_id)

    else:
        raise ValueError(f"Tipo de bloco desconhecido ou inválido: {kind}")


class RequirementDecision(Contract):
    requirement_id: str
    status: Literal["supported", "contradicted", "unanswerable"]
    constraint_application_notes: str = ""


class TransportAnswerDocument(Contract):
    requirements: list[RequirementDecision] = Field(min_length=1)
    final_answer: list[TransportAnswerBlock] = Field(min_length=1)


class AnswerDocument(Contract):
    requirements: list[RequirementDecision] = Field(min_length=1)
    final_answer: list[AnswerBlock] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def canonicalize_document(cls, data: Any) -> Any:
        if isinstance(data, dict):
            fa = data.get("final_answer")
            if isinstance(fa, list):
                data = dict(data)
                data["final_answer"] = [canonicalize_answer_block(b) for b in fa]
        return data

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        schema = TransportAnswerDocument.model_json_schema(*args, **kwargs)
        schema["title"] = "AnswerDocument"
        return schema


class EvidenceSpan(Contract):
    span_id: str
    text: str
    source: str
    source_url: str = ""
    kind: Literal["document", "tool", "human"] = "document"
    content_hash: str = ""


def canonicalize_plan(proposal: PlanProposal) -> tuple[list[Requirement], list[StepProposal]]:
    """Converts model proposals into canonical server requirements with stable IDs."""
    if not proposal.requirements:
        raise ValueError("Plano sem requisitos.")
    if not proposal.steps:
        raise ValueError("Plano sem etapas.")

    num_reqs = len(proposal.requirements)
    canonical_reqs: list[Requirement] = []
    for r_idx, r_prop in enumerate(proposal.requirements, start=1):
        req_id = f"r{r_idx:04d}"
        canonical_bindings: list[Binding] = []
        if r_prop.kind == "operational":
            if not r_prop.bindings:
                raise ValueError("Requisito operacional sem binding.")
            for b_idx, b_prop in enumerate(r_prop.bindings, start=1):
                b_id = f"{req_id}:b{b_idx:04d}"
                canonical_bindings.append(
                    Binding(
                        binding_id=b_id,
                        tool_name=b_prop.tool_name,
                        resource_type=b_prop.resource_type,
                        resource_id=b_prop.resource_id,
                        field_pointer=b_prop.field_pointer,
                        unit=b_prop.unit,
                    )
                )
        canonical_reqs.append(
            Requirement(
                requirement_id=req_id,
                kind=r_prop.kind,
                summary=r_prop.summary,
                constraints=list(dict.fromkeys(r_prop.constraints + r_prop.user_constraints_and_negations)),
                bindings=canonical_bindings,
            )
        )

    # Validate step requirement indices
    for s_idx, step in enumerate(proposal.steps):
        indices = step.requirement_indexes
        if not indices:
            raise ValueError(f"Etapa {s_idx} sem referência de requisito.")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Etapa {s_idx} possui índices de requisitos duplicados.")
        for idx in indices:
            if idx < 0 or idx >= num_reqs:
                raise ValueError(f"Etapa {s_idx} referencia índice de requisito inválido: {idx}.")

    return canonical_reqs, proposal.steps


def answer_status(requirements: list[RequirementDecision]) -> Literal["complete", "partial", "abstained"]:
    if not requirements:
        return "abstained"
    unanswerable = sum(r.status == "unanswerable" for r in requirements)
    if unanswerable == len(requirements):
        return "abstained"
    if unanswerable > 0:
        return "partial"
    return "complete"


# Backwards compatibility aliases for existing unit tests
Plan = PlanProposal
PlannedRequirement = RequirementProposal
