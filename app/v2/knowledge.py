"""Single-pass grounded composition with server-bound structured facts and blocks.

NO lexical regexes, NO prose currency matching, NO fuzzy parsing of operational numbers.
All operational facts are bound and presented directly by the server from verified tool results.
"""
import hashlib
import json
from decimal import Decimal
from typing import Any, Optional

from app.agents.guardrails import GuardrailManager
from app.security.pii import PIIRedactor
from app.v2.contracts import (
    AnswerBlock,
    AnswerDocument,
    DocumentaryBlock,
    EvidenceSpan,
    Fact,
    HumanBlock,
    OperationalBlock,
    ContextBlock,
    PendingBlock,
    Requirement,
    RequirementDecision,
)


class EvidenceError(ValueError):
    def __init__(self, code_or_msg: str, message: str | None = None):
        if message is not None:
            self.code = code_or_msg
            self.message = message
            super().__init__(f"[{code_or_msg}] {message}")
        else:
            self.code = "EVIDENCE_VALIDATION_ERROR"
            self.message = code_or_msg
            super().__init__(code_or_msg)


def format_fact(fact: Fact) -> str:
    value = _format_fact_value(fact)
    return value + ("\n" + fact.scope_notice if fact.scope_notice else "")


def _format_fact_value(fact: Fact) -> str:
    """Server-side deterministic presentation of verified facts."""
    field_name = fact.field_pointer.strip("/").split("/")[-1]
    if fact.availability != "available":
        return f"{field_name} ({fact.resource_id}): Informação indisponível."

    if fact.resource_type == "currency":
        # An exchange rate is a ratio; currency-money rounding would change it.
        return f"{field_name} ({fact.resource_id}): {fact.value} {fact.unit or ''}".rstrip()

    if fact.unit == "BRL":
        try:
            val_num = float(fact.value)
            formatted = f"{val_num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return f"{field_name} ({fact.resource_id}): BRL {formatted}"
        except (ValueError, TypeError):
            return f"{field_name} ({fact.resource_id}): BRL {fact.value}"

    if fact.unit:
        return f"{field_name} ({fact.resource_id}): {fact.value} {fact.unit}"

    return f"{field_name} ({fact.resource_id}): {fact.value}"


def render_final_answer(
    document: AnswerDocument,
    spans: list[EvidenceSpan],
    facts: list[Fact],
    context: dict,
) -> str:
    """Renders structured answer blocks into the public response string.

    The model writes prose ONLY for documentary blocks.
    Operational values and attributions are presented exclusively by the server.
    """
    facts_by_id = {f.fact_id: f for f in facts}
    spans_by_id = {s.span_id: s for s in spans}
    rendered_sections: list[str] = []

    for block in document.final_answer:
        if isinstance(block, DocumentaryBlock):
            rendered_sections.append(block.text)
            for sid in block.span_ids:
                span = spans_by_id.get(sid)
                if span and sid.startswith("web:") and span.source_url.startswith(("https://", "http://")):
                    rendered_sections.append(f"Fonte: {span.source_url}")

        elif isinstance(block, OperationalBlock):
            presented_lines = []
            for fid in block.fact_ids:
                fact = facts_by_id.get(fid)
                if fact:
                    presented_lines.append(format_fact(fact))
            if presented_lines:
                rendered_sections.append("\n".join(presented_lines))

        elif isinstance(block, ContextBlock):
            rendered_sections.append(f"[Contexto informado pelo lojista] {resolve_user_context(block.context_ref, context)}")

        elif isinstance(block, HumanBlock):
            observation = spans_by_id.get(block.observation_id)
            if not observation:
                human_spans = [s for s in spans_by_id.values() if s.kind == "human"]
                if len(human_spans) == 1:
                    observation = human_spans[0]
            if observation:
                rendered_sections.append(f"[Orientação do Operador — {observation.source}] {observation.text}")

        elif isinstance(block, PendingBlock):
            rendered_sections.append(f"[Pendente de Verificação] A informação para o requisito {block.requirement_id} permanece pendente.")

    return "\n\n".join(s for s in rendered_sections if s)


def resolve_user_context(reference: str, context: dict) -> str:
    if reference == "/original_request" and isinstance(context.get("original_request"), str):
        return context["original_request"]
    parts = reference.split("/")
    if len(parts) == 4 and parts[1] == "recent_messages" and parts[3] == "content" and parts[2].isdigit():
        messages = context.get("recent_messages", [])
        index = int(parts[2])
        if index < len(messages) and messages[index].get("role") == "user":
            return messages[index]["content"]
    raise EvidenceError("UNKNOWN_CONTEXT_REFERENCE", "Referência não aponta para contexto real do usuário.")


def validate_answer(
    document: AnswerDocument,
    spans: list[EvidenceSpan],
    facts: list[Fact],
    planned_requirements: list[Requirement],
    context: dict | None = None,
) -> None:
    """Validates structural and referential integrity against server-bound requirements and facts."""
    if not planned_requirements:
        raise EvidenceError("EMPTY_PLANNED_REQUIREMENTS", "Nenhum requisito planejado fornecido para validação.")

    planned_map = {r.requirement_id: r for r in planned_requirements}
    planned_ids = [r.requirement_id for r in planned_requirements]
    doc_ids = [r.requirement_id for r in document.requirements]

    # 1. Exact requirement coverage
    if len(doc_ids) != len(set(doc_ids)):
        raise EvidenceError("DUPLICATE_REQUIREMENT_ID", "Requisitos duplicados na resposta.")
    if doc_ids != planned_ids:
        raise EvidenceError("INCOMPLETE_REQUIREMENT_COVERAGE", "Requisitos da resposta não cobrem exatamente os requisitos planejados.")

    available_spans = {s.span_id: s for s in spans}
    available_facts = {f.fact_id: f for f in facts}
    frozen_bindings = {binding.binding_id: binding for req in planned_requirements for binding in req.bindings}
    rendered_fact_ids = {fid for block in document.final_answer if isinstance(block, OperationalBlock)
                         for fid in block.fact_ids}

    # Map blocks by requirement_id
    blocks_by_req: dict[str, list[AnswerBlock]] = {rid: [] for rid in planned_ids}
    for block in document.final_answer:
        if block.requirement_id not in blocks_by_req:
            raise EvidenceError("UNKNOWN_REQUIREMENT_REFERENCE", f"Bloco referencia requisito desconhecido: {block.requirement_id}")
        if isinstance(block, ContextBlock):
            resolve_user_context(block.context_ref, context or {})
        if isinstance(block, HumanBlock):
            observation = available_spans.get(block.observation_id)
            if not observation:
                human_spans = [s for s in available_spans.values() if s.kind == "human"]
                if len(human_spans) == 1:
                    observation = human_spans[0]
                    block.observation_id = observation.span_id
            if not observation or observation.kind != "human" or not observation.source.startswith("hitl:"):
                raise EvidenceError("UNKNOWN_HUMAN_REFERENCE", "Observação humana inexistente ou sem atribuição.")
            if hashlib.sha256(observation.text.encode("utf-8")).hexdigest() != observation.content_hash:
                raise EvidenceError("HASH_MISMATCH", "Observação humana divergente.")
        blocks_by_req[block.requirement_id].append(block)

    for req_dec in document.requirements:
        req = planned_map[req_dec.requirement_id]
        blocks = blocks_by_req[req_dec.requirement_id]
        allowed = {
            "documentary": (DocumentaryBlock, ContextBlock, HumanBlock, PendingBlock),
            "operational": (OperationalBlock, ContextBlock, HumanBlock, PendingBlock),
            "human": (HumanBlock, ContextBlock, PendingBlock),
        }
        if any(not isinstance(b, allowed[req.kind]) for b in blocks):
            raise EvidenceError("BLOCK_KIND_MISMATCH", "Tipo de bloco incompatível com o requisito.")

        if req_dec.status in ("supported", "contradicted") and not blocks:
            raise EvidenceError("MISSING_EVIDENCE_SPAN", f"Requisito {req.requirement_id} marcado como {req_dec.status} sem nenhum bloco de evidência.")

        if req.kind == "documentary":
            doc_blocks = [b for b in blocks if isinstance(b, DocumentaryBlock)]
            if req_dec.status in ("supported", "contradicted") and not doc_blocks:
                raise EvidenceError("MISSING_EVIDENCE_SPAN", f"Requisito documental {req.requirement_id} sem bloco documental de suporte.")

            for b in doc_blocks:
                for fid in b.fact_ids:
                    fact = available_facts.get(fid)
                    if fact is None:
                        raise EvidenceError("UNKNOWN_FACT_REFERENCE", "Suporte documental referencia fato inexistente.")
                    binding = frozen_bindings.get(fact.binding_id)
                    if (binding is None or fact.source_kind != "tool"
                            or (binding.resource_type, binding.resource_id, binding.field_pointer, binding.unit)
                            != (fact.resource_type, fact.resource_id, fact.field_pointer, fact.unit)):
                        raise EvidenceError("DOCUMENTARY_FACT_BINDING_MISMATCH", "Suporte documental diverge do binding operacional congelado.")
                    # This bridge is narrowly for qualitative customer-profile
                    # synthesis. Numeric, financial, action and terminal facts
                    # remain exclusively operational server-rendered data.
                    if (binding.tool_name != "get_customer_profile" or binding.resource_type != "customer"
                            or not binding.field_pointer.startswith("/profile/") or binding.unit is not None
                            or fact.value_type != "str" or not isinstance(fact.value, str)):
                        raise EvidenceError("INVALID_DOCUMENTARY_FACT_KIND", "Suporte documental aceita somente texto cadastral disponível do perfil.")
                    if fact.availability != "available" or fact.value is None:
                        raise EvidenceError("UNAVAILABLE_DOCUMENTARY_FACT", "Suporte documental exige fato operacional disponível.")
                    if fid not in rendered_fact_ids:
                        raise EvidenceError("UNRENDERED_DOCUMENTARY_FACT", "Fato de suporte deve ser apresentado também em bloco operacional.")
                for sid in b.span_ids:
                    if sid not in available_spans:
                        raise EvidenceError("UNKNOWN_SPAN_REFERENCE", f"Referência a span inexistente: {sid}")
                    span = available_spans[sid]
                    if span.kind != "document":
                        raise EvidenceError("INVALID_DOCUMENTARY_CITATION", f"Fonte operacional {span.source} usada indevidamente como citação documental.")
                    digest = hashlib.sha256(span.text.encode("utf-8")).hexdigest()
                    if span.content_hash and digest != span.content_hash:
                        raise EvidenceError("HASH_MISMATCH", f"Hash divergente para span documental {sid}.")

        elif req.kind == "operational":
            # Check: Documentary blocks or context blocks cannot satisfy operational bindings
            context_blocks = [b for b in blocks if isinstance(b, ContextBlock)]
            if req_dec.status == "supported" and context_blocks and not any(isinstance(b, OperationalBlock) for b in blocks):
                raise EvidenceError("UNSUPPORTED_OPERATIONAL_BINDING", "Bloco contextual não pode satisfazer binding operacional.")

            op_blocks = [b for b in blocks if isinstance(b, OperationalBlock)]
            if req_dec.status in ("supported", "contradicted") and not op_blocks:
                raise EvidenceError("MISSING_EVIDENCE_SPAN", f"Requisito operacional {req.requirement_id} sem bloco operacional de suporte.")

            for b in op_blocks:
                for fid in b.fact_ids:
                    if fid not in available_facts:
                        raise EvidenceError("UNKNOWN_FACT_REFERENCE", f"Referência a fato inexistente: {fid}")
                    fact = available_facts[fid]

                    # Match fact against bindings of this requirement
                    matching_binding = next((bg for bg in req.bindings if bg.binding_id == fact.binding_id), None)

                    if not matching_binding:
                        raise EvidenceError("OPERATIONAL_BINDING_MISMATCH", f"Fato {fid} não corresponde aos bindings do requisito {req.requirement_id}.")

                    # Resource target check: requested target must match observed target
                    if matching_binding.unit != fact.unit or fact.source_kind != "tool":
                        raise EvidenceError("OPERATIONAL_UNIT_MISMATCH", "Unidade/origem do fato diverge do binding.")
                    if matching_binding.resource_type != fact.resource_type or matching_binding.resource_id != fact.resource_id:
                        raise EvidenceError("OPERATIONAL_RESOURCE_MISMATCH", f"Alvo de recurso divergente: esperado {matching_binding.resource_id}, recebido {fact.resource_id}.")

                    # Field pointer check
                    if matching_binding.field_pointer != fact.field_pointer:
                        raise EvidenceError("OPERATIONAL_FIELD_MISMATCH", f"Campo divergente: esperado {matching_binding.field_pointer}, recebido {fact.field_pointer}.")

            # If supported, all required bindings must be available and not null/missing
            if req_dec.status in ("supported", "contradicted"):
                for bg in req.bindings:
                    # Find fact satisfying this binding
                    satisfying = [
                        available_facts[fid]
                        for b in op_blocks
                        for fid in b.fact_ids
                        if available_facts[fid].binding_id == bg.binding_id
                        and available_facts[fid].resource_type == bg.resource_type
                        and available_facts[fid].resource_id == bg.resource_id
                        and available_facts[fid].field_pointer == bg.field_pointer
                    ]
                    if not satisfying or any(f.availability != "available" for f in satisfying):
                        raise EvidenceError("UNSUPPORTED_OPERATIONAL_BINDING", f"Requisito {req.requirement_id} tem binding {bg.binding_id} ({bg.field_pointer}) indisponível ou ausente; não pode ser 'supported'.")

        elif req.kind == "human":
            human_blocks = [b for b in blocks if isinstance(b, HumanBlock)]
            if req_dec.status in ("supported", "contradicted") and not human_blocks:
                raise EvidenceError("MISSING_EVIDENCE_SPAN", f"Requisito humano {req.requirement_id} sem bloco humano de suporte.")


class KnowledgeService:
    def __init__(self, model, retrieve=None):
        self.model = model
        self.retrieve = retrieve or self._retrieve

    @staticmethod
    def _retrieve(query):
        from app.rag.retriever import retriever
        resolved = retriever.resolve_query_vector(query)
        return retriever.retrieve_candidates(query, resolved_context=resolved)

    def search(self, query: str):
        outcome = self.retrieve(query)
        spans = []
        for chunk in outcome.get("candidate_chunks", []):
            body = str(chunk.get("content", ""))
            safe, _ = GuardrailManager.sanitize_untrusted_content(body)
            safe = PIIRedactor.mask_text(safe)
            digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()
            spans.append(EvidenceSpan(
                span_id="doc:" + digest,
                text=safe,
                source=chunk.get("source", chunk.get("chunk_id", "document")),
                source_url=chunk.get("source_url", ""),
                kind="document",
                content_hash=digest,
            ))
        return spans, outcome.get("retrieval_mode", "unknown")

    def compose(
        self,
        message: str,
        spans: list[EvidenceSpan],
        facts: list[Fact],
        context: dict,
        tracer,
        planned_requirements: list[Requirement],
    ) -> tuple[AnswerDocument, str]:
        unique_spans = {s.span_id: s for s in spans}
        evidence_spans = list(unique_spans.values())

        payload = {
            "original_request": message,
            "context": context,
            "planned_requirements": [r.model_dump() for r in planned_requirements],
            "evidence_spans": [s.model_dump() for s in evidence_spans],
            "observed_facts": [f.model_dump() for f in facts],
        }
        provider_document = self.model.generate(AnswerDocument, payload, phase="knowledge", tracer=tracer)
        document = provider_document.model_copy(deep=True)
        # Status is a model proposal; observed availability belongs to the
        # server. Only normalize fully represented, exactly bound facts. Missing
        # references/blocks, wrong resources/fields/units remain strict errors.
        facts_by_id = {fact.fact_id: fact for fact in facts}
        requirements_by_id = {req.requirement_id: req for req in planned_requirements}
        for decision in document.requirements:
            req = requirements_by_id.get(decision.requirement_id)
            if not req or req.kind != "operational" or decision.status not in {"supported", "contradicted"}:
                continue
            blocks = [block for block in document.final_answer
                      if isinstance(block, OperationalBlock) and block.requirement_id == req.requirement_id]
            selected = [facts_by_id.get(fid) for block in blocks for fid in block.fact_ids]
            if not selected or any(fact is None for fact in selected):
                continue
            represented = []
            valid = True
            for fact in selected:
                binding = next((binding for binding in req.bindings if binding.binding_id == fact.binding_id), None)
                if (binding is None or fact.source_kind != "tool"
                        or (binding.resource_type, binding.resource_id, binding.field_pointer, binding.unit)
                        != (fact.resource_type, fact.resource_id, fact.field_pointer, fact.unit)):
                    valid = False
                    break
                represented.append(binding.binding_id)
            if (valid and req.bindings and all(binding.binding_id in represented for binding in req.bindings)
                    and any(fact.availability != "available" for fact in selected)):
                decision.status = "unanswerable"
                tracer.add_step(agent="V2_SUPERVISOR", action="normalize_operational_availability",
                    details={"requirement_id": req.requirement_id, "status": "unanswerable"})
        context = {**context, "original_request": message}
        validate_answer(document, evidence_spans, facts, planned_requirements, context)
        rendered_text = render_final_answer(document, evidence_spans, facts, context)
        return document, rendered_text


def tool_span(result) -> EvidenceSpan | None:
    if not getattr(result, "success", False) or getattr(result, "outcome_state", "") != "completed":
        return None
    data = result.data or {}
    if data.get("found") is False:
        return None
    text = json.dumps(PIIRedactor.mask_structure(data), ensure_ascii=False, sort_keys=True)
    text, _ = GuardrailManager.sanitize_untrusted_content(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return EvidenceSpan(
        span_id="tool:" + digest,
        text=text,
        source=getattr(result, "tool_name", "tool"),
        kind="tool",
        content_hash=digest,
    )


def human_span(ticket_id: str, operator_id: str, text: str) -> EvidenceSpan:
    safe, _ = GuardrailManager.sanitize_untrusted_content(text)
    safe = PIIRedactor.mask_text(safe)
    digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()
    return EvidenceSpan(
        span_id="human:" + digest,
        text=safe,
        source=f"hitl:{ticket_id}:{operator_id}",
        kind="human",
        content_hash=digest,
    )


def configure_embedding_cache(cache_path) -> None:
    from pathlib import Path
    from app.rag.embeddings import SemanticEmbedder
    with SemanticEmbedder._lock:
        SemanticEmbedder.CACHE_PATH = Path(cache_path)
        SemanticEmbedder._cache_loaded = False
        SemanticEmbedder._ensure_cache_loaded()


def persist_embedding_cache(cache_path=None) -> bool:
    from pathlib import Path
    from app.rag.embeddings import SemanticEmbedder
    target = Path(cache_path) if cache_path else SemanticEmbedder.CACHE_PATH
    with SemanticEmbedder._lock:
        if not SemanticEmbedder._neural_vectors:
            return False
        payload = {
            "model": SemanticEmbedder.EMBEDDING_MODEL,
            "dimension": SemanticEmbedder.DIM,
            "alpha": SemanticEmbedder._alpha,
            "mean_unit": SemanticEmbedder._mean_unit,
            "vectors": SemanticEmbedder._neural_vectors,
        }
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.with_suffix(".tmp")
        temp_file.write_text(raw, encoding="utf-8")
        temp_file.replace(target)
        return True
