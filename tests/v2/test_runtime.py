"""Runtime integration and isolated security tests for Getnet Núcleo 2.0 consolidation."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager
from app.security.auth import AuthManager, CrossTenantAccessError, SecurityContext
from app.v2.adapters import StateAdapter, ToolGateway
from app.v2.api import create_app
from app.v2.contracts import (
    AnswerDocument,
    BindingProposal,
    DocumentaryBlock,
    Entities,
    HumanBlock,
    OperationalBlock,
    PendingBlock,
    PlanProposal,
    RequirementDecision,
    RequirementProposal,
    ResumePlan,
    StepProposal,
)
from app.v2.knowledge import KnowledgeService
from app.v2.runtime import Runtime
from tests.v2.fixture_factory import prepare_arms


class ScriptedModel:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.payloads = []

    def generate(self, schema, payload, *, phase, tracer):
        self.payloads.append((phase, payload))
        output = self.outputs.pop(0)
        if callable(output):
            return output(payload)
        if isinstance(output, schema):
            return output
        if isinstance(output, dict):
            return schema.model_validate(output)
        return output


def default_result(payload):
    spans = payload.get("evidence_spans", [])
    facts = payload.get("observed_facts", [])
    reqs = payload.get("planned_requirements") or [{"requirement_id": "r0001", "kind": "documentary"}]
    rid = reqs[0]["requirement_id"]
    kind = reqs[0].get("kind", "documentary")

    if kind == "documentary" and spans:
        blocks = [DocumentaryBlock(requirement_id=rid, text="Consulta documentada com sucesso.", span_ids=[spans[0]["span_id"]])]
        status = "supported"
    elif kind == "operational" and facts:
        blocks = [OperationalBlock(requirement_id=rid, fact_ids=[facts[0]["fact_id"]])]
        status = "supported"
    else:
        blocks = [PendingBlock(requirement_id=rid)]
        status = "unanswerable"

    return AnswerDocument(
        requirements=[RequirementDecision(requirement_id=rid, status=status)],
        final_answer=blocks,
    )


result = default_result


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = StateStore(tmp_path / "state.db")
    # The new state adapter must not use legacy semantic extraction.
    monkeypatch.setattr(store, "_extract_user_referents", Mock(side_effect=AssertionError("legacy extractor")))
    for module in ("app.tools.registry", "app.security.action_policy"):
        monkeypatch.setattr(module + ".state_store", store)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(settings, "HITL_WEBHOOK_URL", "")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    path = tmp_path / "customers.json"
    path.write_text(json.dumps({
        "customers": {
            "merchant-a": {
                "company_name": "Loja Lume",
                "necessidade_registrada_em_perfil": "Estoque no balcão",
                "terminals": [
                    {
                        "terminal_id": "POS-8417",
                        "status": "ativo",
                        "battery_percent": 0,
                        "chip_algar_status": "ativo_multioperadora",
                        "compartimento_detectado": "SIM1",
                    }
                ],
                "sales_settlement": {
                    "yesterday_sales": {
                        "date": "2026-10-20",
                        "net_payout_scheduled_today_brl": 125.0,
                    }
                },
            }
        }
    }), encoding="utf-8")
    adapter = StateAdapter(store)
    queue = HITLQueueManager(store=store)
    gateway = ToolGateway(path, store=store, queue=queue)
    return store, adapter, queue, gateway


def build(env, model):
    _, adapter, queue, gateway = env
    retrieval = lambda query: {
        "candidate_chunks": [
            {
                "chunk_id": "methods",
                "content": "O produto aceita débito, crédito, Pix, Wallet e boleto.",
                "source": "manual",
                "source_url": "https://site.getnet.com.br/",
            }
        ],
        "retrieval_mode": "sparse_only",
    }
    return Runtime(model, KnowledgeService(model, retrieval), adapter, gateway, queue)


def test_context_and_extra_attributes_survive_without_legacy_extractor(env):
    store, state, queue, gateway = env
    sec = SecurityContext("merchant-a", scopes=["chat:write", "customer:read_own"])
    state.append("session", "merchant-a", "user", "Confira meu POS-8417.")
    state.save("session", "merchant-a", {"entities": {"terminal_id": "POS-8417", "product_discussed": "Modelo Lume"}})
    assert state.context("session", "merchant-a")["v2"]["entities"]["terminal_id"] == "POS-8417"
    profile = gateway.execute("get_customer_profile", {}, sec, "session")
    assert "Estoque" in json.dumps(profile.data, ensure_ascii=False)
    diagnostic = gateway.execute("diagnose_terminal_issue", {"terminal_id": "POS-8417"}, sec, "session")
    assert diagnostic.data["terminal"]["battery_percent"] == 0
    assert diagnostic.data["terminal"]["compartimento_detectado"] == "SIM1"
    denied = gateway.execute("get_customer_profile", {"user_id": "someone-else"}, sec, "session")
    assert not denied.success and denied.error_code == "CROSS_TENANT_ACCESS_DENIED"


def test_ticket_lookup_without_prefix_or_id_uses_existing_queue(env):
    store, state, queue, gateway = env
    state.append("session", "merchant-a", "user", "Consulte meu chamado.")
    ticket = queue.create_ticket(session_id="session", user_id="merchant-a", category="access", reason="Acesso pendente")
    response = gateway.execute("check_escalation_ticket", {}, SecurityContext("merchant-a", scopes=["chat:write", "customer:read_own"]), "session")
    assert response.success and response.data["tickets"][0]["ticket_id"] == ticket.ticket_id


def test_runtime_never_enters_legacy_semantic_gates(env, monkeypatch):
    for path in [
        "app.security.request_units.request_units",
        "app.security.fulfillment.evaluate_operational_unit",
        "app.security.request_constraints.constrain_call",
        "app.security.public_projection.project_specialist",
        "app.security.public_projection.project_public_response",
    ]:
        monkeypatch.setattr(path, Mock(side_effect=AssertionError("legacy gate reached")))

    plan_prop = PlanProposal(
        requirements=[
            RequirementProposal(
                kind="operational",
                summary="Consultar terminal",
                bindings=[
                    BindingProposal(
                        tool_name="diagnose_terminal_issue",
                        resource_type="terminal",
                        resource_id="POS-8417",
                        field_pointer="/terminal/status",
                    )
                ],
            ),
            RequirementProposal(
                kind="documentary",
                summary="Explicar meios",
            ),
        ],
        steps=[
            StepProposal(kind="tool", tool_name="diagnose_terminal_issue", arguments={"terminal_id": "POS-8417"}, requirement_indexes=[0]),
            StepProposal(kind="knowledge", query="Produto Lume", requirement_indexes=[1]),
        ],
        entities=Entities(terminal_id="POS-8417"),
    )

    def result_func(payload):
        facts = payload["observed_facts"]
        spans = payload["evidence_spans"]
        doc_spans = [s for s in spans if s.get("kind") == "document"]
        return AnswerDocument(
            requirements=[
                RequirementDecision(requirement_id="r0001", status="supported"),
                RequirementDecision(requirement_id="r0002", status="supported"),
            ],
            final_answer=[
                OperationalBlock(requirement_id="r0001", fact_ids=[facts[0]["fact_id"]]),
                DocumentaryBlock(requirement_id="r0002", text="Aceita débito, crédito e Pix.", span_ids=[doc_spans[0]["span_id"]]),
            ],
        )

    model = ScriptedModel(plan_prop, result_func)
    response = build(env, model).process(
        "Veja no cadastro se meu POS-8417 está ativo e explique os meios.",
        SecurityContext("merchant-a", scopes=["chat:write", "customer:read_own"]),
        "session",
    )
    assert response.answer_status == "complete"
    assert response.semantic_mode == "native_semantic"
    assert "diagnose_terminal_issue" in response.tools_called
    assert "search_getnet_knowledge" in response.tools_called
    assert len(model.payloads) == 2


def test_human_reply_resumes_same_original_request_and_replay_is_durable(env):
    first_plan = PlanProposal(
        requirements=[RequirementProposal(kind="human", summary="Validação de acesso")],
        steps=[StepProposal(kind="human", reason="Validação de acesso", requirement_indexes=[0])],
        entities=Entities(terminal_id="POS-8417"),
    )
    resumed_plan = ResumePlan(
        steps=[StepProposal(kind="tool", tool_name="get_customer_profile", arguments={}, requirement_indexes=[0])],
    )

    def res_func(payload):
        return AnswerDocument(
            requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
            final_answer=[HumanBlock(requirement_id="r0001", observation_id="hitl_obs_1")],
        )

    model = ScriptedModel(first_plan, resumed_plan, res_func)
    runtime = build(env, model)
    original = "Consulte meu cadastro e peça validação humana do acesso."
    pending = runtime.process(original, SecurityContext("merchant-a", scopes=["chat:write", "customer:read_own"]), "session")
    assert pending.escalated and pending.answer_status != "complete"

    env[2].reply_to_ticket(pending.ticket_id, "operator", "Cadastro validado, prossiga com a consulta.", resolve_ticket=True)
    resumed_response = runtime.resume(pending.ticket_id, SecurityContext("operator", role="operator", scopes=["hitl:manage"]))
    assert resumed_response.answer_status == "complete"
    assert model.payloads[1][1]["original_request"] == original
    assert "Cadastro validado" in model.payloads[1][1]["human_guidance"]
    assert runtime.resume(pending.ticket_id, SecurityContext("operator", role="operator", scopes=["hitl:manage"])).response == resumed_response.response
    assert len(model.payloads) == 3


def test_standalone_api_preserves_authenticated_principal(env):
    plan = PlanProposal(
        requirements=[RequirementProposal(kind="documentary", summary="Meios de pagamento")],
        steps=[StepProposal(kind="knowledge", query="meios", requirement_indexes=[0])],
    )
    def answer_func(payload):
        spans = payload["evidence_spans"]
        return AnswerDocument(
            requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
            final_answer=[DocumentaryBlock(requirement_id="r0001", text="Meios disponíveis.", span_ids=[spans[0]["span_id"]])],
        )

    runtime = build(env, ScriptedModel(plan, answer_func))
    client = TestClient(create_app(runtime))
    assert client.post("/api/v2/chat", json={"message": "Meios?", "user_id": "merchant-a"}).status_code == 401
    token = AuthManager.create_jwt_token("merchant-a")
    headers = {"Authorization": "Bearer " + token}
    denied = client.post("/api/v2/chat", headers=headers, json={"message": "Meios?", "user_id": "other"})
    assert denied.status_code == 403
    response = client.post("/api/v2/chat", headers=headers, json={"message": "Meios?", "user_id": "merchant-a"})
    assert response.status_code == 200 and response.json()["answer_status"] == "complete"
