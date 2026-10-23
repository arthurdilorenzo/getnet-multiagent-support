"""Offline contract tests: actual GenAI SDK types, simulated provider transport only."""
from copy import deepcopy

import pytest
from google.genai import types
from pydantic import BaseModel

from app.agents.customer_support_agent import CustomerSupportAgent
from app.agents.router_agent import RouterAgent
from app.config import settings
from app.memory.store import StateStore
from app.schemas import RoutingPlan
from app.security.auth import SecurityContext
from app.tools.registry import ToolRegistry, tool_registry


def completion(*calls, text=None, signature=None):
    parts = [types.Part(function_call=types.FunctionCall(name=name, args=args, id=f"call-{i}"))
             for i, (name, args) in enumerate(calls)]
    if signature and parts:
        parts[0].thought_signature = signature
    if text is not None:
        parts.append(types.Part.from_text(text=text))
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts), finish_reason="STOP"
    )])


class Provider:
    """Only replace the external SDK invocation; registry/handlers remain real."""
    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []
        self.models = self

    def generate_content(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if not self.steps:
            raise RuntimeError("Unexpected extra model turn")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(kwargs) if callable(step) else step


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key-never-sent")
    from app.tools import customer_tools
    import importlib
    from app.security import action_policy
    store = StateStore(tmp_path / "state.db")
    monkeypatch.setattr(customer_tools, "state_store", store)
    monkeypatch.setattr(action_policy, "state_store", store)
    monkeypatch.setattr(importlib.import_module("app.tools.registry"), "state_store", store)
    from app.observability.resilience import breaker_registry
    breaker_registry.reset_all()
    yield
    breaker_registry.reset_all()


def ctx(user_id="cliente1988", **kwargs):
    return SecurityContext(authenticated_user_id=user_id, **kwargs)


def declarations(request):
    config = types.GenerateContentConfig.model_validate(request["config"])
    assert config.automatic_function_calling.disable is True
    assert config.tools and config.tools[0].function_declarations
    return {d.name: d for d in config.tools[0].function_declarations}


def function_results(request):
    return [p.function_response for c in request["contents"] for p in c.parts or []
            if p.function_response]


def test_router_uses_native_delegation_call_without_keyword_preemption():
    provider = Provider(completion(("CUSTOMER_SUPPORT_AGENT", {
        "sub_query": "Examine a ocorrência Q7", "rationale": "É uma consulta privada"})))
    plan = RouterAgent(provider).plan("Ocorrência Q7: pode verificar?", "cliente1988")
    assert isinstance(plan, RoutingPlan)
    assert plan.primary_agent == "CUSTOMER_SUPPORT_AGENT"
    assert plan.plan_steps[0].sub_query == "Examine a ocorrência Q7"
    offered = declarations(provider.requests[0])
    assert set(offered) == {"CUSTOMER_SUPPORT_AGENT", "KNOWLEDGE_AGENT", "WEB_SEARCH_AGENT", "HUMAN_ESCALATION_AGENT"}
    assert "user_id" not in str(offered)
    assert types.GenerateContentConfig.model_validate(provider.requests[0]["config"]).tool_config.function_calling_config.mode == "ANY"


def test_router_validates_ordered_multi_agent_calls():
    provider = Provider(completion(
        ("WEB_SEARCH_AGENT", {"sub_query": "Cotação", "rationale": "Dado público"}),
        ("CUSTOMER_SUPPORT_AGENT", {"sub_query": "Conta", "rationale": "Dado privado"})))
    plan = RouterAgent(provider).plan("Reúna as duas referências solicitadas", "cliente1988")
    assert [s.target_agent for s in plan.plan_steps] == ["WEB_SEARCH_AGENT", "CUSTOMER_SUPPORT_AGENT"]
    assert [s.step_order for s in plan.plan_steps] == [1, 2]
    assert plan.is_multi_step and plan.requires_user_context


@pytest.mark.parametrize("response", [
    completion(("unknown_agent", {"sub_query": "Q7", "rationale": "x"})),
    completion(("KNOWLEDGE_AGENT", {"sub_query": "Q7", "rationale": "x", "user_id": "cliente2045"})),
    completion(("KNOWLEDGE_AGENT", {"sub_query": "Q7"})),
    completion(text='{"primary_agent":"KNOWLEDGE_AGENT"}'),
    completion(*[("KNOWLEDGE_AGENT", {"sub_query": "Q7", "rationale": "x"})] * 5),
    RuntimeError("provider down"),
])
def test_router_invalid_or_failed_provider_never_uses_keyword_fallback(response):
    with pytest.raises(Exception) as caught:
        RouterAgent(Provider(response)).plan("Qual a cotação do euro hoje?", "cliente1988")
    assert type(caught.value).__name__ == "RouterPlanningError"


def test_configured_key_without_client_is_controlled_failure():
    with pytest.raises(Exception) as caught:
        RouterAgent(None).plan("Minha maquininha não conecta", "cliente1988")
    assert type(caught.value).__name__ == "RouterPlanningError"
    out = CustomerSupportAgent(None).run("Minha maquininha não conecta", ctx())
    assert out.abstained and not out.tools_called


def test_support_model_selects_real_tools_and_receives_dependent_results():
    def after_profile(request):
        assert request["contents"][-1].role == "user"
        results = function_results(request)
        profile = results[-1].response
        assert results[-1].name == "get_customer_profile"
        assert results[-1].id == "call-0"
        assert profile["success"] is True
        assert profile["data"]["user_id"] == "cliente1988"
        terminal = profile["data"]["terminals_summary"][1]["terminal_id"]
        assert terminal == "GET-CLASSICA-10442"
        model_content = request["contents"][-2]
        assert model_content.parts[0].thought_signature == b"signature-for-round-trip"
        return completion(("diagnose_terminal_issue", {"terminal_id": terminal, "symptom": "connectivity"}))

    def after_diagnosis(request):
        results = function_results(request)
        assert [r.name for r in results] == ["get_customer_profile", "diagnose_terminal_issue"]
        assert results[-1].response["data"]["terminal"]["terminal_id"] == "GET-CLASSICA-10442"
        return completion(text="A ocorrência Q7 foi verificada com os dados retornados.")

    provider = Provider(completion(("get_customer_profile", {}), signature=b"signature-for-round-trip"),
                        after_profile, after_diagnosis)
    out = CustomerSupportAgent(provider).run("Verifique a ocorrência Q7 e prossiga.", ctx(), session_id="sess-Q7")
    assert out.tools_called == ["get_customer_profile", "diagnose_terminal_issue"]
    assert all(r.success for r in out.tool_results)
    assert "GET-CLASSICA-10442" in out.response
    assert "A ocorrência Q7 foi verificada com os dados retornados." not in out.response
    assert out.evidence
    offered = declarations(provider.requests[0])
    assert set(offered) == {"get_sales_settlement", "diagnose_terminal_issue", "get_customer_profile", "execute_terminal_remote_action"}
    for declaration in offered.values():
        schema = declaration.parameters_json_schema or declaration.parameters.model_dump(exclude_none=True)
        assert "user_id" not in schema["properties"] and "session_id" not in schema["properties"]


@pytest.mark.parametrize("name,args,error", [
    ("not_registered", {}, "TOOL_NOT_ALLOWED"),
    ("create_escalation_ticket", {"reason": "LLM decided"}, "TOOL_NOT_ALLOWED"),
    ("get_sales_settlement", {"payment_modality": "bitcoin"}, "INVALID_TOOL_ARGUMENTS"),
    ("get_customer_profile", {"extra_field": "ignored before"}, "INVALID_TOOL_ARGUMENTS"),
    ("get_customer_profile", {"user_id": "cliente2045"}, "CROSS_TENANT_ACCESS_DENIED"),
    ("get_customer_profile", {"session_id": "some-other-session"}, "CROSS_SESSION_ACCESS_DENIED"),
])
def test_support_returns_rejected_tool_result_to_model(name, args, error):
    def observe(request):
        result = function_results(request)[-1].response
        assert result["success"] is False and result["error_code"] == error
        return completion(text="Não foi possível executar a consulta com segurança.")
    out = CustomerSupportAgent(Provider(completion((name, args)), observe)).run("Ocorrência Q7", ctx(), session_id="sess-Q7")
    assert out.abstained
    assert out.tool_results[0].error_code == error
    assert out.structured_data == {}


def test_support_recovers_invalid_arguments_with_model_selected_retry():
    def fix(request):
        assert function_results(request)[-1].response["error_code"] == "INVALID_TOOL_ARGUMENTS"
        return completion(("get_sales_settlement", {"payment_modality": "pix"}))
    provider = Provider(completion(("get_sales_settlement", {"payment_modality": "unknown"})), fix,
                        completion(text="Consulta corrigida."))
    out = CustomerSupportAgent(provider).run("Ocorrência Q7", ctx())
    assert [r.success for r in out.tool_results] == [False, True]
    assert out.structured_data["get_sales_settlement"]["payment_modality_filter"] == "pix"
    # Protocol recovery does not establish that an opaque original was fulfilled.
    assert out.abstained and out.answer_status == 'partial'
    assert out.unresolved_unit_ids == ['u1']


@pytest.mark.parametrize("failure", [RuntimeError("unavailable"), completion(), completion(text="Saldo inventado")])
def test_support_abstains_without_tool_evidence_instead_of_offline_selection(failure):
    out = CustomerSupportAgent(Provider(failure)).run("Quando cai o dinheiro das vendas de ontem?", ctx())
    assert out.abstained and out.tools_called == []
    assert out.response != "Saldo inventado"


def test_support_preserves_hardware_escalation_after_provider_failure():
    provider = Provider(completion(("diagnose_terminal_issue", {"symptom": "hardware"})), RuntimeError("down"))
    out = CustomerSupportAgent(provider).run("Ocorrência Q7", ctx("cliente3012"))
    assert out.abstained and out.requires_escalation
    assert out.structured_data["diagnose_terminal_issue"]["requires_hardware_replacement"] is True


def test_support_enforces_total_call_budget_before_parallel_batch_executes(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOOL_TURNS", 2)
    provider = Provider(completion(*[("get_customer_profile", {})] * 3))
    out = CustomerSupportAgent(provider).run("Ocorrência Q7", ctx())
    assert out.abstained and out.tools_called == []
    assert "limite" in out.abstention_reason.lower()


def test_support_stops_repeated_model_calls_at_turn_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_TOOL_TURNS", 2)
    provider = Provider(*[completion(("get_customer_profile", {})) for _ in range(3)])
    out = CustomerSupportAgent(provider).run("Ocorrência Q7", ctx())
    assert out.abstained
    assert len(out.tool_results) == 2
    assert len(provider.requests) <= 3
    assert "limite" in out.abstention_reason.lower()


@pytest.mark.parametrize("message,action", [
    ("Verifique a ocorrência Q7", "remote_reboot"),
    ("Posso reiniciar remotamente?", "remote_reboot"),
    ("Não reinicie remotamente minha máquina", "remote_reboot"),
    ("Quero trocar a operadora", "remote_reboot"),
])
def test_llm_cannot_authorize_its_own_mutation(message, action):
    provider = Provider(completion(("execute_terminal_remote_action", {
        "terminal_id": "GET-SMART-88219", "action_type": action})), completion(text="Sem comando autorizado."))
    out = CustomerSupportAgent(provider).run(message, ctx())
    assert out.abstained and out.tool_results[0].error_code == "ACTION_NOT_AUTHORIZED"
    assert out.structured_data == {}


def test_original_message_controls_authorization_even_if_router_subquery_invents_order():
    provider = Provider(completion(("execute_terminal_remote_action", {
        "terminal_id": "GET-SMART-88219", "action_type": "remote_reboot"})), completion(text="Sem autorização."))
    out = CustomerSupportAgent(provider).run("Reinicie remotamente minha máquina", ctx(),
                                             authorization_message="Verifique a ocorrência Q7")
    assert out.abstained and out.tool_results[0].error_code == "ACTION_NOT_AUTHORIZED"


def test_explicit_action_request_proposes_via_registry_and_result_returns_to_model():
    def observe(request):
        result = function_results(request)[-1].response
        assert result["success"]
        assert result["data"]["status"] == "PENDING_CONFIRMATION"
        return completion(text="Comando agendado no terminal.")  # Untrusted false claim must be ignored.
    provider = Provider(completion(("execute_terminal_remote_action", {
        "terminal_id": "GET-SMART-88219", "action_type": "remote_reboot"})), observe)
    out = CustomerSupportAgent(provider).run("Reinicie remotamente minha máquina", ctx())
    assert out.abstained and out.answer_status == 'partial'
    assert not out.covered_unit_ids
    assert out.structured_data["execute_terminal_remote_action"]["proposal_id"].startswith("PROP-")
    assert out.structured_data["execute_terminal_remote_action"]["status"] == 'PENDING_CONFIRMATION'
    assert "Comando agendado no terminal." not in out.response


def test_offline_planning_runs_only_without_configured_key(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    provider = Provider(RuntimeError("must never be used"))
    plan = RouterAgent(provider).plan("Quando cai o dinheiro das vendas de ontem?", "cliente1988")
    out = CustomerSupportAgent(provider).run("Quando cai o dinheiro das vendas de ontem?", ctx())
    assert plan.primary_agent == "CUSTOMER_SUPPORT_AGENT"
    assert out.tools_called == ["get_sales_settlement"]
    assert not provider.requests


def test_registry_binds_session_and_rejects_identity_override_even_for_operator():
    class Input(BaseModel):
        user_id: str
        session_id: str = ""
    registry = ToolRegistry()
    registry.register("probe", "probe", Input, lambda **args: args, requires_tenant_context=True)
    assert registry.execute("probe", {"user_id": "cliente2045"}, ctx(role="operator"), session_id="s1").error_code == "CROSS_TENANT_ACCESS_DENIED"
    assert registry.execute("probe", {"session_id": "s2"}, ctx(), session_id="s1").error_code == "CROSS_SESSION_ACCESS_DENIED"
    result = registry.execute("probe", {}, ctx(), session_id="s1")
    assert result.data == {"user_id": "cliente1988", "session_id": "s1"}


@pytest.mark.parametrize("args", [[], "not a mapping", {"user_id": 123}, {"undeclared": True}])
def test_registry_rejects_malformed_or_undeclared_arguments_without_handler(args):
    result = tool_registry.execute("get_customer_profile", args, ctx())
    assert not result.success
    assert result.error_code in {"INVALID_TOOL_ARGUMENTS", "CROSS_TENANT_ACCESS_DENIED"}


def test_registry_namespaces_write_idempotency_keys_per_tenant_and_session():
    class Input(BaseModel):
        user_id: str
        idempotency_key: str
    registry = ToolRegistry()
    registry.register("write_probe", "probe", Input, lambda **args: args,
                      requires_tenant_context=True, is_write_action=True)
    a = registry.execute("write_probe", {"idempotency_key": "same-key"}, ctx(), session_id="s1")
    b = registry.execute("write_probe", {"idempotency_key": "same-key"}, ctx("cliente2045"), session_id="s1")
    c = registry.execute("write_probe", {"idempotency_key": "same-key"}, ctx(), session_id="s2")
    replay = registry.execute("write_probe", {"idempotency_key": "same-key"}, ctx(), session_id="s1")
    assert a.success and b.success and c.success
    assert len({a.data["idempotency_key"], b.data["idempotency_key"], c.data["idempotency_key"]}) == 3
    assert a.data["idempotency_key"] == replay.data["idempotency_key"]


@pytest.mark.parametrize("action", ["remote_reboot", "sim_carrier_switch"])
def test_request_to_assistant_with_question_mark_authorizes_requested_actions(action):
    def observe(request):
        assert function_results(request)[-1].response["success"] is True
        return completion(text="Comando solicitado agendado.")
    provider = Provider(completion(("execute_terminal_remote_action", {
        "terminal_id": "GET-SMART-88219", "action_type": action})), observe)
    out = CustomerSupportAgent(provider).run(
        "Minha maquininha não conecta na internet, pode reiniciar remotamente e trocar operadora?", ctx()
    )
    assert out.abstained and out.answer_status == 'partial'
    assert not out.covered_unit_ids
    assert out.tool_results[0].success


def test_provider_receives_redacted_message_and_session_history():
    message = "Ocorrência Q7, CPF 123.456.789-01 e senha: secretExample9"
    context = {"recent_messages": [{"role": "user", "content": "Meu CNPJ 12.345.678/0001-99"}]}
    router_provider = Provider(completion(("CUSTOMER_SUPPORT_AGENT", {
        "sub_query": "Q7", "rationale": "Dados operacionais"})))
    RouterAgent(router_provider).plan(message, "cliente1988", conversation_context=context)
    support_provider = Provider(completion(("get_customer_profile", {})), completion(text="Consulta feita."))
    CustomerSupportAgent(support_provider).run(message, ctx(), conversation_context=context)
    for provider in (router_provider, support_provider):
        sent = str(provider.requests[0]["contents"])
        assert "123.456.789-01" not in sent
        assert "secretExample9" not in sent
        assert "12.345.678/0001-99" not in sent


@pytest.mark.parametrize("message", ["Don't reboot my terminal", "Do not reboot my terminal"])
def test_english_negation_cannot_authorize_remote_action(message):
    provider = Provider(completion(("execute_terminal_remote_action", {
        "terminal_id": "GET-SMART-88219", "action_type": "remote_reboot"})), completion(text="Negação respeitada."))
    out = CustomerSupportAgent(provider).run(message, ctx())
    assert out.abstained
    assert out.tool_results[0].error_code == "ACTION_NOT_AUTHORIZED"


def test_explicit_terminal_authorization_does_not_allow_other_own_terminal():
    provider = Provider(completion(("execute_terminal_remote_action", {
        "terminal_id": "GET-SMART-88219", "action_type": "remote_reboot"})), completion(text="Sem autorização para esse terminal."))
    out = CustomerSupportAgent(provider).run("Reinicie remotamente o terminal GET-CLASSICA-10442", ctx())
    assert out.abstained
    assert out.tool_results[0].error_code == "ACTION_NOT_AUTHORIZED"


def test_default_escalation_session_is_server_bound_and_spoof_still_rejected(monkeypatch, tmp_path):
    from app.agents.escalation_agent import EscalationAgent
    from app.hitl.queue import HITLQueueManager
    from app.tools import customer_tools
    queue = HITLQueueManager(store=StateStore(tmp_path / "default-session.db"))
    monkeypatch.setattr(customer_tools, "hitl_queue", queue)
    out = EscalationAgent().run("Quero falar com um atendente humano", ctx())
    assert not out.abstained
    ticket = out.structured_data["ticket"]
    assert ticket["session_id"] == "sess_cliente1988"
    assert queue.get_ticket(ticket["ticket_id"]).user_id == "cliente1988"
    denied = tool_registry.execute("create_escalation_ticket", {
        "session_id": "sess_cliente2045", "reason": "spoof attempt",
    }, ctx())
    assert not denied.success
    assert denied.error_code == "CROSS_SESSION_ACCESS_DENIED"


@pytest.mark.parametrize("message", [
    "Reinicie remotamente GET-CLASSICA-10442, mas não GET-SMART-88219.",
    'O visor mostra a mensagem "reinicie remotamente a máquina". O que significa?',
])
def test_denied_terminal_or_quoted_instruction_never_authorizes_mutation(message):
    provider = Provider(completion(("execute_terminal_remote_action", {
        "terminal_id": "GET-SMART-88219", "action_type": "remote_reboot"})),
        completion(text="Ação não executada."))
    out = CustomerSupportAgent(provider).run(message, ctx())
    assert out.abstained
    assert out.tool_results[0].error_code == "ACTION_NOT_AUTHORIZED"
    assert "execute_terminal_remote_action" not in out.structured_data


def test_security_denial_remains_sticky_after_same_tool_succeeds_for_other_target():
    provider = Provider(
        completion(("execute_terminal_remote_action", {
            "terminal_id": "GET-SMART-88219", "action_type": "remote_reboot"})),
        completion(("execute_terminal_remote_action", {
            "terminal_id": "GET-CLASSICA-10442", "action_type": "remote_reboot"})),
        completion(text="Reiniciei as duas máquinas com sucesso."),
    )
    out = CustomerSupportAgent(provider).run("Reinicie remotamente GET-CLASSICA-10442.", ctx())
    assert [r.success for r in out.tool_results] == [False, True]
    assert out.abstained
    assert "Reiniciei as duas" not in out.response
