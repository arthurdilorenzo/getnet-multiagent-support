import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.schemas import ToolExecutionResult
from app.security.auth import SecurityContext
from .fixture_factory import prepare_arms
from .test_runtime import ScriptedModel, build, env, result


def test_fixture_factory_preserves_extras_and_produces_equivalent_sources(tmp_path):
    cases = [
        {
            "id": "arbitrary",
            "mensagens": [{"role": "user", "content": "Veja meu terminal."}],
            "contexto_premissas": {
                "user_id": "account",
                "dados_sinteticos": {
                    "terminal_id": "POS-6204",
                    "chip_algar_status": "ativo",
                    "compartimento_detectado": "SIM1",
                },
            },
        }
    ]
    arms = prepare_arms(tmp_path, cases)
    a = json.loads(Path(arms["A"]["customers"]).read_text())
    b = json.loads(Path(arms["B"]["customers"]).read_text())
    assert a == b
    terminal = a["customers"]["account"]["terminals"][0]
    assert terminal["terminal_id"] == "POS-6204" and terminal["chip_algar_status"] == "ativo"
    assert "firmware_version" not in terminal and "status" not in terminal
    assert Path(arms["A"]["sqlite"]) != Path(arms["B"]["sqlite"])


def test_live_model_failure_abstains_and_persists_without_false_completion(env):
    model = Mock()
    model.generate.side_effect = TimeoutError("provider unavailable")
    response = build(env, model).process("Consulte meu cadastro.", SecurityContext("merchant-a"), "session")
    assert response.answer_status == "abstained"
    assert env[1].context("session", "merchant-a")["recent_messages"][-1]["role"] == "assistant"


def test_resume_waits_for_other_human_ticket_without_model_call(env):
    model = ScriptedModel(
        {
            "requirements": [{"requirement_id": "req_1", "requirement_summary": "Acesso"}],
            "steps": [{"kind": "human", "reason": "Acesso", "requirement_indexes": [0]}],
            "entities": {},
        }
    )
    runtime = build(env, model)
    pending = runtime.process("Peça validação humana.", SecurityContext("merchant-a"), "session")
    env[2].create_ticket(user_id="merchant-a", session_id="session", category="other", reason="Outra validação")
    env[2].reply_to_ticket(pending.ticket_id, "operator", "Primeiro validado", resolve_ticket=True)
    response = runtime.resume(pending.ticket_id, SecurityContext("operator", role="operator", scopes=["hitl:manage"]))
    assert response.answer_status != "complete" and len(model.payloads) == 1
    assert env[1].context("session", "merchant-a")["status"] == "waiting_human"


def test_foreign_ticket_never_becomes_tool_evidence(env):
    store, state, queue, gateway = env
    state.append("other-session", "other", "user", "Ajuda")
    foreign = queue.create_ticket(user_id="other", session_id="other-session", category="access", reason="PRIVATE DATA")
    state.append("session", "merchant-a", "user", "Consulta")
    response = gateway.execute("check_escalation_ticket", {"ticket_id": foreign.ticket_id}, SecurityContext("merchant-a"), "session")
    assert response.data["found"] is False
    assert "PRIVATE DATA" not in json.dumps(response.data)


def test_unknown_tool_outcome_is_not_retried_or_complete(env, monkeypatch):
    model = ScriptedModel(
        {
            "requirements": [{"requirement_id": "req_1", "requirement_summary": "Perfil"}],
            "steps": [{"kind": "tool", "tool_name": "get_customer_profile", "requirement_indexes": [0]}],
            "entities": {},
        }
    )
    runtime = build(env, model)
    execute = Mock(
        return_value=ToolExecutionResult(
            tool_name="get_customer_profile",
            success=False,
            outcome_state="unknown",
            error_code="UNCERTAIN",
        )
    )
    monkeypatch.setattr(runtime.gateway, "execute", execute)
    response = runtime.process("Consulte o perfil.", SecurityContext("merchant-a"), "session")
    assert response.answer_status == "abstained" and execute.call_count == 1
