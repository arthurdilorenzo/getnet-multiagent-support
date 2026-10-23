"""Handler integration: preserve a verified Pix fact beside an absent POS field."""
import importlib
import json

from fastapi.testclient import TestClient

from app.agents.orchestrator import MultiAgentOrchestrator
from app.config import settings
from app.memory.store import StateStore


def test_chat_keeps_pix_fact_when_requested_firmware_is_absent(tmp_path, monkeypatch):
    orchestrator_module = importlib.import_module("app.agents.orchestrator")
    main_module = importlib.import_module("app.main")
    customer_tools = importlib.import_module("app.tools.customer_tools")
    action_policy = importlib.import_module("app.security.action_policy")
    registry_module = importlib.import_module("app.tools.registry")
    state = StateStore(tmp_path / "mixed-pix-firmware.sqlite")
    for module in (orchestrator_module, main_module, customer_tools, action_policy, registry_module):
        monkeypatch.setattr(module, "state_store", state)

    target = "GET-SYNTH-V10-PARTIAL-1234"
    other = "GET-SYNTH-V10-OTHER-5678"
    fixture = tmp_path / "customers.json"
    fixture.write_text(json.dumps({"customers": {"synthetic": {
        "company_name": "Synthetic",
        "terminals": [
            {"terminal_id": target, "model": "Get Smart", "status": "ONLINE",
             "connection_type": "WIFI", "signal_strength": "3/4", "firmware_version": None},
            {"terminal_id": other, "model": "Get Clássica", "status": "ONLINE",
             "connection_type": "Ethernet", "signal_strength": "4/4",
             "firmware_version": "private-other-terminal-9.99"},
        ],
    }}}))
    monkeypatch.setattr(settings, "MOCK_CUSTOMERS_PATH", fixture)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "AUTH_ENFORCE_STRICT", False)

    message = f"O que é o Pix Getnet; consulte o firmware de {target}."
    orchestrator = MultiAgentOrchestrator()
    monkeypatch.setattr(main_module, "orchestrator", orchestrator)

    response = TestClient(main_module.app).post("/api/chat", json={"message": message, "user_id": "synthetic"})
    assert response.status_code == 200
    body = response.json()

    assert body["answer_status"] == "partial"
    assert body["abstained"] is True
    assert "um dia corrido" in body["response"]
    assert "modelo de Pix" in body["response"]
    assert target in body["response"]
    assert "firmware" in body["response"].lower()
    assert "private-other-terminal-9.99" not in body["response"]
    assert other not in body["response"]

    ledger = body["request_coverage"]
    assert [(row["unit_id"], row["state"]) for row in ledger] == [("u1", "fulfilled"), ("u2", "unresolved")]
    assert body["evidence"]
    evidence_json = json.dumps(body["evidence"])
    assert target in evidence_json
    assert other not in evidence_json
    assert "private-other-terminal-9.99" not in evidence_json
    assert not any(item.get("fact_path") == "terminal.firmware_version" for item in body["evidence"])
    assert not body["action_proposals"]
