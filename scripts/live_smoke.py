"""Opt-in real Gemini smoke; synthetic data, ephemeral state, no secret output.

Run explicitly with GETNET_LIVE_TESTS=1. This is integration evidence, not a V4
score or a replacement for a heldout semantic audit. Provider usage may be billed.
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path
import secrets
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GETNET_LIVE_TESTS") != "1":
        raise SystemExit("Explicit GETNET_LIVE_TESTS=1 is required")
    with tempfile.TemporaryDirectory(prefix="getnet-live-") as directory:
        os.environ.update(SQLITE_DB_PATH=str(Path(directory) / "state.db"),
                          VECTOR_INDEX_PATH=str(Path(directory) / "vectors.json"),
                          JWT_SECRET=secrets.token_urlsafe(48), AUTH_API_KEY=secrets.token_urlsafe(48))
        from app.config import settings
        if not settings.GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is not configured")
        from app.main import app
        from app.agents.orchestrator import orchestrator
        from app.security.auth import AuthManager
        from fastapi.testclient import TestClient
        logging.getLogger("getnet.multiagent").disabled = True
        captured = []
        provider_diagnostics = []
        original_models = orchestrator.gemini_client.models

        class ObservedModels:
            def generate_content(self, **kwargs):
                captured.append(str(kwargs.get("contents", "")))
                generation_started = time.monotonic()
                try:
                    result = original_models.generate_content(**kwargs)
                    provider_diagnostics.append({"status": "responded", "finish_reasons":
                        [str(c.finish_reason) for c in result.candidates or []],
                        "duration_seconds": round(time.monotonic() - generation_started, 3),
                        "thoughts_tokens": getattr(result.usage_metadata, "thoughts_token_count", None),
                        "function_names": [p.function_call.name for c in result.candidates or []
                                           for p in (c.content.parts if c.content else []) or [] if p.function_call]})
                    return result
                except Exception as exc:
                    provider_diagnostics.append({"status": "error", "error_type": type(exc).__name__,
                        "duration_seconds": round(time.monotonic() - generation_started, 3),
                        "code": getattr(exc, "code", None),
                        "message": str(getattr(exc, "message", ""))[:1500].replace(settings.GEMINI_API_KEY, "[REDACTED]")})
                    raise

        class ObservedClient:
            models = ObservedModels()

        proxy = ObservedClient()
        for agent in (orchestrator.router, orchestrator.customer_support_agent, orchestrator.knowledge_agent):
            agent.client = proxy
        cases = [
            ("settlement", "cliente1988", "When will the money from yesterday's sales be deposited?", "get_sales_settlement"),
            ("knowledge", "cliente1988", "Como funciona a antecipação de recebíveis com a Getnet?", "search_getnet_knowledge"),
            ("missing_coverage", "cliente1988", "A Getnet aceita Pix na Argentina?", "search_getnet_knowledge"),
            ("action_proposal", "cliente1988", "Reinicie remotamente meu terminal GET-SMART-88219 agora.", "execute_terminal_remote_action"),
            ("pii", "cliente1988", "Meu CPF é 12345678901 e minha senha é Abacaxi987; meu CVV é 678. Consulte meu perfil.", "get_customer_profile"),
            ("hardware_handoff", "cliente3012", "Minha maquininha está com defeito no leitor de chip, o que está acontecendo?", "diagnose_terminal_issue"),
        ]
        results = []
        client = TestClient(app)
        for name, user, message, expected_tool in cases:
            headers = {"Authorization": "Bearer " + AuthManager.create_jwt_token(user)}
            start = time.monotonic()
            try:
                response = client.post("/api/chat", headers=headers,
                                       json={"user_id": user, "session_id": "live-" + name, "message": message})
                data = response.json()
                passed = response.status_code == 200 and expected_tool in data.get("tools_called", [])
                if name == "missing_coverage":
                    passed &= data.get("abstained") is True
                elif name == "action_proposal":
                    proposals = data.get("action_proposals", [])
                    passed &= bool(proposals) and proposals[0].get("status") == "PENDING_CONFIRMATION"
                    if proposals:
                        proposal = proposals[0]
                        confirmation = client.post(f"/api/actions/{proposal['proposal_id']}/confirm", headers=headers,
                            json={"user_id": user, "expected_payload_hash": proposal["payload_hash"], "confirmation": "CONFIRM"})
                        passed &= confirmation.status_code == 200 and confirmation.json().get("details", {}).get("simulated") is True
                elif name == "hardware_handoff":
                    passed &= data.get("escalated") is True and "create_escalation_ticket" in data.get("tools_called", [])
                else:
                    passed &= data.get("abstained") is False
                if name == "pii":
                    history = client.get(f"/api/sessions/live-{name}?user_id={user}", headers=headers).json()
                    raw = json.dumps(history, ensure_ascii=False) + "\n".join(captured)
                    privacy = {"cpf_absent": "12345678901" not in raw,
                               "password_absent": "Abacaxi987" not in raw,
                               "labeled_cvv_absent": not bool(re.search(r"(?i)\bcvv\s*(?:é|is|[:=])\s*678\b", raw))}
                    passed &= all(privacy.values())
                results.append({"case": name, "passed": bool(passed), "http_status": response.status_code,
                                "agent": data.get("agent_used"), "tools": data.get("tools_called"),
                                "abstained": data.get("abstained"), "escalated": data.get("escalated"),
                                "semantic_mode": data.get("semantic_mode"), "answer_status": data.get("answer_status"),
                                "token_usage": data.get("token_usage"), "trace": data.get("trace", []),
                                "response": data.get("response"), "duration_seconds": round(time.monotonic() - start, 3),
                                **({"privacy_checks":privacy} if name=="pii" else {})})
            except Exception as exc:
                results.append({"case": name, "passed": False, "error_type": type(exc).__name__})
            sys.stdout.write(json.dumps({"case": name, "passed": results[-1]["passed"]}) + "\n")
            sys.stdout.flush()
        import subprocess
        root = Path(__file__).resolve().parents[1]
        report = {"mode": "real_gemini_synthetic_inprocess_http", "model": settings.GEMINI_MODEL,
                  "code_sha": subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                  "code_dirty": bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)),
                  "provider_calls": len(captured), "provider_diagnostics": provider_diagnostics, "cases": results,
                  "passed": all(case["passed"] for case in results)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
