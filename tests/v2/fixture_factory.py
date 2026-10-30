"""Prepare input data, never oracle answers, for isolated A/B processes."""
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path


def get_pilot_cases() -> list[dict]:
    """Returns the 5 flows for the Núcleo 2.0 pilot."""
    repo_root = Path(__file__).resolve().parents[2]
    inputs_file = repo_root / "data" / "inputs.json"
    if not inputs_file.exists():
        inputs_file = repo_root / "archive_audits" / "getnet_blind16_2026-10-28" / "inputs.json"
    if not inputs_file.exists():
        inputs_file = repo_root.parent / "getnet_blind16_2026-10-28" / "inputs.json"
    
    blind_map = {}
    if inputs_file.exists():
        for item in json.loads(inputs_file.read_text(encoding="utf-8")):
            blind_map[item["id"]] = item

    # Synthetic script 5 (pilot_hitl_01) as specified in Section 4 of the plan
    hitl_case = {
        "id": "pilot_hitl_01",
        "categoria": "misto_hitl_sintetico",
        "mensagens": [
            {
                "role": "user",
                "content": (
                    "Explique como testar a comunicação deste terminal e peça à equipe humana que "
                    "confira sua conectividade; depois continue este mesmo atendimento. "
                    "Não reinicie nem altere a máquina."
                ),
            }
        ],
        "contexto_premissas": {
            "user_id": "pilot_hitl_01",
            "modo_operacional": "hitl_orientacao_e_retomada",
            "dados_sinteticos": {
                "terminal_id": "POS-5500",
                "status": "pendente_revisao",
                "chip_algar_status": "ativo",
                "compartimento_detectado": "SIM1",
            },
        },
        "operator_response": "Verificação manual concluída: a conectividade do terminal está ativa.",
    }

    selected_ids = ["blind_01", "blind_07", "blind_13", "blind_14"]
    cases = []
    for cid in selected_ids:
        if cid in blind_map:
            cases.append(copy.deepcopy(blind_map[cid]))
        else:
            raise RuntimeError(f"Caso {cid} não encontrado em inputs.json")
    cases.append(hitl_case)
    return cases


def prepare_arms(root: Path, cases: list[dict]) -> dict:
    import os

    # Ensure environment points inside isolated root before importing any store singleton
    root.mkdir(parents=True, exist_ok=True)
    temp_init_db = root / ".bootstrap_state.db"
    os.environ["SQLITE_DB_PATH"] = str(temp_init_db)

    from app.memory.store import StateStore
    from app.hitl.queue import HITLQueueManager
    from app.v2.adapters import StateAdapter

    repo_data = Path(__file__).resolve().parents[2] / "data"
    customers = {}
    notes = []

    for case in cases:
        context = case.get("contexto_premissas", {})
        user = context.get("user_id", case["id"])
        supplied = copy.deepcopy(context.get("dados_sinteticos") or {})
        customer = {"user_id": user, **supplied}

        if supplied.get("terminal_id"):
            customer["terminals"] = [copy.deepcopy(supplied)]
        elif supplied.get("equipamento_atual_conta"):
            equipment = supplied["equipamento_atual_conta"]
            match = re.fullmatch(r"\s*(\d+)\s*x\s+(.+)", equipment, re.I)
            if match:
                customer["terminals"] = [{"model": match[2], "terminal_id": f"POS-{case['id']}"} for _ in range(int(match[1]))]
                notes.append({"id": case["id"], "note": "Quantidade/modelo explícitos mapeados para terminals."})
        customers[user] = customer

    serialized = json.dumps({"customers": customers}, ensure_ascii=False, indent=2) + "\n"
    arms = {}

    for arm in ("A", "B"):
        arm_dir = root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        customer_file = arm_dir / "customers.json"
        sqlite_file = arm_dir / "state.db"
        vector_index_file = arm_dir / "vector_index.json"
        embedding_cache_file = arm_dir / "embeddings_cache.json"

        customer_file.write_text(serialized, encoding="utf-8")

        # Copy baseline vector index and embeddings cache only if not present (preserve mutated cache)
        if not vector_index_file.exists() and (repo_data / "vector_index.json").exists():
            shutil.copy2(repo_data / "vector_index.json", vector_index_file)
        if not embedding_cache_file.exists() and (repo_data / "embeddings_cache.json").exists():
            shutil.copy2(repo_data / "embeddings_cache.json", embedding_cache_file)

        os.environ["SQLITE_DB_PATH"] = str(sqlite_file)
        store = StateStore(sqlite_file)
        state = StateAdapter(store)
        queue = HITLQueueManager(store)

        for case in cases:
            context = case.get("contexto_premissas", {})
            user = context.get("user_id", case["id"])
            sid = "pilot_" + case["id"]
            store.ensure_session(sid, user)

            # Seed conversation history idempotently (never duplicate messages on repeated prepare)
            with store._connect() as conn:
                existing_msgs = conn.execute("SELECT count(*) FROM messages WHERE session_id=?", (sid,)).fetchone()[0]
            if existing_msgs == 0:
                for message in case.get("mensagens", [])[:-1]:
                    if arm == "A":
                        store.append_message(sid, user, message["role"], message["content"])
                    else:
                        state.append(sid, user, message["role"], message["content"])

            data = context.get("dados_sinteticos") or {}
            if arm == "B":
                existing_v2 = state.context(sid, user).get("v2", {})
                if "entities" not in existing_v2:
                    entities = {}
                    if data.get("terminal_id"):
                        entities["terminal_id"] = data["terminal_id"]
                    if data.get("protocolo_aberto"):
                        entities["ticket_id"] = data["protocolo_aberto"]
                    state.save(sid, user, {"entities": entities})

            if data.get("protocolo_aberto"):
                with store._connect() as conn:
                    existing_t = conn.execute("SELECT 1 FROM escalation_tickets WHERE ticket_id=?", (data["protocolo_aberto"],)).fetchone()
                if not existing_t:
                    ticket = queue.create_ticket(
                        user_id=user,
                        session_id=sid,
                        category="access",
                        reason=data.get("motivo", "Validação humana pendente"),
                        diagnostic_snapshot=data,
                    )
                    with store._connect() as conn:
                        conn.execute("UPDATE escalation_tickets SET ticket_id=? WHERE ticket_id=?",
                                     (data["protocolo_aberto"], ticket.ticket_id))

        arms[arm] = {
            "customers": str(customer_file),
            "sqlite": str(sqlite_file),
            "vector_index": str(vector_index_file),
            "embedding_cache": str(embedding_cache_file),
            "customers_sha256": hashlib.sha256(customer_file.read_bytes()).hexdigest(),
        }

    assert arms["A"]["customers_sha256"] == arms["B"]["customers_sha256"]
    layout_data = {
        "arms": arms,
        "notes": notes,
        "scope": "Fixtures sintéticas isoladas para Arm A e Arm B do piloto Getnet v2.",
        "case_ids": [c["id"] for c in cases],
    }
    (root / "layout.json").write_text(json.dumps(layout_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return arms
