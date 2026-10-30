from datetime import datetime, timezone, timedelta
import hashlib
import json
from typing import Any, Dict, List, Optional
import httpx
from app.config import settings
from app.memory.store import StateStore, state_store
from app.observability.logger import logger
from app.schemas import HitlTicketModel
from app.security.pii import PIIRedactor


class HITLTransitionError(ValueError):
    """The persisted ticket state or operator ownership forbids this mutation."""

    code = "HITL_INVALID_TRANSITION"


class HITLQueueManager:
    """
    Gerenciador real de Human-in-the-Loop (HITL) com persistência SQLite,
    máquina de estados de chamado (OPEN -> CLAIMED -> RESPONDED -> RESOLVED),
    notificação externa opcional (Telegram / Webhook) e retomada de sessão pelo agente.
    """

    def __init__(self, store: Optional[StateStore] = None) -> None:
        self.store = store or state_store

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_ticket(
        self,
        user_id: str,
        session_id: str,
        category: str,
        reason: str,
        priority: str = "high",
        diagnostic_snapshot: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> HitlTicketModel:
        ctx = self.store.get_conversation_context(session_id=session_id, user_id=user_id, limit=10)
        scoped_key = None
        if idempotency_key:
            key_material = json.dumps(["hitl", user_id, session_id, idempotency_key], ensure_ascii=False)
            scoped_key = "hitl:" + hashlib.sha256(key_material.encode()).hexdigest()
        history = PIIRedactor.mask_structure(ctx.get("recent_messages", []))
        masked_snapshot = PIIRedactor.mask_structure(diagnostic_snapshot or {})
        masked_reason = PIIRedactor.mask_text(reason)

        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        sla_mins = {"critical": 15, "high": 60, "medium": 120, "low": 240}.get(priority, 60)
        sla_deadline = (now_dt + timedelta(minutes=sla_mins)).isoformat()
        seed = scoped_key or json.dumps([user_id, session_id, now, reason[:40]], ensure_ascii=False)
        short_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24].upper()
        ticket_id = f"GET-ESC-{short_hash}"

        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self.store._assert_owner(owner, user_id)
            existing = conn.execute("SELECT * FROM escalation_tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            if existing:
                return self._row_to_model(existing)
            conn.execute(
                """
                INSERT INTO escalation_tickets (
                    ticket_id, session_id, user_id, priority, category, reason,
                    status, assigned_operator, operator_response,
                    diagnostic_snapshot_json, conversation_history_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'OPEN', NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    session_id,
                    user_id,
                    priority,
                    category,
                    masked_reason,
                    json.dumps(masked_snapshot, ensure_ascii=False),
                    json.dumps(history, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            self.store.set_session_status(session_id, "waiting_human", user_id=user_id, connection=conn)
            conn.commit()

        ticket = HitlTicketModel(
            ticket_id=ticket_id,
            session_id=session_id,
            user_id=user_id,
            priority=priority if priority in ("low", "medium", "high", "critical") else "high",
            category=category,
            reason=masked_reason,
            status="OPEN",
            assigned_operator=None,
            operator_response=None,
            diagnostic_snapshot=masked_snapshot,
            conversation_history=history,
            created_at=now,
            updated_at=now,
            sla_minutes=sla_mins,
            sla_deadline_at=sla_deadline,
            is_overdue=False,
        )

        self._notify_external_channels(ticket)
        logger.info(
            "hitl_ticket_persisted",
            ticket_id=ticket_id,
            session_id=session_id,
            user_id=user_id,
            priority=priority,
            category=category,
        )
        return ticket

    def _notify_external_channels(self, ticket: HitlTicketModel) -> Dict[str, str]:
        """
        Envia alerta real via Telegram Bot API e/ou Webhook HTTP quando configurados.
        Nunca falha o fluxo caso o canal externo não esteja configurado.
        """
        dispatched: Dict[str, str] = {"sqlite_queue": "enqueued"}

        if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
            try:
                tg_url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
                text = (
                    f"[Getnet HITL Alert]\n"
                    f"Ticket: {ticket.ticket_id}\n"
                    f"Cliente: {ticket.user_id}\n"
                    f"Prioridade: {ticket.priority.upper()}\n"
                    f"Categoria: {ticket.category}\n"
                    f"Motivo: {ticket.reason}"
                )
                with httpx.Client(timeout=2.5) as client:
                    resp = client.post(
                        tg_url,
                        json={"chat_id": settings.TELEGRAM_CHAT_ID, "text": text},
                    )
                    dispatched["telegram"] = f"http_{resp.status_code}"
            except Exception as exc:
                logger.warning("hitl_telegram_notification_failed", ticket_id=ticket.ticket_id, error_type=type(exc).__name__)
                dispatched["telegram"] = "failed"

        if settings.HITL_WEBHOOK_URL:
            try:
                with httpx.Client(timeout=2.5) as client:
                    resp = client.post(settings.HITL_WEBHOOK_URL, json=ticket.model_dump())
                    dispatched["webhook"] = f"http_{resp.status_code}"
            except Exception as exc:
                logger.warning("hitl_webhook_notification_failed", ticket_id=ticket.ticket_id, error_type=type(exc).__name__)
                dispatched["webhook"] = "failed"

        return dispatched

    def get_ticket(self, ticket_id: str) -> Optional[HitlTicketModel]:
        with self.store._lock, self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM escalation_tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_model(row, store=self.store)

    def list_tickets(
        self, status_filter: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[HitlTicketModel]:
        query = "SELECT * FROM escalation_tickets WHERE 1=1"
        params: List[Any] = []
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter.upper())
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " ORDER BY created_at DESC"

        with self.store._lock, self.store._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_model(r, store=self.store) for r in rows]

    def claim_ticket(self, ticket_id: str, operator_id: str) -> Optional[HitlTicketModel]:
        now = self._now_iso()
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM escalation_tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if not row:
                return None
            owner = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (row["session_id"],)).fetchone()
            self.store._assert_owner(owner, row["user_id"])
            self._validate_operator_transition(row, operator_id, claiming=True)
            if row["status"] == "CLAIMED":
                return self._row_to_model(row, store=self.store)
            conn.execute(
                """
                UPDATE escalation_tickets
                SET status = 'CLAIMED', assigned_operator = ?, updated_at = ?
                WHERE ticket_id = ?
                """,
                (operator_id, now, ticket_id),
            )
            conn.commit()
        return self.get_ticket(ticket_id)

    def reply_to_ticket(
        self,
        ticket_id: str,
        operator_id: str,
        reply_message: str,
        resolve_ticket: bool = True,
        resume_agent: bool = False,
        orchestrator: Optional[Any] = None,
    ) -> Optional[HitlTicketModel]:
        """
        Registra a resposta de um especialista humano no ticket,
        injeta a resposta no histórico e libera a sessão somente quando todos
        os tickets estiverem resolvidos. Quando resume_agent=True, aciona a
        retomada ativa da sessão pelo orquestrador multi-agente.
        """
        now = self._now_iso()
        new_status = "RESOLVED" if resolve_ticket else "RESPONDED"
        masked_reply = PIIRedactor.mask_text(reply_message)

        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # Read, validate ownership, transition and inject history under one lock/transaction.
            row = conn.execute("SELECT * FROM escalation_tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            if not row:
                return None
            self._validate_operator_transition(row, operator_id)
            ticket = self._row_to_model(row, store=self.store)
            owner = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (ticket.session_id,)).fetchone()
            self.store._assert_owner(owner, ticket.user_id)
            conn.execute(
                """
                UPDATE escalation_tickets
                SET status = ?, assigned_operator = ?, operator_response = ?, updated_at = ?
                WHERE ticket_id = ?
                """,
                (new_status, operator_id, masked_reply, now, ticket_id),
            )
            # Ticket, message, structured resumption facts and session status commit together after owner validation.
            resumption_summary = self._synthesize_agent_resumption(
                ticket_id=ticket_id,
                category=ticket.category,
                reason=ticket.reason,
                operator_id=operator_id,
                masked_reply=masked_reply,
                diagnostic_snapshot=ticket.diagnostic_snapshot,
                resolved=resolve_ticket,
            )
            self.store.append_message(
                session_id=ticket.session_id,
                user_id=ticket.user_id,
                role="human_operator",
                content=f"[Operador Humano {operator_id} - Ticket {ticket_id}]: {masked_reply}",
                agent_used=f"HUMAN_OPERATOR:{operator_id}",
                tools_called=["hitl_operator_reply"],
                sources=[f"HITL Ticket {ticket_id}"],
                connection=conn,
            )
            sess_row = conn.execute(
                "SELECT session_facts_json FROM sessions WHERE session_id = ?",
                (ticket.session_id,),
            ).fetchone()
            facts = json.loads((sess_row["session_facts_json"] if sess_row else "{}") or "{}")
            facts[f"hitl_resolution_{ticket_id}"] = resumption_summary
            facts["latest_hitl_resumption"] = resumption_summary
            conn.execute(
                "UPDATE sessions SET session_facts_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(facts, ensure_ascii=False), now, ticket.session_id),
            )
            pending = conn.execute(
                "SELECT 1 FROM escalation_tickets WHERE session_id = ? AND user_id = ? "
                "AND status IN ('OPEN', 'CLAIMED', 'RESPONDED') LIMIT 1", (ticket.session_id, ticket.user_id),
            ).fetchone()
            self.store.set_session_status(ticket.session_id, "waiting_human" if pending else "active",
                                          user_id=ticket.user_id, connection=conn)
            conn.commit()

        if resume_agent and resolve_ticket and not pending:
            try:
                orch = orchestrator
                if orch is None:
                    from app.agents.orchestrator import orchestrator as global_orch
                    orch = global_orch
                orch.resume_session_from_hitl(
                    session_id=ticket.session_id,
                    user_id=ticket.user_id,
                    ticket_id=ticket_id,
                    operator_reply=masked_reply,
                    operator_id=operator_id,
                    store=self.store,
                )
            except Exception as exc:
                logger.warning("hitl_active_resumption_failed", ticket_id=ticket_id, error_type=type(exc).__name__)

        logger.info(
            "hitl_operator_replied_and_session_resumed",
            ticket_id=ticket_id,
            session_id=ticket.session_id,
            operator_id=operator_id,
            new_status=new_status,
            active_resumption_triggered=bool(resume_agent and resolve_ticket and not pending),
        )
        return self.get_ticket(ticket_id)

    @staticmethod
    def _validate_operator_transition(row: Any, operator_id: str, claiming: bool = False) -> None:
        if not operator_id or not operator_id.strip():
            raise HITLTransitionError("Operador responsável é obrigatório.")
        if row["status"] == "RESOLVED":
            raise HITLTransitionError("Ticket resolvido exige reabertura explícita antes de nova intervenção.")
        if row["assigned_operator"] and row["assigned_operator"] != operator_id:
            raise HITLTransitionError("Ticket pertence a outro operador; reatribuição explícita é necessária.")
        allowed = {"OPEN", "CLAIMED"} if claiming else {"OPEN", "CLAIMED", "RESPONDED"}
        if row["status"] not in allowed:
            raise HITLTransitionError("Transição de estado inválida para este ticket.")

    @staticmethod
    def _synthesize_agent_resumption(
        ticket_id: str,
        category: str,
        reason: str,
        operator_id: str,
        masked_reply: str,
        diagnostic_snapshot: Dict[str, Any],
        resolved: bool,
    ) -> str:
        terminal = (
            diagnostic_snapshot.get("terminal_id")
            or diagnostic_snapshot.get("terminal_model")
            or "terminal do lojista"
        )
        state_label = "orientação do especialista registrada; ticket resolvido" if resolved else "orientação intermediária registrada"
        return (
            f"[Resumo de Orientação Humana HITL {ticket_id} | {category} | {terminal}]: "
            f"O especialista {operator_id} analisou o motivo '{reason}' e registrou a diretriz: "
            f"{masked_reply} ({state_label})."
        )

    @staticmethod
    def _row_to_model(row: Any, store: Optional[StateStore] = None) -> HitlTicketModel:
        created_at_str = row["created_at"]
        priority = row["priority"]
        sla_mins = {"critical": 15, "high": 60, "medium": 120, "low": 240}.get(priority, 60)
        is_overdue = False
        sla_deadline_str = None
        try:
            created_dt = datetime.fromisoformat(created_at_str)
            deadline_dt = created_dt + timedelta(minutes=sla_mins)
            sla_deadline_str = deadline_dt.isoformat()
            # SLA de Resolução: permanece overdue enquanto não estiver RESOLVED
            if row["status"] != "RESOLVED":
                is_overdue = datetime.now(timezone.utc) > deadline_dt
        except Exception:
            pass

        diag_snapshot = json.loads(row["diagnostic_snapshot_json"] or "{}")
        op_response = row["operator_response"]
        resumption = None
        if op_response:
            resumption = HITLQueueManager._synthesize_agent_resumption(
                ticket_id=row["ticket_id"],
                category=row["category"],
                reason=row["reason"],
                operator_id=row["assigned_operator"] or "operador",
                masked_reply=op_response,
                diagnostic_snapshot=diag_snapshot,
                resolved=(row["status"] == "RESOLVED"),
            )

        resumed_turn = None
        if store is not None:
            try:
                with store._lock, store._connect() as conn:
                    s_row = conn.execute("SELECT session_facts_json FROM sessions WHERE session_id = ?", (row["session_id"],)).fetchone()
                    if s_row and s_row["session_facts_json"]:
                        s_facts = json.loads(s_row["session_facts_json"])
                        resumed_turn = s_facts.get(f"hitl_resumed_{row['ticket_id']}")
            except Exception:
                resumed_turn = None

        return HitlTicketModel(
            ticket_id=row["ticket_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            priority=priority,
            category=row["category"],
            reason=row["reason"],
            status=row["status"],
            assigned_operator=row["assigned_operator"],
            operator_response=op_response,
            diagnostic_snapshot=diag_snapshot,
            conversation_history=json.loads(row["conversation_history_json"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            sla_minutes=sla_mins,
            sla_deadline_at=sla_deadline_str,
            is_overdue=is_overdue,
            agent_resumption_response=resumption,
            resumed_turn=resumed_turn,
        )


hitl_queue = HITLQueueManager()
