from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional
import httpx
from app.config import settings
from app.memory.store import StateStore, state_store
from app.observability.logger import logger
from app.schemas import HitlTicketModel
from app.security.pii import PIIRedactor


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

        now = self._now_iso()
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
                logger.warning("hitl_telegram_notification_failed", ticket_id=ticket.ticket_id, error=str(exc))
                dispatched["telegram"] = "failed"

        if settings.HITL_WEBHOOK_URL:
            try:
                with httpx.Client(timeout=2.5) as client:
                    resp = client.post(settings.HITL_WEBHOOK_URL, json=ticket.model_dump())
                    dispatched["webhook"] = f"http_{resp.status_code}"
            except Exception as exc:
                logger.warning("hitl_webhook_notification_failed", ticket_id=ticket.ticket_id, error=str(exc))
                dispatched["webhook"] = "failed"

        return dispatched

    def get_ticket(self, ticket_id: str) -> Optional[HitlTicketModel]:
        with self.store._lock, self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM escalation_tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_model(row)

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
            return [self._row_to_model(r) for r in rows]

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
    ) -> Optional[HitlTicketModel]:
        """
        Registra a resposta de um especialista humano no ticket,
        injeta a resposta no histórico conversacional da sessão do cliente
        e reativa a sessão ('active') permitindo retomada fluida pelo agente.
        """
        ticket = self.get_ticket(ticket_id)
        if not ticket:
            return None
        self.store.assert_session_access(ticket.session_id, ticket.user_id)

        now = self._now_iso()
        new_status = "RESOLVED" if resolve_ticket else "RESPONDED"
        masked_reply = PIIRedactor.mask_text(reply_message)

        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
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
            # Ticket, message and session status commit together after owner validation.
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
            self.store.set_session_status(ticket.session_id, "active", user_id=ticket.user_id, connection=conn)
            conn.commit()

        logger.info(
            "hitl_operator_replied_and_session_resumed",
            ticket_id=ticket_id,
            session_id=ticket.session_id,
            operator_id=operator_id,
            new_status=new_status,
        )
        return self.get_ticket(ticket_id)

    @staticmethod
    def _row_to_model(row: Any) -> HitlTicketModel:
        return HitlTicketModel(
            ticket_id=row["ticket_id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            priority=row["priority"],
            category=row["category"],
            reason=row["reason"],
            status=row["status"],
            assigned_operator=row["assigned_operator"],
            operator_response=row["operator_response"],
            diagnostic_snapshot=json.loads(row["diagnostic_snapshot_json"] or "{}"),
            conversation_history=json.loads(row["conversation_history_json"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


hitl_queue = HITLQueueManager()
