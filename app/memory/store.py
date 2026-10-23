from datetime import datetime, timezone
from contextlib import nullcontext
import json
import re
from pathlib import Path
import sqlite3
import threading
import unicodedata
from typing import Any, Dict, List, Optional
from app.config import settings
from app.observability.logger import logger
from app.security.pii import PIIRedactor
from app.security.auth import CrossTenantAccessError


class StateStore:
    """
    Camada de Persistência Transacional SQLite para:
    1. Memória conversacional multi-turno (sessions, messages, rolling_summaries).
    2. Fila real Human-in-the-Loop / HITL (escalation_tickets).
    3. Ações operacionais de escrita e controle de idempotência (terminal_actions, idempotency_store).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or settings.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    rolling_summary TEXT NOT NULL DEFAULT '',
                    active_topic TEXT NOT NULL DEFAULT '',
                    last_agent TEXT NOT NULL DEFAULT '',
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_used TEXT DEFAULT '',
                    tools_called_json TEXT DEFAULT '[]',
                    sources_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

                CREATE TABLE IF NOT EXISTS escalation_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    priority TEXT NOT NULL DEFAULT 'medium',
                    category TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    assigned_operator TEXT,
                    operator_response TEXT,
                    diagnostic_snapshot_json TEXT NOT NULL DEFAULT '{}',
                    conversation_history_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tickets_status ON escalation_tickets(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_tickets_user ON escalation_tickets(user_id);

                CREATE TABLE IF NOT EXISTS terminal_actions (
                    action_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    terminal_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS idempotency_store (
                    idempotency_key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            # Additive migration: existing conversations remain readable after restart.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
            for column in ("session_facts_json", "entities_json"):
                if column not in columns:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} TEXT NOT NULL DEFAULT '{{}}'")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def resolve_session_id(self, user_id: str, provided_session_id: Optional[str] = None) -> str:
        if provided_session_id and provided_session_id.strip():
            return provided_session_id.strip()
        return f"sess_{user_id.strip()}"

    def ensure_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        now = self._now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                self._assert_owner(row, user_id)
                return dict(row)
            conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, status, rolling_summary, active_topic, last_agent, turn_count, created_at, updated_at)
                VALUES (?, ?, 'active', '', '', '', 0, ?, ?)
                """,
                (session_id, user_id, now, now),
            )
            conn.commit()
            return {
                "session_id": session_id,
                "user_id": user_id,
                "status": "active",
                "rolling_summary": "",
                "active_topic": "",
                "last_agent": "",
                "turn_count": 0,
                "created_at": now,
                "updated_at": now,
            }

    @staticmethod
    def _assert_owner(row: Any, user_id: str) -> None:
        if not row or row["user_id"] != user_id:
            raise CrossTenantAccessError("Acesso negado: a sessão não pertence ao usuário solicitado.")

    def assert_session_access(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Authorize against the persisted owner; a session name conveys no authority."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._assert_owner(row, user_id)
            return dict(row)

    def append_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        agent_used: str = "",
        tools_called: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        active_topic: Optional[str] = None,
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        now = self._now_iso()
        masked_content = PIIRedactor.mask_text(content)

        with self._lock, (nullcontext(connection) if connection is not None else self._connect()) as conn:
            if connection is None:
                self.ensure_session(session_id, user_id)
                conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._assert_owner(owner, user_id)
            conn.execute(
                """
                INSERT INTO messages (session_id, user_id, role, content, agent_used, tools_called_json, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    role,
                    masked_content,
                    agent_used,
                    json.dumps(tools_called or [], ensure_ascii=False),
                    json.dumps(sources or [], ensure_ascii=False),
                    now,
                ),
            )

            session_row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            new_turn_count = (session_row["turn_count"] if session_row else 0) + (1 if role == "user" else 0)
            topic_to_save = active_topic if active_topic is not None else (session_row["active_topic"] if session_row else "")
            facts = json.loads(session_row["session_facts_json"] or "{}")
            entities = json.loads(session_row["entities_json"] or "{}")
            if role == "user":
                self._extract_user_referents(masked_content, facts, entities)

            conn.execute(
                """
                UPDATE sessions
                SET turn_count = ?,
                    last_agent = CASE WHEN ? != '' THEN ? ELSE last_agent END,
                    active_topic = CASE WHEN ? != '' THEN ? ELSE active_topic END,
                    session_facts_json = ?, entities_json = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (new_turn_count, agent_used, agent_used, topic_to_save, topic_to_save,
                 json.dumps(facts, ensure_ascii=False), json.dumps(entities, ensure_ascii=False), now, session_id),
            )
            self._maybe_compact_summary(conn, session_id)
            if connection is None:
                conn.commit()

    @staticmethod
    def _extract_user_referents(content: str, facts: Dict[str, Any], entities: Dict[str, Any]) -> None:
        """Keep literal, bounded reference slots, never instructions or inferred permissions."""
        norm = "".join(c for c in unicodedata.normalize("NFD", content.lower()) if unicodedata.category(c) != "Mn")
        products = []
        for pattern, label in (
            (r"\b(?:antecipacao|receivables? advance|anticipation)\b", "antecipação"),
            (r"\bpix\b", "Pix"), (r"\bcrediario\b", "crediário"),
            (r"\bget\s+classica\b", "Get Clássica"), (r"\bget\s+smart\b", "Get Smart"),
            (r"\bget\s+tap\b", "Get Tap"), (r"\b(?:link de pagamento|payment link)\b", "link de pagamento"),
            (r"\bget\s+conta\b", "Get Conta"),
        ):
            match = re.search(pattern, norm)
            if match:
                products.append((match.start(), label))
        for _, product in sorted(products):
            previous = entities.get("products", [])
            entities["products"] = [p for p in previous if p != product][-11:] + [product]
            facts["last_product"] = product
            facts["last_subject"] = product
        if len(products) >= 2:
            facts["product_comparison"] = [product for _, product in sorted(products)][-2:]
        terminals = re.findall(r"\bGET-[A-Z]+-\d+\b", content.upper())
        if terminals:
            entities["terminal_id"] = terminals[-1][:80]
        code = re.search(r"\b(?:codigo|code|erro|error)\s*[:#]?\s*(\d{2,3})\b", norm)
        if code:
            entities["decline_code"] = code.group(1)

    def _maybe_compact_summary(self, conn: sqlite3.Connection, session_id: str) -> None:
        rows = conn.execute(
            "SELECT role, content, agent_used FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        if len(rows) > settings.MEMORY_WINDOW_MESSAGES:
            older_rows = rows[: -settings.MEMORY_WINDOW_MESSAGES]
            snippets = []
            for r in older_rows[-6:]:
                snippet = r["content"][:140].replace("\n", " ")
                snippets.append(f"{r['role']}: {snippet}")
            summary = " | ".join(snippets)
            conn.execute(
                "UPDATE sessions SET rolling_summary = ? WHERE session_id = ?",
                (summary, session_id),
            )

    def get_conversation_context(self, session_id: str, user_id: str, limit: int = 8) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            self.ensure_session(session_id, user_id)
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._assert_owner(row, user_id)
            session = dict(row)
            rows = conn.execute(
                """
                SELECT id, role, content, agent_used, tools_called_json, sources_json, created_at
                FROM messages
                WHERE session_id = ? AND user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, user_id, limit),
            ).fetchall()

        messages = [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "agent_used": r["agent_used"],
                "tools_called": json.loads(r["tools_called_json"] or "[]"),
                "sources": json.loads(r["sources_json"] or "[]"),
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]

        return {
            "session_id": session_id,
            "user_id": user_id,
            "status": session.get("status", "active"),
            "rolling_summary": session.get("rolling_summary", ""),
            "summary": session.get("rolling_summary", ""),
            "session_facts": json.loads(session.get("session_facts_json", "{}") or "{}"),
            "entities": json.loads(session.get("entities_json", "{}") or "{}"),
            "active_topic": session.get("active_topic", ""),
            "last_agent": session.get("last_agent", ""),
            "turn_count": session.get("turn_count", 0),
            "recent_messages": messages,
        }

    def has_pending_human(self, session_id: str, user_id: str) -> bool:
        """A paused session is scoped by its persisted owner, never its name."""
        with self._lock, self._connect() as conn:
            owner = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._assert_owner(owner, user_id)
            return conn.execute(
                "SELECT 1 FROM escalation_tickets WHERE session_id = ? AND user_id = ? "
                "AND status IN ('OPEN', 'CLAIMED', 'RESPONDED') LIMIT 1", (session_id, user_id),
            ).fetchone() is not None

    def remember_tool_referents(self, session_id: str, user_id: str, results: List[Dict[str, Any]]) -> None:
        """Persist allowlisted slots from successful operational evidence, not model prose."""
        updates: Dict[str, Any] = {}
        for result in results:
            if not result.get("success") or result.get("tool_name") != "diagnose_terminal_issue":
                continue
            data = result.get("data") or {}
            terminal = data.get("terminal") or {}
            terminal_id = terminal.get("terminal_id")
            if isinstance(terminal_id, str) and re.fullmatch(r"GET-[A-Z]+-\d+", terminal_id):
                updates["terminal_id"] = terminal_id[:80]
            transaction = data.get("last_declined_transaction") or data.get("declined_transaction") or {}
            code = transaction.get("decline_code") or data.get("decline_code")
            if isinstance(code, (str, int)) and re.fullmatch(r"\d{2,3}", str(code)):
                updates["decline_code"] = str(code)
        if not updates:
            return
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT user_id, entities_json FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._assert_owner(row, user_id)
            entities = json.loads(row["entities_json"] or "{}")
            entities.update(updates)
            conn.execute("UPDATE sessions SET entities_json = ?, updated_at = ? WHERE session_id = ?",
                         (json.dumps(entities, ensure_ascii=False), self._now_iso(), session_id))
            conn.commit()

    def set_session_status(
        self, session_id: str, status: str, user_id: str, connection: Optional[sqlite3.Connection] = None
    ) -> None:
        now = self._now_iso()
        with self._lock, (nullcontext(connection) if connection is not None else self._connect()) as conn:
            if connection is None:
                conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._assert_owner(owner, user_id)
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (status, now, session_id),
            )
            if connection is None:
                conn.commit()

    def clear_session(self, session_id: str, user_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            self._assert_owner(owner, user_id)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    # =========================================================================
    # Idempotency & Write Tool Persistence
    # =========================================================================
    def get_idempotent_result(self, idempotency_key: str, user_id: str) -> Optional[Dict[str, Any]]:
        if not idempotency_key:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, response_json FROM idempotency_store WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if not row:
                return None
            if row["user_id"] != user_id:
                raise PermissionError("Chave de idempotência pertence a outro tenant.")
            return json.loads(row["response_json"])

    def save_idempotent_result(
        self, idempotency_key: str, tool_name: str, user_id: str, result: Dict[str, Any]
    ) -> None:
        if not idempotency_key:
            return
        now = self._now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO idempotency_store (idempotency_key, tool_name, user_id, response_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (idempotency_key, tool_name, user_id, json.dumps(result, ensure_ascii=False), now),
            )
            conn.commit()

    def record_terminal_action(
        self,
        action_id: str,
        user_id: str,
        terminal_id: str,
        action_type: str,
        status: str,
        idempotency_key: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = self._now_iso()
        record = {
            "action_id": action_id,
            "user_id": user_id,
            "terminal_id": terminal_id,
            "action_type": action_type,
            "status": status,
            "idempotency_key": idempotency_key,
            "details": details,
            "created_at": now,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO terminal_actions (action_id, user_id, terminal_id, action_type, status, idempotency_key, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    user_id,
                    terminal_id,
                    action_type,
                    status,
                    idempotency_key,
                    json.dumps(details, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
        return record


state_store = StateStore()
