"""Persistent SQLite repository and OTLP-compatible exporter for runtime traces and spans."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.security.pii import PIIRedactor


class TraceRepository:
    """Thread-safe SQLite store for execution traces, spans, and OTLP export payloads."""

    _STORE_BY_PATH: Dict[str, "TraceRepository"] = {}

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path or settings.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @classmethod
    def default(cls) -> "TraceRepository":
        db_path: Path | str = settings.SQLITE_DB_PATH
        main_mod = sys.modules.get("app.main")
        if main_mod is not None:
            active_fn = getattr(main_mod, "active_store", None)
            if callable(active_fn):
                try:
                    active = active_fn()
                    if getattr(active, "db_path", None) is not None:
                        db_path = active.db_path
                except Exception:
                    pass
        key = str(Path(db_path))
        repo = cls._STORE_BY_PATH.get(key)
        if repo is None:
            repo = cls(db_path=key)
            cls._STORE_BY_PATH[key] = repo
        return repo

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
                CREATE TABLE IF NOT EXISTS runtime_traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    runtime_version TEXT NOT NULL DEFAULT 'v2',
                    status TEXT NOT NULL DEFAULT 'running',
                    answer_status TEXT NOT NULL DEFAULT '',
                    total_duration_ms REAL NOT NULL DEFAULT 0.0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                    parent_trace_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_runtime_traces_session
                ON runtime_traces(session_id, started_at);

                CREATE INDEX IF NOT EXISTS idx_runtime_traces_user
                ON runtime_traces(user_id, started_at);

                CREATE TABLE IF NOT EXISTS runtime_spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    step_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    cumulative_ms REAL NOT NULL,
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runtime_spans_trace
                ON runtime_spans(trace_id, step_id);
                """
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_trace(
        self,
        trace_id: str,
        session_id: str,
        user_id: str,
        runtime_version: str = "v2",
        parent_trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_meta = PIIRedactor.mask_structure(metadata or {})
        now = self._now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO runtime_traces (
                    trace_id, session_id, user_id, runtime_version, status,
                    parent_trace_id, metadata_json, started_at
                )
                VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    user_id = excluded.user_id
                """,
                (
                    trace_id,
                    session_id or "",
                    user_id or "",
                    runtime_version,
                    parent_trace_id,
                    json.dumps(safe_meta, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()

    def record_span(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        step_id: int,
        agent: str,
        action: str,
        status: str,
        duration_ms: float,
        cumulative_ms: float,
        attributes: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        user_id: str = "",
    ) -> None:
        safe_attrs = PIIRedactor.mask_structure(attributes or {})
        now = self._now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO runtime_traces (
                    trace_id, session_id, user_id, runtime_version, status, started_at
                )
                VALUES (?, ?, ?, 'v2', 'running', ?)
                ON CONFLICT(trace_id) DO NOTHING
                """,
                (trace_id, session_id or "", user_id or "", now),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO runtime_spans (
                    span_id, trace_id, parent_span_id, step_id, name,
                    agent, action, status, duration_ms, cumulative_ms,
                    attributes_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span_id,
                    trace_id,
                    parent_span_id,
                    int(step_id),
                    f"{agent}:{action}",
                    agent,
                    action,
                    status,
                    float(duration_ms),
                    float(cumulative_ms),
                    json.dumps(safe_attrs, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()

    def finish_trace(
        self,
        trace_id: str,
        *,
        status: str = "completed",
        answer_status: str = "",
        total_duration_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_meta = PIIRedactor.mask_structure(metadata or {})
        now = self._now_iso()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE runtime_traces
                SET status = ?,
                    answer_status = ?,
                    total_duration_ms = ?,
                    prompt_tokens = ?,
                    completion_tokens = ?,
                    total_tokens = ?,
                    estimated_cost_usd = ?,
                    metadata_json = ?,
                    finished_at = ?
                WHERE trace_id = ?
                """,
                (
                    status,
                    answer_status,
                    float(total_duration_ms),
                    int(prompt_tokens),
                    int(completion_tokens),
                    int(total_tokens),
                    float(estimated_cost_usd),
                    json.dumps(safe_meta, ensure_ascii=False),
                    now,
                    trace_id,
                ),
            )
            conn.commit()

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runtime_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if row is None:
                return None
            spans_rows = conn.execute(
                "SELECT * FROM runtime_spans WHERE trace_id = ? ORDER BY step_id ASC",
                (trace_id,),
            ).fetchall()

        spans = [
            {
                "trace_id": s["trace_id"],
                "span_id": s["span_id"],
                "parent_span_id": s["parent_span_id"],
                "step_id": s["step_id"],
                "name": s["name"],
                "kind": "SPAN_KIND_INTERNAL",
                "agent": s["agent"],
                "action": s["action"],
                "status": {
                    "code": "STATUS_CODE_OK" if s["status"] == "success" else "STATUS_CODE_ERROR",
                    "raw": s["status"],
                },
                "duration_ms": s["duration_ms"],
                "cumulative_ms": s["cumulative_ms"],
                "attributes": json.loads(s["attributes_json"] or "{}"),
                "created_at": s["created_at"],
            }
            for s in spans_rows
        ]
        trace_dict = {
            "trace_id": row["trace_id"],
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "runtime_version": row["runtime_version"],
            "status": row["status"],
            "answer_status": row["answer_status"],
            "total_duration_ms": row["total_duration_ms"],
            "token_usage": {
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "estimated_cost_usd": row["estimated_cost_usd"],
            },
            "parent_trace_id": row["parent_trace_id"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "spans": spans,
        }
        trace_dict["otlp_resource_spans"] = self.to_otlp_payload(trace_dict)
        return trace_dict

    def list_traces(
        self,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(int(limit), 200)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM runtime_traces {where_sql} ORDER BY started_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [
            {
                "trace_id": r["trace_id"],
                "session_id": r["session_id"],
                "user_id": r["user_id"],
                "runtime_version": r["runtime_version"],
                "status": r["status"],
                "answer_status": r["answer_status"],
                "total_duration_ms": r["total_duration_ms"],
                "total_tokens": r["total_tokens"],
                "estimated_cost_usd": r["estimated_cost_usd"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
            }
            for r in rows
        ]

    @staticmethod
    def _otlp_hex(raw_id: Optional[str], length: int) -> str:
        if not raw_id:
            return ""
        import hashlib
        import re
        cleaned = re.sub(r"^(?:trc_|span_)", "", str(raw_id)).replace("_", "").lower()
        if len(cleaned) == length and all(c in "0123456789abcdef" for c in cleaned):
            return cleaned
        return hashlib.sha256(str(raw_id).encode("utf-8")).hexdigest()[:length]

    @staticmethod
    def _iso_to_unix_nano(iso_str: Optional[str], offset_ms: float = 0.0) -> str:
        try:
            dt = datetime.fromisoformat(str(iso_str)) if iso_str else datetime.now(timezone.utc)
            base_ns = int(dt.timestamp() * 1_000_000_000)
        except Exception:
            base_ns = int(time.time() * 1_000_000_000)
        return str(max(0, base_ns + int(offset_ms * 1_000_000)))

    @classmethod
    def to_otlp_payload(cls, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a persisted trace into a spec-compliant OpenTelemetry (OTLP/JSON) ResourceSpans envelope."""
        otlp_trace_id = cls._otlp_hex(trace_data.get("trace_id"), 32)
        otlp_spans = []
        for span in trace_data.get("spans", []):
            dur_ms = float(span.get("duration_ms") or 0.0)
            end_ns = cls._iso_to_unix_nano(span.get("created_at"), 0.0)
            start_ns = str(max(0, int(end_ns) - int(dur_ms * 1_000_000)))
            raw_status = (span.get("status") or {}).get("raw", "success")
            status_code = 1 if raw_status == "success" else 2
            attrs = [
                {"key": "getnet.public_trace_id", "value": {"stringValue": str(trace_data.get("trace_id", ""))}},
                {"key": "getnet.public_span_id", "value": {"stringValue": str(span.get("span_id", ""))}},
            ]
            for k, v in (span.get("attributes") or {}).items():
                attrs.append({"key": str(k), "value": {"stringValue": str(v)}})
            otlp_spans.append(
                {
                    "traceId": otlp_trace_id,
                    "spanId": cls._otlp_hex(span.get("span_id"), 16),
                    "parentSpanId": cls._otlp_hex(span.get("parent_span_id"), 16),
                    "name": span["name"],
                    "kind": 1,
                    "startTimeUnixNano": start_ns,
                    "endTimeUnixNano": end_ns,
                    "status": {"code": status_code},
                    "attributes": attrs,
                }
            )
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "getnet-multiagent-support"}},
                            {"key": "service.version", "value": {"stringValue": trace_data.get("runtime_version", "v2")}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "app.observability.tracer"},
                            "spans": otlp_spans,
                        }
                    ],
                }
            ]
        }
