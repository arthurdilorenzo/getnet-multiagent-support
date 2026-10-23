from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional
from app.security.pii import PIIRedactor


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": PIIRedactor.mask_text(record.getMessage()),
        }
        extra_fields = getattr(record, "structured_data", None)
        if isinstance(extra_fields, dict):
            payload.update(PIIRedactor.mask_structure(extra_fields))
        if record.exc_info:
            payload["exception"] = PIIRedactor.mask_text(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


class StructuredLogger:
    """
    Logger estruturado em JSON com correlação de trace_id e redaction automática de PII.
    Encaminha eventos de aplicação ao logging padrão em vez de chamadas de console.
    """

    def __init__(self, name: str = "getnet.multiagent"):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredJsonFormatter())
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False

    def info(self, event: str, trace_id: str = "", **kwargs: Any) -> None:
        self._logger.info(
            event,
            extra={"structured_data": {"event": event, "trace_id": trace_id, **kwargs}},
        )

    def warning(self, event: str, trace_id: str = "", **kwargs: Any) -> None:
        self._logger.warning(
            event,
            extra={"structured_data": {"event": event, "trace_id": trace_id, **kwargs}},
        )

    def error(self, event: str, trace_id: str = "", **kwargs: Any) -> None:
        self._logger.error(
            event,
            extra={"structured_data": {"event": event, "trace_id": trace_id, **kwargs}},
        )


logger = StructuredLogger()
