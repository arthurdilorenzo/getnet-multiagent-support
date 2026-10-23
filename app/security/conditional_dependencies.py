"""Fail-closed terminal predicates for dependent documentary work.

The request supplies the condition. Only a typed diagnostic result supplies its
truth value; model text, tool names in prose, and other terminals do not.
"""

from dataclasses import dataclass
import re
from typing import Optional, Sequence

from app.schemas import ToolExecutionResult


@dataclass(frozen=True)
class TerminalCondition:
    terminal_id: str
    firmware_version: str
    status: str


_VERSION = r"(?=[A-Za-z0-9._-]*\d)[A-Za-z0-9][A-Za-z0-9._-]*"


def parse_terminal_condition(message: str, terminal_id: str) -> Optional[TerminalCondition]:
    """Recognize only a literal firmware and online conjunction for one terminal.

    The finite grammar deliberately rejects disjunction, ranges, implied values,
    extra predicates, and conditions whose terminal is not explicitly named.
    """
    if not isinstance(message, str) or not isinstance(terminal_id, str) or not terminal_id:
        return None
    target = re.escape(terminal_id)
    clauses = re.findall(r"\b(?:if|se)\s+([^,;]+)", message, flags=re.IGNORECASE)
    when_clauses = re.findall(r"\bquando\s+(.+)$", message, flags=re.IGNORECASE)
    if len(clauses) + len(when_clauses) != 1:
        return None
    if clauses:
        recognized = re.search(r"\b(?:if|se)\s+([^,;]+)", message, flags=re.IGNORECASE)
        if message[:recognized.start()].strip() or not re.fullmatch(
            r",\s*(?:tell\s+me\s+(?:the\s+eligible\s+)?benefits|informe\s+os\s+benefícios)\.",
            message[recognized.end():].strip(), flags=re.IGNORECASE,
        ):
            return None
        patterns = (
            rf"firmware\s+of\s+{target}\s+(?:is|equals|==)\s+({_VERSION})"
            rf"\s+and\s+status\s+(?:is|equals|==)\s+(online)",
            rf"(?:o\s+)?firmware\s+do\s+terminal\s+{target}\s+(?:for|estiver|==)\s+({_VERSION})"
            rf"\s+e\s+(?:o\s+)?status\s+(?:for|estiver|==)\s+(online)",
        )
        clause = clauses[0].strip()
    else:
        recognized = re.search(r"\bquando\s+(.+)$", message, flags=re.IGNORECASE)
        if not re.fullmatch(
            r"O\s+(?:recebimento|prazo(?:\s+de\s+recebimento)?)\s+do\s+Crediário\s+"
            r"só\s+deve\s+ser\s+explicado\s+",
            message[:recognized.start()], flags=re.IGNORECASE,
        ):
            return None
        patterns = (
            rf"o\s+{target}\s+tiver\s+firmware\s+({_VERSION})\s+com\s+conexão\s+(online)\.",
            rf"o\s+firmware\s+do\s+meu\s+{target}\s+for\s+exatamente\s+({_VERSION})"
            rf",\s+com\s+conexão\s+(ativa)\.",
        )
        clause = when_clauses[0].strip()
    match = next((found for pattern in patterns if (found := re.fullmatch(
        pattern, clause, flags=re.IGNORECASE,
    ))), None)
    if match is None:
        return None
    return TerminalCondition(terminal_id=terminal_id, firmware_version=match.group(1), status="Online")


def evaluate_terminal_condition(
    condition: Optional[TerminalCondition], observations: Sequence[ToolExecutionResult]
) -> bool:
    """Authorize only when every exact-target diagnostic observation agrees."""
    if not isinstance(condition, TerminalCondition) or not observations:
        return False
    matched = False
    for result in observations:
        if not isinstance(result, ToolExecutionResult) or result.tool_name != "diagnose_terminal_issue":
            continue
        data = result.data
        if not isinstance(data, dict):
            continue
        terminal = data.get("terminal")
        if data.get("requested_terminal_id") == condition.terminal_id and (
            not isinstance(terminal, dict) or terminal.get("terminal_id") != condition.terminal_id
        ):
            return False
        if not isinstance(terminal, dict) or terminal.get("terminal_id") != condition.terminal_id:
            continue
        matched = True
        if (result.success is not True or result.outcome_state != "completed"
                or result.error_code is not None or data.get("found") is not True
                or data.get("requested_terminal_id", condition.terminal_id) != condition.terminal_id
                or terminal.get("firmware_version") != condition.firmware_version
                or terminal.get("status") != condition.status):
            return False
    return matched
