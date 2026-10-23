import re
from typing import Any, Dict, List, Optional
from app.observability.tracer import SystemTracer
from app.schemas import SpecialistOutput, ToolExecutionResult
from app.security.auth import SecurityContext
import app.tools.customer_tools  # noqa: F401 - garante registro de create_escalation_ticket e check_escalation_ticket
from app.tools.registry import tool_registry


class EscalationAgent:
    """
    Agente 4: Human Escalation & HITL Queue Agent.
    Utiliza ferramentas reais registradas no `ToolRegistry` (`create_escalation_ticket`,
    `check_escalation_ticket`) para persistir chamados na fila SQLite (`escalation_tickets`),
    anexar o histórico real da sessão e snapshots de diagnóstico, alterar o estado da sessão
    para `waiting_human` e permitir que operadores humanos respondam via API retomando a conversa.
    """

    def run(
        self,
        message: str,
        security_context: SecurityContext,
        session_id: str = "",
        reason: str = "Solicitação explícita de atendimento humano",
        category: str = "human_handoff",
        priority: str = "high",
        diagnostic_snapshot: Optional[Dict[str, Any]] = None,
        idempotency_key: str = "",
        tracer: Optional[SystemTracer] = None,
    ) -> SpecialistOutput:
        trace_id = tracer.trace_id if tracer else ""
        lower = message.lower()

        tools_called: List[str] = []
        tool_results: List[ToolExecutionResult] = []

        # Se o cliente estiver consultando o andamento de um protocolo existente (GET-ESC-...)
        ticket_match = re.search(r"\b(GET-ESC-[A-Z0-9]+)\b", message, re.I)
        if ticket_match and any(w in lower for w in ["status", "andamento", "consultar", "resposta", "chamado", "protocolo"]):
            t_id = ticket_match.group(1).upper()
            tools_called.append("check_escalation_ticket")
            res = tool_registry.execute(
                tool_name="check_escalation_ticket",
                raw_arguments={"ticket_id": t_id},
                security_context=security_context,
                session_id=session_id,
                trace_id=trace_id,
            )
            tool_results.append(res)
            if tracer:
                tracer.add_step(
                    agent="HUMAN_ESCALATION_TOOL",
                    action="call:check_escalation_ticket",
                    status="success" if res.success else "error",
                    details={"ticket_id": t_id, "success": res.success},
                )
            data = res.data or {}
            if data.get("found") and data.get("ticket"):
                tk = data["ticket"]
                op_reply = tk.get("operator_response")
                reply_section = (
                    f"\n- **Resposta do Operador ({tk.get('assigned_operator')}):** {op_reply}"
                    if op_reply
                    else "\n- **Andamento:** Aguardando manifestação do operador na fila HITL."
                )
                resp_text = (
                    f"**Status do Protocolo `{tk['ticket_id']}`:**\n\n"
                    f"- **Status na Fila:** `{tk['status']}`\n"
                    f"- **Prioridade:** `{tk['priority'].upper()}`\n"
                    f"- **Motivo Registrado:** {tk['reason']}"
                    f"{reply_section}"
                )
            else:
                resp_text = f"Não localizamos o protocolo `{t_id}` vinculado à sua conta."

            return SpecialistOutput(
                agent_name="HUMAN_ESCALATION_AGENT",
                response=resp_text,
                confidence=0.98,
                tools_called=tools_called,
                tool_results=tool_results,
                sources=["Getnet SQLite HITL Ticket Queue"],
                structured_data=data,
            )

        # Caso padrão: Criação real e persistida de ticket na fila HITL
        tools_called.append("create_escalation_ticket")
        res = tool_registry.execute(
            tool_name="create_escalation_ticket",
            raw_arguments={
                "session_id": session_id or f"sess_{security_context.authenticated_user_id}",
                "category": category,
                "reason": reason,
                "priority": priority if priority in ("low", "medium", "high", "critical") else "high",
                "diagnostic_snapshot": diagnostic_snapshot or {},
                "idempotency_key": idempotency_key,
            },
            security_context=security_context,
            session_id=session_id,
            trace_id=trace_id,
        )
        tool_results.append(res)

        if tracer:
            tracer.add_step(
                agent="HUMAN_ESCALATION_TOOL",
                action="call:create_escalation_ticket",
                status="success" if res.success else "error",
                details={
                    "ticket_id": (res.data or {}).get("ticket_id"),
                    "status": (res.data or {}).get("status"),
                    "execution_ms": res.execution_ms,
                },
            )

        ticket_data = res.data or {}
        protocol_number = ticket_data.get("ticket_id")
        if not res.success or not protocol_number:
            return SpecialistOutput(
                agent_name="HUMAN_ESCALATION_AGENT",
                response=(
                    "Não foi possível registrar o chamado na fila de atendimento neste momento. "
                    "Nenhum protocolo de atendimento foi confirmado. Tente novamente em instantes."
                ),
                confidence=0.0,
                abstained=True,
                abstention_reason=res.error_code or "TICKET_NOT_PERSISTED",
                tools_called=tools_called,
                tool_results=tool_results,
            )
        ticket_status = ticket_data.get("status", "OPEN")
        history_len = len(ticket_data.get("conversation_history", []))
        has_diag = bool(ticket_data.get("diagnostic_snapshot"))

        response_text = (
            f"Registramos a sua solicitação na fila de atendimento especializado Human-in-the-Loop (HITL) da Getnet.\n\n"
            f"- **Protocolo Persistido:** `{protocol_number}` (Status: `{ticket_status}`)\n"
            f"- **Prioridade:** `{priority.upper()}` | **Categoria:** `{category}`\n"
            f"- **Motivo do Transbordo:** {reason}\n"
            f"- **Contexto Anexado no Banco:** {history_len} mensagem(ns) da sessão `{session_id or 'ativa'}`"
            + (" + snapshot técnico das ferramentas executadas." if has_diag else ".")
            + f"\n\nO chamado encontra-se aberto na fila operacional (`/api/hitl/tickets/{protocol_number}`). "
            f"Assim que um operador registrar a resposta, ela ficará disponível nesta sessão."
        )

        if tracer:
            tracer.record_tokens(prompt_text=message, completion_text=response_text)

        return SpecialistOutput(
            agent_name="HUMAN_ESCALATION_AGENT",
            response=response_text,
            confidence=0.99,
            requires_escalation=True,
            escalation_reason=reason,
            tools_called=tools_called,
            tool_results=tool_results,
            sources=["Getnet SQLite HITL Ticket Queue"],
            structured_data={"ticket": ticket_data, "protocol_number": protocol_number},
            evidence=[{"kind": "tool_fact", "tool_name": "create_escalation_ticket",
                       "fact_path": key, "value": value}
                      for key, value in ticket_data.items()],
        )

    def handle(
        self,
        message: str,
        user_id: str,
        reason: str = "Solicitação explícita do cliente",
        session_id: str = "",
        diagnostic_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Interface de compatibilidade direta.
        """
        sec_ctx = SecurityContext(authenticated_user_id=user_id)
        out = self.run(
            message=message,
            security_context=sec_ctx,
            session_id=session_id or f"sess_{user_id}",
            reason=reason,
            diagnostic_snapshot=diagnostic_snapshot,
        )
        ticket_obj = out.structured_data.get("ticket", {})
        return {
            "response": out.response,
            "escalated_to_human": True,
            "protocol_number": out.structured_data.get("protocol_number"),
            "ticket": ticket_obj,
            "tools_called": out.tools_called,
            "queue": "Getnet SQLite HITL Ticket Queue",
        }
