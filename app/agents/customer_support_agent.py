import json
import re
from typing import Any, Dict, List, Optional
from app.config import settings
from app.observability.logger import logger
from app.observability.resilience import call_with_resilience
from app.observability.tracer import SystemTracer
from app.rag.embeddings import BilingualDenseEmbedder
from app.schemas import SpecialistOutput, ToolCallSpec, ToolExecutionResult
from app.security.auth import SecurityContext
import app.tools.customer_tools  # noqa: F401 - garante registro das customer tools
from app.tools.registry import tool_registry

SUPPORT_SYSTEM_PROMPT = """Você é o Customer Support Agent autenticado da Getnet.
Sua função é diagnosticar e resolver demandas operacionais de clientes usando estritamente os dados retornados pelas ferramentas (`get_sales_settlement`, `diagnose_terminal_issue`, `get_customer_profile`, `execute_terminal_remote_action`).
Regras estritas:
1. NUNCA invente saldos, datas de depósito, modelos de máquina ou códigos de erro que não estejam nos resultados das tools.
2. O `user_id` já está vinculado com segurança pelo `SecurityContext` — nunca tente acessar dados de outro cliente.
3. Apresente valores financeiros, lotes de liquidação, status de conectividade e passos de resolução de forma estruturada e clara.
"""


class CustomerSupportAgent:
    """
    Agente 3: Customer Support Agent com Loop de Function Calling Multi-Turno,
    isolamento de identidade via `SecurityContext`, suporte a múltiplos terminais POS,
    ações remotas de escrita idempotentes e detecção automática de necessidade de escalonamento.
    """

    AVAILABLE_TOOLS = [
        "get_sales_settlement",
        "diagnose_terminal_issue",
        "get_customer_profile",
        "execute_terminal_remote_action",
    ]

    def __init__(self, gemini_client: Optional[Any] = None) -> None:
        self.client = gemini_client

    def run(
        self,
        message: str,
        security_context: SecurityContext,
        session_id: str = "",
        conversation_context: Optional[Dict[str, Any]] = None,
        tracer: Optional[SystemTracer] = None,
    ) -> SpecialistOutput:
        trace_id = tracer.trace_id if tracer else ""
        contextual_msg = self._enrich_with_session_memory(message, conversation_context)

        tools_called: List[str] = []
        tool_results: List[ToolExecutionResult] = []
        collected_data: Dict[str, Any] = {}
        requires_escalation = False
        escalation_reason: Optional[str] = None

        # Turno 1 do Loop de Tools: planeja chamadas iniciais via Function Calling Schema
        pending_calls = self._plan_initial_tools(contextual_msg, tracer=tracer)

        turn = 0
        while pending_calls and turn < settings.MAX_TOOL_TURNS:
            turn += 1
            call_spec = pending_calls.pop(0)
            tools_called.append(call_spec.tool_name)

            res = tool_registry.execute(
                tool_name=call_spec.tool_name,
                raw_arguments=call_spec.arguments,
                security_context=security_context,
                session_id=session_id,
                trace_id=trace_id,
            )
            tool_results.append(res)

            if tracer:
                tracer.add_step(
                    agent="CUSTOMER_SUPPORT_TOOL",
                    action=f"call:{call_spec.tool_name}",
                    status="success" if res.success else "error",
                    details={
                        "turn": turn,
                        "arguments": call_spec.arguments,
                        "success": res.success,
                        "idempotent_replay": res.idempotent_replay,
                        "error_code": res.error_code,
                        "execution_ms": res.execution_ms,
                    },
                )

            if res.success and isinstance(res.data, dict):
                collected_data[call_spec.tool_name] = res.data

                # Observação do resultado da tool no loop (ReAct Observation -> Next Tool / Escalation)
                if call_spec.tool_name == "diagnose_terminal_issue":
                    diag_data = res.data
                    # Se o diagnóstico apontar falha física de hardware que exige substituição, sinaliza escalação dinâmica!
                    if diag_data.get("requires_hardware_replacement"):
                        requires_escalation = True
                        term_id = (diag_data.get("terminal") or {}).get("terminal_id", "POS")
                        escalation_reason = (
                            f"Diagnóstico de telemetria detectou falha física de hardware no terminal {term_id} "
                            f"exigindo substituição de equipamento (Nível 2)."
                        )

                    # Se o usuário solicitou reinício remoto / troca de operadora ou atualização de tabelas, encadeia a write tool!
                    norm_msg = BilingualDenseEmbedder.normalize(contextual_msg)
                    if (
                        "execute_terminal_remote_action" not in tools_called
                        and any(w in norm_msg for w in ["reiniciar remotamente", "remote reboot", "trocar operadora", "carrier switch", "atualizar tabela"])
                    ):
                        term_obj = diag_data.get("terminal") or {}
                        t_id = term_obj.get("terminal_id", "GET-SMART-88219")
                        act_type = "sim_carrier_switch" if "operadora" in norm_msg or "carrier" in norm_msg else "remote_reboot"
                        pending_calls.append(
                            ToolCallSpec(
                                tool_name="execute_terminal_remote_action",
                                arguments={
                                    "terminal_id": t_id,
                                    "action_type": act_type,
                                    "reason": f"Solicitação durante diagnóstico de {diag_data.get('symptom_classified')}",
                                },
                                rationale="Execução de comando remoto no terminal após diagnóstico preliminar.",
                            )
                        )

        # Verifica se houve bloqueio de segurança Cross-Tenant
        for r in tool_results:
            if not r.success and r.error_code == "CROSS_TENANT_ACCESS_DENIED":
                return SpecialistOutput(
                    agent_name="CUSTOMER_SUPPORT_AGENT",
                    response=f"Acesso negado: {r.error_message}",
                    confidence=1.0,
                    abstained=True,
                    abstention_reason=r.error_message,
                    tools_called=tools_called,
                    tool_results=tool_results,
                )

        response_text = self._synthesize_support_response(
            message=message,
            user_id=security_context.authenticated_user_id,
            collected_data=collected_data,
            tracer=tracer,
        )

        sources = ["Getnet Merchant Settlement DB", "Getnet POS Telemetry Engine"]
        return SpecialistOutput(
            agent_name="CUSTOMER_SUPPORT_AGENT",
            response=response_text,
            confidence=0.95,
            abstained=False,
            requires_escalation=requires_escalation,
            escalation_reason=escalation_reason,
            tools_called=tools_called,
            tool_results=tool_results,
            sources=sources,
            structured_data=collected_data,
        )

    def handle(self, message: str, user_id: str) -> Dict[str, Any]:
        """
        Interface de compatibilidade direta.
        """
        sec_ctx = SecurityContext(authenticated_user_id=user_id)
        out = self.run(message=message, security_context=sec_ctx)
        return {
            "response": out.response,
            "tools_called": out.tools_called,
            "data_used": out.structured_data,
            "requires_escalation": out.requires_escalation,
            "escalation_reason": out.escalation_reason,
        }

    @staticmethod
    def _enrich_with_session_memory(
        message: str, conversation_context: Optional[Dict[str, Any]]
    ) -> str:
        if not conversation_context:
            return message
        norm = BilingualDenseEmbedder.normalize(message)
        if len(message.split()) <= 7 and any(w in norm for w in ("ela", "ele", "maquina", "codigo", "venda", "ontem", "terminal")):
            recent = conversation_context.get("recent_messages", [])
            prev_user = [m["content"] for m in recent if m.get("role") == "user" and m.get("content") != message]
            if prev_user:
                return f"{prev_user[-1]} | Contexto complementar: {message}"
        return message

    def _plan_initial_tools(
        self, message: str, tracer: Optional[SystemTracer] = None
    ) -> List[ToolCallSpec]:
        norm = BilingualDenseEmbedder.normalize(message)
        lower = message.lower()
        calls: List[ToolCallSpec] = []

        # Extração de tentativa explícita de acesso a outro user_id no texto (ex: "consultar dados do cliente2045")
        spoof_match = re.search(r"\b(cliente\d{3,6})\b", lower)
        extracted_target_uid = spoof_match.group(1) if spoof_match else None

        # 1. Liquidação de vendas / depósitos / extrato
        if any(w in norm for w in ["vendas", "ontem", "yesterday", "dinheiro", "deposit", "cair", "liquidacao", "repasse", "saldo", "extrato"]):
            modality = "all"
            if "pix" in norm:
                modality = "pix"
            elif "debito" in norm or "debit" in norm:
                modality = "debit"
            elif "credito" in norm or "credit" in norm:
                modality = "credit"
            args: Dict[str, Any] = {"payment_modality": modality}
            if extracted_target_uid:
                args["user_id"] = extracted_target_uid
            calls.append(
                ToolCallSpec(
                    tool_name="get_sales_settlement",
                    arguments=args,
                    rationale="Consulta de agenda de liquidação e repasse das vendas realizadas.",
                )
            )

        # 2. Diagnóstico de Terminal POS / Conectividade / Transação Recusada / Hardware
        if any(
            w in norm
            for w in [
                "conecta", "connect", "internet", "sinal", "wifi", "chip", "rede",
                "decline", "recusa", "recusada", "erro", "error", "transacao", "transaction",
                "maquininha", "maquina", "machine", "terminal", "codigo", "leitor", "defeito",
            ]
        ):
            symptom = "general"
            if any(w in norm for w in ["decline", "recusa", "recusada", "transacao", "transaction", "codigo"]):
                symptom = "decline"
            elif any(w in norm for w in ["conecta", "connect", "internet", "sinal", "wifi", "chip", "rede"]):
                symptom = "connectivity"
            elif any(w in norm for w in ["leitor", "defeito", "hardware"]):
                symptom = "hardware"

            # Extração de terminal_id específico se mencionado (ex: GET-CLASSICA-10442)
            term_id_match = re.search(r"\b(GET-[A-Z]+-\d+)\b", message, re.I)
            term_id = term_id_match.group(1).upper() if term_id_match else None

            # Extração de modelo se mencionado
            term_model = None
            if "classica" in norm:
                term_model = "Get Clássica"
            elif "smart" in norm:
                term_model = "Get Smart"

            # Extração de código ISO-8583 se mencionado
            code_match = re.search(r"\b(?:codigo|code|erro|error)\s*(05|51|12|54|57|91)\b", norm)
            decline_code = code_match.group(1) if code_match else None

            diag_args: Dict[str, Any] = {
                "terminal_id": term_id,
                "terminal_model": term_model,
                "symptom": symptom,
                "decline_code": decline_code,
                "error_context": message,
            }
            if extracted_target_uid:
                diag_args["user_id"] = extracted_target_uid

            calls.append(
                ToolCallSpec(
                    tool_name="diagnose_terminal_issue",
                    arguments=diag_args,
                    rationale="Diagnóstico de telemetria de terminal POS e análise de códigos ISO-8583.",
                )
            )

        # 3. Se nenhuma das anteriores foi acionada, consulta o perfil cadastral completo
        if not calls:
            prof_args: Dict[str, Any] = {}
            if extracted_target_uid:
                prof_args["user_id"] = extracted_target_uid
            calls.append(
                ToolCallSpec(
                    tool_name="get_customer_profile",
                    arguments=prof_args,
                    rationale="Consulta de perfil cadastral, plano e frota de máquinas do cliente.",
                )
            )

        return calls

    def _synthesize_support_response(
        self,
        message: str,
        user_id: str,
        collected_data: Dict[str, Any],
        tracer: Optional[SystemTracer] = None,
    ) -> str:
        # 1. Síntese via Gemini se configurado
        if self.client and settings.GEMINI_API_KEY:
            try:
                prompt = (
                    f"Cliente Autenticado: {user_id}\n"
                    f"Mensagem: \"{message}\"\n\n"
                    f"Dados retornados pelas ferramentas operacionais:\n"
                    f"{json.dumps(collected_data, ensure_ascii=False, indent=2)}\n\n"
                    f"Responda de forma clara e fundamentada apenas nesses dados."
                )

                def _call_llm() -> Any:
                    return self.client.models.generate_content(
                        model=settings.GEMINI_MODEL,
                        contents=prompt,
                        config={"system_instruction": SUPPORT_SYSTEM_PROMPT, "temperature": 0.1},
                    )

                resp = call_with_resilience("gemini_api", _call_llm, max_retries=1)
                if resp and resp.text:
                    if tracer:
                        tracer.record_tokens(prompt_text=prompt, completion_text=resp.text)
                    return resp.text.strip()
            except Exception as exc:
                logger.warning("support_llm_synthesis_fallback", error=str(exc))

        # 2. Síntese Estruturada Orientada aos Dados Retornados pelas Tools
        parts: List[str] = []

        if "get_sales_settlement" in collected_data:
            st = collected_data["get_sales_settlement"]
            if not st.get("found"):
                parts.append(
                    f"Não localizamos registro de liquidação para o identificador `{user_id}` ({st.get('error')})."
                )
            else:
                sales = st.get("sales_data", {})
                company = st.get("company_name", user_id)
                plan = st.get("account_plan", "Getnet")
                parts.append(
                    f"Olá, **{company}** (`{user_id}` — Plano *{plan}*)!\n\n"
                    f"Consultamos a sua agenda financeira referente às vendas de ontem (**{sales.get('date')}**, Lote `{sales.get('settlement_batch_id')}`):\n\n"
                    f"- **Volume Bruto Total:** R$ {sales.get('total_gross_brl', 0):,.2f} "
                    f"(Débito: R$ {sales.get('debit_brl', 0):,.2f} | Crédito: R$ {sales.get('credit_brl', 0):,.2f} | Pix: R$ {sales.get('pix_brl', 0):,.2f})\n"
                    f"- **Valor Líquido Programado para Hoje:** **R$ {sales.get('net_payout_scheduled_today_brl', 0):,.2f}**\n"
                    f"- **Status da Liquidação:** {sales.get('payout_status')}\n"
                    f"- **Domicílio Bancário de Destino:** {sales.get('target_bank')}\n"
                    f"- **Liquidação Pix:** {sales.get('pix_note')}"
                )

        if "diagnose_terminal_issue" in collected_data:
            diag = collected_data["diagnose_terminal_issue"]
            if not diag.get("found"):
                parts.append(f"Não localizamos terminais vinculados ao identificador `{user_id}`.")
            else:
                symptom = diag.get("symptom_classified", "general")
                term = diag.get("terminal") or {}
                all_terms = diag.get("all_terminals") or []
                term_model = term.get("model", "Terminal POS Getnet")
                term_id = term.get("terminal_id", "N/A")
                signal = term.get("signal_strength", "N/A")
                conn_type = term.get("connection_type", "Rede Móvel/Wi-Fi")
                fleet_note = (
                    f" (frota total cadastrada: {len(all_terms)} terminais)"
                    if len(all_terms) > 1
                    else ""
                )

                if symptom == "decline":
                    last_txn = diag.get("last_declined_transaction")
                    if last_txn:
                        parts.append(
                            f"Analisamos a telemetria do terminal **{term_model}** (`{last_txn.get('terminal_id') or term_id}`){fleet_note} "
                            f"e identificamos a transação recusada `{last_txn.get('transaction_id')}` registrada em `{last_txn.get('timestamp')}`:\n\n"
                            f"- **Valor e Modalidade:** R$ {last_txn.get('amount_brl', 0):,.2f} ({last_txn.get('payment_method', 'Cartão')} — {last_txn.get('card_brand')})\n"
                            f"- **Código ISO-8583 Retornado:** **Código {last_txn.get('decline_code')}**\n"
                            f"- **Diagnóstico Técnico:** {last_txn.get('decline_reason')}\n"
                            f"- **Ação Recomendada:** {last_txn.get('action_required')}\n\n"
                            f"*Nota técnica:* Recusas de código `{last_txn.get('decline_code')}` partem do banco emissor do portador do cartão, indicando que o seu terminal Getnet está comunicando normalmente."
                        )
                    else:
                        resolution = diag.get("recommended_troubleshooting", {}).get("decline_resolution", "")
                        parts.append(
                            f"Diagnóstico do terminal **{term_model}** (`{term_id}`):\n\n{resolution}"
                        )
                else:
                    steps = diag.get("recommended_troubleshooting", {}).get("connectivity", [])
                    steps_str = "\n".join(steps)
                    hw_alert = (
                        "\n\n**Alerta Crítico de Hardware:** A telemetria identificou falha física no leitor deste equipamento (`requires_replacement=True`). "
                        "Acionaremos automaticamente o protocolo de suporte Nível 2 para substituição."
                        if diag.get("requires_hardware_replacement")
                        else ""
                    )
                    parts.append(
                        f"Executamos o diagnóstico de telemetria no terminal **{term_model}** (ID: `{term_id}`){fleet_note}:\n\n"
                        f"- **Status Atual:** {term.get('status')}\n"
                        f"- **Conectividade:** {conn_type} — Sinal: **{signal}** | Firmware: `{term.get('firmware_version')}`\n\n"
                        f"**Procedimento Técnico para Restabelecimento de Conexão:**\n"
                        f"{steps_str}{hw_alert}"
                    )

        if "execute_terminal_remote_action" in collected_data:
            act = collected_data["execute_terminal_remote_action"]
            if act.get("action_id"):
                replay_tag = " *(Replay Idempotente — comando já registrado anteriormente)*" if act.get("idempotent_replay") else ""
                parts.append(
                    f"**Comando Remoto Executado:** `{act.get('action_type')}` enviado para o terminal `{act.get('terminal_id')}` "
                    f"(Protocolo de Ação: `{act.get('action_id')}`, Status: `{act.get('status')}`){replay_tag}."
                )

        if "get_customer_profile" in collected_data and not parts:
            prof = collected_data["get_customer_profile"]
            if not prof.get("found"):
                parts.append(f"Cliente `{user_id}` não encontrado na base cadastral.")
            else:
                dom = prof.get("domicilio_bancario", {})
                parts.append(
                    f"**Perfil Cadastral Getnet — {prof.get('company_name')} (`{user_id}`)**\n\n"
                    f"- **CNPJ (Mascarado):** `{prof.get('cnpj')}`\n"
                    f"- **Plano Ativo:** {prof.get('account_plan')} | **Antecipação Automática:** {'Ativa' if prof.get('antecipacao_automatica') else 'Inativa'}\n"
                    f"- **Domicílio Bancário:** {dom.get('banco')} (Agência `{dom.get('agencia')}` / Conta `{dom.get('conta_corrente')}`)\n"
                    f"- **Terminais Habilitados ({prof.get('terminals_count')}):** "
                    + ", ".join(f"`{t['terminal_id']}` ({t['model']} - {t['status']})" for t in prof.get("terminals_summary", []))
                )

        final_text = "\n\n".join(parts) if parts else f"Atendimento operacional concluído para `{user_id}`."
        if tracer:
            tracer.record_tokens(prompt_text=message, completion_text=final_text)
        return final_text
