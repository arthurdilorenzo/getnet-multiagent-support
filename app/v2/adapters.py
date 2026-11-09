"""Infrastructure adapters and Fact extraction, without legacy interpretation or public projection."""
import copy
import hashlib
import re
import json
from pathlib import Path
from typing import Any, Optional

from app.hitl.queue import HITLQueueManager
from app.memory.store import StateStore
from app.schemas import ToolExecutionResult
from app.security.auth import SecurityContext
from app.security.pii import PIIRedactor
from app.tools.registry import ToolRegistry
from app.tools.customer_tools import (
    GetCustomerProfileInput,
    DiagnoseTerminalIssueInput,
    GetSalesSettlementInput,
    CheckEscalationTicketInput,
    ExecuteTerminalRemoteActionInput,
)
from app.tools.search_tools import (
    WeatherForecastInput,
    CurrencyExchangeInput,
    GeneralWebSearchInput,
    search_weather,
    search_currency_rate,
    general_web_search,
)
from app.v2.contracts import Binding, Fact


class StateAdapter:
    """Reuse SQLite/ownership while bypassing append_message's lexical extraction."""

    def __init__(self, store: StateStore):
        self.store = store

    def append(self, session_id, user_id, role, content, tools=None, sources=None, connection=None):
        from contextlib import nullcontext

        if role not in {"user", "assistant", "human_operator"}:
            raise ValueError("Papel de mensagem inválido.")
        if connection is None:
            self.store.ensure_session(session_id, user_id)
        with self.store._lock, (nullcontext(connection) if connection is not None else self.store._connect()) as conn:
            if connection is None:
                conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT user_id FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            self.store._assert_owner(row, user_id)
            now = self.store._now_iso()
            conn.execute(
                "INSERT INTO messages (session_id,user_id,role,content,agent_used,tools_called_json,sources_json,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    user_id,
                    role,
                    PIIRedactor.mask_text(content),
                    "V2_SUPERVISOR",
                    json.dumps(tools or []),
                    json.dumps(sources or []),
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET turn_count=turn_count+?, last_agent='V2_SUPERVISOR', updated_at=? WHERE session_id=?",
                (int(role == "user"), now, session_id),
            )

    def context(self, session_id, user_id):
        ctx = self.store.get_conversation_context(session_id, user_id)
        return {
            "recent_messages": ctx["recent_messages"],
            "summary": ctx["summary"],
            "status": ctx["status"],
            "v2": ctx["session_facts"].get("v2", {}),
        }

    def save(self, session_id, user_id, value):
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            self.store._assert_owner(row, user_id)
            facts = json.loads(row["session_facts_json"] or "{}")
            payload = dict(value)
            payload.setdefault("version", "2.0")
            facts["v2"] = PIIRedactor.mask_structure(payload)
            conn.execute(
                "UPDATE sessions SET session_facts_json=?,updated_at=? WHERE session_id=?",
                (json.dumps(facts, ensure_ascii=False), self.store._now_iso(), session_id),
            )

    def finish_resumption(self, session_id, user_id, ticket_id, reservation_id, response, outcome):
        self.store.finish_hitl_resumption(
            session_id, user_id, ticket_id, reservation_id,
            response.model_dump(mode="json"), "v2_support", "operator", status=outcome,
        )


# Compatibility alias: ToolRegistry now supports store injection directly
IsolatedToolRegistry = ToolRegistry


class ToolGateway:
    """Tenant-bound reads and immutable proposals on the existing scopes registry.

    Data views preserve business extras from the authenticated fixture.
    No model may select another customer; terminal confirmation is HTTP-only.
    Uses ToolRegistry with injected store - NO subclassing and NO global monkeypatching.
    """

    TOOL_DESCRIPTORS = [
        {
            "tool_name": "get_customer_profile",
            "resource_type": "customer",
            "fields": [
                {"field_pointer": "/profile/status", "value_type": "str", "unit": None, "description": "Status cadastral do lojista"},
                {"field_pointer": "/profile/segment", "value_type": "str", "unit": None, "description": "Segmento comercial"},
            ],
        },
        {
            "tool_name": "diagnose_terminal_issue",
            "resource_type": "terminal",
            "fields": [
                {"field_pointer": "/terminal/status", "value_type": "str", "unit": None, "description": "Status operacional da máquina"},
                {"field_pointer": "/terminal/connectivity", "value_type": "str", "unit": None, "description": "Tipo de conectividade da máquina"},
                {"field_pointer": "/terminal/model", "value_type": "str", "unit": None, "description": "Modelo da máquina"},
                {"field_pointer": "/ultimo_erro_transacao", "value_type": "str", "unit": None, "description": "Último código de erro de transação registrado"},
                {"field_pointer": "/chip_algar_status", "value_type": "str", "unit": None, "description": "Status do chip da operadora"},
                {"field_pointer": "/observacao_sistema", "value_type": "str", "unit": None, "description": "Observação técnica de sistema"},
                {"field_pointer": "/compartimento_detectado", "value_type": "str", "unit": None, "description": "Compartimento detectado"},
            ],
        },
        {
            "tool_name": "get_sales_settlement",
            "resource_type": "customer",
            "fields": [
                {"field_pointer": "/sales_data/saldo_disponivel_brl", "value_type": "decimal", "unit": "BRL", "description": "Saldo disponível para liquidação em reais"},
                {"field_pointer": "/sales_data/net_payout_scheduled_today_brl", "value_type": "decimal", "unit": "BRL", "description": "Valor líquido programado para hoje em reais"},
                {"field_pointer": "/sales_data/net_amount_brl", "value_type": "decimal", "unit": "BRL", "description": "Valor líquido apurado em reais"},
                {"field_pointer": "/sales_data/gross_amount_brl", "value_type": "decimal", "unit": "BRL", "description": "Valor bruto de vendas em reais"},
                {"field_pointer": "/sales_data/date", "value_type": "str", "unit": None, "description": "Data do snapshot de liquidação"},
            ],
        },
        {
            "tool_name": "check_escalation_ticket",
            "resource_type": "ticket",
            "fields": [
                {"field_pointer": "/ticket/status", "value_type": "str", "unit": None, "description": "Status atual do chamado HITL"},
                {"field_pointer": "/ticket/operator_response", "value_type": "str", "unit": None, "description": "Resposta fornecida pelo operador humano"},
            ],
        },
        {
            "tool_name": "get_weather_forecast",
            "resource_type": "weather",
            "fields": [
                {"field_pointer": "/city", "value_type": "str", "unit": None, "description": "Nome da cidade"},
                {"field_pointer": "/target_date", "value_type": "str", "unit": None, "description": "Data da previsão"},
                {"field_pointer": "/forecast_horizon", "value_type": "str", "unit": None, "description": "Horizonte da previsão"},
                {"field_pointer": "/source", "value_type": "str", "unit": None, "description": "Fonte meteorológica observada"},
                {"field_pointer": "/condition", "value_type": "str", "unit": None, "description": "Condição do tempo"},
                {"field_pointer": "/max_temp_celsius", "value_type": "decimal", "unit": "°C", "description": "Temperatura máxima em graus Celsius"},
                {"field_pointer": "/min_temp_celsius", "value_type": "decimal", "unit": "°C", "description": "Temperatura mínima em graus Celsius"},
                {"field_pointer": "/rain_probability_percent", "value_type": "decimal", "unit": "%", "description": "Probabilidade de chuva"},
            ],
        },
        {
            "tool_name": "get_currency_exchange_rate",
            "resource_type": "currency",
            "fields": [
                {"field_pointer": "/pair", "value_type": "str", "unit": None, "description": "Par de moedas (ex: EUR/BRL)"},
                {"field_pointer": "/rate_type", "value_type": "str", "unit": None, "description": "Natureza observada da cotação (mercado ou referência diária)"},
                {"field_pointer": "/rate", "value_type": "decimal", "unit": None, "description": "Cotação de câmbio"},
                {"field_pointer": "/date", "value_type": "str", "unit": None, "description": "Data de referência da cotação"},
                {"field_pointer": "/source", "value_type": "str", "unit": None, "description": "Fonte da cotação"},
            ],
        },
        {
            "tool_name": "execute_terminal_remote_action", "resource_type": "terminal",
            "fields": [
                {"field_pointer": "/" + field, "value_type": "str", "unit": None,
                 "description": description}
                for field, description in (
                    ("proposal_id", "Identificador da proposta; não comprova execução"),
                    ("status", "Estado da proposta; PENDING_CONFIRMATION aguarda consentimento"),
                    ("action_type", "Ação proposta"), ("payload_hash", "Hash para confirmação autenticada"),
                    ("expires_at", "Expiração da proposta"),
                    ("confirmation_instruction", "Instrução de confirmação; nenhum comando enviado"),
                )
            ],
        },
    ]

    def __init__(self, customers_path: Path, *, store: StateStore, queue: HITLQueueManager):
        self.path = Path(customers_path)
        self.store = store
        self.queue = queue
        self.registry = ToolRegistry(store=self.store)
        for name, schema, handler, req_tenant, scopes in [
            ("get_customer_profile", GetCustomerProfileInput, self.profile, True, ["customer:read_own"]),
            ("diagnose_terminal_issue", DiagnoseTerminalIssueInput, self.diagnose, True, ["customer:read_own"]),
            ("get_sales_settlement", GetSalesSettlementInput, self.settlement, True, ["customer:read_own"]),
            ("check_escalation_ticket", CheckEscalationTicketInput, self.tickets, True, ["customer:read_own"]),
            ("execute_terminal_remote_action", ExecuteTerminalRemoteActionInput, self.action, True, ["customer:read_own", "chat:write", "terminal:action_own"]),
            ("get_weather_forecast", WeatherForecastInput, self.weather, False, []),
            ("get_currency_exchange_rate", CurrencyExchangeInput, self.currency, False, []),
            ("web_search_general", GeneralWebSearchInput, self.web_search, False, []),
        ]:
            self.registry.register(
                name,
                schema.__name__,
                schema,
                handler,
                requires_tenant_context=req_tenant,
                required_scopes=scopes,
                is_write_action=name == "execute_terminal_remote_action",
            )

    def declarations(self):
        return self.registry.get_gemini_function_declarations()

    def get_tool_descriptors(self, user_id=None) -> list[dict]:
        """Expose field names from this tenant's actual records, never invented slots."""
        descriptors = copy.deepcopy(self.TOOL_DESCRIPTORS)
        if user_id is None:
            return descriptors
        customer = self._customer(user_id) or {}
        terminals = customer.get("terminals", [])
        for descriptor in descriptors:
            name = descriptor["tool_name"]
            if name not in {"get_customer_profile", "diagnose_terminal_issue"}:
                continue
            known = {f["field_pointer"]: f for f in descriptor["fields"]}
            available = {}
            if name == "get_customer_profile":
                descriptor["resource_ids"] = [user_id]
                available = {"/profile/" + k: v for k, v in customer.items()
                             if isinstance(v, (str, int, float, bool)) or v is None}
            else:
                descriptor["resource_ids"] = [t["terminal_id"] for t in terminals if t.get("terminal_id")]
                for terminal in terminals:
                    available.update({"/terminal/" + k: v for k, v in terminal.items()
                                      if isinstance(v, (str, int, float, bool)) or v is None})
                    for field in ("ultimo_erro_transacao", "chip_algar_status", "observacao_sistema", "compartimento_detectado"):
                        if field in terminal:
                            available["/" + field] = terminal[field]
                        elif len(terminals) == 1 and field in customer:
                            available["/" + field] = customer[field]
            descriptor["fields"] = [known.get(pointer) or {
                "field_pointer": pointer, "value_type": self._scalar_type(value),
                "unit": None, "description": pointer.rsplit("/", 1)[-1].replace("_", " "),
            } for pointer, value in available.items()]
        return descriptors

    @staticmethod
    def _scalar_type(value):
        if isinstance(value, bool): return "bool"
        if isinstance(value, int): return "int"
        if isinstance(value, float): return "decimal"
        if value is None: return "null"
        return "str"

    def resolve_resource_id(self, resource_type, resource_id, user_id, arguments=None, result=None):
        """Resolve explicit symbolic aliases before extraction; concrete IDs stay exact."""
        arguments = arguments or {}
        data = (result.data or {}) if result is not None else {}
        if resource_type == "customer" and resource_id in {"current", "current_customer"}:
            return user_id
        if resource_type == "terminal" and resource_id in {"current", "current_terminal"}:
            terminals = (self._customer(user_id) or {}).get("terminals", [])
            selected = arguments.get("terminal_id")
            ids = [t["terminal_id"] for t in terminals if t.get("terminal_id")]
            if selected in ids:
                return selected
            if len(ids) == 1:
                return ids[0]
        # Public geocoding canonicalizes only the city actually requested by this call.
        if resource_type == "weather" and resource_id == arguments.get("city") and data.get("city"):
            return data["city"]
        if resource_type == "currency":
            pair = data.get("currency_pair") or data.get("pair")
            if pair:
                return pair
        return resource_id

    def _customer(self, user_id):
        if not self.path.exists():
            return None
        records = json.loads(self.path.read_text(encoding="utf-8")).get("customers", {})
        value = records.get(user_id)
        if value is None:
            return None

        def clean(val):
            if isinstance(val, dict):
                return {
                    k: ("[REDACTED]" if any(s in k.lower() for s in ("password", "senha", "secret", "api_key", "token")) else clean(v))
                    for k, v in val.items()
                }
            if isinstance(val, list):
                return [clean(v) for v in val]
            return val

        return PIIRedactor.mask_structure(clean(value))

    def profile(self, user_id):
        record = self._customer(user_id)
        return {"found": record is not None, "user_id": user_id, "profile": record or {}}

    def diagnose(self, user_id, terminal_id=None, transaction_id=None, terminal_model=None,
                 symptom="general", decline_code=None, error_context=""):
        customer = self._customer(user_id) or {}
        terminals = customer.get("terminals", [])
        selected = [t for t in terminals if t.get("terminal_id") == terminal_id] if terminal_id else terminals
        if terminal_id is None and len(selected) != 1:
            return {"found": False, "user_id": user_id, "error_code": "TERMINAL_NEEDS_CLARIFICATION"}
        terminal = selected[0] if selected else None
        transactions = customer.get("recent_transactions", [])
        if transaction_id:
            transaction = next(
                (t for t in transactions if t.get("transaction_id") == transaction_id and
                 (not terminal or t.get("terminal_id") == terminal.get("terminal_id"))),
                None,
            )
        else:
            transaction = transactions[0] if transactions else None

        diagnostic_fields = {}
        for field in ("ultimo_erro_transacao", "chip_algar_status", "observacao_sistema", "compartimento_detectado"):
            if terminal and field in terminal:
                diagnostic_fields[field] = terminal[field]
            elif terminal and len(terminals) == 1 and field in customer:
                diagnostic_fields[field] = customer[field]

        return {
            "found": terminal is not None or bool(transactions),
            "user_id": user_id,
            "requested_terminal_id": terminal_id,
            "terminal": terminal,
            "requested_transaction_id": transaction_id,
            "transaction": transaction,
            **diagnostic_fields,
        }

    def settlement(self, user_id, payment_modality="all", sales_date=None):
        customer = self._customer(user_id) or {}
        snapshot = customer.get("sales_settlement", {}).get("yesterday_sales") or customer.get("agenda_recebiveis")
        found = snapshot is not None and (not sales_date or snapshot.get("date") == sales_date)
        return {
            "found": found,
            "user_id": user_id,
            "data_scope": "synthetic_historical_snapshot",
            "sales_data": snapshot if found else None,
            "temporal_notice": "Snapshot histórico; não confirma liquidação atual.",
        }

    def tickets(self, user_id, ticket_id=None):
        if ticket_id:
            ticket = self.queue.get_ticket(ticket_id)
            found = bool(ticket and ticket.user_id == user_id)
            return {
                "found": found,
                "ticket": ticket.model_dump() if found else None,
                "tickets": [ticket.model_dump()] if found else [],
            }
        tickets = self.queue.list_tickets(user_id=user_id)
        return {"found": bool(tickets), "tickets": [t.model_dump() for t in tickets]}

    def action(self, user_id, terminal_id, action_type, reason="Diagnóstico automatizado de suporte", idempotency_key="", session_id=""):
        from app.security.action_policy import propose_action
        proposal = propose_action(
            user_id=user_id,
            session_id=session_id,
            terminal_id=terminal_id,
            action_type=action_type,
            reason=reason,
            idempotency_key=idempotency_key,
            store=self.store,
            customers_path=self.path,
        )
        return proposal

    def weather(self, city="Porto Alegre", days_ahead=1):
        from app.tools.search_tools import search_weather
        return search_weather(city=city, days_ahead=days_ahead)

    def currency(self, base_currency="EUR", target_currency="BRL"):
        from app.tools.search_tools import search_currency_rate
        return search_currency_rate(base_currency=base_currency, target_currency=target_currency)

    def web_search(self, query="", max_results=3):
        from app.tools.search_tools import general_web_search
        return general_web_search(query=query, max_results=max_results)

    def execute(self, name, arguments, context, session_id, trace_id=""):
        self.store.assert_session_access(session_id, context.authenticated_user_id)
        return self.registry.execute(name, arguments, context, session_id, trace_id)

    def extract_facts(self, tool_name: str, result: ToolExecutionResult, bindings: list[Binding]) -> list[Fact]:
        """Server-side extraction of verified facts from completed tool execution.

        Facts are strictly anchored in actual observed records and fields, never prose.
        """
        if not result.success or result.outcome_state != "completed":
            return []
        data = result.data or {}
        if data.get("found") is False:
            return []

        observed_facts: dict[tuple[str, str, str], tuple[Any, str, Optional[str]]] = {}
        # key: (resource_type, resource_id, field_pointer) -> (value, value_type, unit)

        def infer_type(val: Any) -> str:
            if isinstance(val, bool):
                return "bool"
            if isinstance(val, int):
                return "int"
            if isinstance(val, float):
                return "decimal"
            if isinstance(val, str):
                return "str"
            if val is None:
                return "null"
            return "str"

        if tool_name == "get_customer_profile":
            user_id = data.get("user_id", "")
            profile = data.get("profile", {})
            for k, v in profile.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    observed_facts[("customer", user_id, f"/profile/{k}")] = (v, infer_type(v), None)

        elif tool_name == "diagnose_terminal_issue":
            terminal = data.get("terminal")
            # Resource ID is the actual selected terminal ID from the database record!
            term_id = terminal.get("terminal_id", "") if terminal else ""
            if terminal:
                for k, v in terminal.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        observed_facts[("terminal", term_id, f"/terminal/{k}")] = (v, infer_type(v), None)
            # Top-level operational fields on customer terminal diagnostic
            for field in ("ultimo_erro_transacao", "chip_algar_status", "observacao_sistema", "compartimento_detectado"):
                v = data.get(field)
                if v is None and terminal:
                    v = terminal.get(field)
                if term_id:
                    observed_facts[("terminal", term_id, f"/{field}")] = (v, infer_type(v), None)

        elif tool_name == "get_sales_settlement":
            user_id = data.get("user_id", "")
            sales = data.get("sales_data") or {}
            for k, v in sales.items():
                unit = "BRL" if k.endswith("_brl") or k.endswith("_reais") else None
                observed_facts[("customer", user_id, f"/sales_data/{k}")] = (v, infer_type(v), unit)

        elif tool_name == "check_escalation_ticket":
            ticket = data.get("ticket")
            ticket_id = ticket.get("ticket_id", "") if ticket else ""
            if ticket:
                for k, v in ticket.items():
                    if k == "operator_response":
                        continue  # Human guidance is presented only through attributed human spans.
                    observed_facts[("ticket", ticket_id, f"/ticket/{k}")] = (v, infer_type(v), None)

        elif tool_name == "get_weather_forecast":
            city = data.get("city")
            if not city:
                return []
            max_t = data.get("max_temp_celsius")
            min_t = data.get("min_temp_celsius")
            rain = data.get("rain_probability_percent")
            cond = data.get("condition")
            t_date = data.get("target_date")
            horizon = data.get("forecast_horizon")
            res_id = city
            observed_facts[("weather", res_id, "/source")] = (data.get("source"), "str", None)
            observed_facts[("weather", res_id, "/city")] = (city, "str", None)
            observed_facts[("weather", res_id, "/target_date")] = (t_date, "str", None)
            observed_facts[("weather", res_id, "/forecast_horizon")] = (horizon, "str", None)
            observed_facts[("weather", res_id, "/condition")] = (cond, "str", None)
            observed_facts[("weather", res_id, "/max_temp_celsius")] = (max_t, infer_type(max_t), "°C")
            observed_facts[("weather", res_id, "/min_temp_celsius")] = (min_t, infer_type(min_t), "°C")
            observed_facts[("weather", res_id, "/rain_probability_percent")] = (rain, infer_type(rain), "%")

        elif tool_name == "get_currency_exchange_rate":
            pair = data.get("currency_pair") or data.get("pair")
            if pair is None and data.get("base_currency") and data.get("target_currency"):
                pair = f"{data['base_currency']}/{data['target_currency']}"
            if not isinstance(pair, str) or not re.fullmatch(r"[A-Z]{3}/[A-Z]{3}", pair):
                return []
            base, target = pair.split("/")
            rate = data.get("current_rate_brl") if data.get("current_rate_brl") is not None else data.get("rate")
            date = data.get("updated_at") or data.get("observation_date") or data.get("date")
            source = data.get("source")
            res_id = pair
            observed_facts[("currency", res_id, "/pair")] = (pair, "str", None)
            observed_facts[("currency", res_id, "/rate")] = (rate, infer_type(rate), target)
            observed_facts[("currency", res_id, "/date")] = (date, "str", None)
            observed_facts[("currency", res_id, "/source")] = (source, "str", None)
            observed_facts[("currency", res_id, "/rate_type")] = (data.get("rate_type"), "str", None)

        elif tool_name == "execute_terminal_remote_action":
            term_id = data.get("terminal_id")
            if term_id:
                for field in ("proposal_id", "status", "action_type", "payload_hash", "expires_at", "confirmation_instruction"):
                    value = data.get(field)
                    observed_facts[("terminal", term_id, "/" + field)] = (value, infer_type(value), None)

        facts: list[Fact] = []
        op_id = result.operation_id

        # Bind observed facts to requested bindings, or auto-bind for public tools (weather/currency)
        effective_bindings = list(bindings)
        if tool_name in ("get_weather_forecast", "get_currency_exchange_rate"):
            has_matching = any(b.tool_name == tool_name for b in bindings)
            if not has_matching:
                desc = next((tool for tool in self.TOOL_DESCRIPTORS if tool["tool_name"] == tool_name), None)
                if desc:
                    res_type = desc["resource_type"]
                    res_id = next((k[1] for k in observed_facts.keys() if k[0] == res_type), res_type)
                    for fld in desc["fields"]:
                        effective_bindings.append(
                            Binding(
                                binding_id=f"auto:{tool_name}:{fld['field_pointer'].strip('/')}",
                                tool_name=tool_name,
                                resource_type=res_type,
                                resource_id=res_id,
                                field_pointer=fld["field_pointer"],
                                unit=observed_facts.get((res_type, res_id, fld["field_pointer"]), (None, None, fld["unit"]))[2],
                            )
                        )

        for binding in effective_bindings:
            if binding.tool_name != tool_name:
                continue

            lookup_key = (binding.resource_type, binding.resource_id, binding.field_pointer)
            if lookup_key in observed_facts:
                val, v_type, unit = observed_facts[lookup_key]
                descriptor = next((field for tool in self.TOOL_DESCRIPTORS
                                   if tool["tool_name"] == tool_name for field in tool["fields"]
                                   if field["field_pointer"] == binding.field_pointer), None)
                expected = descriptor["value_type"] if descriptor else None
                correct_type = expected is None or v_type == expected or (expected == "decimal" and v_type in ("int", "decimal"))
                # Known field types and units come from server output contracts.
                if not correct_type or binding.unit != unit:
                    avail = "missing"
                elif val is None:
                    avail = "unavailable"
                else:
                    avail = "available"

                h_val = json.dumps({"res": binding.resource_id, "ptr": binding.field_pointer, "val": str(val), "u": unit}, sort_keys=True)
                c_hash = hashlib.sha256(h_val.encode()).hexdigest()
                f_id = f"fact_{hashlib.sha256(f'{op_id}:{binding.resource_type}:{binding.resource_id}:{binding.field_pointer}'.encode()).hexdigest()[:12]}"
                
                # Context notice for public / financial / snapshot facts
                if tool_name == "get_sales_settlement":
                    scope_notice = " | ".join(str(x) for x in (data.get("data_scope"), (data.get("sales_data") or {}).get("date"), data.get("temporal_notice")) if x)
                elif tool_name == "get_currency_exchange_rate":
                    scope_notice = " | ".join(str(x) for x in (pair, date, source, data.get("rate_type"), data.get("disclaimer")) if x)
                elif tool_name == "get_weather_forecast":
                    scope_notice = " | ".join(str(x) for x in (data.get("city"), data.get("target_date"), data.get("forecast_horizon"), data.get("source")) if x)
                else:
                    scope_notice = ""

                facts.append(
                    Fact(
                        fact_id=f_id,
                        binding_id=binding.binding_id,
                        operation_id=op_id,
                        resource_type=binding.resource_type,
                        resource_id=binding.resource_id,
                        field_pointer=binding.field_pointer,
                        availability=avail,
                        value=str(val) if isinstance(val, float) else val,
                        value_type=v_type,
                        unit=unit,
                        source_kind="tool",
                        scope_notice=scope_notice,
                        content_hash=c_hash,
                    )
                )
            else:
                # Field or resource missing
                f_id = f"fact_{hashlib.sha256(f'{op_id}:{binding.resource_type}:{binding.resource_id}:{binding.field_pointer}:missing'.encode()).hexdigest()[:12]}"
                facts.append(
                    Fact(
                        fact_id=f_id,
                        binding_id=binding.binding_id,
                        operation_id=op_id,
                        resource_type=binding.resource_type,
                        resource_id=binding.resource_id,
                        field_pointer=binding.field_pointer,
                        availability="missing",
                        value=None,
                        value_type="null",
                        unit=binding.unit,
                        source_kind="tool",
                        content_hash="",
                    )
                )

        return facts
