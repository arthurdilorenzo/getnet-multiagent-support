from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError
from app.config import settings
from app.observability.logger import logger
from app.observability.metrics import metrics_collector
from app.observability.resilience import CircuitBreakerOpenError, ExternalServiceUnavailableError
from app.schemas import ToolExecutionResult
from app.security.auth import CrossTenantAccessError, SecurityContext
from app.security.pii import PIIRedactor


@dataclass
class RegisteredTool:
    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., Dict[str, Any]]
    requires_tenant_context: bool = False
    is_write_action: bool = False
    timeout_sec: float = settings.TOOL_TIMEOUT_SECONDS


class ToolRegistry:
    """
    Registro Central de Ferramentas (Tools) com:
    - Validação estrita de argumentos via Pydantic Schema.
    - Exportação de declarações JSON Schema para Function Calling nativo (Gemini).
    - Isolamento Zero-Trust de `user_id` via `SecurityContext` (o LLM nunca escolhe outro tenant).
    - Timeout por execução de tool, tratamento estruturado de erros e redaction de PII.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}
        self._executor = ThreadPoolExecutor(max_workers=8)

    def register(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        handler: Callable[..., Dict[str, Any]],
        requires_tenant_context: bool = False,
        is_write_action: bool = False,
        timeout_sec: float = settings.TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            args_schema=args_schema,
            handler=handler,
            requires_tenant_context=requires_tenant_context,
            is_write_action=is_write_action,
            timeout_sec=timeout_sec,
        )

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def list_tools(self, names: Optional[List[str]] = None) -> List[RegisteredTool]:
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def get_gemini_function_declarations(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Gera declarações de Function Calling compatíveis com Google GenAI SDK / OpenAI Schema.
        Nota de Segurança: o campo `user_id` nunca é exposto para escolha livre do LLM;
        ele é injetado pelo `ToolRegistry` a partir do `SecurityContext` autenticado.
        """
        declarations: List[Dict[str, Any]] = []
        for tool in self.list_tools(names):
            raw_schema = tool.args_schema.model_json_schema()
            properties = dict(raw_schema.get("properties", {}))
            required = list(raw_schema.get("required", []))
            # Remove user_id e session_id da escolha livre do LLM
            for protected_field in ("user_id", "session_id"):
                properties.pop(protected_field, None)
                if protected_field in required:
                    required.remove(protected_field)

            declarations.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return declarations

    def execute(
        self,
        tool_name: str,
        raw_arguments: Dict[str, Any],
        security_context: SecurityContext,
        session_id: str = "",
        trace_id: str = "",
    ) -> ToolExecutionResult:
        start = time.perf_counter()
        tool = self._tools.get(tool_name)
        if not tool:
            metrics_collector.record_tool_error(tool_name)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_code="TOOL_NOT_FOUND",
                error_message=f"Ferramenta '{tool_name}' não está registrada no sistema.",
                retryable=False,
                execution_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        args = dict(raw_arguments or {})

        # 1. Verificação de Segurança e Isolamento de Tenant (Zero-Trust Identity Binding)
        if tool.requires_tenant_context:
            if "user_id" in args and args["user_id"]:
                try:
                    verified_uid = security_context.assert_tenant_access(str(args["user_id"]))
                except CrossTenantAccessError as exc:
                    metrics_collector.record_tool_error(tool_name)
                    logger.warning(
                        "tool_cross_tenant_blocked",
                        trace_id=trace_id,
                        tool_name=tool_name,
                        authenticated_user=security_context.authenticated_user_id,
                        attempted_user=args.get("user_id"),
                    )
                    return ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error_code="CROSS_TENANT_ACCESS_DENIED",
                        error_message=str(exc),
                        retryable=False,
                        execution_ms=round((time.perf_counter() - start) * 1000, 2),
                    )
                args["user_id"] = verified_uid
            else:
                args["user_id"] = security_context.authenticated_user_id

            if "session_id" in tool.args_schema.model_fields and not args.get("session_id"):
                args["session_id"] = session_id or f"sess_{security_context.authenticated_user_id}"

        # 2. Validação de Schema via Pydantic
        try:
            validated_model = tool.args_schema.model_validate(args)
            validated_kwargs = validated_model.model_dump()
        except ValidationError as exc:
            metrics_collector.record_tool_error(tool_name)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_code="INVALID_TOOL_ARGUMENTS",
                error_message=f"Erro de validação nos parâmetros da tool '{tool_name}': {exc}",
                retryable=False,
                execution_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # 3. Execução com Timeout Estrito e Tratamento de Erros de Resiliência
        try:
            future = self._executor.submit(tool.handler, **validated_kwargs)
            raw_output = future.result(timeout=tool.timeout_sec)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            masked_output = PIIRedactor.mask_structure(raw_output)
            is_replay = bool(isinstance(masked_output, dict) and masked_output.get("idempotent_replay"))

            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                data=masked_output,
                execution_ms=elapsed_ms,
                idempotent_replay=is_replay,
            )
        except FuturesTimeoutError:
            metrics_collector.record_tool_error(tool_name)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_code="TOOL_TIMEOUT",
                error_message=f"Execução da tool '{tool_name}' excedeu o timeout de {tool.timeout_sec}s.",
                retryable=True,
                execution_ms=elapsed_ms,
            )
        except (CircuitBreakerOpenError, ExternalServiceUnavailableError) as exc:
            metrics_collector.record_tool_error(tool_name)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_code="EXTERNAL_SERVICE_UNAVAILABLE",
                error_message=str(exc),
                retryable=True,
                execution_ms=elapsed_ms,
            )
        except CrossTenantAccessError as exc:
            metrics_collector.record_tool_error(tool_name)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_code="CROSS_TENANT_ACCESS_DENIED",
                error_message=str(exc),
                retryable=False,
                execution_ms=elapsed_ms,
            )
        except Exception as exc:
            metrics_collector.record_tool_error(tool_name)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "tool_execution_exception",
                trace_id=trace_id,
                tool_name=tool_name,
                error=str(exc),
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_code="TOOL_EXECUTION_ERROR",
                error_message=str(exc),
                retryable=False,
                execution_ms=elapsed_ms,
            )


tool_registry = ToolRegistry()
