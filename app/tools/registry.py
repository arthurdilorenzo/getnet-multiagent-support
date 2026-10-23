from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError
from app.config import settings
from app.memory.store import state_store
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
    required_scopes: Tuple[str, ...] = ()


class ToolRegistry:
    """Validated, scoped tools with bounded workers and persisted execution status.

    A thread deadline ends waiting, not an in-flight write. Such a write is
    reported as unknown/non-retryable and its operation can be reconciled later.
    """

    def __init__(self, max_workers: int = 8, max_pending: int = 8) -> None:
        if max_workers < 1 or max_pending < 0:
            raise ValueError('Invalid tool worker capacity')
        self._tools: Dict[str, RegisteredTool] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._capacity = threading.BoundedSemaphore(max_workers + max_pending)

    def register(self, name: str, description: str, args_schema: Type[BaseModel],
                 handler: Callable[..., Dict[str, Any]], requires_tenant_context: bool = False,
                 is_write_action: bool = False, timeout_sec: float = settings.TOOL_TIMEOUT_SECONDS,
                 required_scopes: Optional[List[str]] = None) -> None:
        self._tools[name] = RegisteredTool(name=name, description=description, args_schema=args_schema,
            handler=handler, requires_tenant_context=requires_tenant_context,
            is_write_action=is_write_action, timeout_sec=timeout_sec,
            required_scopes=tuple(required_scopes or ()))

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def list_tools(self, names: Optional[List[str]] = None) -> List[RegisteredTool]:
        return list(self._tools.values()) if names is None else [self._tools[n] for n in names if n in self._tools]

    def get_gemini_function_declarations(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        declarations = []
        for tool in self.list_tools(names):
            schema = tool.args_schema.model_json_schema()
            properties = dict(schema.get('properties', {}))
            required = list(schema.get('required', []))
            for protected in ('user_id', 'session_id'):
                properties.pop(protected, None)
                if protected in required:
                    required.remove(protected)
            declarations.append({'name': tool.name, 'description': tool.description, 'parameters': {
                **{k: v for k, v in schema.items() if k not in {'properties', 'required', 'title'}},
                'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}})
        return declarations

    @staticmethod
    def _schema(conn) -> None:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS tool_operations (
                operation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                required_scopes_json TEXT NOT NULL,
                namespace_key TEXT UNIQUE,
                payload_hash TEXT NOT NULL,
                outcome_state TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        ''')

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _scope_allowed(scopes, required) -> bool:
        # A role alone never grants scopes. Only explicit admin:all does.
        return 'admin:all' in scopes or set(required).issubset(set(scopes))

    def get_operation(self, operation_id: str, security_context: SecurityContext) -> dict:
        with state_store._lock, state_store._connect() as conn:
            self._schema(conn)
            row = conn.execute('SELECT * FROM tool_operations WHERE operation_id=?', (operation_id,)).fetchone()
            if not row or row['user_id'] != security_context.authenticated_user_id:
                raise CrossTenantAccessError('Operação não encontrada para o principal autenticado.')
            if not self._scope_allowed(security_context.scopes, json.loads(row['required_scopes_json'])):
                raise PermissionError('Escopo insuficiente para consultar esta operação.')
            return {'operation_id': row['operation_id'], 'tool_name': row['tool_name'],
                    'outcome_state': row['outcome_state'], 'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'result': json.loads(row['result_json']) if row['result_json'] else None}

    def _existing(self, namespace_key: Optional[str], payload_hash: str) -> Optional[ToolExecutionResult]:
        if not namespace_key:
            return None
        with state_store._lock, state_store._connect() as conn:
            self._schema(conn)
            row = conn.execute('SELECT * FROM tool_operations WHERE namespace_key=?', (namespace_key,)).fetchone()
            if not row:
                return None
            if row['payload_hash'] != payload_hash:
                return ToolExecutionResult(tool_name=row['tool_name'], success=False,
                    error_code='IDEMPOTENCY_CONFLICT', error_message='Conflito de idempotência: a chave já identifica outro payload.',
                    retryable=False, operation_id=row['operation_id'], outcome_state='failed')
            if row['result_json']:
                result = ToolExecutionResult.model_validate_json(row['result_json'])
                return result.model_copy(update={'idempotent_replay': True})
            return ToolExecutionResult(tool_name=row['tool_name'], success=False,
                error_code='TOOL_OPERATION_PENDING', error_message='A operação ainda está pendente ou tem resultado incerto; consulte seu operation_id antes de uma nova tentativa.',
                retryable=False, operation_id=row['operation_id'], outcome_state=row['outcome_state'])

    @staticmethod
    def _save_result(result: ToolExecutionResult) -> None:
        with state_store._lock, state_store._connect() as conn:
            conn.execute('UPDATE tool_operations SET outcome_state=?,result_json=?,updated_at=? WHERE operation_id=?',
                (result.outcome_state, result.model_dump_json(), ToolRegistry._now(), result.operation_id))
            conn.commit()

    def execute(self, tool_name: str, raw_arguments: Dict[str, Any], security_context: SecurityContext,
                session_id: str = '', trace_id: str = '') -> ToolExecutionResult:
        start = time.perf_counter()
        def reject(code: str, message: str, retryable: bool = False, **kwargs) -> ToolExecutionResult:
            metrics_collector.record_tool_error(tool_name)
            return ToolExecutionResult(tool_name=tool_name, success=False, error_code=code,
                error_message=PIIRedactor.mask_text(message), retryable=retryable,
                execution_ms=round((time.perf_counter() - start) * 1000, 2), outcome_state='failed', **kwargs)
        tool = self._tools.get(tool_name)
        if not tool:
            return reject('TOOL_NOT_FOUND', 'Ferramenta não registrada no sistema.')
        if not self._scope_allowed(security_context.scopes, tool.required_scopes):
            return reject('INSUFFICIENT_TOOL_SCOPE', 'O principal autenticado não possui os escopos necessários para esta ferramenta.')
        if 'session_id' in tool.args_schema.model_fields:
            session_id = session_id or f'sess_{security_context.authenticated_user_id}'
        if not isinstance(raw_arguments, dict) or not all(isinstance(k, str) for k in raw_arguments):
            return reject('INVALID_TOOL_ARGUMENTS', 'Os argumentos da ferramenta devem ser um objeto JSON.')
        args = dict(raw_arguments)
        if 'user_id' in args and args['user_id'] != security_context.authenticated_user_id:
            logger.warning('tool_cross_tenant_blocked', trace_id=trace_id, tool_name=tool_name)
            return reject('CROSS_TENANT_ACCESS_DENIED', 'A ferramenta não pode selecionar outra conta.')
        if 'session_id' in args and args['session_id'] != session_id:
            return reject('CROSS_SESSION_ACCESS_DENIED', 'A ferramenta não pode selecionar outra sessão.')
        args.pop('user_id', None)
        args.pop('session_id', None)
        if tool.requires_tenant_context:
            args['user_id'] = security_context.authenticated_user_id
        if 'session_id' in tool.args_schema.model_fields:
            args['session_id'] = session_id
        if set(args) - set(tool.args_schema.model_fields):
            return reject('INVALID_TOOL_ARGUMENTS', 'A chamada contém parâmetros não declarados.')
        try:
            validated = tool.args_schema.model_validate(args, strict=True).model_dump()
        except ValidationError:
            # ValidationError includes raw invalid values; never copy it into a trace or FunctionResponse.
            return reject('INVALID_TOOL_ARGUMENTS', 'Parâmetros inválidos para o schema declarado da ferramenta.')
        namespace_key = None
        if tool.is_write_action and validated.get('idempotency_key'):
            identity = json.dumps([tool_name, security_context.authenticated_user_id, session_id, validated['idempotency_key']],
                                  ensure_ascii=False, separators=(',', ':'))
            namespace_key = hashlib.sha256(identity.encode()).hexdigest()
            validated['idempotency_key'] = namespace_key
        payload_hash = hashlib.sha256(json.dumps(validated, sort_keys=True, ensure_ascii=False,
                                                separators=(',', ':')).encode()).hexdigest()
        existing = self._existing(namespace_key, payload_hash)
        if existing:
            return existing
        if not self._capacity.acquire(blocking=False):
            return reject('TOOL_QUEUE_FULL', 'A fila de ferramentas está cheia; nenhuma execução foi iniciada.', retryable=True)
        operation_id = f'OP-{uuid.uuid4().hex.upper()}'
        try:
            with state_store._lock, state_store._connect() as conn:
                self._schema(conn)
                conn.execute('BEGIN IMMEDIATE')
                # Serialize competing identical requests before dispatching either worker.
                existing = self._existing(namespace_key, payload_hash)
                if existing:
                    self._capacity.release()
                    return existing
                now = self._now()
                conn.execute('''INSERT INTO tool_operations (operation_id,user_id,tool_name,required_scopes_json,
                    namespace_key,payload_hash,outcome_state,created_at,updated_at) VALUES (?,?,?,?,?,?,'pending',?,?)''',
                    (operation_id, security_context.authenticated_user_id, tool_name,
                     json.dumps(tool.required_scopes), namespace_key, payload_hash, now, now))
                conn.commit()
        except Exception:
            self._capacity.release()
            raise

        def invoke() -> ToolExecutionResult:
            try:
                data = PIIRedactor.mask_structure(tool.handler(**PIIRedactor.mask_structure(validated)))
                result = ToolExecutionResult(tool_name=tool_name, success=True, data=data,
                    idempotent_replay=bool(isinstance(data, dict) and data.get('idempotent_replay')),
                    operation_id=operation_id, outcome_state='completed',
                    execution_ms=round((time.perf_counter() - start) * 1000, 2))
            except CrossTenantAccessError:
                result = reject('CROSS_TENANT_ACCESS_DENIED', 'Acesso negado ao recurso solicitado.', operation_id=operation_id)
            except (CircuitBreakerOpenError, ExternalServiceUnavailableError):
                result = reject('EXTERNAL_SERVICE_UNAVAILABLE', 'O serviço externo está indisponível.', retryable=not tool.is_write_action,
                                operation_id=operation_id)
                if tool.is_write_action:
                    result.outcome_state = 'unknown'
            except Exception as exc:
                from app.security.action_policy import ActionPolicyError
                code = exc.code if isinstance(exc, ActionPolicyError) else 'TOOL_EXECUTION_ERROR'
                message = str(exc) if isinstance(exc, ActionPolicyError) else 'A ferramenta não pôde concluir a operação.'
                logger.error('tool_execution_exception', trace_id=trace_id, tool_name=tool_name, error_type=type(exc).__name__)
                result = reject(code, message, operation_id=operation_id)
                if tool.is_write_action and not isinstance(exc, ActionPolicyError):
                    # An exception after invocation is not proof that the remote
                    # side rolled back. Only explicit policy denials are failed.
                    result.outcome_state = 'unknown'
            self._save_result(result)
            return result

        try:
            future = self._executor.submit(invoke)
        except Exception:
            self._capacity.release()
            result = reject('TOOL_EXECUTION_ERROR', 'A execução não pôde ser iniciada.', operation_id=operation_id)
            self._save_result(result)
            return result
        future.add_done_callback(lambda _: self._capacity.release())
        try:
            return future.result(timeout=tool.timeout_sec)
        except FuturesTimeoutError:
            if future.cancel():
                result = reject('TOOL_TIMEOUT', 'A execução expirou na fila e foi cancelada antes de iniciar.', retryable=True,
                                operation_id=operation_id)
                self._save_result(result)
                return result
            # A finished worker may have crossed the deadline concurrently. Never
            # overwrite its terminal result with a stale unknown/pending marker.
            state = 'unknown' if tool.is_write_action else 'pending'
            with state_store._lock, state_store._connect() as conn:
                conn.execute('UPDATE tool_operations SET outcome_state=?,updated_at=? WHERE operation_id=? AND result_json IS NULL',
                             (state, self._now(), operation_id))
                conn.commit()
            metrics_collector.record_tool_error(tool_name)
            return ToolExecutionResult(tool_name=tool_name, success=False, error_code='TOOL_TIMEOUT',
                error_message=('Resultado da escrita desconhecido; não repita a ação. Consulte seu operation_id.'
                               if tool.is_write_action else 'A consulta excedeu o prazo de espera.'),
                retryable=not tool.is_write_action, operation_id=operation_id, outcome_state=state,
                execution_ms=round((time.perf_counter() - start) * 1000, 2))


tool_registry = ToolRegistry()
