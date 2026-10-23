"""Server-owned, persisted proposal/confirmation policy for simulated POS actions.

The agent can create proposals only. Authenticated HTTP confirmation (not an LLM
function) is the only caller permitted to turn a proposal into a local receipt.
Principal/scopes must be checked by that HTTP boundary and by ToolRegistry.
"""
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
import time
import uuid
from typing import Any, Dict

from app.memory.store import state_store
from app.security.auth import CrossTenantAccessError
from app.security.pii import PIIRedactor

PROPOSAL_TTL_SECONDS = 300
ALLOWED_ACTIONS = frozenset({'remote_reboot', 'sim_carrier_switch', 'parameter_table_update'})


class ActionPolicyError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _schema(conn) -> None:
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS terminal_action_proposals (
            proposal_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            expires_at REAL NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING_CONFIRMATION',
            receipt_json TEXT,
            UNIQUE(user_id, session_id, idempotency_key)
        );
        CREATE TRIGGER IF NOT EXISTS immutable_terminal_action_proposal
        BEFORE UPDATE OF user_id, session_id, idempotency_key, payload_json,
                         payload_hash, expires_at, created_at ON terminal_action_proposals
        BEGIN SELECT RAISE(ABORT, 'Action proposal payload is immutable'); END;
    ''')


def _session_authorized(conn, session_id: str, user_id: str) -> None:
    row = conn.execute('SELECT user_id FROM sessions WHERE session_id=?', (session_id,)).fetchone()
    if not row or row['user_id'] != user_id:
        raise CrossTenantAccessError('A sessão da proposta não pertence ao cliente autenticado.')


def _human_pause(conn, session_id: str, user_id: str) -> None:
    # The same SQLite transaction guards the check and mutation. Merely checking
    # a session status outside this transaction races with a newly opened ticket.
    pending = conn.execute('''SELECT 1 FROM escalation_tickets WHERE session_id=?
        AND user_id=? AND status IN ('OPEN','CLAIMED','RESPONDED') LIMIT 1''',
        (session_id, user_id)).fetchone()
    if pending:
        raise ActionPolicyError('HUMAN_INTERVENTION_PENDING',
                                'Ação pausada enquanto houver atendimento humano pendente nesta sessão.')


def _terminal(user_id: str, terminal_id: str) -> Dict[str, Any]:
    from app.tools.customer_tools import _load_customers_db
    customer = _load_customers_db().get(user_id)
    target = next((t for t in (customer or {}).get('terminals', [])
                   if t.get('terminal_id', '').casefold() == terminal_id.casefold()), None)
    if not target:
        raise CrossTenantAccessError('O terminal solicitado não pertence à frota autenticada.')
    return target


def _view(row, replay: bool = False) -> Dict[str, Any]:
    payload = json.loads(row['payload_json'])
    return {**payload, 'proposal_id': row['proposal_id'], 'payload_hash': row['payload_hash'],
            'status': row['status'], 'expires_at': _iso(row['expires_at']),
            'idempotent_replay': replay,
            'confirmation_instruction': 'Confirme esta proposta pela API autenticada usando seu proposal_id e payload_hash. Nenhum comando foi enviado.'}


def propose_action(user_id: str, session_id: str, terminal_id: str, action_type: str,
                   reason: str = '', idempotency_key: str = '') -> dict:
    """Create one immutable intent; never dispatch an action from a chat turn."""
    if not all(isinstance(value, str) for value in (user_id, session_id, terminal_id, action_type, reason, idempotency_key)):
        raise ActionPolicyError('INVALID_ACTION_ARGUMENTS', 'Parâmetros da proposta inválidos.')
    if action_type not in ALLOWED_ACTIONS:
        raise ActionPolicyError('INVALID_ACTION_TYPE', 'Tipo de ação remota não permitido.')
    if not user_id.strip() or not session_id.strip() or not terminal_id.strip():
        raise ActionPolicyError('INVALID_ACTION_ARGUMENTS', 'Cliente, sessão e terminal são obrigatórios.')
    if max(len(user_id), len(session_id), len(terminal_id), len(idempotency_key)) > 256 or len(reason) > 2000:
        raise ActionPolicyError('INVALID_ACTION_ARGUMENTS', 'Parâmetros da proposta excedem o limite permitido.')
    terminal = _terminal(user_id, terminal_id.strip())
    state_store.ensure_session(session_id, user_id)
    payload = {'user_id': user_id, 'session_id': session_id,
               'terminal_id': terminal['terminal_id'], 'action_type': action_type,
               'reason': PIIRedactor.mask_text(reason)}
    payload_json = _json(payload)
    payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
    key = idempotency_key.strip() or uuid.uuid4().hex
    now = time.time()
    with state_store._lock, state_store._connect() as conn:
        _schema(conn)
        conn.execute('BEGIN IMMEDIATE')
        _session_authorized(conn, session_id, user_id)
        _human_pause(conn, session_id, user_id)
        existing = conn.execute('''SELECT * FROM terminal_action_proposals
            WHERE user_id=? AND session_id=? AND idempotency_key=?''', (user_id, session_id, key)).fetchone()
        if existing:
            if not hmac.compare_digest(existing['payload_hash'], payload_hash):
                raise ActionPolicyError('IDEMPOTENCY_CONFLICT', 'Conflito de idempotência: a chave já identifica outro payload.')
            return _view(existing, replay=True)
        proposal_id = f'PROP-{uuid.uuid4().hex.upper()}'
        conn.execute('''INSERT INTO terminal_action_proposals
            (proposal_id,user_id,session_id,idempotency_key,payload_json,payload_hash,expires_at,created_at)
            VALUES (?,?,?,?,?,?,?,?)''', (proposal_id, user_id, session_id, key, payload_json,
                                       payload_hash, now + PROPOSAL_TTL_SECONDS, _iso(now)))
        row = conn.execute('SELECT * FROM terminal_action_proposals WHERE proposal_id=?', (proposal_id,)).fetchone()
        conn.commit()
        return _view(row)


def confirm_action(proposal_id: str, user_id: str, expected_payload_hash: str) -> dict:
    """Validate authenticated consent, then record one simulated POS receipt atomically.

    This function is deliberately absent from all model function declarations.
    A receipt proves the local simulated scheduling record, not physical execution.
    """
    if not all(isinstance(v, str) and v for v in (proposal_id, user_id, expected_payload_hash)):
        raise ActionPolicyError('INVALID_CONFIRMATION', 'Confirmação inválida.')
    if not re.fullmatch(r'[0-9a-f]{64}', expected_payload_hash):
        raise ActionPolicyError('PROPOSAL_HASH_MISMATCH', 'Hash da proposta inválido.')
    with state_store._lock, state_store._connect() as conn:
        _schema(conn)
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT * FROM terminal_action_proposals WHERE proposal_id=?', (proposal_id,)).fetchone()
        if not row or row['user_id'] != user_id:
            raise CrossTenantAccessError('Proposta não encontrada para o cliente autenticado.')
        _session_authorized(conn, row['session_id'], user_id)
        actual_hash = hashlib.sha256(row['payload_json'].encode()).hexdigest()
        if not hmac.compare_digest(actual_hash, row['payload_hash']) or not hmac.compare_digest(row['payload_hash'], expected_payload_hash):
            raise ActionPolicyError('PROPOSAL_HASH_MISMATCH', 'Hash da proposta diverge do payload confirmado.')
        # Replay is a read of an already committed receipt, even after expiry.
        if row['receipt_json']:
            receipt = json.loads(row['receipt_json'])
            receipt['idempotent_replay'] = True
            return receipt
        _human_pause(conn, row['session_id'], user_id)
        if time.time() >= row['expires_at']:
            raise ActionPolicyError('PROPOSAL_EXPIRED', 'Proposta expirada; solicite uma nova proposta.')
        if row['status'] != 'PENDING_CONFIRMATION':
            raise ActionPolicyError('INVALID_PROPOSAL_STATE', 'A proposta não aceita confirmação neste estado.')
        payload = json.loads(row['payload_json'])
        terminal = _terminal(user_id, payload['terminal_id'])
        action_id = f'ACT-{uuid.uuid4().hex.upper()}'
        now = _iso(time.time())
        details = {'proposal_id': proposal_id, 'payload_hash': row['payload_hash'],
                   'reason': payload['reason'], 'terminal_model': terminal.get('model'),
                   'command_dispatched': payload['action_type'], 'simulated': True}
        receipt = {'action_id': action_id, 'proposal_id': proposal_id, **payload,
                   'payload_hash': row['payload_hash'], 'status': 'SCHEDULED_ON_TERMINAL',
                   'created_at': now, 'details': details, 'idempotent_replay': False}
        conn.execute('''INSERT INTO terminal_actions
            (action_id,user_id,terminal_id,action_type,status,idempotency_key,details_json,created_at)
            VALUES (?,?,?,?,?,?,?,?)''', (action_id, user_id, payload['terminal_id'],
                payload['action_type'], receipt['status'], proposal_id, _json(details), now))
        conn.execute('UPDATE terminal_action_proposals SET status=?, receipt_json=? WHERE proposal_id=?',
                     ('CONFIRMED', _json(receipt), proposal_id))
        conn.commit()
        return receipt
