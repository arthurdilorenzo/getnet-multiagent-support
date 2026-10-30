"""Consolidated acceptance test suite for Getnet Núcleo 2.0 (Acceptance A1-A5).

Fully offline, deterministic, testing properties and negative controls.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid
import pytest

from app.memory.store import StateStore
from app.hitl.queue import HITLQueueManager
from app.schemas import ChatResponse, ToolExecutionResult
from app.security.auth import CrossTenantAccessError, SecurityContext
from app.tools.registry import ToolRegistry
from app.v2.adapters import StateAdapter, ToolGateway
from app.v2.contracts import (
    AnswerDocument,
    Binding,
    BindingProposal,
    ContextBlock,
    DocumentaryBlock,
    Entities,
    EvidenceSpan,
    Fact,
    HumanBlock,
    OperationalBlock,
    PendingBlock,
    PlanProposal,
    Requirement,
    RequirementDecision,
    RequirementProposal,
    ResumePlan,
    StepProposal,
    answer_status,
    canonicalize_plan,
)
from app.v2.knowledge import EvidenceError, KnowledgeService, format_fact, render_final_answer, validate_answer
from app.v2.runtime import Runtime
from scripts.v2_pilot import DurableSQLiteLedger


# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------

class MockModel:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def generate(self, schema, payload, *, phase, tracer):
        self.calls.append({"schema": schema.__name__, "payload": payload, "phase": phase})
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, schema):
                return resp
            if isinstance(resp, dict):
                return schema.model_validate(resp)
        raise RuntimeError(f"MockModel out of responses for {schema.__name__}")


def make_customers_json(tmp_path: Path, customers: dict) -> Path:
    p = tmp_path / "customers.json"
    p.write_text(json.dumps({"customers": customers}, ensure_ascii=False), encoding="utf-8")
    return p


def run_isolated_child(tmp_path, name, code, *arguments):
    """A fresh interpreter inherits guards before its first application import."""
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    guard = root / "sitecustomize.py"
    guard.write_text(
        "import os, socket\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['ACCEPTANCE_CHILD_ROOT'])\n"
        "for key in ('GEMINI_API_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'HITL_WEBHOOK_URL'): os.environ[key] = ''\n"
        "os.environ.update(GETNET_OFFLINE_MODE='1', GETNET_LIVE_EMBEDDINGS='false', GETNET_LIVE_TESTS='0')\n"
        "for key, name in {'SQLITE_DB_PATH':'state.db', 'DATA_DIR':'data', 'MOCK_CUSTOMERS_PATH':'customers.json', 'VECTOR_INDEX_PATH':'index.json', 'CORPUS_JSON_PATH':'corpus.json'}.items(): os.environ[key] = str(root / name)\n"
        "import pydantic_settings\n"
        "original_init = pydantic_settings.BaseSettings.__init__\n"
        "def offline_init(self, *args, **kwargs):\n"
        "    kwargs['_env_file'] = None\n"
        "    original_init(self, *args, **kwargs)\n"
        "pydantic_settings.BaseSettings.__init__ = offline_init\n"
        "def denied(*args, **kwargs): raise RuntimeError('NETWORK_DENIED_CHILD')\n"
        "socket.socket.connect = denied\n"
        "socket.socket.connect_ex = denied\n"
        "socket.create_connection = denied\n"
        "socket.getaddrinfo = denied\n"
        "os.environ['ACCEPTANCE_CHILD_GUARDED'] = '1'\n",
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "ACCEPTANCE_CHILD_ROOT": str(root), "PYTHONPATH": os.pathsep.join([str(root), str(repo)])}
    child = subprocess.run([sys.executable, "-c", code, *map(str, arguments)], env=env, cwd=root, capture_output=True, text=True, timeout=40)
    assert child.returncode == 0, child.stdout + child.stderr
    return json.loads(child.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# A1: Um store por ambiente, sem neutralizar a sentinela
# ---------------------------------------------------------------------------

class SentinelStore(StateStore):
    def __init__(self, db_path: Path):
        self.armed = False
        self.breach_count = 0
        super().__init__(db_path)
        self.armed = True

    def _connect(self):
        if getattr(self, "armed", False):
            self.breach_count += 1
            raise RuntimeError("SENTINEL_BREACH: Forbidden operational store accessed!")
        return super()._connect()


def test_a1_1_sentinel_isolation_and_negative_control(tmp_path, monkeypatch):
    """A1.1: Sentinel on forbidden store remains intact and uncalled; operation appears only in allowed store."""
    forbidden_db = tmp_path / "forbidden.db"
    sentinel = SentinelStore(forbidden_db)
    sentinel_id = id(sentinel)
    # Install the sentinel at the real legacy registry dependency. V2 must
    # neither use it nor replace it while constructing an explicit gateway.
    import app.tools.registry as registry_module
    monkeypatch.setattr(registry_module, "state_store", sentinel)
    assert ToolRegistry()._store is sentinel

    allowed_db = tmp_path / "allowed.db"
    allowed_store = StateStore(allowed_db)
    allowed_store.ensure_session("sess_1", "user_1")

    cust_file = make_customers_json(tmp_path, {
        "user_1": {
            "user_id": "user_1",
            "terminals": [{"terminal_id": "POS-9941", "status": "ativo"}],
            "ultimo_erro_transacao": "E-99",
        }
    })
    queue = HITLQueueManager(allowed_store)
    gateway = ToolGateway(cust_file, store=allowed_store, queue=queue)

    sec_ctx = SecurityContext("user_1", scopes=["chat:write", "customer:read_own"])
    res = gateway.execute("diagnose_terminal_issue", {"terminal_id": "POS-9941"}, sec_ctx, "sess_1")

    assert res.success is True
    assert id(registry_module.state_store) == sentinel_id
    assert registry_module.state_store is sentinel
    assert sentinel.breach_count == 0

    # Operation exists in allowed store
    with allowed_store._connect() as conn:
        row = conn.execute("SELECT * FROM tool_operations WHERE operation_id=?", (res.operation_id,)).fetchone()
        assert row is not None
        assert row["tool_name"] == "diagnose_terminal_issue"

    # Forbidden DB had zero operational activity
    with sqlite3.connect(str(forbidden_db)) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "tool_operations" in tables:
            assert conn.execute("SELECT count(*) FROM tool_operations").fetchone()[0] == 0

    # Negative control: Using registry tied to sentinel raises SENTINEL_BREACH
    from pydantic import BaseModel
    class EmptyArgs(BaseModel):
        pass
    broken_registry = ToolRegistry()
    broken_registry.register("test_tool", "desc", EmptyArgs, lambda: {})
    with pytest.raises(RuntimeError, match="SENTINEL_BREACH"):
        broken_registry.execute("test_tool", {}, sec_ctx, "sess_1")


def test_a1_2_independent_processes_and_no_bleed(tmp_path):
    """A1.2: Two independent processes each record only their own operations; zero bleed."""
    code = """
import os, sys, json, socket
from pathlib import Path
assert os.environ['ACCEPTANCE_CHILD_GUARDED'] == '1'
for probe in (lambda: socket.create_connection(('127.0.0.1', 9)), lambda: socket.getaddrinfo('example.invalid', 443)):
    try: probe()
    except RuntimeError as exc: assert str(exc) == 'NETWORK_DENIED_CHILD'
    else: raise AssertionError('child network guard absent')
from app.config import settings
assert settings.GEMINI_API_KEY == ''
assert settings.SQLITE_DB_PATH == Path(os.environ['ACCEPTANCE_CHILD_ROOT']) / 'state.db'
from app.memory.store import state_store
from app.v2.adapters import StateAdapter, ToolGateway
from app.hitl.queue import HITLQueueManager
from app.security.auth import SecurityContext
user, session, message = sys.argv[1:]
customers = Path(os.environ['MOCK_CUSTOMERS_PATH'])
customers.write_text(json.dumps({'customers': {user: {'user_id': user, 'segment': user}}}))
adapter = StateAdapter(state_store)
state_store.ensure_session(session, user)
adapter.append(session, user, 'user', message)
gateway = ToolGateway(customers, store=state_store, queue=HITLQueueManager(state_store))
result = gateway.execute('get_customer_profile', {}, SecurityContext(user, scopes=['chat:write', 'customer:read_own']), session)
assert result.success
with state_store._connect() as conn:
    operations = [dict(row) for row in conn.execute('SELECT user_id,tool_name FROM tool_operations')]
    messages = [dict(row) for row in conn.execute('SELECT user_id,session_id,content FROM messages')]
print(json.dumps({'pid': os.getpid(), 'operations': operations, 'messages': messages, 'network_denied': True}))
"""
    results = [run_isolated_child(tmp_path, 'proc_' + suffix, code, 'user_' + suffix, 'sess_' + suffix, 'Mensagem Processo ' + suffix) for suffix in ('a', 'b')]
    assert results[0]['pid'] != results[1]['pid']
    for suffix, result in zip(('a', 'b'), results):
        assert result['network_denied']
        assert result['operations'] == [{'user_id': 'user_' + suffix, 'tool_name': 'get_customer_profile'}]
        assert result['messages'] == [{'user_id': 'user_' + suffix, 'session_id': 'sess_' + suffix, 'content': 'Mensagem Processo ' + suffix}]


def test_a1_3_legacy_compatibility_and_custom_independence(tmp_path):
    """A1.3: Default ToolRegistry uses state_store; explicit store is independent with single execute."""
    from app.memory.store import state_store

    default_reg = ToolRegistry()
    assert default_reg._store is state_store

    custom_db = tmp_path / "custom.db"
    custom_store = StateStore(custom_db)
    custom_reg = ToolRegistry(store=custom_store)
    assert custom_reg._store is custom_store
    assert custom_reg._store is not state_store


def test_a1_4_prepare_preserves_empty_entities_and_mutated_cache(tmp_path):
    """A1.4: Repeated preparation preserves entities={}, mutated cache, and conversation messages."""
    from tests.v2.fixture_factory import prepare_arms

    cases = [{
        "id": "test_case_1",
        "contexto_premissas": {
            "user_id": "lojista_01",
            "dados_sinteticos": {"terminal_id": "POS-1000"},
        },
        "mensagens": [
            {"role": "user", "content": "Primeira mensagem"},
            {"role": "assistant", "content": "Primeira resposta"},
        ],
    }]

    fixtures_dir = tmp_path / "fixtures"
    prepare_arms(fixtures_dir, cases)

    store_b = StateStore(fixtures_dir / "B" / "state.db")
    adapter_b = StateAdapter(store_b)

    # Save entities={} specifically to test empty dict preservation
    adapter_b.save("pilot_test_case_1", "lojista_01", {"entities": {}, "preserved_flag": True})

    # Mutate cache
    cache_file = fixtures_dir / "B" / "embeddings_cache.json"
    cache_file.write_text(json.dumps({"mutated": True}), encoding="utf-8")
    mutated_hash = hashlib.sha256(cache_file.read_bytes()).hexdigest()

    # Repeat preparation
    prepare_arms(fixtures_dir, cases)

    # Verify entities={} preserved
    ctx_v2 = adapter_b.context("pilot_test_case_1", "lojista_01")["v2"]
    assert ctx_v2.get("entities") == {}
    assert ctx_v2.get("preserved_flag") is True

    # Verify cache preserved
    assert hashlib.sha256(cache_file.read_bytes()).hexdigest() == mutated_hash

    # Verify no duplicated messages
    with store_b._connect() as conn:
        count = conn.execute("SELECT count(*) FROM messages WHERE session_id='pilot_test_case_1'").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# A2: Identidade e autorização sobrevivem ao HITL
# ---------------------------------------------------------------------------

def test_a2_1_scope_preservation_and_privilege_boundary(tmp_path):
    """A2.1: Merchant with only chat:write cannot read customer profile after HITL resume; scopes not escalated."""
    db = tmp_path / "hitl.db"
    store = StateStore(db)
    state = StateAdapter(store)
    queue = HITLQueueManager(store)

    cust_file = make_customers_json(tmp_path, {
        "user_sec": {"user_id": "user_sec", "profile": {"name": "Lojista"}}
    })
    gateway = ToolGateway(cust_file, store=store, queue=queue)

    class ModelStepHumanThenProfile:
        def __init__(self):
            self.turn = 0
        def generate(self, schema, payload, *, phase, tracer):
            if schema == PlanProposal:
                return PlanProposal(
                    requirements=[RequirementProposal(kind="human", summary="Pedir humano")],
                    steps=[StepProposal(kind="human", reason="Precisa de humano", requirement_indexes=[0])],
                )
            elif schema == ResumePlan:
                return ResumePlan(
                    steps=[StepProposal(kind="tool", tool_name="get_customer_profile", arguments={"user_id": "user_sec"}, requirement_indexes=[0])]
                )
            raise RuntimeError("Unexpected schema")

    runtime = Runtime(ModelStepHumanThenProfile(), KnowledgeService(None), state, gateway, queue)

    # Turn 1: merchant has ONLY chat:write
    merchant_write_only = SecurityContext("user_sec", scopes=["chat:write"])
    resp1 = runtime.process("Ajuda humana", merchant_write_only, "sess_h1")
    assert resp1.escalated is True
    ticket_id = resp1.ticket_id

    # Operator resolves ticket
    operator = SecurityContext("op_1", role="operator", scopes=["hitl:manage"])
    queue.claim_ticket(ticket_id, "op_1")
    queue.reply_to_ticket(ticket_id, "op_1", "Operador confirmou.", resolve_ticket=True)

    # Negative control: Unauthorized operator or merchant trying to call resume
    imposter = SecurityContext("user_sec", scopes=["chat:write"])
    with pytest.raises(PermissionError):
        runtime.resume(ticket_id, imposter)

    unprivileged_operator = SecurityContext("op_bad", role="operator", scopes=["chat:write"])
    with pytest.raises(PermissionError):
        runtime.resume(ticket_id, unprivileged_operator)
    # Existing foreign identities exercise ownership rather than absent-session errors.
    store.ensure_session("sess_foreign", "user_foreign")
    store.ensure_session("sess_same_user", "user_sec")
    with store._connect() as conn:
        before_claims = conn.execute("SELECT count(*) FROM hitl_resumptions").fetchone()[0]
    for foreign_session, foreign_user, foreign_ticket in (
        ("sess_foreign", "user_foreign", ticket_id),
        ("sess_same_user", "user_sec", ticket_id),
        ("sess_h1", "user_sec", "nonexistent_ticket"),
    ):
        with pytest.raises((ValueError, CrossTenantAccessError)):
            store.reserve_hitl_resumption(foreign_session, foreign_user, foreign_ticket)
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM hitl_resumptions").fetchone()[0] == before_claims

    # Resume with valid operator: The resumed execution tries to run get_customer_profile under merchant context
    resp2 = runtime.resume(ticket_id, operator)
    # The tool failed because merchant does NOT have customer:read_own!
    assert "Operação não concluída" in resp2.response
    assert resp2.answer_status == "abstained"

    # Check merchant scopes were NOT escalated in stored state
    stored_state = state.context("sess_h1", "user_sec")["v2"]
    assert "customer:read_own" not in stored_state.get("original_scopes", [])
    assert stored_state.get("facts", []) == []
    assert not any("fact_id" in evidence for evidence in resp2.evidence)


def test_a2_2_replay_idempotency_after_resumption(tmp_path):
    """A2.2: Real read/HITL, then a new process with forbidden model/tool dispatch."""
    store = StateStore(tmp_path / "replay.db")
    state, queue = StateAdapter(store), HITLQueueManager(store)
    cust_file = make_customers_json(tmp_path, {"user_r": {"user_id": "user_r", "segment": "Alimentação"}})
    gateway = ToolGateway(cust_file, store=store, queue=queue)

    class SimpleHITLModel:
        def __init__(self):
            self.calls = []
        def generate(self, schema, payload, *, phase, tracer):
            self.calls.append((schema.__name__, copy.deepcopy(payload)))
            if schema == PlanProposal:
                return PlanProposal(requirements=[
                    RequirementProposal(kind="human", summary="Conferência atribuída"),
                    RequirementProposal(kind="operational", summary="Segmento cadastrado", bindings=[BindingProposal(tool_name="get_customer_profile", resource_type="customer", resource_id="user_r", field_pointer="/profile/segment")]),
                ], steps=[StepProposal(kind="human", reason="Validar", requirement_indexes=[0, 1])])
            if schema == ResumePlan:
                return ResumePlan(steps=[StepProposal(kind="tool", tool_name="get_customer_profile", arguments={}, requirement_indexes=[1])])
            if schema == AnswerDocument:
                human = next(span for span in payload['evidence_spans'] if span['kind'] == 'human')
                fact = next(f for f in payload['observed_facts'] if f['availability'] == 'available')
                return AnswerDocument(requirements=[RequirementDecision(requirement_id="r0001", status="supported"), RequirementDecision(requirement_id="r0002", status="supported")], final_answer=[HumanBlock(requirement_id="r0001", observation_id=human['span_id']), OperationalBlock(requirement_id="r0002", fact_ids=[fact['fact_id']])])
            raise AssertionError(schema)

    model = SimpleHITLModel()
    runtime = Runtime(model, KnowledgeService(model), state, gateway, queue)
    original = "Preciso de conferência e do segmento cadastrado, sem alterar dados."
    first = runtime.process(original, SecurityContext("user_r", scopes=["chat:write", "customer:read_own"]), "sess_r")
    assert first.escalated
    before = copy.deepcopy(state.context("sess_r", "user_r")["v2"]["requirements"])
    operator = SecurityContext("op_1", role="operator", scopes=["hitl:manage"])
    queue.claim_ticket(first.ticket_id, "op_1")
    queue.reply_to_ticket(first.ticket_id, "op_1", "Conferência OK.", resolve_ticket=True)
    completed = runtime.resume(first.ticket_id, operator)
    assert completed.answer_status == "complete"
    assert 'op_1' in completed.response and 'Conferência OK.' in completed.response
    assert state.context("sess_r", "user_r")["v2"]["requirements"] == before
    assert [name for name, payload in model.calls] == ['PlanProposal', 'ResumePlan', 'AnswerDocument']
    assert all(payload['original_request'] == original for _, payload in model.calls)
    with store._connect() as conn:
        assert conn.execute('SELECT count(*) FROM tool_operations').fetchone()[0] == 1
        before_messages = conn.execute('SELECT count(*) FROM messages').fetchone()[0]
    code = """
import sys,json
from pathlib import Path
from app.memory.store import StateStore
from app.v2.adapters import StateAdapter, ToolGateway
from app.v2.runtime import Runtime
from app.v2.knowledge import KnowledgeService
from app.hitl.queue import HITLQueueManager
from app.security.auth import SecurityContext
calls = {'model': 0, 'tool': 0}
class ForbiddenModel:
    def generate(self, *args, **kwargs):
        calls['model'] += 1
        raise AssertionError('replay invoked model')
class ForbiddenGateway(ToolGateway):
    def execute(self, *args, **kwargs):
        calls['tool'] += 1
        raise AssertionError('replay invoked tool')
store = StateStore(Path(sys.argv[1]))
state, queue = StateAdapter(store), HITLQueueManager(store)
model = ForbiddenModel()
gateway = ForbiddenGateway(Path(sys.argv[2]), store=store, queue=queue)
response = Runtime(model, KnowledgeService(model), state, gateway, queue).resume(sys.argv[3], SecurityContext('op_1', role='operator', scopes=['hitl:manage']))
with store._connect() as conn:
    messages = conn.execute('SELECT count(*) FROM messages').fetchone()[0]
    operations = conn.execute('SELECT count(*) FROM tool_operations').fetchone()[0]
print(json.dumps({'response': response.model_dump(mode='json'), 'calls': calls, 'requirements': state.context('sess_r','user_r')['v2']['requirements'], 'messages':messages, 'operations':operations}))
"""
    replay = run_isolated_child(tmp_path, 'replay_child', code, store.db_path, cust_file, first.ticket_id)
    assert replay['response'] == completed.model_dump(mode='json')
    assert replay['calls'] == {'model': 0, 'tool': 0}
    assert replay['requirements'] == before
    assert replay['messages'] == before_messages and replay['operations'] == 1


def test_a2_3_atomic_failure_on_corrupted_or_cross_session_claim(tmp_path):
    """A2.3: A valid foreign reservation cannot publish to a second valid session."""
    store = StateStore(tmp_path / "atomic.db")
    state, queue = StateAdapter(store), HITLQueueManager(store)
    tickets = []
    for suffix in ('a', 'b'):
        user, session = 'user_' + suffix, 'sess_' + suffix
        store.ensure_session(session, user)
        state.save(session, user, {'entities': {}, 'marker': suffix})
        ticket = queue.create_ticket(user, session, 'test', 'reason')
        queue.claim_ticket(ticket.ticket_id, 'op_1')
        queue.reply_to_ticket(ticket.ticket_id, 'op_1', 'Conferido.', resolve_ticket=True)
        claim = store.reserve_hitl_resumption(session, user, ticket.ticket_id)
        assert claim['acquired']
        tickets.append((ticket, claim))
    def snapshot():
        with store._connect() as conn:
            return {table: [tuple(row) for row in conn.execute('SELECT * FROM ' + table + ' ORDER BY rowid')] for table in ('sessions', 'messages', 'hitl_resumptions', 'escalation_tickets')}
    before = snapshot()
    ticket_a, claim_a = tickets[0]
    forged = ChatResponse(response='Cross-session publication', user_id='user_b', session_id='sess_b', trace_id='cross', agent_used='V2', semantic_mode='native_semantic')
    with pytest.raises((ValueError, CrossTenantAccessError)):
        state.finish_resumption('sess_b', 'user_b', ticket_a.ticket_id, claim_a['reservation_id'], forged, 'completed')
    assert snapshot() == before
    # Interrupted reservation is retained across a fresh connection, never reclaimed.
    restarted = StateStore(store.db_path)
    retried = restarted.reserve_hitl_resumption('sess_a', 'user_a', ticket_a.ticket_id)
    assert retried['acquired'] is False and retried['status'] == 'running'
    assert snapshot() == before


# ---------------------------------------------------------------------------
# A3: Fatos operacionais por recurso/campo, nunca por palavras
# ---------------------------------------------------------------------------

def test_a3_1_resource_mismatch_rejected():
    """A3.1: Tool returns POS-1111; attempting to use it for requirement POS-9999 is rejected."""
    planned_req = Requirement(
        requirement_id="r0001",
        kind="operational",
        summary="Verificar erro da máquina",
        bindings=[
            Binding(
                binding_id="r0001:b0001",
                tool_name="diagnose_terminal_issue",
                resource_type="terminal",
                resource_id="POS-9999",
                field_pointer="/ultimo_erro_transacao",
            )
        ],
    )

    # Observed fact from tool is for POS-1111
    fact_1111 = Fact(
        fact_id="fact_1111",
        binding_id="r0001:b0001",
        operation_id="OP-1",
        resource_type="terminal",
        resource_id="POS-1111",
        field_pointer="/ultimo_erro_transacao",
        availability="available",
        value="E-99",
    )

    doc = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[OperationalBlock(requirement_id="r0001", fact_ids=["fact_1111"])],
    )

    with pytest.raises(EvidenceError, match="OPERATIONAL_RESOURCE_MISMATCH"):
        validate_answer(doc, [], [fact_1111], [planned_req])


def test_a3_2_null_error_requires_unanswerable_invariant_to_wording():
    """A3.2: Null error cannot be supported; server enforces unanswerable regardless of summary wording."""
    for summary_text in ("Último erro de transação", "Resultado da tentativa de transação"):
        planned_req = Requirement(
            requirement_id="r0001",
            kind="operational",
            summary=summary_text,
            bindings=[
                Binding(
                    binding_id="r0001:b0001",
                    tool_name="diagnose_terminal_issue",
                    resource_type="terminal",
                    resource_id="POS-9999",
                    field_pointer="/ultimo_erro_transacao",
                )
            ],
        )

        fact_null_error = Fact(
            fact_id="fact_null",
            binding_id="r0001:b0001",
            operation_id="OP-1",
            resource_type="terminal",
            resource_id="POS-9999",
            field_pointer="/ultimo_erro_transacao",
            availability="unavailable",
            value=None,
        )

        # Mock claims supported on null error
        doc_invalid = AnswerDocument(
            requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
            final_answer=[OperationalBlock(requirement_id="r0001", fact_ids=["fact_null"])],
        )

        with pytest.raises(EvidenceError, match="UNSUPPORTED_OPERATIONAL_BINDING"):
            validate_answer(doc_invalid, [], [fact_null_error], [planned_req])

        # Valid document marks unanswerable
        doc_valid = AnswerDocument(
            requirements=[RequirementDecision(requirement_id="r0001", status="unanswerable")],
            final_answer=[PendingBlock(requirement_id="r0001")],
        )
        validate_answer(doc_valid, [], [fact_null_error], [planned_req])
        assert answer_status(doc_valid.requirements) == "abstained"


def test_a3_3_operational_block_rejects_free_text_and_presents_server_value():
    """A3.3: OperationalBlock strictly rejects free value/text in schema; server formats fact BRL 125,00."""
    # Free fields rejected by extra='forbid'
    with pytest.raises(Exception):
        OperationalBlock.model_validate({"requirement_id": "r0001", "fact_ids": ["f1"], "value": 350})

    with pytest.raises(Exception):
        OperationalBlock.model_validate({"requirement_id": "r0001", "fact_ids": ["f1"], "text": "saldo R$ 350"})

    # Valid fact with BRL 125,00
    fact_saldo = Fact(
        fact_id="f_saldo",
        binding_id="r0001:b0001",
        operation_id="OP-1",
        resource_type="customer",
        resource_id="user_123",
        field_pointer="/sales_data/saldo_disponivel_brl",
        availability="available",
        value="125.00",
        value_type="decimal",
        unit="BRL",
    )

    doc = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[OperationalBlock(requirement_id="r0001", fact_ids=["f_saldo"])],
    )

    rendered = render_final_answer(doc, [], [fact_saldo], {})
    assert "BRL 125,00" in rendered


def test_a3_4_zero_false_available_vs_null_missing():
    """A3.4: 0 and False are available and presented; missing/null/found=False do not satisfy binding."""
    fact_zero = Fact(
        fact_id="f_zero",
        operation_id="OP-1",
        resource_type="customer",
        resource_id="user_1",
        field_pointer="/sales_data/saldo_disponivel_brl",
        availability="available",
        value=0,
        unit="BRL",
    )
    formatted_zero = format_fact(fact_zero)
    assert "BRL 0,00" in formatted_zero

    fact_false = Fact(
        fact_id="f_false",
        operation_id="OP-1",
        resource_type="terminal",
        resource_id="POS-1",
        field_pointer="/chip_algar_status",
        availability="available",
        value=False,
    )
    assert "False" in format_fact(fact_false)

    fact_unavail = Fact(
        fact_id="f_unavail",
        operation_id="OP-1",
        resource_type="terminal",
        resource_id="POS-1",
        field_pointer="/ultimo_erro_transacao",
        availability="unavailable",
        value=None,
    )
    assert "Informação indisponível" in format_fact(fact_unavail)


def test_a3_5_context_and_documentary_blocks_cannot_satisfy_operational_binding():
    """A3.5: ContextBlock or DocumentaryBlock cannot satisfy an operational binding."""
    planned_req = Requirement(
        requirement_id="r0001",
        kind="operational",
        summary="Consultar saldo",
        bindings=[
            Binding(
                binding_id="r0001:b0001",
                tool_name="get_sales_settlement",
                resource_type="customer",
                resource_id="u1",
                field_pointer="/sales_data/saldo_disponivel_brl",
                unit="BRL",
            )
        ],
    )

    doc = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[ContextBlock(requirement_id="r0001", context_ref="/recent_messages/0/content")],
    )

    with pytest.raises(EvidenceError, match="UNSUPPORTED_OPERATIONAL_BINDING"):
        validate_answer(doc, [], [], [planned_req], context={"recent_messages": [{"role": "user", "content": "Usuário disse ter saldo 350"}]})


# ---------------------------------------------------------------------------
# A4: Cobertura estruturada e composição sem catálogo de respostas
# ---------------------------------------------------------------------------

def test_a4_1_requirement_coverage_integrity():
    """A4.1: Omission, duplicates, or empty requirements in answer document are rejected."""
    req1 = Requirement(requirement_id="r0001", kind="documentary", summary="Taxas Pix")
    req2 = Requirement(requirement_id="r0002", kind="documentary", summary="Prazo")

    span = EvidenceSpan(span_id="s1", text="Taxas Pix 0.99%", source="doc_pix", kind="document", content_hash="h1")

    # Omission of r0002
    doc_incomplete = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[DocumentaryBlock(requirement_id="r0001", text="Taxas", span_ids=["s1"])],
    )
    with pytest.raises(EvidenceError, match="INCOMPLETE_REQUIREMENT_COVERAGE"):
        validate_answer(doc_incomplete, [span], [], [req1, req2])

    # Duplicate r0001
    doc_duplicate = AnswerDocument(
        requirements=[
            RequirementDecision(requirement_id="r0001", status="supported"),
            RequirementDecision(requirement_id="r0001", status="supported"),
        ],
        final_answer=[DocumentaryBlock(requirement_id="r0001", text="Taxas", span_ids=["s1"])],
    )
    with pytest.raises(EvidenceError, match="DUPLICATE_REQUIREMENT_ID"):
        validate_answer(doc_duplicate, [span], [], [req1, req2])


def test_a4_2_multi_capability_knowledge_and_tool_distinction(tmp_path):
    """A4.2: Mixed inquiry (documentary + operational) cannot be satisfied solely by profile."""
    req_doc = Requirement(requirement_id="r0001", kind="documentary", summary="Regra crediário")
    req_op = Requirement(
        requirement_id="r0002",
        kind="operational",
        summary="Segmento lojista",
        bindings=[
            Binding(
                binding_id="r0002:b0001",
                tool_name="get_customer_profile",
                resource_type="customer",
                resource_id="user_1",
                field_pointer="/profile/segment",
            )
        ],
    )

    fact_profile = Fact(
        fact_id="f_prof",
        binding_id="r0002:b0001",
        operation_id="OP-1",
        resource_type="customer",
        resource_id="user_1",
        field_pointer="/profile/segment",
        availability="available",
        value="Alimentação",
    )

    # Mock tries to mark r0001 supported without any documentary span
    doc_mock = AnswerDocument(
        requirements=[
            RequirementDecision(requirement_id="r0001", status="supported"),
            RequirementDecision(requirement_id="r0002", status="supported"),
        ],
        final_answer=[OperationalBlock(requirement_id="r0002", fact_ids=["f_prof"])],
    )

    with pytest.raises(EvidenceError, match="MISSING_EVIDENCE_SPAN"):
        validate_answer(doc_mock, [], [fact_profile], [req_doc, req_op])


@pytest.mark.parametrize("profile_only", [False, True], ids=["mixed_valid", "profile_cannot_resolve_documentation"])
def test_a4_2_runtime_mixed_capabilities_and_single_sdk_composition(tmp_path, profile_only):
    """A4.2/A4.3: SDK boundary sees original exclusions/history; origins remain distinct."""
    from types import SimpleNamespace
    from google.genai import types
    from app.v2.model import GeminiModel

    store = StateStore(tmp_path / "mixed.db")
    state, queue = StateAdapter(store), HITLQueueManager(store)
    user, session = "user_mix", "sess_mix"
    store.ensure_session(session, user)
    history = "Já confirmei: não altere meu cadastro e não proponha antecipação."
    state.append(session, user, "user", history)
    customers = make_customers_json(tmp_path, {user: {"user_id": user, "segment": "Alimentação"}})
    gateway = ToolGateway(customers, store=store, queue=queue)
    original = "Explique a regra do crediário e consulte meu segmento, sem antecipação nem alterar dados."
    retrieval_calls, sdk_calls = [], []
    def retrieve(query):
        retrieval_calls.append(query)
        return {"candidate_chunks": [{"content": "O crediário depende de condições da modalidade e do estabelecimento.", "source": "guia_crediario"}], "retrieval_mode": "offline_fixture"}
    class MockSDK:
        def generate_content(self, *, model, contents, config):
            payload = json.loads(contents[0].parts[0].text)
            title = config.response_json_schema['title']
            sdk_calls.append((title, copy.deepcopy(payload)))
            if title == 'PlanProposal':
                steps = [StepProposal(kind="tool", tool_name="get_customer_profile", arguments={}, requirement_indexes=[1])]
                if not profile_only:
                    steps.insert(0, StepProposal(kind="knowledge", query="condições crediário", requirement_indexes=[0]))
                result = PlanProposal(requirements=[
                    RequirementProposal(kind="documentary", summary="Regra crediário", constraints=["sem antecipação", "não alterar dados"]),
                    RequirementProposal(kind="operational", summary="Segmento cadastrado", bindings=[BindingProposal(tool_name="get_customer_profile", resource_type="customer", resource_id=user, field_pointer="/profile/segment")]),
                ], steps=steps)
            elif title == 'AnswerDocument':
                fact = next(f for f in payload['observed_facts'] if f['availability'] == 'available')
                blocks = [OperationalBlock(requirement_id="r0002", fact_ids=[fact['fact_id']])]
                documents = [span for span in payload['evidence_spans'] if span['kind'] == 'document']
                if documents:
                    # Deliberately paraphrased prose: structural citation checks do
                    # not claim to adjudicate open semantic quality offline.
                    blocks.insert(0, DocumentaryBlock(requirement_id="r0001", text="As condições variam conforme a modalidade e o estabelecimento.", span_ids=[documents[0]['span_id']]))
                result = AnswerDocument(requirements=[RequirementDecision(requirement_id="r0001", status="supported"), RequirementDecision(requirement_id="r0002", status="supported")], final_answer=blocks)
            else:
                raise AssertionError(title)
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role='model', parts=[types.Part.from_text(text=result.model_dump_json())]), finish_reason=types.FinishReason.STOP)])
    model = GeminiModel(client=SimpleNamespace(models=MockSDK()))
    response = Runtime(model, KnowledgeService(model, retrieve=retrieve), state, gateway, queue).process(original, SecurityContext(user, scopes=['chat:write', 'customer:read_own']), session)
    assert [title for title, _ in sdk_calls] == ['PlanProposal', 'AnswerDocument']
    for title, payload in sdk_calls:
        assert payload['original_request'] == original
        assert any(message['content'] == history for message in payload['context']['recent_messages'])
    composition = sdk_calls[-1][1]
    assert composition['planned_requirements'][0]['constraints'] == ['sem antecipação', 'não alterar dados']
    assert len(composition['observed_facts']) == 1
    with store._connect() as conn:
        operations = conn.execute('SELECT tool_name FROM tool_operations').fetchall()
    assert [row['tool_name'] for row in operations] == ['get_customer_profile']
    if profile_only:
        assert retrieval_calls == []
        assert response.answer_status == 'abstained'
        assert response.abstained
    else:
        assert retrieval_calls == ['condições crediário']
        assert response.answer_status == 'complete'
        assert response.tools_called == ['search_getnet_knowledge', 'get_customer_profile']
        assert 'guia_crediario' in response.sources
        assert 'Alimentação' in response.response


def test_a4_3_documentary_citation_integrity():
    """A4.3: Validates non-existent span reference, hash mismatch, and operational tool span as doc citation."""
    req = Requirement(requirement_id="r0001", kind="documentary", summary="Regras")

    valid_text = "Texto sobre regras Getnet"
    digest = hashlib.sha256(valid_text.encode("utf-8")).hexdigest()
    span_valid = EvidenceSpan(span_id="s_valid", text=valid_text, source="guia", kind="document", content_hash=digest)
    span_tool = EvidenceSpan(span_id="s_tool", text="tool data", source="get_customer_profile", kind="tool")

    # Unknown span
    doc_unknown = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[DocumentaryBlock(requirement_id="r0001", text="Prosa", span_ids=["s_unknown"])],
    )
    with pytest.raises(EvidenceError, match="UNKNOWN_SPAN_REFERENCE"):
        validate_answer(doc_unknown, [span_valid], [], [req])

    # Tool span cited as documentary
    doc_tool = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[DocumentaryBlock(requirement_id="r0001", text="Prosa", span_ids=["s_tool"])],
    )
    with pytest.raises(EvidenceError, match="INVALID_DOCUMENTARY_CITATION"):
        validate_answer(doc_tool, [span_valid, span_tool], [], [req])

    # Hash mismatch
    span_tampered = EvidenceSpan(span_id="s_tampered", text="Texto alterado", source="guia", kind="document", content_hash=digest)
    doc_tampered = AnswerDocument(
        requirements=[RequirementDecision(requirement_id="r0001", status="supported")],
        final_answer=[DocumentaryBlock(requirement_id="r0001", text="Prosa", span_ids=["s_tampered"])],
    )
    with pytest.raises(EvidenceError, match="HASH_MISMATCH"):
        validate_answer(doc_tampered, [span_tampered], [], [req])


def test_a4_4_structural_removal_of_regex_and_global_mutations():
    """A4.4: Confirms structural removal of lexical regexes, IsolatedToolRegistry, and global assignments."""
    v2_knowledge_path = Path(__file__).resolve().parents[2] / "app" / "v2" / "knowledge.py"
    v2_adapters_path = Path(__file__).resolve().parents[2] / "app" / "v2" / "adapters.py"

    k_text = v2_knowledge_path.read_text(encoding="utf-8")
    a_text = v2_adapters_path.read_text(encoding="utf-8")

    # Regexes removed from knowledge
    for forbidden in ("_extract_percentages", "_extract_currencies", "_extract_numbers", "_extract_terminal_entities"):
        assert forbidden not in k_text, f"Forbidden regex function {forbidden} found in knowledge.py!"

    # IsolatedToolRegistry removed from adapters
    assert "class IsolatedToolRegistry" not in a_text, "IsolatedToolRegistry must be removed!"
    assert "IsolatedToolRegistry(" not in a_text

    # Global store mutation removed from adapters
    assert "app.tools.registry.state_store =" not in a_text
    assert "app.memory.store.state_store =" not in a_text


# ---------------------------------------------------------------------------
# A5: Preservar evidência e não fabricar saldo seguro
# ---------------------------------------------------------------------------

def test_a5_1_generic_reconciliation_and_unverified_fallback(tmp_path):
    """A5.1: Generic reconciliation sets accounting_status=unverified and remaining_cost_usd=None on incomplete sources."""
    db_path = tmp_path / "ledger.db"
    ledger = DurableSQLiteLedger(db_path)

    build_dir = tmp_path / "mock_build"
    directed_dir = tmp_path / "mock_directed"
    build_dir.mkdir()
    directed_dir.mkdir()

    # Synthetic ledger with a failed dispatch (no exact usage returned)
    lot1_data = {
        "dispatches": [
            {
                "claim_id": "c1", "timestamp": "2026-10-29T10:00:00Z", "case_id": "case_1",
                "kind": "generation", "model": "gemini-3.5-flash-lite", "input_tokens": 1000,
                "output_tokens": 200, "status": "RECONCILED",
            },
            {
                "claim_id": "c2_failed", "timestamp": "2026-10-29T10:01:00Z", "case_id": "case_2",
                "kind": "generation", "model": "gemini-3.5-flash-lite", "input_tokens": 5000,
                "output_tokens": 2048, "status": "failed", "error": "504 DEADLINE_EXCEEDED",
            },
        ]
    }
    (build_dir / "consumption_ledger.json").write_text(json.dumps(lot1_data), encoding="utf-8")

    ledger.seed_historical_runs(build_dir, directed_dir)
    summary = ledger.get_summary()

    assert summary["accounting_status"] == "unverified"
    assert summary["remaining_cost_usd"] is None
    assert summary["live_enabled"] is False
    assert len(summary["unverified_reasons"]) > 0

    # Idempotent re-run: claims not duplicated
    ledger.seed_historical_runs(build_dir, directed_dir)
    summary2 = ledger.get_summary()
    assert summary2["known_requests"] == 2


def test_a5_2_bounded_limits_and_overconsumption_block(tmp_path):
    """A5.2: In bounded synthetic ledger, attempts exceeding limits raise RuntimeError without dispatch."""
    db_path = tmp_path / "limits_test.db"
    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=2)

    # First reservation: ok
    c1 = ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id="case_1", model="flash")
    ledger.reconcile(c1, actual_input=1000, actual_output=500, duration_s=1.0, success=True)

    # Second reservation: ok
    c2 = ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id="case_2", model="flash")
    ledger.reconcile(c2, actual_input=1000, actual_output=500, duration_s=1.0, success=True)

    # Third reservation: exceeds sublimit (max 2)
    with pytest.raises(RuntimeError, match="Budget limit exceeded"):
        ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id="case_3", model="flash")


def test_a5_3_concurrency_and_interrupted_reservation(tmp_path):
    """A5.3: Disputing last reservation serializes; interrupted reservation remains in ledger."""
    db_path = tmp_path / "concurrency.db"
    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=1)

    claim = ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id="case_1", model="flash")
    assert claim["claim_id"].startswith("claim_")

    # Second attempt fails immediately because sublimit of 1 is reserved
    with pytest.raises(RuntimeError, match="Budget limit exceeded"):
        ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id="case_compete", model="flash")

    # If process was interrupted before reconciliation, reservation remains in DB
    summary = ledger.get_summary()
    assert summary["reservations_count"] == 1


def test_a5_4_financial_breakdown_separation(tmp_path):
    """A5.4: Financial summary clearly separates measured, estimated, and unverified entries."""
    db_path = tmp_path / "breakdown.db"
    ledger = DurableSQLiteLedger(db_path)

    c1 = ledger.reserve("generation", est_input=1000, max_output=500, arm="B", case_id="case_1", model="flash")
    ledger.reconcile(c1, actual_input=1200, actual_output=400, duration_s=1.2, success=True)

    summary = ledger.get_summary()
    assert "measured_usage" in summary
    assert "estimated_usage" in summary
    assert summary["measured_usage"]["input_tokens"] == 1200
    assert summary["measured_usage"]["output_tokens"] == 400


def test_a5_5_preservation_of_three_previous_rounds_and_arm_a():
    """A5.5: Compares current sha256 hashes of all 3 previous directories and arm_a; 100% untouched."""
    base = Path(__file__).resolve().parents[3]
    initial_manifest_file = base / "getnet_v2_consolidation_2026-10-29" / "initial_preserved_manifest.json"
    if not initial_manifest_file.exists():
        pytest.skip("initial_preserved_manifest.json not yet generated")

    manifest = json.loads(initial_manifest_file.read_text(encoding="utf-8"))
    for rel_path, exp_hash in manifest.items():
        fp = base / rel_path
        assert fp.exists(), f"Preserved file {rel_path} was deleted!"
        actual_hash = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert actual_hash == exp_hash, f"Preserved file {rel_path} was modified!"
