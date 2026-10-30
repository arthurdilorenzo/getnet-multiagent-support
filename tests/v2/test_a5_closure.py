"""A5 synthetic counterproofs: no SDK/network calls or historical writes."""
import json
import multiprocessing
import sqlite3
from pathlib import Path
import pytest
from scripts.v2_pilot import DurableSQLiteLedger

def reserve(l, kind="generation", inp=10, out=10):
    return l.reserve(kind, inp, out, "B", "synthetic", "mock")

def sources(tmp_path, rows):
    a=tmp_path/"source_a"; b=tmp_path/"source_b"
    a.mkdir(); b.mkdir()
    (a/"consumption_ledger.json").write_text(json.dumps({"dispatches": rows}))
    (b/"consumption_ledger.json").write_text(json.dumps({"reservations": []}))
    return a,b

def row(claim="arbitrary-attempt"):
    return dict(claim_id=claim, timestamp="2026-01-01", case_id="x", kind="generation", model="mock", status="RECONCILED", input_tokens=7, output_tokens=3)

@pytest.mark.parametrize("fault", ["unknown", "missing", "corrupt", "conflicting"])
def test_a5_1_unverified_blocks_actual_reserve(tmp_path, fault):
    r=row()
    if fault=="unknown": r.update(status="UNKNOWN", input_tokens=None, output_tokens=None)
    a,b=sources(tmp_path,[r])
    if fault=="missing": (b/"consumption_ledger.json").unlink()
    if fault=="corrupt": (b/"consumption_ledger.json").write_text("{")
    if fault=="conflicting": (b/"consumption_ledger.json").write_text(json.dumps({"reservations":[dict(row(), input_tokens=800)]}))
    l=DurableSQLiteLedger(tmp_path/"ledger.db")
    l.seed_historical_runs(a,b)
    s=l.get_summary()
    assert s["accounting_status"]=="unverified"
    assert s["remaining_cost_usd"] is None
    with pytest.raises(RuntimeError): reserve(l)

def test_a5_1_generic_overlap_restart_and_no_estimate_as_measured(tmp_path):
    a,b=sources(tmp_path,[row("unusual-id-42")])
    (b/"consumption_ledger.json").write_text(json.dumps({"reservations":[row("unusual-id-42")]}))
    l=DurableSQLiteLedger(tmp_path/"ledger.db"); l.seed_historical_runs(a,b)
    l=DurableSQLiteLedger(tmp_path/"ledger.db"); l.seed_historical_runs(a,b)
    assert l.get_summary()["known_requests"]==1

def test_a5_2_generation_embedding_tokens_separate(tmp_path):
    l=DurableSQLiteLedger(tmp_path/"ledger.db")
    with l._connect() as c: c.execute("UPDATE limits SET max_input_tokens=10, max_emb_tokens=10")
    e=reserve(l,"embedding",10,0); l.reconcile(e,10,0,0,True)
    g=reserve(l,inp=10,out=0); l.reconcile(g,10,0,0,True)
    assert l.get_summary()["committed_usage"]["input_tokens"]==10

@pytest.mark.parametrize("limit,kind,inp,out",[("max_gen","generation",1,1),("max_emb","embedding",1,0),("max_input_tokens","generation",11,0),("max_output_tokens","generation",0,11),("max_emb_tokens","embedding",11,0),("max_cost_usd","generation",100000,0)])
def test_a5_2_all_caps_before_transport(tmp_path,limit,kind,inp,out):
    l=DurableSQLiteLedger(tmp_path/"ledger.db")
    with l._connect() as c: c.execute(f"UPDATE limits SET {limit}=?",(0 if limit in ("max_gen","max_emb","max_cost_usd") else 10,))
    calls=[]
    with pytest.raises(RuntimeError):
        reserve(l,kind,inp,out); calls.append("transport")
    assert calls==[]

def _compete(path, ready, go, q):
    l=DurableSQLiteLedger(Path(path),max_generations_sublimit=1)
    ready.put(True); go.wait()
    try: reserve(l); q.put("dispatch")
    except RuntimeError: q.put("blocked")

def test_a5_3_two_processes_last_slot(tmp_path):
    ctx=multiprocessing.get_context("spawn"); ready=ctx.Queue(); q=ctx.Queue(); go=ctx.Event()
    path=tmp_path/"ledger.db"; DurableSQLiteLedger(path)
    ps=[ctx.Process(target=_compete,args=(str(path),ready,go,q)) for _ in range(2)]
    for p in ps: p.start()
    for p in ps: assert ready.get(timeout=10)
    go.set()
    results=[q.get(timeout=10) for p in ps]
    for p in ps: p.join(10); assert p.exitcode==0
    assert sorted(results)==["blocked","dispatch"]

def test_a5_3_restart_and_idempotent_reconciliation(tmp_path):
    path=tmp_path/"ledger.db"; l=DurableSQLiteLedger(path,max_generations_sublimit=1)
    claim=reserve(l,inp=100,out=50)
    l=DurableSQLiteLedger(path,max_generations_sublimit=1)
    with pytest.raises(RuntimeError): reserve(l)
    l.reconcile(claim,10,5,0,True)
    first=l.get_summary()
    l.reconcile(claim,10,5,0,True)
    assert l.get_summary()==first
    with pytest.raises(RuntimeError): l.reconcile(claim,0,0,0,True)
    assert l.get_summary()==first

def test_a5_2_sdk_payload_reserve_and_missing_bound(tmp_path):
    from types import SimpleNamespace
    from scripts.v2_pilot import _install_sdk_interceptor
    calls=[]; payloads=[]
    class MockModels:
        def generate_content(self, **kw):
            # Real gate must have committed a reservation before entering transport.
            with l._connect() as c:
                assert c.execute("SELECT COUNT(*) FROM reservations WHERE status='RESERVED'").fetchone()[0]==1
            calls.append(kw)
            return SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=1,candidates_token_count=1,thoughts_token_count=1))
        def embed_content(self, **kw): calls.append(kw)
    l=DurableSQLiteLedger(tmp_path/"ledger.db")
    _install_sdk_interceptor(l,models_class=MockModels)
    with pytest.raises(RuntimeError,match="defensible"): MockModels().generate_content(model="mock",contents="input")
    assert calls==[] and l.get_summary()["known_requests"]==0
    # A separate mock transport's synthetic tokenizer defines its own exact cost;
    # this is not a Gemini tokenizer or a certified live token bound.
    class Synthetic(MockModels):
        pass
    def transport(self,**kw):
        with l._connect() as c:
            assert c.execute("SELECT COUNT(*) FROM reservations WHERE status='RESERVED'").fetchone()[0]==1
        calls.append(kw)
        return SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=1,candidates_token_count=1,thoughts_token_count=1))
    Synthetic.generate_content=transport
    def bound(kind,args,kwargs):
        payload=json.dumps([args,kwargs],sort_keys=True)
        payloads.append(payload)
        return dict(certified=True,input_tokens=len(payload.encode()),output_tokens=10 if kind=="generation" else 0)
    now=[100.0]
    _install_sdk_interceptor(l,models_class=Synthetic,bound_provider=bound,clock=lambda:now[0],min_interval_s=2)
    m=Synthetic()
    m.generate_content(model="mock",contents="input",config={"system_instruction":"a"})
    with pytest.raises(RuntimeError,match="interval"): m.generate_content(model="mock",contents="input")
    now[0]+=2
    m.generate_content(model="mock",contents="input",config={"system_instruction":"expanded","schema":{"x":"y"},"tools":["tool"]})
    assert len(calls)==2
    with l._connect() as c:
        rows=c.execute("SELECT est_input,actual_output FROM reservations ORDER BY timestamp").fetchall()
    assert rows[1]["est_input"]>rows[0]["est_input"]
    assert all(r["actual_output"]==2 for r in rows)


def _interrupt_after_reserve(path):
    import os
    l=DurableSQLiteLedger(Path(path),max_generations_sublimit=1)
    reserve(l,inp=100,out=50)
    os._exit(17)


def test_a5_3_process_interrupt_keeps_unknown_outcome_reservation(tmp_path):
    path=tmp_path/"ledger.db"
    p=multiprocessing.get_context("spawn").Process(target=_interrupt_after_reserve,args=(str(path),))
    p.start(); p.join(10); assert p.exitcode==17
    l=DurableSQLiteLedger(path,max_generations_sublimit=1)
    s=l.get_summary()
    assert s["known_requests"]==1 and s["reserved_cost_upper_usd"]>0
    assert s["measured_usage"]["generations"]==0
    with pytest.raises(RuntimeError): reserve(l)


def test_a5_4_estimates_are_not_measured_and_unknown_sdk_blocked(tmp_path):
    from scripts.v2_pilot import _install_sdk_interceptor
    a,b=sources(tmp_path,[dict(row(),kind="embedding",status="ESTIMATED_RECONCILED")])
    l=DurableSQLiteLedger(tmp_path/"ledger.db"); l.seed_historical_runs(a,b)
    calls=[]
    class SDK:
        def generate_content(self,**kw): calls.append(kw)
        def embed_content(self,**kw): calls.append(kw)
    _install_sdk_interceptor(l,models_class=SDK,bound_provider=lambda *a:dict(certified=True,input_tokens=1,output_tokens=1))
    with pytest.raises(RuntimeError,match="unverified"): SDK().generate_content(model="mock",contents="test")
    assert calls==[]
    s=l.get_summary()
    assert s["measured_usage"]["embeddings"]==0
    assert s["estimated_usage"]["estimated_embedding_tokens"]==7
    assert s["live_enabled"] is False

def test_a5_2_default_sdk_without_bound_never_calls_transport(tmp_path,monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from scripts.v2_pilot import _install_sdk_interceptor
    calls=[]
    class SDK:
        def generate_content(self,**kw):
            calls.append(kw)
            return SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=1,candidates_token_count=1,thoughts_token_count=0))
        def embed_content(self,**kw): calls.append(kw)
    for name in ("google","google.genai","google.genai.models"):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules["google.genai.models"].Models=SDK
    l=DurableSQLiteLedger(tmp_path/"ledger.db")
    _install_sdk_interceptor(l)
    with pytest.raises(RuntimeError): SDK().generate_content(model="mock",contents="input",config=SimpleNamespace(system_instruction="a",response_json_schema={},tools=[]))
    assert calls==[]

def test_a5_1_required_source_removed_after_import_blocks_reserve(tmp_path):
    a,b=sources(tmp_path,[row()])
    l=DurableSQLiteLedger(tmp_path/"ledger.db"); l.seed_historical_runs(a,b)
    assert l.get_summary()["accounting_status"]=="bounded"
    (b/"consumption_ledger.json").unlink()
    assert l.get_summary()["remaining_cost_usd"] is None
    with pytest.raises(RuntimeError): reserve(l)


def test_a5_3_minimum_interval_shared_by_restarted_ledger(tmp_path):
    path=tmp_path/"ledger.db"
    l=DurableSQLiteLedger(path)
    claim=l.reserve("generation",10,10,"B","synthetic","mock",dispatch_time=100,min_interval_s=2)
    l.reconcile(claim,1,1,0,True)
    l=DurableSQLiteLedger(path)
    with pytest.raises(RuntimeError,match="interval"):
        l.reserve("generation",10,10,"B","synthetic","mock",dispatch_time=101,min_interval_s=2)
    l.reserve("generation",10,10,"B","synthetic","mock",dispatch_time=102,min_interval_s=2)
    assert l.get_summary()["known_requests"]==2
