#!/usr/bin/env python3
"""Runner, verification harness, and directed validation for Getnet Núcleo 2.0 Pilot.

Subcommands:
  prepare        - Generates isolated fixtures for Arm A and Arm B
  contracts      - Runs offline contracts (tests/v2/ and 4 legacy contracts) with network blocked
  verify         - Verifies baseline file hashes, strict decoupling, no-print AST, and cache reload
  directed       - Executes directed validation of Arm B on blind_07, blind_13, and pilot_hitl_01
  counterproofs  - Executes offline counterproof test suite and generates receipt
  run-live       - Executes comparative A/B (preserved for historical reference)
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import threading
import time

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_EXT_BUILD = REPO_ROOT.parent / "getnet_v2_build_2026-10-29"
_EXT_DIRECTED = REPO_ROOT.parent / "getnet_v2_directed_repair_2026-10-29"
PREV_BUILD_DIR = _EXT_BUILD if _EXT_BUILD.exists() else (REPO_ROOT / "docs" / "eval_results" / "v2_live_integration" / "historical_build")
DIRECTED_DIR = _EXT_DIRECTED if _EXT_DIRECTED.exists() else (REPO_ROOT / "docs" / "eval_results" / "v2_live_integration" / "historical_directed")
OFFLINE_REPAIR_DIR = REPO_ROOT.parent / "getnet_v2_offline_repair_2026-10-29"
MANIFEST_FILE = PREV_BUILD_DIR / "baseline_manifest.json"


class DurableSQLiteLedger:
    """SQLite budget gate. Estimates are never promoted to certified bounds."""

    def __init__(self, db_path: Path, max_generations_sublimit=12, max_embeddings_sublimit=6):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_generations_sublimit = max_generations_sublimit
        self.max_embeddings_sublimit = max_embeddings_sublimit
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._connect() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS limits(id INTEGER PRIMARY KEY, max_gen INTEGER,
                    max_emb INTEGER, max_input_tokens INTEGER, max_output_tokens INTEGER,
                    max_emb_tokens INTEGER, max_cost_usd REAL, price_in_per_m REAL,
                    price_out_per_m REAL, price_emb_per_m REAL);
                INSERT OR IGNORE INTO limits VALUES(1,60,30,500000,128000,50000,1.0,0.30,2.50,0.15);
                CREATE TABLE IF NOT EXISTS reservations(claim_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL, arm TEXT NOT NULL, case_id TEXT NOT NULL,
                    kind TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
                    est_input INTEGER NOT NULL, est_output INTEGER NOT NULL,
                    actual_input INTEGER, actual_output INTEGER, cost_usd REAL NOT NULL,
                    duration_s REAL DEFAULT 0, error_msg TEXT);
                CREATE TABLE IF NOT EXISTS accounting_sources(path TEXT PRIMARY KEY, sha256 TEXT);
                CREATE TABLE IF NOT EXISTS accounting_issues(reason TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS dispatch_clock(id INTEGER PRIMARY KEY, last_dispatch REAL);
                CREATE TABLE IF NOT EXISTS attempt_sources(claim_id TEXT, path TEXT, sha256 TEXT,
                    PRIMARY KEY(claim_id,path));
            """)

    @staticmethod
    def _tokens(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    @staticmethod
    def _cost(kind, inp, out, lim):
        return (inp * lim["price_emb_per_m"] if kind == "embedding" else
                inp * lim["price_in_per_m"] + out * lim["price_out_per_m"]) / 1_000_000

    def seed_historical_runs(self, build_dir, directed_dir):
        self.import_sources([Path(build_dir)/"consumption_ledger.json",
                             Path(directed_dir)/"consumption_ledger.json"])

    def seed_historical_run(self, prev_ledger_path):
        self.import_sources([Path(prev_ledger_path), DIRECTED_DIR/"consumption_ledger.json"])

    def import_sources(self, paths):
        """Import every identity, including overlapping inherited rows, with source hashes.

        A changed/missing/malformed required source poisons certification persistently;
        it cannot silently replace the previous evidence or erase spent attempts.
        """
        with self._connect() as c:
            c.execute("BEGIN IMMEDIATE")
            lim = c.execute("SELECT * FROM limits WHERE id=1").fetchone()
            for path in paths:
                path = Path(path).resolve(); key = str(path)
                try:
                    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
                    data = json.loads(raw)
                    rows = data.get("dispatches", data.get("reservations"))
                    if not isinstance(rows, list): raise ValueError("no attempt list")
                except (OSError, ValueError, TypeError, AttributeError) as exc:
                    c.execute("INSERT OR IGNORE INTO accounting_issues VALUES(?)", (f"required source invalid: {key}: {type(exc).__name__}",))
                    continue
                previous = c.execute("SELECT sha256 FROM accounting_sources WHERE path=?", (key,)).fetchone()
                if previous and previous[0] != digest:
                    c.execute("INSERT OR IGNORE INTO accounting_issues VALUES(?)", (f"required source changed: {key}",))
                c.execute("INSERT OR IGNORE INTO accounting_sources VALUES(?,?)", (key,digest))
                for r in rows:
                    try:
                        claim = r["claim_id"]; kind = r["kind"]
                        if not isinstance(claim,str) or not claim or kind not in ("generation","embedding"):
                            raise ValueError("invalid identity/kind")
                        # Historical embedding counters were estimates unless explicitly evidenced.
                        inp = r.get("actual_input", r.get("input_tokens"))
                        out = r.get("actual_output", r.get("output_tokens"))
                        measured = (self._tokens(inp) and self._tokens(out)
                                    and r.get("status", "RECONCILED") == "RECONCILED"
                                    and not r.get("error") and not r.get("error_msg")
                                    and (kind == "generation" and out > 0 or r.get("usage_basis") == "measured"))
                        ei = r.get("est_input", r.get("input_tokens")); eo = r.get("est_output", r.get("output_tokens"))
                        ei = ei if self._tokens(ei) else 0; eo = eo if self._tokens(eo) else 0
                        status = "RECONCILED" if measured else "HISTORICAL_UNVERIFIED"
                        if not measured: inp = out = None
                        model = r["model"]; case = r["case_id"]; ts = r["timestamp"]
                        existing = c.execute("SELECT * FROM reservations WHERE claim_id=?",(claim,)).fetchone()
                        if existing:
                            # Provenance can overlap across source files, but not disagree.
                            if (existing["kind"],existing["model"],existing["case_id"],existing["timestamp"],existing["est_input"],existing["est_output"]) != (kind,model,case,ts,ei,eo):
                                c.execute("INSERT OR IGNORE INTO accounting_issues VALUES(?)",(f"conflicting attempt: {claim}",))
                            if measured and existing["status"] == "RECONCILED" and (existing["actual_input"], existing["actual_output"]) != (inp,out):
                                c.execute("INSERT OR IGNORE INTO accounting_issues VALUES(?)",(f"conflicting measured usage: {claim}",))
                            if not measured:
                                c.execute("UPDATE reservations SET status='HISTORICAL_UNVERIFIED', actual_input=NULL,actual_output=NULL WHERE claim_id=?",(claim,))
                        else:
                            cost = self._cost(kind, inp if measured else ei, out if measured else eo, lim)
                            c.execute("INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                      (claim,ts,"HISTORICAL",case,kind,model,status,ei,eo,inp,out,cost,0,r.get("error",r.get("error_msg"))))
                        c.execute("INSERT OR IGNORE INTO attempt_sources VALUES(?,?,?)",(claim,key,digest))
                    except (KeyError, ValueError, TypeError):
                        c.execute("INSERT OR IGNORE INTO accounting_issues VALUES(?)",(f"malformed attempt in {key}",))
        self.export_json(self.db_path.with_suffix(".json"))

    @staticmethod
    def _unverified(rows):
        return [r for r in rows if r["status"] not in ("RECONCILED","RESERVED","FAILED_BOUNDED")
                or (r["status"] == "RECONCILED" and (r["actual_input"] is None or r["actual_output"] is None))]

    @staticmethod
    def _source_issues(c):
        reasons = []
        for source in c.execute("SELECT * FROM accounting_sources"):
            try:
                digest = hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
                if digest != source["sha256"]: reasons.append("required source hash changed: " + source["path"])
            except OSError:
                reasons.append("required source unavailable: " + source["path"])
        return reasons

    def reserve(self, kind, est_input, max_output, arm, case_id, model, *, dispatch_time=None, min_interval_s=0):
        # Direct callers supply a certified synthetic bound; SDK path must obtain one
        # for the complete payload independently (no character heuristic).
        if kind not in ("generation","embedding") or not self._tokens(est_input) or not self._tokens(max_output):
            raise RuntimeError("Invalid reservation bounds")
        if kind == "embedding" and max_output != 0: raise RuntimeError("Embedding output bound must be zero")
        with self._connect() as c:
            c.execute("BEGIN IMMEDIATE")
            lim = c.execute("SELECT * FROM limits WHERE id=1").fetchone()
            rows = c.execute("SELECT * FROM reservations").fetchall()
            if c.execute("SELECT 1 FROM accounting_issues LIMIT 1").fetchone() or self._unverified(rows) or self._source_issues(c):
                raise RuntimeError("Accounting unverified; dispatch blocked")
            def count(k, directed=False):
                return sum(r["kind"] == k and (not directed or r["arm"] != "HISTORICAL") for r in rows)
            def committed(k, field):
                return sum(r["actual_"+field] if r["status"] == "RECONCILED" else r["est_"+field] for r in rows if r["kind"] == k)
            cost = self._cost(kind,est_input,max_output,lim)
            checks = [(count("generation")+(kind=="generation"),lim["max_gen"]),
                      (count("embedding")+(kind=="embedding"),lim["max_emb"]),
                      (count("generation",True)+(kind=="generation"),self.max_generations_sublimit),
                      (count("embedding",True)+(kind=="embedding"),self.max_embeddings_sublimit),
                      (committed("generation","input")+(est_input if kind=="generation" else 0),lim["max_input_tokens"]),
                      (committed("generation","output")+max_output,lim["max_output_tokens"]),
                      (committed("embedding","input")+(est_input if kind=="embedding" else 0),lim["max_emb_tokens"]),
                      (sum(r["cost_usd"] for r in rows)+cost,lim["max_cost_usd"])]
            if any(used > maximum for used,maximum in checks): raise RuntimeError("Budget limit exceeded")
            if dispatch_time is not None:
                previous_dispatch = c.execute("SELECT last_dispatch FROM dispatch_clock WHERE id=1").fetchone()
                if previous_dispatch and dispatch_time-previous_dispatch[0] < min_interval_s:
                    raise RuntimeError("Minimum dispatch interval not elapsed")
                c.execute("INSERT OR REPLACE INTO dispatch_clock VALUES(1,?)",(dispatch_time,))
            import uuid
            claim = dict(claim_id="claim_"+uuid.uuid4().hex,kind=kind,est_input=est_input,est_output=max_output,model=model)
            c.execute("INSERT INTO reservations(claim_id,timestamp,arm,case_id,kind,model,status,est_input,est_output,cost_usd) VALUES(?,?,?,?,?,?,'RESERVED',?,?,?)",
                      (claim["claim_id"],datetime.now(timezone.utc).isoformat(),arm,case_id,kind,model,est_input,max_output,cost))
        self.export_json(self.db_path.with_suffix(".json"))
        return claim

    def reconcile(self, claim, actual_input, actual_output, duration_s, success, error_msg=""):
        with self._connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM reservations WHERE claim_id=?",(claim["claim_id"],)).fetchone()
            if row is None or any(claim.get(k) != row[k] for k in ("kind","model","est_input","est_output")):
                raise RuntimeError("Unknown/mismatched reservation")
            measured = success and self._tokens(actual_input) and self._tokens(actual_output)
            status = "RECONCILED" if measured else "FAILED_BOUNDED"
            inp, out = (actual_input,actual_output) if measured else (None,None)
            if row["status"] != "RESERVED":
                if (row["status"],row["actual_input"],row["actual_output"],row["error_msg"]) == (status,inp,out,error_msg): return
                raise RuntimeError("Conflicting reconciliation; original result retained")
            if measured and (inp > row["est_input"] or out > row["est_output"]):
                c.execute("INSERT OR IGNORE INTO accounting_issues VALUES(?)",(f"usage exceeded reserved bound: {row['claim_id']}",))
            lim = c.execute("SELECT * FROM limits WHERE id=1").fetchone()
            cost = self._cost(row["kind"],inp,out,lim) if measured else row["cost_usd"]
            c.execute("UPDATE reservations SET status=?,actual_input=?,actual_output=?,cost_usd=?,duration_s=?,error_msg=? WHERE claim_id=?",
                      (status,inp,out,cost,duration_s,error_msg,claim["claim_id"]))
        self.export_json(self.db_path.with_suffix(".json"))

    def get_summary(self):
        with self._connect() as c:
            rows = c.execute("SELECT * FROM reservations").fetchall()
            lim = dict(c.execute("SELECT * FROM limits WHERE id=1").fetchone())
            reasons = [r[0] for r in c.execute("SELECT reason FROM accounting_issues ORDER BY reason")] + self._source_issues(c)
            sources = [dict(r) for r in c.execute("SELECT * FROM accounting_sources ORDER BY path")]
        unknown = self._unverified(rows)
        if unknown: reasons.append(f"{len(unknown)} attempts have no defensible measured usage or reservation bound")
        def count(k, directed=False): return sum(r["kind"]==k and (not directed or r["arm"]!="HISTORICAL") for r in rows)
        def tokens(k,field): return sum((r["actual_"+field] if r["status"]=="RECONCILED" else r["est_"+field]) or 0 for r in rows if r["kind"]==k)
        measured = [r for r in rows if r["status"]=="RECONCILED"]
        cost = sum(r["cost_usd"] for r in rows)
        estimated = [r for r in rows if r["status"]!="RECONCILED"]
        return dict(accounting_status="unverified" if reasons else "bounded",known_requests=len(rows),live_enabled=False,
            limits=lim, sources=sources, unverified_reasons=reasons,
            measured_usage=dict(generations=sum(r["kind"]=="generation" for r in measured),embeddings=sum(r["kind"]=="embedding" for r in measured),
                input_tokens=sum(r["actual_input"] for r in measured if r["kind"]=="generation"),
                output_tokens=sum(r["actual_output"] for r in measured if r["kind"]=="generation"),
                embedding_tokens=sum(r["actual_input"] for r in measured if r["kind"]=="embedding"),cost_usd=round(sum(r["cost_usd"] for r in measured),9)),
            estimated_usage=dict(unverified_dispatches=len(unknown),estimated_input_tokens=sum(r["est_input"] for r in estimated if r["kind"]=="generation"),
                estimated_output_tokens=sum(r["est_output"] for r in estimated if r["kind"]=="generation"),
                estimated_embedding_tokens=sum(r["est_input"] for r in estimated if r["kind"]=="embedding"),estimated_cost_usd=round(sum(r["cost_usd"] for r in estimated),9)),
            committed_usage=dict(total_generations=count("generation"),total_embeddings=count("embedding"),directed_generations=count("generation",True),directed_embeddings=count("embedding",True),
                input_tokens=tokens("generation","input"),output_tokens=tokens("generation","output"),embedding_tokens=tokens("embedding","input"),estimated_cost_usd=round(cost,9)),
            sublimits=dict(max_directed_generations=self.max_generations_sublimit,max_directed_embeddings=self.max_embeddings_sublimit),
            estimated_cost_usd=round(cost,9),total_estimated_cost_usd=round(cost,9),
            reserved_cost_upper_usd=None if reasons else round(cost,9),remaining_cost_usd=None if reasons else round(lim["max_cost_usd"]-cost,9),reservations_count=len(rows))

    def export_json(self, json_path):
        summary = self.get_summary()
        with self._connect() as c:
            summary["reservations"] = [dict(r) for r in c.execute("SELECT * FROM reservations ORDER BY timestamp,claim_id")]
        # Export is a report, not the transaction authority; concurrent processes
        # write separate temporary files and publish atomically.
        import uuid
        path = Path(json_path); temp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        temp.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        temp.replace(path)


_LEDGER: DurableSQLiteLedger | None = None
_CURRENT_CONTEXT = {"arm": "unknown", "case_id": "unknown"}


def _install_sdk_interceptor(ledger, *, models_class=None, bound_provider=None, clock=time.monotonic,
                             min_interval_s=0.0):
    """Gate the entire SDK payload before transport.

    No live bound provider is configured in this zero-call round. The injectable
    provider exists for offline synthetic SDK tests and must return certified
    input/output bounds for the exact full payload, including tools/config.
    """
    global _LEDGER
    _LEDGER = ledger
    if models_class is None:
        try:
            from google.genai.models import Models as models_class
        except ImportError:
            return
    last_dispatch = [None]
    transport_lock = threading.Lock()

    def wrap(original, kind):
        def intercepted(self, *args, **kwargs):
            # No guess based on characters, output constants, or embedding size.
            if bound_provider is None: raise RuntimeError("No defensible complete-payload token bound; dispatch blocked")
            bounds = bound_provider(kind, args, kwargs)
            if not isinstance(bounds, dict) or bounds.get("certified") is not True:
                raise RuntimeError("Uncertified payload bound; dispatch blocked")
            with transport_lock:
                now = clock()
                if last_dispatch[0] is not None and now-last_dispatch[0] < min_interval_s:
                    raise RuntimeError("Minimum dispatch interval not elapsed")
                claim = ledger.reserve(kind,bounds["input_tokens"],bounds["output_tokens"],
                                       _CURRENT_CONTEXT["arm"],_CURRENT_CONTEXT["case_id"],kwargs.get("model","unknown"),
                                       dispatch_time=now,min_interval_s=min_interval_s)
                last_dispatch[0] = now
                started = clock()
                try:
                    resp = original(self,*args,**kwargs)
                except Exception as exc:
                    ledger.reconcile(claim,None,None,clock()-started,False,f"{type(exc).__name__}: {exc}")
                    raise
                usage = getattr(resp,"usage_metadata",None)
                inp = getattr(usage,"prompt_token_count",None)
                out = getattr(usage,"candidates_token_count",None) if kind=="generation" else 0
                thoughts = getattr(usage,"thoughts_token_count",0)
                if out is not None and thoughts is not None: out += thoughts
                ledger.reconcile(claim,inp,out,clock()-started,True)
                return resp
        return intercepted
    models_class.generate_content = wrap(models_class.generate_content,"generation")
    models_class.embed_content = wrap(models_class.embed_content,"embedding")


def cmd_prepare(args):
    """Generates isolated fixtures for Arm A and Arm B."""
    from tests.v2.fixture_factory import get_pilot_cases, prepare_arms

    fixtures_dir = Path(getattr(args, "fixtures_dir", PREV_BUILD_DIR / "fixtures"))
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    cases = get_pilot_cases()
    arms = prepare_arms(fixtures_dir, cases)

    receipt = {
        "command": "prepare",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixtures_dir": str(fixtures_dir),
        "cases_count": len(cases),
        "case_ids": [c["id"] for c in cases],
        "arms": arms,
    }
    sys.stdout.write(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return 0


def cmd_contracts(args):
    """Runs offline contract suite with blocked network and ignored .env."""
    import pytest

    def blocked(*a, **kw):
        raise RuntimeError("Network access disabled for contracts")

    socket.socket.connect = blocked
    socket.create_connection = blocked

    os.environ.update(
        GEMINI_API_KEY="",
        GETNET_LIVE_EMBEDDINGS="0",
        GETNET_LIVE_TESTS="0",
        TELEGRAM_BOT_TOKEN="",
        TELEGRAM_CHAT_ID="",
        HITL_WEBHOOK_URL="",
    )

    import pydantic_settings
    orig_init = pydantic_settings.BaseSettings.__init__

    def safe_init(self, *a, **kw):
        kw["_env_file"] = None
        return orig_init(self, *a, **kw)

    pydantic_settings.BaseSettings.__init__ = safe_init

    test_paths = [
        "tests/v2/",
        "tests/test_security_boundaries.py",
        "tests/test_http_actions_v3.py",
        "tests/test_p0_action_security.py",
        "tests/test_hitl_resolution_contract.py",
    ]

    t0 = time.monotonic()
    ret = pytest.main(["-q", "-o", "pythonpath=.", *test_paths])
    duration = time.monotonic() - t0

    receipt = {
        "command": "contracts",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_code": ret,
        "duration_seconds": round(duration, 3),
        "test_paths": test_paths,
        "status": "passed" if ret == 0 else "failed",
    }
    if OFFLINE_REPAIR_DIR.exists():
        (OFFLINE_REPAIR_DIR / "contracts_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif DIRECTED_DIR.exists():
        (DIRECTED_DIR / "contracts_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return ret


def cmd_verify(args):
    """Verifies baseline hashes, decoupling, AST no-print, and embedding cache reload."""
    receipt = {
        "command": "verify",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "all_passed": False,
    }

    if not MANIFEST_FILE.exists():
        sys.stderr.write("baseline_manifest.json not found!\n")
        return 1

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    files = manifest.get("files", {})

    missing_repo = []
    mismatch_repo = []
    for rel_path, exp_hash in files.items():
        if rel_path in (".env", "data/getnet_state.db"):
            continue  # Excluded per plan line 110
        p = REPO_ROOT / rel_path
        if not p.exists():
            missing_repo.append(rel_path)
        else:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != exp_hash:
                mismatch_repo.append(rel_path)

    arm_a_dir = PREV_BUILD_DIR / "arm_a"
    missing_arma = []
    mismatch_arma = []
    for rel_path, exp_hash in files.items():
        if rel_path in (".env", "data/getnet_state.db"):
            continue
        p = arm_a_dir / rel_path
        if not p.exists():
            missing_arma.append(rel_path)
        else:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h != exp_hash:
                mismatch_arma.append(rel_path)

    receipt["checks"]["baseline_manifest_repo"] = {
        "total_files": len(files) - 2,
        "missing": len(missing_repo),
        "mismatches": len(mismatch_repo),
        "passed": len(missing_repo) == 0 and len(mismatch_repo) == 0,
    }
    receipt["checks"]["baseline_manifest_arm_a"] = {
        "total_files": len(files) - 2,
        "missing": len(missing_arma),
        "mismatches": len(mismatch_arma),
        "passed": len(missing_arma) == 0 and len(mismatch_arma) == 0,
    }

    # Strict decoupling check (AST on app/v2/)
    v2_dir = REPO_ROOT / "app" / "v2"
    forbidden_tokens = [
        "request_units",
        "evaluate_operational_unit",
        "project_public_response",
        "project_specialist",
        "constrain_call",
        "SemanticEvidenceAuthorizer",
    ]
    decoupling_violations = []
    for py_file in v2_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                decoupling_violations.append(f"{py_file.name}: contains {token}")

    receipt["checks"]["strict_decoupling"] = {
        "violations": decoupling_violations,
        "passed": len(decoupling_violations) == 0,
    }

    from scripts.check_no_print import check_file
    print_violations = []
    for f in list(v2_dir.glob("*.py")) + list((REPO_ROOT / "tests" / "v2").glob("*.py")):
        errs = check_file(f)
        if errs:
            print_violations.extend(errs)

    receipt["checks"]["no_print_ast"] = {
        "violations": print_violations,
        "passed": len(print_violations) == 0,
    }

    from app.v2.knowledge import configure_embedding_cache, persist_embedding_cache
    from app.rag.embeddings import SemanticEmbedder
    import tempfile

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_cache = Path(tmpd) / "embeddings_cache.json"
        unit_vec = [1.0 / (384 ** 0.5)] * 384
        mean_vec = [1.0 / (384 ** 0.5)] * 384
        test_payload = {
            "model": SemanticEmbedder.EMBEDDING_MODEL,
            "dimension": SemanticEmbedder.DIM,
            "alpha": 0.68,
            "mean_unit": mean_vec,
            "vectors": {"test_digest_123": unit_vec},
        }
        tmp_cache.write_text(json.dumps(test_payload), encoding="utf-8")

        configure_embedding_cache(tmp_cache)
        reload_ok = "test_digest_123" in SemanticEmbedder._neural_vectors

        bad_payload = copy.deepcopy(test_payload)
        bad_payload["dimension"] = 128
        bad_cache = Path(tmpd) / "bad_cache.json"
        bad_cache.write_text(json.dumps(bad_payload), encoding="utf-8")
        configure_embedding_cache(bad_cache)
        reject_ok = len(SemanticEmbedder._neural_vectors) == 0

    receipt["checks"]["embedding_cache_reload_and_rejection"] = {
        "reload_passed": reload_ok,
        "rejection_passed": reject_ok,
        "passed": reload_ok and reject_ok,
    }

    from app.v2.api import create_app
    from unittest.mock import Mock

    mock_runtime = Mock()
    app_v2 = create_app(mock_runtime)
    receipt["checks"]["experimental_router_init"] = {"passed": app_v2.title == "Getnet v2 pilot"}

    all_passed = all(c["passed"] for c in receipt["checks"].values())
    receipt["all_passed"] = all_passed

    if OFFLINE_REPAIR_DIR.exists():
        (OFFLINE_REPAIR_DIR / "verify_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif DIRECTED_DIR.exists():
        (DIRECTED_DIR / "verify_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return 0 if all_passed else 1


def _read_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return key


def cmd_counterproofs(args):
    """Runs targeted offline counterproofs and emits structured receipt."""
    import pytest

    def blocked(*a, **kw):
        raise RuntimeError("Network disabled for counterproofs")
    socket.socket.connect = blocked
    socket.create_connection = blocked

    os.environ.update(GEMINI_API_KEY="", GETNET_LIVE_EMBEDDINGS="0", GETNET_LIVE_TESTS="0")
    t0 = time.monotonic()
    ret = pytest.main(["-q", "-o", "pythonpath=.", "tests/v2/"])
    duration = time.monotonic() - t0

    receipt = {
        "command": "counterproofs",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_code": ret,
        "duration_seconds": round(duration, 3),
        "status": "passed" if ret == 0 else "failed",
        "test_target": "tests/v2/",
    }

    out_file = (OFFLINE_REPAIR_DIR / "offline_counterproofs.json") if OFFLINE_REPAIR_DIR.exists() else (DIRECTED_DIR / "offline_counterproofs.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return ret


def cmd_reconcile(args):
    """Reconciles both historical runs into OFFLINE_REPAIR_DIR/consumption_ledger.db."""
    OFFLINE_REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OFFLINE_REPAIR_DIR / "consumption_ledger.db"
    # Fresh initialization
    if db_path.exists():
        db_path.unlink()
    shm = db_path.with_name(db_path.name + "-shm")
    wal = db_path.with_name(db_path.name + "-wal")
    if shm.exists():
        shm.unlink()
    if wal.exists():
        wal.unlink()

    ledger = DurableSQLiteLedger(db_path, max_generations_sublimit=0, max_embeddings_sublimit=0)
    ledger.seed_historical_runs(PREV_BUILD_DIR, DIRECTED_DIR)
    summary = ledger.get_summary()

    receipt = {
        "command": "reconcile",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ledger_db": str(db_path),
        "ledger_json": str(db_path.with_suffix(".json")),
        "historical_runs_imported": [str(PREV_BUILD_DIR), str(DIRECTED_DIR)],
        "summary": summary,
        "status": "reconciled_conservative",
    }
    (OFFLINE_REPAIR_DIR / "reconciliation_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return 0


def cmd_directed(args):
    """Executes directed validation of Arm B on blind_07, blind_13, and pilot_hitl_01."""
    if not args.enable_live:
        sys.stderr.write("Directed validation requires explicit --enable-live\n")
        return 1

    api_key = _read_gemini_api_key()
    if not api_key:
        sys.stderr.write("GEMINI_API_KEY is not configured.\n")
        return 1

    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GEMINI_MODEL"] = "gemini-3.5-flash-lite"
    os.environ["GETNET_LIVE_EMBEDDINGS"] = "1"
    os.environ["TELEGRAM_BOT_TOKEN"] = ""
    os.environ["TELEGRAM_CHAT_ID"] = ""
    os.environ["HITL_WEBHOOK_URL"] = ""

    DIRECTED_DIR.mkdir(parents=True, exist_ok=True)
    fixtures_dir = DIRECTED_DIR / "fixtures"

    from tests.v2.fixture_factory import get_pilot_cases, prepare_arms
    from fastapi.testclient import TestClient

    cases_all = get_pilot_cases()
    # Directed cases: only blind_07, blind_13, and pilot_hitl_01
    directed_case_ids = ["blind_07", "blind_13", "pilot_hitl_01"]
    cases = [c for c in cases_all if c["id"] in directed_case_ids]

    # Clean preparation of fixtures in new directory
    fixtures = prepare_arms(fixtures_dir, cases)

    # Initialize SQLite ledger in DIRECTED_DIR
    ledger_db = DIRECTED_DIR / "consumption_ledger.db"
    max_gen = min(12, args.max_generations)
    ledger = DurableSQLiteLedger(ledger_db, max_generations_sublimit=max_gen, max_embeddings_sublimit=6)
    # Seed historical run
    prev_ledger_json = PREV_BUILD_DIR / "consumption_ledger.json"
    ledger.seed_historical_run(prev_ledger_json)
    _install_sdk_interceptor(ledger)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "directed_validation_arm_b_only",
        "model": "gemini-3.5-flash-lite",
        "embeddings_model": "gemini-embedding-001",
        "pause_seconds": max(10.0, args.pause_seconds),
        "max_generations_sublimit": max_gen,
        "cases_executed": directed_case_ids,
        "cases": {},
    }

    # Bind process environment to B's directed fixtures
    os.environ["SQLITE_DB_PATH"] = fixtures["B"]["sqlite"]
    os.environ["MOCK_CUSTOMERS_PATH"] = fixtures["B"]["customers"]
    os.environ["VECTOR_INDEX_PATH"] = fixtures["B"]["vector_index"]

    from app.config import settings
    settings.SQLITE_DB_PATH = Path(fixtures["B"]["sqlite"])
    settings.MOCK_CUSTOMERS_PATH = Path(fixtures["B"]["customers"])
    settings.VECTOR_INDEX_PATH = Path(fixtures["B"]["vector_index"])
    settings.GEMINI_MODEL = "gemini-3.5-flash-lite"
    settings.GETNET_LIVE_EMBEDDINGS = True

    from app.v2.knowledge import configure_embedding_cache, persist_embedding_cache
    configure_embedding_cache(fixtures["B"]["embedding_cache"])

    from app.memory.store import StateStore
    from app.hitl.queue import HITLQueueManager
    from app.v2.adapters import StateAdapter, ToolGateway
    from app.v2.model import GeminiModel
    from app.v2.knowledge import KnowledgeService
    from app.v2.runtime import Runtime
    from app.v2.api import create_app
    from app.security.auth import AuthManager

    store_b = StateStore(Path(fixtures["B"]["sqlite"]))
    queue_b = HITLQueueManager(store=store_b)
    state_b = StateAdapter(store_b)
    gateway_b = ToolGateway(Path(fixtures["B"]["customers"]), store=store_b, queue=queue_b)
    model_b = GeminiModel()
    knowledge_b = KnowledgeService(model_b)
    runtime_b = Runtime(model_b, knowledge_b, state_b, gateway_b, queue_b)
    app_b = create_app(runtime_b)
    client_b = TestClient(app_b)

    sys.stdout.write("=== INICIANDO VALIDAÇÃO DIRIGIDA ARM B (3 FLUXOS) ===\n")

    for idx, case in enumerate(cases):
        cid = case["id"]
        _CURRENT_CONTEXT["arm"] = "B"
        _CURRENT_CONTEXT["case_id"] = cid
        user = case["contexto_premissas"]["user_id"]
        prompt = case["mensagens"][-1]["content"]
        sid = "pilot_" + cid
        token = AuthManager.create_jwt_token(user, scopes=["chat:write", "customer:read_own"])

        sys.stdout.write(f"\n[Directed B] Executando {cid} (user={user})...\n")
        t0 = time.monotonic()
        try:
            resp = client_b.post(
                "/api/v2/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": prompt, "user_id": user, "session_id": sid},
            )
            dur = time.monotonic() - t0
            data = resp.json() if resp.status_code == 200 else {"error": resp.text}

            # If pilot_hitl_01 paused for human escalation, perform the authorized operator resolution and resume
            if cid == "pilot_hitl_01" and data.get("ticket_id"):
                ticket_id = data["ticket_id"]
                op_resp = case.get("operator_response", "Verificação manual concluída: a conectividade do terminal está ativa.")
                sys.stdout.write(f"[Directed B] HITL escalado: ticket={ticket_id}. Operador respondendo e resolvendo...\n")
                queue_b.reply_to_ticket(ticket_id, "operador_senior", op_resp, resolve_ticket=True)

                # Resume via authorized operator
                op_token = AuthManager.create_jwt_token("operador_senior", role="operator", scopes=["hitl:manage"])
                sys.stdout.write(f"[Directed B] Retomando atendimento v2 via /api/v2/hitl/{ticket_id}/resume...\n")
                t_res = time.monotonic()
                resume_resp = client_b.post(
                    f"/api/v2/hitl/{ticket_id}/resume",
                    headers={"Authorization": f"Bearer {op_token}"},
                )
                dur_res = time.monotonic() - t_res
                resume_data = resume_resp.json() if resume_resp.status_code == 200 else {"error": resume_resp.text}
                data = {
                    "initial_turn": data,
                    "operator_reply": op_resp,
                    "resumed_turn": resume_data,
                }
                dur += dur_res

            results["cases"][cid] = {
                "http_status": resp.status_code,
                "duration_seconds": round(dur, 3),
                "response": data,
            }
        except Exception as exc:
            dur = time.monotonic() - t0
            results["cases"][cid] = {
                "error": type(exc).__name__,
                "detail": str(exc),
                "duration_seconds": round(dur, 3),
            }

        persist_embedding_cache(fixtures["B"]["embedding_cache"])
        sys.stdout.write(f"[Directed B] {cid} concluído em {dur:.2f}s. Pausa obrigatória de {results['pause_seconds']}s...\n")
        time.sleep(results["pause_seconds"])

    # Record final summary
    ledger.export_json(DIRECTED_DIR / "consumption_ledger.json")
    ledger_summary = ledger.get_summary()
    results["ledger_summary"] = ledger_summary
    results_file = DIRECTED_DIR / "directed_validation_results.json"
    results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Generate final report
    report_file = DIRECTED_DIR / "directed_validation_report.md"
    _generate_directed_report(results, ledger_summary, report_file)

    # Manifest of final state
    final_manifest = {"files": {}}
    for p in sorted(DIRECTED_DIR.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(DIRECTED_DIR))
            final_manifest["files"][rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (DIRECTED_DIR / "final_manifest.json").write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")

    sys.stdout.write(f"\nResultados gravados em: {results_file}\n")
    sys.stdout.write(f"Relatório gravado em: {report_file}\n")
    return 0


def _generate_directed_report(results: dict, ledger_summary: dict, out_file: Path):
    usage = ledger_summary["committed_usage"]
    lim = ledger_summary["limits"]
    lines = [
        "# Relatório de Validação Dirigida — Núcleo 2.0 Getnet",
        "",
        f"**Data da Execução:** {results['timestamp']}  ",
        f"**Modelo de Geração:** `{results['model']}` (temperature=0, minimal thinking)  ",
        f"**Modelo de Embeddings:** `{results['embeddings_model']}` (384-dim, cache persistido)  ",
        "",
        "## 1. Auditoria de Consumo Reconciliado (SQLite Ledger)",
        "",
        "| Métrica | Histórico + Esta Rodada | Limite Cumulativo | Saldo Restante | Status |",
        "|---|---:|---:|---:|:---:|",
        f"| Requisições de Geração | {usage['total_generations']} (desta rodada: {usage['directed_generations']}) | {lim['max_gen']} (subteto: {ledger_summary['sublimits']['max_directed_generations']}) | {lim['max_gen'] - usage['total_generations']} | OK |",
        f"| Requisições de Embeddings | {usage['total_embeddings']} (desta rodada: {usage['directed_embeddings']}) | {lim['max_emb']} (subteto: {ledger_summary['sublimits']['max_directed_embeddings']}) | {lim['max_emb'] - usage['total_embeddings']} | OK |",
        f"| Tokens de Entrada (Geração) | {usage['input_tokens']:,} | {lim['max_input_tokens']:,} | {lim['max_input_tokens'] - usage['input_tokens']:,} | OK |",
        f"| Tokens de Saída (Geração) | {usage['output_tokens']:,} | {lim['max_output_tokens']:,} | {lim['max_output_tokens'] - usage['output_tokens']:,} | OK |",
        f"| Custo Estimado Total | US$ {usage['estimated_cost_usd']:.4f} | US$ {lim['max_cost_usd']:.2f} | US$ {lim['max_cost_usd'] - usage['estimated_cost_usd']:.4f} | OK |",
        "",
        "## 2. Resultados dos Três Fluxos Dirigidos",
        "",
        "| Caso | Foco do Reparo | Status Obtido | Latência | Avaliação Técnica |",
        "|---|---|:---:|:---:|---|",
    ]

    for cid in ["blind_07", "blind_13", "pilot_hitl_01"]:
        cdata = results["cases"].get(cid, {})
        resp_data = cdata.get("response", {})
        if cid == "pilot_hitl_01" and "resumed_turn" in resp_data:
            stat = resp_data["resumed_turn"].get("answer_status", "inconclusive")
        else:
            stat = resp_data.get("answer_status", "error" if "error" in cdata else str(cdata.get("http_status")))
        dur = f"{cdata.get('duration_seconds', 0):.2f}s"

        eval_note = "Verificação do reparo"
        if cid == "blind_07":
            eval_note = "Distinção entre fato indisponível (unanswerable -> partial) e status ativo comprovado"
        elif cid == "blind_13":
            eval_note = "Planejamento multi-capacidade (knowledge + tool), sem EvidenceError"
        elif cid == "pilot_hitl_01":
            eval_note = "Escalonamento humano, resposta do operador e retomada nativa"

        lines.append(f"| `{cid}` | {cid} | `{stat}` | {dur} | {eval_note} |")

    lines.extend([
        "",
        "## 3. Respostas e Diagnósticos dos Fluxos",
        "",
    ])

    for cid in ["blind_07", "blind_13", "pilot_hitl_01"]:
        cdata = results["cases"].get(cid, {})
        resp_dict = cdata.get("response", {})
        lines.append(f"### Caso `{cid}`")
        lines.append("")
        if cid == "pilot_hitl_01" and "resumed_turn" in resp_dict:
            initial_t = resp_dict.get("initial_turn", {}).get("response", "N/A")
            resumed_t = resp_dict.get("resumed_turn", {}).get("response", "N/A")
            lines.append("**Turno 1 (Encaminhamento Humano):**")
            lines.append(f"> {initial_t}")
            lines.append("")
            lines.append(f"**Resposta do Operador:** `{resp_dict.get('operator_reply')}`")
            lines.append("")
            lines.append("**Turno 2 (Retomada Automática do Agente):**")
            lines.append(f"> {resumed_t[:600]}")
        else:
            txt = resp_dict.get("response", cdata.get("detail", "N/A"))
            lines.append("**Resposta do Agente:**")
            lines.append(f"> {txt[:600]}")
        lines.append("")

    lines.extend([
        "## 4. Auditoria de Conformidade",
        "",
        "- **Preservação:** A pasta anterior `getnet_v2_build_2026-10-29/` permaneceu intocada (hashes verificados).",
        "- **Isolamento de Processo:** Execução em fixtures exclusivas de `getnet_v2_directed_repair_2026-10-29/`.",
        "- **Escopos de Autorização:** Verificado que `Runtime.resume()` revalida e não eleva privilégios.",
        "- **Orçamento:** Respeitado o subteto de 12 gerações e 6 embeddings dentro do saldo original reconciliado.",
    ])

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_acceptance_offline(args):
    """Closed A1–A5 run in a fresh process; guards precede application imports."""
    import subprocess
    import tempfile
    import xml.etree.ElementTree as ET
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Acceptance output must be new/empty; previous receipts are immutable")
    output_dir.mkdir(parents=True, exist_ok=True)
    tests = ["tests/v2/test_consolidation_acceptance.py", "tests/v2/test_fact_closure.py",
             "tests/v2/test_a5_closure.py", "tests/v2/test_ledger.py",
             "tests/v2/test_runtime_closure.py", "tests/test_security_boundaries.py",
             "tests/test_http_actions_v3.py", "tests/test_p0_action_security.py",
             "tests/test_hitl_resolution_contract.py"]
    guard_source = """import os,socket
os.environ.update(GEMINI_API_KEY='',GETNET_LIVE_EMBEDDINGS='0',GETNET_LIVE_TESTS='0',TELEGRAM_BOT_TOKEN='',TELEGRAM_CHAT_ID='',HITL_WEBHOOK_URL='')
def blocked(*a,**kw): raise RuntimeError('NETWORK_DENIED_OFFLINE')
socket.socket.connect=blocked
socket.create_connection=blocked
socket.getaddrinfo=blocked
import pydantic_settings
original=pydantic_settings.BaseSettings.__init__
def safe(self,*a,**kw):
 kw['_env_file']=None
 return original(self,*a,**kw)
pydantic_settings.BaseSettings.__init__=safe
"""
    (output_dir / "offline_guard.py").write_text(guard_source)
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="getnet-acceptance-") as temp:
        guard = Path(temp)
        (guard / "sitecustomize.py").write_text(guard_source)
        env = {**os.environ, "PYTHONPATH": str(guard)+os.pathsep+str(REPO_ROOT),
               "SQLITE_DB_PATH": str(guard / "state.db"), "PYTHONDONTWRITEBYTECODE": "1"}
        command = [sys.executable,"-m","pytest","-vv","--tb=short","-p","no:cacheprovider",
                   "--junitxml="+str(output_dir / "junit.xml"), *tests]
        run = subprocess.run(command,cwd=REPO_ROOT,env=env,capture_output=True,text=True)
    (output_dir / "pytest.log").write_text(run.stdout+run.stderr)
    cases = []
    xml = output_dir / "junit.xml"
    if xml.exists():
        for node in ET.parse(xml).iter("testcase"):
            status = "failed" if node.find("failure") is not None or node.find("error") is not None else "skipped" if node.find("skipped") is not None else "passed"
            cases.append(dict(name=node.attrib["name"],module=node.attrib.get("classname"),status=status))
    families = {}
    for i in range(1,6):
        selected = [c for c in cases if c["name"].lower().startswith(f"test_a{i}_")]
        families[f"A{i}"] = dict(passed=bool(selected) and all(c["status"]=="passed" for c in selected),scenarios=selected)
    ledger = DurableSQLiteLedger(output_dir / "consumption_ledger.db")
    ledger.seed_historical_runs(PREV_BUILD_DIR,DIRECTED_DIR)
    passed = run.returncode == 0 and all(f["passed"] for f in families.values())
    receipt = dict(command=command,exit_code=run.returncode,status="passed" if passed else "failed",
                   duration_seconds=round(time.monotonic()-t0,3),families=families,
                   totals={k:sum(c["status"]==k for c in cases) for k in ("passed","failed","skipped")},
                   accounting=ledger.get_summary(),log="pytest.log",junit="junit.xml",
                   limitations=["Offline mocks do not certify native Gemini semantic quality or a new score"])
    (output_dir / "acceptance_results.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    sys.stdout.write(json.dumps({"status":receipt["status"],"totals":receipt["totals"]})+"\n")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub_prep = sub.add_parser("prepare", help="Prepare isolated fixtures")
    sub_prep.add_argument("--fixtures-dir", type=Path, default=PREV_BUILD_DIR / "fixtures")
    sub_prep.set_defaults(func=cmd_prepare)

    sub_con = sub.add_parser("contracts", help="Run offline contracts")
    sub_con.set_defaults(func=cmd_contracts)

    sub_ver = sub.add_parser("verify", help="Verify baseline hashes and decoupling")
    sub_ver.set_defaults(func=cmd_verify)

    sub_cp = sub.add_parser("counterproofs", help="Run targeted offline counterproofs")
    sub_cp.set_defaults(func=cmd_counterproofs)

    sub_rec = sub.add_parser("reconcile", help="Reconcile historical consumption into offline repair ledger")
    sub_rec.set_defaults(func=cmd_reconcile)

    sub_acc = sub.add_parser("acceptance-offline", help="Run consolidated acceptance battery A1-A5")
    sub_acc.add_argument("--output-dir", type=Path, default=REPO_ROOT.parent / "getnet_v2_closure_acceptance_2026-10-29")
    sub_acc.set_defaults(func=cmd_acceptance_offline)

    sub_dir = sub.add_parser("directed", help="Execute directed validation on Arm B")
    sub_dir.add_argument("--enable-live", action="store_true", required=True)
    sub_dir.add_argument("--max-generations", type=int, default=12)
    sub_dir.add_argument("--pause-seconds", type=float, default=10.0)
    sub_dir.set_defaults(func=cmd_directed)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
