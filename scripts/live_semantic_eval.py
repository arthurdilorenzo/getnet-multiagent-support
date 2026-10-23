"""Opt-in semantic component receipt; not an end-to-end or independent grade.

Input is JSON {"cases": [{"id": "...", "message": "..."}]}. Defaults to at most
four cases; increasing the limit is explicit. Credentials are loaded by config.
"""
import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4)
    args = parser.parse_args()
    if os.environ.get('GETNET_LIVE_TESTS') != '1':
        raise SystemExit('Explicit GETNET_LIVE_TESTS=1 is required')
    if not 1 <= args.limit <= 48:
        raise SystemExit('Choose 1..48 cases explicitly')
    payload = json.loads(args.input.read_text())
    cases = payload['cases'][:args.limit]
    if not cases:
        raise SystemExit('Empty dataset')
    with tempfile.TemporaryDirectory(prefix='getnet-live-semantic-') as directory:
        os.environ.update(SQLITE_DB_PATH=str(Path(directory)/'state.db'),
                          VECTOR_INDEX_PATH=str(Path(directory)/'vectors.json'))
        from google import genai
        from app.config import settings
        from app.observability.tracer import SystemTracer
        from app.rag.retriever import retriever
        from app.rag.semantic import SemanticEvidenceAuthorizer
        from app.security.pii import PIIRedactor
        if not settings.GEMINI_API_KEY:
            raise SystemExit('GEMINI_API_KEY is not configured')
        logging.getLogger('getnet.multiagent').disabled = True
        client = genai.Client(api_key=settings.GEMINI_API_KEY,
                              http_options={'timeout':int(settings.LLM_TIMEOUT_SECONDS*1000)})
        authorizer = SemanticEvidenceAuthorizer(client)
        root = Path(__file__).resolve().parents[1]
        records = []
        for case in cases:
            message = PIIRedactor.mask_text(case['message'])
            identifier = str(case['id'])
            tracer = SystemTracer(session_id='live-semantic-'+identifier, user_id='synthetic')
            started = time.monotonic()
            candidates = retriever.retrieve_candidates(message, top_k=13)['candidate_chunks']
            try:
                decision = authorizer.authorize(message, candidates, tracer=tracer)
                result = {'decision':asdict(decision)}
            except Exception as exc:
                result = {'error_type':type(exc).__name__}
            records.append({'id':identifier, 'message':message, 'candidate_count':len(candidates),
                            'duration_seconds':round(time.monotonic()-started,3),
                            'trace':[step.model_dump() for step in tracer.get_steps()],
                            'token_usage':tracer.get_token_usage().model_dump(), **result})
            # Flush after each case so interruption preserves completed receipts.
            report = {'mode':'native_semantic_component_only', 'independent_evaluation':False,
                      'semantic_grade_assigned':False, 'model':settings.GEMINI_MODEL,
                      'code_sha':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                      'code_dirty':bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)),
                      'cases':records}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            sys.stdout.write(json.dumps({'id':identifier,'status':result.get('decision',{}).get('status'),
                                         'error_type':result.get('error_type')})+'\n')
            sys.stdout.flush()
    return 0 if not any('error_type' in record for record in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
