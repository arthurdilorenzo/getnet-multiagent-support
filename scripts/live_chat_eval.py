"""Bounded opt-in native HTTP requests from label-free synthetic cases.

Input: {"cases": [{"id": "identifier", "message": "synthetic request"}]}.
Outputs observations only; semantic adjudication belongs to an independent
reviewer. A successful HTTP response is never assigned a semantic pass here.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import re
import secrets
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
    if not 1 <= args.limit <= 12:
        raise SystemExit('Choose 1..12 requests explicitly')
    cases = json.loads(args.input.read_text())['cases'][:args.limit]
    if not cases or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', str(c['id'])) for c in cases):
        raise SystemExit('Nonempty cases and safe unique identifiers are required')
    if len({c['id'] for c in cases}) != len(cases):
        raise SystemExit('Duplicate identifiers')
    root = Path(__file__).resolve().parents[1]

    def revision():
        return {'code_sha': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                'code_dirty': bool(subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True))}

    initial_revision = revision()
    with tempfile.TemporaryDirectory(prefix='getnet-native-chat-') as directory:
        os.environ.update(SQLITE_DB_PATH=str(Path(directory) / 'state.db'),
                          VECTOR_INDEX_PATH=str(Path(directory) / 'vectors.json'),
                          JWT_SECRET=secrets.token_urlsafe(48), AUTH_API_KEY=secrets.token_urlsafe(48))
        from app.config import settings
        if not settings.GEMINI_API_KEY:
            raise SystemExit('GEMINI_API_KEY is not configured')
        from app.main import app
        from app.security.auth import AuthManager
        from app.security.pii import PIIRedactor
        from fastapi.testclient import TestClient
        logging.getLogger('getnet.multiagent').disabled = True
        records = []
        with TestClient(app) as client:
            for case in cases:
                if revision() != initial_revision:
                    raise SystemExit('Candidate revision changed during execution')
                user = 'cliente1988'  # Known synthetic fixture, never a real account.
                message = PIIRedactor.mask_text(case['message'])
                started = time.monotonic()
                try:
                    response = client.post('/api/chat',
                        headers={'Authorization': 'Bearer ' + AuthManager.create_jwt_token(user)},
                        json={'user_id': user, 'session_id': 'native-eval-' + case['id'], 'message': message})
                    record = {'http_status': response.status_code, 'observation': response.json()}
                except Exception as exc:
                    record = {'error_type': type(exc).__name__}
                records.append({'id': case['id'], 'message': message,
                                'duration_seconds': round(time.monotonic() - started, 3), **record})
                report = {'mode': 'native_semantic_inprocess_http', 'synthetic_data': True,
                          'model': settings.GEMINI_MODEL, **initial_revision,
                          'independent_evaluation': False, 'semantic_grade_assigned': False,
                          'retry_policy': 'no replacement of failed cases', 'cases': records}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                sys.stdout.write(json.dumps({'id': case['id'], 'http_status': record.get('http_status'),
                                            'error_type': record.get('error_type')}) + '\n')
                sys.stdout.flush()
    return 0 if all(c.get('http_status') == 200 for c in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
