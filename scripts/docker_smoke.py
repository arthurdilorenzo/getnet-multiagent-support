"""Real container verification, using synthetic identities and ephemeral secrets only."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import date


def command(*args, capture=False, env=None):
    result = subprocess.run(args, check=True, text=True, capture_output=capture, env=env)
    return result.stdout.strip() if capture else None


def validate_historical_chat(response):
    """This scenario cannot prove today's deposit from a static old snapshot."""
    assert 'get_sales_settlement' in response.get('tools_called', [])
    assert response.get('abstained') is True and response.get('answer_status') == 'partial'
    facts = {item.get('fact_path'): item.get('value') for item in response.get('evidence', [])
             if item.get('kind') == 'tool_fact' and item.get('tool_name') == 'get_sales_settlement'}
    assert facts.get('data_scope') == 'synthetic_historical_snapshot'
    snapshot_date = facts.get('sales_data.date')
    assert isinstance(snapshot_date, str)
    assert date.fromisoformat(snapshot_date).isoformat() == snapshot_date
    text = response.get('response', '')
    assert snapshot_date in text and 'históric' in text.lower()
    assert 'não confirma depósito' in text.lower()
    pending = [row for row in response.get('request_coverage', [])
               if row.get('state') == 'unresolved' and row.get('text', '').strip()]
    assert pending and all(row['text'].strip() in text for row in pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="getnet-v4-smoke:local")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code_sha = command("git", "-C", str(root), "rev-parse", "HEAD", capture=True)
    code_dirty = bool(command("git", "-C", str(root), "status", "--porcelain", capture=True))
    if not args.skip_build:
        command("docker", "build", "--build-arg", f"VCS_REF={code_sha}", "--build-arg",
                f"SOURCE_DIRTY={str(code_dirty).lower()}", "-t", args.image, str(root))
    image_id = command("docker", "image", "inspect", "--format", "{{.Id}}", args.image, capture=True)
    image_revision = command("docker", "image", "inspect", "--format",
                             '{{index .Config.Labels "org.opencontainers.image.revision"}}', args.image, capture=True)
    image_dirty = command("docker", "image", "inspect", "--format",
                          '{{index .Config.Labels "com.getnet.source.dirty"}}', args.image, capture=True)
    if args.skip_build and (image_revision != code_sha or code_dirty or image_dirty != 'false'):
        raise RuntimeError("A imagem reutilizada não comprova o checkout limpo atual; reconstrua sem --skip-build.")
    identifier = "getnet-smoke-" + uuid.uuid4().hex[:12]
    container, volume = identifier, identifier + "-state"
    environment = {**os.environ, "GEMINI_API_KEY": "", "AUTH_API_KEY": secrets.token_urlsafe(48),
                   "JWT_SECRET": secrets.token_urlsafe(48), "RATE_LIMIT_PER_MINUTE": "200"}
    checks = []
    started = time.monotonic()
    try:
        command("docker", "volume", "create", volume, capture=True)
        command("docker", "run", "-d", "--name", container, "--read-only", "--tmpfs", "/tmp",
                "--security-opt", "no-new-privileges:true", "-p", "127.0.0.1::8000",
                "--mount", f"type=volume,source={volume},target=/app/state",
                "--env", "AUTH_API_KEY", "--env", "JWT_SECRET", "--env", "GEMINI_API_KEY",
                "--env", "RATE_LIMIT_PER_MINUTE", args.image, capture=True, env=environment)
        port_map = json.loads(command("docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", container, capture=True))
        base = "http://127.0.0.1:" + port_map["8000/tcp"][0]["HostPort"]

        def request(path, method="GET", body=None, headers=None):
            headers = {"Content-Type": "application/json", **(headers or {})}
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                return exc.code, json.loads(exc.read())

        def healthy():
            for _ in range(60):
                try:
                    code, health = request("/api/health")
                    if code == 200 and health["status"] == "healthy":
                        return health
                except (OSError, ValueError):
                    pass
                time.sleep(.5)
            raise RuntimeError("Container did not become healthy")

        health = healthy()
        assert health["gemini_configured"] is False
        checks.append("health_offline")
        user = command("docker", "inspect", "--format", "{{.Config.User}}", container, capture=True)
        assert user in {"appuser", "10001", "10001:10001"}
        checks.append("non_root_read_only")
        body = {"user_id": "cliente1988", "session_id": "smoke-persist", "message": "Quando cai o dinheiro das vendas de ontem?"}
        assert request("/api/chat", "POST", body)[0] == 401
        assert request("/api/hitl/tickets")[0] == 401
        checks.append("anonymous_rejected")
        code, issued = request("/api/auth/token", "POST", {"user_id": "cliente1988"},
                               {"X-API-Key": environment["AUTH_API_KEY"]})
        assert code == 200
        headers = {"Authorization": "Bearer " + issued["access_token"]}
        code, response = request("/api/chat", "POST", body, headers)
        assert code == 200
        validate_historical_chat(response)
        checks.append("authenticated_chat")
        assert request("/api/hitl/tickets", headers=headers)[0] == 403
        assert request("/api/sessions/smoke-persist?user_id=cliente2045", headers=headers)[0] == 403
        checks.append("tenant_and_role_isolation")
        code, before = request("/api/sessions/smoke-persist?user_id=cliente1988", headers=headers)
        assert code == 200 and len(before["recent_messages"]) >= 2
        command("docker", "restart", container, capture=True)
        port_map = json.loads(command("docker", "inspect", "--format", "{{json .NetworkSettings.Ports}}", container, capture=True))
        base = "http://127.0.0.1:" + port_map["8000/tcp"][0]["HostPort"]
        healthy()
        code, after = request("/api/sessions/smoke-persist?user_id=cliente1988", headers=headers)
        assert code == 200 and after["recent_messages"] == before["recent_messages"]
        checks.append("restart_persistence")
        summary = {"status": "passed", "checks": checks, "image": args.image,
                   "image_id": image_id, "code_sha": code_sha, "code_dirty": code_dirty,
                   "image_revision": image_revision, "built_during_run": not args.skip_build,
                   "image_source_dirty": image_dirty,
                   "duration_seconds": round(time.monotonic() - started, 3)}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
    except Exception:
        diagnostic = subprocess.run(["docker", "logs", "--tail", "50", container], capture_output=True, text=True)
        safe_log = diagnostic.stdout + diagnostic.stderr
        for name in ("AUTH_API_KEY", "JWT_SECRET"):
            safe_log = safe_log.replace(environment[name], "[REDACTED]")
        import sys
        sys.stderr.write(safe_log)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        subprocess.run(["docker", "volume", "rm", volume], capture_output=True)


if __name__ == "__main__":
    main()
