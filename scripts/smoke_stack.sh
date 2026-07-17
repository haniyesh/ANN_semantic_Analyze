#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end production smoke test. Requires a populated .env, Docker daemon,
# and Docker Compose v2. Set KEEP_STACK=1 to leave successful containers up.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v docker >/dev/null || { echo "ERROR: docker is not installed"; exit 1; }
docker compose version >/dev/null 2>&1 || {
  echo "ERROR: Docker Compose v2 is unavailable (expected: docker compose)"
  exit 1
}
docker info >/dev/null 2>&1 || {
  echo "ERROR: Docker daemon is unavailable; enable Docker Desktop WSL integration"
  exit 1
}
[[ -f .env ]] || { echo "ERROR: create .env from .env.example first"; exit 1; }

cleanup() {
  if [[ "${KEEP_STACK:-0}" != "1" ]]; then
    docker compose down --remove-orphans
  fi
}
trap cleanup EXIT

echo "[1/7] Validating Compose configuration"
docker compose config --quiet

echo "[2/7] Building production images"
docker compose build

echo "[3/7] Starting postgres, API, bot, and dashboard"
docker compose up -d

echo "[4/7] Waiting for declared container health checks"
deadline=$((SECONDS + 240))
unhealthy="not-started"
while (( SECONDS < deadline )); do
  unhealthy="$(docker compose ps --format json | python -c '
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    rows = []
else:
    try:
        data = json.loads(raw)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
bad = [r.get("Service", "?") for r in rows
       if r.get("Health") not in ("", "healthy") or r.get("State") != "running"]
print(",".join(bad) if len(rows) == 4 else "missing-service")
')"
  [[ -z "$unhealthy" ]] && break
  sleep 5
done
[[ -z "$unhealthy" ]] || {
  docker compose ps
  docker compose logs --tail=100
  echo "ERROR: unhealthy services: $unhealthy"
  exit 1
}

echo "[5/7] Checking PostgreSQL, API, dashboard, and production artifacts"
docker compose exec -T postgres sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
curl --fail --silent --show-error http://localhost:8000/health | python -m json.tool >/dev/null
curl --fail --silent --show-error http://localhost:5173/ >/dev/null
docker compose exec -T api python -c '
import hashlib, json
from pathlib import Path
m = json.loads(Path("model_manifest.json").read_text())
for section in ("model", "scaler"):
    p = Path(m[section]["path"])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == m[section]["sha256"]
print("artifact checksums: ok")
'

echo "[6/7] Checking Qdrant RAG readiness"
curl --fail --silent --show-error http://localhost:8000/health/qdrant | python -m json.tool

echo "[7/7] Checking bot heartbeat and model readiness"
sleep 35
curl --fail --silent --show-error http://localhost:8000/health | python -c '
import json, sys
h = json.load(sys.stdin)
assert h["model"]["ready"], h
assert h["bot_status"] == "ok", h
print("model: ready; bot heartbeat: ok; full-stack smoke test: PASS")
'
