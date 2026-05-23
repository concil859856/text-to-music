"""Non-GPU smoke test for the Vocence /studio/ops integration on text-to-music.

Verifies the FastAPI app instantiates without loading the ACE-Step pipeline,
/healthz + /metrics respond with the standardized schema, the inflight
cap is respected, and bearer auth gates these endpoints when configured.

Run:  python3 test_ops_smoke.py
"""
from __future__ import annotations

import os
import sys
import types


def _stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# Stub the heavy deps so importing api.py doesn't need CUDA / model weights.
if "acestep" not in sys.modules:
    _stub_module("acestep")
    _stub_module("acestep.pipeline_ace_step", ACEStepPipeline=type("ACEStepPipeline", (), {"loaded": False}))

os.environ["MUSIC_API_KEY"] = "test-music-key"
os.environ["MUSIC_CAP"] = "1"
os.environ["PORT"] = "9115"

import api  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


def main_test() -> int:
    print(f"MUSIC_API_KEY set = {bool(api.MUSIC_API_KEY)}    (expected: True)")
    print(f"MUSIC_CAP = {api.MUSIC_CAP}    (expected: 1)")
    print(f"_inflight.cap = {api._inflight.cap}    (expected: 1)")

    client = TestClient(api.app, raise_server_exceptions=False)

    failures = 0

    # /healthz without bearer -> 401
    r = client.get("/healthz")
    if r.status_code != 401:
        print(f"FAIL: /healthz without bearer should be 401, got {r.status_code}")
        failures += 1
    else:
        print("PASS: /healthz without bearer -> 401")

    # /healthz with bearer -> 200
    r = client.get("/healthz", headers={"Authorization": "Bearer test-music-key"})
    if r.status_code != 200:
        print(f"FAIL: /healthz with bearer should be 200, got {r.status_code}")
        failures += 1
    else:
        body = r.json()
        required = {"status", "service", "model_id", "sample_rate", "inflight", "cap", "dev_stub"}
        missing = required - set(body)
        if missing:
            print(f"FAIL: /healthz missing fields: {missing}")
            failures += 1
        elif body["service"] != "music":
            print(f"FAIL: /healthz service={body['service']!r}, expected 'music'")
            failures += 1
        elif body["cap"] != 1:
            print(f"FAIL: /healthz cap={body['cap']!r}, expected 1")
            failures += 1
        else:
            print(f"PASS: /healthz with bearer -> 200 service={body['service']} cap={body['cap']}")

    # /metrics with bearer -> 200
    r = client.get("/metrics", headers={"Authorization": "Bearer test-music-key"})
    if r.status_code != 200:
        print(f"FAIL: /metrics should be 200, got {r.status_code}")
        failures += 1
    else:
        body = r.json()
        required = {"uptime_seconds", "requests_total", "requests_ok", "requests_err",
                    "duration_ms_sum", "duration_ms_count", "duration_ms_p50",
                    "duration_ms_p95", "duration_ms_p99", "inflight", "cap", "service"}
        missing = required - set(body)
        if missing:
            print(f"FAIL: /metrics missing fields: {missing}")
            failures += 1
        else:
            print(f"PASS: /metrics with bearer -> 200 requests_total={body['requests_total']}")

    # /metrics with wrong bearer -> 401
    r = client.get("/metrics", headers={"Authorization": "Bearer wrong-key"})
    if r.status_code != 401:
        print(f"FAIL: /metrics with wrong bearer should be 401, got {r.status_code}")
        failures += 1
    else:
        print("PASS: /metrics with wrong bearer -> 401")

    # Legacy /health open + 200
    r = client.get("/health")
    if r.status_code != 200:
        print(f"FAIL: legacy /health should be 200, got {r.status_code}")
        failures += 1
    else:
        body = r.json()
        if "status" not in body:
            print(f"FAIL: legacy /health body changed: {body}")
            failures += 1
        else:
            print(f"PASS: legacy /health -> 200")

    # /healthz repeated -> all 200 (middleware skips it)
    for _ in range(5):
        r = client.get("/healthz", headers={"Authorization": "Bearer test-music-key"})
        if r.status_code != 200:
            print(f"FAIL: /healthz loop iteration returned {r.status_code}")
            failures += 1
            break
    else:
        print("PASS: /healthz 5x consecutive -> all 200 (middleware skips it)")

    # All expected routes registered
    expected_routes = {"/generate/text2music", "/generate/audio2audio", "/generate/retake",
                       "/generate/repaint", "/generate/edit", "/generate/extend",
                       "/healthz", "/metrics", "/health"}
    actual = {r.path for r in api.app.routes if hasattr(r, "path")}
    missing = expected_routes - actual
    if missing:
        print(f"FAIL: missing routes: {missing}")
        failures += 1
    else:
        print(f"PASS: all expected routes registered ({len(expected_routes)} checked)")

    print()
    if failures == 0:
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print(f"{failures} TEST(S) FAILED")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main_test())
