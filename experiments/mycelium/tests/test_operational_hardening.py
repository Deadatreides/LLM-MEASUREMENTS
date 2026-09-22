"""
tests/test_operational_hardening.py - Operational behavioral tests for hardening.
"""
import os
import sys
import tempfile
import json
import logging
import pytest
import numpy as np

sys.path.insert(0, ".")

# 1. Import smoke check tests
def test_import_smoke_checks():
    """Verify that all core and bridge packages are fully importable."""
    modules = [
        "aiohttp", "faiss", "networkx", "numpy", "dotenv", "yaml",
        "schedule", "sentence_transformers", "sympy", "tqdm", "pytest",
        "fastapi", "uvicorn"
    ]
    for mod in modules:
        try:
            __import__(mod)
        except ImportError as e:
            pytest.fail(f"Failed to import core dependency '{mod}': {e}")

# 2. UTF-8 logging behavior test
def test_utf8_logging():
    """Verify that Cyrillic and Greek math characters write to file cleanly without UnicodeEncodeError."""
    from core.logging_setup import setup_logging
    
    # Run setup logging
    setup_logging()
    
    test_logger = logging.getLogger("main")
    unicode_msg = "Mycelium Hardening | λ_L ≈ 0 | Edge of Chaos | Пригожин Φ | → Δ"
    
    try:
        test_logger.info(unicode_msg)
    except Exception as e:
        pytest.fail(f"Logger failed to handle UTF-8 Unicode characters on Windows: {e}")
        
    # Read the log file and verify content
    log_path = "logs/main.log"
    assert os.path.exists(log_path), "main.log file was not created."
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "λ_L ≈ 0" in content, "Unicode message was not found or was corrupted in log file."
        assert "Пригожин Φ" in content, "Cyrillic message was not found in log file."

# 3. Telemetry JSONL recording test
def test_telemetry_jsonl():
    """Verify telemetry JSONL file creation, appending, and valid JSON serializing."""
    from core.telemetry import log_telemetry
    
    telemetry_path = "logs/telemetry.jsonl"
    if os.path.exists(telemetry_path):
        try:
            os.remove(telemetry_path)
        except OSError:
            pass
            
    payload = {
        'step': 99,
        'task_type': 'code',
        'lambda': -0.0123,
        'phi': 0.824,
        'entropy': 1.025,
        'delta_h': 0.124,
        'e_total': 0.742,
        'regime': 'EDGE',
        'k_act': 6,
        'retrieval_count': 3,
        'memory_nodes': 12,
        'memory_edges': 45,
        'alive_agents': 24,
        'provider_failures': 0,
        'tokens_used': 15000,
        'latency': 5.25
    }
    
    # Try writing telemetry
    try:
        log_telemetry(payload)
    except Exception as e:
        pytest.fail(f"Telemetry logging crashed: {e}")
        
    assert os.path.exists(telemetry_path), "telemetry.jsonl was not created."
    
    # Read back and parse JSON
    with open(telemetry_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed['step'] == 99
        assert parsed['task_type'] == 'code'
        assert abs(parsed['lambda'] - (-0.0123)) < 1e-5
        assert parsed['regime'] == 'EDGE'

# 4. Windows Path Fallback Resilience test
def test_windows_path_fallback():
    """Verify that unwriteable paths gracefully fall back to local data/ folder."""
    from core.memory import _resolve_resilient_path
    
    # Emulate an invalid/unwriteable Windows path (e.g. non-existent drive Z:\)
    invalid_path = "Z:\\nonexistent_drive\\subfolder\\my_traces.db"
    
    resolved = _resolve_resilient_path(invalid_path, "traces.db")
    
    # Should fallback to local data/traces.db
    assert "data" in resolved, f"Fallback failed to redirect to data/. Path remained: {resolved}"
    assert "traces.db" in resolved
    assert os.path.exists("data"), "Local fallback folder data/ was not created."

# 5. OpenCode Bridge API client test
def test_opencode_bridge_api():
    """Verify bridge endpoints status and uvicorn routing using FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from opencode_bridge import app
    
    client = TestClient(app)
    
    # Test status endpoint
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "step" in data
    assert "I" in data
    assert "lambda_L" in data
    assert "regime" in data
    assert "alive_agents" in data

# 6. Provider Cooldown and Cooldown Triggering test
def test_provider_cooldown_triggering():
    """Verify that a fatal error on call immediately puts the provider on 60s cooldown."""
    from core.bandit import SwarmBandit
    from orchestrator import MyceliumOrchestrator
    
    orch = MyceliumOrchestrator()
    bandit = orch.bandit
    
    # Check initial provider state
    provider = "groq"
    assert bandit.provider_available(provider), "Groq provider should be initially active."
    
    # Mocking a fatal failure outcome (representing a 401/402/insufficient balance)
    # This should be mapped inside _record_provider_results with fatal=True
    fatal_result = {
        'id': 'groq_llama33_70b',
        'ok': False,
        'fatal': True,
        'error': 'http_401',
        'latency': 0.05
    }
    
    # Record fatal provider result
    orch._record_provider_results([fatal_result])
    
    # Confirm immediate cooldown
    assert not bandit.provider_available(provider), "Fatal error failed to trigger immediate provider cooldown!"
    assert bandit.provider_health[provider].consecutive_failures >= 3
