"""Client for a local llama-server. One model, blocking calls, full accounting.

Every call is logged with token counts and wall time -- the pilot exists to
measure cost, so nothing is estimated that can be counted.
"""
import json, time, subprocess, os

HOST = os.environ.get("KAN_LLM_HOST", "http://127.0.0.1:8080")

CALL_LOG = []


def _post(path, payload, timeout=1800):
    """POST via curl: python's ssl/urllib is not needed for localhost, but curl
    keeps one code path with the rest of the harness."""
    cmd = ["curl", "-s", "--max-time", str(timeout),
           "-H", "Content-Type: application/json",
           "-X", "POST", "--data-binary", "@-", HOST + path]
    p = subprocess.run(cmd, input=json.dumps(payload), capture_output=True,
                       text=True, encoding="utf-8")
    if not p.stdout:
        raise RuntimeError(f"empty response from {path}: {p.stderr[:300]}")
    return json.loads(p.stdout)


def health(timeout=5):
    p = subprocess.run(["curl", "-s", "--max-time", str(timeout), HOST + "/health"],
                       capture_output=True, text=True, encoding="utf-8")
    return p.returncode == 0 and "ok" in (p.stdout or "").lower()


def chat(system, user, max_tokens=1600, temperature=0.0, reasoning="low", tag=""):
    """One completion. Returns (text, meta)."""
    payload = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": False,
        "cache_prompt": True,
        "chat_template_kwargs": {"reasoning_effort": reasoning},
    }
    t0 = time.time()
    r = _post("/v1/chat/completions", payload)
    dt = time.time() - t0

    if "choices" not in r:
        raise RuntimeError(f"bad response: {str(r)[:400]}")
    msg = r["choices"][0]["message"]
    text = msg.get("content") or ""
    reasoning_text = msg.get("reasoning_content") or ""
    usage = r.get("usage", {})

    meta = dict(tag=tag, seconds=round(dt, 2),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                reasoning_chars=len(reasoning_text),
                finish=r["choices"][0].get("finish_reason"))
    CALL_LOG.append(meta)
    return text, meta


def totals():
    return dict(
        calls=len(CALL_LOG),
        seconds=round(sum(c["seconds"] for c in CALL_LOG), 1),
        prompt_tokens=sum(c["prompt_tokens"] for c in CALL_LOG),
        completion_tokens=sum(c["completion_tokens"] for c in CALL_LOG),
    )


def reset():
    CALL_LOG.clear()
