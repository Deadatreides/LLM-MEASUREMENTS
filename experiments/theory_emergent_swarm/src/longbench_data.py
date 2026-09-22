"""longbench_data.py — LongBench v2, Multi-Document QA / hard / short.
Loader, deterministic chunker, prompts, seams.

The gold letter is used ONLY for scoring: never shown to a model, never
used to build chunks, evidence or candidates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "lb2_mdqa_hard_short.json"
WORDS_PER_CHUNK = 1000          # ~1400 tokens, fits n_ctx=2048 with room for the question
LETTERS = ("A", "B", "C", "D")

_LETTER_RE = re.compile(r"\b(?:ANSWER|OTVET)\s*[:=]?\s*\(?([ABCD])\)?", re.IGNORECASE)
_BARE_RE = re.compile(r"^\s*\(?([ABCD])\)?[\.\):]?\s*$", re.MULTILINE)
_NONE_RE = re.compile(r"\bNONE\b", re.IGNORECASE)


def load() -> list:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    out = []
    for i, x in enumerate(raw):
        out.append({
            "task_id": f"LB_{i:03d}", "qid": x["_id"], "question": x["question"],
            "choices": {L: x[f"choice_{L}"] for L in LETTERS},
            "gold": x["answer"].strip().upper()[:1],
            "context": x["context"], "sub_domain": x["sub_domain"],
        })
    return out


def chunks_of(task: dict) -> list:
    w = task["context"].split()
    return [" ".join(w[i:i + WORDS_PER_CHUNK]) for i in range(0, len(w), WORDS_PER_CHUNK)]


def _choices_block(task: dict) -> str:
    return "\n".join(f"{L}) {task['choices'][L]}" for L in LETTERS)


MAX_CHUNK_CHARS = 5600      # hard cap: 5600 chars ~1400 tok, leaves room inside n_ctx=2048


def prompt_evidence(task: dict, chunk: str) -> str:
    """Discipline-level work: pull evidence, never decide the answer."""
    chunk = chunk[:MAX_CHUNK_CHARS]
    return (f"Text fragment:\n{chunk}\n\n"
            f"Question: {task['question']}\n\n"
            "Copy the sentences from the fragment above that help answer the question. "
            "Copy them EXACTLY as written, at most 2 sentences.\n"
            "If the fragment contains nothing relevant, reply with the single word NONE.")


def prompt_answer(task: dict, evidence: str) -> str:
    return (f"Notes:\n{evidence}\n\n"
            f"Question: {task['question']}\n{_choices_block(task)}\n\n"
            "Answer with a single letter. Final line exactly:\nANSWER: <A|B|C|D>")


def prompt_truncated(task: dict, chunk: str) -> str:
    """Baseline: no decomposition -- the model sees only what fits at n_ctx."""
    chunk = chunk[:MAX_CHUNK_CHARS]
    return (f"Text:\n{chunk}\n\n"
            f"Question: {task['question']}\n{_choices_block(task)}\n\n"
            "Answer with a single letter. Final line exactly:\nANSWER: <A|B|C|D>")


# ------------------------------- seams -------------------------------------

def extract_letter(text: str):
    hits = _LETTER_RE.findall(text or "")
    if hits:
        return hits[-1].upper()
    bare = _BARE_RE.findall(text or "")
    if bare:
        return bare[-1].upper()
    return None


def extract_evidence(text: str, chunk: str) -> str:
    """ВДП: keep only sentences that VERIFIABLY occur in the chunk.
    A quote the model invented is dropped -- a deterministic hallucination
    filter, not a judgement of relevance."""
    t = (text or "").strip()
    if not t or _NONE_RE.search(t):
        return ""
    norm_chunk = " ".join(chunk.split()).lower()
    kept = []
    for line in re.split(r"(?<=[.!?])\s+|\n+", t):
        s = " ".join(line.split()).strip(' "\'`*-')
        if len(s) < 25:
            continue
        if " ".join(s.split()).lower() in norm_chunk:
            kept.append(s)
    return "\n".join(kept[:2])


MIN_SPAN_WORDS = 8


def extract_evidence_v2(text: str, chunk: str) -> str:
    """ВДП v2 — SPAN-level grounding instead of sentence-level.

    v1 compared whole sentences verbatim and rejected 362/870 outputs; a
    hand-checked sample showed ~60% of those rejections were FALSE: the
    model had quoted the chunk correctly but added a lead-in ("The press
    release states:"), re-bulleted the text, or merged sentences. That is
    a formatting difference, not fabrication, and v1 discarded real
    grounded evidence -- a defect in my filter, not in the models.

    v2 keeps the longest contiguous word span of the output that occurs
    verbatim in the chunk, and keeps it only if it is >= MIN_SPAN_WORDS.
    Model-invented additions are still dropped, so this stays a
    hallucination filter -- it just measures grounding at the right
    granularity.
    """
    t = (text or "").strip()
    if not t or _NONE_RE.search(t):
        return ""
    norm_chunk = " ".join(chunk.split()).lower()
    words = " ".join(t.split()).split()
    best_i = best_len = 0
    i = 0
    while i < len(words):
        if len(words) - i < MIN_SPAN_WORDS:
            break
        lo, hi = MIN_SPAN_WORDS, len(words) - i
        if " ".join(words[i:i + MIN_SPAN_WORDS]).lower() not in norm_chunk:
            i += 1
            continue
        while lo < hi:                       # binary search for the longest match at i
            mid = (lo + hi + 1) // 2
            if " ".join(words[i:i + mid]).lower() in norm_chunk:
                lo = mid
            else:
                hi = mid - 1
        if lo > best_len:
            best_i, best_len = i, lo
        i += max(1, lo)
    return " ".join(words[best_i:best_i + best_len]) if best_len >= MIN_SPAN_WORDS else ""


def evidence_supports(evidence: str, task: dict, letter: str) -> bool:
    """МДП across the evidence->answer seam: does the assembled evidence
    share any content word with the chosen option? Checks the INTERFACE,
    it does not re-decide the question."""
    if not letter or not evidence:
        return False
    opt = set(re.findall(r"[a-z]{5,}", task["choices"][letter].lower()))
    ev = set(re.findall(r"[a-z]{5,}", evidence.lower()))
    return bool(opt & ev)
