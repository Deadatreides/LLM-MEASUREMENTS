"""
Section 16-17 (SPLIT_TASK / artifact-level compute): a scoped test of
"local retry of one disputed claim" vs "full task/artifact retry",
reusing experiment 3's claim taxonomy and A1/CONTRACT infrastructure
directly (imported, not duplicated).

Scoped to ONE well-documented, genuinely variable claim rather than a
full parallel experiment: CODE_01/C7_EDGE_SPACES for qwen3-1.7b -- its
persistent, previously-documented blind spot (see experiment 3's
REPORT3.md and experiment 4's REPORT4.md). "Full retry" reuses experiment
3/4's existing A1 pool (no new generation); "local retry" is generated
here: a narrow prompt that shows the model only the specific disputed
claim and asks it to resolve just that, nothing else.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiment3", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiment3", "tasks"))

from llm_client import load_model, generate
from storage6 import save_raw, JsonlWriter, EXP_DIR
from tasks_pipeline import TASKS_BY_ID
from claims_def import CLAIMS_BY_TASK
import re

# claims_def.py's compiled correct_re/incorrect_re bake a tight "[^.]{0,50}"
# same-clause window into the pattern itself (tuned for compact A1 JSON
# fields). Local-retry answers are short, focused natural-language prose
# that routinely splits the relevant assertion across two sentences (e.g.
# "...should remain as-is. The function will not strip them.") and use
# synonyms ("remain") not in the original pattern -- reusing those
# compiled patterns as-is under-matches here (verified empirically: a
# genuinely correct real answer was misclassified OMITTED). A dedicated,
# negation-aware classifier per claim (same technique as experiment 3's
# C6_EDGE_Y fix) is used instead of trying to force-fit the original regex.

_ANCHOR_RE = re.compile(r"\b(space|punctuation)\w*", re.I)
_TARGET_RE = re.compile(r"\b(strip\w*|remov\w*|ignor\w*)\b", re.I)
_NEG_RE = re.compile(r"\b(not|never|n't)\b", re.I)
_PRESERVE_RE = re.compile(r"\b(preserv\w*|remain\w*|kept|keep\w*|includ\w*|as.is)\b", re.I)

LOCAL_CLASSIFIERS = {
    ("CODE_01", "C7_EDGE_SPACES"): "spaces_not_stripped",
}


def _classify_spaces_not_stripped(text):
    """CORRECT = the answer says spaces/punctuation are NOT stripped
    (matches reference); INCORRECT = says they ARE stripped/ignored.

    Two earlier versions of this function were each tested against real
    and synthetic examples and each failed differently: a character-window
    approach missed cases where "strip" preceded "spaces" in the sentence,
    and widening the window then caused an unrelated "ignoring case"
    earlier in the answer to falsely satisfy the space/punctuation anchor
    check by sheer character proximity. Fixed by restricting the
    strip/remove/ignore-vs-space/punctuation co-occurrence check to within
    the SAME SENTENCE (split on . ! ?), which matches how these clauses
    are actually written and stops cross-clause contamination -- verified
    against 7 held-out examples covering both word orders and an
    unrelated "ignoring case" distractor before use."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        if not _ANCHOR_RE.search(sent):
            continue
        for m in _TARGET_RE.finditer(sent):
            # a compound clause like "...ignoring case, but spaces should
            # remain as-is" contains an unrelated "ignoring" (about case,
            # not spaces) -- "case" follows immediately AFTER that
            # "ignoring", not before it. Skip strip/remove/ignore mentions
            # immediately FOLLOWED by "case" and keep scanning this same
            # sentence for the next candidate match.
            if re.search(r"^\s*case\b", sent[m.end(): m.end() + 15]):
                continue
            pre = sent[:m.start()]
            if _NEG_RE.search(pre):
                return "CORRECT", sent.strip()
            return "INCORRECT", sent.strip()
        if _PRESERVE_RE.search(sent):
            return "CORRECT", sent.strip()
    return "OMITTED", None


def _classify_local_answer(text, task_id, claim_id):
    key = (task_id, claim_id)
    fn_name = LOCAL_CLASSIFIERS.get(key)
    if fn_name == "spaces_not_stripped":
        return _classify_spaces_not_stripped(text)
    raise ValueError(f"no local-retry classifier registered for {key}")

RUNS_DIR = os.path.join(EXP_DIR, "runs")
MAX_TOKENS = 200  # a local, single-claim answer should be short
TOP_P = 1.0
TOP_K = 40

LOCAL_RETRY_TEMPLATE = (
    "You previously produced a specification for this task:\n\n{question}\n\n"
    "Focus ONLY on this specific aspect of the specification: \"{claim_text}\"\n\n"
    "State clearly, in one or two sentences, whether this is true or false for the "
    "function's required behavior, and briefly why. Do not restate the rest of the "
    "specification, do not write code."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--claim_id", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--seeds", required=True)
    args = ap.parse_args()

    task = TASKS_BY_ID[args.task_id]
    claim_def = next(c for c in CLAIMS_BY_TASK[args.task_id] if c["claim_id"] == args.claim_id)
    seeds = [int(s) for s in args.seeds.split(",")]

    llm, model_entry, load_time = load_model(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2)}))

    index_path = os.path.join(RUNS_DIR, f"claim_local_retry.{args.model_id}.{args.task_id}.{args.claim_id}.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    prompt_text = LOCAL_RETRY_TEMPLATE.format(question=task["question"], claim_text=claim_def["claim_text"])

    n_done = 0
    for seed in seeds:
        run_id = f"{args.model_id}--{args.task_id}--{args.claim_id}--T{args.temperature}--seed{seed}--LOCALRETRY"
        if writer.has(run_id):
            continue
        gen = generate(llm, args.model_id, model_entry, prompt_text, temperature=args.temperature,
                        top_p=TOP_P, top_k=TOP_K, seed=seed, max_tokens=MAX_TOKENS)

        text = (gen["raw_text"] or "").lower()
        cls, evidence = _classify_local_answer(text, args.task_id, args.claim_id)

        raw_path = save_raw(run_id, gen["raw_text"], {"run_id": run_id, "rendered_prompt": gen["rendered_prompt"]})
        row = {
            "run_id": run_id, "task_id": args.task_id, "claim_id": args.claim_id, "model": args.model_id,
            "temperature": args.temperature, "seed": seed, "prompt_id": "LOCAL_RETRY",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classification": cls, "evidence": evidence,
            "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
            "generation_time_sec": round(gen["generation_time_sec"], 4),
            "generation_failed": gen["generation_failed"], "raw_path": raw_path,
        }
        writer.write(row, key=run_id)
        n_done += 1

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
