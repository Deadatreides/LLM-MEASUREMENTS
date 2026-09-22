"""
Re-applies the current verification.py logic to every row's stored RAW
text and rewrites the verification/result blocks in place.

Why this exists: verification is a pure function of (domain, task, raw
text) and RAW is never modified once written, so re-verifying is fully
reproducible and costs no GPU time. This lets us fix a labeling bug in
verification.py (an early version mislabeled "model produced no usable
text" as TECHNICAL_FAILURE instead of MECHANICAL_FAIL/FORMAT_ERROR) and
apply the corrected logic uniformly to rows written before and after the
fix, without re-running any generation.

Only touches result/verification fields derived from raw_text; leaves
model/prompt/condition/performance/generation fields untouched. Rows whose
generation itself failed (result_status == "generation_failed") are left
as-is, since there is no RAW text to re-verify and that distinction
(generation error vs. solution error) must not be blurred.
"""
import json
import os

from verification import verify
from extraction import text_characteristics

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    tasks_by_id = {t["task_id"]: t for t in load_jsonl(os.path.join(EXP_DIR, "tasks", "tasks.jsonl"))}

    for fname in ["index.jsonl"] + sorted(
        f for f in os.listdir(os.path.join(EXP_DIR, "runs")) if f.startswith("index.") and f.endswith(".jsonl") and f != "index.jsonl" and f != "index_enriched.jsonl"
    ):
        path = os.path.join(EXP_DIR, "runs", fname)
        if not os.path.exists(path):
            continue
        rows = load_jsonl(path)
        n_changed = 0
        for row in rows:
            if row["result_status"] == "generation_failed":
                continue
            raw_txt_path = os.path.join(EXP_DIR, "runs", row["raw_path"], "result.txt")
            if not os.path.exists(raw_txt_path):
                continue
            with open(raw_txt_path, encoding="utf-8") as rf:
                raw_text = rf.read()
            task = tasks_by_id[row["task_id"]]
            old_status = row["verification"]["status"]
            verification = verify(task["domain"], task, raw_text)
            chars = text_characteristics(raw_text)
            has_final_conclusion = (
                verification["has_code"] if task["domain"] == "code" else verification["has_final_answer"]
            )
            row["result"]["length_chars"] = chars["length_chars"]
            row["result"]["has_final_answer"] = verification["has_final_answer"]
            row["result"]["extracted_final_answer"] = verification["extracted_final_answer"]
            row["result"]["answer_format"] = verification["answer_format"]
            row["result"]["has_code"] = verification["has_code"]
            row["result"]["code_length"] = verification["code_length"]
            row["result"]["has_multiple_candidates"] = verification["has_multiple_candidates"]
            row["result"]["has_explicit_uncertainty"] = chars["has_explicit_uncertainty"]
            row["result"]["has_self_correction_language"] = chars["has_self_correction_language"]
            row["result"]["has_contradiction"] = chars["has_contradiction"]
            row["result"]["has_final_conclusion"] = has_final_conclusion
            row["verification"]["status"] = verification["verification_status"]
            row["verification"]["error_class"] = verification["error_class"]
            row["verification"]["error_signature"] = verification["error_signature"]
            row["verification"]["syntax_valid"] = verification["syntax_valid"]
            row["verification"]["execution_status"] = verification["execution_status"]
            row["verification"]["tests_passed"] = verification["tests_passed"]
            row["verification"]["tests_failed"] = verification["tests_failed"]
            row["verification"]["exception_type"] = verification["exception_type"]
            if old_status != verification["verification_status"]:
                n_changed += 1
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{fname}: {len(rows)} rows, {n_changed} verification_status changed")


if __name__ == "__main__":
    main()
