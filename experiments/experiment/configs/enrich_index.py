"""
Post-processing enrichment pass: adds an OUTPUT_TEXT_HASH (sha256 of the raw
generated text, read from RAW) to each compact-index row, so text-diversity
metrics (NUMBER_OF_UNIQUE_OUTPUTS) can be computed from a single file instead
of re-opening every RAW result.txt each time.

Does not modify runs/index.<model_id>.jsonl or runs/index.jsonl (those stay
exactly as run_batch.py produced them -- the authoritative, unmodified
record). Writes a derived copy: runs/index_enriched.jsonl.
"""
import hashlib
import json
import os

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment"


def main():
    index_path = os.path.join(EXP_DIR, "runs", "index.jsonl")
    out_path = os.path.join(EXP_DIR, "runs", "index_enriched.jsonl")

    rows = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_missing_raw = 0
    for row in rows:
        raw_txt_path = os.path.join(EXP_DIR, "runs", row["raw_path"], "result.txt")
        if os.path.exists(raw_txt_path):
            with open(raw_txt_path, encoding="utf-8") as rf:
                text = rf.read()
            row["output_text_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        else:
            row["output_text_hash"] = None
            n_missing_raw += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Enriched {len(rows)} rows -> {out_path} ({n_missing_raw} missing RAW)")


if __name__ == "__main__":
    main()
