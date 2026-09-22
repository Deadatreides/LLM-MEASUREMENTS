"""Two-level storage (RAW + compact index) shared by baseline and pipeline
runners, mirroring experiment 1's approach."""
import json
import os

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment3"
RAW_DIR = os.path.join(EXP_DIR, "runs", "raw")


def save_raw(artifact_id, raw_text, meta):
    d = os.path.join(RAW_DIR, artifact_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "result.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text or "")
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return f"raw/{artifact_id}/"


class JsonlWriter:
    """key_field must match exactly what callers pass to has()/write(key=...)
    for this file -- e.g. "run_id" for baseline.jsonl/pipeline_runs.jsonl,
    "artifact_id" for artifacts.jsonl, "seam_id" for seams.jsonl. Auto-
    detecting the key from whichever field happens to be present is wrong
    when a row carries more than one id-like field (baseline rows have
    both run_id and artifact_id) -- that previously caused resumability to
    silently key off the wrong field and re-generate (duplicate) rows on
    resume."""

    def __init__(self, path, key_field):
        self.path = path
        self.key_field = key_field
        self._existing_ids = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            key = row.get(key_field)
                            if key:
                                self._existing_ids.add(key)
                        except Exception:
                            pass
        self._f = open(path, "a", encoding="utf-8")

    def has(self, key):
        return key in self._existing_ids

    def write(self, row, key=None):
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        if key:
            self._existing_ids.add(key)

    def close(self):
        self._f.close()


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
