import json
import os

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment6"
RAW_DIR = os.path.join(EXP_DIR, "runs", "raw")


def save_raw(run_id, raw_text, meta):
    d = os.path.join(RAW_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "result.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text or "")
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)
    return f"raw/{run_id}/"


class JsonlWriter:
    def __init__(self, path, key_field):
        self.path = path
        self.key_field = key_field
        self._existing = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            k = row.get(key_field)
                            if k:
                                self._existing.add(k)
                        except Exception:
                            pass
        self._f = open(path, "a", encoding="utf-8")

    def has(self, key):
        return key in self._existing

    def write(self, row, key=None):
        self._f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        self._f.flush()
        if key:
            self._existing.add(key)

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
