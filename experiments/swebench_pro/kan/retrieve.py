"""Deterministic retrieval over a repo checkout. No LLM, no gold-patch leakage.

This is layer 0 of the Kan scheme ("CXEMA"): the division of work is decided by
a fixed procedure, not by a model. LIFE-9 measured an LLM planner at 22%, so the
scheme must not be produced by the model it feeds.
"""
import os, re, math, collections

SRC_EXT = {
    "python": {".py"},
    "go": {".go"},
    "js": {".js", ".jsx", ".mjs", ".cjs"},
    "ts": {".ts", ".tsx"},
}
ALSO = {".yml", ".yaml", ".cfg", ".ini", ".toml"}

SKIP_DIRS = {".git", "node_modules", "vendor", "dist", "build", "__pycache__",
             ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", "site-packages"}

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
CAMEL_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def split_ident(tok):
    """foo_bar_baz / FooBarBaz -> component words, plus the whole token."""
    out = {tok.lower()}
    for part in tok.split("_"):
        if len(part) >= 3:
            out.add(part.lower())
        for m in CAMEL_RE.findall(part):
            if len(m) >= 3:
                out.add(m.lower())
    return out


def tokenize(text):
    bag = []
    for t in TOKEN_RE.findall(text):
        bag.extend(split_ident(t))
    return bag


def walk_repo(root, language, max_bytes=400_000):
    exts = SRC_EXT.get(language, {".py"}) | ALSO
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            full = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(full) > max_bytes:
                    continue
            except OSError:
                continue
            files.append(os.path.relpath(full, root).replace("\\", "/"))
    return sorted(files)


class BM25:
    """Plain BM25 over (path words + content words). Deterministic."""

    def __init__(self, root, files, k1=1.5, b=0.75):
        self.root, self.files, self.k1, self.b = root, files, k1, b
        self.tf = []
        self.len = []
        df = collections.Counter()
        for rel in files:
            try:
                with open(os.path.join(root, rel), encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                content = ""
            # path words weigh 3x -- a file named after the subject is strong evidence
            bag = tokenize(rel.replace("/", " ").replace(".", " ")) * 3 + tokenize(content)
            c = collections.Counter(bag)
            self.tf.append(c)
            self.len.append(max(1, len(bag)))
            df.update(c.keys())
        n = len(files)
        self.avg = sum(self.len) / max(1, n)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query_text):
        q = collections.Counter(tokenize(query_text))
        out = []
        for i, rel in enumerate(self.files):
            tf, dl, s = self.tf[i], self.len[i], 0.0
            for term, qn in q.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            if s > 0:
                out.append((s, rel))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out


DEF_RE = {
    "python": re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w+.*$", re.M),
    "go": re.compile(r"^\s*(?:func|type)\s+.*$", re.M),
    "js": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const\s+\w+\s*=\s*(?:async\s*)?\()\s*.*$", re.M),
}
DEF_RE["ts"] = DEF_RE["js"]


def read_file(root, rel):
    try:
        with open(os.path.join(root, rel), encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return ""


def read_file_raw(root, rel):
    """Bytes as they sit on disk, line endings untouched.

    Anything that computes an OFFSET must use this. These repos check out with
    core.autocrlf=true, so the universal-newline reader above silently drops a
    byte per line and every span built on it lands one \\r short per line -- the
    same class of defect the first polygon shipped and had to fix.
    """
    try:
        with open(os.path.join(root, rel), encoding="utf-8", errors="ignore",
                  newline="") as fh:
            return fh.read()
    except OSError:
        return ""


def skeleton(text, language, max_lines=60):
    """Signature-only view of a file: definitions with their line numbers."""
    rx = DEF_RE.get(language, DEF_RE["python"])
    lines = text.split("\n")
    out = []
    for m in rx.finditer(text):
        ln = text[:m.start()].count("\n") + 1
        out.append(f"{ln:5d}| {m.group(0).rstrip()}")
        if len(out) >= max_lines:
            out.append("      | ... (truncated)")
            break
    return "\n".join(out)


def numbered(text, max_lines=None):
    lines = text.split("\n")
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]
    return "\n".join(f"{i+1:5d}| {ln}" for i, ln in enumerate(lines))


def est_tokens(s):
    """~3.6 chars/token for source code with this tokenizer family."""
    return int(len(s) / 3.6)


def build_pool(root, language, query, pool_files=10):
    """Rank the repo and return the retrieval pool: [(rel, score)] top-N."""
    files = walk_repo(root, language)
    if not files:
        return [], 0
    bm = BM25(root, files, )
    ranked = bm.score(query)
    return ranked[:pool_files], len(files)


LINES_PER_BLOCK = 60


def regions(text, language):
    """Split a file into quotable regions covering every byte.

    Definition boundaries when the file has them; otherwise fixed line blocks.
    The fallback matters: a YAML/config file has no `def`, so boundary-splitting
    returns one region spanning the whole file, and a budget check that only
    counts whole regions then lets a 100 KB file through untouched.
    """
    rx = DEF_RE.get(language, DEF_RE["python"])
    bounds = sorted({0} | {m.start() for m in rx.finditer(text) if m.start() > 0})

    if len(bounds) < 3:
        bounds, off = [0], 0
        for i, ln in enumerate(text.split("\n"), 1):
            off += len(ln) + 1
            if i % LINES_PER_BLOCK == 0 and off < len(text):
                bounds.append(off)

    return [(bounds[i], bounds[i + 1] if i + 1 < len(bounds) else len(text))
            for i in range(len(bounds))]


def window_file(text, language, query, budget):
    """The most query-relevant slices of a large file, verbatim.

    A skeleton cannot be quoted, so a file reduced to signatures can never yield
    a SEARCH anchor and every edit against it dies in the VDP. This keeps real
    text for the parts that matter and marks the gaps, so the executor still has
    something it can copy.
    """
    regs = regions(text, language)
    if not regs:
        return text[:budget], [(0, min(len(text), budget))]

    q = collections.Counter(tokenize(query))
    scored = []
    for i, (a, b) in enumerate(regs):
        seg = text[a:b]
        bag = collections.Counter(tokenize(seg))
        # length-damped overlap: long regions should not win on size alone
        s = sum(min(q[t], bag[t]) for t in q) / (1 + math.log(1 + len(seg)))
        scored.append((s, i))
    scored.sort(key=lambda x: (-x[0], x[1]))

    chosen, spent = {}, 0
    for s, i in scored:
        a, b = regs[i]
        room = budget - spent
        if room < 400:
            break
        if b - a > room:
            # keep a line-aligned prefix: still verbatim text, just less of it
            cut = text.rfind("\n", a, a + room)
            if cut <= a:
                continue
            b = cut
        chosen[i] = b
        spent += b - a
    if not chosen:
        a, _ = regs[scored[0][1]]
        cut = text.rfind("\n", a, a + budget)
        chosen[scored[0][1]] = cut if cut > a else min(len(text), a + budget)

    out, kept_spans = [], []
    for i, (a, b_full) in enumerate(regs):
        if i in chosen:
            b = chosen[i]
            line0 = text[:a].count("\n") + 1
            out.append(f"@@ lines {line0}.. @@\n{text[a:b]}")
            kept_spans.append((a, b))
            if b < b_full:
                out.append("[... omitted ...]\n")
        elif out and not out[-1].endswith("[... omitted ...]\n"):
            out.append("[... omitted ...]\n")
    return "\n".join(out), kept_spans


def render_file(root, rel, language, budget_chars, query=""):
    """Verbatim text within budget: whole file if it fits, else the relevant
    windows of it. Never a bare skeleton -- see window_file."""
    text = read_file(root, rel)
    nlines = text.count("\n") + 1
    if len(text) <= budget_chars:
        return (f"--- FILE {rel} (full, {nlines} lines)\n{text}\n"
                f"--- END FILE {rel}\n"), True
    body, _ = window_file(text, language, query, budget_chars)
    return (f"--- FILE {rel} (excerpts, {nlines} lines total; text between the "
            f"markers is verbatim and may be quoted, gaps are marked)\n{body}\n"
            f"--- END FILE {rel}\n"), True
