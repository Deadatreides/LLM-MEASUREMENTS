"""Deterministic scoring against the gold patch. No LLM, no Docker.

What this measures and what it does NOT:
  - It DOES measure localisation (did the arm edit the files/lines the gold
    patch edits) and normcontrol (does the produced patch apply).
  - It does NOT measure "resolved". That requires running fail_to_pass inside
    the instance's Docker image, which is a separate, much more expensive step.
    Nothing here may be reported as a SWE-bench Pro score.
"""
import re

FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)

DOC_EXT = (".rst", ".md", ".asciidoc", ".txt")
CHANGELOG_HINT = ("changelog", "changelogs/", "doc/", "docs/")


def files_of(patch):
    return sorted(set(f for f, _ in FILE_RE.findall(patch or "")))


def is_doc(path):
    return path.lower().endswith(DOC_EXT) or any(h in path.lower() for h in CHANGELOG_HINT)


def line_spans(patch):
    """{file: set(original line numbers touched)} from a unified diff."""
    out, cur = {}, None
    for ln in (patch or "").split("\n"):
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", ln)
        if m:
            cur = m.group(1)
            out.setdefault(cur, set())
            continue
        h = re.match(r"^@@ -(\d+)(?:,(\d+))? \+", ln)
        if h and cur:
            start = int(h.group(1))
            length = int(h.group(2) or 1)
            out[cur].update(range(start, start + max(1, length)))
    return out


def score(pred_patch, gold_patch, tolerance=8):
    """Localisation metrics. `tolerance` = lines of slack when matching hunks."""
    pf, gf = set(files_of(pred_patch)), set(files_of(gold_patch))
    gf_code = {f for f in gf if not is_doc(f)}
    pf_code = {f for f in pf if not is_doc(f)}

    ps, gs = line_spans(pred_patch), line_spans(gold_patch)

    hit_lines, gold_lines = 0, 0
    for f, gl in gs.items():
        if is_doc(f):
            continue
        gold_lines += len(gl)
        pl = ps.get(f, set())
        if not pl:
            continue
        widened = set()
        for x in pl:
            widened.update(range(x - tolerance, x + tolerance + 1))
        hit_lines += len(gl & widened)

    def f1(a, b):
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if not inter:
            return 0.0
        p, r = inter / len(a), inter / len(b)
        return 2 * p * r / (p + r)

    return dict(
        gold_files=sorted(gf), gold_code_files=sorted(gf_code),
        pred_files=sorted(pf),
        file_recall=(len(pf_code & gf_code) / len(gf_code)) if gf_code else 0.0,
        file_precision=(len(pf_code & gf_code) / len(pf_code)) if pf_code else 0.0,
        file_f1=f1(pf_code, gf_code),
        exact_file_set=(pf_code == gf_code),
        hunk_line_recall=(hit_lines / gold_lines) if gold_lines else 0.0,
        pred_empty=not bool((pred_patch or "").strip()),
    )
