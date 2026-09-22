"""SEARCH/REPLACE edit blocks: parsing, the VDP filter, assembly into a patch.

Why SEARCH/REPLACE and not a raw unified diff: a model cannot reliably emit
correct @@ line numbers, and a wrong number is a formatting failure, not a
reasoning failure -- it would pollute the measurement. With SEARCH/REPLACE the
anchor is *quoted text*, which is exactly the LB2 VDP: "the quote must occur
verbatim in the chunk, otherwise it is dropped".

LB2 also measured that a too-strict verbatim filter rejected 362/870 answers of
which ~60% were false rejections (the model quoted correctly but reflowed
whitespace). VDP v2 there switched to the longest verbatim fragment. The same
lesson is applied here: matching is whitespace-normalised, and a block that
fails exact match is retried against its longest anchor run.
"""
import re, os, difflib

BLOCK_RE = re.compile(
    r"^###\s*EDIT\s*\n"
    r"file:\s*(?P<file>[^\n]+)\n"
    r"<{5,}\s*SEARCH\s*\n"
    r"(?P<search>.*?)\n"
    r"={5,}\s*\n"
    r"(?P<replace>.*?)\n"
    r">{5,}\s*REPLACE\s*$",
    re.M | re.S)


def parse_edits(text):
    """Extract edit blocks from a model answer. Deterministic, no LLM."""
    out = []
    for m in BLOCK_RE.finditer(text or ""):
        out.append(dict(file=m.group("file").strip().strip("`").lstrip("./"),
                        search=m.group("search"),
                        replace=m.group("replace")))
    return out


def _norm(s):
    """Whitespace-normalised view used for tolerant matching."""
    return "\n".join(ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n"))


def _find(haystack, needle):
    """Return ((start, end), how) of needle in haystack, or (None, reason).

    Matching is line-wise, and spans are always measured on the RAW string:
    these repos are checked out with core.autocrlf=true, so every line carries a
    trailing \\r and a span measured on a normalised copy would drift by one byte
    per line.

    Two strictness levels, strictest first -- which one fired is recorded per
    edit, because LB2 measured that a verbatim-only filter produced ~60% false
    rejections and the report has to show how much tolerance was actually used.
    """
    if not needle.strip():
        return None, "empty"

    raw_lines = haystack.split("\n")
    need_lines = needle.replace("\r\n", "\n").split("\n")
    while need_lines and not need_lines[-1].strip():
        need_lines.pop()
    if not need_lines:
        return None, "empty"

    def span_at(i, n):
        start = sum(len(x) + 1 for x in raw_lines[:i])
        end = start + sum(len(x) + 1 for x in raw_lines[i:i + n]) - 1
        # raw_lines keep the trailing \r of a CRLF file; leave it outside the
        # span so the file's own terminator survives the splice intact
        if raw_lines[i + n - 1].endswith("\r"):
            end -= 1
        return start, end

    # level 1: identical up to line terminator only
    strict_h = [ln.rstrip("\r") for ln in raw_lines]
    strict_n = [ln.rstrip("\r") for ln in need_lines]
    for i in range(len(strict_h) - len(strict_n) + 1):
        if strict_h[i:i + len(strict_n)] == strict_n:
            return span_at(i, len(strict_n)), "exact"

    # level 2: identical up to trailing whitespace
    loose_h = [ln.rstrip() for ln in raw_lines]
    loose_n = [ln.rstrip() for ln in need_lines]
    for i in range(len(loose_h) - len(loose_n) + 1):
        if loose_h[i:i + len(loose_n)] == loose_n:
            return span_at(i, len(loose_n)), "whitespace"

    return None, "not_found"


VDP_REASONS = ("exact", "whitespace", "not_found", "empty", "no_such_file",
               "outside_block", "no_op")


def vdp(edits, repo_root, allowed_files=None):
    """Layer 2 -- intra-discipline check. Deterministic, 0 model calls.

    An edit survives only if (a) its file exists in the checkout, (b) if the
    block was assigned a file slice, the file is inside that slice, and (c) its
    SEARCH text actually occurs in that file. Anything else is a hallucination
    and is dropped, exactly as K1 dropped ids that fell outside their block.
    """
    kept, dropped = [], []
    for e in edits:
        # A block whose REPLACE restates its SEARCH changes nothing. Left in, it
        # would apply "successfully", produce an empty diff, and be scored as a
        # well-formed answer -- so it is a hallucination of work and is dropped
        # here rather than allowed to look like an edit.
        if _norm(e["search"]).strip() == _norm(e["replace"]).strip():
            dropped.append(dict(e, reason="no_op"))
            continue
        path = os.path.join(repo_root, e["file"].replace("/", os.sep))
        if not os.path.isfile(path):
            dropped.append(dict(e, reason="no_such_file"))
            continue
        if allowed_files is not None and e["file"] not in allowed_files:
            dropped.append(dict(e, reason="outside_block"))
            continue
        # newline="" -- spans must be measured on the same raw bytes that
        # apply_edits will slice
        with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
            content = fh.read()
        span, how = _find(content, e["search"])
        if span is None:
            dropped.append(dict(e, reason=how))
            continue
        kept.append(dict(e, span=span, match=how))
    return kept, dropped


def apply_edits(edits, repo_root):
    """Layer 3 -- assembly. Applies surviving edits to the working tree.

    Returns (applied, conflicts). Edits to the same file are applied back to
    front so earlier spans stay valid.
    """
    by_file = {}
    for e in edits:
        by_file.setdefault(e["file"], []).append(e)

    applied, conflicts = [], []
    for rel, group in by_file.items():
        path = os.path.join(repo_root, rel.replace("/", os.sep))
        with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
            content = fh.read()
        # match the file's own line ending so the diff shows the edit, not a
        # whole-file CRLF/LF churn
        eol = "\r\n" if content.count("\r\n") > content.count("\n") / 2 else "\n"
        group = sorted(group, key=lambda e: -e["span"][0])
        used = []
        for e in group:
            s, t = e["span"]
            if any(not (t <= us or s >= ut) for us, ut in used):
                conflicts.append(dict(e, reason="overlapping_span"))
                continue
            repl = e["replace"].replace("\r\n", "\n").replace("\n", eol)
            content = content[:s] + repl + content[t:]
            used.append((s, t))
            applied.append(e)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
    return applied, conflicts


def mdp(edits, repo_root):
    """Layer 5 -- inter-discipline check on the seams. Deterministic.

    A symbol introduced by one block and referenced by another must resolve.
    Reports seams rather than deleting work: K1 measured MDP as degenerate when
    blocks are independent, so it is recorded, not enforced.
    """
    defined, referenced = {}, {}
    defre = re.compile(r"(?:def|class|func|function)\s+(\w+)")
    for e in edits:
        for name in defre.findall(e["replace"]):
            defined.setdefault(name, set()).add(e.get("block", "?"))
        for name in re.findall(r"\b(\w+)\s*\(", e["replace"]):
            referenced.setdefault(name, set()).add(e.get("block", "?"))

    seams = []
    for name, blocks in defined.items():
        users = referenced.get(name, set()) - blocks
        if users:
            seams.append(dict(symbol=name, defined_in=sorted(blocks), used_in=sorted(users)))
    return seams
