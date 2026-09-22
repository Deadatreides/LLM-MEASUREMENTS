"""Slice-aware VDP and the inter-discipline check (MDP) with something to check.

VDP here is stricter than in the first polygon: the anchor must lie inside the
bytes the producing call was actually shown, not merely inside a file that was
mentioned. With the surface cut through files, "the file was given" is no longer
the same statement as "this text was visible".

MDP finally has load. When cuts fall inside function bodies, an edit can call a
helper it never saw, or invent one. A referenced bare name that resolves nowhere
in the whole retrieved surface is a broken seam -- and when the producing
disciplines differ, it is a seam between disciplines, which is precisely what
Kahn's inter-discipline review exists for.
"""
import re

from . import edits as E
from . import slices as S

BUILTINS = {
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "type", "isinstance", "issubclass", "getattr", "setattr", "hasattr",
    "delattr", "super", "open", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "sum", "min", "max", "abs", "round", "any", "all", "repr",
    "format", "hash", "id", "iter", "next", "callable", "vars", "dir", "bytes",
    "bytearray", "frozenset", "complex", "divmod", "pow", "slice", "object",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "OSError",
    "RuntimeError", "NotImplementedError", "AttributeError", "StopIteration",
    "if", "for", "while", "return", "and", "or", "not", "in", "is", "lambda",
    "def", "class", "try", "except", "finally", "with", "as", "assert", "yield",
    "elif", "else", "import", "from", "raise", "pass", "del", "global",
}

# bare `name(` -- attribute calls like os.path.exists(...) are skipped on
# purpose: their resolution lives in an import we did not retrieve, so counting
# them would manufacture violations
BARE_CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")


def vdp(parsed, repo_root, surface, visible_slices, block=None):
    """Layer 2. Deterministic, 0 model calls.

    An edit survives only if its file is in the surface, its SEARCH text occurs
    in that file, and the matched span lies inside bytes this call was shown.
    """
    kept, dropped = [], []
    allowed = set(surface.rel_list())
    for e in parsed:
        rec = dict(e)
        if block is not None:
            rec["block"] = block
        if E._norm(e["search"]).strip() == E._norm(e["replace"]).strip():
            dropped.append(dict(rec, reason="no_op"))
            continue
        if e["file"] not in allowed:
            dropped.append(dict(rec, reason="no_such_file"))
            continue
        text = surface.text[e["file"]]

        # Search INSIDE the bytes this call was shown, not the whole file.
        # Searching the file and then testing membership is wrong twice over:
        # a repeated snippet resolves to its first occurrence, so a legitimate
        # quote from a later slice is rejected, and -- worse -- if that first
        # occurrence happened to sit in another visible slice the edit would be
        # applied at the wrong place. Measured: 2 of 5 instances had an anchor
        # occurring twice in its file.
        span = how = None
        for sl in visible_slices:
            for rel, a, b in sl.spans:
                if rel != e["file"]:
                    continue
                found, kind = E._find(text[a:b], e["search"])
                if found:
                    span, how = (found[0] + a, found[1] + a), kind
                    break
            if span:
                break

        if span is None:
            # not in what was shown; say which, so the two cases stay separable
            whole, kind = E._find(text, e["search"])
            dropped.append(dict(rec, reason="outside_slice" if whole else kind))
            continue
        kept.append(dict(rec, span=span, match=how))
    return kept, dropped


def mdp(kept, surface, language="python"):
    """Layer 5. Deterministic. Reports seams, does not delete work.

    Applied identically to every arm, so it measures the edits rather than
    separating the arms.
    """
    defined = set()
    for rel in surface.rel_list():
        defined |= S.defined_names(surface.text[rel], language)
    for e in kept:
        defined |= S.defined_names(e["replace"], language)

    violations, cross = [], []
    for e in kept:
        for name in BARE_CALL.findall(e["replace"] or ""):
            if name in BUILTINS or name in defined:
                continue
            v = dict(file=e["file"], symbol=name, block=e.get("block"))
            violations.append(v)
            # a seam BETWEEN disciplines: some other block defined it
            for other in kept:
                if other is e or other.get("block") == e.get("block"):
                    continue
                if name in S.defined_names(other["replace"], language):
                    cross.append(dict(v, defined_in=other.get("block")))
                    break
    return dict(unresolved=violations, cross_discipline=cross)
