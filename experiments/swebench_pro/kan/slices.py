"""The retrieved surface, and slicing it THROUGH files rather than by file.

Why this module exists. The first polygon retrieved a handful of files, squeezed
them to 58% of one window, and then compared one call against three calls inside
that window. Decomposition buys exactly one thing -- the ability to read what
does not fit -- so in a regime where the baseline never hit the limit there was
nothing to buy, and the arms could not differ for the right reason.

Here the surface is the top-K relevant files at full size (measured: 2.5x-6.1x
the window) and it is cut into window-sized slices at fixed offsets. A cut falls
wherever it falls, including in the middle of a function. That is deliberate: a
symbol defined in one slice and used in another is a SEAM, and seams are what the
inter-discipline check (MDP) exists to inspect. Handing each discipline whole
files, as the first polygon did, produced `seams: []` by construction -- the
degenerate case K1 already knew about.
"""
import re

FILE_HDR = "--- FILE {rel} [chars {a}..{b} of {n}]{note}\n"
FILE_END = "--- END {rel}\n"


class Surface:
    """Top-K retrieved files, kept whole, addressable by (rel, offset)."""

    def __init__(self, files):
        # files: [(rel, text)] in retrieval-rank order
        self.files = list(files)
        self.sizes = {rel: len(text) for rel, text in self.files}
        self.text = {rel: text for rel, text in self.files}
        self.total = sum(self.sizes.values())

    def rel_list(self):
        return [rel for rel, _ in self.files]


class Slice:
    """One window-sized cut of the surface.

    `spans` is [(rel, a, b)] -- exactly the bytes this slice shows. VDP uses it
    to decide whether an anchor was actually visible to the call that produced
    the edit, which is stricter and more honest than "the file was mentioned".
    """

    def __init__(self, idx, spans, surface):
        self.idx = idx
        self.spans = spans
        self.surface = surface

    def files(self):
        return [rel for rel, _, _ in self.spans]

    def chars(self):
        return sum(b - a for _, a, b in self.spans)

    def contains(self, rel, start, end):
        return any(r == rel and a <= start and end <= b for r, a, b in self.spans)

    def render(self, cut_map=None):
        """Verbatim text with explicit cut markers.

        The markers are not decoration: a call must be able to tell that it is
        looking at a fragment, otherwise it quotes across a boundary it cannot
        see and the anchor silently fails.
        """
        out = []
        for rel, a, b in self.spans:
            n = self.surface.sizes[rel]
            note = ""
            if a > 0:
                note += " (continued from an earlier slice)"
            if b < n:
                note += " (continues in a later slice)"
            out.append(FILE_HDR.format(rel=rel, a=a, b=b, n=n, note=note))
            out.append(self.surface.text[rel][a:b])
            if not out[-1].endswith("\n"):
                out.append("\n")
            out.append(FILE_END.format(rel=rel))
        return "".join(out)


def cut(surface, slice_chars):
    """Cut the whole surface into slices of ~slice_chars, crossing file borders.

    Cuts are snapped to the nearest line break so that no slice ends mid-line --
    a half line is unquotable and would only manufacture failures. Everything
    else about the boundary is left alone: cutting inside a function is the
    point.
    """
    slices, spans, spent, idx = [], [], 0, 0
    for rel, text in surface.files:
        pos = 0
        n = len(text)
        while pos < n:
            room = slice_chars - spent
            if room < 1000:
                slices.append(Slice(idx, spans, surface))
                idx, spans, spent = idx + 1, [], 0
                room = slice_chars
            end = min(n, pos + room)
            if end < n:
                nl = text.rfind("\n", pos, end)
                if nl > pos:
                    end = nl + 1
            spans.append((rel, pos, end))
            spent += end - pos
            pos = end
    if spans:
        slices.append(Slice(idx, spans, surface))
    return slices


def seam_index(slices):
    """Which files are split across slices, and where the cuts land."""
    where = {}
    for s in slices:
        for rel, a, b in s.spans:
            where.setdefault(rel, []).append((s.idx, a, b))
    return {rel: v for rel, v in where.items() if len(v) > 1}


DEF_PAT = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)", re.M)
GO_PAT = re.compile(r"^[ \t]*func(?:[ \t]*\([^)]*\))?[ \t]*(\w+)", re.M)


def defined_names(text, language="python"):
    pat = GO_PAT if language == "go" else DEF_PAT
    return set(pat.findall(text or ""))


def split_across(slices, language="python"):
    """Names whose definition line and body end up in different slices.

    This is the population MDP is supposed to protect: a discipline that only
    sees the tail of a function cannot know its signature, and one that only
    sees the head cannot know what it returns.
    """
    per_slice = {}
    for s in slices:
        names = set()
        for rel, a, b in s.spans:
            names |= defined_names(s.surface.text[rel][a:b], language)
        per_slice[s.idx] = names
    out = {}
    for i, names in per_slice.items():
        for j, other in per_slice.items():
            if i < j:
                for nm in names & other:
                    out.setdefault(nm, set()).update({i, j})
    return {k: sorted(v) for k, v in out.items()}
