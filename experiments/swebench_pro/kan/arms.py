"""The three arms, in the regime where decomposition is the only way to read.

  TRUNC        one call, the first slice only -- "as it is at n_ctx".
  SOLO_SERIAL  one model walks every slice in order, carrying its own notes.
  KAN          the same slices dealt to disciplines; VDP per call, assembly,
               normcontrol, MDP over the seams the cuts created.

SOLO_SERIAL and KAN read the SAME slices and make the SAME number of calls, so
the read budget is equal by construction rather than by arithmetic -- the first
polygon proved that equal sums can still hide an unequal ceiling. The only
difference is whether one reader holds all requirements or several hold one
discipline each.
"""
import os

from . import llm, edits as E, seams as SE, slices as SL

SYSTEM = (
    "You are a senior engineer fixing a real bug in a real repository. "
    "You are shown one SLICE of the relevant code at a time. You output code "
    "edits and nothing else -- no explanation, no prose.\n\n"
    "Every edit MUST use exactly this format:\n\n"
    "### EDIT\n"
    "file: <path exactly as in the FILE header>\n"
    "<<<<<<< SEARCH\n"
    "<lines copied VERBATIM from the slice>\n"
    "=======\n"
    "<the replacement lines>\n"
    ">>>>>>> REPLACE\n\n"
    "Rules checked mechanically; a broken rule discards the edit:\n"
    "- SEARCH must appear in THIS slice character for character. Copy it from "
    "the text above; do not retype it from memory and do not quote a part of the "
    "file that is not shown here.\n"
    "- REPLACE must DIFFER from SEARCH. Restating unchanged lines is not an edit.\n"
    "- A header saying '(continues in a later slice)' means the file is cut: do "
    "not quote past the end of what you were shown.\n"
    "- If this slice contains nothing relevant to your requirements, reply with "
    "exactly NO EDITS. That is a valid answer and costs you nothing.\n"
    "- Keep SEARCH short: a few unique lines are enough."
)

NOTE_LIMIT = 1200


def build_prompt(problem, reqs, slice_text, slice_idx, n_slices, notes,
                 discipline=None):
    head = f"# Issue\n{problem.strip()}\n"
    if discipline:
        head += (f"\n# Your discipline ({discipline})\n"
                 "You are responsible for ONLY these requirements. Other "
                 "engineers cover the rest; do not touch their parts.\n")
    else:
        head += "\n# Requirements\n"
    head += "\n".join(reqs) + "\n"
    if notes:
        head += f"\n# What you noted in earlier slices\n{notes[:NOTE_LIMIT]}\n"
    head += (f"\n# Slice {slice_idx + 1} of {n_slices}\n{slice_text}\n"
             "\nProduce ### EDIT blocks for this slice, or NO EDITS.")
    return head


def _note(kept, dropped, slice_idx):
    """A one-line memo carried to the next slice. Bounded on purpose: an
    unbounded scratchpad would quietly hand the serial arm a growing context and
    break the equal-read comparison."""
    if not kept and not dropped:
        return f"slice {slice_idx + 1}: nothing relevant."
    files = sorted({e["file"].split("/")[-1] for e in kept})
    return (f"slice {slice_idx + 1}: {len(kept)} edit(s) accepted"
            + (f" in {', '.join(files)}" if files else "") + ".")


def _one_call(prompt, repo_root, surface, visible, max_tokens, tag, block=None):
    txt, meta = llm.chat(SYSTEM, prompt, max_tokens=max_tokens, tag=tag)
    parsed = E.parse_edits(txt)
    kept, dropped = SE.vdp(parsed, repo_root, surface, visible, block=block)
    return dict(text=txt, meta=meta, raw=len(parsed), kept=kept, dropped=dropped,
                truncated=meta["finish"] == "length",
                declined=txt.strip().upper().startswith("NO EDITS"))


# --------------------------------------------------------------------------- arms

def run_trunc(inst, root, surface, slices, max_tokens, ctx):
    """Baseline: whatever fits the window, one call."""
    s = slices[0]
    prompt = build_prompt(inst["problem_statement"], inst["req_bullets"],
                          s.render(), 0, len(slices), "")
    r = _one_call(prompt, root, surface, [s], max_tokens, "trunc")
    return dict(kept=r["kept"], dropped=r["dropped"], calls=[r["meta"]],
                answers=[r["text"]], raw=r["raw"],
                read_chars=s.chars(), slices_read=1,
                declined=int(r["declined"]), truncated=int(r["truncated"]))


def run_solo_serial(inst, root, surface, slices, max_tokens, ctx):
    """One reader, every slice, notes carried forward."""
    kept, dropped, calls, answers, notes = [], [], [], [], []
    raw = declined = truncated = 0
    for s in slices:
        prompt = build_prompt(inst["problem_statement"], inst["req_bullets"],
                              s.render(), s.idx, len(slices), "\n".join(notes))
        r = _one_call(prompt, root, surface, [s], max_tokens, f"solo-s{s.idx+1}")
        calls.append(r["meta"])
        answers.append(r["text"])
        kept.extend(r["kept"])
        dropped.extend(r["dropped"])
        raw += r["raw"]
        declined += int(r["declined"])
        truncated += int(r["truncated"])
        notes.append(_note(r["kept"], r["dropped"], s.idx))
        notes[:] = notes[-6:]
    return dict(kept=kept, dropped=dropped, calls=calls, answers=answers, raw=raw,
                read_chars=sum(s.chars() for s in slices), slices_read=len(slices),
                declined=declined, truncated=truncated)


def deal_slices(slices, n_blocks):
    """Round-robin, as LB2 dealt chunks: parallel coverage, not redundancy.

    Contiguity is deliberately NOT preserved. Dealing block-wise would give each
    discipline a contiguous run and hide the seams again; round-robin guarantees
    that a file split by a cut lands in different disciplines, which is the case
    MDP exists to inspect.
    """
    out = [[] for _ in range(n_blocks)]
    for i, s in enumerate(slices):
        out[i % n_blocks].append(s)
    return out


def run_kan(inst, root, surface, slices, max_tokens, ctx, n_blocks=3,
            blocks_of_reqs=None):
    """The scheme: disciplines, VDP per call, assembly, MDP over the seams."""
    reqs_blocks = blocks_of_reqs or [inst["req_bullets"]]
    dealt = deal_slices(slices, len(reqs_blocks))
    kept, dropped, calls, answers, report = [], [], [], [], []
    raw = declined = truncated = 0

    for bi, (reqs, my_slices) in enumerate(zip(reqs_blocks, dealt), start=1):
        notes, bkept = [], 0
        for s in my_slices:
            prompt = build_prompt(inst["problem_statement"], reqs, s.render(),
                                  s.idx, len(slices), "\n".join(notes),
                                  discipline=f"block {bi} of {len(reqs_blocks)}")
            r = _one_call(prompt, root, surface, [s], max_tokens,
                          f"kan-b{bi}-s{s.idx+1}", block=bi)
            calls.append(r["meta"])
            answers.append(r["text"])
            kept.extend(r["kept"])
            dropped.extend(r["dropped"])
            raw += r["raw"]
            bkept += len(r["kept"])
            declined += int(r["declined"])
            truncated += int(r["truncated"])
            notes.append(_note(r["kept"], r["dropped"], s.idx))
            notes[:] = notes[-6:]
        report.append(dict(block=bi, reqs=len(reqs), slices=[s.idx for s in my_slices],
                           kept=bkept, chars=sum(s.chars() for s in my_slices)))

    return dict(kept=kept, dropped=dropped, calls=calls, answers=answers, raw=raw,
                read_chars=sum(s.chars() for s in slices), slices_read=len(slices),
                declined=declined, truncated=truncated, blocks=report)
