"""The Kan wrapper for SWE-bench Pro, and the SOLO arm it is measured against.

Layer map (PROTOCOL_K1.md §1), one model, blocks as disciplines:

  0  SCHEME        deterministic: requirements -> B blocks; retrieval pool
                   split across blocks. No LLM (LIFE-9: LLM planner 22%).
  1  EXECUTORS     B independent calls of the SAME model, one per discipline.
                   K1 measured the pool worth at most +0.050 and never any
                   emergence, so the executor layer is one-model-per-block.
  2  VDP           deterministic: file must exist, must be inside the block's
                   slice, SEARCH text must occur in it. Otherwise dropped.
  3  ASSEMBLY      deterministic: surviving edits applied, unified diff cut.
  4  NORMCONTROL   deterministic: git apply --check on the produced patch.
  5  MDP           deterministic: symbols on the seams between blocks.
  6  MARGIN GATE   a block that survives VDP with nothing gets one revision.

SOLO reads the SAME pool in one call with all requirements: equal read budget,
the LB2 design. The only difference is one reader or several.
"""
import os, re, subprocess, json, time

from . import llm, retrieve, edits as E

SYSTEM = (
    "You are a senior engineer fixing a real bug in a real repository. "
    "You output code edits and nothing else -- no explanation, no prose.\n\n"
    "Every edit MUST use exactly this format:\n\n"
    "### EDIT\n"
    "file: <path exactly as shown in the FILE headers>\n"
    "<<<<<<< SEARCH\n"
    "<lines copied VERBATIM from that file>\n"
    "=======\n"
    "<the replacement lines>\n"
    ">>>>>>> REPLACE\n\n"
    "Rules that are checked mechanically and will discard your edit if broken:\n"
    "- The SEARCH text must appear in the file character for character. Copy it "
    "from the file body, do not retype it from memory.\n"
    "- REPLACE must DIFFER from SEARCH. Restating the same lines is not an edit "
    "and is discarded.\n"
    "- Include enough surrounding lines in SEARCH to make it unique in the file.\n"
    "- Only edit the files given under FILE headers. A file marked '(excerpts)' "
    "shows real text between its markers -- quote from there, never across a "
    "'[... omitted ...]' gap.\n"
    "- Emit several ### EDIT blocks if several places need changing. Prefer a "
    "few precise edits over many; unfinished output is discarded."
)


def split_requirements(bullets, n_blocks):
    """Layer 0. Contiguous split, order preserved -- K1 kept the original order
    inside a block so the positional measurement stayed meaningful."""
    if not bullets:
        return []
    n_blocks = max(1, min(n_blocks, len(bullets)))
    per = (len(bullets) + n_blocks - 1) // n_blocks
    return [bullets[i:i + per] for i in range(0, len(bullets), per)]


def assign_files(pool, blocks, root, language):
    """Layer 0. Distribute the retrieval pool across blocks by per-block BM25.

    Every file goes to exactly one block, so the total read budget of the Kan
    arm equals the pool -- the same bytes SOLO reads. This is the LB2 "equal
    number of chunk reads" design, not redundancy.
    """
    files = [rel for _, rel in pool]
    if not files:
        return [[] for _ in blocks]
    bm = retrieve.BM25(root, files)
    # score every file against every block
    want = []
    for b in blocks:
        s = dict((rel, sc) for sc, rel in bm.score("\n".join(b)))
        want.append(s)

    # Balanced draft: blocks take turns picking their own highest-scoring file
    # that nobody has taken yet. Argmax-per-file was measured to collapse -- on
    # openlibrary one block took 7 of 8 files, two blocks got the same file as a
    # fallback, and the arm then read that file twice, which also broke the
    # equal-read-budget claim against SOLO.
    assigned = [[] for _ in blocks]
    left = list(files)
    turn = 0
    while left:
        bi = turn % len(blocks)
        s = want[bi]
        pick = max(left, key=lambda rel: (s.get(rel, 0.0), -files.index(rel)))
        assigned[bi].append(pick)
        left.remove(pick)
        turn += 1
    return assigned


def render_context(root, rels, language, char_budget, query="", per_file_cap=None):
    """Render the slice as verbatim text inside a character budget.

    What a file contributes is always literal text, windowed if need be, never a
    signature list -- a skeleton yields no quotable SEARCH anchor, and the
    executor said so itself: "only a skeleton is provided, I cannot edit it".

    Budget is shared by rank with decay 1/(i+1) and a hard floor, then whatever
    the small files leave over is handed back down the ranking. Both extremes
    were measured and both fail: pure greed by rank let configdata.yml (99 KB)
    crowd qtargs.py out of its own block entirely, and equal shares cut the
    ansible target file from whole to 55%. Decay keeps the top file whole while
    nothing is starved. The rule is identical in both arms.
    """
    if not rels:
        return ""
    sizes = {rel: len(retrieve.read_file(root, rel)) for rel in rels}
    weights = [1.0 / (i + 1) for i in range(len(rels))]
    total_w = sum(weights)
    floor = min(2500, char_budget // max(1, len(rels)))

    caps, leftover = {}, 0
    for i, rel in enumerate(rels):
        share = max(floor, int(char_budget * weights[i] / total_w))
        caps[rel] = min(sizes[rel], share)
        leftover += max(0, share - caps[rel])
    for rel in rels:                       # hand the slack back down the ranking
        if leftover <= 0:
            break
        need = sizes[rel] - caps[rel]
        if need > 0:
            give = min(need, leftover)
            caps[rel] += give
            leftover -= give

    parts, spent = [], 0
    for rel in rels:
        remaining = char_budget - spent
        if remaining < 1200:
            break
        chunk, _ = retrieve.render_file(root, rel, language,
                                        min(caps[rel], remaining), query)
        if len(chunk) > remaining:
            # hard clamp: the context has to fit n_ctx no matter what any
            # renderer decides. Measured need -- a definition-less YAML came
            # back at 99 KB against a 34 KB budget and would have blown n_ctx.
            cut = chunk.rfind("\n", 0, remaining)
            chunk = (chunk[:cut if cut > 0 else remaining]
                     + f"\n[... truncated ...]\n--- END FILE {rel}\n")
        parts.append(chunk)
        spent += len(chunk)
    return "\n".join(parts)


def quotable_files(context):
    """Files whose text was actually shown -- only those may be edited."""
    return set(re.findall(r"^--- FILE (\S+) \((?:full|excerpts)", context, re.M))


def build_prompt(problem, reqs, interface, context, block_label=None):
    head = f"# Issue\n{problem.strip()}\n"
    if block_label:
        head += (f"\n# Your discipline ({block_label})\n"
                 "You are responsible for ONLY these requirements. Other engineers "
                 "handle the rest; do not touch their parts.\n")
    else:
        head += "\n# Requirements\n"
    head += "\n".join(reqs) + "\n"
    if interface and "No new interfaces" not in interface:
        head += f"\n# Interface\n{interface.strip()}\n"
    head += f"\n# Repository files you may edit\n{context}\n"
    head += "\nProduce the ### EDIT blocks now."
    return head


def git(root, *args):
    return subprocess.run(["git", "-C", root] + list(args),
                          capture_output=True, text=True, encoding="utf-8", errors="ignore")


def cut_patch(root):
    """Layers 3 and 4: cut the diff, restore the tree, then check the patch
    really applies to the CLEAN tree.

    The check has to run against a pristine checkout -- that is what a benchmark
    harness does with a submitted patch. Checking it against the already-edited
    tree measures nothing.
    """
    d = git(root, "diff", "--no-color", "--no-ext-diff")
    patch = d.stdout or ""
    reset(root)
    if not patch.strip():
        return "", False
    pf = os.path.join(root, ".kan_patch.diff")
    try:
        with open(pf, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(patch if patch.endswith("\n") else patch + "\n")
        p = git(root, "apply", "--check", ".kan_patch.diff")
        ok = p.returncode == 0
    finally:
        if os.path.exists(pf):
            os.remove(pf)
    return patch, ok


def reset(root):
    git(root, "checkout", "--", ".")
    git(root, "clean", "-qfd")


REVISION_NOTE = {
    "no_op": "your SEARCH and REPLACE were identical, so nothing would change",
    "not_found": "your SEARCH text does not occur in the file",
    "whitespace": "",
    "no_such_file": "you edited a file that does not exist",
    "outside_block": "you edited a file you were not given in full",
    "empty": "your SEARCH block was empty",
}


def ask_with_gate(prompt, root, allowed, max_tokens, tag):
    """Layer 1 + 2 + 6: one executor call, VDP, and the margin gate.

    Used by BOTH arms. The revision is part of the Kan scheme ("3 ревизии"), so
    giving it only to KAN would measure the gate, not the decomposition.
    """
    calls, answers = [], []
    txt, meta = llm.chat(SYSTEM, prompt, max_tokens=max_tokens, tag=tag)
    calls.append(meta)
    answers.append(txt)
    parsed = E.parse_edits(txt)
    kept, dropped = E.vdp(parsed, root, allowed_files=allowed)
    revised = False
    truncated = meta["finish"] == "length"

    # A cut-off answer has no closing REPLACE marker, so nothing parses and the
    # gate used to stay shut: the block's whole turn was lost in silence. Every
    # truncation measured was a Kan block hitting the ceiling that SOLO, writing
    # one answer, never reached -- so it has to count as a failure and retry.
    if not kept and (parsed or truncated):
        why = "; ".join(sorted({REVISION_NOTE.get(d["reason"], d["reason"])
                                for d in dropped if REVISION_NOTE.get(d["reason"], d["reason"])}))
        if truncated and not why:
            why = "your answer was cut off before any edit was complete"
        retry = prompt + (
            f"\n\n# Revision\nEvery edit you produced was rejected: {why}. "
            "The SEARCH block must be text copied verbatim out of the file body "
            "above, and the REPLACE block must differ from it. Keep SEARCH short "
            "-- a few unique lines are enough. Try again.")
        txt2, meta2 = llm.chat(SYSTEM, retry, max_tokens=max_tokens, tag=tag + "-rev")
        calls.append(meta2)
        answers.append(txt2)
        parsed2 = E.parse_edits(txt2)
        kept2, dropped2 = E.vdp(parsed2, root, allowed_files=allowed)
        # keep BOTH rejection sets: overwriting them hid 19 first-attempt drops
        # in the pilot and made the VDP look far cleaner than it was
        dropped = [dict(d, attempt=1) for d in dropped] + \
                  [dict(d, attempt=2) for d in dropped2]
        if kept2:
            kept, revised = kept2, True
        raw_total = len(parsed) + len(parsed2)
        truncated = truncated or meta2["finish"] == "length"
    else:
        raw_total = len(parsed)

    return dict(kept=kept, dropped=dropped, raw=raw_total, revised=revised,
                truncated=truncated, calls=calls, answers=answers)


# --------------------------------------------------------------------------- arms

def run_solo(inst, root, pool, language, char_budget, max_tokens):
    reset(root)
    query = inst["problem_statement"] + "\n" + inst["requirements"]
    ctx = render_context(root, [rel for _, rel in pool], language, char_budget, query)
    prompt = build_prompt(inst["problem_statement"], inst["req_bullets"],
                          inst["interface"], ctx)
    r = ask_with_gate(prompt, root, quotable_files(ctx), max_tokens, "solo")
    applied, conflicts = E.apply_edits(r["kept"], root)
    patch, ok = cut_patch(root)
    out = dict(arm="solo", raw_edits=r["raw"], kept=len(r["kept"]), dropped=r["dropped"],
               applied=len(applied), conflicts=len(conflicts), raw_answers=r["answers"],
               patch=patch, normcontrol=ok, calls=r["calls"], revised=r["revised"],
               files_touched=sorted({e["file"] for e in applied}),
               context_chars=len(ctx), pool=[rel for _, rel in pool])
    return out


def run_kan(inst, root, pool, language, char_budget, max_tokens, n_blocks=3):
    reset(root)
    blocks = split_requirements(inst["req_bullets"], n_blocks)
    slices = assign_files(pool, blocks, root, language)

    per_block_budget = max(3000, char_budget // max(1, len(blocks)))
    all_kept, all_dropped, calls, block_report = [], [], [], []

    raw_answers = []
    for bi, (reqs, rels) in enumerate(zip(blocks, slices)):
        ctx = render_context(root, rels, language, per_block_budget,
                             inst["problem_statement"] + "\n" + "\n".join(reqs),
                             per_file_cap=max(4000, char_budget // 3))
        prompt = build_prompt(inst["problem_statement"], reqs, inst["interface"], ctx,
                              block_label=f"block {bi+1} of {len(blocks)}")
        # VDP: the file must be inside this block's slice -- K1's "id outside its
        # own block is a hallucination" -- and must have been shown in full
        allowed = set(rels) & quotable_files(ctx)
        r = ask_with_gate(prompt, root, allowed, max_tokens, f"kan-b{bi+1}")
        calls.extend(r["calls"])
        raw_answers.extend(r["answers"])
        kept, dropped = r["kept"], r["dropped"]

        for e in kept:
            e["block"] = bi + 1
        all_kept.extend(kept)
        all_dropped.extend(dict(d, block=bi + 1) for d in dropped)
        block_report.append(dict(block=bi + 1, reqs=len(reqs), files=rels,
                                 raw=r["raw"], kept=len(kept), revised=r["revised"],
                                 context_chars=len(ctx)))

    applied, conflicts = E.apply_edits(all_kept, root)
    seams = E.mdp(all_kept, root)
    patch, ok = cut_patch(root)
    out = dict(arm="kan", raw_edits=sum(b["raw"] for b in block_report),
               kept=len(all_kept), dropped=all_dropped, applied=len(applied),
               conflicts=len(conflicts), patch=patch, normcontrol=ok, calls=calls,
               blocks=block_report, seams=seams, raw_answers=raw_answers,
               files_touched=sorted({e["file"] for e in applied}),
               context_chars=sum(b["context_chars"] for b in block_report),
               pool=[rel for _, rel in pool])
    reset(root)
    return out
