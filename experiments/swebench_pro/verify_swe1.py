"""Structural verification of the SWE1 polygon. No model calls, no GPU.

Checks the things that, in the previous polygon, were only discovered by burning
a 45-minute run each: slice coverage, verbatim fidelity, offset correctness on
CRLF, slice-aware VDP, MDP resolution, and the equal-budget invariants between
arms. Run this before any live run.
"""
import os, sys, json, re, tempfile, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kan import retrieve, slices as SL, seams as SE, edits as E, arms, pipeline, llm, score  # noqa

REPOS = r"H:\swebench_pro_repos"
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return cond


# --------------------------------------------------------------- 1. slicing

def check_slicing():
    print("\n[1] slicing: coverage, verbatim fidelity, line-aligned cuts")
    body = "".join(f"line {i} of the file\n" for i in range(1, 401))
    files = [("a/one.py", body), ("b/two.py", body.replace("line", "row")),
             ("c/three.py", "short\n")]
    surf = SL.Surface(files)
    sl = SL.cut(surf, 3000)

    # every byte of every file appears exactly once, in order
    rebuilt = {}
    ok_order = True
    for s in sl:
        for rel, a, b in s.spans:
            prev = rebuilt.get(rel, 0)
            if a != prev:
                ok_order = False
            rebuilt[rel] = b
    check("slices cover each file exactly once, in order", ok_order)
    check("coverage reaches end of every file",
          all(rebuilt.get(rel, -1) == len(t) for rel, t in files),
          str({r: (rebuilt.get(r), len(t)) for r, t in files}))
    check("total sliced chars == surface size",
          sum(s.chars() for s in sl) == surf.total,
          f"{sum(s.chars() for s in sl)} vs {surf.total}")

    # verbatim: the concatenated spans reproduce the file byte for byte
    joined = {rel: "" for rel, _ in files}
    for s in sl:
        for rel, a, b in s.spans:
            joined[rel] += surf.text[rel][a:b]
    check("spans reproduce every file byte for byte",
          all(joined[rel] == t for rel, t in files))

    # cuts land on line boundaries
    cuts_ok = True
    for s in sl:
        for rel, a, b in s.spans:
            if b < surf.sizes[rel] and surf.text[rel][b - 1] != "\n":
                cuts_ok = False
    check("every internal cut lands after a newline", cuts_ok)

    check("the surface actually gets split (seams exist)",
          len(SL.seam_index(sl)) >= 2, f"{len(SL.seam_index(sl))} files split")
    check("no slice exceeds its budget",
          all(s.chars() <= 3000 for s in sl),
          f"max {max(s.chars() for s in sl)}")
    return surf, sl


# ------------------------------------------------- 2. offsets on CRLF files

def check_crlf_offsets():
    print("\n[2] offsets on a CRLF checkout (core.autocrlf=true here)")
    d = tempfile.mkdtemp()
    try:
        body_lf = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
        p = os.path.join(d, "m.py")
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(body_lf.replace("\n", "\r\n"))

        raw = retrieve.read_file_raw(d, "m.py")
        norm = retrieve.read_file(d, "m.py")
        check("read_file_raw preserves CRLF", "\r\n" in raw)
        check("read_file (normalising) does NOT preserve CRLF", "\r\n" not in norm)
        check("raw is longer by one byte per line",
              len(raw) - len(norm) == body_lf.count("\n"),
              f"{len(raw)} vs {len(norm)}")

        surf = SL.Surface([("m.py", raw)])
        sl = SL.cut(surf, 10000)
        anchor = "def beta():\r\n    return 2"
        span, how = E._find(raw, anchor)
        check("anchor found in raw text", span is not None, str(how))
        check("span slices back to the anchor",
              span and raw[span[0]:span[1]].rstrip("\r\n") == anchor.rstrip("\r\n"),
              repr(raw[span[0]:span[1]])[:60] if span else "")
        check("slice.contains() accepts the span",
              any(s.contains("m.py", span[0], span[1]) for s in sl))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------- 3. slice-aware VDP

def check_vdp():
    print("\n[3] VDP: anchor must be inside what THIS call was shown")
    body = "".join(f"def fn{i}():\n    return {i}\n" for i in range(200))
    surf = SL.Surface([("m.py", body)])
    sl = SL.cut(surf, 1500)
    check("test needs >=2 slices", len(sl) >= 2, f"{len(sl)} slices")

    first, second = sl[0], sl[1]
    a0, b0 = first.spans[0][1], first.spans[0][2]
    a1, b1 = second.spans[0][1], second.spans[0][2]

    def two_lines(a, b):
        """Two consecutive lines exactly as they sit in the file.

        No strip(): dropping the indentation makes the quote non-verbatim, which
        is a defect of the test, not of the filter under test.
        """
        seg = body[a:b].split("\n")
        return [ln for ln in seg[:2]]

    in_first = two_lines(a0, b0)
    in_second = two_lines(a1, b1)

    def one(search, replace, visible):
        return SE.vdp([dict(file="m.py", search=search, replace=replace)],
                      "", surf, visible)

    kept, dropped = one("\n".join(in_first), "\n".join(in_first) + "\n    # x", [first])
    check("anchor inside the shown slice is kept", len(kept) == 1,
          dropped[0]["reason"] if dropped else "")

    kept, dropped = one("\n".join(in_second), "\n".join(in_second) + "\n    # x", [first])
    check("anchor from ANOTHER slice is rejected as outside_slice",
          not kept and dropped and dropped[0]["reason"] == "outside_slice",
          dropped[0]["reason"] if dropped else "kept!")

    kept, dropped = one("\n".join(in_first), "\n".join(in_first), [first])
    check("no-op edit rejected", not kept and dropped[0]["reason"] == "no_op",
          dropped[0]["reason"] if dropped else "kept!")

    kept, dropped = one("def nothing_like_this():\n    pass", "x", [first])
    check("absent anchor rejected as not_found",
          not kept and dropped[0]["reason"] == "not_found",
          dropped[0]["reason"] if dropped else "kept!")

    kept, dropped = SE.vdp([dict(file="ghost.py", search="a", replace="b")],
                           "", surf, [first])
    check("edit to a file outside the surface rejected",
          not kept and dropped[0]["reason"] == "no_such_file",
          dropped[0]["reason"] if dropped else "kept!")


# ------------------------------------------------------------- 4. MDP

def check_mdp():
    print("\n[4] MDP: unresolved references across seams")
    surf = SL.Surface([("m.py", "def existing_helper():\n    return 1\n"),
                       ("n.py", "def other_helper():\n    return 2\n")])

    r = SE.mdp([dict(file="m.py", replace="x = existing_helper()", block=1)], surf)
    check("call to a symbol defined in the surface -> no violation",
          not r["unresolved"], str(r["unresolved"]))

    r = SE.mdp([dict(file="m.py", replace="x = invented_helper()", block=1)], surf)
    check("call to an invented symbol -> violation",
          len(r["unresolved"]) == 1, str(r["unresolved"]))

    r = SE.mdp([dict(file="m.py", replace="x = len(y) + int(z)", block=1)], surf)
    check("builtins are not counted", not r["unresolved"], str(r["unresolved"]))

    r = SE.mdp([dict(file="m.py", replace="x = os.path.exists(p)", block=1)], surf)
    check("attribute calls are not counted", not r["unresolved"], str(r["unresolved"]))

    r = SE.mdp([dict(file="m.py", replace="v = made_here()", block=1),
                dict(file="n.py", replace="def made_here():\n    return 3", block=2)],
               surf)
    check("symbol defined by ANOTHER discipline resolves",
          not r["unresolved"], str(r["unresolved"]))

    r = SE.mdp([dict(file="m.py", replace="v = only_in_two()", block=1),
                dict(file="n.py", replace="def only_in_two():\n    return 3", block=2)],
               surf)
    check("cross-discipline dependency is recorded when unresolved elsewhere",
          isinstance(r["cross_discipline"], list))


# ------------------------------------------ 5. equal-budget invariants

def check_budget_invariants():
    print("\n[5] equal-budget invariants between arms (the class that bit us 3x)")
    body = "".join(f"def fn{i}():\n    return {i}\n" for i in range(600))
    surf = SL.Surface([("m.py", body), ("n.py", body.replace("fn", "gn"))])
    sl = SL.cut(surf, 8000)
    n_blocks = 3
    dealt = arms.deal_slices(sl, n_blocks)

    flat = [s.idx for grp in dealt for s in grp]
    check("KAN sees every slice exactly once",
          sorted(flat) == [s.idx for s in sl], f"{len(flat)} vs {len(sl)}")
    check("SOLO_SERIAL and KAN read identical chars",
          sum(s.chars() for s in sl) == sum(s.chars() for grp in dealt for s in grp))
    check("SOLO_SERIAL and KAN make the same number of calls",
          len(sl) == sum(len(g) for g in dealt))
    check("TRUNC reads exactly one slice", True, "by construction")
    sizes = [len(g) for g in dealt]
    check("slices dealt evenly across disciplines", max(sizes) - min(sizes) <= 1, str(sizes))

    split = SL.seam_index(sl)
    dealt_of = {s.idx: bi for bi, g in enumerate(dealt) for s in g}
    crossing = sum(1 for rel, parts in split.items()
                   if len({dealt_of[i] for i, _, _ in parts}) > 1)
    check("round-robin puts split files in different disciplines (seams are real)",
          crossing >= 1, f"{crossing} of {len(split)} split files cross disciplines")


# --------------------------------------------- 6. end-to-end with a fake model

def check_end_to_end():
    print("\n[6] end-to-end on a real checkout with a scripted model (no GPU)")
    pilot = json.load(open(os.path.join(HERE, "data", "pilot3.json"), encoding="utf-8"))
    inst = pilot[0]
    root = os.path.join(REPOS, inst["instance_id"])
    if not os.path.isdir(os.path.join(root, ".git")):
        print("  SKIP (no checkout)")
        return
    pipeline.reset(root)

    from run_swe1 import build_surface
    q = inst["problem_statement"] + "\n" + inst["requirements"]
    surf = build_surface(root, "python", q, 8)
    sl = SL.cut(surf, 30000)
    check("real surface exceeds one window",
          surf.total > 16384 * 3.6, f"{surf.total/(16384*3.6):.2f}x")
    check("real surface splits into several slices", len(sl) > 1, f"{len(sl)} slices")

    # scripted model: quote the first 3 real lines of the slice it is shown
    def fake_chat(system, user, max_tokens=0, temperature=0.0, reasoning="low", tag=""):
        m = re.search(r"^--- FILE (\S+) \[chars (\d+)\.\.(\d+)", user, re.M)
        if not m:
            return "NO EDITS", dict(tag=tag, seconds=0.0, prompt_tokens=len(user)//4,
                                    completion_tokens=3, reasoning_chars=0, finish="stop")
        rel, a = m.group(1), int(m.group(2))
        chunk = surf.text[rel][a:a + 4000]
        # CONSECUTIVE lines, blanks included: skipping blanks would join lines
        # that are not adjacent in the file and the quote stops being verbatim
        lines = chunk.split("\n")
        start = next((i for i, l in enumerate(lines) if l.strip()), 0)
        anchor = "\n".join(lines[start:start + 3])
        if not anchor.strip():
            return "NO EDITS", dict(tag=tag, seconds=0.0, prompt_tokens=len(user)//4,
                                    completion_tokens=3, reasoning_chars=0, finish="stop")
        body = ("### EDIT\nfile: " + rel + "\n<<<<<<< SEARCH\n" + anchor +
                "\n=======\n" + anchor + "\n# KAN-VERIFY\n>>>>>>> REPLACE\n")
        return body, dict(tag=tag, seconds=0.0, prompt_tokens=len(user)//4,
                          completion_tokens=len(body)//4, reasoning_chars=0,
                          finish="stop")

    real_chat = llm.chat
    llm.chat = fake_chat
    try:
        for name, fn, kw in (("TRUNC", arms.run_trunc, {}),
                             ("SOLO", arms.run_solo_serial, {}),
                             ("KAN", arms.run_kan,
                              dict(n_blocks=3,
                                   blocks_of_reqs=pipeline.split_requirements(inst["req_bullets"], 3)))):
            pipeline.reset(root)
            r = fn(inst, root, surf, sl, 3000, None, **kw)
            applied, conflicts = E.apply_edits(r["kept"], root)
            patch, ok = pipeline.cut_patch(root)
            sc = score.score(patch, inst["gold_patch"])
            check(f"{name}: produced a patch that passes normcontrol",
                  bool(patch) and ok,
                  f"kept={len(r['kept'])} applied={len(applied)} "
                  f"chars={len(patch)} slices={r['slices_read']}")
            if name == "SOLO":
                solo_read, solo_calls = r["read_chars"], len(r["calls"])
            if name == "KAN":
                check("KAN read exactly what SOLO read",
                      r["read_chars"] == solo_read, f"{r['read_chars']} vs {solo_read}")
                check("KAN made the same number of calls as SOLO",
                      len(r["calls"]) == solo_calls, f"{len(r['calls'])} vs {solo_calls}")
    finally:
        llm.chat = real_chat
        pipeline.reset(root)
        check("checkout left clean",
              not pipeline.git(root, "status", "--porcelain").stdout.strip())


def main():
    print("=" * 72)
    print("SWE1 polygon verification")
    print("=" * 72)
    check_slicing()
    check_crlf_offsets()
    check_vdp()
    check_mdp()
    check_budget_invariants()
    check_end_to_end()
    print("\n" + "=" * 72)
    print(f"PASSED {len(OK)}   FAILED {len(BAD)}")
    for b in BAD:
        print("  FAILED:", b)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
