"""Pick the SWE1 instance set and lay out one checkout per instance.

Reuses the three repos already cloned: a new instance is a different
base_commit, so the existing object store is copied and only the delta is
fetched. Selection is on task shape (files, hunks, requirement count) only --
never on retrieval success or on any outcome.
"""
import os, re, sys, json, shutil, subprocess, collections
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "swebench_pro_test.parquet")
OUT = os.path.join(HERE, "data", "swe1.json")
REPOS = r"H:\swebench_pro_repos"

REPO_SEED = {                      # existing checkouts used as object caches
    "ansible/ansible": "instance_ansible__ansible-748f534312f2073a25a87871f5bd05882891b8c4-v0f01c69f1e2528b935359cfe578530722bca2c59",
    "internetarchive/openlibrary": "instance_internetarchive__openlibrary-a7b7dc5735a1b3a9824376b1b469b556dd413981-va4315b5dc369c1ef66ae22f9ae4267aa3114e1b3",
    "qutebrowser/qutebrowser": "instance_qutebrowser__qutebrowser-ef5ba1a0360b39f9eff027fbdc57f363597c3c3b-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d",
}
PER_REPO = {"ansible/ansible": 7, "internetarchive/openlibrary": 7,
            "qutebrowser/qutebrowser": 6}

fre = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
hre = re.compile(r"^@@ ", re.M)


def unesc(s):
    s = s or ""
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")


def bullets(s):
    return [b.strip() for b in unesc(s).split("\n") if b.strip().startswith("- ")]


def git(d, *a):
    return subprocess.run(["git", "-C", d] + list(a), capture_output=True,
                          text=True, encoding="utf-8", errors="ignore")


def pick():
    rows = pq.read_table(DATA).to_pylist()
    by_repo = collections.defaultdict(list)
    for r in rows:
        if r["repo"] not in PER_REPO:
            continue
        p = r["patch"] or ""
        files = sorted(set(f for f, _ in fre.findall(p)))
        nb = len(bullets(r["requirements"]))
        if not (2 <= len(files) <= 4 and 4 <= len(hre.findall(p)) <= 12 and 4 <= nb <= 10):
            continue
        by_repo[r["repo"]].append((r, files, nb))

    chosen = []
    for repo, want in PER_REPO.items():
        lst = sorted(by_repo[repo], key=lambda c: (len(c[1]), c[2], c[0]["instance_id"]))
        step = max(1, len(lst) // max(1, want + 1))
        for k in range(want):
            idx = min(len(lst) - 1, step * (k + 1))
            chosen.append(lst[idx])
        print(f"  {repo}: {len(lst)} candidates -> took {want}")
    return chosen


def setup(chosen):
    out = []
    for r, files, nb in chosen:
        iid = r["instance_id"]
        d = os.path.join(REPOS, iid)
        sha = r["base_commit"].strip()
        if not os.path.isdir(os.path.join(d, ".git")):
            seed = os.path.join(REPOS, REPO_SEED[r["repo"]])
            print(f"  copying object store for {iid[:44]} ...")
            shutil.copytree(os.path.join(seed, ".git"), os.path.join(d, ".git"))
            git(d, "config", "core.longpaths", "true")
            git(d, "reset", "-q", "--hard")
        head = git(d, "rev-parse", "HEAD").stdout.strip()
        if head != sha:
            git(d, "fetch", "-q", "--depth", "1", "origin", sha)
            git(d, "checkout", "-q", "-f", sha)
            head = git(d, "rev-parse", "HEAD").stdout.strip()
        status = "OK " if head == sha else "BAD"
        print(f"  {status} {r['repo']:30s} {sha[:12]} files={len(files)} reqs={nb}")
        if head != sha:
            continue
        out.append(dict(
            instance_id=iid, repo=r["repo"], base_commit=sha,
            repo_language=r["repo_language"],
            problem_statement=unesc(r["problem_statement"]),
            requirements=unesc(r["requirements"]),
            interface=unesc(r["interface"]),
            req_bullets=bullets(r["requirements"]),
            gold_files=files, gold_patch=r["patch"],
            fail_to_pass=r["fail_to_pass"], dockerhub_tag=r["dockerhub_tag"]))
    return out


def interleave(out):
    """Round-robin the final list across repos.

    The run carries a wall-clock deadline and stops between instances, so any
    PREFIX of this list has to be a fair sample. Grouped by repo, a run cut
    short would report seven ansible, seven openlibrary and nothing else.
    """
    byrepo = {}
    for r in out:
        byrepo.setdefault(r["repo"], []).append(r)
    order, i = [], 0
    while any(byrepo.values()):
        for repo in sorted(byrepo):
            if byrepo[repo]:
                order.append(byrepo[repo].pop(0))
        i += 1
    return order


if __name__ == "__main__":
    print("picking instances (on task shape only)")
    chosen = pick()
    print("\nlaying out checkouts")
    out = interleave(setup(chosen))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT} with {len(out)} instances")
