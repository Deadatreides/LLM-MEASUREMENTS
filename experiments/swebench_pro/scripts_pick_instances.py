"""Pick 3 pilot instances for SWE-bench Pro + dump them as JSON."""
import json, re, os, collections
import pyarrow.parquet as pq

DATA = r"<PROJECT_ROOT>\trace-probe\swebench_pro\data\swebench_pro_test.parquet"
OUT = r"<PROJECT_ROOT>\trace-probe\swebench_pro\data\pilot3.json"

rows = pq.read_table(DATA).to_pylist()
fre = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
hre = re.compile(r"^@@ ", re.M)


def unesc(s):
    """The text fields are stored as python-repr-ish strings with literal \\n."""
    s = s or ""
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")


def req_bullets(s):
    return [b.strip() for b in unesc(s).split("\n") if b.strip().startswith("- ")]


cand = []
for r in rows:
    p = r["patch"] or ""
    files = sorted(set(f for f, _ in fre.findall(p)))
    nb = len(req_bullets(r["requirements"]))
    cand.append(dict(r=r, files=files, hunks=len(hre.findall(p)), nreq=nb))

# pilot criteria: python, 2-4 files, 4-12 hunks, 4-10 requirement bullets (so blocks are meaningful)
sel = [c for c in cand
       if c["r"]["repo_language"] == "python"
       and 2 <= len(c["files"]) <= 4
       and 4 <= c["hunks"] <= 12
       and 4 <= c["nreq"] <= 10]

print(f"candidates: {len(sel)}")
print(collections.Counter(c["r"]["repo"] for c in sel).most_common())

# take one from each of the distinct python repos, prefer median-sized
by_repo = collections.defaultdict(list)
for c in sel:
    by_repo[c["r"]["repo"]].append(c)

pilot = []
for repo in sorted(by_repo):
    lst = sorted(by_repo[repo], key=lambda c: (c["hunks"], len(c["files"])))
    pilot.append(lst[len(lst) // 2])
pilot = pilot[:3]

out = []
for c in pilot:
    r = c["r"]
    out.append(dict(
        instance_id=r["instance_id"], repo=r["repo"], base_commit=r["base_commit"],
        problem_statement=unesc(r["problem_statement"]),
        requirements=unesc(r["requirements"]),
        interface=unesc(r["interface"]),
        gold_files=c["files"], gold_hunks=c["hunks"],
        req_bullets=req_bullets(r["requirements"]),
        gold_patch=r["patch"],
        fail_to_pass=r["fail_to_pass"],
        dockerhub_tag=r["dockerhub_tag"],
    ))
    print(f"\n{r['instance_id'][:70]}")
    print(f"  repo={r['repo']}  commit={r['base_commit'][:12]}  files={len(c['files'])}  hunks={c['hunks']}  reqs={c['nreq']}")
    for f in c["files"]:
        print(f"    {f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(f"\nwrote {OUT}")
