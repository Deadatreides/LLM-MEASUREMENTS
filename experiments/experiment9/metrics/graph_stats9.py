"""TABLE 1: graph statistics, computed directly from the frozen schema."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))
from schema9 import CLAIMS, DEPENDS_ON, DEPENDENCY_EDGES, TASK_IDS


def out_degree():
    reverse = {c: 0 for c in CLAIMS}
    for c, deps in DEPENDS_ON.items():
        for d in deps:
            reverse[d] += 1
    return reverse


def longest_path_depth():
    memo = {}

    def depth(cid):
        if cid in memo:
            return memo[cid]
        deps = DEPENDS_ON.get(cid, [])
        if not deps:
            memo[cid] = 1
        else:
            memo[cid] = 1 + max(depth(d) for d in deps)
        return memo[cid]

    return max(depth(c) for c in CLAIMS)


def main():
    od = out_degree()
    in_deg = {c: len(DEPENDS_ON.get(c, [])) for c in CLAIMS}
    stats = {
        "n_tasks": len(TASK_IDS), "n_claims_per_task": len(CLAIMS), "n_artifacts_per_task": 6,
        "n_dependency_edges_per_task": len(DEPENDENCY_EDGES),
        "out_degree_by_claim": od, "in_degree_by_claim": in_deg,
        "max_branching_factor": max(od.values()), "max_merge_in_degree": max(in_deg.values()),
        "dependency_depth": longest_path_depth(),
        "dependency_types_used": sorted({e["dependency_type"] for e in DEPENDENCY_EDGES}),
        "n_edges_uncertain": sum(1 for e in DEPENDENCY_EDGES if e["confidence"] != "CERTAIN"),
        "total_claims_all_tasks": len(TASK_IDS) * len(CLAIMS),
        "total_dependency_edges_all_tasks": len(TASK_IDS) * len(DEPENDENCY_EDGES),
    }
    with open(os.path.join(os.path.dirname(__file__), "graph_stats9.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
