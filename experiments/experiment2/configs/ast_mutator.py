"""
Purely mechanical mutation generator (ast-based, no LLM involved). Applies
standard mutation-testing operators to a reference implementation's source:
  - comparison operator swap (<, <=, >, >=, ==, != each swapped for a
    "close" wrong operator)
  - arithmetic operator swap (+ <-> -)
  - boolean constant flip (True <-> False)
  - integer constant off-by-one (n -> n+1, n -> n-1)

One mutation per mutant (classic mutation-testing convention: each mutant
differs from the original by exactly one edit), so mutation score reflects
how many *individual* injected bugs a test suite catches.
"""
import ast
import copy

CMP_SWAP = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
ARITH_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add}


class _Finder(ast.NodeVisitor):
    def __init__(self, target_type, lineno, col_offset):
        self.target_type = target_type
        self.lineno = lineno
        self.col_offset = col_offset
        self.found = None

    def generic_visit(self, node):
        if (
            self.found is None
            and isinstance(node, self.target_type)
            and getattr(node, "lineno", None) == self.lineno
            and getattr(node, "col_offset", None) == self.col_offset
        ):
            self.found = node
        super().generic_visit(node)


def _find_node(tree, node_type, lineno, col_offset):
    f = _Finder(node_type, lineno, col_offset)
    f.visit(tree)
    return f.found


def generate_mutants(source, max_mutants=40):
    """Returns a list of {mutation_id, description, mutant_source}."""
    tree = ast.parse(source)
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP_SWAP:
            sites.append(("cmp", node.lineno, node.col_offset, type(node.ops[0])))
        elif isinstance(node, ast.BinOp) and type(node.op) in ARITH_SWAP:
            sites.append(("arith", node.lineno, node.col_offset, type(node.op)))
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            sites.append(("bool", node.lineno, node.col_offset, node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            sites.append(("int_plus", node.lineno, node.col_offset, node.value))
            sites.append(("int_minus", node.lineno, node.col_offset, node.value))

    mutants = []
    for i, (kind, lineno, col, extra) in enumerate(sites[:max_mutants]):
        tree_copy = copy.deepcopy(tree)
        desc = None
        if kind == "cmp":
            target = _find_node(tree_copy, ast.Compare, lineno, col)
            if target is None:
                continue
            old_op = type(target.ops[0])
            new_op_cls = CMP_SWAP[old_op]
            target.ops[0] = new_op_cls()
            desc = f"compare {old_op.__name__}->{new_op_cls.__name__} @L{lineno}"
        elif kind == "arith":
            target = _find_node(tree_copy, ast.BinOp, lineno, col)
            if target is None:
                continue
            old_op = type(target.op)
            new_op_cls = ARITH_SWAP[old_op]
            target.op = new_op_cls()
            desc = f"arith {old_op.__name__}->{new_op_cls.__name__} @L{lineno}"
        elif kind == "bool":
            target = _find_node(tree_copy, ast.Constant, lineno, col)
            if target is None:
                continue
            target.value = not extra
            desc = f"bool {extra}->{not extra} @L{lineno}"
        elif kind in ("int_plus", "int_minus"):
            target = _find_node(tree_copy, ast.Constant, lineno, col)
            if target is None:
                continue
            delta = 1 if kind == "int_plus" else -1
            target.value = extra + delta
            desc = f"const {extra}->{extra + delta} @L{lineno}"

        ast.fix_missing_locations(tree_copy)
        try:
            mutant_source = ast.unparse(tree_copy)
        except Exception:
            continue
        if mutant_source.strip() == source.strip():
            continue  # mutation had no textual effect (shouldn't normally happen)
        mutants.append(
            {"mutation_id": f"MUT_{i:03d}", "description": desc, "mutant_source": mutant_source}
        )
    return mutants
