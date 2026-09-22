"""
FROZEN PROTOCOL (section 5/23) -- 20 real tasks, all sharing the same
6-node branching DAG (section 1's minimal example):

    C1 (root: frames both categories/branches)
      -> C2 (branch A: states the method for branch A)
           -> C4 (branch A: computes branch A's numeric subtotal)
      -> C3 (branch B: states the method for branch B)
           -> C5 (branch B: computes branch B's numeric subtotal)
    C4 + C5 -> C6 (merge: combines both subtotals into the final answer)

The structure is not forced onto unrelated content -- all 20 tasks are
genuine two-category combined-total word problems (shopping, wages,
interest, area, payroll, ...), a natural real task type that actually
has this shape (branch A and branch B are computed independently from a
shared framing, then summed). Frozen BEFORE any generation or ground-truth
work in this experiment; not retuned afterward.

Every dependency edge is typed (section 3) with an explicit reason, never
"just adjacent":
  C1->C2, C1->C3 : CAUSAL      (the root framing is what tells you branch
                                 A/B's method applies at all)
  C2->C4, C3->C5 : DERIVATION  (the numeric subtotal is derived FROM the
                                 stated method -- a claim about applying it)
  C4->C6, C5->C6 : COMPUTATIONAL (the merge is an arithmetic combination
                                 of both branch subtotals)
None are UNCERTAIN -- for this specific task family the causal/derivation
reading is unambiguous, unlike some cases deliberately tested in
metrics/hidden_false_dependency9.py.
"""

TASKS = {
    "BRANCH_01": {"keywordA": 'shirt', "keywordB": 'pant', "labelA": "shirts", "qtyA": 2, "rateA": 20, "subtotalA": 40, "labelB": "pants", "qtyB": 3, "rateB": 35, "subtotalB": 105, "final_total": 145, "question": "A store sells shirts at $20 each and pants at $35 each. If you buy 2 shirts and 3 pants, what is the total amount paid?"},
    "BRANCH_02": {"keywordA": 'worker a', "keywordB": 'worker b', "labelA": "worker A hours", "qtyA": 5, "rateA": 18, "subtotalA": 90, "labelB": "worker B hours", "qtyB": 7, "rateB": 15, "subtotalB": 105, "final_total": 195, "question": "Worker A is paid $18 per hour and works 5 hours. Worker B is paid $15 per hour and works 7 hours. What is the combined pay for both workers?"},
    "BRANCH_03": {"keywordA": 'account a', "keywordB": 'account b', "labelA": "account A", "qtyA": 300, "rateA": 0.05, "subtotalA": 15.0, "labelB": "account B", "qtyB": 450, "rateB": 0.03, "subtotalB": 13.5, "final_total": 28.5, "question": "Account A holds $300 earning 5% simple annual interest. Account B holds $450 earning 3% simple annual interest. What is the combined interest earned by both accounts in one year?"},
    "BRANCH_04": {"keywordA": 'leg a', "keywordB": 'leg b', "labelA": "leg A miles", "qtyA": 40, "rateA": 0.175, "subtotalA": 7.0, "labelB": "leg B miles", "qtyB": 60, "rateB": 0.14, "subtotalB": 8.4, "final_total": 15.4, "question": "On leg A of a trip, a car uses fuel costing $0.175 per mile over 40 miles. On leg B, it uses fuel costing $0.14 per mile over 60 miles. What is the total fuel cost for the trip?"},
    "BRANCH_05": {"keywordA": 'room a', "keywordB": 'room b', "labelA": "room A area", "qtyA": 20, "rateA": 4, "subtotalA": 80, "labelB": "room B area", "qtyB": 18, "rateB": 4, "subtotalB": 72, "final_total": 152, "question": "Room A has an area of 20 square meters and room B has an area of 18 square meters. Flooring costs $4 per square meter for both rooms. What is the total flooring cost?"},
    "BRANCH_06": {"keywordA": 'apple', "keywordB": 'orange', "labelA": "apples kg", "qtyA": 3, "rateA": 2.5, "subtotalA": 7.5, "labelB": "oranges kg", "qtyB": 2, "rateB": 3, "subtotalB": 6, "final_total": 13.5, "question": "A shopper buys 3 kg of apples at $2.5 per kg and 2 kg of oranges at $3 per kg. What is the total cost?"},
    "BRANCH_07": {"keywordA": 'recipe a', "keywordB": 'recipe b', "labelA": "recipe A flour", "qtyA": 2, "rateA": 250, "subtotalA": 500, "labelB": "recipe B flour", "qtyB": 3, "rateB": 180, "subtotalB": 540, "final_total": 1040, "question": "Recipe A needs 2 batches of 250 grams of flour each. Recipe B needs 3 batches of 180 grams of flour each. What is the total flour needed in grams?"},
    "BRANCH_08": {"keywordA": 'tank a', "keywordB": 'tank b', "labelA": "tank A", "qtyA": 5, "rateA": 12, "subtotalA": 60, "labelB": "tank B", "qtyB": 9, "rateB": 8, "subtotalB": 72, "final_total": 132, "question": "Tank A fills at 12 liters per minute for 5 minutes. Tank B fills at 8 liters per minute for 9 minutes. What is the total water collected from both tanks?"},
    "BRANCH_09": {"keywordA": 'salesperson a', "keywordB": 'salesperson b', "labelA": "salesperson A commission base", "qtyA": 2000, "rateA": 0.08, "subtotalA": 160.0, "labelB": "salesperson B commission base", "qtyB": 3500, "rateB": 0.06, "subtotalB": 210.0, "final_total": 370.0, "question": "Salesperson A sold $2000 worth of goods earning an 8% commission. Salesperson B sold $3500 worth of goods earning a 6% commission. What is the total commission paid to both salespeople?"},
    "BRANCH_10": {"keywordA": 'bookstore a', "keywordB": 'bookstore b', "labelA": "bookstore A", "qtyA": 15, "rateA": 12, "subtotalA": 180, "labelB": "bookstore B", "qtyB": 22, "rateB": 9, "subtotalB": 198, "final_total": 378, "question": "Bookstore A ordered 15 books at $12 each. Bookstore B ordered 22 books at $9 each. What is the total cost of both orders?"},
    "BRANCH_11": {"keywordA": 'vip', "keywordB": 'general', "labelA": "VIP tickets", "qtyA": 4, "rateA": 85, "subtotalA": 340, "labelB": "general tickets", "qtyB": 12, "rateB": 40, "subtotalB": 480, "final_total": 820, "question": "A concert sold 4 VIP tickets at $85 each and 12 general tickets at $40 each. What is the total revenue from ticket sales?"},
    "BRANCH_12": {"keywordA": 'wall a', "keywordB": 'wall b', "labelA": "wall A paint", "qtyA": 20, "rateA": 0.3, "subtotalA": 6.0, "labelB": "wall B paint", "qtyB": 35, "rateB": 0.25, "subtotalB": 8.75, "final_total": 14.75, "question": "Wall A is 20 square meters and needs 0.3 liters of paint per square meter. Wall B is 35 square meters and needs 0.25 liters of paint per square meter. What is the total paint needed in liters?"},
    "BRANCH_13": {"keywordA": 'fund a', "keywordB": 'fund b', "labelA": "fund A growth", "qtyA": 1000, "rateA": 0.07, "subtotalA": 70.0, "labelB": "fund B growth", "qtyB": 1500, "rateB": 0.04, "subtotalB": 60.0, "final_total": 130.0, "question": "Fund A holds $1000 and grew by 7% this year. Fund B holds $1500 and grew by 4% this year. What is the combined growth amount (not including the original principal) from both funds?"},
    "BRANCH_14": {"keywordA": 'cookie', "keywordB": 'brownie', "labelA": "cookies", "qtyA": 40, "rateA": 1.5, "subtotalA": 60.0, "labelB": "brownies", "qtyB": 25, "rateB": 2.25, "subtotalB": 56.25, "final_total": 116.25, "question": "A bake sale sold 40 cookies at $1.5 each and 25 brownies at $2.25 each. What is the total revenue?"},
    "BRANCH_15": {"keywordA": 'bed a', "keywordB": 'bed b', "labelA": "garden bed A", "qtyA": 24, "rateA": 4, "subtotalA": 96, "labelB": "garden bed B", "qtyB": 20, "rateB": 4, "subtotalB": 80, "final_total": 176, "question": "Garden bed A covers 24 square meters and garden bed B covers 20 square meters. Soil costs $4 per square meter for both beds. What is the total soil cost?"},
    "BRANCH_16": {"keywordA": 'department a', "keywordB": 'department b', "labelA": "dept A payroll", "qtyA": 12, "rateA": 50000, "subtotalA": 600000, "labelB": "dept B payroll", "qtyB": 8, "rateB": 65000, "subtotalB": 520000, "final_total": 1120000, "question": "Department A has 12 employees earning an average of $50000 each per year. Department B has 8 employees earning an average of $65000 each per year. What is the combined total payroll for both departments?"},
    "BRANCH_17": {"keywordA": 'car a', "keywordB": 'car b', "labelA": "car A fuel", "qtyA": 500, "rateA": 0.12, "subtotalA": 60.0, "labelB": "car B fuel", "qtyB": 350, "rateB": 0.15, "subtotalB": 52.5, "final_total": 112.5, "question": "Car A travels 500 miles at a fuel cost of $0.12 per mile. Car B travels 350 miles at a fuel cost of $0.15 per mile. What is the total fuel cost for both cars?"},
    "BRANCH_18": {"keywordA": 'morning', "keywordB": 'afternoon', "labelA": "conference morning", "qtyA": 45, "rateA": 30, "subtotalA": 1350, "labelB": "conference afternoon", "qtyB": 60, "rateB": 25, "subtotalB": 1500, "final_total": 2850, "question": "A conference sold 45 morning-session passes at $30 each and 60 afternoon-session passes at $25 each. What is the total revenue?"},
    "BRANCH_19": {"keywordA": 'warehouse a', "keywordB": 'warehouse b', "labelA": "warehouse A boxes", "qtyA": 120, "rateA": 3.2, "subtotalA": 384.0, "labelB": "warehouse B boxes", "qtyB": 95, "rateB": 4.1, "subtotalB": 389.5, "final_total": 773.5, "question": "Warehouse A ships 120 boxes at a shipping cost of $3.2 per box. Warehouse B ships 95 boxes at a shipping cost of $4.1 per box. What is the total shipping cost?"},
    "BRANCH_20": {"keywordA": 'team a', "keywordB": 'team b', "labelA": "team A bonus", "qtyA": 6, "rateA": 450, "subtotalA": 2700, "labelB": "team B bonus", "qtyB": 9, "rateB": 300, "subtotalB": 2700, "final_total": 5400, "question": "Team A has 6 members each receiving a $450 bonus. Team B has 9 members each receiving a $300 bonus. What is the total bonus paid across both teams?"},
}

TASK_IDS = list(TASKS.keys())

ARTIFACTS = ["A1_ROOT", "A2_METHOD_A", "A3_METHOD_B", "A4_SUBTOTAL_A", "A5_SUBTOTAL_B", "A6_FINAL_TOTAL"]
CLAIMS = ["C1_ROOT", "C2_METHOD_A", "C3_METHOD_B", "C4_SUBTOTAL_A", "C5_SUBTOTAL_B", "C6_FINAL_TOTAL"]
CLAIM_ARTIFACT = {"C1_ROOT": "A1_ROOT", "C2_METHOD_A": "A2_METHOD_A", "C3_METHOD_B": "A3_METHOD_B",
                   "C4_SUBTOTAL_A": "A4_SUBTOTAL_A", "C5_SUBTOTAL_B": "A5_SUBTOTAL_B", "C6_FINAL_TOTAL": "A6_FINAL_TOTAL"}

DEPENDS_ON = {
    "C1_ROOT": [],
    "C2_METHOD_A": ["C1_ROOT"],
    "C3_METHOD_B": ["C1_ROOT"],
    "C4_SUBTOTAL_A": ["C2_METHOD_A"],
    "C5_SUBTOTAL_B": ["C3_METHOD_B"],
    "C6_FINAL_TOTAL": ["C4_SUBTOTAL_A", "C5_SUBTOTAL_B"],
}

DEPENDENCY_EDGES = [
    {"source": "C1_ROOT", "target": "C2_METHOD_A", "dependency_type": "CAUSAL", "evidence": "root framing determines that branch A's category/method applies", "confidence": "CERTAIN"},
    {"source": "C1_ROOT", "target": "C3_METHOD_B", "dependency_type": "CAUSAL", "evidence": "root framing determines that branch B's category/method applies", "confidence": "CERTAIN"},
    {"source": "C2_METHOD_A", "target": "C4_SUBTOTAL_A", "dependency_type": "DERIVATION", "evidence": "the numeric subtotal is a direct application of the stated method", "confidence": "CERTAIN"},
    {"source": "C3_METHOD_B", "target": "C5_SUBTOTAL_B", "dependency_type": "DERIVATION", "evidence": "the numeric subtotal is a direct application of the stated method", "confidence": "CERTAIN"},
    {"source": "C4_SUBTOTAL_A", "target": "C6_FINAL_TOTAL", "dependency_type": "COMPUTATIONAL", "evidence": "the final total is an arithmetic combination (sum) of both subtotals", "confidence": "CERTAIN"},
    {"source": "C5_SUBTOTAL_B", "target": "C6_FINAL_TOTAL", "dependency_type": "COMPUTATIONAL", "evidence": "the final total is an arithmetic combination (sum) of both subtotals", "confidence": "CERTAIN"},
]

# Independent branches for section 11's test: everything reachable only
# through C2/C4 is "branch A", only through C3/C5 is "branch B".
BRANCH_A_CLAIMS = ["C2_METHOD_A", "C4_SUBTOTAL_A"]
BRANCH_B_CLAIMS = ["C3_METHOD_B", "C5_SUBTOTAL_B"]
SHARED_CLAIMS = ["C1_ROOT"]
MERGE_CLAIMS = ["C6_FINAL_TOTAL"]
