"""arithmetic_multistep_tasks.py — 30 свежих многошаговых (4-6 шагов)
арифметических задач для эксперимента 12.

Без пошаговой структуры K1/K2 (структурный контекст без/с
подтверждёнными значениями) невыразимы -- однократные задачи
Эксперимента 11 для этого не годятся. Каждый шаг -- именованная
промежуточная величина, вычисленная либо из сырых чисел условия, либо
из УЖЕ вычисленных предыдущих шагов по имени (запрещены опережающие
ссылки). Оракульные значения вычисляются программно (`eval` формулы с
подстановкой уже посчитанных шагов), не вписываются вручную -- та же
дисциплина, что в `experiment11/tasks/arithmetic_tasks.py`.

Формулы (`formula`) используются ТОЛЬКО для вычисления оракула -- сами
по себе они никогда не показываются модели (там сырые числа условия),
структурное описание для K1/K2 строит `context_builder.describe_step`
по AST формулы, заменяя все числовые литералы плейсхолдером.
"""

from __future__ import annotations

# (question, [(step_name, formula), ...]) -- formula ссылается только на
# сырые числа условия и/или уже перечисленные выше имена шагов.
_RAW = [
    (
        "A store sells shirts at $18 each and pants at $32 each. A customer buys 3 shirts "
        "and 2 pants. A 10% sales tax is applied to the subtotal. What is the total amount "
        "paid, in dollars?",
        [("shirts_cost", "3 * 18"), ("pants_cost", "2 * 32"), ("subtotal", "shirts_cost + pants_cost"),
         ("tax", "subtotal * 0.1"), ("total", "subtotal + tax")],
    ),
    (
        "Worker A is paid $16/hour for 6 hours; Worker B is paid $21/hour for 4 hours. "
        "Their combined pay receives a 5% team bonus. What is the total pay including the "
        "bonus, in dollars?",
        [("hoursA_pay", "6 * 16"), ("hoursB_pay", "4 * 21"), ("combined", "hoursA_pay + hoursB_pay"),
         ("bonus", "combined * 0.05"), ("total", "combined + bonus")],
    ),
    (
        "Room 1 is 5m by 7m; Room 2 is 4m by 6m. Paint covers 10 square meters per liter "
        "and costs $25 per liter. How much does painting both rooms cost, in dollars?",
        [("room1_area", "5 * 7"), ("room2_area", "4 * 6"), ("total_area", "room1_area + room2_area"),
         ("paint_needed", "total_area / 10"), ("paint_cost", "paint_needed * 25")],
    ),
    (
        "A car travels 60 mph for 3 hours, then 45 mph for 2 hours, then 30 mph for 1 hour. "
        "What is the total distance travelled, in miles?",
        [("leg1", "60 * 3"), ("leg2", "45 * 2"), ("leg3", "30 * 1"), ("total", "leg1 + leg2 + leg3")],
    ),
    (
        "A recipe needs 250g flour and 80g sugar per batch. You make 4 batches. Each storage "
        "container holds 440g. How many containers are needed for the combined flour and "
        "sugar weight?",
        [("flour_total", "250 * 4"), ("sugar_total", "80 * 4"), ("combined_weight", "flour_total + sugar_total"),
         ("containers_needed", "combined_weight / 440")],
    ),
    (
        "A shop had 340 apples. It sold 128 in the morning and 95 in the afternoon. It then "
        "restocked 50 apples, then donated 20 apples to a food bank. How many apples remain "
        "at the end?",
        [("total_sold", "128 + 95"), ("remaining", "340 - total_sold"), ("restocked", "remaining + 50"),
         ("final", "restocked - 20")],
    ),
    (
        "A tank starts with 200 liters. It fills at 12 L/min and simultaneously drains at "
        "5 L/min, for 15 minutes. What is the final water level, in liters?",
        [("fill_amount", "12 * 15"), ("drain_amount", "5 * 15"), ("net_change", "fill_amount - drain_amount"),
         ("final_level", "200 + net_change")],
    ),
    (
        "A school has 8 classrooms with 24 students each, 3 classrooms with 18 students each, "
        "and 2 classrooms with 30 students each. What is the total number of students?",
        [("classroom_a_total", "8 * 24"), ("classroom_b_total", "3 * 18"), ("classroom_c_total", "2 * 30"),
         ("total_students", "classroom_a_total + classroom_b_total + classroom_c_total")],
    ),
    (
        "An account holds $500 at 4% annual interest, compounded yearly. What is the balance "
        "after 2 years, in dollars?",
        [("interest_year1", "500 * 0.04"), ("balance_year1", "500 + interest_year1"),
         ("interest_year2", "balance_year1 * 0.04"), ("balance_year2", "balance_year1 + interest_year2")],
    ),
    (
        "A factory produces 145 units per day for 12 days. 2% of units are defective and "
        "discarded. Good units sell for $8 each. What is the total revenue, in dollars?",
        [("units_12days", "145 * 12"), ("defects", "units_12days * 0.02"), ("good_units", "units_12days - defects"),
         ("revenue", "good_units * 8")],
    ),
    (
        "A bus company runs 4 buses with 48 seats each. 3 buses are completely full, and the "
        "4th has 27 passengers. How many empty seats are there in total across all 4 buses?",
        [("three_full_buses", "3 * 48"), ("filled_total", "three_full_buses + 27"), ("total_capacity", "4 * 48"),
         ("empty_seats", "total_capacity - filled_total")],
    ),
    (
        "A gardener plants 6 rows of 14 tulip bulbs each, costing $40 total for the bulbs, "
        "and removes 9 that don't grow. Each surviving tulip sells for $3. What is the net "
        "profit, in dollars?",
        [("planted", "6 * 14"), ("after_removal", "planted - 9"), ("revenue", "after_removal * 3"),
         ("net_profit", "revenue - 40")],
    ),
    (
        "A book costs $13 with a $4 discount applied per book. 7 books are bought, plus a 5% "
        "shipping fee on the discounted subtotal. What is the total cost, in dollars?",
        [("price_after_discount", "13 - 4"), ("subtotal", "7 * price_after_discount"),
         ("shipping", "subtotal * 0.05"), ("total", "subtotal + shipping")],
    ),
    (
        "A train travels 220 km in 4 hours at constant speed, then continues at the same "
        "speed for another 7 hours. What is the total distance travelled, in kilometers?",
        [("leg1_distance", "220"), ("speed", "leg1_distance / 4"), ("leg2_distance", "speed * 7"),
         ("total_distance", "leg1_distance + leg2_distance")],
    ),
    (
        "A warehouse ships 85 boxes at $3/box and 60 boxes at $5/box shipping cost. A 10% "
        "bulk discount applies to the total shipping cost. What is the final shipping cost, "
        "in dollars?",
        [("cost_a", "85 * 3"), ("cost_b", "60 * 5"), ("total_cost", "cost_a + cost_b"),
         ("discount", "total_cost * 0.1"), ("final_cost", "total_cost - discount")],
    ),
    (
        "A concert sold 120 tickets at $45 and 80 tickets at $30. A 3% payment processing "
        "fee is deducted from gross revenue. What is the net revenue, in dollars?",
        [("revenue_a", "120 * 45"), ("revenue_b", "80 * 30"), ("gross_revenue", "revenue_a + revenue_b"),
         ("fee", "gross_revenue * 0.03"), ("net_revenue", "gross_revenue - fee")],
    ),
    (
        "A pool holds 8000 liters. A pump removes 250 L/hour while a hose adds 100 L/hour, "
        "for 9 hours. What is the final water level, in liters?",
        [("removed", "250 * 9"), ("added", "100 * 9"), ("net_change", "removed - added"),
         ("final_level", "8000 - net_change")],
    ),
    (
        "A bakery uses 2.5kg sugar and 4kg flour per cake. It makes 11 cakes. Combined "
        "ingredient weight costs $2/kg. What is the total ingredient cost, in dollars?",
        [("sugar_total", "2.5 * 11"), ("flour_total", "4 * 11"), ("combined_weight", "sugar_total + flour_total"),
         ("cost", "combined_weight * 2")],
    ),
    (
        "Team A has 9 members earning a $60 bonus each; Team B has 6 members earning an $85 "
        "bonus each. A 15% tax is deducted from the total bonus pool. What is the net bonus "
        "paid, in dollars?",
        [("teamA_bonus", "9 * 60"), ("teamB_bonus", "6 * 85"), ("gross_total", "teamA_bonus + teamB_bonus"),
         ("tax", "gross_total * 0.15"), ("net_total", "gross_total - tax")],
    ),
    (
        "A field has 4 sides of 34 meters each, fenced at $6/meter. Two gates give a $15 "
        "discount each. What is the total fencing cost, in dollars?",
        [("side_length_total", "4 * 34"), ("perimeter_cost", "side_length_total * 6"),
         ("gate_discount", "15 * 2"), ("total_cost", "perimeter_cost - gate_discount")],
    ),
    (
        "A grocery order: 3 apples at $4 each, 2 loaves of bread at $6 each, 5 cartons of "
        "milk at $2 each. A 5% discount applies to the subtotal. What is the total cost, in "
        "dollars?",
        [("apples_cost", "3 * 4"), ("bread_cost", "2 * 6"), ("milk_cost", "5 * 2"),
         ("subtotal", "apples_cost + bread_cost + milk_cost"), ("discount", "subtotal * 0.05"),
         ("total", "subtotal - discount")],
    ),
    (
        "A car rental costs $45/day for 4 days plus $0.20/mile for 300 miles driven. An 8% "
        "tax applies to the subtotal. What is the total cost, in dollars?",
        [("daily_cost", "45 * 4"), ("mileage_cost", "0.2 * 300"), ("subtotal", "daily_cost + mileage_cost"),
         ("tax", "subtotal * 0.08"), ("total", "subtotal + tax")],
    ),
    (
        "An electricity bill charges $0.10/kWh for the first 200 kWh and $0.15/kWh for the "
        "next 150 kWh, plus a flat $10 service fee. What is the total bill, in dollars?",
        [("tier1_cost", "200 * 0.1"), ("tier2_cost", "150 * 0.15"), ("total_usage_cost", "tier1_cost + tier2_cost"),
         ("total_bill", "total_usage_cost + 10")],
    ),
    (
        "A laptop costs $800 with a 15% discount, then 8% tax on the discounted price, plus "
        "a flat $15 shipping fee. What is the total price, in dollars?",
        [("discount_amount", "800 * 0.15"), ("price_after_discount", "800 - discount_amount"),
         ("tax", "price_after_discount * 0.08"), ("price_with_tax", "price_after_discount + tax"),
         ("total", "price_with_tax + 15")],
    ),
    (
        "A farm collects 15 dozen eggs. 5% of the eggs break. The remaining eggs are sold by "
        "the dozen at $3/dozen. What is the total revenue, in dollars?",
        [("total_eggs", "15 * 12"), ("broken", "total_eggs * 0.05"), ("sellable", "total_eggs - broken"),
         ("dozens_sold", "sellable / 12"), ("revenue", "dozens_sold * 3")],
    ),
    (
        "Parking costs $4/hour for 5 hours, plus a 10% weekend surcharge, minus a flat $3 "
        "loyalty discount. What is the total parking cost, in dollars?",
        [("hourly_cost", "4 * 5"), ("weekend_surcharge", "hourly_cost * 0.1"),
         ("subtotal", "hourly_cost + weekend_surcharge"), ("total", "subtotal - 3")],
    ),
    (
        "A phone plan costs $40/month, plus $10/GB for 2 GB of overage data, plus 7% tax on "
        "the subtotal. What is the total monthly bill, in dollars?",
        [("overage_cost", "2 * 10"), ("subtotal", "40 + overage_cost"), ("tax", "subtotal * 0.07"),
         ("total", "subtotal + tax")],
    ),
    (
        "4 movie tickets at $12 each and 3 snack combos at $7 each. A 10% member discount "
        "applies to the subtotal. What is the total cost, in dollars?",
        [("ticket_cost", "4 * 12"), ("snack_cost", "3 * 7"), ("subtotal", "ticket_cost + snack_cost"),
         ("discount", "subtotal * 0.1"), ("total", "subtotal - discount")],
    ),
    (
        "A runner covers the first 10km at a 6 min/km pace, the next 10km at 6.5 min/km, and "
        "the final 10km at 7 min/km. What is the total time, in minutes?",
        [("split1_time", "10 * 6"), ("split2_time", "10 * 6.5"), ("split3_time", "10 * 7"),
         ("total_time", "split1_time + split2_time + split3_time")],
    ),
    (
        "50 shares are bought at $20 each. A 3% dividend is paid on the share value, then a "
        "1% management fee is deducted from the total. What is the net value, in dollars?",
        [("share_value", "50 * 20"), ("dividend", "share_value * 0.03"), ("total_value", "share_value + dividend"),
         ("fee", "total_value * 0.01"), ("net_value", "total_value - fee")],
    ),
]


def _build() -> dict:
    tasks = {}
    for i, (question, step_specs) in enumerate(_RAW, start=1):
        task_id = f"MSARITH_{i:02d}"
        computed: dict = {}
        steps = []
        for name, formula in step_specs:
            value = float(eval(formula, {"__builtins__": {}}, computed))  # noqa: S307 -- fixed formulas, no user input
            computed[name] = value
            steps.append({"name": name, "formula": formula, "value": value})
        tasks[task_id] = {
            "question": question,
            "steps": steps,
            "final_step_name": steps[-1]["name"],
            "answer": steps[-1]["value"],
        }
    return tasks


TASKS = _build()
TASK_IDS = tuple(TASKS)


def generation_prompt(task: dict) -> str:
    step_names = ", ".join(s["name"] for s in task["steps"])
    return (
        f"{task['question']}\n\n"
        "Solve this step by step. Report EVERY step as its own line in the format "
        "`name = value`, using exactly these step names in this order: "
        f"{step_names}.\n"
        "Output nothing except these lines -- no explanation, no extra text, no other lines."
    )
