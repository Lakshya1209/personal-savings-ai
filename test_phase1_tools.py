"""
test_phase1_tools.py — Unit test script to verify state.py and all tools/ against sample_data.csv
Verifies exact mathematical outputs with zero LLM dependence.
"""

import pandas as pd
from state import AnalysisState, ExpenseAgentOutput, BudgetAgentOutput, GoalAgentOutput
from tools.expense_tools import category_totals, essential_vs_discretionary, detect_recurring
from tools.budget_tools import calculate_savings_rate, calculate_discretionary_capacity
from tools.goal_tools import calculate_required_monthly_saving, project_completion_date
from tools.savings_tools import simulate_category_reduction, combine_strategies
from tools.safety_tools import check_essential_floor, check_emergency_buffer

def run_tests():
    print("=" * 60)
    print("RUNNING PHASE 1 TOOL VERIFICATION AGAINST sample_data.csv")
    print("=" * 60)
    
    df = pd.read_csv("data/sample_data.csv")
    print(f"Loaded dataset with {len(df)} rows and {df['user_id'].nunique()} unique users.\n")
    
    # Test User 101 (Rahul Verma)
    u101_df = df[df["user_id"] == 101]
    u101_profile = u101_df.iloc[0].to_dict()
    
    print("--- 1. Testing expense_tools ---")
    cat_tot = category_totals(u101_df)
    print(f"User 101 Average Category Totals: {cat_tot}")
    assert "rent" in cat_tot and cat_tot["rent"] == 22000.0, "Rent total mismatch"
    assert "dining_out" in cat_tot, "dining_out missing"
    
    split = essential_vs_discretionary(cat_tot)
    print(f"Essential vs Discretionary Split: {split}")
    assert split["essential_total"] > 0 and split["discretionary_total"] > 0
    assert abs(split["total_expenses"] - (split["essential_total"] + split["discretionary_total"])) < 0.05
    
    recurring = detect_recurring(u101_df)
    print(f"Detected Recurring Items: {[r['item'] for r in recurring]}")
    
    print("\n--- 2. Testing budget_tools ---")
    avg_income = float(u101_df["monthly_income"].mean())
    avg_expense = split["total_expenses"]
    budget = calculate_savings_rate(avg_income, avg_expense)
    print(f"Savings Rate Analysis: {budget}")
    assert budget["current_monthly_savings"] == round(avg_income - avg_expense, 2)
    
    disc_cap = calculate_discretionary_capacity(split["discretionary_total"])
    print(f"Discretionary Capacity: {disc_cap}")
    assert disc_cap["safe_to_cut_capacity"] == round(split["discretionary_total"] * 0.5, 2)
    
    print("\n--- 3. Testing goal_tools ---")
    goal_res = calculate_required_monthly_saving(
        goal_amount=u101_profile["goal_amount"],
        deadline_months=u101_profile["goal_deadline_months"],
        current_savings=u101_profile["existing_savings"],
        current_monthly_saving=budget["current_monthly_savings"]
    )
    print(f"Goal Calculation: {goal_res}")
    # Net target: 200,000 - 150,000 = 50,000. 12 months => 4166.67/mo
    assert goal_res["net_target_to_accumulate"] == 50000.0
    assert goal_res["required_monthly_saving"] == 4166.67
    
    proj = project_completion_date(budget["current_monthly_savings"], u101_profile["goal_amount"], u101_profile["existing_savings"])
    print(f"Projected Completion: {proj}")
    
    print("\n--- 4. Testing savings_tools ---")
    sim1 = simulate_category_reduction("dining_out", cat_tot["dining_out"], 30.0)
    sim2 = simulate_category_reduction("shopping", cat_tot["shopping"], 25.0)
    print(f"Simulate Dining Out 30%: {sim1}")
    print(f"Simulate Shopping 25%: {sim2}")
    combo = combine_strategies([sim1, sim2])
    print(f"Combined Strategies: {combo}")
    assert combo["total_monthly_saving"] == round(sim1["monthly_saving"] + sim2["monthly_saving"], 2)
    
    print("\n--- 5. Testing safety_tools (Approvals & Rejection Demo) ---")
    # Test safe approval
    app_check = check_essential_floor(
        strategy_name="Moderate Dining Out Cut",
        category="dining_out",
        cut_amount=sim1["monthly_saving"],
        current_category_amount=cat_tot["dining_out"],
        reduction_percent=30.0,
        discretionary_total=split["discretionary_total"],
        total_proposed_discretionary_cuts=sim1["monthly_saving"]
    )
    print(f"Safe Strategy Decision: {app_check['decision']} - {app_check['reason']}")
    assert app_check["is_safe"] is True
    
    # Test protected essential rejection (rent cut)
    rent_check = check_essential_floor(
        strategy_name="Cut Rent by 20%",
        category="rent",
        cut_amount=4400.0,
        current_category_amount=22000.0,
        reduction_percent=20.0
    )
    print(f"Rent Cut Decision: {rent_check['decision']} - {rent_check['reason']}")
    assert rent_check["is_safe"] is False
    assert rent_check["decision"] == "rejected"
    
    # Test over-capacity rejection for User 103 (Amit Patel - Demo Test Case)
    u103_df = df[df["user_id"] == 103]
    u103_totals = category_totals(u103_df)
    u103_split = essential_vs_discretionary(u103_totals)
    print(f"\nUser 103 Discretionary Total: Rs. {u103_split['discretionary_total']}")
    
    # An aggressive attempt to cut ₹12,000 from a ₹10,700 discretionary pool
    aggressive_check = check_essential_floor(
        strategy_name="Extreme Lifestyle Compression",
        category="dining_out",
        cut_amount=12000.0,
        current_category_amount=u103_totals["dining_out"],
        reduction_percent=90.0,
        discretionary_total=u103_split["discretionary_total"],
        total_proposed_discretionary_cuts=12000.0
    )
    print(f"Aggressive Cut Decision: {aggressive_check['decision']} - {aggressive_check['reason']}")
    assert aggressive_check["is_safe"] is False
    assert aggressive_check["decision"] == "rejected"
    
    print("\n" + "=" * 60)
    print("ALL PHASE 1 TOOLS PASSED VERIFICATION WITH ZERO ARITHMETIC ERRORS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
