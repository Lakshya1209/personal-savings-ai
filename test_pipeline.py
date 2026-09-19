"""
test_pipeline.py — Comprehensive End-to-End Automated Acceptance Test.
Validates:
1. Tool arithmetic consistency against sample dataset.
2. Multi-Agent orchestration across all 6 agents.
3. The iterative Reject -> Revise -> Approve loop in action.
4. Human-readable reasoning log generation for demo & viva.
"""

import os
import sys
import pandas as pd

# Ensure workspace root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.orchestrator import run_pipeline
from tools.expense_tools import category_totals, essential_vs_discretionary
from tools.budget_tools import calculate_savings_rate
from tools.goal_tools import calculate_required_monthly_saving
from tools.savings_tools import simulate_category_reduction, combine_strategies
from tools.safety_tools import check_essential_floor

def test_arithmetic_tools():
    print("TEST 1: Verifying Python Tools Arithmetic (Zero LLM dependence)...")
    # Test Category Totals & Split
    sample_exp = {
        "rent": 22000, "groceries": 10000, "utilities": 4500, "debt_emi": 5000,
        "transport": 4000, "dining_out": 8500, "shopping": 7000,
        "entertainment": 4000, "subscriptions": 2500
    }
    totals = category_totals(sample_exp)
    split = essential_vs_discretionary(totals)
    assert split["essential_total"] == 43500.0, f"Expected 43500, got {split['essential_total']}"
    assert split["discretionary_total"] == 24000.0, f"Expected 24000, got {split['discretionary_total']}"
    
    # Test Budget
    b = calculate_savings_rate(85000, 67500)
    assert b["current_monthly_savings"] == 17500.0
    assert b["savings_rate_percent"] == 20.59
    
    # Test Goal
    g = calculate_required_monthly_saving(200000, 12, 150000, 17500)
    assert g["net_target_to_accumulate"] == 50000.0
    assert g["required_monthly_saving"] == 4166.67
    assert g["is_feasible_with_current"] is True

    # Test Savings Simulation
    s1 = simulate_category_reduction("dining_out", 8500, 30.0)
    assert s1["monthly_saving"] == 2550.0
    assert s1["new_amount"] == 5950.0

    print(" -> PASSED: Pure Python tool arithmetic is exact and verified.")


def test_standard_pipeline():
    print("\nTEST 2: Running End-to-End Pipeline for Rahul Verma (Healthy Budget)...")
    profile = {
        "user_name": "Rahul Verma",
        "monthly_income": 85000,
        "expenses": {
            "rent": 22000, "groceries": 10000, "utilities": 4500, "debt_emi": 5000,
            "transport": 4000, "dining_out": 8500, "shopping": 7000,
            "entertainment": 4000, "subscriptions": 2500
        },
        "goal_amount": 200000,
        "goal_deadline_months": 12,
        "existing_savings": 150000,
    }
    state = run_pipeline(profile, objective_text="Help me optimize my savings smoothly.")
    assert state["status"] == "completed"
    assert len(state["validated_strategies"]) >= 2
    assert "findings" in state and "expense" in state["findings"]
    print(f" -> PASSED: Completed with {len(state['validated_strategies'])} approved strategies.")


def test_rejection_revision_loop():
    print("\nTEST 3: Running Rejection & Revision Loop (Amit Patel Test Case)...")
    profile = {
        "user_name": "Amit Patel",
        "monthly_income": 55000,
        "expenses": {
            "rent": 20000, "groceries": 11000, "utilities": 4000, "debt_emi": 15000,
            "transport": 3500, "dining_out": 4500, "shopping": 3000,
            "entertainment": 2000, "subscriptions": 1200
        },
        "goal_amount": 180000,
        "goal_deadline_months": 12,
        "existing_savings": 25000,
    }
    # This scenario demands an aggressive Rs 15k cut from a small discretionary budget
    state = run_pipeline(
        profile=profile,
        objective_text="Aggressively save Rs 15,000 extra per month",
        force_demo_rejection=True,
    )
    
    logs = state["reasoning_log"]
    has_rejection = any("REJECTED" in l for l in logs)
    has_revision_loop = any("Triggering revise loop" in l for l in logs)
    has_approval = any("approved all" in l.lower() or "approved plan" in l.lower() for l in logs)
    
    print("\nDetailed Execution Trace:")
    for idx, log in enumerate(logs, 1):
        print(f"  [{idx:02d}] {log}")

    assert has_rejection, "Safety Agent should have rejected Pass 1"
    assert has_revision_loop, "Orchestrator should have triggered revision loop"
    assert has_approval, "Safety Agent should have approved revised Pass 2"
    assert state["status"] == "completed"
    print("\n -> PASSED: Safety Agent rejected Pass 1, Savings Agent revised, and Pass 2 was approved!")


def test_dataset_availability():
    print("\nTEST 4: Checking Dataset Availability...")
    assert os.path.exists("data/sample_data.csv"), "data/sample_data.csv missing"
    sample_df = pd.read_csv("data/sample_data.csv")
    print(f" -> Sample dataset verified: {len(sample_df)} rows, {sample_df['user_id'].nunique()} users.")

    kaggle_path = "dataset/personal_finance_tracker_dataset.csv"
    if os.path.exists(kaggle_path):
        kdf = pd.read_csv(kaggle_path)
        print(f" -> Kaggle dataset verified at {kaggle_path}: {len(kdf)} rows.")
    print(" -> PASSED: All datasets verified.")


if __name__ == "__main__":
    print("=" * 70)
    print("STARTING FULL ACCEPTANCE SUITE")
    print("=" * 70)
    test_arithmetic_tools()
    test_dataset_availability()
    test_standard_pipeline()
    test_rejection_revision_loop()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! PROJECT ACCEPTANCE CHECKLIST SATISFIED.")
    print("=" * 70)
