"""
tools/expense_tools.py — Pure Python expense analysis tools.
No LLM arithmetic.
"""

from typing import Dict, Any, List, Union
import pandas as pd

ESSENTIAL_CATEGORIES = {
    "rent", "mortgage", "groceries", "utilities", "debt_emi", 
    "loan_payment", "insurance", "healthcare", "essential_spending"
}

DISCRETIONARY_CATEGORIES = {
    "dining_out", "food_delivery", "shopping", "entertainment", 
    "subscriptions", "travel", "discretionary_spending"
}

# Transport can have an essential baseline, but in typical categorization can be split or placed conservatively
ESSENTIAL_BASELINE_MINIMUMS = {
    "groceries": 3000.0,  # absolute subsistence minimum for food
    "utilities": 1000.0,  # power, water, connectivity floor
}

def category_totals(data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, float]:
    """
    Computes aggregated expense amounts across individual expense categories.
    Works on either a single user profile dictionary or a pandas DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        # Average across rows if multiple months exist
        res = {}
        cols_to_skip = {
            "user_id", "user_name", "archetype", "month", "date", "monthly_income", 
            "savings_rate", "budget_goal", "credit_score", "actual_savings", "savings_goal_met",
            "existing_savings", "goal_amount", "goal_deadline_months", "transaction_count", "fraud_flag"
        }
        for col in data.columns:
            if col not in cols_to_skip and pd.api.types.is_numeric_dtype(data[col]):
                # If explicit total columns exist, exclude them from category breakdown
                if col not in ("monthly_expense_total", "essential_spending", "discretionary_spending"):
                    res[col] = round(float(data[col].mean()), 2)
        return res
    
    elif isinstance(data, dict):
        # Extract categories from profile dict
        categories = {}
        # If nested "expenses" dict is provided
        raw_expenses = data.get("expenses", data)
        for k, v in raw_expenses.items():
            if k in (
                "rent", "groceries", "utilities", "debt_emi", "loan_payment",
                "transport", "dining_out", "shopping", "entertainment", 
                "subscriptions", "healthcare", "insurance"
            ):
                try:
                    categories[k] = round(float(v), 2)
                except (ValueError, TypeError):
                    continue
        return categories
    else:
        return {}


def essential_vs_discretionary(cat_totals: Dict[str, float]) -> Dict[str, float]:
    """
    Partitions category totals into essential vs discretionary totals.
    Essential: rent/mortgage, baseline groceries, utilities, debt/loans, healthcare, insurance.
    Discretionary: dining out, shopping, entertainment, subscriptions, transport surplus.
    """
    essential_sum = 0.0
    discretionary_sum = 0.0

    for cat, amount in cat_totals.items():
        cat_lower = cat.lower()
        if cat_lower in ESSENTIAL_CATEGORIES:
            essential_sum += amount
        elif cat_lower in DISCRETIONARY_CATEGORIES:
            discretionary_sum += amount
        elif "rent" in cat_lower or "mortgage" in cat_lower or "debt" in cat_lower or "loan" in cat_lower or "utilit" in cat_lower:
            essential_sum += amount
        elif "dining" in cat_lower or "shop" in cat_lower or "entertain" in cat_lower or "subscri" in cat_lower:
            discretionary_sum += amount
        else:
            # Transport / other defaults to 50% essential commute, 50% discretionary
            if "transport" in cat_lower:
                essential_sum += round(amount * 0.5, 2)
                discretionary_sum += round(amount * 0.5, 2)
            else:
                discretionary_sum += amount

    total = round(essential_sum + discretionary_sum, 2)
    return {
        "essential_total": round(essential_sum, 2),
        "discretionary_total": round(discretionary_sum, 2),
        "total_expenses": total,
        "essential_ratio": round((essential_sum / total * 100), 2) if total > 0 else 0.0,
        "discretionary_ratio": round((discretionary_sum / total * 100), 2) if total > 0 else 0.0,
    }


def detect_recurring(data: Union[Dict[str, Any], pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Detects recurring fixed monthly commitments such as rent, EMI/loans, and subscriptions.
    """
    recurring_items = []
    
    if isinstance(data, pd.DataFrame):
        # Look for low variance columns
        check_cols = ["rent", "debt_emi", "loan_payment", "subscriptions", "utilities"]
        for col in check_cols:
            if col in data.columns:
                mean_val = float(data[col].mean())
                std_val = float(data[col].std()) if len(data) > 1 else 0.0
                if mean_val > 0 and (std_val / mean_val < 0.15 or len(data) <= 1):
                    recurring_items.append({
                        "item": col,
                        "monthly_cost": round(mean_val, 2),
                        "type": "Fixed Recurring",
                        "variance": round(std_val, 2)
                    })
    elif isinstance(data, dict):
        raw = data.get("expenses", data)
        for key in ["rent", "debt_emi", "loan_payment", "subscriptions"]:
            if key in raw and float(raw[key]) > 0:
                recurring_items.append({
                    "item": key,
                    "monthly_cost": round(float(raw[key]), 2),
                    "type": "Fixed Recurring",
                    "variance": 0.0
                })
    return recurring_items


if __name__ == "__main__":
    test_data = {
        "rent": 20000,
        "groceries": 10000,
        "utilities": 4000,
        "debt_emi": 5000,
        "transport": 4000,
        "dining_out": 8000,
        "shopping": 6000,
        "entertainment": 3000,
        "subscriptions": 2000,
    }
    totals = category_totals(test_data)
    split = essential_vs_discretionary(totals)
    recurring = detect_recurring(test_data)
    print("Totals:", totals)
    print("Split:", split)
    print("Recurring:", recurring)
