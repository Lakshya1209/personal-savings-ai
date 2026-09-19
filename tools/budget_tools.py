"""
tools/budget_tools.py — Pure Python budget and capacity calculation tools.
No LLM arithmetic.
"""

from typing import Dict, Any


def calculate_savings_rate(income: float, expenses: float) -> Dict[str, float]:
    """
    Computes current monthly savings and savings rate percentage.
    savings = income - expenses
    savings_rate = (savings / income) * 100
    """
    income = float(income)
    expenses = float(expenses)
    savings = round(income - expenses, 2)
    
    if income > 0:
        rate = round((savings / income) * 100, 2)
    else:
        rate = 0.0

    if savings < 0:
        status = "Deficit"
    elif rate < 15.0:
        status = "Strained"
    elif rate < 30.0:
        status = "Moderate"
    else:
        status = "Healthy"

    return {
        "monthly_income": income,
        "monthly_expenses": expenses,
        "current_monthly_savings": savings,
        "savings_rate_percent": rate,
        "budget_status": status,
    }


def calculate_discretionary_capacity(
    discretionary_total: float, 
    safe_cut_ratio: float = 0.50
) -> Dict[str, float]:
    """
    Calculates the realistic, sustainable capacity to trim from discretionary spending.
    By default, up to 50% of discretionary spending is deemed safe-to-cut without lifestyle collapse.
    Cutting more than 60% of discretionary is considered aggressive/fragile.
    """
    discretionary_total = float(discretionary_total)
    safe_capacity = round(discretionary_total * safe_cut_ratio, 2)
    aggressive_capacity = round(discretionary_total * 0.75, 2)
    minimum_retained = round(discretionary_total - safe_capacity, 2)

    return {
        "discretionary_total": discretionary_total,
        "safe_to_cut_capacity": safe_capacity,
        "aggressive_cut_capacity": aggressive_capacity,
        "safe_remaining_discretionary": minimum_retained,
        "recommended_max_cut_ratio": safe_cut_ratio,
    }


if __name__ == "__main__":
    b = calculate_savings_rate(85000, 62000)
    c = calculate_discretionary_capacity(25000)
    print("Budget:", b)
    print("Capacity:", c)
