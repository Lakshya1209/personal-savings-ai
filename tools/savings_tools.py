"""
tools/savings_tools.py — Pure Python strategy simulation and combination tools.
No LLM arithmetic.
"""

from typing import Dict, Any, List


def simulate_category_reduction(
    category: str,
    current_amount: float,
    percent_reduction: float
) -> Dict[str, Any]:
    """
    Simulates reducing a single expense category by a given percentage.
    monthly_saving = current_amount * (percent_reduction / 100)
    new_amount = current_amount - monthly_saving
    """
    current_amount = max(0.0, float(current_amount))
    percent_reduction = max(0.0, min(100.0, float(percent_reduction)))
    
    monthly_saving = round(current_amount * (percent_reduction / 100.0), 2)
    new_amount = round(current_amount - monthly_saving, 2)
    annual_saving = round(monthly_saving * 12.0, 2)

    return {
        "category": category,
        "original_amount": current_amount,
        "percent_reduction": percent_reduction,
        "monthly_saving": monthly_saving,
        "new_amount": new_amount,
        "annual_saving": annual_saving,
    }


def combine_strategies(reductions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combines a list of simulated category reductions into a unified composite strategy.
    Calculates total monthly saving, annualized saving, and per-category breakdown.
    """
    total_monthly = 0.0
    breakdown = []

    for item in reductions:
        cat = item.get("category", "unknown")
        saving = float(item.get("monthly_saving", 0.0))
        pct = float(item.get("percent_reduction", 0.0))
        orig = float(item.get("original_amount", 0.0))
        
        total_monthly += saving
        breakdown.append({
            "category": cat,
            "monthly_saving": round(saving, 2),
            "percent_reduction": pct,
            "original_amount": orig,
        })

    total_monthly = round(total_monthly, 2)
    annual_saving = round(total_monthly * 12.0, 2)

    return {
        "total_monthly_saving": total_monthly,
        "annual_saving": annual_saving,
        "item_count": len(reductions),
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    r1 = simulate_category_reduction("dining_out", 8000, 30)
    r2 = simulate_category_reduction("subscriptions", 2500, 50)
    combo = combine_strategies([r1, r2])
    print("Reduction 1:", r1)
    print("Combination:", combo)
