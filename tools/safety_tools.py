"""
tools/safety_tools.py — Pure Python safety verification and floor checking tools.
No LLM arithmetic.
"""

from typing import Dict, Any, List

PROTECTED_ESSENTIALS = {
    "rent": "Rent and mortgage obligations are strictly protected and cannot be reduced.",
    "mortgage": "Rent and mortgage obligations are strictly protected and cannot be reduced.",
    "debt_emi": "Debt/EMI repayments are legally binding; reducing them requires formal refinancing.",
    "loan_payment": "Debt/loan payments cannot be cut arbitrarily without defaulting.",
    "insurance": "Core health and life insurance must not be compromised.",
}

CATEGORY_FLOORS = {
    "groceries": 3000.0,   # Absolute minimum nutritional floor
    "utilities": 1000.0,   # Essential electricity, water, connectivity floor
}

MAX_SAFE_DISCRETIONARY_CUT_RATIO = 0.65  # Max 65% cut in discretionary spending before lifestyle breakdown


def check_essential_floor(
    strategy_name: str,
    category: str,
    cut_amount: float,
    current_category_amount: float,
    reduction_percent: float,
    discretionary_total: float = 0.0,
    total_proposed_discretionary_cuts: float = 0.0,
) -> Dict[str, Any]:
    """
    Validates whether a proposed savings strategy respects essential expense floors
    and sustainable lifestyle bounds.
    """
    category_clean = category.lower().strip()
    cut_amount = float(cut_amount)
    current_category_amount = float(current_category_amount)
    reduction_percent = float(reduction_percent)
    
    # 1. Protected non-negotiable categories
    for protected, reason in PROTECTED_ESSENTIALS.items():
        if protected in category_clean and cut_amount > 0:
            return {
                "is_safe": False,
                "decision": "rejected",
                "violation_type": "ESSENTIAL_PROTECTION_VIOLATION",
                "reason": f"Safety Agent Rejected [{strategy_name}]: {reason}",
                "suggested_fix": "Exclude rent and loan obligations from savings plans.",
            }

    # 2. Hard subsistence floors for groceries and utilities
    if "groceries" in category_clean or "food_essentials" in category_clean:
        floor = CATEGORY_FLOORS["groceries"]
        remaining = current_category_amount - cut_amount
        if remaining < floor or reduction_percent > 25.0:
            return {
                "is_safe": False,
                "decision": "rejected",
                "violation_type": "SUBSISTENCE_FLOOR_VIOLATION",
                "reason": f"Safety Agent Rejected [{strategy_name}]: Leaves groceries at Rs. {remaining:,.2f}, breaching nutritional floor of Rs. {floor:,.2f}.",
                "suggested_fix": f"Limit grocery cuts to at most 15-20% while retaining at least Rs. {floor:,.2f}.",
            }

    if "utilit" in category_clean:
        floor = CATEGORY_FLOORS["utilities"]
        remaining = current_category_amount - cut_amount
        if remaining < floor or reduction_percent > 30.0:
            return {
                "is_safe": False,
                "decision": "rejected",
                "violation_type": "UTILITY_FLOOR_VIOLATION",
                "reason": f"Safety Agent Rejected [{strategy_name}]: Cuts essential utilities below minimum operational floor of Rs. {floor:,.2f}.",
                "suggested_fix": "Limit utility savings to energy conservation (max 10-15%).",
            }

    # 3. Discretionary sustainability ceiling check
    if discretionary_total > 0 and total_proposed_discretionary_cuts > 0:
        max_allowed_cuts = round(discretionary_total * MAX_SAFE_DISCRETIONARY_CUT_RATIO, 2)
        if total_proposed_discretionary_cuts > max_allowed_cuts:
            excess = round(total_proposed_discretionary_cuts - max_allowed_cuts, 2)
            return {
                "is_safe": False,
                "decision": "rejected",
                "violation_type": "DISCRETIONARY_CAPACITY_EXCEEDED",
                "reason": (
                    f"Safety Agent Rejected [{strategy_name}]: Proposed total cuts of Rs. {total_proposed_discretionary_cuts:,.2f} "
                    f"exceed sustainable capacity limit of Rs. {max_allowed_cuts:,.2f} (65% of discretionary spend) by Rs. {excess:,.2f}."
                ),
                "suggested_fix": f"Cap total discretionary cuts at or below Rs. {max_allowed_cuts:,.2f} to prevent severe lifestyle fatigue.",
            }

    # 4. Single category excessive reduction check (e.g., cutting dining out 75%+)
    if reduction_percent >= 75.0:
        return {
            "is_safe": False,
            "decision": "rejected",
            "violation_type": "EXTREME_REDUCTION_RATIO",
            "reason": (
                f"Safety Agent Rejected [{strategy_name}]: Proposed {reduction_percent:.0f}% cut in {category} "
                f"breaches sustainability threshold (maximum safe cut is 50%). Extreme cuts lead to immediate relapse."
            ),
            "suggested_fix": f"Limit reduction in {category} to 30-40% for sustainable adherence.",
        }

    return {
        "is_safe": True,
        "decision": "approved",
        "violation_type": "NONE",
        "reason": f"Safety Agent Approved [{strategy_name}]: Verified within safe discretionary thresholds and preserves all essential floors.",
        "suggested_fix": None,
    }


def check_emergency_buffer(
    monthly_expenses: float,
    existing_savings: float,
    monthly_savings_after_plan: float,
    months_buffer_target: int = 3
) -> Dict[str, Any]:
    """
    Verifies that the user maintains or builds toward an emergency buffer of at least 3 months expenses.
    """
    monthly_expenses = max(1.0, float(monthly_expenses))
    existing_savings = max(0.0, float(existing_savings))
    monthly_savings_after_plan = float(monthly_savings_after_plan)
    
    current_buffer_months = round(existing_savings / monthly_expenses, 1)
    target_buffer_amount = round(monthly_expenses * months_buffer_target, 2)
    buffer_deficit = max(0.0, round(target_buffer_amount - existing_savings, 2))

    is_adequate = current_buffer_months >= months_buffer_target
    healthy_cashflow = monthly_savings_after_plan > 0

    return {
        "current_buffer_months": current_buffer_months,
        "target_buffer_months": months_buffer_target,
        "target_buffer_amount": target_buffer_amount,
        "buffer_deficit": buffer_deficit,
        "is_adequate": is_adequate,
        "healthy_cashflow": healthy_cashflow,
        "assessment": (
            "Adequate emergency reserve (>3 months expenses)"
            if is_adequate
            else f"Low emergency reserve ({current_buffer_months} mo vs recommended {months_buffer_target} mo)"
        ),
    }


if __name__ == "__main__":
    # Test protected essential rejection
    test_rej = check_essential_floor("Cut Rent", "rent", 5000, 20000, 25.0)
    print("Rent Test:", test_rej)
    
    # Test capacity overflow rejection
    test_cap = check_essential_floor("Aggressive Cuts", "dining_out", 14000, 15000, 93.0, discretionary_total=18000, total_proposed_discretionary_cuts=14000)
    print("Capacity Test:", test_cap)
    
    # Test approval
    test_app = check_essential_floor("Moderate Dining Cut", "dining_out", 2400, 8000, 30.0, discretionary_total=20000, total_proposed_discretionary_cuts=2400)
    print("Approval Test:", test_app)
