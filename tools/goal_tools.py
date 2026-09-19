"""
tools/goal_tools.py — Pure Python goal feasibility and timeline tools.
No LLM arithmetic.
"""

from typing import Dict, Any


def calculate_required_monthly_saving(
    goal_amount: float,
    deadline_months: int,
    current_savings: float = 0.0,
    current_monthly_saving: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes required monthly savings to achieve target goal within the deadline.
    Remaining goal = max(0, goal_amount - current_savings)
    Required monthly = Remaining goal / deadline_months
    Gap = Required monthly - current_monthly_saving
    """
    goal_amount = float(goal_amount)
    deadline_months = max(1, int(deadline_months))
    current_savings = max(0.0, float(current_savings))
    current_monthly_saving = float(current_monthly_saving)

    net_target = max(0.0, goal_amount - current_savings)
    required_monthly = round(net_target / deadline_months, 2)
    gap = round(required_monthly - current_monthly_saving, 2)
    is_feasible_with_current = bool(current_monthly_saving >= required_monthly and required_monthly > 0)

    return {
        "goal_amount": goal_amount,
        "existing_savings": current_savings,
        "net_target_to_accumulate": round(net_target, 2),
        "deadline_months": deadline_months,
        "current_monthly_saving": current_monthly_saving,
        "required_monthly_saving": required_monthly,
        "gap_vs_current": gap,
        "is_feasible_with_current": is_feasible_with_current,
    }


def project_completion_date(
    current_monthly_saving: float,
    goal_amount: float,
    current_savings: float = 0.0
) -> Dict[str, Any]:
    """
    Projects estimated months needed to reach goal at a given monthly saving rate.
    """
    goal_amount = float(goal_amount)
    current_savings = max(0.0, float(current_savings))
    current_monthly_saving = float(current_monthly_saving)
    net_target = max(0.0, goal_amount - current_savings)

    if net_target <= 0:
        return {
            "months_needed": 0.0,
            "is_reachable": True,
            "projected_timeline_str": "Goal already achieved with current accumulated savings!"
        }

    if current_monthly_saving <= 0:
        return {
            "months_needed": 999.0,
            "is_reachable": False,
            "projected_timeline_str": "Unreachable at current savings rate (no positive monthly savings)"
        }

    months_needed = round(net_target / current_monthly_saving, 1)
    return {
        "months_needed": months_needed,
        "is_reachable": True,
        "projected_timeline_str": f"Estimated {months_needed} months to reach goal"
    }


if __name__ == "__main__":
    g = calculate_required_monthly_saving(200000, 12, 50000, 8000)
    p = project_completion_date(8000, 200000, 50000)
    print("Goal calc:", g)
    print("Project completion:", p)
