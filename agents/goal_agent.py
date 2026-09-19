"""
agents/goal_agent.py — Financial Goal Feasibility Agent.
Tools: calculate_required_monthly_saving, project_completion_date.
Strictly delegates all arithmetic to Python tools.
"""

import os
import sys
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from state import AnalysisState, GoalAgentOutput
from tools.goal_tools import calculate_required_monthly_saving, project_completion_date
from agents.base_agent import invoke_gemini_structured

SYSTEM_PROMPT = """
You are the Goal Feasibility Agent in a multi-agent personal finance advisory system.
Your role:
1. Evaluate whether the user's financial goal is achievable by their target deadline.
2. Determine required monthly savings vs current savings velocity.
3. Compute the monthly shortfall/gap and estimate timeline to completion.
4. Output an actionable feasibility assessment.

CRITICAL RULE:
You MUST NOT perform any math or arithmetic yourself. All figures must strictly match
the figures returned by the goal tools.
"""


def run_goal_agent(state: AnalysisState) -> AnalysisState:
    """
    Executes Goal Agent:
    - Reads goal_amount, goal_deadline_months, existing_savings from profile
    - Reads current_monthly_savings from state['findings']['budget']
    - Calls calculate_required_monthly_saving and project_completion_date
    - Updates state['findings']['goal'] and appends to state['reasoning_log']
    """
    profile = state["profile"]
    budget_findings = state["findings"].get("budget", {})
    
    goal_amount = float(profile.get("goal_amount", 0.0))
    deadline_months = int(profile.get("goal_deadline_months", profile.get("deadline_months", 12)))
    existing_savings = float(profile.get("existing_savings", 0.0))
    curr_monthly_saving = float(budget_findings.get("current_savings", 0.0))
    
    # Tool Calls (Pure Python Arithmetic)
    goal_metrics = calculate_required_monthly_saving(
        goal_amount=goal_amount,
        deadline_months=deadline_months,
        current_savings=existing_savings,
        current_monthly_saving=curr_monthly_saving,
    )
    timeline_metrics = project_completion_date(
        current_monthly_saving=curr_monthly_saving,
        goal_amount=goal_amount,
        current_savings=existing_savings,
    )

    req_monthly = goal_metrics["required_monthly_saving"]
    gap = goal_metrics["gap_vs_current"]
    is_feasible = goal_metrics["is_feasible_with_current"]
    proj_months = timeline_metrics["months_needed"]

    prompt = f"""
    The Python goal tools evaluated:
    - Target Goal Amount: Rs. {goal_amount:,.2f}
    - Existing Liquid Savings: Rs. {existing_savings:,.2f}
    - Deadline: {deadline_months} months
    - Current Monthly Savings: Rs. {curr_monthly_saving:,.2f}
    - Required Monthly Savings: Rs. {req_monthly:,.2f}
    - Monthly Savings Shortfall/Gap: Rs. {gap:,.2f}
    - Feasible with Current Trajectory: {is_feasible}
    - Projected Completion Timeline: {proj_months} months

    Return a structured JSON object conforming to GoalAgentOutput with goal_amount, deadline_months,
    current_savings_rate, required_monthly_saving, gap_vs_current, is_feasible_with_current,
    projected_completion_months, and a concise reasoning string.
    """

    llm_result = invoke_gemini_structured(
        system_instruction=SYSTEM_PROMPT,
        prompt=prompt,
        pydantic_schema=GoalAgentOutput,
        tools=[calculate_required_monthly_saving, project_completion_date],
        temperature=0.1
    )

    if llm_result:
        finding_data = llm_result.model_dump()
        finding_data["goal_amount"] = goal_amount
        finding_data["deadline_months"] = deadline_months
        finding_data["current_savings_rate"] = curr_monthly_saving
        finding_data["required_monthly_saving"] = req_monthly
        finding_data["gap_vs_current"] = gap
        finding_data["is_feasible_with_current"] = is_feasible
        finding_data["projected_completion_months"] = proj_months
        reasoning_text = llm_result.reasoning
    else:
        status_word = "on track" if is_feasible else f"short by Rs. {gap:,.2f}/month"
        reasoning_text = (
            f"Goal of Rs. {goal_amount:,.2f} in {deadline_months} months requires Rs. {req_monthly:,.2f}/mo. "
            f"Current trajectory is {status_word} ({proj_months} months projected)."
        )
        finding_data = {
            "goal_amount": goal_amount,
            "deadline_months": deadline_months,
            "current_savings_rate": curr_monthly_saving,
            "required_monthly_saving": req_monthly,
            "gap_vs_current": gap,
            "is_feasible_with_current": is_feasible,
            "projected_completion_months": proj_months,
            "reasoning": reasoning_text,
        }

    state["findings"]["goal"] = finding_data
    status_tag = "Feasible without cuts" if is_feasible else f"Deficit of Rs. {gap:,.2f}/mo"
    log_msg = f"Goal Agent: Target Rs. {goal_amount:,.2f} in {deadline_months} mo needs Rs. {req_monthly:,.2f}/mo ({status_tag})."
    state["reasoning_log"].append(log_msg)
    state["status"] = "goal_analyzed"
    return state


if __name__ == "__main__":
    from agents.expense_agent import run_expense_agent
    from agents.budget_agent import run_budget_agent
    
    sample_state: AnalysisState = {
        "profile": {
            "monthly_income": 85000,
            "expenses": {
                "rent": 22000, "groceries": 10000, "utilities": 4500,
                "debt_emi": 5000, "transport": 4000, "dining_out": 8500,
                "shopping": 7000, "entertainment": 4000, "subscriptions": 2500
            },
            "goal_amount": 350000,
            "goal_deadline_months": 12,
            "existing_savings": 50000,
        },
        "constraints": {"protect_essentials": True},
        "findings": {},
        "candidate_strategies": [],
        "validated_strategies": [],
        "reasoning_log": [],
        "status": "initialized",
    }
    run_expense_agent(sample_state)
    run_budget_agent(sample_state)
    updated = run_goal_agent(sample_state)
    print("Goal Findings:", updated["findings"]["goal"])
    print("Reasoning Log:", updated["reasoning_log"])
