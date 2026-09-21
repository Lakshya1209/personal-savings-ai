"""
agents/budget_agent.py — Budget Analysis Agent.
Tools: calculate_savings_rate, calculate_discretionary_capacity.
Strictly delegates all arithmetic to Python tools.
"""

import os
import sys
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from state import AnalysisState, BudgetAgentOutput
from tools.budget_tools import calculate_savings_rate, calculate_discretionary_capacity
from agents.base_agent import invoke_llm_structured

SYSTEM_PROMPT = """
You are the Budget Analysis Agent in a multi-agent personal finance advisory system.
Your role:
1. Evaluate cash flow health based on monthly income vs total expenses.
2. Determine current monthly savings and savings rate percentage.
3. Compute the safe-to-cut capacity from discretionary spending.
4. Output a clear, concise evaluation of the user's budgetary breathing room.

CRITICAL RULE:
You MUST NOT perform any math or arithmetic yourself. All numbers must strictly match
the figures returned by the budget tools.
"""


def run_budget_agent(state: AnalysisState) -> AnalysisState:
    """
    Executes Budget Agent:
    - Reads income from profile and expense totals from state['findings']['expense']
    - Calls calculate_savings_rate and calculate_discretionary_capacity
    - Updates state['findings']['budget'] and appends to state['reasoning_log']
    """
    profile = state["profile"]
    expense_findings = state["findings"].get("expense", {})
    
    income = float(profile.get("monthly_income", profile.get("income", 0.0)))
    total_expenses = float(expense_findings.get("essential_total", 0.0) + expense_findings.get("discretionary_total", 0.0))
    discretionary_total = float(expense_findings.get("discretionary_total", 0.0))
    
    # Tool Calls (Pure Python Arithmetic)
    budget_metrics = calculate_savings_rate(income, total_expenses)
    capacity_metrics = calculate_discretionary_capacity(discretionary_total)
    
    curr_savings = budget_metrics["current_monthly_savings"]
    savings_rate = budget_metrics["savings_rate_percent"]
    safe_capacity = capacity_metrics["safe_to_cut_capacity"]
    health = budget_metrics["budget_status"]

    prompt = f"""
    The Python budget tools calculated:
    - Total Monthly Income: Rs. {income:,.2f}
    - Total Monthly Expenses: Rs. {total_expenses:,.2f}
    - Current Monthly Savings: Rs. {curr_savings:,.2f}
    - Savings Rate: {savings_rate}%
    - Safe Discretionary Cut Capacity: Rs. {safe_capacity:,.2f}
    - Budget Status: {health}

    Return a structured JSON object conforming to BudgetAgentOutput with total_income, total_expenses,
    current_savings, savings_rate, discretionary_capacity, health_status, and a concise reasoning string.
    """

    llm_result = invoke_llm_structured(
        system_instruction=SYSTEM_PROMPT,
        prompt=prompt,
        pydantic_schema=BudgetAgentOutput,
        tools=[calculate_savings_rate, calculate_discretionary_capacity],
        temperature=0.1
    )

    if llm_result:
        finding_data = llm_result.model_dump()
        finding_data["total_income"] = income
        finding_data["total_expenses"] = total_expenses
        finding_data["current_savings"] = curr_savings
        finding_data["savings_rate"] = savings_rate
        finding_data["discretionary_capacity"] = safe_capacity
        finding_data["health_status"] = health
        reasoning_text = llm_result.reasoning
    else:
        reasoning_text = (
            f"Current savings: Rs. {curr_savings:,.2f} ({savings_rate}% savings rate, status: {health}). "
            f"Safe discretionary cut ceiling: Rs. {safe_capacity:,.2f}/month."
        )
        finding_data = {
            "total_income": income,
            "total_expenses": total_expenses,
            "current_savings": curr_savings,
            "savings_rate": savings_rate,
            "discretionary_capacity": safe_capacity,
            "health_status": health,
            "reasoning": reasoning_text,
        }

    state["findings"]["budget"] = finding_data
    log_msg = f"Budget Agent: Savings rate is {savings_rate}% (Rs. {curr_savings:,.2f}/mo); safe discretionary trim ceiling is Rs. {safe_capacity:,.2f}."
    state["reasoning_log"].append(log_msg)
    state["status"] = "budget_analyzed"
    return state


if __name__ == "__main__":
    from agents.expense_agent import run_expense_agent
    sample_state: AnalysisState = {
        "profile": {
            "monthly_income": 85000,
            "expenses": {
                "rent": 22000, "groceries": 10000, "utilities": 4500,
                "debt_emi": 5000, "transport": 4000, "dining_out": 8500,
                "shopping": 7000, "entertainment": 4000, "subscriptions": 2500
            },
            "goal_amount": 200000,
            "goal_deadline_months": 12,
            "existing_savings": 150000,
        },
        "constraints": {"protect_essentials": True},
        "findings": {},
        "candidate_strategies": [],
        "validated_strategies": [],
        "reasoning_log": [],
        "status": "initialized",
    }
    run_expense_agent(sample_state)
    updated = run_budget_agent(sample_state)
    print("Budget Findings:", updated["findings"]["budget"])
    print("Reasoning Log:", updated["reasoning_log"])
