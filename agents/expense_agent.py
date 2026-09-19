"""
agents/expense_agent.py — Expense Analysis Agent.
Tools: category_totals, essential_vs_discretionary, detect_recurring.
Strictly delegates all arithmetic to Python tools.
"""

import os
import sys
from typing import Dict, Any, Union
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from state import AnalysisState, ExpenseAgentOutput
from tools.expense_tools import category_totals, essential_vs_discretionary, detect_recurring
from agents.base_agent import invoke_gemini_structured


SYSTEM_PROMPT = """
You are the Expense Analysis Agent in a multi-agent personal finance advisory system.
Your role:
1. Take category totals and partition expenses into essential commitments vs discretionary spending.
2. Identify fixed recurring charges (rent, EMI, subscriptions).
3. Provide a concise, clear plain-language summary of where the user's money is going.

CRITICAL RULE:
You MUST NOT perform any math or arithmetic yourself. All totals, splits, and recurring figures
must strictly match the provided tool results.
"""


def run_expense_agent(state: AnalysisState, df: Union[pd.DataFrame, None] = None) -> AnalysisState:
    """
    Executes Expense Analysis Agent:
    - Calls category_totals, essential_vs_discretionary, detect_recurring
    - Invokes Gemini for structured synthesis with exact tool figures
    - Updates state['findings']['expense'] and appends to state['reasoning_log']
    """
    profile = state["profile"]
    source_data = df if df is not None else profile
    
    # Tool Calls (Pure Python Arithmetic)
    cat_totals = category_totals(source_data)
    split = essential_vs_discretionary(cat_totals)
    recurring = detect_recurring(source_data)
    
    essential_tot = split["essential_total"]
    disc_tot = split["discretionary_total"]
    
    # Prompt for Gemini synthesis
    prompt = f"""
    The Python expense tools have computed the following exact financial breakdown:
    - Category Totals: {cat_totals}
    - Essential Expenses Total: Rs. {essential_tot:,.2f} ({split['essential_ratio']}%)
    - Discretionary Expenses Total: Rs. {disc_tot:,.2f} ({split['discretionary_ratio']}%)
    - Detected Recurring Commitments: {recurring}
    
    Return a structured JSON object with category_totals, essential_total, discretionary_total,
    recurring_items, and a concise 1-2 sentence reasoning summary.
    """
    
    # Attempt LLM structured generation
    llm_result = invoke_gemini_structured(
        system_instruction=SYSTEM_PROMPT,
        prompt=prompt,
        pydantic_schema=ExpenseAgentOutput,
        tools=[category_totals, essential_vs_discretionary, detect_recurring],
        temperature=0.1
    )
    
    if llm_result:
        # Enforce tool arithmetic onto output
        finding_data = llm_result.model_dump()
        finding_data["category_totals"] = cat_totals
        finding_data["essential_total"] = essential_tot
        finding_data["discretionary_total"] = disc_tot
        finding_data["recurring_items"] = recurring
        reasoning_text = llm_result.reasoning
    else:
        # Graceful fallback with tool outputs
        reasoning_text = (
            f"Expenses categorized across {len(cat_totals)} buckets: "
            f"Rs. {essential_tot:,.2f} ({split['essential_ratio']}%) essential vs "
            f"Rs. {disc_tot:,.2f} ({split['discretionary_ratio']}%) discretionary."
        )
        finding_data = {
            "category_totals": cat_totals,
            "essential_total": essential_tot,
            "discretionary_total": disc_tot,
            "recurring_items": recurring,
            "reasoning": reasoning_text,
        }

    # Update State
    state["findings"]["expense"] = finding_data
    log_msg = f"Expense Agent: Categorized {len(cat_totals)} expense areas (Essential: Rs. {essential_tot:,.2f}, Discretionary: Rs. {disc_tot:,.2f})."
    state["reasoning_log"].append(log_msg)
    state["status"] = "expense_analyzed"
    return state


if __name__ == "__main__":
    sample_state: AnalysisState = {
        "profile": {
            "monthly_income": 85000,
            "expenses": {
                "rent": 22000,
                "groceries": 10000,
                "utilities": 4500,
                "debt_emi": 5000,
                "transport": 4000,
                "dining_out": 8500,
                "shopping": 7000,
                "entertainment": 4000,
                "subscriptions": 2500,
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
    updated = run_expense_agent(sample_state)
    print("Expense Findings:", updated["findings"]["expense"])
    print("Reasoning Log:", updated["reasoning_log"])
