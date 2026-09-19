"""
agents/safety_agent.py — Safety and Feasibility Verification Agent.
Tools: check_essential_floor, check_emergency_buffer.
Strictly delegates all arithmetic and constraint verification to Python tools.
Triggers reject/revise feedback loop when strategies breach safety thresholds.
"""

import os
import sys
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from state import AnalysisState, SafetyAgentOutput, SafetyEvaluationItem
from tools.safety_tools import check_essential_floor, check_emergency_buffer
from agents.base_agent import invoke_gemini_structured

SYSTEM_PROMPT = """
You are the Safety & Feasibility Agent in a multi-agent personal finance advisory system.
Your role:
1. Act as the protective fiduciary barrier ensuring the user's financial stability.
2. Verify that NO strategy cuts non-negotiable essentials (rent, debt/EMI, nutritional food floor).
3. Verify that total discretionary cuts do not exceed safe capacity or cause lifestyle burnout.
4. If ANY strategy violates a safety rule, REJECT it with an explicit, constructive reason so the
   Savings Agent can revise its plan.

CRITICAL RULE:
You MUST NOT perform any math or arithmetic yourself. All floor and buffer checks must strictly
delegate to the Python safety tools: check_essential_floor and check_emergency_buffer.
"""


def run_safety_agent(state: AnalysisState) -> AnalysisState:
    """
    Executes Safety Agent:
    - Evaluates each candidate strategy in state['candidate_strategies']
    - Calls check_essential_floor and check_emergency_buffer
    - If any strategy is rejected, records rejection reasons to state['constraints']['rejection_reasons']
      and returns overall_approved=False
    - If all pass, updates state['validated_strategies']
    - Appends clear human-readable verification logs to state['reasoning_log']
    """
    profile = state["profile"]
    expense_findings = state["findings"].get("expense", {})
    budget_findings = state["findings"].get("budget", {})
    candidate_strategies = state.get("candidate_strategies", [])
    
    cat_totals = expense_findings.get("category_totals", {})
    essential_tot = float(expense_findings.get("essential_total", 0.0))
    disc_tot = float(expense_findings.get("discretionary_total", 0.0))
    monthly_income = float(profile.get("monthly_income", profile.get("income", 0.0)))
    existing_savings = float(profile.get("existing_savings", 0.0))
    
    evaluations: List[SafetyEvaluationItem] = []
    rejection_reasons: List[str] = []
    validated_list: List[Dict[str, Any]] = []
    
    overall_approved = True

    # Check emergency buffer
    buffer_check = check_emergency_buffer(
        monthly_expenses=essential_tot + disc_tot,
        existing_savings=existing_savings,
        monthly_savings_after_plan=float(budget_findings.get("current_savings", 0.0))
    )

    for strat in candidate_strategies:
        name = strat.get("name", "Unknown Strategy")
        cat = strat.get("category", "discretionary")
        saving = float(strat.get("estimated_monthly_saving", 0.0))
        pct = float(strat.get("reduction_percent", 0.0))
        orig_cat_amt = float(cat_totals.get(cat, disc_tot))

        # Tool Call: check_essential_floor
        floor_eval = check_essential_floor(
            strategy_name=name,
            category=cat,
            cut_amount=saving,
            current_category_amount=orig_cat_amt,
            reduction_percent=pct,
            discretionary_total=disc_tot,
            total_proposed_discretionary_cuts=saving
        )

        decision = floor_eval["decision"]
        reason = floor_eval["reason"]
        suggested_fix = floor_eval.get("suggested_fix")

        evaluations.append(SafetyEvaluationItem(
            strategy_name=name,
            decision=decision,
            reason=reason,
            suggested_fix=suggested_fix
        ))

        if decision == "rejected":
            overall_approved = False
            rejection_reasons.append(reason)
            strat_copy = dict(strat)
            strat_copy["status"] = "rejected"
            strat_copy["safety_notes"] = reason
            state["reasoning_log"].append(reason)
        else:
            strat_copy = dict(strat)
            strat_copy["status"] = "approved" if decision == "approved" else "modified"
            strat_copy["safety_notes"] = reason
            validated_list.append(strat_copy)

    # Prompt Gemini for structured review summary
    prompt = f"""
    The Python safety tools evaluated candidate strategies:
    {[e.model_dump() for e in evaluations]}
    
    Emergency Buffer Check: {buffer_check}
    Overall Approved: {overall_approved}
    Rejection Reasons: {rejection_reasons}

    Return a structured JSON object conforming to SafetyAgentOutput.
    """

    llm_result = invoke_gemini_structured(
        system_instruction=SYSTEM_PROMPT,
        prompt=prompt,
        pydantic_schema=SafetyAgentOutput,
        tools=[check_essential_floor, check_emergency_buffer],
        temperature=0.1
    )

    if overall_approved:
        state["validated_strategies"] = validated_list
        state["status"] = "safety_approved"
        log_msg = f"Safety Agent: Approved all {len(validated_list)} strategies. Preserves essential floors and emergency buffer."
        state["reasoning_log"].append(log_msg)
    else:
        state["status"] = "safety_rejected"
        state["constraints"]["rejection_reasons"] = rejection_reasons
        log_msg = f"Safety Agent: REJECTED candidate plan ({len(rejection_reasons)} violations found). Demanding revision from Savings Agent."
        state["reasoning_log"].append(log_msg)

    return state


if __name__ == "__main__":
    from agents.expense_agent import run_expense_agent
    from agents.budget_agent import run_budget_agent
    from agents.goal_agent import run_goal_agent
    from agents.savings_agent import run_savings_agent
    
    # Test case: Amit Patel (Demo Rejection Scenario)
    sample_state: AnalysisState = {
        "profile": {
            "monthly_income": 55000,
            "expenses": {
                "rent": 20000, "groceries": 11000, "utilities": 4000,
                "debt_emi": 15000, "transport": 3500, "dining_out": 4500,
                "shopping": 3000, "entertainment": 2000, "subscriptions": 1200
            },
            "goal_amount": 180000,
            "goal_deadline_months": 12,
            "existing_savings": 25000,
        },
        "constraints": {"protect_essentials": True, "revision_count": 0, "force_demo_rejection": True},
        "findings": {},
        "candidate_strategies": [],
        "validated_strategies": [],
        "reasoning_log": [],
        "status": "initialized",
    }
    
    print("--- Running Expense Agent ---")
    run_expense_agent(sample_state)
    print("--- Running Budget Agent ---")
    run_budget_agent(sample_state)
    print("--- Running Goal Agent ---")
    run_goal_agent(sample_state)
    print("--- Running Savings Agent (Pass 1) ---")
    run_savings_agent(sample_state)
    print("--- Running Safety Agent (Pass 1) ---")
    run_safety_agent(sample_state)
    
    print(f"\nFinal State Status: {sample_state['status']}")
    print("Rejection Reasons in Constraints:", sample_state["constraints"].get("rejection_reasons"))
    print("\nReasoning Log:")
    for l in sample_state["reasoning_log"]:
        print(f" -> {l}")
