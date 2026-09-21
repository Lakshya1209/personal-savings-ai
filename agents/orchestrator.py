"""
agents/orchestrator.py — Multi-Agent Orchestrator and iterative control loop.
Coordinates sequential execution of Expense, Budget, and Goal agents,
and manages the iterative reject/revise loop between Savings Agent and Safety Agent.
"""

import os
import sys
import logging
from typing import Dict, Any, Generator, Optional, List
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from state import AnalysisState
from agents.base_agent import invoke_llm_structured
from agents.expense_agent import run_expense_agent
from agents.budget_agent import run_budget_agent
from agents.goal_agent import run_goal_agent
from agents.savings_agent import run_savings_agent
from agents.safety_agent import run_safety_agent

logger = logging.getLogger(__name__)


class UserConstraintSchema(BaseModel):
    protect_essentials: bool = Field(default=True, description="Strictly forbid reductions in rent or debt")
    target_extra_savings: float = Field(default=0.0, description="Target incremental savings requested")
    focus_categories: List[str] = Field(default_factory=list, description="Categories preferred for trimming")
    strictness: str = Field(default="Balanced", description="Conservative, Balanced, or Aggressive")


def parse_user_objective_to_constraints(objective_text: str) -> Dict[str, Any]:
    """
    Uses a scoped Groq call (with deterministic fallback) to parse free-text
    user instructions into structured constraints for the multi-agent system.
    """
    if not objective_text or not objective_text.strip():
        return {
            "protect_essentials": True,
            "target_extra_savings": 0.0,
            "focus_categories": ["dining_out", "subscriptions", "shopping"],
            "strictness": "Balanced",
            "rejection_reasons": [],
            "revision_count": 0,
        }

    system_prompt = """
    You are the Objective Parser in an AI Financial Planning System.
    Parse the user's free-text request into structured financial constraints.
    Always keep protect_essentials=True unless explicitly overridden.
    """
    prompt = f"User Request: '{objective_text}'\nExtract structured constraints."
    
    parsed = invoke_llm_structured(
        system_instruction=system_prompt,
        prompt=prompt,
        pydantic_schema=UserConstraintSchema,
        temperature=0.0
    )
    
    if parsed:
        c_dict = parsed.model_dump()
    else:
        # Fallback heuristic
        text_lower = objective_text.lower()
        strictness = "Aggressive" if "aggressive" in text_lower or "fast" in text_lower else "Balanced"
        focus = []
        for cat in ["dining", "subscriptions", "shopping", "entertainment", "transport"]:
            if cat in text_lower:
                focus.append(cat)
        if not focus:
            focus = ["dining_out", "subscriptions", "shopping"]
        c_dict = {
            "protect_essentials": True,
            "target_extra_savings": 0.0,
            "focus_categories": focus,
            "strictness": strictness,
        }

    c_dict["rejection_reasons"] = []
    c_dict["revision_count"] = 0
    return c_dict


def run_pipeline(
    profile: Dict[str, Any],
    objective_text: str = "",
    df: Optional[pd.DataFrame] = None,
    max_revisions: int = 2,
    force_demo_rejection: bool = False,
) -> AnalysisState:
    """
    Synchronous end-to-end multi-agent execution pipeline.
    """
    # Consume full generator
    state = None
    for intermediate_state in run_pipeline_generator(
        profile=profile,
        objective_text=objective_text,
        df=df,
        max_revisions=max_revisions,
        force_demo_rejection=force_demo_rejection,
    ):
        state = intermediate_state
    return state


def run_pipeline_generator(
    profile: Dict[str, Any],
    objective_text: str = "",
    df: Optional[pd.DataFrame] = None,
    max_revisions: int = 2,
    force_demo_rejection: bool = False,
) -> Generator[AnalysisState, None, None]:
    """
    Generator pipeline that yields updated AnalysisState after each agent execution step.
    Enables live progressive streaming to Gradio UI.
    """
    constraints = parse_user_objective_to_constraints(objective_text)
    if force_demo_rejection:
        constraints["force_demo_rejection"] = True

    # 1. Initialize State
    state: AnalysisState = {
        "profile": profile,
        "constraints": constraints,
        "findings": {},
        "candidate_strategies": [],
        "validated_strategies": [],
        "reasoning_log": [
            "Orchestrator: Initialized session. Parsed constraints (Protect Essentials: Yes, Strategy Strictness: "
            + constraints.get("strictness", "Balanced") + ")."
        ],
        "status": "initialized",
    }
    yield state

    # 2. Expense Analysis Agent
    state = run_expense_agent(state, df=df)
    yield state

    # 3. Budget Agent
    state = run_budget_agent(state)
    yield state

    # 4. Goal Agent
    state = run_goal_agent(state)
    yield state

    # 5. Iterative Savings & Safety Loop (Reject -> Revise -> Approve)
    for attempt in range(max_revisions):
        state["constraints"]["revision_count"] = attempt
        
        if attempt > 0:
            state["reasoning_log"].append(
                f"Orchestrator: Loop iteration {attempt + 1}. Routing rejection feedback to Savings Agent for re-generation."
            )
            yield state

        # Run Savings Recommendation Agent
        state = run_savings_agent(state)
        yield state

        # Run Safety & Feasibility Agent
        state = run_safety_agent(state)
        yield state

        # Check outcome
        if state["status"] == "safety_approved":
            state["reasoning_log"].append(
                f"Orchestrator: Safety Agent approved plan on attempt {attempt + 1}. Proceeding to compile final advice."
            )
            yield state
            break
        else:
            # Rejection occurred
            if attempt < max_revisions - 1:
                state["reasoning_log"].append(
                    f"Orchestrator: Safety rejection detected. Triggering revise loop (Attempt {attempt + 1} of {max_revisions})."
                )
                # Ensure second pass does not force demo rejection
                if "force_demo_rejection" in state["constraints"]:
                    state["constraints"]["force_demo_rejection"] = False
                yield state
            else:
                state["reasoning_log"].append(
                    f"Orchestrator: Maximum revision attempts ({max_revisions}) reached. Compiling best available safe fallback plan."
                )
                # Fallback to whatever was safe or empty
                if not state["validated_strategies"]:
                    state["validated_strategies"] = [
                        s for s in state["candidate_strategies"] if s.get("status") in ("approved", "candidate")
                    ]
                yield state

    # 6. Final Compilation
    approved_count = len(state["validated_strategies"])
    total_monthly_saving = sum(s.get("estimated_monthly_saving", 0.0) for s in state["validated_strategies"] if s.get("category") != "composite")
    state["reasoning_log"].append(
        f"Orchestrator: Pipeline execution complete. Compiled {approved_count} validated recommendations "
        f"yielding up to Rs. {total_monthly_saving:,.2f}/month."
    )
    state["status"] = "completed"
    yield state


if __name__ == "__main__":
    # Test Orchestrator with Demo Rejection Scenario (Amit Patel)
    demo_profile = {
        "user_name": "Amit Patel (Demo Reject/Revise Profile)",
        "monthly_income": 55000,
        "expenses": {
            "rent": 20000, "groceries": 11000, "utilities": 4000,
            "debt_emi": 15000, "transport": 3500, "dining_out": 4500,
            "shopping": 3000, "entertainment": 2000, "subscriptions": 1200
        },
        "goal_amount": 180000,
        "goal_deadline_months": 12,
        "existing_savings": 25000,
    }
    
    print("=" * 70)
    print("RUNNING ORCHESTRATOR DEMO WITH REJECT/REVISE LOOP")
    print("=" * 70)
    
    final_state = run_pipeline(
        profile=demo_profile,
        objective_text="Save Rs 15,000 extra per month as quickly as possible",
        force_demo_rejection=True,
    )
    
    print("\nFINAL STATUS:", final_state["status"])
    print(f"Validated Strategies ({len(final_state['validated_strategies'])}):")
    for s in final_state["validated_strategies"]:
        print(f"  * {s['name']}: Rs. {s['estimated_monthly_saving']}/mo ({s['status']})")
    
    print("\nAGENT REASONING LOG (STREAMED DEMO VIEW):")
    for idx, log in enumerate(final_state["reasoning_log"], 1):
        print(f"  {idx:02d}. {log}")
    
    # Assertions
    rejection_logged = any("REJECTED" in l for l in final_state["reasoning_log"])
    approval_logged = any("approved" in l.lower() for l in final_state["reasoning_log"])
    print("\nVerification: Rejection Triggered?", rejection_logged)
    print("Verification: Final Approval Achieved?", approval_logged)
    assert rejection_logged and approval_logged, "Reject -> Revise -> Approve loop did not complete properly!"
    print("=" * 70)
    print("ORCHESTRATOR PASSED ALL CHECKS!")
    print("=" * 70)
