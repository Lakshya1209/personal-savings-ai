"""
app.py — Personal Savings & Cash Flow Advisory (SaaS Product Edition).
Includes complete SQLite-backed authentication, per-user data isolation, central dashboard,
dedicated saved plans section, responsive fintech design, and preserved multi-agent AI pipeline.
"""

import os
import sys
import time
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Ensure workspace root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database
from state import AnalysisState
from agents.orchestrator import run_pipeline_generator
from tools.budget_tools import calculate_savings_rate
from tools.goal_tools import calculate_required_monthly_saving, project_completion_date
from tools.savings_tools import simulate_category_reduction, combine_strategies
from agents.base_agent import get_llm_client, get_candidate_models
from agents.faq_agent import get_faq_response

# ---------------------------------------------------------------------------
# Global Session Fallbacks (Isolated per user in DB)
# ---------------------------------------------------------------------------
CURRENT_SESSION_STATE = {"state": None, "source_df": None}


# ---------------------------------------------------------------------------
# Visual Financial Charts
# ---------------------------------------------------------------------------
def create_expense_chart(state: AnalysisState):
    """Generates a clean horizontal bar chart of spending categories."""
    findings = state.get("findings", {})
    expense_data = findings.get("expense", {})
    cat_totals = expense_data.get("category_totals", {})
    if not cat_totals:
        return None

    labels = []
    values = []
    colors = []
    essentials = {"rent", "mortgage", "groceries", "utilities", "debt_emi", "loan_payment"}
    
    for cat, val in cat_totals.items():
        clean_name = cat.replace("_", " ").title()
        labels.append(clean_name)
        values.append(val)
        if any(e in cat.lower() for e in essentials):
            colors.append("#1C232B")  # Primary ink for essentials
        else:
            colors.append("#C97A3E")  # Warmer gold-orange accent for flexible spending

    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8A8578", labelsize=9, left=False, bottom=False)
    ax.grid(axis="x", linestyle="--", alpha=0.6, color="#F0EDE5")
    ax.set_xlabel("Monthly Spending (₹)", color="#8A8578", fontsize=9, labelpad=8)
    ax.set_title("Monthly Spending Distribution (Ink: Essential, Orange: Flexible)", loc="left", color="#1C232B", fontsize=11, weight="bold", pad=12)
    
    max_val = max(values) if values else 1
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max_val * 0.02, bar.get_y() + bar.get_height() / 2, f"₹{width:,.0f}",
                va="center", ha="left", color="#1C232B", fontsize=8.5, weight="bold")
                
    plt.tight_layout()
    return fig


def create_goal_chart(state: AnalysisState):
    """Generates a comparison bar chart for target goal requirements."""
    findings = state.get("findings", {})
    goal_data = findings.get("goal", {})
    budget_data = findings.get("budget", {})
    
    curr_saving = budget_data.get("current_savings", 0.0)
    req_saving = goal_data.get("required_monthly_saving", 0.0)
    
    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    
    metrics = ["Current Savings", "Required for Target"]
    values = [max(0, curr_saving), req_saving]
    bar_colors = ["#3A7A5A" if curr_saving >= req_saving else "#A8532F", "#C97A3E"]
    
    bars = ax.bar(metrics, values, color=bar_colors, width=0.42)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8A8578", labelsize=9, left=False, bottom=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#F0EDE5")
    ax.set_ylabel("Monthly Amount (₹)", color="#8A8578", fontsize=9)
    ax.set_title("Monthly Savings Velocity vs Goal Target", loc="left", color="#1C232B", fontsize=11, weight="bold", pad=12)

    max_val = max(values) if values else 1
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + max_val * 0.03, f"₹{yval:,.0f}",
                ha="center", va="bottom", color="#1C232B", fontsize=9, weight="bold")

    plt.tight_layout()
    return fig


def create_dashboard_cashflow_chart(profile: dict):
    """Generates a cash flow chart for the central dashboard."""
    income = float(profile.get("monthly_income", 60000.0))
    expenses = (
        float(profile.get("rent", 18000.0)) + float(profile.get("groceries", 9000.0)) +
        float(profile.get("utilities", 3500.0)) + float(profile.get("debt_emi", 5000.0)) +
        float(profile.get("transport", 3500.0)) + float(profile.get("dining_out", 5000.0)) +
        float(profile.get("shopping", 4000.0)) + float(profile.get("entertainment", 2500.0)) +
        float(profile.get("subscriptions", 1500.0))
    )
    net_savings = income - expenses

    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    categories = ["Income", "Expenses", "Net Savings"]
    vals = [income, expenses, max(0.0, net_savings)]
    colors = ["#1C232B", "#A8532F" if expenses > income else "#5C5648", "#3A7A5A" if net_savings >= 0 else "#A8532F"]

    bars = ax.bar(categories, vals, color=colors, width=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8A8578", labelsize=9.5, left=False, bottom=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#F0EDE5")
    ax.set_ylabel("Amount (₹)", color="#8A8578", fontsize=9)
    ax.set_title("Monthly Cash Flow Dynamics", loc="left", color="#1C232B", fontsize=11, weight="bold", pad=12)

    max_v = max(vals) if vals else 1
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_v * 0.03, f"₹{h:,.0f}",
                ha="center", va="bottom", color="#1C232B", fontsize=8.5, weight="bold")

    plt.tight_layout()
    return fig


def create_dashboard_breakdown_chart(profile: dict):
    """Generates essential vs discretionary expense split for the dashboard."""
    essential = (
        float(profile.get("rent", 18000.0)) + float(profile.get("groceries", 9000.0)) +
        float(profile.get("utilities", 3500.0)) + float(profile.get("debt_emi", 5000.0))
    )
    discretionary = (
        float(profile.get("transport", 3500.0)) + float(profile.get("dining_out", 5000.0)) +
        float(profile.get("shopping", 4000.0)) + float(profile.get("entertainment", 2500.0)) +
        float(profile.get("subscriptions", 1500.0))
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    categories = ["Essential Commitments", "Flexible Lifestyle"]
    vals = [essential, discretionary]
    colors = ["#1C232B", "#C97A3E"]

    bars = ax.bar(categories, vals, color=colors, width=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8A8578", labelsize=9.5, left=False, bottom=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#F0EDE5")
    ax.set_ylabel("Spending (₹)", color="#8A8578", fontsize=9)
    ax.set_title("Essential vs Flexible Spending Split", loc="left", color="#1C232B", fontsize=11, weight="bold", pad=12)

    max_v = max(vals) if vals else 1
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_v * 0.03, f"₹{h:,.0f}",
                ha="center", va="bottom", color="#1C232B", fontsize=8.5, weight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Human-Centered UI Generators
# ---------------------------------------------------------------------------
PROGRESS_STEPS = [
    ("Reviewing your income and expenses", "Analyzing your cash flow and fixed commitments"),
    ("Checking what's essential vs. flexible", "Protecting housing, food, and essentials"),
    ("Working out your goal timeline", "Calculating savings velocity against your target"),
    ("Building your savings plan", "Finding realistic areas to free up monthly cash"),
    ("Double-checking it's realistic for you", "Ensuring cuts are sustainable and safe"),
]

def render_human_progress(active_idx=0, is_complete=False, status_note=None):
    """Renders a clean, human progress sequence during plan generation."""
    if is_complete:
        badge_html = '<span class="status-pill complete">✓ Plan Ready</span>'
    else:
        badge_html = '<span class="status-pill active">Analyzing Finances</span>'

    items_html = ""
    for i, (label, sub) in enumerate(PROGRESS_STEPS):
        if is_complete or i < active_idx:
            state_class = "step-done"
            icon_html = '<div class="step-check">✓</div>'
        elif i == active_idx:
            state_class = "step-active"
            icon_html = '<div class="step-pulse"></div>'
        else:
            state_class = "step-pending"
            icon_html = '<div class="step-dot"></div>'

    items_html = "".join([
        f"""
        <div class="progress-step-row {state_class}">
            <div class="step-indicator-col">{icon_html}</div>
            <div class="step-text-col">
                <div class="step-label-title">{label}</div>
                <div class="step-sub-desc">{sub}</div>
            </div>
        </div>
        """ for i, (label, sub) in enumerate(PROGRESS_STEPS)
    ])

    note_html = f'<div class="progress-note">{status_note}</div>' if status_note else ""

    return f"""
    <div class="human-progress-card">
        <div class="progress-card-header">
            <div>
                <h3 class="progress-card-title">Personal Financial Analysis</h3>
                <p class="progress-card-subtitle">Building your customized savings path</p>
            </div>
            {badge_html}
        </div>
        <div class="progress-steps-list">
            {items_html}
        </div>
        {note_html}
    </div>
    """


def render_plan_hero(state: AnalysisState):
    """Generates the hero financial metrics banner with high-weight ₹ numbers and goal progress."""
    if not state or "findings" not in state:
        return ""
    
    findings = state["findings"]
    profile = state.get("profile", {})
    budget_data = findings.get("budget", {})
    goal_data = findings.get("goal", {})
    validated = state.get("validated_strategies", [])

    composite_strat = next((s for s in validated if "Synergy" in s.get("name", "") or "Composite" in s.get("name", "")), None)
    if composite_strat:
        unlocked_amt = float(composite_strat.get("estimated_monthly_saving", 0))
    elif validated:
        unlocked_amt = float(validated[0].get("estimated_monthly_saving", 0))
    else:
        unlocked_amt = 0.0

    current_savings = float(budget_data.get("current_savings", 0.0))
    new_monthly_savings = current_savings + unlocked_amt
    monthly_income = float(profile.get("monthly_income", 1.0))
    new_savings_rate = (new_monthly_savings / monthly_income * 100) if monthly_income > 0 else 0.0
    
    goal_amount = float(profile.get("goal_amount", 0.0))
    existing_savings = float(profile.get("existing_savings", 0.0))
    net_needed = max(0.0, goal_amount - existing_savings)
    progress_pct = min(100.0, (existing_savings / goal_amount * 100.0)) if goal_amount > 0 else 0.0

    if new_monthly_savings > 0:
        months_to_goal = int(net_needed / new_monthly_savings) + (1 if (net_needed % new_monthly_savings) > 0 else 0)
        target_deadline = int(profile.get("goal_deadline_months", 12))
        if months_to_goal <= target_deadline:
            timeline_str = f"Estimated in {months_to_goal} months (On track for your {target_deadline}-mo target)"
        else:
            timeline_str = f"Estimated in {months_to_goal} months (Target horizon: {target_deadline} months)"
    else:
        timeline_str = "Timeline pending surplus generation"

    return f"""
    <div class="hero-plan-dashboard">
        <div class="hero-metric-strip">
            <div class="hero-metric-cell hero-highlight">
                <div class="metric-cell-label">MONTHLY SAVINGS UNLOCKED</div>
                <div class="metric-cell-amount positive">₹{unlocked_amt:,.0f}</div>
                <div class="metric-cell-sub">Freed up from flexible spending</div>
            </div>
            <div class="hero-metric-cell">
                <div class="metric-cell-label">NEW TOTAL MONTHLY SAVINGS</div>
                <div class="metric-cell-amount">₹{new_monthly_savings:,.0f}</div>
                <div class="metric-cell-sub">{new_savings_rate:.1f}% of your monthly income</div>
            </div>
            <div class="hero-metric-cell">
                <div class="metric-cell-label">SAVINGS GOAL TARGET</div>
                <div class="metric-cell-amount">₹{goal_amount:,.0f}</div>
                <div class="metric-cell-sub">{timeline_str}</div>
            </div>
        </div>

        <div class="goal-progress-card">
            <div class="goal-progress-header">
                <span class="goal-progress-title">Goal Reserve Progress</span>
                <span class="goal-progress-badge">{progress_pct:.1f}% achieved</span>
            </div>
            <div class="goal-progress-track">
                <div class="goal-progress-fill" style="width: {progress_pct:.1f}%;"></div>
            </div>
            <div class="goal-progress-footer">
                <span class="progress-sub-stat">Current reserve: <b>₹{existing_savings:,.0f}</b></span>
                <span class="progress-sub-stat">Target: <b>₹{goal_amount:,.0f}</b></span>
            </div>
        </div>
    </div>
    """


def format_recommendations_html(validated_strategies, has_revision=False):
    """Renders recommendations as clean financial habit cards."""
    if not validated_strategies:
        return """
        <div class="empty-recommendations-box">
            <p>Your tailored recommendations will appear here once you build your plan.</p>
        </div>
        """

    cards_html = ""
    for s in validated_strategies:
        name = s.get("name", "Savings Strategy")
        action = s.get("action", "")
        reason = s.get("reason", "")
        monthly = s.get("estimated_monthly_saving", 0)
        annual = s.get("annual_impact", monthly * 12)

        revision_banner = ""
        if has_revision:
            revision_banner = """
            <div class="revision-note-banner">
                <span class="note-bullet">✦</span>
                <span>We scaled this back from a more aggressive cut to keep your daily lifestyle comfortable and realistic.</span>
            </div>
            """

        cards_html += f"""
        <div class="recommendation-item-card">
            <div class="card-header-row">
                <div class="card-title-text">{name}</div>
                <div class="status-badge-realistic">✓ Realistic for you</div>
            </div>
            <div class="card-action-text">{action}</div>
            <div class="card-reason-text">{reason}</div>

            <div class="card-numbers-row">
                <div class="number-stat-item">
                    <span class="stat-caption">ESTIMATED MONTHLY SAVING</span>
                    <span class="stat-value-highlight">₹{monthly:,.0f}</span>
                </div>
                <div class="stat-divider"></div>
                <div class="number-stat-item">
                    <span class="stat-caption">ANNUALIZED IMPACT</span>
                    <span class="stat-value">₹{annual:,.0f}</span>
                </div>
            </div>
            {revision_banner}
        </div>
        """

    return f"""
    <div class="recommendations-container">
        <div class="recommendations-intro">
            <h3 class="recommendations-title">Recommended Spending Adjustments</h3>
            <p class="recommendations-subtitle">Curated habits that protect your essential baseline while reaching your goal.</p>
        </div>
        <div class="recommendations-list">
            {cards_html}
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Dashboard Renderers
# ---------------------------------------------------------------------------
def render_dashboard_metrics(user: dict, profile: dict, latest_plan: dict = None) -> str:
    """Renders the top summary metric cards on the central dashboard."""
    income = float(profile.get("monthly_income", 60000.0))
    expenses = (
        float(profile.get("rent", 18000.0)) + float(profile.get("groceries", 9000.0)) +
        float(profile.get("utilities", 3500.0)) + float(profile.get("debt_emi", 5000.0)) +
        float(profile.get("transport", 3500.0)) + float(profile.get("dining_out", 5000.0)) +
        float(profile.get("shopping", 4000.0)) + float(profile.get("entertainment", 2500.0)) +
        float(profile.get("subscriptions", 1500.0))
    )
    net_savings = income - expenses
    savings_rate = (net_savings / income * 100.0) if income > 0 else 0.0

    potential_savings = float(latest_plan.get("total_monthly_cut", 0.0)) if latest_plan else (
        (float(profile.get("dining_out", 5000)) + float(profile.get("shopping", 4000)) + float(profile.get("subscriptions", 1500))) * 0.25
    )

    name = user.get("full_name", "Valued Member") if user else "Valued Member"

    return f"""
    <div class="dashboard-welcome-banner">
        <div class="welcome-text-wrap">
            <h2 class="welcome-title">Welcome back, {name}</h2>
            <p class="welcome-subtitle">Here is your verified personal finance overview and current cash flow trajectory.</p>
        </div>
        <div class="quick-action-strip">
            <span class="sync-pill">● Database Synced</span>
        </div>
    </div>

    <div class="dashboard-metric-grid">
        <div class="dash-card">
            <div class="dash-card-caption">MONTHLY NET INCOME</div>
            <div class="dash-card-val">₹{income:,.0f}</div>
            <div class="dash-card-foot">Verified inflow</div>
        </div>
        <div class="dash-card">
            <div class="dash-card-caption">MONTHLY EXPENSES</div>
            <div class="dash-card-val">₹{expenses:,.0f}</div>
            <div class="dash-card-foot">Fixed + discretionary</div>
        </div>
        <div class="dash-card">
            <div class="dash-card-caption">CURRENT NET SAVINGS</div>
            <div class="dash-card-val {"positive" if net_savings >= 0 else "negative"}">₹{net_savings:,.0f}</div>
            <div class="dash-card-foot">Monthly surplus</div>
        </div>
        <div class="dash-card">
            <div class="dash-card-caption">SAVINGS RATE</div>
            <div class="dash-card-val highlight">{savings_rate:.1f}%</div>
            <div class="dash-card-foot">Target benchmark: >20%</div>
        </div>
        <div class="dash-card dash-card-accent">
            <div class="dash-card-caption">POTENTIAL SAVINGS UNLOCKED</div>
            <div class="dash-card-val accent">₹{potential_savings:,.0f}</div>
            <div class="dash-card-foot">From safe discretionary trimming</div>
        </div>
    </div>
    """


def render_dashboard_active_goal(profile: dict) -> str:
    """Renders the active savings goal card on the dashboard."""
    goal_amt = float(profile.get("goal_amount", 0.0))
    current_res = float(profile.get("existing_savings", 0.0))
    deadline = int(profile.get("goal_deadline_months", 12))
    objective = profile.get("objective", "Build emergency cushion.")

    if goal_amt <= 0:
        return """
        <div class="dash-goal-box empty">
            <div class="empty-goal-icon">🎯</div>
            <h4 class="empty-goal-title">No Active Savings Goal Configured</h4>
            <p class="empty-goal-sub">Head over to the <b>Your Finances</b> tab to set a target goal and horizon.</p>
        </div>
        """

    remaining = max(0.0, goal_amt - current_res)
    progress_pct = min(100.0, (current_res / goal_amt * 100.0)) if goal_amt > 0 else 0.0
    req_monthly = remaining / deadline if deadline > 0 else remaining

    return f"""
    <div class="dash-goal-box">
        <div class="goal-header-line">
            <div>
                <span class="goal-tag">ACTIVE SAVINGS TARGET</span>
                <h3 class="goal-main-title">{objective}</h3>
            </div>
            <div class="goal-status-badge">
                <span>{progress_pct:.1f}% Complete</span>
            </div>
        </div>

        <div class="goal-progress-track">
            <div class="goal-progress-fill" style="width: {progress_pct:.1f}%;"></div>
        </div>

        <div class="goal-stat-row">
            <div class="goal-mini-stat">
                <span class="mini-caption">TARGET AMOUNT</span>
                <span class="mini-val">₹{goal_amt:,.0f}</span>
            </div>
            <div class="goal-mini-stat">
                <span class="mini-caption">CURRENT RESERVE</span>
                <span class="mini-val">₹{current_res:,.0f}</span>
            </div>
            <div class="goal-mini-stat">
                <span class="mini-caption">REMAINING TO ACCUMULATE</span>
                <span class="mini-val highlight">₹{remaining:,.0f}</span>
            </div>
            <div class="goal-mini-stat">
                <span class="mini-caption">TARGET HORIZON</span>
                <span class="mini-val">{deadline} Months</span>
            </div>
            <div class="goal-mini-stat">
                <span class="mini-caption">REQUIRED MONTHLY SAVINGS</span>
                <span class="mini-val accent">₹{req_monthly:,.0f}/mo</span>
            </div>
        </div>
    </div>
    """


def render_saved_plans_html(user_id: int) -> str:
    """Renders the user-isolated saved plans list."""
    if not user_id:
        return "<p>Please sign in to view your saved plans.</p>"

    plans = database.get_user_plans(user_id)
    if not plans:
        return """
        <div class="empty-state-box">
            <div class="empty-state-icon">📋</div>
            <h3 class="empty-state-title">No saved plans yet</h3>
            <p class="empty-state-desc">Create your first savings plan using the multi-agent recommendation engine and save it to track your progress over time.</p>
        </div>
        """

    items_html = ""
    for p in plans:
        strategies = p.get("strategies", [])
        strategies_html = ""
        for s in strategies:
            s_name = s.get("name", "Cut")
            s_cut = float(s.get("estimated_monthly_saving", 0))
            strategies_html += f"""
            <span class="plan-strat-tag">
                <b>{s_name}</b>: -₹{s_cut:,.0f}/mo
            </span>
            """

        items_html += f"""
        <div class="saved-plan-card">
            <div class="saved-plan-top">
                <div>
                    <h3 class="saved-plan-title">{p['title']}</h3>
                    <p class="saved-plan-meta">Saved on {p.get('created_at_fmt', 'Recently')} • Plan ID #{p['id']}</p>
                </div>
                <div class="saved-plan-badge">✓ Verified Safe</div>
            </div>

            <div class="saved-plan-metrics-grid">
                <div class="saved-metric-cell">
                    <span class="sm-label">GOAL AMOUNT</span>
                    <span class="sm-value">₹{p['goal_amount']:,.0f}</span>
                </div>
                <div class="saved-metric-cell">
                    <span class="sm-label">REQUIRED SAVING</span>
                    <span class="sm-value">₹{p['required_monthly_saving']:,.0f}/mo</span>
                </div>
                <div class="saved-metric-cell">
                    <span class="sm-label">TARGET DEADLINE</span>
                    <span class="sm-value">{p['deadline_months']} Months</span>
                </div>
                <div class="saved-metric-cell highlight">
                    <span class="sm-label">MONTHLY DISCRETIONARY TRIM</span>
                    <span class="sm-value highlight">₹{p['total_monthly_cut']:,.0f}/mo</span>
                </div>
            </div>

            <div class="saved-plan-strats-section">
                <span class="strats-header-label">Approved Strategies Included:</span>
                <div class="strats-tags-wrap">
                    {strategies_html if strategies_html else '<span class="plan-strat-tag">Custom savings plan</span>'}
                </div>
            </div>
        </div>
        """

    return f"""
    <div class="saved-plans-container">
        {items_html}
    </div>
    """


def render_account_html(user: dict, profile: dict) -> str:
    """Renders the account and profile details card."""
    if not user:
        return ""
    name = user.get("full_name", "")
    email = user.get("email", "")
    created = user.get("created_at", "")
    income = float(profile.get("monthly_income", 0))

    return f"""
    <div class="account-profile-card">
        <div class="account-header">
            <div class="account-avatar">{name[:1].upper() if name else 'U'}</div>
            <div class="account-title-wrap">
                <h2 class="account-name">{name}</h2>
                <p class="account-email">{email}</p>
            </div>
        </div>

        <div class="account-detail-grid">
            <div class="account-detail-item">
                <span class="detail-label">ACCOUNT STATUS</span>
                <span class="detail-val active">● Active Member</span>
            </div>
            <div class="account-detail-item">
                <span class="detail-label">MEMBER SINCE</span>
                <span class="detail-val">{created}</span>
            </div>
            <div class="account-detail-item">
                <span class="detail-label">RECORDED MONTHLY INCOME</span>
                <span class="detail-val">₹{income:,.0f}</span>
            </div>
            <div class="account-detail-item">
                <span class="detail-label">AUTHENTICATION LEVEL</span>
                <span class="detail-val">PBKDF2-HMAC-SHA256 Encrypted</span>
            </div>
            <div class="account-detail-item">
                <span class="detail-label">DATA ISOLATION</span>
                <span class="detail-val">Private User Partition (Neon PostgreSQL)</span>
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Streaming Plan Execution (Unchanged Multi-Agent AI Pipeline)
# ---------------------------------------------------------------------------
def run_analysis_stream(
    user_name, income, rent, groceries, utilities, debt_emi, transport,
    dining_out, shopping, entertainment, subscriptions, goal_amount,
    goal_deadline, existing_savings, objective_text, force_rejection
):
    """Streams plan generation with human progress, updating calculation details in background."""
    profile = {
        "user_name": user_name or "User",
        "monthly_income": float(income),
        "expenses": {
            "rent": float(rent),
            "groceries": float(groceries),
            "utilities": float(utilities),
            "debt_emi": float(debt_emi),
            "transport": float(transport),
            "dining_out": float(dining_out),
            "shopping": float(shopping),
            "entertainment": float(entertainment),
            "subscriptions": float(subscriptions),
        },
        "goal_amount": float(goal_amount),
        "goal_deadline_months": int(goal_deadline),
        "existing_savings": float(existing_savings),
    }

    final_state = None
    step_index = 0

    for state in run_pipeline_generator(
        profile=profile,
        objective_text=objective_text,
        max_revisions=2,
        force_demo_rejection=bool(force_rejection),
    ):
        final_state = state
        CURRENT_SESSION_STATE["state"] = state
        
        findings = state.get("findings", {})
        if "savings" in findings:
            step_index = 4
        elif "goal" in findings:
            step_index = 3
        elif "budget" in findings:
            step_index = 2
        elif "expense" in findings:
            step_index = 1
        else:
            step_index = 0

        raw_log = "\n\n".join(
            f"{i+1:02d}. {line}" for i, line in enumerate(state.get("reasoning_log", []))
        )

        status_note = None
        if state.get("status") == "safety_rejected":
            status_note = "Fine-tuning recommended cuts to protect your essential spending floor..."
        
        progress_html = render_human_progress(active_idx=step_index, is_complete=False, status_note=status_note)
        yield progress_html, "", "", None, None, raw_log, state
        time.sleep(0.12)

    # Completed state
    has_rev = (
        final_state.get("constraints", {}).get("revision_count", 0) > 0
        or any("rejection" in l.lower() or "revised" in l.lower() for l in final_state.get("reasoning_log", []))
    )

    progress_html = render_human_progress(active_idx=5, is_complete=True)
    hero_html = render_plan_hero(final_state)
    rec_html = format_recommendations_html(final_state.get("validated_strategies", []), has_revision=has_rev)
    exp_fig = create_expense_chart(final_state)
    goal_fig = create_goal_chart(final_state)

    raw_log = "\n\n".join(
        f"{i+1:02d}. {line}" for i, line in enumerate(final_state.get("reasoning_log", []))
    )

    yield progress_html, hero_html, rec_html, exp_fig, goal_fig, raw_log, final_state


def simulate_what_if(dining_cut_pct, shopping_cut_pct, sub_cut_pct, income_change_pct, session_user, current_profile):
    """Simulates custom financial adjustments and projects new trajectory."""
    profile = current_profile if current_profile else {
        "monthly_income": 60000, "rent": 18000, "groceries": 9000, "utilities": 3500,
        "debt_emi": 5000, "transport": 3500, "dining_out": 5000, "shopping": 4000,
        "subscriptions": 1500, "existing_savings": 30000, "goal_amount": 150000
    }
    
    base_income = float(profile.get("monthly_income", 60000))
    adj_income = round(base_income * (1.0 + (income_change_pct / 100.0)), 2)
    
    dining_base = float(profile.get("dining_out", 5000))
    shopping_base = float(profile.get("shopping", 4000))
    sub_base = float(profile.get("subscriptions", 1500))

    r_dining = simulate_category_reduction("dining_out", dining_base, dining_cut_pct)
    r_shop = simulate_category_reduction("shopping", shopping_base, shopping_cut_pct)
    r_sub = simulate_category_reduction("subscriptions", sub_base, sub_cut_pct)
    
    combo = combine_strategies([r_dining, r_shop, r_sub])
    total_trimmed = combo["total_monthly_saving"]
    
    orig_expenses = (
        float(profile.get("rent", 18000)) + float(profile.get("groceries", 9000)) +
        float(profile.get("utilities", 3500)) + float(profile.get("debt_emi", 5000)) +
        float(profile.get("transport", 3500)) + dining_base + shopping_base +
        float(profile.get("entertainment", 2500)) + sub_base
    )
    new_expenses = round(orig_expenses - total_trimmed, 2)
    new_budget = calculate_savings_rate(adj_income, new_expenses)
    
    goal_amt = float(profile.get("goal_amount", 150000))
    cur_sav = float(profile.get("existing_savings", 30000))
    proj = project_completion_date(new_budget["current_monthly_savings"], goal_amt, cur_sav)
    
    # Optionally persist scenario if authenticated
    if session_user and "id" in session_user:
        database.save_scenario_history(
            user_id=session_user["id"],
            dining_cut=dining_cut_pct,
            shopping_cut=shopping_cut_pct,
            sub_cut=sub_cut_pct,
            income_change=income_change_pct,
            projected_monthly_saving=new_budget["current_monthly_savings"],
            projected_timeline=proj["projected_timeline_str"]
        )

    report = f"""
### Scenario Projection Results

- **Adjusted Monthly Income:** ₹{adj_income:,.2f} ({income_change_pct:+0.1f}%)
- **Monthly Spending Trimmed:** ₹{total_trimmed:,.2f}
  - Dining moderation ({dining_cut_pct}%): ₹{r_dining['monthly_saving']:,.2f}
  - Shopping adjustment ({shopping_cut_pct}%): ₹{r_shop['monthly_saving']:,.2f}
  - Subscriptions audit ({sub_cut_pct}%): ₹{r_sub['monthly_saving']:,.2f}
- **New Net Monthly Savings:** ₹{new_budget['current_monthly_savings']:,.2f} (Savings Rate: **{new_budget['savings_rate_percent']}%**)
- **Annualized Extra Savings:** ₹{combo['annual_saving']:,.2f} per year
- **Goal Completion Projection:** {proj['projected_timeline_str']}
    """
    
    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    categories = ["Current Savings", "Simulated Savings"]
    orig_net = max(0.0, base_income - orig_expenses)
    values = [orig_net, max(0.0, new_budget["current_monthly_savings"])]
    
    bars = ax.bar(categories, values, color=["#1C232B", "#C97A3E"], width=0.42)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#8A8578", labelsize=9, left=False, bottom=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#F0EDE5")
    ax.set_ylabel("Monthly Surplus (₹)", color="#8A8578", fontsize=8.5)
    
    max_v = max(values) if values else 1
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_v * 0.03, f"₹{h:,.0f}",
                ha="center", va="bottom", color="#1C232B", fontsize=8.5, weight="bold")
                
    plt.tight_layout()
    return report, fig


# ---------------------------------------------------------------------------
# Custom Design Tokens & CSS
# ---------------------------------------------------------------------------
custom_css = """
/* Reset color scheme & force light canvas across all containers */
:root, html, body, .gradio-container, gradio-app,
:root.dark, html.dark, body.dark, .gradio-container.dark {
    color-scheme: light !important;
    background-color: #FAF9F6 !important;
    background: #FAF9F6 !important;
    color: #1C232B !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    /* CSS variables — NOTE: !important is NOT valid on custom properties.
       These must be set without it. Gradio 6 reads these to color labels/inputs. */
    --body-text-color: #000000;
    --block-label-text-color: #000000;
}

/* Global light surface override for all Gradio blocks, groups, cards, forms */
.block, .dark .block,
.form, .dark .form,
fieldset, .dark fieldset,
.gr-panel, .dark .gr-panel,
.gr-box, .dark .gr-box,
.gr-group, .dark .gr-group,
.section-card, .dark .section-card,
.group, .dark .group,
.card, .dark .card {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    border: 1px solid #EDEAE2 !important;
    border-color: #EDEAE2 !important;
    border-radius: 12px !important;
    color: #1C232B !important;
    box-shadow: none !important;
}

/* Section Card Containers */
.section-card {
    padding: 16px 20px !important;
    margin-bottom: 20px !important;
    background: #FFFFFF !important;
    border: 1px solid #EDEAE2 !important;
    border-radius: 12px !important;
}

/* High-contrast, clearly spaced Section Headings */
.section-heading-label {
    color: #1C232B !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    margin: 18px 0 12px 0 !important;
    padding: 0 !important;
    display: block !important;
    line-height: 1.4 !important;
    letter-spacing: -0.01em !important;
}

/* Brand Header */
.brand-header-box {
    background: #FFFFFF !important;
    border: 1px solid #EDEAE2 !important;
    border-radius: 12px !important;
    padding: 18px 24px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.brand-title {
    font-size: 22px;
    font-weight: 700;
    color: #1C232B;
    margin: 0;
    letter-spacing: -0.01em;
}
.brand-subtitle {
    font-size: 13px;
    color: #8A8578;
    margin: 4px 0 0 0;
}
.security-status-badge {
    background: #FFFFFF;
    color: #3A7A5A;
    border: 1px solid #3A7A5A;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ai-status-pill {
    background: #FAF9F6;
    color: #C97A3E;
    border: 1px solid #C97A3E;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Primary Action Buttons */
button.primary-action-btn, button.gr-button-primary {
    background-color: #C97A3E !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    transition: background-color 0.15s ease !important;
}
button.primary-action-btn:hover, button.gr-button-primary:hover {
    background-color: #B3672F !important;
}

/* Secondary Buttons */
button.secondary-action-btn, button.gr-button-secondary {
    background-color: #FFFFFF !important;
    border: 1px solid #EDEAE2 !important;
    color: #1C232B !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
button.secondary-action-btn:hover, button.gr-button-secondary:hover {
    background-color: #F0EDE5 !important;
    border-color: #EDEAE2 !important;
}

/* ==========================================================================
   NAVIGATION HOVER & ACTIVE STYLING (BRIGHT ORANGE, NEVER BLACK)
   ========================================================================== */
.tabs {
    border-bottom: 1px solid #EDEAE2 !important;
    margin-bottom: 18px !important;
}
.tab-nav button, .tabs button {
    font-size: 14.5px !important;
    font-weight: 600 !important;
    color: #5C5648 !important;
    padding: 13px 20px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.18s ease, border-color 0.18s ease, background-color 0.18s ease !important;
}
.tab-nav button:hover, .tabs button:hover {
    color: #C97A3E !important; /* Bright orange hover */
    background-color: #FAF6F0 !important; /* Gentle light orange tint */
    border-bottom: 2px solid #C97A3E !important;
}
.tab-nav button.selected, .tabs button.selected {
    color: #C97A3E !important;
    border-bottom: 2px solid #C97A3E !important;
    background: transparent !important;
    font-weight: 700 !important;
}

/* ==========================================================================
   DROPDOWN READABILITY FIX (WHITE BACKGROUND, DARK TEXT, ORANGE HOVER)
   ========================================================================== */
.gr-dropdown, .dark .gr-dropdown,
div[data-testid="dropdown"],
ul[role="listbox"],
.dropdown-menu,
.options,
.gr-select,
select,
option {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #1C232B !important;
    border-color: #EDEAE2 !important;
}

ul[role="listbox"] li,
.item,
.option,
[role="option"] {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #1C232B !important;
    padding: 10px 14px !important;
    font-size: 13.5px !important;
}

ul[role="listbox"] li:hover,
.item:hover,
.option:hover,
[role="option"]:hover,
[role="option"].highlighted,
.item.selected {
    background-color: #FDF5EC !important;
    background: #FDF5EC !important;
    color: #C97A3E !important;
    font-weight: 600 !important;
}

div[data-testid="dropdown"] ul,
.gr-dropdown ul,
[role="listbox"] {
    z-index: 9999 !important;
    box-shadow: 0 8px 24px rgba(28, 35, 43, 0.12) !important;
    border: 1px solid #EDEAE2 !important;
    border-radius: 8px !important;
    max-height: 280px !important;
    overflow-y: auto !important;
}

/* All Inputs */
.gr-input, .dark .gr-input,
input[type="text"], .dark input[type="text"],
input[type="number"], .dark input[type="number"],
input[type="password"], .dark input[type="password"],
textarea, .dark textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #EDEAE2 !important;
    border-radius: 8px !important;
    color: #1C232B !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
}
input:focus, textarea:focus {
    border-color: #C97A3E !important;
    box-shadow: 0 0 0 1px #C97A3E !important;
}

/* ==========================================================================
   PUBLIC LANDING & AUTHENTICATION STYLES
   ========================================================================== */
.landing-hero-card {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 16px;
    padding: 40px 36px;
    text-align: center;
    margin-bottom: 24px;
}
.landing-badge {
    background: #FAF6F0;
    color: #C97A3E;
    border: 1px solid #EBD5C2;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 14px;
}
.landing-title {
    font-size: 32px;
    font-weight: 800;
    color: #1C232B;
    margin: 0 0 12px 0;
    letter-spacing: -0.02em;
}
.landing-desc {
    font-size: 16px;
    color: #5C5648;
    max-width: 680px;
    margin: 0 auto 24px auto;
    line-height: 1.6;
}
.landing-feature-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
    margin-bottom: 28px;
}
.feature-box {
    background: #FAF9F6;
    border: 1px solid #EDEAE2;
    border-radius: 12px;
    padding: 22px 20px;
    text-align: left;
}
.feature-box-icon {
    font-size: 24px;
    margin-bottom: 8px;
}
.feature-box-title {
    font-size: 15px;
    font-weight: 700;
    color: #1C232B;
    margin: 0 0 6px 0;
}
.feature-box-desc {
    font-size: 13px;
    color: #8A8578;
    line-height: 1.5;
    margin: 0;
}

.auth-container-card {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 16px;
    padding: 30px;
    max-width: 520px;
    margin: 0 auto 40px auto;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
}

/* ── Auth page: Sign In + Sign Up ──────────────────────────────────────
   Root cause fix: Override the CSS variables that Gradio's Svelte components
   actually read, scoped to the auth container. This beats any specificity fight
   because the variables resolve at the point of use.

   Confirmed Gradio 6.9.0 variable usage:
   - Field labels:  label.svelte-19djge9  reads --block-label-text-color
   - Input values:  input/textarea.svelte-1hguek3  reads --body-text-color
   - Placeholders:  ::placeholder reads --input-placeholder-color (untouched)
─────────────────────────────────────────────────────────────────────── */

/* Set CSS variables on the auth container — NO !important needed on vars.
   Gradio's Svelte components read var(--block-label-text-color) for labels
   and var(--body-text-color) for input value text. Setting them here
   overrides :root values for all descendants. */
.auth-container-card,
.auth-container-card * {
    --block-label-text-color: #000000;
    --body-text-color: #000000;
}

/* Belt + suspenders: also target directly with Svelte hashes */
span.svelte-19djge9,
label.svelte-19djge9 {
    color: #000000 !important;
}

textarea.svelte-1hguek3,
input[type="password"].svelte-1hguek3,
input[type="text"].svelte-1hguek3,
input[type="email"].svelte-1hguek3 {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Placeholder: explicitly restore grey so -webkit-text-fill-color above doesn't bleed in */
textarea.svelte-1hguek3::placeholder,
input.svelte-1hguek3::placeholder {
    -webkit-text-fill-color: unset !important;
    color: var(--input-placeholder-color) !important;
}

/* ── Tab nav (Sign In / Create Account tab text) */
.auth-container-card .tab-nav button,
.auth-container-card .tabs > div > button {
    color: #1C232B !important;
}
.auth-container-card .tab-nav button.selected,
.auth-container-card .tabs > div > button.selected {
    color: #C97A3E !important;
    border-bottom-color: #C97A3E !important;
}

/* ── Auth buttons: solid orange, always visible, compact, centered ── */

/* Override Gradio's primary button variant directly.
   Gradio 6 button uses variant="primary" which maps to --button-primary-background-fill.
   We also target by elem_classes="auth-compact-btn" for extra specificity. */
.auth-compact-btn,
.auth-container-card button[class*="auth-compact-btn"] {
    background: #C97A3E !important;
    background-color: #C97A3E !important;
    border: 2px solid #C97A3E !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    padding: 10px 36px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    width: auto !important;
    min-width: 130px !important;
    max-width: 200px !important;
    display: block !important;
    margin: 8px auto 0 auto !important;
    flex: 0 0 auto !important;
    box-shadow: 0 2px 8px rgba(201, 122, 62, 0.35) !important;
}
.auth-compact-btn *,
.auth-container-card button[class*="auth-compact-btn"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.auth-compact-btn:hover {
    background: #B3672F !important;
    background-color: #B3672F !important;
    border-color: #B3672F !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(201, 122, 62, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* Neutralise Gradio flex-stretch on button row */
.auth-container-card [class*="row"],
.auth-container-card .block.gr-Row {
    justify-content: center !important;
    align-items: center !important;
}
.auth-container-card [class*="row"] > div {
    flex: 0 0 auto !important;
    width: auto !important;
}

/* ==========================================================================
   DASHBOARD CARDS & WIDGETS
   ========================================================================== */
.dashboard-welcome-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.welcome-title {
    font-size: 20px;
    font-weight: 700;
    color: #1C232B;
    margin: 0;
}
.welcome-subtitle {
    font-size: 13px;
    color: #8A8578;
    margin: 4px 0 0 0;
}
.sync-pill {
    font-size: 12px;
    font-weight: 600;
    color: #3A7A5A;
    background: #F2F8F4;
    padding: 4px 12px;
    border-radius: 12px;
    border: 1px solid #CBE2D3;
}

.dashboard-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.dash-card {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 12px;
    padding: 18px 20px;
}
.dash-card-accent {
    border-color: #C97A3E;
    background: #FCFBF9;
}
.dash-card-caption {
    font-size: 11px;
    font-weight: 700;
    color: #8A8578;
    letter-spacing: 0.04em;
}
.dash-card-val {
    font-size: 22px;
    font-weight: 800;
    color: #1C232B;
    margin: 6px 0 4px 0;
    font-variant-numeric: tabular-nums;
}
.dash-card-val.positive { color: #3A7A5A; }
.dash-card-val.negative { color: #A8532F; }
.dash-card-val.highlight { color: #C97A3E; }
.dash-card-val.accent { color: #C97A3E; }
.dash-card-foot {
    font-size: 11.5px;
    color: #8A8578;
}

.dash-goal-box {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 24px;
}
.dash-goal-box.empty {
    text-align: center;
    padding: 36px 24px;
}
.empty-goal-icon { font-size: 32px; margin-bottom: 8px; }
.empty-goal-title { font-size: 16px; font-weight: 700; color: #1C232B; margin: 0 0 6px 0; }
.empty-goal-sub { font-size: 13px; color: #8A8578; margin: 0; }

.goal-header-line {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}
.goal-tag {
    font-size: 10.5px;
    font-weight: 700;
    color: #C97A3E;
    letter-spacing: 0.05em;
    display: block;
}
.goal-main-title {
    font-size: 17px;
    font-weight: 700;
    color: #1C232B;
    margin: 4px 0 0 0;
}
.goal-status-badge {
    background: #FAF6F0;
    border: 1px solid #EBD5C2;
    color: #C97A3E;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 12px;
}
.goal-stat-row {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #F0EDE5;
}
.goal-mini-stat {
    display: flex;
    flex-direction: column;
}
.mini-caption {
    font-size: 10.5px;
    font-weight: 600;
    color: #8A8578;
}
.mini-val {
    font-size: 15px;
    font-weight: 700;
    color: #1C232B;
    margin-top: 2px;
}
.mini-val.highlight { color: #A8532F; }
.mini-val.accent { color: #3A7A5A; }

/* ==========================================================================
   SAVED PLANS STYLING
   ========================================================================== */
.saved-plans-container {
    display: flex;
    flex-direction: column;
    gap: 18px;
    margin-top: 12px;
}
.saved-plan-card {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 12px;
    padding: 22px 24px;
    transition: border-color 0.2s ease;
}
.saved-plan-card:hover {
    border-color: #C97A3E;
}
.saved-plan-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}
.saved-plan-title {
    font-size: 18px;
    font-weight: 700;
    color: #1C232B;
    margin: 0;
}
.saved-plan-meta {
    font-size: 12px;
    color: #8A8578;
    margin: 4px 0 0 0;
}
.saved-plan-badge {
    background: #FFFFFF;
    border: 1px solid #3A7A5A;
    color: #3A7A5A;
    font-size: 11.5px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
}
.saved-plan-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    background: #FAF9F6;
    border: 1px solid #EDEAE2;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
}
.saved-metric-cell {
    display: flex;
    flex-direction: column;
}
.saved-metric-cell.highlight .sm-value {
    color: #C97A3E;
}
.sm-label {
    font-size: 10.5px;
    font-weight: 600;
    color: #8A8578;
}
.sm-value {
    font-size: 15px;
    font-weight: 700;
    color: #1C232B;
    margin-top: 2px;
}
.saved-plan-strats-section {
    margin-top: 10px;
}
.strats-header-label {
    font-size: 11.5px;
    font-weight: 600;
    color: #8A8578;
    margin-bottom: 6px;
    display: block;
}
.strats-tags-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.plan-strat-tag {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #1C232B;
}

.empty-state-box {
    text-align: center;
    padding: 48px 24px;
    background: #FFFFFF;
    border: 1px dashed #D8D3C7;
    border-radius: 12px;
    margin-top: 14px;
}
.empty-state-icon { font-size: 40px; margin-bottom: 12px; }
.empty-state-title { font-size: 18px; font-weight: 700; color: #1C232B; margin: 0 0 6px 0; }
.empty-state-desc { font-size: 13.5px; color: #8A8578; max-width: 480px; margin: 0 auto; }

/* ==========================================================================
   ACCOUNT PROFILE VIEW
   ========================================================================== */
.account-profile-card {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 12px;
    padding: 28px;
    margin-top: 12px;
}
.account-header {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 24px;
    padding-bottom: 20px;
    border-bottom: 1px solid #EDEAE2;
}
.account-avatar {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: #FAF6F0;
    color: #C97A3E;
    border: 2px solid #C97A3E;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: 800;
}
.account-name { font-size: 22px; font-weight: 700; color: #1C232B; margin: 0; }
.account-email { font-size: 14px; color: #8A8578; margin: 4px 0 0 0; }
.account-detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
}
.account-detail-item {
    background: #FAF9F6;
    border: 1px solid #EDEAE2;
    border-radius: 8px;
    padding: 14px 16px;
}
.detail-label { font-size: 11px; font-weight: 600; color: #8A8578; }
.detail-val { font-size: 14.5px; font-weight: 700; color: #1C232B; margin-top: 4px; display: block; }
.detail-val.active { color: #3A7A5A; }

/* Existing Human Progress & Recommendation Styles */
.human-progress-card {
    background: #FFFFFF;
    border: 1px solid #EDEAE2;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
}
.progress-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
}
.progress-card-title { font-size: 16px; font-weight: 600; color: #1C232B; margin: 0; }
.progress-card-subtitle { font-size: 12.5px; color: #8A8578; margin: 2px 0 0 0; }
.status-pill.complete {
    background: #FFFFFF;
    color: #3A7A5A;
    border: 1px solid #3A7A5A;
    font-size: 11.5px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 12px;
}
.status-pill.active {
    background: #FAF9F6;
    color: #C97A3E;
    border: 1px solid #C97A3E;
    font-size: 11.5px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 12px;
}
.progress-steps-list { display: flex; flex-direction: column; gap: 12px; }
.progress-step-row { display: flex; align-items: flex-start; gap: 14px; }
.step-indicator-col { width: 22px; display: flex; justify-content: center; margin-top: 1px; }
.step-check {
    width: 20px; height: 20px; border-radius: 50%;
    background: #3A7A5A; color: #FFFFFF;
    font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.step-pulse {
    width: 20px; height: 20px; border-radius: 50%;
    background: #F0EDE5; border: 2px solid #C97A3E;
}
.step-dot {
    width: 20px; height: 20px; border-radius: 50%;
    background: #F0EDE5; border: 1px solid #EDEAE2;
}
.step-label-title { font-size: 13.5px; font-weight: 500; color: #1C232B; }
.step-pending .step-label-title { color: #8A8578; }
.step-sub-desc { font-size: 11.5px; color: #8A8578; margin-top: 2px; }
.progress-note {
    margin-top: 14px; padding: 8px 12px;
    background: #FAF9F6; border-left: 3px solid #C97A3E;
    border-radius: 4px; font-size: 12.5px; color: #1C232B;
}

.hero-plan-dashboard { margin-bottom: 24px; }
.hero-metric-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px; margin-bottom: 16px;
}
.hero-metric-cell {
    background: #FFFFFF; border: 1px solid #EDEAE2;
    border-radius: 12px; padding: 18px 20px;
}
.hero-metric-cell.hero-highlight { border: 1px solid #C97A3E; }
.metric-cell-label { font-size: 11px; font-weight: 600; color: #8A8578; letter-spacing: 0.04em; }
.metric-cell-amount {
    font-size: 26px; font-weight: 700; color: #1C232B;
    margin: 4px 0; font-variant-numeric: tabular-nums;
}
.metric-cell-amount.positive { color: #3A7A5A; }
.metric-cell-sub { font-size: 12px; color: #8A8578; }

.goal-progress-card {
    background: #FFFFFF; border: 1px solid #EDEAE2;
    border-radius: 12px; padding: 16px 20px;
}
.goal-progress-header { display: flex; justify-content: space-between; align-items: center; }
.goal-progress-title { font-size: 13px; font-weight: 600; color: #1C232B; }
.goal-progress-badge { font-size: 12px; font-weight: 600; color: #C97A3E; }
.goal-progress-track {
    width: 100%; height: 10px; background-color: #F0EDE5;
    border-radius: 6px; overflow: hidden; margin: 8px 0;
}
.goal-progress-fill {
    height: 100%; background-color: #C97A3E;
    border-radius: 6px; transition: width 0.3s ease;
}
.goal-progress-footer { display: flex; justify-content: space-between; font-size: 11.5px; color: #8A8578; }
.goal-progress-footer b { color: #1C232B; }

.recommendations-container { margin-bottom: 24px; }
.recommendations-intro { margin-bottom: 16px; }
.recommendations-title { font-size: 17px; font-weight: 600; color: #1C232B; margin: 0; }
.recommendations-subtitle { font-size: 12.5px; color: #8A8578; margin: 4px 0 0 0; }
.recommendations-list { display: flex; flex-direction: column; gap: 14px; }
.recommendation-item-card {
    background: #FFFFFF; border: 1px solid #EDEAE2;
    border-radius: 12px; padding: 18px 22px;
}
.card-header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.card-title-text { font-size: 15px; font-weight: 600; color: #1C232B; }
.status-badge-realistic {
    background: #FFFFFF; color: #3A7A5A; font-size: 11px;
    font-weight: 600; padding: 3px 10px; border-radius: 12px; border: 1px solid #3A7A5A;
}
.card-action-text { font-size: 13.5px; font-weight: 500; color: #1C232B; margin-bottom: 6px; line-height: 1.45; }
.card-reason-text { font-size: 12.5px; color: #8A8578; margin-bottom: 14px; line-height: 1.45; }
.card-numbers-row {
    display: flex; align-items: center; gap: 20px;
    background: #FAF9F6; border: 1px solid #EDEAE2;
    border-radius: 8px; padding: 10px 16px;
}
.number-stat-item { display: flex; flex-direction: column; }
.stat-caption { font-size: 10.5px; color: #8A8578; font-weight: 600; }
.stat-value { font-size: 15px; font-weight: 700; color: #1C232B; font-variant-numeric: tabular-nums; margin-top: 2px; }
.stat-value-highlight { font-size: 15px; font-weight: 700; color: #3A7A5A; font-variant-numeric: tabular-nums; margin-top: 2px; }
.stat-divider { width: 1px; height: 24px; background: #EDEAE2; }
.revision-note-banner {
    margin-top: 12px; padding: 8px 12px; background: #FAF9F6;
    border-left: 3px solid #C97A3E; border-radius: 4px;
    font-size: 12px; color: #1C232B; display: flex; align-items: center; gap: 8px; line-height: 1.4;
}
.note-bullet { color: #C97A3E; font-size: 13px; }
.empty-recommendations-box {
    padding: 24px; background: #FFFFFF; border: 1px dashed #EDEAE2;
    border-radius: 12px; text-align: center; color: #8A8578; font-size: 13.5px;
}

/* Hide Gradio Branding */
footer, .footer, gradio-app footer, div[data-testid="footer"], .show-api, a[href*="gradio.app"], a[href*="/api"] {
    display: none !important; visibility: hidden !important; height: 0 !important;
}

:root, .dark, .gradio-container {
    --body-text-color: #1C232B !important;
    --body-text-color-subdued: #1C232B !important;
    --block-label-text-color: #1C232B !important;
    --block-title-text-color: #1C232B !important;
    --block-info-text-color: #1C232B !important;
    --input-placeholder-color: #6B6B6B !important;
    --button-primary-background-fill: #C97A3E !important;
    --button-primary-background-fill-hover: #B3672F !important;
    --button-primary-border-color: #C97A3E !important;
    --button-primary-border-color-hover: #B3672F !important;
    --button-primary-text-color: #FFFFFF !important;
    --button-primary-text-color-hover: #FFFFFF !important;
}

[data-testid="block-info"],
[data-testid="block-title"],
.label-wrap span,
.block-title,
.block-info,
label, label span, label *,
input, textarea,
input::placeholder, textarea::placeholder {
    color: #1C232B !important;
    -webkit-text-fill-color: #1C232B !important;
    opacity: 1 !important;
}

.auth-compact-btn,
button.auth-compact-btn,
button.primary.auth-compact-btn,
.auth-container-card button,
.auth-container-card button.primary,
.auth-container-card button[class*="auth-compact-btn"] {
    background: #C97A3E !important;
    background-color: #C97A3E !important;
    background-image: none !important;
    border: 2px solid #C97A3E !important;
    border-color: #C97A3E !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    filter: none !important;
}
.auth-compact-btn *,
button.auth-compact-btn *,
.auth-container-card button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.auth-compact-btn:hover,
button.auth-compact-btn:hover,
button.primary.auth-compact-btn:hover,
.auth-container-card button:hover,
.auth-container-card button.primary:hover,
.auth-container-card button[class*="auth-compact-btn"]:hover {
    background: #B3672F !important;
    background-color: #B3672F !important;
    background-image: none !important;
    border-color: #B3672F !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus,
textarea:-webkit-autofill {
    -webkit-text-fill-color: #1C232B !important;
    -webkit-box-shadow: 0 0 0 1000px #FFFFFF inset !important;
    box-shadow: 0 0 0 1000px #FFFFFF inset !important;
}
"""

# Custom Theme object for Gradio base tokens
custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
).set(
    body_background_fill="#FAF9F6",
    body_background_fill_dark="#FAF9F6",
    body_text_color="#000000",
    body_text_color_dark="#000000",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_color="#EDEAE2",
    block_border_color_dark="#EDEAE2",
    block_radius="12px",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#EDEAE2",
    input_border_color_dark="#EDEAE2",
    button_primary_background_fill="#C97A3E",
    button_primary_background_fill_hover="#B3672F",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F0EDE5",
    button_secondary_border_color="#EDEAE2",
    button_secondary_text_color="#1C232B",
)

# ---------------------------------------------------------------------------
# Gradio Interface Assembly
# ---------------------------------------------------------------------------
with gr.Blocks(title="Personal Savings & Cash Flow Advisory") as demo:
    # State tracking per session
    session_user = gr.State(None)
    current_profile = gr.State(None)
    active_generated_plan = gr.State(None)

    # CSS & JS injection
    gr.HTML(f"""
    <script>
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    </script>
    <style>
        {custom_css}
    </style>
    """)

    # -----------------------------------------------------------------------
    # 1. PUBLIC LANDING & AUTHENTICATION VIEW (Visible when session_user is None)
    # -----------------------------------------------------------------------
    with gr.Column(visible=True) as landing_container:
        gr.HTML("""
        <div class="landing-hero-card">
            <span class="landing-badge">✨ Autonomous Multi-Agent AI System</span>
            <h1 class="landing-title">Personal Savings & Cash Flow Advisory</h1>
            <p class="landing-desc">
                Intelligent, sustainable financial guidance powered by autonomous multi-agent reasoning. 
                Experience guaranteed arithmetic precision with zero hallucinated math, tailored to your real life.
            </p>
            <div class="landing-feature-strip">
                <div class="feature-box">
                    <div class="feature-box-icon">🔍</div>
                    <h3 class="feature-box-title">Expense Auditing</h3>
                    <p class="feature-box-desc">Automated categorization of essential commitments vs flexible lifestyle leaks.</p>
                </div>
                <div class="feature-box">
                    <div class="feature-box-icon">🛡️</div>
                    <h3 class="feature-box-title">Essential Floor Protection</h3>
                    <p class="feature-box-desc">Safety agent automatically rejects aggressive or unrealistic cuts to protect your wellbeing.</p>
                </div>
                <div class="feature-box">
                    <div class="feature-box-icon">🎯</div>
                    <h3 class="feature-box-title">Goal Velocity</h3>
                    <p class="feature-box-desc">Exact timeline modeling to hit emergency reserves, home downpayments, or major targets.</p>
                </div>
            </div>
        </div>
        """)

        with gr.Group(elem_classes=["auth-container-card"]):
            gr.HTML("<h3 style='margin:0 0 16px 0; color:#1C232B; font-weight:700; font-size:18px;'>Access Your Financial Dashboard</h3>")
            
            with gr.Tabs() as auth_tabs:
                with gr.TabItem("Sign In", id="tab_signin"):
                    login_email_input = gr.Textbox(label="Email Address", value="demo@finance.ai", placeholder="you@example.com")
                    login_password_input = gr.Textbox(label="Password", type="password", value="password123")
                    with gr.Row():
                        login_btn = gr.Button("Sign In", variant="primary", elem_classes=["auth-compact-btn"])
                    login_status_msg = gr.Markdown("")

                with gr.TabItem("Create Account", id="tab_signup"):
                    reg_name_input = gr.Textbox(label="Full Name", placeholder="e.g. Maya Lin")
                    reg_email_input = gr.Textbox(label="Email Address", placeholder="maya@example.com")
                    reg_pass1_input = gr.Textbox(label="Password (min 6 characters)", type="password")
                    reg_pass2_input = gr.Textbox(label="Confirm Password", type="password")
                    with gr.Row():
                        signup_btn = gr.Button("Sign Up", variant="primary", elem_classes=["auth-compact-btn"])
                    signup_status_msg = gr.Markdown("")

    # -----------------------------------------------------------------------
    # 2. AUTHENTICATED APPLICATION VIEW (Visible when session_user is set)
    # -----------------------------------------------------------------------
    with gr.Column(visible=False) as app_container:
        # Header Box with User Status
        with gr.Row(elem_classes=["brand-header-box"]):
            with gr.Column(scale=3):
                gr.HTML("""
                <h1 class='brand-title'>Personal Savings & Cash Flow Advisory</h1>
                <p class='brand-subtitle'>Personalized, sustainable financial planning tailored to your goals.</p>
                """)
            with gr.Column(scale=2):
                with gr.Row():
                    gr.HTML("""
                    <div style="display:flex; gap:10px; align-items:center; justify-content:flex-end; flex-wrap:wrap;">
                        <span class="ai-status-pill">⚡ Groq AI Active (openai/gpt-oss-20b)</span>
                        <span class="security-status-badge"><span>●</span> Essential Floor Active</span>
                    </div>
                    """)
                    logout_btn = gr.Button("Sign Out", variant="secondary", size="sm")

        # Main Navigation Tabs
        with gr.Tabs() as main_tabs:
            # ---------------------------------------------------------------
            # TAB 1: CENTRAL DASHBOARD
            # ---------------------------------------------------------------
            with gr.TabItem("Dashboard", id="tab_dash"):
                dashboard_hero_display = gr.HTML("")
                
                with gr.Row():
                    dash_to_finances_btn = gr.Button("📝 Update Your Finances", variant="secondary")
                    dash_to_plan_btn = gr.Button("🚀 Build Savings Plan", variant="primary")
                    dash_to_sim_btn = gr.Button("📈 Test Scenarios", variant="secondary")
                    dash_to_saved_btn = gr.Button("💾 View Saved Plans", variant="secondary")

                gr.HTML('<div class="section-heading-label">Cash Flow Overview & Distribution</div>')
                with gr.Row():
                    dash_cashflow_plot = gr.Plot(label="Monthly Cash Flow Dynamics")
                    dash_breakdown_plot = gr.Plot(label="Essential vs Flexible Split")

                gr.HTML('<div class="section-heading-label">Active Savings Goal Tracking</div>')
                dashboard_goal_display = gr.HTML("")

            # ---------------------------------------------------------------
            # TAB 2: YOUR FINANCES (Sample Profile Dropdown Completely Removed!)
            # ---------------------------------------------------------------
            with gr.TabItem("Your Finances", id="tab_profile"):
                gr.HTML("<p style='color:#8A8578; font-size:13.5px; margin-bottom:14px;'>Manage your custom financial numbers below. These are stored securely in your private account.</p>")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-heading-label">Profile & Monthly Income</div>')
                        with gr.Group(elem_classes=["section-card"]):
                            user_name_input = gr.Textbox(label="Full Name / Profile Label", value="Alex Morgan")
                            monthly_income_input = gr.Number(label="Monthly Net Income (₹)", value=75000)

                        gr.HTML('<div class="section-heading-label">Essential Fixed Commitments</div>')
                        with gr.Group(elem_classes=["section-card"]):
                            rent_input = gr.Number(label="Housing / Rent (₹)", value=20000)
                            groceries_input = gr.Number(label="Groceries & Food (₹)", value=9500)
                            utilities_input = gr.Number(label="Utilities (Power, Water, Net) (₹)", value=4000)
                            debt_input = gr.Number(label="Debt / Loan EMI (₹)", value=6000)

                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-heading-label">Flexible Lifestyle Spending</div>')
                        with gr.Group(elem_classes=["section-card"]):
                            transport_input = gr.Number(label="Transport (₹)", value=3500)
                            dining_input = gr.Number(label="Dining & Delivery (₹)", value=7000)
                            shopping_input = gr.Number(label="Shopping & Retail (₹)", value=5000)
                            entertainment_input = gr.Number(label="Entertainment (₹)", value=3000)
                            subscriptions_input = gr.Number(label="Subscriptions (₹)", value=2000)

                        gr.HTML('<div class="section-heading-label">Savings Target & Reserve</div>')
                        with gr.Group(elem_classes=["section-card"]):
                            goal_amount_input = gr.Number(label="Target Goal Amount (₹)", value=200000)
                            goal_deadline_input = gr.Number(label="Target Horizon (Months)", value=12)
                            existing_savings_input = gr.Number(label="Current Reserve (₹)", value=45000)
                            objective_input = gr.Textbox(
                                label="Personal Financial Objective",
                                value="Build a 6-month safety net while preserving essential comforts.",
                                placeholder="e.g. Save Rs 8,000 extra per month, do not adjust groceries."
                            )
                            force_reject_cb = gr.Checkbox(
                                label="Stress-test plan sustainability",
                                value=False,
                                info="Tests safety constraints against aggressive targets to ensure adjustments remain realistic."
                            )

                with gr.Row():
                    save_profile_btn = gr.Button("💾 Save Financial Profile", variant="secondary", scale=1)
                    proceed_to_plan_btn = gr.Button("🚀 Proceed to Savings Plan", variant="primary", scale=2)
                profile_save_status = gr.Markdown("")

            # ---------------------------------------------------------------
            # TAB 3: SAVINGS PLANS (Hero Product Experience)
            # ---------------------------------------------------------------
            with gr.TabItem("Savings Plans", id="tab_plan"):
                with gr.Row():
                    run_btn = gr.Button("Build My Savings Plan", variant="primary", scale=2)
                
                progress_html_display = gr.HTML(
                    render_human_progress(active_idx=0, is_complete=False)
                )

                hero_metrics_display = gr.HTML("")

                recommendations_html_display = gr.HTML(
                    format_recommendations_html([])
                )

                gr.HTML('<div class="section-heading-label">Cash Flow & Goal Trajectory</div>')
                with gr.Row():
                    expense_chart_plot = gr.Plot(label="Spending Breakdown")
                    goal_chart_plot = gr.Plot(label="Savings Velocity")

                with gr.Row():
                    approve_plan_btn = gr.Button("💾 Save This Plan to My Account", variant="primary")
                    modify_goals_btn = gr.Button("Adjust Financial Goals", variant="secondary")
                    edit_finances_btn = gr.Button("Modify Spending Breakdown", variant="secondary")
                plan_action_status = gr.Markdown("")

                with gr.Accordion("View calculation details & multi-agent technical log", open=False):
                    agent_log_display = gr.Textbox(
                        label="Calculation Details & Audit Log",
                        lines=12,
                        interactive=False,
                        placeholder="Full calculation logs and safety audit records will appear here after building a plan..."
                    )

            # ---------------------------------------------------------------
            # TAB 4: SCENARIO PLANNER
            # ---------------------------------------------------------------
            with gr.TabItem("Scenario Planner", id="tab_simulator"):
                gr.HTML('<div class="section-heading-label">Test Spending Scenarios</div>')
                gr.Markdown("Adjust sliders below to recalculate cash flow and goal timelines without modifying your base profile.")
                
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["section-card"]):
                            sim_dining_slider = gr.Slider(0, 60, value=25, step=5, label="Adjust Dining & Delivery (%)")
                            sim_shopping_slider = gr.Slider(0, 50, value=20, step=5, label="Adjust Shopping & Retail (%)")
                            sim_sub_slider = gr.Slider(0, 80, value=40, step=10, label="Adjust Subscriptions (%)")
                            sim_income_slider = gr.Slider(-20, 30, value=0, step=5, label="Anticipated Income Change (%)")
                            sim_run_btn = gr.Button("Recalculate Scenario", variant="primary")
                    
                    with gr.Column():
                        with gr.Group(elem_classes=["section-card"]):
                            what_if_output_md = gr.Markdown("Adjust sliders and click 'Recalculate Scenario' to test outcomes.")
                            what_if_plot = gr.Plot(label="Simulated Savings Comparison")

            # ---------------------------------------------------------------
            # TAB 5: SAVED PLANS (Dedicated Section)
            # ---------------------------------------------------------------
            with gr.TabItem("Saved Plans", id="tab_saved_plans"):
                gr.HTML('<div class="section-heading-label">Your Saved Financial Plans</div>')
                gr.Markdown("Review previously approved savings strategies, milestones, and habit commitments.")
                saved_plans_html_display = gr.HTML("")

                with gr.Row():
                    refresh_saved_btn = gr.Button("🔄 Refresh Saved Plans", variant="secondary")
                    with gr.Row():
                        delete_plan_id_input = gr.Number(label="Plan ID to Delete", precision=0)
                        delete_plan_btn = gr.Button("🗑️ Delete Selected Plan", variant="secondary")
                delete_status_md = gr.Markdown("")

            # ---------------------------------------------------------------
            # TAB 6: PROFILE / ACCOUNT
            # ---------------------------------------------------------------
            with gr.TabItem("Account & Profile", id="tab_account"):
                gr.HTML('<div class="section-heading-label">Your Account Information</div>')
                account_html_display = gr.HTML("")
                with gr.Row():
                    account_logout_btn = gr.Button("Sign Out of Account", variant="secondary")

            # ---------------------------------------------------------------
            # TAB 7: FAQ & FINANCIAL INFORMATION BOT
            # ---------------------------------------------------------------
            with gr.TabItem("Financial Assistant & FAQ", id="tab_faq"):
                gr.HTML("""
                <div class="section-card" style="margin-bottom:15px; border-left: 4px solid #C97A3E;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div>
                            <div style="font-weight:700; font-size:15px; color:#1C232B;">Educational Financial Assistant & System Guide</div>
                            <div style="font-size:13px; color:#5C5648; margin-top:3px;">Ask questions about our 6-agent architecture, safety audit floors, scenario modeling, or general personal finance concepts.</div>
                        </div>
                        <span class="security-status-badge"><span>●</span> Privacy Guard Active (Zero Access to User Financial Records)</span>
                    </div>
                </div>
                """)

                faq_chatbot = gr.Chatbot(
                    label="Assistant Dialogue",
                    height=450,
                    show_label=False
                )

                gr.HTML('<div style="font-size:12.5px; font-weight:600; color:#8A8578; margin: 8px 0 4px 0;">Suggested Quick Questions:</div>')
                with gr.Row():
                    faq_chip1 = gr.Button("💡 What does this application do?", size="sm", variant="secondary")
                    faq_chip2 = gr.Button("🤖 What are the 6 autonomous agents?", size="sm", variant="secondary")
                    faq_chip3 = gr.Button("🛡️ What is an emergency fund?", size="sm", variant="secondary")
                    faq_chip4 = gr.Button("📊 What is the 50/30/20 rule?", size="sm", variant="secondary")

                with gr.Row():
                    faq_input = gr.Textbox(
                        placeholder="Ask a question about personal finance or how this advisory system works...",
                        label="Your Question",
                        scale=6,
                        lines=1,
                        max_lines=3,
                        container=False
                    )
                    faq_send_btn = gr.Button("Ask Assistant", variant="primary", scale=1)
                    faq_clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

                gr.HTML("""
                <div style="font-size:11.5px; color:#8A8578; margin-top:10px; text-align:center;">
                    ⚠️ <em>Educational guidance only. This assistant does not provide certified financial, investment, or tax advice. To build your personal savings plan, use the 'Build Savings Plan' tab.</em>
                </div>
                """)

    # -----------------------------------------------------------------------
    # Wire Authentication & Event Handlers
    # -----------------------------------------------------------------------
    def handle_login(email, password):
        user, err = database.authenticate_user(email, password)
        if err:
            return None, gr.update(visible=True), gr.update(visible=False), f"❌ {err}", *([gr.update()] * 21)
        
        # Load user profile
        prof = database.get_user_profile(user["id"])
        plans = database.get_user_plans(user["id"])
        latest_plan = plans[0] if plans else None

        hero_html = render_dashboard_metrics(user, prof, latest_plan)
        goal_html = render_dashboard_active_goal(prof)
        cash_fig = create_dashboard_cashflow_chart(prof)
        split_fig = create_dashboard_breakdown_chart(prof)
        saved_html = render_saved_plans_html(user["id"])
        acc_html = render_account_html(user, prof)

        return (
            user,
            gr.update(visible=False),  # hide landing
            gr.update(visible=True),   # show app
            "",                        # clear login status
            hero_html,
            goal_html,
            cash_fig,
            split_fig,
            saved_html,
            acc_html,
            prof,                      # current_profile state
            # Form field values from user profile:
            user["full_name"],
            prof.get("monthly_income", 60000.0),
            prof.get("rent", 18000.0),
            prof.get("groceries", 9000.0),
            prof.get("utilities", 3500.0),
            prof.get("debt_emi", 5000.0),
            prof.get("transport", 3500.0),
            prof.get("dining_out", 5000.0),
            prof.get("shopping", 4000.0),
            prof.get("entertainment", 2500.0),
            prof.get("subscriptions", 1500.0),
            prof.get("goal_amount", 150000.0),
            prof.get("goal_deadline_months", 12),
            prof.get("existing_savings", 30000.0),
            prof.get("objective", "Build emergency savings."),
        )

    login_btn.click(
        fn=handle_login,
        inputs=[login_email_input, login_password_input],
        outputs=[
            session_user, landing_container, app_container, login_status_msg,
            dashboard_hero_display, dashboard_goal_display, dash_cashflow_plot, dash_breakdown_plot,
            saved_plans_html_display, account_html_display, current_profile,
            user_name_input, monthly_income_input, rent_input, groceries_input,
            utilities_input, debt_input, transport_input, dining_input,
            shopping_input, entertainment_input, subscriptions_input,
            goal_amount_input, goal_deadline_input, existing_savings_input,
            objective_input
        ]
    )


    def handle_signup(name, email, pass1, pass2):
        if pass1 != pass2:
            return None, gr.update(visible=True), gr.update(visible=False), "❌ Passwords do not match.", *([gr.update()] * 21)
        user, err = database.create_user(email, pass1, name)
        if err:
            return None, gr.update(visible=True), gr.update(visible=False), f"❌ {err}", *([gr.update()] * 21)
        return handle_login(email, pass1)

    signup_btn.click(
        fn=handle_signup,
        inputs=[reg_name_input, reg_email_input, reg_pass1_input, reg_pass2_input],
        outputs=[
            session_user, landing_container, app_container, signup_status_msg,
            dashboard_hero_display, dashboard_goal_display, dash_cashflow_plot, dash_breakdown_plot,
            saved_plans_html_display, account_html_display, current_profile,
            user_name_input, monthly_income_input, rent_input, groceries_input,
            utilities_input, debt_input, transport_input, dining_input,
            shopping_input, entertainment_input, subscriptions_input,
            goal_amount_input, goal_deadline_input, existing_savings_input,
            objective_input
        ]
    )

    def handle_logout():
        return None, None, None, gr.update(visible=True), gr.update(visible=False), ""

    logout_btn.click(
        fn=handle_logout,
        inputs=[],
        outputs=[session_user, current_profile, active_generated_plan, landing_container, app_container, login_status_msg]
    )
    account_logout_btn.click(
        fn=handle_logout,
        inputs=[],
        outputs=[session_user, current_profile, active_generated_plan, landing_container, app_container, login_status_msg]
    )

    # Save Profile Handler
    def handle_save_profile(user, name, inc, rnt, gro, util, dbt, trn, din, shp, ent, sub, goal, dline, res, obj):
        if not user or "id" not in user:
            return "❌ Please sign in to save your profile.", None, None, None, None
        
        prof_data = {
            "monthly_income": inc, "rent": rnt, "groceries": gro, "utilities": util,
            "debt_emi": dbt, "transport": trn, "dining_out": din, "shopping": shp,
            "entertainment": ent, "subscriptions": sub, "goal_amount": goal,
            "goal_deadline_months": dline, "existing_savings": res, "objective": obj
        }
        database.save_user_profile(user["id"], prof_data)
        
        # Refresh dashboard metrics
        plans = database.get_user_plans(user["id"])
        latest_plan = plans[0] if plans else None
        hero_html = render_dashboard_metrics(user, prof_data, latest_plan)
        goal_html = render_dashboard_active_goal(prof_data)
        cash_fig = create_dashboard_cashflow_chart(prof_data)
        split_fig = create_dashboard_breakdown_chart(prof_data)

        return "✓ **Financial profile saved successfully to your account.**", hero_html, goal_html, cash_fig, split_fig

    save_profile_btn.click(
        fn=handle_save_profile,
        inputs=[
            session_user, user_name_input, monthly_income_input, rent_input, groceries_input,
            utilities_input, debt_input, transport_input, dining_input,
            shopping_input, entertainment_input, subscriptions_input,
            goal_amount_input, goal_deadline_input, existing_savings_input,
            objective_input
        ],
        outputs=[
            profile_save_status, dashboard_hero_display, dashboard_goal_display,
            dash_cashflow_plot, dash_breakdown_plot
        ]
    )

    # Navigation Shortcuts from Dashboard
    dash_to_finances_btn.click(fn=lambda: gr.update(selected="tab_profile"), inputs=[], outputs=[main_tabs])
    dash_to_plan_btn.click(fn=lambda: gr.update(selected="tab_plan"), inputs=[], outputs=[main_tabs])
    dash_to_sim_btn.click(fn=lambda: gr.update(selected="tab_simulator"), inputs=[], outputs=[main_tabs])
    dash_to_saved_btn.click(fn=lambda: gr.update(selected="tab_saved_plans"), inputs=[], outputs=[main_tabs])

    proceed_to_plan_btn.click(fn=lambda: gr.update(selected="tab_plan"), inputs=[], outputs=[main_tabs])
    modify_goals_btn.click(fn=lambda: gr.update(selected="tab_profile"), inputs=[], outputs=[main_tabs])
    edit_finances_btn.click(fn=lambda: gr.update(selected="tab_profile"), inputs=[], outputs=[main_tabs])

    # Run Analysis Stream
    run_btn.click(
        fn=run_analysis_stream,
        inputs=[
            user_name_input, monthly_income_input, rent_input, groceries_input,
            utilities_input, debt_input, transport_input, dining_input,
            shopping_input, entertainment_input, subscriptions_input,
            goal_amount_input, goal_deadline_input, existing_savings_input,
            objective_input, force_reject_cb
        ],
        outputs=[
            progress_html_display,
            hero_metrics_display,
            recommendations_html_display,
            expense_chart_plot,
            goal_chart_plot,
            agent_log_display,
            active_generated_plan
        ]
    )

    # Save Plan Handler
    def handle_save_plan(user, plan_state, profile_data):
        if not user or "id" not in user:
            return "❌ Please sign in to save your plan.", gr.update()
        if not plan_state or "validated_strategies" not in plan_state:
            return "❌ No generated plan found. Please click 'Build My Savings Plan' first.", gr.update()

        strategies = plan_state.get("validated_strategies", [])
        total_cut = sum(float(s.get("estimated_monthly_saving", 0)) for s in strategies)
        goal_findings = plan_state.get("findings", {}).get("goal", {})
        req_monthly = float(goal_findings.get("required_monthly_saving", 0))
        goal_amt = float(plan_state.get("profile", {}).get("goal_amount", 100000))
        dline = int(plan_state.get("profile", {}).get("goal_deadline_months", 12))
        obj = plan_state.get("profile", {}).get("objective", "Savings Plan")
        title = f"{obj[:40]} Plan" if obj else "Personal Savings Plan"

        database.save_plan(
            user_id=user["id"],
            title=title,
            goal_amount=goal_amt,
            required_monthly_saving=req_monthly,
            deadline_months=dline,
            total_monthly_cut=total_cut,
            strategies=strategies
        )
        saved_html = render_saved_plans_html(user["id"])
        return "✓ **Plan Saved Successfully!** You can review, track, and manage it anytime in the **Saved Plans** tab.", saved_html

    approve_plan_btn.click(
        fn=handle_save_plan,
        inputs=[session_user, active_generated_plan, current_profile],
        outputs=[plan_action_status, saved_plans_html_display]
    )

    # Saved Plans Tab Handlers
    refresh_saved_btn.click(
        fn=lambda u: render_saved_plans_html(u["id"]) if u else "",
        inputs=[session_user],
        outputs=[saved_plans_html_display]
    )

    def handle_delete_plan(user, plan_id):
        if not user or "id" not in user:
            return "❌ Please sign in.", gr.update()
        if not plan_id:
            return "❌ Please enter a valid Plan ID.", gr.update()
        
        ok = database.delete_plan(int(plan_id), user["id"])
        if ok:
            saved_html = render_saved_plans_html(user["id"])
            return f"✓ Plan #{int(plan_id)} deleted successfully.", saved_html
        return f"❌ Could not delete Plan #{int(plan_id)}. Make sure the ID belongs to your account.", gr.update()

    delete_plan_btn.click(
        fn=handle_delete_plan,
        inputs=[session_user, delete_plan_id_input],
        outputs=[delete_status_md, saved_plans_html_display]
    )

    # Scenario Simulator
    sim_run_btn.click(
        fn=simulate_what_if,
        inputs=[sim_dining_slider, sim_shopping_slider, sim_sub_slider, sim_income_slider, session_user, current_profile],
        outputs=[what_if_output_md, what_if_plot]
    )

    # FAQ Bot Event Handlers
    def handle_faq_chat(user_msg, history):
        if not user_msg or not user_msg.strip():
            return "", history
        
        chat_turns = []
        if history:
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    chat_turns.append((item[0], item[1]))
                elif isinstance(item, dict):
                    chat_turns.append((item.get("role", ""), item.get("content", "")))

        bot_reply = get_faq_response(user_msg.strip(), chat_turns)
        new_history = list(history) if history else []
        new_history.append((user_msg.strip(), bot_reply))
        return "", new_history

    faq_send_btn.click(
        fn=handle_faq_chat,
        inputs=[faq_input, faq_chatbot],
        outputs=[faq_input, faq_chatbot]
    )
    faq_input.submit(
        fn=handle_faq_chat,
        inputs=[faq_input, faq_chatbot],
        outputs=[faq_input, faq_chatbot]
    )
    faq_clear_btn.click(
        fn=lambda: [],
        inputs=[],
        outputs=[faq_chatbot]
    )
    faq_chip1.click(
        fn=lambda h: handle_faq_chat("What does this application do?", h),
        inputs=[faq_chatbot],
        outputs=[faq_input, faq_chatbot]
    )
    faq_chip2.click(
        fn=lambda h: handle_faq_chat("What are the 6 autonomous agents in this system?", h),
        inputs=[faq_chatbot],
        outputs=[faq_input, faq_chatbot]
    )
    faq_chip3.click(
        fn=lambda h: handle_faq_chat("What is an emergency fund?", h),
        inputs=[faq_chatbot],
        outputs=[faq_input, faq_chatbot]
    )
    faq_chip4.click(
        fn=lambda h: handle_faq_chat("What is the 50/30/20 budgeting rule?", h),
        inputs=[faq_chatbot],
        outputs=[faq_input, faq_chatbot]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        footer_links=[],
        theme=custom_theme,
        css=custom_css,
        js="() => { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); }"
    )
