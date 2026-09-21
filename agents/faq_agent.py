"""
agents/faq_agent.py — Dedicated FAQ & Financial Education Assistant.
Answers application architecture and general personal finance questions using Groq.
Maintains strict educational safety boundaries, never touches private user financial records,
and gracefully redirects personal plan requests to the main Savings Plan Builder.
"""

import os
import sys
import logging
from typing import List, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.base_agent import invoke_llm_text

logger = logging.getLogger(__name__)

# Grounded system prompt capturing system architecture and personal finance principles
FAQ_SYSTEM_PROMPT = """
You are the FAQ & Financial Information Assistant for the "Personal Savings & Cash Flow Advisory" application.

YOUR MISSION:
Provide clear, accurate, educational explanations about:
1. HOW THIS APPLICATION WORKS (Multi-Agent Architecture, Agents, Scenario Planner, Security)
2. GENERAL FINANCIAL EDUCATION (Emergency funds, EMI, 50/30/20 rule, compound interest, budgeting, saving vs. investing)

CORE PRINCIPLES & SYSTEM GROUNDING:
- The system is a Multi-Agentic AI Personal Savings Recommendation System built with pure Python deterministic tools and Groq LLMs.
- The system coordinates 6 specialized agents:
  1. Orchestrator Agent: Parses user objectives into structured constraints and manages the multi-agent pipeline and feedback loop.
  2. Expense Analysis Agent: Categorizes spending into essential commitments (rent, utilities, groceries, debt) vs. flexible/discretionary spending, and detects recurring obligations.
  3. Budget Assessment Agent: Computes savings rate, evaluates baseline health, and sets discretionary trimming capacity ceilings.
  4. Financial Goal Feasibility Agent: Calculates exact monthly savings required to hit target deadlines and evaluates current trajectory feasibility.
  5. Savings Recommendation Agent: Formulates concrete, non-essential expense reduction strategies (e.g. trimming dining out, retail subscriptions).
  6. Financial Safety & Feasibility Agent: Enforces strict essential spending floors and minimum emergency buffers; rejects aggressive or unsustainable recommendations and demands revisions.
- Zero LLM mental math: All calculations (percentages, savings rates, required monthly savings, dates) trace directly to pure Python tool functions.
- Rejection/Revision loop: If a strategy compromises living essentials or cuts discretionary spending by more than 50-65%, the Safety Agent rejects it and forces the Savings Agent to generate a safer, sustainable alternative.
- Scenario Planner: Allows users to test what-if adjustments to dining, retail, subscriptions, and income without modifying their base account numbers.
- Security & Privacy: User accounts use PBKDF2-HMAC-SHA256 password hashing. Data is isolated per user in Neon PostgreSQL.

STRICT BOUNDARIES & SAFETY GUIDELINES:
1. EDUCATIONAL ROLE ONLY: You are an educational assistant, NOT a certified financial planner, tax advisor, or investment broker. Do not provide certified legal, investment, or tax advice.
2. NO PERSONAL FINANCIAL RECORD ACCESS: You do NOT have access to the user's private financial records, balance, or income. Never attempt to read or modify database records.
3. NO DIRECT PLAN GENERATION: If the user asks for a personal savings calculation or specific plan (e.g., "I earn ₹50,000 and spend ₹38,000, how much should I save?"), provide general educational context and gently direct them to use the "Build Savings Plan" tab in the dashboard for exact multi-agent analysis.
4. HONESTY: Do not fabricate internal features or pretend the system does algorithmic stock trading or cryptocurrency speculation.

Format your responses with clean markdown, bullet points, and brief, encouraging paragraphs.
"""

# Curated deterministic offline knowledge base for instant answers
OFFLINE_FAQ_KB = {
    "what does this application do": """
### What Does This Application Do?

The **Personal Savings & Cash Flow Advisory** is a safety-first, multi-agent AI personal finance platform. It helps individuals:
- **Analyze Cash Flow**: Automatically categorizes monthly spending into essential commitments (housing, utilities, groceries, debt EMI) and flexible lifestyle spending.
- **Generate Realistic Savings Plans**: Uses autonomous agents to recommend sustainable expense adjustments tailored to your specific financial goals.
- **Enforce Safety Floors**: An autonomous Safety Agent ensures recommendations never compromise essential needs or living security.
- **Test What-If Scenarios**: Interactively simulate the impact of lifestyle adjustments on your savings timeline.

*All math is calculated using pure Python algorithms—ensuring 100% mathematical accuracy with zero AI guesswork.*
""",
    "how does the savings recommendation system work": """
### How the Savings Recommendation System Works

The recommendation system uses an **iterative multi-agent feedback loop**:
1. **Constraint Parsing**: The Orchestrator extracts your goals and strictness preferences.
2. **Expense & Budget Analysis**: Specialized agents calculate your essential vs. discretionary split and safe trimming capacity.
3. **Strategy Formulation**: The Savings Agent proposes concrete reductions exclusively in flexible categories (e.g., dining, subscriptions).
4. **Safety Audit**: The Safety Agent checks each recommendation against strict safety floors. If a recommendation is too extreme (over 50-65% discretionary reduction), it **rejects** the plan and orders the Savings Agent to revise it.
5. **Final Plan**: Only verified, sustainable strategies are presented and saved to your account.
""",
    "what are the different agents": """
### The 6 Specialized Autonomous Agents

1. **Orchestrator Agent**: Coordinates the workflow, parses user goals into structured constraints, and manages the iterative revise loop.
2. **Expense Analysis Agent**: Computes category totals, divides expenses into essential vs. discretionary, and tracks recurring bills.
3. **Budget Assessment Agent**: Calculates current savings rate and determines the maximum safe trimming capacity.
4. **Financial Goal Agent**: Calculates the required monthly savings to achieve target goals within designated deadlines.
5. **Savings Recommendation Agent**: Formulates high-impact, realistic reduction strategies for flexible spending.
6. **Financial Safety & Feasibility Agent**: Audits every proposed cut against essential living floors and emergency cushions, rejecting unsafe recommendations.
""",
    "what is an emergency fund": """
### What is an Emergency Fund?

An **emergency fund** is a dedicated cash buffer set aside exclusively for unexpected life events, such as medical emergencies, urgent home repairs, or sudden income loss.

- **Recommended Size**: Typically **3 to 6 months of essential living expenses** (rent, groceries, utilities, and debt EMIs).
- **Where to Keep It**: In a liquid, low-risk account (like a high-yield savings account or liquid mutual fund) that you can access immediately without penalties.
- **Why It Matters**: Having an emergency fund prevents you from taking on high-interest debt or liquidating long-term investments during a crisis.
""",
    "what is the 50/30/20 budgeting rule": """
### The 50/30/20 Budgeting Rule

The **50/30/20 rule** is an intuitive budgeting framework popularized by Senator Elizabeth Warren:

- **50% Needs (Essential Expenses)**: Rent/mortgage, groceries, basic utilities, transportation, and minimum loan repayments.
- **30% Wants (Discretionary Spending)**: Dining out, entertainment, shopping, vacations, and streaming subscriptions.
- **20% Savings & Debt Acceleration**: Emergency fund contributions, retirement accounts, and extra debt principal reduction.

*Our application actively categorizes your spending to help you identify how close your current cash flow is to this benchmark.*
""",
    "what is an emi": """
### What is an EMI?

**EMI** stands for **Equated Monthly Installment**. It is a fixed payment amount made by a borrower to a lender at a specified date each calendar month.

- **Components**: Each EMI payment includes both the **principal component** (repaying the loan amount) and the **interest charge**.
- **Crucial Rule**: EMIs for home, auto, or personal loans are treated as **essential fixed commitments** in our system and are strictly protected from reduction recommendations.
""",
    "what is compound interest": """
### What is Compound Interest?

**Compound interest** is the interest you earn on both your original principal investment and on the accumulated interest from prior periods. Albert Einstein famously referred to it as the "eighth wonder of the world."

- **The Formula**: $A = P(1 + r/n)^{nt}$
- **Why It Matters**: Over long time horizons, reinvested earnings grow exponentially. Starting to save and invest even modest amounts early produces significantly greater wealth than saving larger amounts later in life.
""",
    "what is discretionary spending": """
### What is Discretionary Spending?

**Discretionary spending** consists of non-essential lifestyle expenses that you can choose to modify or eliminate without threatening your physical health, housing, or contractual obligations.

- **Examples**: Dining out, food delivery, designer clothing, subscription services, weekend leisure, and consumer gadgets.
- **Contrast with Essentials**: Essentials include housing, essential food, utilities, and debt commitments. Our Savings Agent focuses exclusively on optimizing discretionary spending.
""",
    "difference between saving and investing": """
### The Difference Between Saving and Investing

- **Saving**: Setting aside cash in safe, liquid accounts (like bank savings or fixed deposits) for short-term goals (under 3 years) or emergency funds. Capital preservation is the priority; returns are modest.
- **Investing**: Allocating capital to assets (like diversified equity index funds, bonds, or real estate) to generate long-term capital growth (5+ years). It involves risk and volatility, but historically outpaces inflation.
"""
}


def find_offline_answer(user_query: str) -> Optional[str]:
    """Checks the curated knowledge base for a matching answer."""
    query_lower = user_query.strip().lower()
    for key, answer in OFFLINE_FAQ_KB.items():
        if key in query_lower or all(w in query_lower for w in key.split() if len(w) > 3):
            return answer
    return None


def get_faq_response(user_query: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
    """
    Generates a helpful, educational response to user questions using Groq.
    Falls back to curated domain knowledge if offline or rate-limited.
    """
    if not user_query or not user_query.strip():
        return "Please ask a question about personal finance or how our advisory system works!"

    query = user_query.strip()
    
    # Check for direct calculation queries and politely redirect
    q_lower = query.lower()
    has_income_indicators = any(kw in q_lower for kw in ["i earn", "my income", "i make", "my salary", "spend ₹", "spend rs"])
    has_advice_request = any(kw in q_lower for kw in ["how much should i save", "calculate for me", "build me a plan", "make me a plan"])
    
    if has_income_indicators and has_advice_request:
        return (
            "💡 **Personalized Financial Advice Notice:**\n\n"
            "As an educational assistant, I don't calculate individual financial plans directly in this chat or modify your records.\n\n"
            "To generate an exact, customized savings plan using our 6 autonomous agents and deterministic tools:\n"
            "1. Switch to the **'Your Finances'** tab to verify your income and expenses.\n"
            "2. Head to the **'Build Savings Plan'** tab and click **'🚀 Generate Savings Recommendation'**.\n\n"
            "Our multi-agent system will calculate your exact capacity, project your timeline, and enforce safety floors!"
        )

    # Contextual history summary
    history_context = ""
    if chat_history:
        formatted_turns = []
        for item in chat_history[-6:]:
            if isinstance(item, dict):
                role = str(item.get("role", "user")).capitalize()
                content = item.get("content", "")
                if content:
                    formatted_turns.append(f"{role}: {content}")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                formatted_turns.append(f"User: {item[0]}\nAssistant: {item[1]}")
        if formatted_turns:
            history_context = "\n".join(formatted_turns)

    context_block = f"Previous conversation context:\n{history_context}\n\n" if history_context else ""
    prompt = f"""
{context_block}User Question: {query}

Provide a helpful, structured educational answer grounded in our system design and sound financial principles.
"""

    response = invoke_llm_text(
        system_instruction=FAQ_SYSTEM_PROMPT,
        prompt=prompt,
        temperature=0.3,
        max_tokens=600,
    )

    if response and response.strip():
        return response.strip()

    # Deterministic offline knowledge base fallback
    offline_match = find_offline_answer(query)
    if offline_match:
        return offline_match.strip()

    return (
        "### Personal Finance & Advisory Assistant\n\n"
        "Thank you for your question! Here are key areas I can help you with:\n"
        "- **How This App Works**: Ask about our 6 autonomous agents, the safety floor audit, or scenario testing.\n"
        "- **General Finance**: Ask about emergency funds, the 50/30/20 rule, debt EMIs, or compound interest.\n\n"
        "For custom personal savings plans with your actual income and expenses, please use the **'Build Savings Plan'** tab in the main navigation."
    )
