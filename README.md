# Multi-Agentic AI Personal Savings Recommendation System

A multi-agent artificial intelligence application demonstrating authentic agentic AI architecture (planning, delegation, tool use, inter-agent state, and an iterative reject/revise feedback loop) designed for personal financial planning and savings optimization.

Built with **Python 3.11+**, **Groq SDK (`groq`)**, **Neon PostgreSQL**, **pandas**, and **Gradio Blocks**.

---

## 🏛️ Architecture & System Design

The system coordinates 6 specialized agents operating over a centralized typed blackboard state (`AnalysisState`). **Crucially, the LLM never performs mental arithmetic—every single financial number, savings rate, and timeline calculation traces directly to deterministic Python tool functions.**

```
                                  +-----------------------+
                                  |      User Profile     |
                                  |  (CSV Upload or Form) |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |     Orchestrator      |
                                  |  (Constraint Parsing) |
                                  +-----------+-----------+
                                              |
                     +------------------------+------------------------+
                     |                        |                        |
                     v                        v                        v
          +--------------------+    +--------------------+    +--------------------+
          |   Expense Agent    |    |    Budget Agent    |    |     Goal Agent     |
          |  (category_totals, |    |  (savings_rate,    |    |  (required_saving, |
          |  ess_vs_disc,      |    |   capacity_calc)   |    |   project_timeline)|
          |  detect_recurring) |    +--------------------+    +--------------------+
          +--------------------+
                     |
                     +------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Savings Recommendation     | <----------------+
                              |             Agent             |                  | (Rejection Feedback Loop:
                              |   (simulate_category_cut,     |                  |  rejection_reasons injected
                              |    combine_strategies)        |                  |  into constraints)
                              +---------------+---------------+                  |
                                              |                                  |
                                              | Candidate Strategies             |
                                              v                                  |
                              +-------------------------------+                  |
                              |   Safety & Feasibility Agent  |                  |
                              |     (check_essential_floor,   | --- [REJECT] ----+
                              |      check_emergency_buffer)  |
                              +---------------+---------------+
                                              |
                                      [ALL APPROVED]
                                              |
                                              v
                                  +-----------------------+
                                  |     Orchestrator      |
                                  |  (Final Compilation)  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |    Gradio Web UI      |
                                  | (Dashboard + Approvals|
                                  |   + What-If Sim)      |
                                  +-----------------------+
```

---

## 🎓 Why This Is Genuinely Agentic (Viva / Evaluation Guide)

| Feature | Single Prompt LLM / Traditional ML | This Multi-Agentic System |
| :--- | :--- | :--- |
| **Arithmetic Integrity** | Prone to hallucinations, incorrect percentage calculations, and compounding rounding errors. | **Zero LLM Arithmetic**: LLMs generate qualitative synthesis while all numbers originate strictly from Python tool calls (`simulate_category_reduction`, `calculate_savings_rate`). |
| **Feedback & Revision** | One-shot output with no self-correction or internal debate. | **Closed-loop Reject/Revise**: The Safety Agent evaluates candidate plans against hard essential floors and sustainability bounds. On violation, it rejects the plan and routes explicit reasons back to the Savings Agent to re-plan. |
| **Separation of Concerns** | Single giant prompt trying to categorize, calculate, optimize, and guardrail at once. | **6 Scoped Agents**: Each agent possesses dedicated tools, isolated prompt scope, and validated Pydantic output schemas. |
| **Explainable Audit Trail** | Black-box output. | **Sequential Reasoning Log**: Every agent appends a structured, timestamped audit log (`reasoning_log`) streamed directly into the UI. |
| **Human-in-the-Loop** | Autonomous risk or static text. | **Safe Proposal Model**: System never executes or applies a budget without explicit human review and approval in the UI. |

---

## 🛠️ Tech Stack & Constraints

- **Language:** Python 3.11+
- **LLM SDK:** Groq Python SDK (`groq` / `openai/gpt-oss-20b`)
- **Database:** Neon PostgreSQL (`psycopg2-binary`) with HTTPS fallback
- **Data:** pandas, numpy
- **UI:** Gradio (Blocks API)
- **Charts:** Matplotlib
- **Design Decisions:**
  - **No LLM Math:** Hard constraint strictly enforced.
  - **Conservative Safeguards:** Rent and loan EMIs are non-negotiable (100% protected). Nutritional grocery floor set to Rs. 3,000/month.
  - **Offline / Fallback Resilience:** If `GROQ_API_KEY` is not set or network fails, agents seamlessly execute deterministic tool logic without crashing.

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
cd "d:\Personnel saving recommnedation Ai'"
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and provide your Groq API key and Neon Database URL:
```bash
copy .env.example .env
```
Inside `.env`:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
DATABASE_URL=postgresql://neondb_owner:your_password@ep-example.ap-southeast-1.aws.neon.tech/neondb?sslmode=require
```

### 3. Generate or Verify Sample Data
Generate 6 months of data for 6 distinct user profiles:
```bash
python data/generate_sample_data.py
```
*(The Kaggle dataset `dataset/personal_finance_tracker_dataset.csv` is also pre-loaded and supported).*

### 4. Run Automated Verification Tests
```bash
python test_pipeline.py
```

### 5. Launch the Web Interface
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 📱 Gradio App Features

1. **Tab 1: Profile Input & Data Setup**
   - ⚡ Quick-load 3 curated presets (Rahul Verma, Priya Sharma, or Amit Patel).
   - Upload any CSV or pick users from `sample_data.csv` or Kaggle dataset.
   - Edit financial categories, goals, deadlines, and natural language objectives.
   - Checkbox to toggle the **Demo Rejection Loop**.
2. **Tab 2: Run Analysis & Live Activity Feed**
   - Click `⚡ Run Multi-Agent Analysis Pipeline`.
   - Real-time streaming log displaying each agent's execution, tool calls, and rejection/revision loops.
3. **Tab 3: Financial Dashboard**
   - Horizontal bar chart comparing Essential (Blue) vs Discretionary (Pink) expenses.
   - Goal timeline gauge comparing Current Savings Velocity vs Required Velocity.
4. **Tab 4: Validated Recommendations**
   - Card view of each safety-approved strategy detailing Action, Monthly Savings, Annual Impact, and Risk Profile.
   - Interactive human-in-the-loop buttons: **Approve Plan**, **Reject Strategy**, **Refine Constraints**.
5. **Tab 5: What-If Simulator**
   - Interactive sliders for dining cuts, shopping trims, subscription cancellations, and income adjustments.
   - Instant re-evaluation using Budget and Savings tools without re-running the full pipeline.

---

## 🧪 Demo Test Case: Triggering the Reject/Revise Loop

To demonstrate authentic multi-agent feedback in a presentation or viva:
1. Go to **Tab 1 (Profile Input)**.
2. Select preset: **"Demo 2: Amit Patel (Overleveraged - REJECT/REVISE TEST CASE)"**.
   - Income: Rs. 55,000 | Rent: Rs. 20,000 | Debt EMI: Rs. 15,000 | Discretionary: Rs. 12,450.
   - Goal: Rs. 180,000 in 12 months (requires ~Rs. 12,916/mo savings).
3. Notice that **"Force Demo Rejection Loop"** is checked.
4. Switch to **Tab 2 (Run Analysis)** and click **Run Multi-Agent Analysis Pipeline**.
5. Observe the live reasoning feed:
   - `Pass 1`: Savings Agent generates an aggressive 90% dining slash.
   - `Safety Agent`: **REJECTS** the strategy because a 90% cut violates sustainability bounds.
   - `Orchestrator`: Detects rejection, increments revision counter, and loops back.
   - `Pass 2`: Savings Agent reads the rejection feedback and generates a balanced, compliant strategy.
   - `Safety Agent`: **APPROVES** all revised strategies.
   - `Orchestrator`: Compiles the final recommendations.
