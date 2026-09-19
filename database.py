"""
database.py — SQLite persistence layer for Multi-Agent AI Personal Finance SaaS.
Handles secure user authentication (PBKDF2-HMAC-SHA256), user-isolated financial profiles,
saved savings plans, and scenario history.
"""

import os
import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "finance.db")


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hashes a password with PBKDF2-HMAC-SHA256 and a 16-byte random salt."""
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()
    return pwd_hash, salt


def verify_password(password: str, pwd_hash: str, salt: str) -> bool:
    """Verifies a password against the stored hash and salt."""
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == pwd_hash


def init_db() -> None:
    """Initializes SQLite schema and seeds a default demo account."""
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Users Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. User Financial Profiles (One per user)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS financial_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            monthly_income REAL DEFAULT 60000.0,
            rent REAL DEFAULT 18000.0,
            groceries REAL DEFAULT 9000.0,
            utilities REAL DEFAULT 3500.0,
            debt_emi REAL DEFAULT 5000.0,
            transport REAL DEFAULT 3500.0,
            dining_out REAL DEFAULT 5000.0,
            shopping REAL DEFAULT 4000.0,
            entertainment REAL DEFAULT 2500.0,
            subscriptions REAL DEFAULT 1500.0,
            existing_savings REAL DEFAULT 30000.0,
            goal_amount REAL DEFAULT 150000.0,
            goal_deadline_months INTEGER DEFAULT 12,
            objective TEXT DEFAULT 'Optimize discretionary spending to build an emergency fund.',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # 3. User Saved Savings Plans
    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            goal_amount REAL NOT NULL,
            required_monthly_saving REAL NOT NULL,
            deadline_months INTEGER NOT NULL,
            total_monthly_cut REAL NOT NULL,
            strategies_json TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # 4. User Scenario History
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scenario_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dining_cut REAL DEFAULT 0.0,
            shopping_cut REAL DEFAULT 0.0,
            sub_cut REAL DEFAULT 0.0,
            income_change REAL DEFAULT 0.0,
            projected_monthly_saving REAL NOT NULL,
            projected_timeline TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()

    # Pre-seed demo user if not already present
    cur.execute("SELECT id FROM users WHERE email = 'demo@finance.ai'")
    if not cur.fetchone():
        pwd_hash, salt = hash_password("password123")
        cur.execute(
            "INSERT INTO users (email, password_hash, salt, full_name) VALUES (?, ?, ?, ?)",
            ("demo@finance.ai", pwd_hash, salt, "Alex Morgan")
        )
        demo_user_id = cur.lastrowid
        cur.execute("""
            INSERT INTO financial_profiles (
                user_id, monthly_income, rent, groceries, utilities, debt_emi,
                transport, dining_out, shopping, entertainment, subscriptions,
                existing_savings, goal_amount, goal_deadline_months, objective
            ) VALUES (?, 75000, 20000, 9500, 4000, 6000, 3500, 7000, 5000, 3000, 2000, 45000, 200000, 12, ?)
        """, (demo_user_id, "Build a 6-month safety net while preserving essential comforts."))

        # Pre-seed one realistic saved plan for demo user
        strategies = [
            {
                "name": "Moderate Dining & Food Delivery",
                "action": "Trim weekly takeout and weekend dining to 4 meals/month",
                "category": "dining_out",
                "reduction_percent": 30.0,
                "estimated_monthly_saving": 2100.0,
                "annual_impact": 25200.0,
                "reason": "Reduces flexible dining without impacting grocery nutrition.",
                "risk_level": "Low",
                "status": "approved"
            },
            {
                "name": "Streaming & Subscription Prune",
                "action": "Consolidate multiple streaming platforms to one primary service",
                "category": "subscriptions",
                "reduction_percent": 40.0,
                "estimated_monthly_saving": 800.0,
                "annual_impact": 9600.0,
                "reason": "Eliminates duplicate digital entertainment subscriptions.",
                "risk_level": "Low",
                "status": "approved"
            }
        ]
        cur.execute("""
            INSERT INTO saved_plans (
                user_id, title, goal_amount, required_monthly_saving,
                deadline_months, total_monthly_cut, strategies_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            demo_user_id,
            "Emergency Fund Acceleration",
            200000.0,
            12916.67,
            12,
            2900.0,
            json.dumps(strategies)
        ))
        conn.commit()

    conn.close()


def create_user(email: str, password: str, full_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Registers a new user. Returns (user_dict, None) or (None, error_message)."""
    email = email.strip().lower()
    full_name = full_name.strip()

    if not email or "@" not in email or "." not in email:
        return None, "Please provide a valid email address."
    if not full_name:
        return None, "Please enter your full name."
    if len(password) < 6:
        return None, "Password must be at least 6 characters long."

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        pwd_hash, salt = hash_password(password)
        cur.execute(
            "INSERT INTO users (email, password_hash, salt, full_name) VALUES (?, ?, ?, ?)",
            (email, pwd_hash, salt, full_name)
        )
        user_id = cur.lastrowid
        # Create default financial profile
        cur.execute(
            "INSERT INTO financial_profiles (user_id) VALUES (?)",
            (user_id,)
        )
        conn.commit()
        user = {
            "id": user_id,
            "email": email,
            "full_name": full_name,
            "created_at": datetime.now().strftime("%Y-%m-%d")
        }
        return user, None
    except sqlite3.IntegrityError:
        return None, "An account with this email address already exists. Please sign in."
    except Exception as e:
        return None, f"Registration failed: {str(e)}"
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Authenticates a user by email and password. Returns (user_dict, None) or (None, error_msg)."""
    email = email.strip().lower()
    if not email or not password:
        return None, "Please provide both email and password."

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash, salt, full_name, created_at FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None, "Invalid email or password. Please try again."

    if not verify_password(password, row["password_hash"], row["salt"]):
        return None, "Invalid email or password. Please try again."

    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "created_at": str(row["created_at"])[:10]
    }, None


def get_user_profile(user_id: int) -> Dict[str, Any]:
    """Retrieves the financial profile for a specific user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM financial_profiles WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()

    if row:
        return dict(row)
    # Default fallback if row missing
    return {
        "user_id": user_id,
        "monthly_income": 60000.0,
        "rent": 18000.0,
        "groceries": 9000.0,
        "utilities": 3500.0,
        "debt_emi": 5000.0,
        "transport": 3500.0,
        "dining_out": 5000.0,
        "shopping": 4000.0,
        "entertainment": 2500.0,
        "subscriptions": 1500.0,
        "existing_savings": 30000.0,
        "goal_amount": 150000.0,
        "goal_deadline_months": 12,
        "objective": "Optimize discretionary spending to build an emergency fund.",
    }


def save_user_profile(user_id: int, profile_data: Dict[str, Any]) -> bool:
    """Saves or updates a user's financial profile."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO financial_profiles (
            user_id, monthly_income, rent, groceries, utilities, debt_emi,
            transport, dining_out, shopping, entertainment, subscriptions,
            existing_savings, goal_amount, goal_deadline_months, objective, updated_at
        ) VALUES (
            :user_id, :monthly_income, :rent, :groceries, :utilities, :debt_emi,
            :transport, :dining_out, :shopping, :entertainment, :subscriptions,
            :existing_savings, :goal_amount, :goal_deadline_months, :objective, CURRENT_TIMESTAMP
        )
        ON CONFLICT(user_id) DO UPDATE SET
            monthly_income = excluded.monthly_income,
            rent = excluded.rent,
            groceries = excluded.groceries,
            utilities = excluded.utilities,
            debt_emi = excluded.debt_emi,
            transport = excluded.transport,
            dining_out = excluded.dining_out,
            shopping = excluded.shopping,
            entertainment = excluded.entertainment,
            subscriptions = excluded.subscriptions,
            existing_savings = excluded.existing_savings,
            goal_amount = excluded.goal_amount,
            goal_deadline_months = excluded.goal_deadline_months,
            objective = excluded.objective,
            updated_at = CURRENT_TIMESTAMP
    """, {
        "user_id": user_id,
        "monthly_income": float(profile_data.get("monthly_income", 60000.0)),
        "rent": float(profile_data.get("rent", 18000.0)),
        "groceries": float(profile_data.get("groceries", 9000.0)),
        "utilities": float(profile_data.get("utilities", 3500.0)),
        "debt_emi": float(profile_data.get("debt_emi", 5000.0)),
        "transport": float(profile_data.get("transport", 3500.0)),
        "dining_out": float(profile_data.get("dining_out", 5000.0)),
        "shopping": float(profile_data.get("shopping", 4000.0)),
        "entertainment": float(profile_data.get("entertainment", 2500.0)),
        "subscriptions": float(profile_data.get("subscriptions", 1500.0)),
        "existing_savings": float(profile_data.get("existing_savings", 30000.0)),
        "goal_amount": float(profile_data.get("goal_amount", 150000.0)),
        "goal_deadline_months": int(profile_data.get("goal_deadline_months", 12)),
        "objective": str(profile_data.get("objective", "Build emergency savings.")),
    })
    conn.commit()
    conn.close()
    return True


def save_plan(
    user_id: int,
    title: str,
    goal_amount: float,
    required_monthly_saving: float,
    deadline_months: int,
    total_monthly_cut: float,
    strategies: List[Dict[str, Any]]
) -> int:
    """Saves an approved savings plan for a user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO saved_plans (
            user_id, title, goal_amount, required_monthly_saving,
            deadline_months, total_monthly_cut, strategies_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        title,
        float(goal_amount),
        float(required_monthly_saving),
        int(deadline_months),
        float(total_monthly_cut),
        json.dumps(strategies)
    ))
    plan_id = cur.lastrowid
    conn.commit()
    conn.close()
    return plan_id


def get_user_plans(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all saved plans for a specific user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, user_id, title, goal_amount, required_monthly_saving,
               deadline_months, total_monthly_cut, strategies_json, status, created_at
        FROM saved_plans
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    plans = []
    for r in rows:
        p = dict(r)
        try:
            p["strategies"] = json.loads(p["strategies_json"])
        except Exception:
            p["strategies"] = []
        p["created_at_fmt"] = str(p["created_at"])[:16].replace("T", " ")
        plans.append(p)
    return plans


def delete_plan(plan_id: int, user_id: int) -> bool:
    """Deletes a saved plan belonging to a user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_plans WHERE id = ? AND user_id = ?", (plan_id, user_id))
    rows_deleted = cur.rowcount
    conn.commit()
    conn.close()
    return rows_deleted > 0


def save_scenario_history(
    user_id: int,
    dining_cut: float,
    shopping_cut: float,
    sub_cut: float,
    income_change: float,
    projected_monthly_saving: float,
    projected_timeline: str
) -> int:
    """Records a what-if scenario run for the user."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO scenario_history (
            user_id, dining_cut, shopping_cut, sub_cut, income_change,
            projected_monthly_saving, projected_timeline
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        float(dining_cut),
        float(shopping_cut),
        float(sub_cut),
        float(income_change),
        float(projected_monthly_saving),
        str(projected_timeline)
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_user_recent_activity(user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Returns a unified timeline of recent saved plans and scenario runs for the user."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 'plan' as type, title as description, total_monthly_cut as metric,
               created_at
        FROM saved_plans
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))
    plan_rows = [dict(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT 'scenario' as type,
               'What-If Scenario: Projected ₹' || printf('%,d', CAST(projected_monthly_saving AS INT)) || '/mo' as description,
               projected_monthly_saving as metric,
               created_at
        FROM scenario_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))
    scen_rows = [dict(r) for r in cur.fetchall()]

    conn.close()

    combined = plan_rows + scen_rows
    combined.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return combined[:limit]


# Auto-initialize on first import
init_db()
