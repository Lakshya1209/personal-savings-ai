"""
database.py — Unified persistence layer for Multi-Agent AI Personal Finance Platform.
Supports Neon PostgreSQL with automatic failover to Neon HTTPS (port 443) and local SQLite fallback.
Maintains secure user authentication (PBKDF2-HMAC-SHA256), strict user data isolation,
saved savings plans, and scenario history.
"""

import os
import sys
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
SQLITE_DB_PATH = os.path.join(DATA_DIR, "finance.db")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


# ---------------------------------------------------------------------------
# Password Security (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Database Client / Abstraction Layer
# ---------------------------------------------------------------------------
class NeonHTTPClient:
    """Client that executes SQL against Neon via official HTTPS endpoint on port 443."""
    def __init__(self, database_url: str):
        self.database_url = database_url
        host_part = database_url.split("@")[1].split("/")[0].split(":")[0]
        self.endpoint = host_part.replace("-pooler", "")
        self.url = f"https://{self.endpoint}/sql"

    def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        import urllib.request
        body: Dict[str, Any] = {"query": query}
        if params is not None:
            # Convert params into positional $1, $2, etc. if needed, or pass as array
            body["params"] = params
        
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Neon-Connection-String": self.database_url,
                "Content-Type": "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("rows", [])


def _get_db_mode() -> str:
    """Detects whether PostgreSQL (Neon) or SQLite should be used."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgres://") or url.startswith("postgresql://"):
        return "postgres"
    return "sqlite"


_DIRECT_AVAILABLE = None
_HTTP_CLIENT = None


def execute_sql(query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Executes a SQL query with parameters and returns rows as dictionaries.
    Prioritizes direct PostgreSQL (psycopg2); falls back to Neon HTTPS port 443 if port 5432 is blocked;
    falls back to SQLite if DATABASE_URL is not set.
    """
    global _DIRECT_AVAILABLE, _HTTP_CLIENT
    db_mode = _get_db_mode()

    if db_mode == "postgres":
        db_url = os.getenv("DATABASE_URL", "").strip()
        if db_url.startswith("postgres://"):
            db_url = "postgresql://" + db_url[len("postgres://"):]

        # Normalize query: convert any %s placeholders into $1, $2
        normalized_query = query
        if params and "%s" in normalized_query:
            parts = normalized_query.split("%s")
            rebuilt = []
            for idx, part in enumerate(parts[:-1]):
                rebuilt.append(part + f"${idx + 1}")
            rebuilt.append(parts[-1])
            normalized_query = "".join(rebuilt)

        # 1. Attempt direct psycopg2 connection if not previously determined to be blocked
        if _DIRECT_AVAILABLE is not False:
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(db_url, connect_timeout=10)
                try:
                    with conn:
                        with conn.cursor(cursor_factory=RealDictCursor) as cur:
                            pg_query = normalized_query
                            if params:
                                for i in range(len(params), 0, -1):
                                    pg_query = pg_query.replace(f"${i}", "%s")
                                cur.execute(pg_query, params)
                            else:
                                cur.execute(pg_query)
                            
                            if cur.description:
                                rows = [dict(r) for r in cur.fetchall()]
                            else:
                                rows = []
                    _DIRECT_AVAILABLE = True
                    return rows
                finally:
                    conn.close()
            except Exception as pg_err:
                _DIRECT_AVAILABLE = False
                logger.info(f"Direct PostgreSQL port 5432 unreachable ({pg_err}). Switching to Neon HTTPS (port 443).")

        # 2. Execute via Neon's official HTTPS SQL API (port 443)
        try:
            if _HTTP_CLIENT is None:
                _HTTP_CLIENT = NeonHTTPClient(db_url)
            return _HTTP_CLIENT.execute(normalized_query, params)
        except Exception as http_err:
            logger.error(f"Neon database query error: {http_err}")
            raise http_err

    # 2. Local SQLite Fallback (only if DATABASE_URL is not set)
    import sqlite3
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        sqlite_query = query
        if params:
            # Replace $1, $2 or %s with ? for SQLite
            for i in range(len(params), 0, -1):
                sqlite_query = sqlite_query.replace(f"${i}", "?")
            sqlite_query = sqlite_query.replace("%s", "?")
            cur.execute(sqlite_query, params)
        else:
            cur.execute(sqlite_query)
        
        # Must consume results BEFORE commit() to prevent "cannot commit transaction - SQL statements in progress"
        rows = [dict(r) for r in cur.fetchall()] if cur.description else []
        conn.commit()
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema Initialization
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Initializes schema and ensures demo user exists."""
    db_mode = _get_db_mode()

    if db_mode == "postgres":
        statements = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                salt VARCHAR(255) NOT NULL,
                full_name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS financial_profiles (
                id SERIAL PRIMARY KEY,
                user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                monthly_income DOUBLE PRECISION DEFAULT 60000.0,
                rent DOUBLE PRECISION DEFAULT 18000.0,
                groceries DOUBLE PRECISION DEFAULT 9000.0,
                utilities DOUBLE PRECISION DEFAULT 3500.0,
                debt_emi DOUBLE PRECISION DEFAULT 5000.0,
                transport DOUBLE PRECISION DEFAULT 3500.0,
                dining_out DOUBLE PRECISION DEFAULT 5000.0,
                shopping DOUBLE PRECISION DEFAULT 4000.0,
                entertainment DOUBLE PRECISION DEFAULT 2500.0,
                subscriptions DOUBLE PRECISION DEFAULT 1500.0,
                existing_savings DOUBLE PRECISION DEFAULT 30000.0,
                goal_amount DOUBLE PRECISION DEFAULT 150000.0,
                goal_deadline_months INTEGER DEFAULT 12,
                objective TEXT DEFAULT 'Optimize discretionary spending to build an emergency fund.',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS saved_plans (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(255) NOT NULL,
                goal_amount DOUBLE PRECISION NOT NULL,
                required_monthly_saving DOUBLE PRECISION NOT NULL,
                deadline_months INTEGER NOT NULL,
                total_monthly_cut DOUBLE PRECISION NOT NULL,
                strategies_json TEXT NOT NULL,
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS scenario_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                dining_cut DOUBLE PRECISION DEFAULT 0.0,
                shopping_cut DOUBLE PRECISION DEFAULT 0.0,
                sub_cut DOUBLE PRECISION DEFAULT 0.0,
                income_change DOUBLE PRECISION DEFAULT 0.0,
                projected_monthly_saving DOUBLE PRECISION NOT NULL,
                projected_timeline TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]
        for stmt in statements:
            try:
                execute_sql(stmt)
            except Exception as e:
                logger.warning(f"Error during Postgres table init: {e}")

        # Seed demo user in Postgres if missing
        demo_user = execute_sql("SELECT id FROM users WHERE email = $1", ["demo@finance.ai"])
        if not demo_user:
            pwd_hash, salt = hash_password("password123")
            inserted = execute_sql(
                "INSERT INTO users (email, password_hash, salt, full_name) VALUES ($1, $2, $3, $4) RETURNING id",
                ["demo@finance.ai", pwd_hash, salt, "Alex Morgan"]
            )
            demo_user_id = inserted[0]["id"] if inserted else 1
            execute_sql("""
                INSERT INTO financial_profiles (
                    user_id, monthly_income, rent, groceries, utilities, debt_emi,
                    transport, dining_out, shopping, entertainment, subscriptions,
                    existing_savings, goal_amount, goal_deadline_months, objective
                ) VALUES ($1, 75000, 20000, 9500, 4000, 6000, 3500, 7000, 5000, 3000, 2000, 45000, 200000, 12, $2)
                ON CONFLICT (user_id) DO NOTHING
            """, [demo_user_id, "Build a 6-month safety net while preserving essential comforts."])

    else:
        # SQLite initialization
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
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
            conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# User Authentication Operations
# ---------------------------------------------------------------------------
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

    try:
        # Check existing email
        existing = execute_sql("SELECT id FROM users WHERE email = $1", [email])
        if existing:
            return None, "An account with this email address already exists. Please sign in."

        pwd_hash, salt = hash_password(password)
        inserted = execute_sql(
            "INSERT INTO users (email, password_hash, salt, full_name) VALUES ($1, $2, $3, $4) RETURNING id, email, full_name, created_at",
            [email, pwd_hash, salt, full_name]
        )
        if not inserted:
            return None, "Failed to create user record."

        user_id = inserted[0]["id"]
        # Create default financial profile
        execute_sql(
            "INSERT INTO financial_profiles (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
            [user_id]
        )

        user = {
            "id": user_id,
            "email": email,
            "full_name": full_name,
            "created_at": str(inserted[0].get("created_at", datetime.now().strftime("%Y-%m-%d")))[:10]
        }
        return user, None
    except Exception as e:
        err_msg = str(e)
        if "unique" in err_msg.lower() or "integrity" in err_msg.lower():
            return None, "An account with this email address already exists. Please sign in."
        return None, f"Registration failed: {err_msg}"


def authenticate_user(email: str, password: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Authenticates a user by email and password. Returns (user_dict, None) or (None, error_msg)."""
    email = email.strip().lower()
    if not email or not password:
        return None, "Please provide both email and password."

    rows = execute_sql(
        "SELECT id, email, password_hash, salt, full_name, created_at FROM users WHERE email = $1",
        [email]
    )
    if not rows:
        return None, "Invalid email or password. Please try again."

    user_row = rows[0]
    if not verify_password(password, user_row["password_hash"], user_row["salt"]):
        return None, "Invalid email or password. Please try again."

    return {
        "id": user_row["id"],
        "email": user_row["email"],
        "full_name": user_row["full_name"],
        "created_at": str(user_row.get("created_at", ""))[:10]
    }, None


# ---------------------------------------------------------------------------
# User Financial Profile Operations
# ---------------------------------------------------------------------------
def get_user_profile(user_id: int) -> Dict[str, Any]:
    """Retrieves the financial profile for a specific user."""
    rows = execute_sql("SELECT * FROM financial_profiles WHERE user_id = $1", [user_id])
    if rows:
        return dict(rows[0])
    
    # Default fallback
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
    upsert_sql = """
    INSERT INTO financial_profiles (
        user_id, monthly_income, rent, groceries, utilities, debt_emi,
        transport, dining_out, shopping, entertainment, subscriptions,
        existing_savings, goal_amount, goal_deadline_months, objective, updated_at
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, CURRENT_TIMESTAMP
    )
    ON CONFLICT (user_id) DO UPDATE SET
        monthly_income = EXCLUDED.monthly_income,
        rent = EXCLUDED.rent,
        groceries = EXCLUDED.groceries,
        utilities = EXCLUDED.utilities,
        debt_emi = EXCLUDED.debt_emi,
        transport = EXCLUDED.transport,
        dining_out = EXCLUDED.dining_out,
        shopping = EXCLUDED.shopping,
        entertainment = EXCLUDED.entertainment,
        subscriptions = EXCLUDED.subscriptions,
        existing_savings = EXCLUDED.existing_savings,
        goal_amount = EXCLUDED.goal_amount,
        goal_deadline_months = EXCLUDED.goal_deadline_months,
        objective = EXCLUDED.objective,
        updated_at = CURRENT_TIMESTAMP
    """
    execute_sql(upsert_sql, [
        user_id,
        float(profile_data.get("monthly_income", 60000.0)),
        float(profile_data.get("rent", 18000.0)),
        float(profile_data.get("groceries", 9000.0)),
        float(profile_data.get("utilities", 3500.0)),
        float(profile_data.get("debt_emi", 5000.0)),
        float(profile_data.get("transport", 3500.0)),
        float(profile_data.get("dining_out", 5000.0)),
        float(profile_data.get("shopping", 4000.0)),
        float(profile_data.get("entertainment", 2500.0)),
        float(profile_data.get("subscriptions", 1500.0)),
        float(profile_data.get("existing_savings", 30000.0)),
        float(profile_data.get("goal_amount", 150000.0)),
        int(profile_data.get("goal_deadline_months", 12)),
        str(profile_data.get("objective", "Build emergency savings.")),
    ])
    return True


# ---------------------------------------------------------------------------
# Saved Financial Plans Operations
# ---------------------------------------------------------------------------
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
    insert_sql = """
        INSERT INTO saved_plans (
            user_id, title, goal_amount, required_monthly_saving,
            deadline_months, total_monthly_cut, strategies_json
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
    """
    res = execute_sql(insert_sql, [
        user_id,
        title,
        float(goal_amount),
        float(required_monthly_saving),
        int(deadline_months),
        float(total_monthly_cut),
        json.dumps(strategies)
    ])
    if res and "id" in res[0]:
        return res[0]["id"]
    return 1


def get_user_plans(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all saved plans for a specific user, strictly user-isolated."""
    rows = execute_sql("""
        SELECT id, user_id, title, goal_amount, required_monthly_saving,
               deadline_months, total_monthly_cut, strategies_json, status, created_at
        FROM saved_plans
        WHERE user_id = $1
        ORDER BY created_at DESC
    """, [user_id])

    plans = []
    for r in rows:
        p = dict(r)
        try:
            p["strategies"] = json.loads(p["strategies_json"])
        except Exception:
            p["strategies"] = []
        p["created_at_fmt"] = str(p.get("created_at", ""))[:16].replace("T", " ")
        plans.append(p)
    return plans


def delete_plan(plan_id: int, user_id: int) -> bool:
    """Deletes a saved plan belonging to a user."""
    res = execute_sql(
        "DELETE FROM saved_plans WHERE id = $1 AND user_id = $2 RETURNING id",
        [plan_id, user_id]
    )
    return len(res) > 0


# ---------------------------------------------------------------------------
# Scenario History Operations
# ---------------------------------------------------------------------------
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
    insert_sql = """
        INSERT INTO scenario_history (
            user_id, dining_cut, shopping_cut, sub_cut, income_change,
            projected_monthly_saving, projected_timeline
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
    """
    res = execute_sql(insert_sql, [
        user_id,
        float(dining_cut),
        float(shopping_cut),
        float(sub_cut),
        float(income_change),
        float(projected_monthly_saving),
        str(projected_timeline)
    ])
    if res and "id" in res[0]:
        return res[0]["id"]
    return 1


def get_user_recent_activity(user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    """Returns a unified timeline of recent saved plans and scenario runs for the user."""
    plan_rows = execute_sql("""
        SELECT 'plan' as type, title as description, total_monthly_cut as metric,
               created_at
        FROM saved_plans
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2
    """, [user_id, limit])

    scen_rows_raw = execute_sql("""
        SELECT 'scenario' as type, projected_monthly_saving as metric,
               created_at
        FROM scenario_history
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2
    """, [user_id, limit])

    scen_rows = []
    for r in scen_rows_raw:
        saving = float(r.get("metric", 0.0))
        scen_rows.append({
            "type": "scenario",
            "description": f"What-If Scenario: Projected ₹{int(saving):,}/mo",
            "metric": saving,
            "created_at": r.get("created_at")
        })

    combined = [dict(p) for p in plan_rows] + scen_rows
    combined.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return combined[:limit]


# Auto-initialize schema on import
init_db()
