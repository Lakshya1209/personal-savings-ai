"""
test_database.py — Unit tests for SQLite persistence, authentication, and user data isolation.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import database

class TestDatabasePersistence(unittest.TestCase):
    def setUp(self):
        database.init_db()

    def test_demo_user_authentication(self):
        user, err = database.authenticate_user("demo@finance.ai", "password123")
        self.assertIsNotNone(user)
        self.assertIsNone(err)
        self.assertEqual(user["email"], "demo@finance.ai")

    def test_invalid_login(self):
        user, err = database.authenticate_user("demo@finance.ai", "wrongpassword")
        self.assertIsNone(user)
        self.assertIsNotNone(err)

    def test_create_and_isolate_users(self):
        import uuid
        uid1 = f"user_{uuid.uuid4().hex[:6]}@test.com"
        uid2 = f"user_{uuid.uuid4().hex[:6]}@test.com"

        u1, err1 = database.create_user(uid1, "pass12345", "Alice")
        self.assertIsNotNone(u1)
        self.assertIsNone(err1)

        u2, err2 = database.create_user(uid2, "pass12345", "Bob")
        self.assertIsNotNone(u2)
        self.assertIsNone(err2)

        # Save profile for User 1
        database.save_user_profile(u1["id"], {
            "monthly_income": 99000,
            "rent": 30000,
            "groceries": 12000,
            "objective": "Alice specific goal"
        })

        # Save profile for User 2
        database.save_user_profile(u2["id"], {
            "monthly_income": 45000,
            "rent": 15000,
            "groceries": 8000,
            "objective": "Bob specific goal"
        })

        prof1 = database.get_user_profile(u1["id"])
        prof2 = database.get_user_profile(u2["id"])

        self.assertEqual(prof1["monthly_income"], 99000.0)
        self.assertEqual(prof2["monthly_income"], 45000.0)
        self.assertEqual(prof1["objective"], "Alice specific goal")
        self.assertEqual(prof2["objective"], "Bob specific goal")

        # Test saved plan isolation
        plan_id1 = database.save_plan(
            user_id=u1["id"],
            title="Alice Vacation Fund",
            goal_amount=50000,
            required_monthly_saving=5000,
            deadline_months=10,
            total_monthly_cut=4000,
            strategies=[{"name": "Cut dining", "saving": 2000}]
        )

        plans_u1 = database.get_user_plans(u1["id"])
        plans_u2 = database.get_user_plans(u2["id"])

        self.assertEqual(len(plans_u1), 1)
        self.assertEqual(plans_u1[0]["title"], "Alice Vacation Fund")
        self.assertEqual(len(plans_u2), 0)  # Bob must NOT see Alice's plan!

        # Bob cannot delete Alice's plan
        deleted_by_bob = database.delete_plan(plan_id1, u2["id"])
        self.assertFalse(deleted_by_bob)

        # Alice deletes her plan
        deleted_by_alice = database.delete_plan(plan_id1, u1["id"])
        self.assertTrue(deleted_by_alice)
        self.assertEqual(len(database.get_user_plans(u1["id"])), 0)

if __name__ == "__main__":
    unittest.main()
