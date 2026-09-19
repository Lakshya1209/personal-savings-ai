"""
data/generate_sample_data.py — Generates synthetic 6-month financial snapshots
for 6 distinct user profiles with rich categories.
"""

import os
import pandas as pd
import numpy as np

def generate_sample_dataset(output_path: str = "data/sample_data.csv") -> pd.DataFrame:
    """
    Generates a realistic multi-user, multi-month personal finance dataset.
    Covers 6 distinct user archetypes:
    1. User 101 - Rahul: Young Software Engineer (moderate discretionary spend, comfortable surplus)
    2. User 102 - Priya: High-earning Consultant (high dining & shopping, large goal)
    3. User 103 - Amit: Overextended Mid-career (heavy EMI, high rent, tight savings - ideal for rejection demo)
    4. User 104 - Sneha: Frugal Freelancer (variable income, conservative essentials)
    5. User 105 - Vikram: Recent Graduate (low income, high subscriptions/entertainment)
    6. User 106 - Ananya: Family Primary Earner (high essentials, balanced spending)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
    
    users = [
        {
            "user_id": 101,
            "name": "Rahul Verma",
            "archetype": "Tech Professional",
            "base_income": 85000,
            "rent": 22000,
            "groceries": 10000,
            "utilities": 4500,
            "debt_emi": 5000,
            "transport": 4000,
            "dining_out": 8500,
            "shopping": 7000,
            "entertainment": 4000,
            "subscriptions": 2500,
            "existing_savings": 150000,
            "goal_amount": 200000,
            "goal_deadline_months": 12,
        },
        {
            "user_id": 102,
            "name": "Priya Sharma",
            "archetype": "Corporate Consultant",
            "base_income": 140000,
            "rent": 38000,
            "groceries": 14000,
            "utilities": 6500,
            "debt_emi": 12000,
            "transport": 7500,
            "dining_out": 16000,
            "shopping": 15000,
            "entertainment": 8000,
            "subscriptions": 4500,
            "existing_savings": 320000,
            "goal_amount": 500000,
            "goal_deadline_months": 18,
        },
        {
            "user_id": 103,
            "name": "Amit Patel",
            "archetype": "Overleveraged Salaried (Demo Test Case)",
            "base_income": 55000,
            "rent": 20000,
            "groceries": 11000,
            "utilities": 4000,
            "debt_emi": 15000,  # High debt
            "transport": 3500,
            "dining_out": 4500,
            "shopping": 3000,
            "entertainment": 2000,
            "subscriptions": 1200,
            "existing_savings": 25000,
            "goal_amount": 180000,
            "goal_deadline_months": 12,  # Requires ₹15,000/mo cut, which exceeds safe discretionary capacity!
        },
        {
            "user_id": 104,
            "name": "Sneha Roy",
            "archetype": "Freelance Designer",
            "base_income": 62000,
            "rent": 16000,
            "groceries": 8000,
            "utilities": 3200,
            "debt_emi": 0,
            "transport": 2500,
            "dining_out": 5500,
            "shopping": 4000,
            "entertainment": 2500,
            "subscriptions": 1800,
            "existing_savings": 80000,
            "goal_amount": 100000,
            "goal_deadline_months": 8,
        },
        {
            "user_id": 105,
            "name": "Vikram Singh",
            "archetype": "Junior Developer",
            "base_income": 42000,
            "rent": 14000,
            "groceries": 7000,
            "utilities": 2800,
            "debt_emi": 3000,
            "transport": 3000,
            "dining_out": 6000,
            "shopping": 4500,
            "entertainment": 3500,
            "subscriptions": 2200,
            "existing_savings": 15000,
            "goal_amount": 80000,
            "goal_deadline_months": 10,
        },
        {
            "user_id": 106,
            "name": "Ananya Mukherjee",
            "archetype": "Senior Healthcare Specialist",
            "base_income": 95000,
            "rent": 26000,
            "groceries": 13000,
            "utilities": 5000,
            "debt_emi": 8000,
            "transport": 5000,
            "dining_out": 7000,
            "shopping": 6500,
            "entertainment": 3500,
            "subscriptions": 2000,
            "existing_savings": 220000,
            "goal_amount": 300000,
            "goal_deadline_months": 15,
        },
    ]

    records = []
    for u in users:
        for idx, m in enumerate(months):
            # Realistic monthly fluctuation (+/- 4%)
            fluct = 1.0 + (np.random.rand() - 0.5) * 0.08
            income_fluct = 1.0 if idx != 2 else 1.05 # occasional bonus
            
            income = round(u["base_income"] * income_fluct, 2)
            rent = u["rent"]  # fixed
            groceries = round(u["groceries"] * fluct, 2)
            utilities = round(u["utilities"] * (1.0 + (np.random.rand() - 0.5) * 0.1), 2)
            debt_emi = u["debt_emi"]  # fixed
            transport = round(u["transport"] * fluct, 2)
            dining_out = round(u["dining_out"] * fluct, 2)
            shopping = round(u["shopping"] * (1.0 + (np.random.rand() - 0.5) * 0.15), 2)
            entertainment = round(u["entertainment"] * fluct, 2)
            subscriptions = u["subscriptions"]  # fixed
            
            essential_total = round(rent + groceries + utilities + debt_emi, 2)
            discretionary_total = round(transport + dining_out + shopping + entertainment + subscriptions, 2)
            total_expense = round(essential_total + discretionary_total, 2)
            monthly_savings = round(income - total_expense, 2)
            savings_rate = round((monthly_savings / income) * 100, 2) if income > 0 else 0.0
            
            records.append({
                "user_id": u["user_id"],
                "user_name": u["name"],
                "archetype": u["archetype"],
                "month": m,
                "monthly_income": income,
                "rent": rent,
                "groceries": groceries,
                "utilities": utilities,
                "debt_emi": debt_emi,
                "transport": transport,
                "dining_out": dining_out,
                "shopping": shopping,
                "entertainment": entertainment,
                "subscriptions": subscriptions,
                "essential_spending": essential_total,
                "discretionary_spending": discretionary_total,
                "monthly_expense_total": total_expense,
                "actual_savings": monthly_savings,
                "savings_rate": savings_rate,
                "existing_savings": u["existing_savings"],
                "goal_amount": u["goal_amount"],
                "goal_deadline_months": u["goal_deadline_months"],
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated sample dataset at: {output_path} ({len(df)} rows, {len(users)} users)")
    return df

if __name__ == "__main__":
    generate_sample_dataset()
