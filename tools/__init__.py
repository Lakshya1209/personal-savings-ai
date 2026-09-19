"""
tools package initialization
"""
from .expense_tools import category_totals, essential_vs_discretionary, detect_recurring
from .budget_tools import calculate_savings_rate, calculate_discretionary_capacity
from .goal_tools import calculate_required_monthly_saving, project_completion_date
from .savings_tools import simulate_category_reduction, combine_strategies
from .safety_tools import check_essential_floor, check_emergency_buffer
