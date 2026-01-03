"""
Simulation module for multi-day field experiment planning.

This module simulates multi-day field experiments to estimate sample sizes
and generate daily work plans.
"""
from .day1 import generate_day1_plan
from .multiday import simulate_multiday_experiment

__all__ = ["generate_day1_plan", "simulate_multiday_experiment"]
