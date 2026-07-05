"""Shared pytest configuration.

Adds the project root to sys.path so every test can import project modules
without needing an installed package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
