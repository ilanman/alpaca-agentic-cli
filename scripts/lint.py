#!/usr/bin/env python3
"""
Code linting script for finAI Trading Agent

Checks code quality using various tools:
- flake8 for style and basic errors
- black for code formatting
- mypy for type checking

Usage:
    python scripts/lint.py [--fix] [--check-only]
"""

import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    """Run all linting checks."""
    project_root = Path(__file__).parent.parent

    # Files to check
    python_files = [
        "main.py",
        "config.py",
        "chat/mcp_client.py",
        "chat/langchain_agent.py",
        "scripts/lint.py",
    ]

    # Check if files exist
    existing_files = [f for f in python_files if (project_root / f).exists()]

    if not existing_files:
        print("❌ No Python files found to lint")
        return 1

    print(f"📁 Linting {len(existing_files)} Python files...")

    # Run flake8
    flake8_cmd = ["flake8"] + existing_files
    flake8_success = run_command(flake8_cmd, "Flake8 style check")

    # Run black check
    black_cmd = ["black", "--check"] + existing_files
    black_success = run_command(black_cmd, "Black formatting check")

    # Run mypy
    mypy_cmd = ["mypy"] + existing_files
    mypy_success = run_command(mypy_cmd, "MyPy type checking")

    # Summary
    all_passed = flake8_success and black_success and mypy_success

    if all_passed:
        print("\n🎉 All linting checks passed!")
        return 0
    else:
        print("\n❌ Some linting checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
