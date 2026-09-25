"""Smoke test to verify that all core scientific, automation, and dashboard dependencies import cleanly."""
import sys
import importlib

def test_imports():
    packages = [
        "asyncua",
        "numpy",
        "pandas",
        "scipy",
        "ruptures",
        "hmmlearn",
        "streamlit",
        "plotly",
        "pytest",
    ]
    for pkg in packages:
        module = importlib.import_module(pkg)
        assert module is not None, f"Failed to import {pkg}"
        print(f"Loaded {pkg} v{getattr(module, '__version__', 'unknown')}")

if __name__ == "__main__":
    test_imports()
    print("All dependency imports passed successfully.")
