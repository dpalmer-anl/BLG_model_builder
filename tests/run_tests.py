"""
run_tests.py
------------
Run all tests in the tests folder.

Usage
-----
    # From project root (recommended):
    python tests/run_tests.py

    # Or with pytest directly:
    pytest tests/ -v

    # Run specific test file:
    python tests/run_tests.py tests/test_calculators.py
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run all tests: pytest on test_*.py, then legacy test_interface."""
    root = Path(__file__).resolve().parent.parent
    tests_dir = Path(__file__).resolve().parent

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    exit_code = 0

    # 1. Run pytest on test_calculators, test_pod_torch, test_tetb
    try:
        import pytest
        # Exclude test_interface (legacy runner) from pytest
        args = (
            [str(tests_dir), "-v", f"--ignore={tests_dir / 'test_interface.py'}"]
            + sys.argv[1:]
        )
        exit_code = pytest.main(args)
    except ImportError:
        print("pytest not found; skipping pytest-based tests.")

    # 2. Run legacy test_interface (Tersoff, KC, DRIP, POD, compute_hoppings)
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("  Legacy test_interface")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, str(tests_dir / "test_interface.py")],
            cwd=str(root),
        )
        if result.returncode != 0:
            exit_code = result.returncode

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
