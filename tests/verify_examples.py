"""Verify the parser against all example cases.

Run from the project root:
    python tests/verify_examples.py
"""

import sys
import os
import ast
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES = [
    ("examples/case_01_matrix.png",       10, 10, "examples/case_01_matrix_expected.py"),
    ("examples/case_02_basic.png",         10, 10, "examples/case_02_basic_expected.py"),
    ("examples/case_03_double_digits.png", 15, 15, "examples/case_03_double_digits_expected.py"),
    ("examples/case_04_rectangular.png",   10, 15, "examples/case_04_rectangular_expected.py"),
    ("examples/case_05_matrix.png",        15, 15, "examples/case_05_matrix_expected.py"),
]


def load_expected(path: str) -> tuple[list, list]:
    with open(path) as f:
        src = f.read()
    ns: dict = {}
    exec(src, ns)
    return ns["row_clues"], ns["col_clues"]


def run_parser(image: str, rows: int, cols: int) -> tuple[list, list] | None:
    result = subprocess.run(
        [sys.executable, "src/parser.py", image, str(rows), str(cols)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    stderr: {result.stderr.strip()}")
        return None

    row_clues = col_clues = None
    for line in result.stdout.splitlines():
        if line.startswith("row_clues"):
            row_clues = ast.literal_eval(line.split("=", 1)[1].strip())
        elif line.startswith("col_clues"):
            col_clues = ast.literal_eval(line.split("=", 1)[1].strip())

    if row_clues is None or col_clues is None:
        print(f"    stdout: {result.stdout.strip()}")
        return None
    return row_clues, col_clues


def main() -> None:
    passed = 0
    failed = 0

    for image, rows, cols, expected_path in CASES:
        name = os.path.basename(image)
        print(f"\n{'─'*60}")
        print(f"  {name}  ({rows}×{cols})")

        expected_row, expected_col = load_expected(expected_path)
        parsed = run_parser(image, rows, cols)

        if parsed is None:
            print("  FAIL  — parser returned no output")
            failed += 1
            continue

        parsed_row, parsed_col = parsed
        row_ok = parsed_row == expected_row
        col_ok = parsed_col == expected_col

        if row_ok and col_ok:
            print("  PASS")
            passed += 1
        else:
            print("  FAIL")
            failed += 1
            if not row_ok:
                print(f"    row_clues mismatch:")
                for i, (got, exp) in enumerate(zip(parsed_row, expected_row)):
                    if got != exp:
                        print(f"      row {i}: got {got}, expected {exp}")
                if len(parsed_row) != len(expected_row):
                    print(f"      lengths: got {len(parsed_row)}, expected {len(expected_row)}")
            if not col_ok:
                print(f"    col_clues mismatch:")
                for i, (got, exp) in enumerate(zip(parsed_col, expected_col)):
                    if got != exp:
                        print(f"      col {i}: got {got}, expected {exp}")
                if len(parsed_col) != len(expected_col):
                    print(f"      lengths: got {len(parsed_col)}, expected {len(expected_col)}")

    print(f"\n{'─'*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(CASES)} cases")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
