"""30x30 nonogram stress test.

Generates a known solution, derives clues from it,
runs the solver, verifies the result, and reports timing.

Run from the project root:
    python tests/test_30x30.py
"""

import sys
import os
import time

# Make the project root importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout.reconfigure(encoding="utf-8")

from src.solver import UNKNOWN, EMPTY, FILLED, solve, validate, get_column
from src.formatter import format_board


# ---------------------------------------------------------------------------
# Build a 30x30 solution with a visible pattern
# (varied diagonals and blocks — diverse clues per row/column so propagation
#  alone can resolve most of the puzzle)
# ---------------------------------------------------------------------------

SIZE = 30

solution: list[list[int]] = [
    [
        FILLED if (
            (r % 7 < 3 and c % 5 < 2) or
            (r % 11 > 6 and c % 7 < 4) or
            (abs(r - c) % 9 < 2) or
            (r + c) % 13 < 3
        ) else EMPTY
        for c in range(SIZE)
    ]
    for r in range(SIZE)
]


def extract_clue(line: list[int]) -> list[int]:
    clue: list[int] = []
    count = 0
    for cell in line:
        if cell == FILLED:
            count += 1
        elif count > 0:
            clue.append(count)
            count = 0
    if count > 0:
        clue.append(count)
    return clue or [0]


row_clues = [extract_clue(solution[r]) for r in range(SIZE)]
col_clues = [extract_clue(get_column(solution, c)) for c in range(SIZE)]


# ---------------------------------------------------------------------------
# Print puzzle info
# ---------------------------------------------------------------------------

n_rows = SIZE
n_cols = SIZE
print(f"Grid size   : {n_rows}×{n_cols}")
print(f"Total cells : {n_rows * n_cols}")
filled_count = sum(cell == FILLED for row in solution for cell in row)
print(f"Filled cells: {filled_count}  ({100 * filled_count // (n_rows * n_cols)}%)")
print()

print("Sample row clues (first 5):")
for i, c in enumerate(row_clues[:5]):
    print(f"  Row {i+1:2d}: {c}")
print()
print("Sample col clues (first 5):")
for i, c in enumerate(col_clues[:5]):
    print(f"  Col {i+1:2d}: {c}")
print()

# ---------------------------------------------------------------------------
# Solve and time
# ---------------------------------------------------------------------------

board = [[UNKNOWN] * n_cols for _ in range(n_rows)]

print("Solving...", flush=True)
t0 = time.perf_counter()
result = solve(board, row_clues, col_clues)
elapsed = time.perf_counter() - t0

print(f"Time: {elapsed:.3f}s")
print()

if result is None:
    print("No solution found.")
    sys.exit(1)

if not validate(result, row_clues, col_clues):
    print("INVALID solution — validation failed.")
    sys.exit(1)

if result != solution:
    print("WRONG solution — does not match expected.")
    sys.exit(1)

print("Solution is CORRECT.")
print()
print(format_board(result, row_clues, col_clues))
