"""Entry point for the nonogram solver.

Usage:
    python main.py

The puzzle clues are defined below. Edit row_clues and col_clues to change
the puzzle. The solver works for any rectangular grid; dimensions are inferred
automatically from the number of clue lists.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os

from src.solver import UNKNOWN, solve, validate
from src.formatter import render_png

# ---------------------------------------------------------------------------
# Puzzle definition (20 rows × 25 columns)
# Edit these lists to solve a different puzzle.
# ---------------------------------------------------------------------------

row_clues = [[5, 4], [2, 2, 6], [2, 2, 2, 5], [1, 1, 1, 1, 4], [1, 1, 1, 1, 2], [2, 2, 2, 2, 1], [2, 2, 3, 1, 4], [5, 4, 6], [1, 2, 2, 1, 2, 2], [5, 3, 2, 1, 2, 2], [2, 2, 2, 1, 6], [2, 3, 5, 8], [1, 5, 1, 3, 6], [1, 6, 1, 5, 2, 2], [2, 3, 11, 2, 2], [2, 2, 2, 2, 6], [5, 5, 1, 4], [2, 2], [1, 2, 2], [4, 1], [4, 1], [4, 2], [4, 2], [10, 3, 2, 12], [3], [10, 3, 2, 11], [2, 2], [9, 2, 5, 9], [4, 6], [7, 5, 10]]
col_clues = [[1, 1, 1, 1], [1, 1, 1, 1], [4, 1, 1, 1, 1], [4, 2, 2, 1, 1, 1, 1], [2, 2, 2, 2, 2, 1, 1, 1, 1], [2, 2, 1, 4, 1, 1, 1, 1, 1], [1, 1, 1, 4, 1, 1, 1, 1, 1], [1, 3, 4, 1, 1, 1, 1], [1, 1, 2, 2, 2, 1, 1, 1, 1], [2, 3, 2, 3, 1, 1, 2], [2, 2, 2, 1, 1, 2], [4, 1, 4, 2, 5], [1, 1, 1, 3, 5], [3, 2, 5, 4, 1], [1, 2, 2, 4, 4, 1], [2, 1, 2, 2, 1, 4, 2, 2], [2, 1, 4, 2, 1, 3, 4, 2], [3, 2, 1, 3, 2, 1, 2, 5], [4, 2, 5, 2, 2, 1, 3], [6, 10, 2, 1, 1, 2], [4, 1, 1, 1, 1], [2, 3, 1, 1, 1, 1], [9, 1, 1, 1, 1], [11, 1, 1, 1, 1], [2, 3, 2, 1, 1, 1, 1], [2, 3, 2, 1, 1, 1, 1], [5, 5, 1, 1, 1, 1], [3, 3, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]



def main() -> None:
    """Run the nonogram solver and save the result as solution.png."""
    rows = len(row_clues)
    cols = len(col_clues)
    board = [[UNKNOWN] * cols for _ in range(rows)]

    print("=== Nonogram Solver ===")
    print(f"Grid: {rows} rows × {cols} cols")
    print()

    solution = solve(board, row_clues, col_clues)

    if solution is None:
        print("No solution found.")
        return

    if not validate(solution, row_clues, col_clues):
        print("Solver returned an invalid solution.")
        return

    path = render_png(solution)
    print(f"Saved: {path}")
    if sys.platform == "win32":
        os.startfile(path)


if __name__ == "__main__":
    main()
