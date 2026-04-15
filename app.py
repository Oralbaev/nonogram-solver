"""Entry point for the nonogram solver."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os

from solver import UNKNOWN, solve, validate
from formatter import render_png
from parser import parse_png

row_clues = [
    [7, 4, 2],
    [5, 4, 4],
    [5, 4, 1],
    [2, 5, 6],
    [2, 6, 2, 5],
    [2, 3, 2, 2, 2, 1, 1],
    [1, 3, 1, 1, 1, 1, 1],
    [4, 6, 1, 3],
    [1, 11],
    [4, 1, 4, 4],
    [2, 2, 2, 5, 4],
    [2, 2, 1, 5, 4],
    [3, 5, 4, 1, 1, 3],
    [2, 3, 2, 3, 3],
    [1, 3, 4, 4, 2],
    [4, 3, 2, 1, 1, 1, 1],
    [1, 3, 3, 3, 3],
    [2, 7, 2, 2],
    [2, 6],
    [14, 1],
]
col_clues = [
    [1, 1, 1],
    [8],
    [2, 1, 1, 2],
    [1, 3, 1, 2, 2, 2],
    [2, 2, 1, 1, 1, 1, 1],
    [5, 2, 2, 2, 2, 1],
    [4, 3, 2, 1, 1, 1],
    [3, 3, 2, 2, 1, 1],
    [2, 3, 1, 1, 1, 1],
    [1, 3, 1, 2, 1, 1],
    [7, 1, 1, 1, 1],
    [6, 3, 1, 2, 1],
    [4, 4, 2, 1, 1],
    [2, 3, 3, 2, 1],
    [1, 2, 2, 3, 2, 1],
    [1, 2, 6, 1, 2],
    [2, 1, 12, 2],
    [1, 3, 5, 2, 3],
    [1, 2, 3, 2, 4],
    [9, 2],
    [1, 2, 3, 2, 4],
    [3, 4, 2, 2],
    [10],
    [6],
    [2, 2],
]
pathpng = sys.argv[1] if len(sys.argv) > 1 else "matrix.png"


def main() -> None:
    """Run the nonogram solver and print the result."""
    rows = len(row_clues)
    cols = len(col_clues)
    board = [[UNKNOWN] * cols for _ in range(rows)]

    print("=== Nonogram Solver ===")
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
    os.startfile(path)


if __name__ == "__main__":
    main()
