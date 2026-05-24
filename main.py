"""Entry point for the nonogram solver.

Usage:
    python main.py

Prompts for an image file name and board dimensions, parses clues from the
image, solves the puzzle, and saves the result as solution.png.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os

from src.solver import UNKNOWN, solve, validate
from src.formatter import render_png
from src.parser import parse_nonogram_image


def _prompt_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("  Please enter a positive integer.")


def main() -> None:
    print("=== Nonogram Solver ===")

    filename = input("Enter image file name: ").strip()
    img_path = os.path.join("examples", filename)
    if not os.path.isfile(img_path):
        print(f"Error: file not found: {img_path}")
        return

    n_cols = _prompt_int("Enter number of columns: ")
    n_rows = _prompt_int("Enter number of rows: ")

    print(f"\nParsing {img_path} ({n_rows} rows × {n_cols} cols)...")
    try:
        row_clues, col_clues = parse_nonogram_image(img_path, n_rows, n_cols)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Parser error: {e}")
        return

    if len(row_clues) != n_rows or len(col_clues) != n_cols:
        print(
            f"Error: parsed {len(row_clues)} row clue(s) and {len(col_clues)} col clue(s), "
            f"expected {n_rows} and {n_cols}."
        )
        return

    board = [[UNKNOWN] * n_cols for _ in range(n_rows)]
    print("Solving...")
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
