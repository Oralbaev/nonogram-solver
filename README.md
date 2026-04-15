# Nonogram Solver

A Python CLI tool that solves nonogram puzzles of any size from row and column clues.

## What it does

Given row and column clues, the solver fills in a grid using:

1. **Constraint propagation** — for each line, finds all patterns compatible with
   the clue and the current board state, then fixes any cell that is the same
   across every compatible pattern.
2. **Backtracking** — if propagation stalls, guesses a cell value and recurses.

The grid size is derived automatically from the number of clues.

---

## Algorithm optimizations

Three bottlenecks from the naive version were identified and fixed:

### 1. Redundant pattern generation → LRU cache

**Old:** `generate_patterns(clue, length)` was called on every `infer_line` call,
regenerating the same pattern list repeatedly — even for lines that had not changed.
For a 30-cell line with a clue like `[5, 5, 5]` this can mean thousands of patterns
rebuilt from scratch on every propagation pass.

**Fix:** `@lru_cache` on `_generate_patterns(tuple(clue), length)`.
Patterns for a given `(clue, length)` pair are generated once and reused forever.

---

### 2. Full board scan on every iteration → Dirty queue

**Old:** The propagation loop iterated every row, then every column, on every pass,
even if nothing in that row/column had changed since the last check.
Cost per pass: `O(rows × cols)` even when most lines are already stable.

**Fix:** A dirty-queue tracks only lines with recent cell changes.
When a cell in row `r` changes, column `c` is added to the dirty set, and vice versa.
Only those lines are re-examined — untouched lines are skipped entirely.

---

### 3. Naive cell selection for backtracking → MRV heuristic

**Old:** Backtracking simply picked the first unknown cell `(r=0, c=0)` in scan order,
regardless of how constrained its row or column was. This leads to deep, wide search
trees when the chosen line still has many valid arrangements.

**Fix:** MRV (Minimum Remaining Values) — scan all lines with unknowns, count their
compatible patterns, and branch on the cell whose line has the fewest options.
The most constrained line is the one most likely to produce a contradiction quickly,
which prunes the search tree dramatically.

---

## How nonograms are stored

```python
row_clues: list[list[int]]
col_clues: list[list[int]]
```

Each inner list is the clue for one row or column — consecutive block lengths,
left to right / top to bottom. The grid size is `len(row_clues) × len(col_clues)`.

Example:
```python
row_clues = [
    [1, 1],
    [5],
    [5],
    [3],
    [1],
]

col_clues = [
    [2],
    [4],
    [4],
    [4],
    [2],
]
```

## Cell states

| Symbol | Meaning  | Internal value |
|--------|----------|----------------|
| `■`    | Filled   | `1`            |
| ` `    | Empty    | `0`            |
| `?`    | Unknown  | `-1`           |

## File structure

```
Nanogramm/
├── app.py        # Entry point — defines the puzzle and runs the solver
├── solver.py     # All solving logic (propagation, backtracking, validation)
├── formatter.py  # Board and clue formatting for console output
└── README.md     # This file
```

## How to run

```bash
python app.py
```

Requires Python 3.12+. No external dependencies.
