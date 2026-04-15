"""Nonogram solver using constraint propagation and backtracking.

Optimizations over the naive version:
  1. LRU-cached pattern generation  — patterns for the same (clue, length)
     pair are generated once and reused across all calls.
  2. Dirty-queue propagation        — only rows/columns whose cells changed
     are re-examined, instead of scanning the whole board every iteration.
  3. MRV backtracking heuristic     — when guessing, pick the unknown cell
     whose line has the fewest compatible patterns (most constrained first).
"""

import copy
from functools import lru_cache

UNKNOWN = -1
EMPTY = 0
FILLED = 1

Board = list[list[int]]


# ---------------------------------------------------------------------------
# Pattern generation (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _generate_patterns(clue: tuple[int, ...], length: int) -> list[tuple[int, ...]]:
    """Generate all valid patterns for a clue and line length.

    Results are cached: identical (clue, length) pairs are computed only once.
    Patterns are stored as tuples so they are hashable and immutable.
    """
    if not clue:
        return [(EMPTY,) * length]

    patterns: list[tuple[int, ...]] = []

    def backtrack(pos: int, block_idx: int, current: list[int]) -> None:
        if block_idx == len(clue):
            full = current + [EMPTY] * (length - pos)
            patterns.append(tuple(full))
            return

        block_size = clue[block_idx]
        remaining = clue[block_idx:]
        min_space = sum(remaining) + len(remaining) - 1

        for start in range(pos, length - min_space + 1):
            prefix = [EMPTY] * (start - pos)
            block = [FILLED] * block_size

            if block_idx < len(clue) - 1:
                segment = prefix + block + [EMPTY]
                next_pos = start + block_size + 1
            else:
                segment = prefix + block
                next_pos = start + block_size

            backtrack(next_pos, block_idx + 1, current + segment)

    backtrack(0, 0, [])
    return patterns


def _compatible_patterns(clue: list[int], line: list[int]) -> list[tuple[int, ...]]:
    """Filter cached patterns to those compatible with the current line state."""
    return [
        p for p in _generate_patterns(tuple(clue), len(line))
        if all(l == UNKNOWN or p[i] == l for i, l in enumerate(line))
    ]


# ---------------------------------------------------------------------------
# Line inference
# ---------------------------------------------------------------------------

def infer_line(clue: list[int], line: list[int]) -> list[int] | None:
    """Return an updated line with newly determined cells, or None on contradiction.

    A cell is fixed if every compatible pattern agrees on its value.
    """
    compatible = _compatible_patterns(clue, line)
    if not compatible:
        return None

    result = line[:]
    for i in range(len(line)):
        if result[i] != UNKNOWN:
            continue
        values = {p[i] for p in compatible}
        if len(values) == 1:
            result[i] = values.pop()

    return result


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def get_column(board: Board, col: int) -> list[int]:
    """Extract column col from the board."""
    return [board[row][col] for row in range(len(board))]


def set_column(board: Board, col: int, line: list[int]) -> None:
    """Write line back as column col."""
    for row in range(len(board)):
        board[row][col] = line[row]


# ---------------------------------------------------------------------------
# Propagation (dirty-queue)
# ---------------------------------------------------------------------------

def propagate(
    board: Board, row_clues: list[list[int]], col_clues: list[list[int]]
) -> Board | None:
    """Apply constraint propagation using a dirty queue.

    Only rows and columns whose cells have actually changed are re-examined.
    This avoids the O(rows*cols) full scan on every iteration of the old loop.
    """
    rows = len(board)
    cols = len(board[0])

    dirty_rows: set[int] = set(range(rows))
    dirty_cols: set[int] = set(range(cols))

    while dirty_rows or dirty_cols:
        while dirty_rows:
            r = dirty_rows.pop()
            new_line = infer_line(row_clues[r], board[r])
            if new_line is None:
                return None
            for c in range(cols):
                if new_line[c] != board[r][c]:
                    dirty_cols.add(c)
            board[r] = new_line

        while dirty_cols:
            c = dirty_cols.pop()
            col = get_column(board, c)
            new_col = infer_line(col_clues[c], col)
            if new_col is None:
                return None
            for r in range(rows):
                if new_col[r] != col[r]:
                    dirty_rows.add(r)
            set_column(board, c, new_col)

    return board


# ---------------------------------------------------------------------------
# MRV cell selection
# ---------------------------------------------------------------------------

def _most_constrained_cell(
    board: Board, row_clues: list[list[int]], col_clues: list[list[int]]
) -> tuple[int, int] | None:
    """Return the unknown cell belonging to the most constrained line.

    'Most constrained' means the line with the fewest compatible patterns.
    Branching here minimises the size of the backtracking search tree (MRV).
    """
    rows = len(board)
    cols = len(board[0])

    best_count = float("inf")
    best_cell: tuple[int, int] | None = None

    for r in range(rows):
        if UNKNOWN not in board[r]:
            continue
        count = len(_compatible_patterns(row_clues[r], board[r]))
        if count < best_count:
            best_count = count
            for c in range(cols):
                if board[r][c] == UNKNOWN:
                    best_cell = (r, c)
                    break

    for c in range(cols):
        col = get_column(board, c)
        if UNKNOWN not in col:
            continue
        count = len(_compatible_patterns(col_clues[c], col))
        if count < best_count:
            best_count = count
            for r in range(rows):
                if board[r][c] == UNKNOWN:
                    best_cell = (r, c)
                    break

    return best_cell


# ---------------------------------------------------------------------------
# Solver entry point
# ---------------------------------------------------------------------------

def solve(
    board: Board, row_clues: list[list[int]], col_clues: list[list[int]]
) -> Board | None:
    """Solve the nonogram using propagation followed by MRV backtracking.

    Works for any grid size; dimensions are inferred from the board.
    Returns the solved board, or None if no solution exists.
    """
    board = copy.deepcopy(board)
    board = propagate(board, row_clues, col_clues)
    if board is None:
        return None

    cell = _most_constrained_cell(board, row_clues, col_clues)
    if cell is None:
        return board  # No unknowns left — solved

    r, c = cell
    for value in (FILLED, EMPTY):
        candidate = copy.deepcopy(board)
        candidate[r][c] = value
        result = solve(candidate, row_clues, col_clues)
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    board: Board, row_clues: list[list[int]], col_clues: list[list[int]]
) -> bool:
    """Validate that the solved board satisfies all row and column clues."""
    rows = len(board)
    cols = len(board[0]) if board else 0

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
        return clue

    for r in range(rows):
        if extract_clue(board[r]) != row_clues[r]:
            return False
    for c in range(cols):
        if extract_clue(get_column(board, c)) != col_clues[c]:
            return False

    return True
