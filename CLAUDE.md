# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Solve a puzzle (edit row_clues / col_clues in main.py first)
python main.py

# Parse clues from a screenshot (requires Tesseract + opencv-python + numpy)
python src/parser.py examples/case_01_matrix.png 10 10

# Stress-test the solver with a generated 30×30 puzzle
python tests/test_30x30.py

# Verify the image parser against all 5 reference examples
python -X utf8 tests/verify_examples.py
```

Always run `verify_examples.py` with `-X utf8` — the test runner prints Unicode box-drawing characters that fail on Windows without it.

Install parser dependencies (solver itself needs nothing beyond stdlib):

```bash
pip install -r requirements.txt
# Also requires Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
# Default path assumed: C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Architecture

The project has two independent pipelines that share no runtime state:

### Solver pipeline (`main.py` → `src/solver.py` → `src/formatter.py`)

Pure stdlib, no external dependencies.

- `main.py` hard-codes `row_clues` / `col_clues` and calls `solve()` then `render_png()`.
- `src/solver.py` exposes three public symbols: `UNKNOWN`, `solve()`, `validate()`.  
  Internally: `_generate_patterns()` (LRU-cached by `(clue_tuple, length)`) → `infer_line()` → `propagate()` (dirty-queue) → `_most_constrained_cell()` (MRV) → recursive `solve()`.  
  Cell values are `FILLED=1`, `EMPTY=0`, `UNKNOWN=-1`.
- `src/formatter.py` renders to PNG using only `struct` + `zlib` (no Pillow). Also provides `format_board()` for a Unicode console view.

### Parser pipeline (`src/parser.py`)

Reads a browser screenshot and extracts clues as Python code. Requires opencv-python, numpy, pytesseract.

```
detect_grid_lines() → infer_clue_dimensions() → extract_cell() → ocr_cell() → build_clues() → validate_clues()
```

Key design decisions in `ocr_cell()`:
- Tries three binarization thresholds in order (190 → 170 → 150) to handle black, blue, and orange digit colors.
- Runs Tesseract in PSM modes 8, 7, 6 at each threshold and applies resolution heuristics:
  - Requires ≥ 2 PSM reads before accepting a result (prevents single-PSM false positives).
  - When all PSM results are two-digit numbers, trusts PSM 8 (single-word mode is most accurate for complete multi-digit reads, e.g. distinguishes "10" from "19").
  - When PSM 8 returns one digit but PSM 7 and 6 agree on a two-digit result, PSM 8 dropped a digit — uses the two-digit result.

`infer_clue_dimensions()` derives `n_top_rows` and `n_left_cols` from total detected grid lines minus the known board dimensions (`n_rows`, `n_cols` come from the CLI arguments).

### Test/example data

`examples/case_NN_*_expected.py` are ground-truth files with `row_clues` and `col_clues` variables. `tests/verify_examples.py` runs the parser as a subprocess and diffs its stdout against the expected files.

Empty row/column clues are represented as `[0]` (never an empty list).
