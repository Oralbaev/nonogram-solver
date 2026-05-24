"""Nonogram image parser.

Reads a clean browser screenshot of a nonogram puzzle and outputs
the row and column clue arrays as Python-ready code.

Usage:
    python src/parser.py IMAGE ROWS COLS

    IMAGE  path to the puzzle screenshot
    ROWS   number of board rows  (not counting the clue area)
    COLS   number of board columns

Example:
    python src/parser.py examples/case_01_matrix.png 10 10

Requirements:
    opencv-python, numpy, pytesseract, Pillow
    Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
    If tesseract.exe is not on PATH, set TESSERACT_CMD below.
"""

import sys
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ---------------------------------------------------------------------------
# Tesseract binary path (Windows)
# If tesseract.exe is not on your PATH, set it here:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ---------------------------------------------------------------------------
_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(_TESSERACT_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

DARK_THRESH  = 160  # gray < this → dark pixel (grid lines are ~0–80)
LINE_FRAC    = 0.25 # fraction of row/col that must be dark to count as a grid line
MERGE_DIST   = 5    # merge adjacent detected line midpoints within this many px
MIN_CELL_PX  = 15   # minimum valid cell size; gaps smaller than this are thick
                    # separators (border between clue area and board) and are filtered
INK_THRESH   = 190  # binarize cell: < 190 → ink, ≥ 190 → background
                    # captures black, blue (~29), and orange (~155) text
MIN_INK_PX   = 15   # cells with fewer dark pixels are treated as empty → None
UPSCALE      = 4    # upscale factor applied before OCR
PAD_PX       = 6    # white padding (px) added around the upscaled cell for OCR
CELL_TRIM    = 2    # px trimmed from each cell edge to remove grid-line bleed

_TESS_PSMS = [8, 7, 6]
_TESS_BASE = "--oem 3 -c tessedit_char_whitelist=0123456789"
# Binarization thresholds tried in order (190 → 170 → 150).
# Lower thresholds are stricter (fewer pixels count as ink).  They help when
# the primary threshold leaves too little contrast for Tesseract to read.
_INK_THRESHOLDS = [190, 170, 150]


# ---------------------------------------------------------------------------
# Step 1 – Grid line detection
# ---------------------------------------------------------------------------

def _runs_to_midpoints(flags: list[bool]) -> list[int]:
    """Convert a boolean list to a list of midpoints of True-runs."""
    midpoints: list[int] = []
    in_run = False
    run_start = 0
    for i, flag in enumerate(flags):
        if flag and not in_run:
            in_run = True
            run_start = i
        elif not flag and in_run:
            in_run = False
            midpoints.append((run_start + i - 1) // 2)
    if in_run:
        midpoints.append((run_start + len(flags) - 1) // 2)
    return midpoints


def _merge_close(points: list[int], dist: int) -> list[int]:
    """Merge points that are within dist of each other (keep average)."""
    if not points:
        return []
    merged = [points[0]]
    for p in points[1:]:
        if p - merged[-1] <= dist:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    return merged


def _filter_min_gap(lines: list[int], min_gap: int) -> list[int]:
    """Drop any line that is less than min_gap pixels after the previous kept line.

    Thick visual separators (e.g. between the clue area and the board) create
    two or more closely-spaced dark bands. This pass collapses them: the first
    line of the pair is kept, subsequent lines that are too close are skipped.
    """
    if not lines:
        return []
    result = [lines[0]]
    for line in lines[1:]:
        if line - result[-1] >= min_gap:
            result.append(line)
    return result


def detect_grid_lines(gray: np.ndarray) -> tuple[list[int], list[int]]:
    """Find horizontal and vertical grid lines by dark-pixel projection.

    Returns two sorted lists of pixel positions: h_lines and v_lines.
    Each entry is the midpoint (pixel index) of one grid line band.
    """
    h, w = gray.shape
    dark = gray < DARK_THRESH

    h_flags = [(int(dark[r].sum()) / w) > LINE_FRAC for r in range(h)]
    v_flags = [(int(dark[:, c].sum()) / h) > LINE_FRAC for c in range(w)]

    h_lines = _filter_min_gap(_merge_close(_runs_to_midpoints(h_flags), MERGE_DIST), MIN_CELL_PX)
    v_lines = _filter_min_gap(_merge_close(_runs_to_midpoints(v_flags), MERGE_DIST), MIN_CELL_PX)

    if len(h_lines) < 2:
        raise RuntimeError(
            f"Only {len(h_lines)} horizontal grid line(s) found. "
            "Check that the image is a clean nonogram screenshot."
        )
    if len(v_lines) < 2:
        raise RuntimeError(
            f"Only {len(v_lines)} vertical grid line(s) found."
        )
    return h_lines, v_lines


# ---------------------------------------------------------------------------
# Step 2 – Infer clue region dimensions
# ---------------------------------------------------------------------------

def infer_clue_dimensions(
    h_lines: list[int], v_lines: list[int], n_rows: int, n_cols: int
) -> tuple[int, int]:
    """Compute how many rows/cols of cells the clue areas occupy.

    The total grid is (n_top_rows + n_rows) × (n_left_cols + n_cols).
    Grid lines bound each cell, so:
        n_total_rows = len(h_lines) - 1
        n_top_rows   = n_total_rows - n_rows
    """
    n_total_rows = len(h_lines) - 1
    n_total_cols = len(v_lines) - 1
    n_top_rows   = n_total_rows - n_rows
    n_left_cols  = n_total_cols - n_cols

    if n_top_rows < 1:
        raise RuntimeError(
            f"n_top_rows = {n_top_rows} (need ≥ 1). "
            f"Detected {len(h_lines)} horizontal lines for {n_rows} board rows. "
            "Try decreasing MERGE_DIST or check the image."
        )
    if n_left_cols < 1:
        raise RuntimeError(
            f"n_left_cols = {n_left_cols} (need ≥ 1). "
            f"Detected {len(v_lines)} vertical lines for {n_cols} board cols."
        )
    return n_top_rows, n_left_cols


# ---------------------------------------------------------------------------
# Step 3 – Cell extraction
# ---------------------------------------------------------------------------

def extract_cell(
    img_bgr: np.ndarray,
    h_lines: list[int],
    v_lines: list[int],
    row_idx: int,
    col_idx: int,
) -> np.ndarray:
    """Extract the BGR crop for the cell at grid position (row_idx, col_idx)."""
    y1 = h_lines[row_idx]     + CELL_TRIM
    y2 = h_lines[row_idx + 1] - CELL_TRIM
    x1 = v_lines[col_idx]     + CELL_TRIM
    x2 = v_lines[col_idx + 1] - CELL_TRIM
    if y2 <= y1 or x2 <= x1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return img_bgr[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# Step 4 – OCR a single cell
# ---------------------------------------------------------------------------

def ocr_cell(cell_bgr: np.ndarray) -> int | None:
    """Read the digit(s) from a single clue cell.

    Returns an integer (1–99) or None for an empty cell.

    Per-threshold strategy (190 → 170 → 150):
      1. Skip if fewer than MIN_INK_PX dark pixels (treat as empty).
      2. Run all three Tesseract PSM modes; reject values outside 1–99.
      3. Resolve:
         a. All two-digit results → majority vote; PSM 8 as tiebreaker.
         b. Majority vote (≥ 2 PSMs agree).
         c. PSM 8 dropped a digit → use two-digit result that PSM 7/6 agree on.
      4. If resolved, return immediately; otherwise continue to next threshold.

    Cross-threshold fallbacks (when per-threshold resolution fails):
      F1. Count every PSM/threshold read. If one value has ≥ 2 votes, return it,
          preferring the value first seen at the HIGHEST threshold when tied.
          (Fixes "9": PSM 6 reads "9" at thresholds 190 and 170 for 2 votes, while
          PSM 8 reads the wrong "3" only at 170 and 150 — same count, but "9" was
          first seen at the highest threshold so it wins.)
      F2. Single-vote from the lowest threshold only: digit invisible at higher
          thresholds but reads cleanly on the sharpest binarisation (e.g. "8" in
          some cases that only PSM 8 reads at thresh=150).
      F3. Topology correction: "8" is misread as "3" by PSM 8 at mid-thresholds
          because the digit's two loops resemble the two arcs of "3". Detect the
          difference via enclosed-region count: an "8" has 2 holes in its binary
          representation, "3" has 0.
    """
    if cell_bgr.size == 0:
        return None

    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Quick empty check at the base threshold.
    if int((gray < INK_THRESH).sum()) < MIN_INK_PX:
        return None

    def _read_psms(binary: np.ndarray) -> dict[int, int]:
        """Run all PSM modes and return {psm: value} for successful reads."""
        big = cv2.resize(binary, (w * UPSCALE, h * UPSCALE),
                         interpolation=cv2.INTER_LANCZOS4)
        padded = cv2.copyMakeBorder(big, PAD_PX, PAD_PX, PAD_PX, PAD_PX,
                                    cv2.BORDER_CONSTANT, value=255)
        pil_img = Image.fromarray(padded)
        out: dict[int, int] = {}
        for psm in _TESS_PSMS:
            raw = pytesseract.image_to_string(
                pil_img, config=f"--psm {psm} {_TESS_BASE}"
            ).strip()
            if raw.isdigit():
                v = int(raw)
                if 1 <= v <= 99:  # reject impossible clue values (e.g. "415")
                    out[psm] = v
        return out

    def _resolve(psm_results: dict[int, int]) -> int | None:
        """Pick one value from {psm: value} using heuristics for digit ambiguity."""
        if not psm_results:
            return None
        values = list(psm_results.values())
        counts: dict[int, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1

        # (a) All two-digit: use majority, but only commit when PSM 8 agrees with
        #     (or is absent from) the majority. If PSM 8 holds a different value,
        #     return None so the next threshold is tried — at lower thresholds the
        #     PSMs often converge on the correct reading.
        #     Examples this handles:
        #       "10": {8:10, 7:19, 6:19} at thresh=190 → defer; at thresh=150 all
        #             agree on 10 (PSM 7/6 confuse 0→9 at higher ink levels).
        #       "15": {8:45, 7:15, 6:15} at thresh=190 → defer; at thresh=170
        #             PSM 8 returns "415" (filtered out), leaving {7:15,6:15} with
        #             no PSM 8 to disagree → returns 15 immediately.
        if all(v >= 10 for v in values):
            best = max(counts, key=counts.get)
            if counts[best] >= 2:
                if 8 not in psm_results or psm_results[8] == best:
                    return best
                return None  # PSM 8 disagrees — try next threshold
            if 8 in psm_results:
                return psm_results[8]
            return values[0]

        # Need at least 2 reads for confidence.
        if len(values) < 2:
            return None

        # (b) Simple majority vote.
        best = max(counts, key=counts.get)
        if counts[best] >= 2:
            return best

        # (c) If PSM 8 gave a one-digit result but PSM 7 and 6 both gave the
        #     same two-digit result, PSM 8 likely dropped a digit.
        if (8 in psm_results and psm_results[8] < 10
                and 7 in psm_results and 6 in psm_results
                and psm_results[7] == psm_results[6]
                and psm_results[7] >= 10):
            return psm_results[7]

        return None

    global_votes: dict[int, int] = {}
    first_seen_idx: dict[int, int] = {}  # value → index in _INK_THRESHOLDS (0 = highest)

    for idx, thresh in enumerate(_INK_THRESHOLDS):
        binary = np.where(gray < thresh, 0, 255).astype(np.uint8)
        if int((binary == 0).sum()) < MIN_INK_PX:
            continue
        psm_results = _read_psms(binary)
        for v in psm_results.values():
            global_votes[v] = global_votes.get(v, 0) + 1
            if v not in first_seen_idx:
                first_seen_idx[v] = idx
        val = _resolve(psm_results)
        if val is not None:
            return val

    if not global_votes:
        return None

    best_count = max(global_votes.values())

    # F1: Cross-threshold vote — prefer the value first seen at the highest threshold
    # when there is a tie, since a read at thresh=190 (most ink, most complete image)
    # is more reliable than a read at thresh=150 (fewest ink pixels).
    if best_count >= 2:
        candidates = [v for v, c in global_votes.items() if c == best_count]
        return min(candidates, key=lambda v: first_seen_idx[v])

    # F2: Single-vote at the lowest threshold only — digit that only becomes readable
    # after aggressive binarisation (e.g. "8" visible only at thresh=150).
    last_idx = len(_INK_THRESHOLDS) - 1
    if (len(global_votes) == 1
            and all(first_seen_idx[v] == last_idx for v in global_votes)):
        return list(global_votes.keys())[0]

    # F3: Topology correction for "8" misread as "3" — an "8" has 2 enclosed regions
    # (the two loops), while "3" has 0. When the only vote is for "3" and the cell
    # topology has ≥ 2 holes, override to "8".
    if set(global_votes.keys()) == {3} and global_votes[3] == 1:
        thresh_3 = _INK_THRESHOLDS[first_seen_idx[3]]
        binary_3 = np.where(gray < thresh_3, 0, 255).astype(np.uint8)
        inv = cv2.bitwise_not(binary_3)
        _, hier = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            holes = sum(1 for i in range(len(hier[0])) if hier[0][i][3] >= 0)
            if holes >= 2:
                return 8

    return None


# ---------------------------------------------------------------------------
# Step 5 – Assemble clue arrays
# ---------------------------------------------------------------------------

def build_clues(
    img_bgr: np.ndarray,
    h_lines: list[int],
    v_lines: list[int],
    n_top_rows: int,
    n_left_cols: int,
    n_rows: int,
    n_cols: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """OCR all clue cells and assemble row_clues and col_clues."""

    # Top clue grid: n_top_rows × n_cols
    top_grid: list[list[int | None]] = []
    for r in range(n_top_rows):
        row_vals: list[int | None] = []
        for c in range(n_cols):
            cell = extract_cell(img_bgr, h_lines, v_lines, r, n_left_cols + c)
            row_vals.append(ocr_cell(cell))
        top_grid.append(row_vals)

    # Left clue grid: n_rows × n_left_cols
    left_grid: list[list[int | None]] = []
    for r in range(n_rows):
        row_vals = []
        for c in range(n_left_cols):
            cell = extract_cell(img_bgr, h_lines, v_lines, n_top_rows + r, c)
            row_vals.append(ocr_cell(cell))
        left_grid.append(row_vals)

    # Assemble col_clues: for each column c, collect top_grid[r][c] dropping None.
    col_clues: list[list[int]] = []
    for c in range(n_cols):
        clue = [top_grid[r][c] for r in range(n_top_rows) if top_grid[r][c] is not None]
        col_clues.append(clue if clue else [0])

    # Assemble row_clues: for each row r, collect left_grid[r][c] dropping None.
    row_clues: list[list[int]] = []
    for r in range(n_rows):
        clue = [v for v in left_grid[r] if v is not None]
        row_clues.append(clue if clue else [0])

    return row_clues, col_clues


# ---------------------------------------------------------------------------
# Step 6 – Validation
# ---------------------------------------------------------------------------

def validate_clues(
    row_clues: list[list[int]],
    col_clues: list[list[int]],
    n_rows: int,
    n_cols: int,
) -> None:
    """Raise ValueError if the parsed clues violate basic nonogram constraints."""
    if len(row_clues) != n_rows:
        raise ValueError(f"Expected {n_rows} row clues, got {len(row_clues)}")
    if len(col_clues) != n_cols:
        raise ValueError(f"Expected {n_cols} col clues, got {len(col_clues)}")

    for r, clue in enumerate(row_clues):
        needed = sum(clue) + max(len(clue) - 1, 0)
        if needed > n_cols:
            raise ValueError(f"Row {r}: clue {clue} needs {needed} cells but grid is only {n_cols} wide")

    for c, clue in enumerate(col_clues):
        needed = sum(clue) + max(len(clue) - 1, 0)
        if needed > n_rows:
            raise ValueError(f"Col {c}: clue {clue} needs {needed} cells but grid is only {n_rows} tall")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_nonogram_image(
    img_path: str, n_rows: int, n_cols: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Parse a nonogram screenshot and return (row_clues, col_clues).

    Raises FileNotFoundError, RuntimeError, or ValueError on failure.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    # cv2.imread() silently fails on paths with non-ASCII characters (e.g. Cyrillic).
    # np.fromfile + imdecode works correctly with any Unicode path.
    img_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not load image: {img_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = detect_grid_lines(gray)
    n_top_rows, n_left_cols = infer_clue_dimensions(h_lines, v_lines, n_rows, n_cols)
    row_clues, col_clues = build_clues(
        img_bgr, h_lines, v_lines, n_top_rows, n_left_cols, n_rows, n_cols
    )
    validate_clues(row_clues, col_clues, n_rows, n_cols)
    return row_clues, col_clues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple[str, int, int]:
    if len(sys.argv) != 4:
        print("Usage: python src/parser.py IMAGE ROWS COLS", file=sys.stderr)
        print("  IMAGE  path to the puzzle screenshot", file=sys.stderr)
        print("  ROWS   number of board rows (not counting clue area)", file=sys.stderr)
        print("  COLS   number of board columns", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"Error: image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    try:
        n_rows = int(sys.argv[2])
        n_cols = int(sys.argv[3])
        if n_rows < 1 or n_cols < 1:
            raise ValueError
    except ValueError:
        print("Error: ROWS and COLS must be positive integers", file=sys.stderr)
        sys.exit(1)

    return img_path, n_rows, n_cols


def main() -> None:
    img_path, n_rows, n_cols = parse_args()

    img_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: could not load image: {img_path}", file=sys.stderr)
        sys.exit(1)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    try:
        h_lines, v_lines = detect_grid_lines(gray)
    except RuntimeError as e:
        print(f"Error (grid detection): {e}", file=sys.stderr)
        sys.exit(1)

    try:
        n_top_rows, n_left_cols = infer_clue_dimensions(h_lines, v_lines, n_rows, n_cols)
    except RuntimeError as e:
        print(f"Error (clue dimensions): {e}", file=sys.stderr)
        sys.exit(1)

    row_clues, col_clues = build_clues(
        img_bgr, h_lines, v_lines, n_top_rows, n_left_cols, n_rows, n_cols
    )

    try:
        validate_clues(row_clues, col_clues, n_rows, n_cols)
    except ValueError as e:
        print(f"Warning (validation failed): {e}", file=sys.stderr)
        # Print anyway — the user can see what was parsed.

    print(f"row_clues = {row_clues}")
    print(f"col_clues = {col_clues}")


if __name__ == "__main__":
    main()
