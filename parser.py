"""
parser.py — Nonogram image parser.

Usage:
    python parser.py matrix.png ROWS COLS

ROWS, COLS: size of the inner board (not counting the clue areas).

Pipeline (implemented in stages):

  Stage 1 — Geometry
      Load image, detect puzzle bounds, split into three named regions:
          top_clues_area   (→ col_clues)
          left_clues_area  (→ row_clues)
          board_area       (game grid, no clues)

  Stage 2 — Cell extraction
      Split top_clues_area into N_top_rows × COLS individual cells.
      Split left_clues_area into ROWS × N_left_cols individual cells.
      Save every cell as a raw crop.
      Save grid overlay visualizations.
      No recognition yet.

  Stage 3 — Cell cleaning
      For every raw clue-cell from Stage 2:
          trim outer margins     (removes grid-line bleed)
          upscale 4×             (smooth anti-aliasing before threshold)
          suppress border band   (removes residual bleed in upscaled image)
          Otsu binarize          (stable ink / background separation)
          remove small noise     (drops isolated pixels and thin remnants)
          morphological closing  (heals tiny gaps in digit strokes)
      Save: top_cells_clean/, left_cells_clean/

  Stage 4 — Cell recognition
      For every cleaned clue-cell from Stage 3:
          extract foreground bounding box
          detect single vs two-digit via valley detection
          match each digit region against rendered OpenCV templates
          using pixel MSE + weighted projection-profile L2 distance
      Returns None (empty) or one integer per cell — never a list.
      Save: top_cells_recognized/, left_cells_recognized/
            cell_recognition.json

  Stage 5 — Clue-matrix assembly  ← implemented here
      From per-cell results (Stage 4):
          row_clues: left_results → for each row, drop None, keep ints
          col_clues: top_results  → for each col, drop None, keep ints
      Validate lengths and types.
      Save: result.json, final_cells_used.json
      Print: row_clues = [...], col_clues = [...]

Design rule: top_clues_area and left_clues_area are NEVER processed together.
  top_clues_area  → will produce col_clues
  left_clues_area → will produce row_clues

Outputs (debug_output/):
  original.png
  puzzle.png, top.png, left.png, board.png
  puzzle_with_lines.png
  top_grid_overlay.png, left_grid_overlay.png
  top_cells/r00_c00.png ...
  left_cells/r00_c00.png ...
  top_cells_clean/r00_c00.png ...
  left_cells_clean/r00_c00.png ...
  top_cells_recognized/r00_c00.png ...
  left_cells_recognized/r00_c00.png ...
  cell_recognition.json
  final_cells_used.json
  result.json
  metadata.json
"""

import cv2
import numpy as np
import os
import sys
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEBUG_DIR = "debug_output"
DARK_THRESH = 160    # gray < this → dark pixel
LINE_FRAC   = 0.25   # row/col is a grid line if frac of its pixels are dark

# Stage 3 — cell cleaning
CELL_UPSCALE    = 4      # upscale factor applied to the trimmed cell
CELL_TRIM_PX    = 2      # pixels to trim from each side before upscaling;
                         # removes the 1-2 px of grid-line bleed at cell edges
CELL_BORDER_PX  = 3      # upscaled-image border to force white after upscaling;
                         # at 4× this equals ~0.75 px in the original
NOISE_MIN_AREA  = 10     # connected components with fewer pixels are noise
MIN_INK_PIXELS  = 8      # (Stage 4 is_empty_cell) largest component < this → empty
                         # lowered from 30 → avoids rejecting narrow cells
MIN_INK_FRAC    = 0.004  # ink fraction of largest component < this → empty
                         # lowered from 0.015 → same reason

# Stage 4 — recognition
TMPL_CANONICAL_H    = 28    # canonical height to which every digit region is resized
TMPL_CANONICAL_W    = 28    # canonical width
UNCERTAIN_THRESHOLD = 1.00  # best-match score above this → return None (uncertain)
                             # The score formula is: pix_mse/65025 + PROJ_WEIGHT*proj_L2
                             # Even a good font-vs-template match can score 0.30–0.80,
                             # so threshold must be >> typical good-match scores.
                             # 0.20 (original) was too strict: rejected all real digits.
                             # 1.00 effectively trusts is_empty_cell to handle empty cells.
PROJ_WEIGHT         = 2.5   # weight of projection-feature term vs pixel MSE term

# Stage 2 — left clue column merging
MIN_COL_WIDTH_FRAC  = 0.45  # a detected left clue-column narrower than this fraction
                             # of board_cell_w is considered a spurious band from digit
                             # strokes and merged with its neighbour


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path!r}")
    return img


def safe_save_image(path: str, img: np.ndarray) -> bool:
    """
    Write img to path.  Creates parent directories automatically.
    Returns True on success, False on failure (logs a warning).
    """
    if img is None or img.size == 0:
        log_warn(f"skipping save of empty image → {path}")
        return False
    try:
        ensure_dir(os.path.dirname(path) or ".")
        cv2.imwrite(path, img)
        return True
    except Exception as exc:
        log_warn(f"could not save {path}: {exc}")
        return False


def save_debug(name: str, img: np.ndarray) -> None:
    path = os.path.join(DEBUG_DIR, name)
    if safe_save_image(path, img):
        print(f"  saved  {path}")


def crop_bbox(img: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple:
    """
    Parse and validate CLI arguments.

    Expected: python parser.py <image_path> <ROWS> <COLS>

    Returns (img_path, n_rows, n_cols) on success.
    Prints a usage message and exits with code 1 on any error.
    """
    usage = (
        "Usage:  python parser.py <image> <ROWS> <COLS>\n"
        "\n"
        "  image  path to the nonogram puzzle image\n"
        "  ROWS   number of rows in the inner game board  (positive integer)\n"
        "  COLS   number of columns in the inner game board  (positive integer)\n"
        "\n"
        "Example:\n"
        "  python parser.py matrix.png 10 15\n"
    )

    if len(sys.argv) != 4:
        print(f"[ERROR] Expected 3 arguments, got {len(sys.argv) - 1}.\n")
        print(usage)
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.isfile(img_path):
        print(f"[ERROR] Image file not found: {img_path!r}\n")
        print(usage)
        sys.exit(1)

    for name, raw in (("ROWS", sys.argv[2]), ("COLS", sys.argv[3])):
        try:
            val = int(raw)
        except ValueError:
            print(f"[ERROR] {name} must be a positive integer, got: {raw!r}\n")
            print(usage)
            sys.exit(1)
        if val <= 0:
            print(f"[ERROR] {name} must be > 0, got: {val}\n")
            print(usage)
            sys.exit(1)

    return img_path, int(sys.argv[2]), int(sys.argv[3])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_metadata_json(path: str, img_path: str,
                       n_rows: int, n_cols: int,
                       layout: dict,
                       n_top_rows: int, n_left_cols: int) -> None:
    """
    Save a JSON file describing the detected geometry and input parameters.

    All bboxes are stored as [x, y, w, h] for easy consumption by other tools.
    """
    def to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]

    meta = {
        "input_image":    os.path.abspath(img_path),
        "rows":           n_rows,
        "cols":           n_cols,
        "puzzle_bbox":    to_xywh(layout["puzzle_bbox"]),
        "board_bbox":     to_xywh(layout["board_bbox"]),
        "top_bbox":       to_xywh(layout["top_clues_bbox"]),
        "left_bbox":      to_xywh(layout["left_clues_bbox"]),
        "top_clue_rows":  n_top_rows,
        "left_clue_cols": n_left_cols,
        "top_cell_count": n_top_rows * n_cols,
        "left_cell_count": n_rows * n_left_cols,
    }
    try:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  saved  {path}")
    except Exception as exc:
        log_warn(f"could not save metadata: {exc}")


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

def validate_geometry(layout: dict, img_shape: tuple) -> None:
    """
    Check that the detected bboxes are non-empty and lie within the image.
    Logs warnings for any suspicious values; does not raise.
    """
    img_h, img_w = img_shape[:2]
    checks = [
        ("puzzle",     layout["puzzle_bbox"]),
        ("board",      layout["board_bbox"]),
        ("top_clues",  layout["top_clues_bbox"]),
        ("left_clues", layout["left_clues_bbox"]),
    ]
    for name, (x1, y1, x2, y2) in checks:
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            log_warn(f"{name} bbox has zero or negative size: "
                     f"({x1},{y1})→({x2},{y2})")
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            log_warn(f"{name} bbox extends outside image bounds "
                     f"(image {img_w}×{img_h}): ({x1},{y1})→({x2},{y2})")


# ---------------------------------------------------------------------------
# Projection profiles and line-band detection
# ---------------------------------------------------------------------------

def dark_projection(gray: np.ndarray, axis: int) -> np.ndarray:
    """
    Count dark pixels per row (axis=1) or per column (axis=0).
    """
    return (gray < DARK_THRESH).astype(np.int32).sum(axis=axis)


def find_bands(proj: np.ndarray, span: int,
               frac: float = LINE_FRAC) -> list:
    """
    Find contiguous runs where proj > frac * span.

    'span': the perpendicular dimension (width for rows, height for cols).
    Returns list of (start, end) tuples — both inclusive.
    """
    threshold = span * frac
    bands = []
    start = None
    for i, v in enumerate(proj.tolist()):
        if v > threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                bands.append((start, i - 1))
                start = None
    if start is not None:
        bands.append((start, len(proj) - 1))
    return bands


# ---------------------------------------------------------------------------
# Stage 1 — Puzzle layout detection
# ---------------------------------------------------------------------------

def select_board_bands(all_bands: list, n_lines: int) -> list:
    """
    Return the last n_lines bands (board is always at bottom/right).
    """
    if len(all_bands) < n_lines:
        raise ValueError(
            f"Need {n_lines} bands but only {len(all_bands)} detected. "
            "Try reducing DARK_THRESH or LINE_FRAC."
        )
    return all_bands[-n_lines:]


def detect_puzzle_layout(gray: np.ndarray,
                         n_rows: int, n_cols: int) -> dict:
    """
    Detect the full puzzle layout from a grayscale image.

    Returns a dict with keys:
        puzzle_bbox, top_clues_bbox, left_clues_bbox, board_bbox
        h_bands, v_bands
        board_h_bands, board_v_bands
        h_separator, v_separator
        board_cell_h, board_cell_w
    """
    img_h, img_w = gray.shape

    h_proj = dark_projection(gray, axis=1)
    v_proj = dark_projection(gray, axis=0)
    h_bands = find_bands(h_proj, span=img_w)
    v_bands = find_bands(v_proj, span=img_h)

    if not h_bands:
        raise RuntimeError("No horizontal grid lines detected.")
    if not v_bands:
        raise RuntimeError("No vertical grid lines detected.")

    print(f"  detected {len(h_bands)} H-bands, {len(v_bands)} V-bands")

    # Outer puzzle bounds
    puzzle_bbox = (v_bands[0][0], h_bands[0][0],
                   v_bands[-1][1] + 1, h_bands[-1][1] + 1)

    # Board grid lines = last ROWS+1 H-bands and last COLS+1 V-bands
    board_h_bands = select_board_bands(h_bands, n_rows + 1)
    board_v_bands = select_board_bands(v_bands, n_cols + 1)

    # Separators: top/left borders of the board
    h_separator = board_h_bands[0]
    v_separator = board_v_bands[0]

    # Board interior
    board_bbox = (
        board_v_bands[0][1] + 1,
        board_h_bands[0][1] + 1,
        board_v_bands[-1][0],
        board_h_bands[-1][0],
    )

    # top_clues_area: above board, aligned with board columns
    top_clues_bbox = (
        board_v_bands[0][0],
        puzzle_bbox[1],
        board_v_bands[-1][1] + 1,
        h_separator[1] + 1,
    )

    # left_clues_area: left of board, aligned with board rows
    left_clues_bbox = (
        puzzle_bbox[0],
        board_h_bands[0][0],
        v_separator[1] + 1,
        board_h_bands[-1][1] + 1,
    )

    bx1, by1, bx2, by2 = board_bbox
    return {
        "puzzle_bbox":     puzzle_bbox,
        "top_clues_bbox":  top_clues_bbox,
        "left_clues_bbox": left_clues_bbox,
        "board_bbox":      board_bbox,
        "h_bands":         h_bands,
        "v_bands":         v_bands,
        "board_h_bands":   board_h_bands,
        "board_v_bands":   board_v_bands,
        "h_separator":     h_separator,
        "v_separator":     v_separator,
        "board_cell_h":    (by2 - by1) / n_rows,
        "board_cell_w":    (bx2 - bx1) / n_cols,
    }


# ---------------------------------------------------------------------------
# Stage 1 — Named region crops
# ---------------------------------------------------------------------------

def crop_top_clues_area(img: np.ndarray, layout: dict) -> np.ndarray:
    """
    top_clues_area: above the board, aligned with board columns.
    Contains column clues — will produce col_clues.
    """
    return crop_bbox(img, layout["top_clues_bbox"])


def crop_left_clues_area(img: np.ndarray, layout: dict) -> np.ndarray:
    """
    left_clues_area: left of the board, aligned with board rows.
    Contains row clues — will produce row_clues.
    """
    return crop_bbox(img, layout["left_clues_bbox"])


def crop_board_area(img: np.ndarray, layout: dict) -> np.ndarray:
    """
    board_area: the inner game grid.
    Contains no clue numbers.
    """
    return crop_bbox(img, layout["board_bbox"])


def split_puzzle_regions(img: np.ndarray, layout: dict) -> dict:
    """
    Split the full puzzle image into its three strictly separate regions.

    Returns {top_clues_area, left_clues_area, board_area}.
    These are kept separate throughout the pipeline.
    """
    return {
        "top_clues_area":  crop_top_clues_area(img, layout),
        "left_clues_area": crop_left_clues_area(img, layout),
        "board_area":      crop_board_area(img, layout),
    }


# ---------------------------------------------------------------------------
# Stage 1 — Visualization overlay
# ---------------------------------------------------------------------------

def draw_puzzle_overlay(img: np.ndarray, layout: dict) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]

    for a, b in layout["h_bands"]:
        cv2.line(vis, (0, (a+b)//2), (w-1, (a+b)//2), (60, 160, 220), 1)
    for a, b in layout["v_bands"]:
        cv2.line(vis, ((a+b)//2, 0), ((a+b)//2, h-1), (60, 160, 220), 1)
    for a, b in layout["board_h_bands"]:
        cv2.line(vis, (0, (a+b)//2), (w-1, (a+b)//2), (0, 210, 0), 2)
    for a, b in layout["board_v_bands"]:
        cv2.line(vis, ((a+b)//2, 0), ((a+b)//2, h-1), (0, 210, 0), 2)

    hy = (layout["h_separator"][0] + layout["h_separator"][1]) // 2
    cv2.line(vis, (0, hy), (w-1, hy), (0, 0, 230), 3)
    vx = (layout["v_separator"][0] + layout["v_separator"][1]) // 2
    cv2.line(vis, (vx, 0), (vx, h-1), (0, 0, 230), 3)

    def box(bbox, color):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2-1, y2-1), color, 2)

    box(layout["puzzle_bbox"],     (0,   230, 230))
    box(layout["board_bbox"],      (210, 0,   210))
    box(layout["top_clues_bbox"],  (0,   130, 255))
    box(layout["left_clues_bbox"], (200, 100, 0  ))

    for bbox, label in [
        (layout["top_clues_bbox"],  "top_clues_area"),
        (layout["left_clues_bbox"], "left_clues_area"),
        (layout["board_bbox"],      "board_area"),
    ]:
        cv2.putText(vis, label, (bbox[0]+4, bbox[1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, label, (bbox[0]+4, bbox[1]+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return vis


# ===========================================================================
# Stage 2 — Clue-cell extraction
# ===========================================================================

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def translate_bands(bands: list, offset: int) -> list:
    """Shift band coordinates into a cropped image's local frame."""
    return [(a - offset, b - offset) for a, b in bands]


# ---------------------------------------------------------------------------
# 2a. Detect the unknown clue dimensions from within each clue area
# ---------------------------------------------------------------------------

def detect_top_row_bands(top_clues_area: np.ndarray) -> list:
    """
    Detect horizontal line bands inside top_clues_area.

    These bands define the boundaries of the clue ROWS (the unknown
    dimension N_top_rows).  The number of clue rows = len(bands) - 1.

    All detected lines span the full width of top_clues_area, so the
    standard LINE_FRAC threshold applies without modification.
    """
    gray = to_gray(top_clues_area)
    h, w = gray.shape
    proj  = dark_projection(gray, axis=1)        # count per row
    bands = find_bands(proj, span=w)
    if len(bands) < 2:
        raise RuntimeError(
            f"top_clues_area: only {len(bands)} H-band(s) detected "
            f"(need ≥ 2).  Check debug_output/top.png."
        )
    return bands


def _merge_narrow_col_bands(bands: list, min_gap_px: int) -> list:
    """
    Merge consecutive line bands when the cell gap between them is < min_gap_px.

    Over-segmentation of the left clue area happens when digit strokes project
    as spurious vertical bands.  A gap narrower than ~half the board cell width
    is almost certainly a digit stroke, not a real grid-line separator.

    Algorithm:
        Walk the band list left→right.  If the gap between the end of the
        last accepted band and the start of the current band is < min_gap_px,
        absorb the current band into the last accepted one (extend its right
        edge).  Otherwise accept it as a new band.
    """
    if len(bands) < 3:   # need ≥3 bands (≥2 cells) to be worth merging
        return bands

    result = [bands[0]]
    for band in bands[1:]:
        gap = band[0] - result[-1][1] - 1   # pixels between prev end and curr start
        if gap < min_gap_px:
            # Merge: extend the last accepted band rightward
            result[-1] = (result[-1][0], band[1])
        else:
            result.append(band)
    return result


def detect_left_col_bands(left_clues_area: np.ndarray) -> list:
    """
    Detect vertical line bands inside left_clues_area.

    These bands define the boundaries of the clue COLUMNS (the unknown
    dimension N_left_cols).  The number of clue cols = len(bands) - 1.
    """
    gray = to_gray(left_clues_area)
    h, w = gray.shape
    proj  = dark_projection(gray, axis=0)        # count per col
    bands = find_bands(proj, span=h)
    if len(bands) < 2:
        raise RuntimeError(
            f"left_clues_area: only {len(bands)} V-band(s) detected "
            f"(need ≥ 2).  Check debug_output/left.png."
        )
    return bands


# ---------------------------------------------------------------------------
# 2b. Core cell extraction
# ---------------------------------------------------------------------------

def extract_cells(area: np.ndarray,
                  row_bands: list,
                  col_bands: list) -> list:
    """
    Extract individual cell crops from a clue area.

    Cell (r, c) is the interior between:
        row_bands[r]   (top border band)   and  row_bands[r+1]  (bottom)
        col_bands[c]   (left border band)  and  col_bands[c+1]  (right)

    Returns cells[row][col] = np.ndarray crop, or None if bbox is empty.

    Empty cells (no digits) are still extracted — they remain as valid
    slots whose content is blank.  They are NOT skipped.
    """
    n_rows = len(row_bands) - 1
    n_cols = len(col_bands) - 1

    cells = []
    for r in range(n_rows):
        row_cells = []
        y1 = row_bands[r][1]     + 1   # one pixel below top band
        y2 = row_bands[r + 1][0]       # one pixel above bottom band
        for c in range(n_cols):
            x1 = col_bands[c][1]     + 1
            x2 = col_bands[c + 1][0]
            if y2 > y1 and x2 > x1:
                row_cells.append(area[y1:y2, x1:x2].copy())
            else:
                row_cells.append(None)
        cells.append(row_cells)

    return cells


# ---------------------------------------------------------------------------
# 2c. High-level splitters for each clue area
# ---------------------------------------------------------------------------

def split_top_clue_area(top_clues_area: np.ndarray,
                        layout: dict) -> tuple:
    """
    Split top_clues_area into a (N_top_rows × COLS) grid of cells.

    Column boundaries come from board_v_bands (guarantees exact alignment
    with board columns — the known dimension).

    Row boundaries are detected from horizontal lines inside top_clues_area
    (the unknown dimension N_top_rows).

    Returns
    -------
    cells          : list[list[np.ndarray | None]]  shape [N_top_rows][COLS]
    h_bands_local  : detected horizontal bands in top_clues_area coordinates
    col_bands_local: board_v_bands translated to top_clues_area coordinates
    """
    top_x1 = layout["top_clues_bbox"][0]    # x-origin of top_clues_area

    # Column bands: translate board_v_bands into top_clues_area local coords.
    # This guarantees alignment with board columns.
    col_bands_local = translate_bands(layout["board_v_bands"], top_x1)

    # Row bands: detected from the image (N_top_rows is unknown).
    h_bands_local = detect_top_row_bands(top_clues_area)

    cells = extract_cells(top_clues_area, h_bands_local, col_bands_local)

    n_top_rows = len(cells)
    n_cols     = len(cells[0]) if cells else 0
    print(f"  top_clues_area  → {n_top_rows} clue rows × {n_cols} cols"
          f"  =  {n_top_rows * n_cols} cells")

    return cells, h_bands_local, col_bands_local


def split_left_clue_area(left_clues_area: np.ndarray,
                         layout: dict) -> tuple:
    """
    Split left_clues_area into a (ROWS × N_left_cols) grid of cells.

    Row boundaries come from board_h_bands (guarantees exact alignment
    with board rows — the known dimension).

    Column boundaries are detected from vertical lines inside left_clues_area
    (the unknown dimension N_left_cols).

    Returns
    -------
    cells          : list[list[np.ndarray | None]]  shape [ROWS][N_left_cols]
    row_bands_local: board_h_bands translated to left_clues_area coordinates
    v_bands_local  : detected vertical bands in left_clues_area coordinates
    """
    left_y1 = layout["left_clues_bbox"][1]   # y-origin of left_clues_area

    # Row bands: translate board_h_bands into left_clues_area local coords.
    # This guarantees alignment with board rows.
    row_bands_local = translate_bands(layout["board_h_bands"], left_y1)

    # Column bands: detected from the image (N_left_cols is unknown).
    v_bands_local = detect_left_col_bands(left_clues_area)

    # Merge bands that create cells narrower than MIN_COL_WIDTH_FRAC * board_cell_w.
    # Digit strokes in the horizontal projection can produce spurious narrow bands,
    # causing the left area to appear to have far more clue columns than it actually has.
    min_col_px = max(int(layout["board_cell_w"] * MIN_COL_WIDTH_FRAC), 4)
    n_before   = len(v_bands_local) - 1
    v_bands_local = _merge_narrow_col_bands(v_bands_local, min_col_px)
    n_after    = len(v_bands_local) - 1
    if n_after != n_before:
        log_warn(f"left_clues_area: merged {n_before} → {n_after} clue cols "
                 f"(min cell width threshold: {min_col_px} px  = "
                 f"{MIN_COL_WIDTH_FRAC:.0%} × board_cell_w {layout['board_cell_w']:.1f})")

    # Log detected column widths for inspection
    col_gaps = [v_bands_local[i+1][0] - v_bands_local[i][1] - 1
                for i in range(len(v_bands_local) - 1)]
    print(f"  left clue col widths (px): {col_gaps}")

    cells = extract_cells(left_clues_area, row_bands_local, v_bands_local)

    n_rows      = len(cells)
    n_left_cols = len(cells[0]) if cells else 0
    print(f"  left_clues_area → {n_rows} rows × {n_left_cols} clue cols"
          f"  =  {n_rows * n_left_cols} cells")

    return cells, row_bands_local, v_bands_local


# ---------------------------------------------------------------------------
# 2d. Save cell crops
# ---------------------------------------------------------------------------

def save_cell_crops(cells: list, out_dir: str) -> int:
    """
    Save every cell crop to out_dir/r{row}_c{col}.png.

    Empty cells (None or zero-size) are skipped; their absence in the
    directory signals "no crop possible" and they will be treated as
    empty slots during recognition.

    Returns the number of files written.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            if cell is not None and cell.size > 0:
                fname = f"r{r:02d}_c{c:02d}.png"
                cv2.imwrite(os.path.join(out_dir, fname), cell)
                saved += 1
    print(f"  saved {saved} cell crops → {out_dir}/")
    return saved


# ---------------------------------------------------------------------------
# 2e. Grid overlay visualization
# ---------------------------------------------------------------------------

def draw_grid_overlay(area: np.ndarray,
                      row_bands: list,
                      col_bands: list,
                      label: str = "") -> np.ndarray:
    """
    Draw cell boundary lines on a copy of area_img.

    row_bands and col_bands must be in area's local coordinate system.

    Color legend:
        red      — row boundary bands (horizontal lines)
        green    — col boundary bands (vertical lines)
        cyan     — cell interior rectangles
    """
    vis = area.copy()
    h, w = vis.shape[:2]

    # Draw each boundary band as two lines (its start and end pixel)
    for a, b in row_bands:
        cv2.line(vis, (0, a),   (w-1, a),   (40,  40, 220), 1)
        cv2.line(vis, (0, b),   (w-1, b),   (40,  40, 220), 1)

    for a, b in col_bands:
        cv2.line(vis, (a, 0),   (a, h-1),   (40, 200,  40), 1)
        cv2.line(vis, (b, 0),   (b, h-1),   (40, 200,  40), 1)

    # Draw cell interior rectangles
    for r in range(len(row_bands) - 1):
        y1 = row_bands[r][1]     + 1
        y2 = row_bands[r + 1][0]
        for c in range(len(col_bands) - 1):
            x1 = col_bands[c][1]     + 1
            x2 = col_bands[c + 1][0]
            if x2 > x1 and y2 > y1:
                cv2.rectangle(vis, (x1, y1), (x2-1, y2-1), (200, 200, 0), 1)

    if label:
        # White shadow + colored text for readability over any background
        cv2.putText(vis, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2)
        cv2.putText(vis, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 180), 1)

    return vis


# ===========================================================================
# Stage 3 — Per-cell cleaning
# ===========================================================================

# ---------------------------------------------------------------------------
# 3a. Individual cleaning steps
# ---------------------------------------------------------------------------

def trim_cell_margins(gray: np.ndarray,
                      trim_px: int = CELL_TRIM_PX) -> np.ndarray:
    """
    Remove trim_px pixels from every side of the grayscale cell.

    Purpose: the raw crop starts 1 px after the band boundary, but grid
    lines can bleed 1-2 px inward.  A 2 px trim removes this without
    cutting into digit strokes (the narrowest stroke in the font is ~3 px).
    """
    h, w = gray.shape
    if h <= 2 * trim_px or w <= 2 * trim_px:
        return gray   # cell too small — skip trim
    return gray[trim_px : h - trim_px, trim_px : w - trim_px]


def remove_cell_borders(upscaled: np.ndarray,
                        border_px: int = CELL_BORDER_PX) -> np.ndarray:
    """
    Force the outermost border_px pixels of the UPSCALED image to white.

    Any residual bleed that survived the initial trim is clamped to
    background here.  At CELL_UPSCALE=4, border_px=3 is ~0.75 px in
    the original — too small to touch digit strokes.
    """
    result = upscaled.copy()
    h, w   = result.shape
    if h <= 2 * border_px or w <= 2 * border_px:
        return result
    result[:border_px,  :]  = 255
    result[h-border_px:, :] = 255
    result[:,  :border_px]  = 255
    result[:, w-border_px:] = 255
    return result


def binarize_cell(gray: np.ndarray) -> np.ndarray:
    """
    Otsu binarization: returns binary image where 255 = ink, 0 = background.

    For nearly-uniform (empty) images the standard deviation is very low;
    forcing the result to all-zero prevents noise pixels from being called ink.

    Why Otsu?  The nonogram font produces a clear bimodal histogram in each
    cell: one cluster for the white background (~230-255) and one for the
    dark digit strokes (~20-100).  Otsu reliably finds the valley between
    them regardless of the exact ink color or background shade.
    """
    if gray.size == 0:
        return np.zeros_like(gray)
    if float(gray.std()) < 6.0:
        # Nearly uniform → no digit; avoid thresholding noise
        return np.zeros_like(gray)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def remove_small_noise(binary: np.ndarray,
                       min_area: int = NOISE_MIN_AREA) -> np.ndarray:
    """
    Drop connected components smaller than min_area pixels.

    After binarization, isolated pixels and thin grid-remnant lines appear
    as tiny components.  Real digit strokes at 4× upscale are 100+ pixels.
    Removing components < 10 px eliminates noise without touching any digit.
    """
    if binary is None or binary.size == 0 or not binary.any():
        return binary if binary is not None else np.zeros((1, 1), np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary,
                                                           connectivity=8)
    out = np.zeros_like(binary)
    for lbl in range(1, n):                     # 0 = background
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 255
    return out


def is_nearly_empty(binary: np.ndarray) -> bool:
    """
    Return True if the binary image is too sparse to contain a digit.

    Two independent checks — either is sufficient to declare the cell empty:
      • absolute ink count  < MIN_INK_PIXELS  (catches very small cells)
      • ink fraction        < MIN_INK_FRAC    (catches proportionally sparse)

    These thresholds apply to the UPSCALED binary, so MIN_INK_PIXELS=30 at
    CELL_UPSCALE=4 corresponds to ~2 px in the original — safely below any
    real digit but above sensor noise.
    """
    if binary is None or binary.size == 0:
        return True
    ink   = int((binary > 0).sum())
    total = int(binary.size)
    return ink < MIN_INK_PIXELS or ink < total * MIN_INK_FRAC


# ---------------------------------------------------------------------------
# 3b. Full per-cell pipeline
# ---------------------------------------------------------------------------

def clean_clue_cell(cell_bgr: np.ndarray) -> tuple:
    """
    Clean one raw clue-cell crop.  Returns (cleaned_bgr, is_empty).

    Steps
    -----
    1  Grayscale        — luminance-weighted, handles colored digit fonts.
    2  Trim margins     — removes grid-line bleed at cell edges.
    3  Upscale 4×       — LANCZOS gives smooth sub-pixel edges; upscaling
                          BEFORE binarising lets Otsu see anti-aliased
                          gradients rather than blocky pixels.
    4  Suppress border  — forces outermost upscaled pixels to white, removes
                          any bleed that survived step 2.
    5  Otsu binarize    — separates ink from background per-cell adaptively.
    6  Remove noise     — drops tiny components (< NOISE_MIN_AREA px).
    7  Morphological    — 2×2 closing heals single-pixel gaps in digit strokes
       closing            without bridging the gap between '1' and '0' in "10"
                          (gap at 4× upscale is 12–20 px; kernel is only 2 px).

    Why these steps target the known failure modes
    -----------------------------------------------
    8→3, 5→3   Caused by grid-line bleed making the LEFT stroke of the digit
                look like it isn't there.  Steps 2+4 remove bleed; step 5
                uses Otsu so gray border pixels go to background, not ink.
    2→1        Curved top of "2" blends with a border line.  Same fix.
    10 damage  Multi-digit "10" is most fragile to noise; step 7 uses a
                minimal kernel so "1" and "0" stay separate.
    Empty→fake  Steps 5+6 together: low-std guard in binarize_cell prevents
    digit       Otsu from over-thresholding a blank cell, and noise removal
                drops any stray pixels that pass the guard.

    Returns
    -------
    cleaned_bgr : BGR image, white background, black strokes, 4× upscaled.
                  None if the input was invalid or too small to process.
    is_empty    : True if the cell contains no recognizable digit.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return None, True

    # 1. Grayscale
    gray = (cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
            if len(cell_bgr.shape) == 3 else cell_bgr.copy())

    # 2. Trim margins
    trimmed = trim_cell_margins(gray, CELL_TRIM_PX)
    if trimmed.size == 0:
        return None, True

    # 3. Upscale with LANCZOS
    th, tw  = trimmed.shape
    upscaled = cv2.resize(trimmed, (tw * CELL_UPSCALE, th * CELL_UPSCALE),
                          interpolation=cv2.INTER_LANCZOS4)

    # 4. Suppress border band in the upscaled image
    no_border = remove_cell_borders(upscaled, CELL_BORDER_PX)

    # 5. Otsu binarization  (ink = 255, background = 0)
    binary = binarize_cell(no_border)

    # 6. Remove small noise components
    denoised = remove_small_noise(binary, NOISE_MIN_AREA)

    # 7. Morphological closing — heal tiny gaps without merging digit parts
    if denoised.any():
        kernel   = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel,
                                    iterations=1)

    empty = is_nearly_empty(denoised)

    # Convert to displayable BGR: invert so ink=black on white background
    cleaned_bgr = cv2.cvtColor(255 - denoised, cv2.COLOR_GRAY2BGR)
    return cleaned_bgr, empty


# ---------------------------------------------------------------------------
# 3c. Batch processor
# ---------------------------------------------------------------------------

def clean_cells_batch(cells: list, clean_dir: str) -> tuple:
    """
    Run clean_clue_cell on every cell in the grid and save results.

    Naming is identical to the raw crops so the two directories can be
    compared side-by-side:  top_cells/r02_c05.png  ←→  top_cells_clean/r02_c05.png

    Returns
    -------
    stats_dict   : {total_cleaned, empty_count, sizes}
    cleaned_grid : 2-D list parallel to `cells`; each entry is
                   (cleaned_bgr, is_empty) — cleaned_bgr may be None.
    """
    os.makedirs(clean_dir, exist_ok=True)
    total        = 0
    empty        = 0
    sizes        = []
    cleaned_grid = []

    for r, row in enumerate(cells):
        cleaned_row = []
        for c, cell in enumerate(row):
            cleaned, is_empty = clean_clue_cell(cell)
            cleaned_row.append((cleaned, is_empty))
            if cleaned is not None:
                fname = f"r{r:02d}_c{c:02d}.png"
                cv2.imwrite(os.path.join(clean_dir, fname), cleaned)
                total += 1
                sizes.append(cleaned.shape[:2])   # (height, width)
                if is_empty:
                    empty += 1
        cleaned_grid.append(cleaned_row)

    print(f"  {total} cells cleaned  ({empty} empty)  → {clean_dir}/")
    return {"total_cleaned": total, "empty_count": empty, "sizes": sizes}, cleaned_grid


# ===========================================================================
# Stage 4 — Per-cell recognition
# ===========================================================================

# ---------------------------------------------------------------------------
# 4a. Template construction
# ---------------------------------------------------------------------------

def build_digit_templates() -> tuple:
    """
    Render digits 0-9 using cv2.FONT_HERSHEY_SIMPLEX, tight-crop each one,
    resize to (TMPL_CANONICAL_W × TMPL_CANONICAL_H), and precompute their
    projection-profile feature vectors.

    Returns (templates, tmpl_features) where both are dicts keyed 0..9.
    templates[d]     : uint8 array shape (H, W), values 0=background 255=ink
    tmpl_features[d] : float64 array shape (20,) normalised H+V projections
    """
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 3.0
    thick = 4

    templates    = {}
    tmpl_features = {}

    for d in range(10):
        text = str(d)
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        canvas_h = th + baseline + 20
        canvas_w = tw + 20
        canvas   = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
        cv2.putText(canvas, text, (10, th + 10),
                    font, scale, 0, thick, cv2.LINE_AA)

        # Tight-crop to the inked region
        fg   = (canvas < 128)
        rows = np.where(fg.any(axis=1))[0]
        cols = np.where(fg.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            # Fallback: use full canvas
            tight_255 = fg.astype(np.uint8) * 255
        else:
            tight_255 = fg[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1].astype(np.uint8) * 255

        resized = cv2.resize(tight_255, (TMPL_CANONICAL_W, TMPL_CANONICAL_H),
                             interpolation=cv2.INTER_AREA)
        templates[d]     = resized
        tmpl_features[d] = compute_projection_features(resized)

    return templates, tmpl_features


_digit_templates_cache = None


def get_digit_templates() -> tuple:
    """Return cached (templates, tmpl_features); build on first call."""
    global _digit_templates_cache
    if _digit_templates_cache is None:
        _digit_templates_cache = build_digit_templates()
    return _digit_templates_cache


# ---------------------------------------------------------------------------
# 4b. Projection-profile feature vector
# ---------------------------------------------------------------------------

def compute_projection_features(binary: np.ndarray,
                                n_bins: int = 10) -> np.ndarray:
    """
    Compute a 20-element normalised projection-profile feature vector.

    Steps:
        1. H-projection (sum per row)   → resample to n_bins values
        2. V-projection (sum per col)   → resample to n_bins values
        3. Concatenate → 20 values, normalise entire vector to [0, 1].

    This captures the global shape of the ink distribution independently
    of exact pixel alignment.  It strongly discriminates digit pairs that
    look similar pixel-by-pixel (e.g. 8 vs 3, 2 vs 1, 5 vs 3).
    """
    h_proj = binary.sum(axis=1).astype(float)
    v_proj = binary.sum(axis=0).astype(float)

    def resample(proj: np.ndarray) -> np.ndarray:
        if proj.size == 0:
            return np.zeros(n_bins)
        xs   = np.linspace(0, proj.size - 1, n_bins)
        return np.interp(xs, np.arange(proj.size), proj)

    feat = np.concatenate([resample(h_proj), resample(v_proj)])
    mx   = feat.max()
    if mx > 0:
        feat /= mx
    return feat


# ---------------------------------------------------------------------------
# 4c. Foreground helpers
# ---------------------------------------------------------------------------

def cell_to_fg(cleaned_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a cleaned BGR cell (white bg, black strokes) to a binary mask
    where 255 = ink, 0 = background.
    """
    if cleaned_bgr is None or cleaned_bgr.size == 0:
        return None
    gray = (cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
            if len(cleaned_bgr.shape) == 3 else cleaned_bgr.copy())
    return (gray < 128).astype(np.uint8) * 255


def extract_foreground_bbox(fg: np.ndarray):
    """
    Return (rmin, cmin, rmax_excl, cmax_excl) tight bounding box of all ink,
    or None if there is no ink.
    """
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


# ---------------------------------------------------------------------------
# 4d. Two-digit splitting
# ---------------------------------------------------------------------------

def split_digit_regions(fg_crop: np.ndarray) -> list:
    """
    Decide if fg_crop contains one or two horizontally adjacent digits and
    return the column ranges.

    Strategy: look for a valley in the column-wise ink histogram in the
    middle half of the image.  Split only when:
        valley_value < peak * 0.30   (clear gap between the two digits)
        left width  >= 8 px
        right width >= 8 px

    Why "0" and "8" don't false-split: their inner hollow produces a local
    minimum around 40% of peak — safely above the 30% threshold.

    Returns list of (x_start, x_end) pairs (end is exclusive).
    One pair  → single digit.
    Two pairs → two digits; value = d0 * 10 + d1.
    """
    h, w = fg_crop.shape
    if w < 16:
        return [(0, w)]

    col_hist = fg_crop.sum(axis=0).astype(float)
    peak = col_hist.max()
    if peak == 0:
        return [(0, w)]

    # Inspect only the middle half to avoid edge effects
    mid_start = w // 4
    mid_end   = 3 * w // 4
    mid_hist  = col_hist[mid_start:mid_end]

    valley_local = int(np.argmin(mid_hist))
    valley_idx   = mid_start + valley_local
    valley_val   = col_hist[valley_idx]

    if valley_val >= peak * 0.30:
        return [(0, w)]

    left_w  = valley_idx
    right_w = w - valley_idx
    if left_w < 8 or right_w < 8:
        return [(0, w)]

    return [(0, valley_idx), (valley_idx, w)]


# ---------------------------------------------------------------------------
# 4e. Single-digit matching
# ---------------------------------------------------------------------------

def match_digit(component_fg: np.ndarray,
                templates: dict,
                tmpl_features: dict) -> tuple:
    """
    Match a single-digit foreground crop against all templates.

    Score = pixel_MSE / 65025  +  PROJ_WEIGHT * projection_L2

    The pixel term penalises gross shape differences; the projection term
    breaks ties between visually similar digits (e.g. 8 vs 3).

    Returns (best_digit, best_score).
    """
    resized  = cv2.resize(component_fg,
                          (TMPL_CANONICAL_W, TMPL_CANONICAL_H),
                          interpolation=cv2.INTER_AREA)
    cell_feat = compute_projection_features(resized)

    best_digit = 0
    best_score = float("inf")

    for d, tmpl in templates.items():
        diff      = resized.astype(float) - tmpl.astype(float)
        pix_mse   = float((diff ** 2).mean()) / 65025.0
        proj_l2   = float(np.linalg.norm(cell_feat - tmpl_features[d]))
        score     = pix_mse + PROJ_WEIGHT * proj_l2
        if score < best_score:
            best_score = score
            best_digit = d

    return best_digit, best_score


# ---------------------------------------------------------------------------
# 4f. Empty-cell guard
# ---------------------------------------------------------------------------

def is_empty_cell(cell_img: np.ndarray) -> bool:
    """
    Determine whether a cleaned clue-cell image contains no digit.

    Works on the cleaned BGR cell (white background, black strokes).

    Strategy: connected-component analysis on the foreground mask.
    Uses the LARGEST component area rather than raw ink count, so isolated
    noise pixels do not falsely flag an empty cell as non-empty, and a
    real digit stroke is not falsely discarded as empty.

    Decision (empty = True) when:
        largest_component_area < MIN_INK_PIXELS    (too few absolute pixels)
        OR
        largest_component_area < cell_area * MIN_INK_FRAC  (too sparse)

    If no foreground component exists at all the cell is empty by definition.

    Returns True  → empty slot   → recognize_cell returns None.
    Returns False → digit present → proceed with recognition.
    """
    if cell_img is None or cell_img.size == 0:
        return True
    fg = cell_to_fg(cell_img)
    if fg is None or not fg.any():
        return True

    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n < 2:   # only the background label (0)
        return True

    max_area  = int(max(stats[i, cv2.CC_STAT_AREA] for i in range(1, n)))
    cell_area = int(fg.size)
    return max_area < MIN_INK_PIXELS or max_area < cell_area * MIN_INK_FRAC


# ---------------------------------------------------------------------------
# 4g. Full per-cell recogniser
# ---------------------------------------------------------------------------

def recognize_cell(cell_img: np.ndarray):
    """
    Recognise a single already-cleaned clue-cell image.

    Input
    -----
    cell_img : cleaned BGR image (white background, black strokes, 4× upscaled)
               as produced by clean_clue_cell().

    Returns
    -------
    None  — cell is empty or confidence is too low
    int   — recognised integer (0–9 for single digit, 10–99 for two digits)

    Guarantees:
        • never returns a list
        • one cell → one value or None
        • '1'+'0' → 10, not [1, 0]

    Pipeline
    --------
    1. Reject trivially empty cells via is_empty_cell().
    2. Extract the tight foreground bounding box.
    3. Detect single- vs two-digit layout via column-histogram valley.
    4. Match each digit region against rendered OpenCV templates using
       pixel MSE + projection-profile L2 distance.
    5. Combine two recognised digits into one integer (d0*10 + d1).
    6. Reject if the best match score exceeds UNCERTAIN_THRESHOLD.
    """
    if cell_img is None or cell_img.size == 0:
        return None
    if is_empty_cell(cell_img):
        return None

    fg   = cell_to_fg(cell_img)
    bbox = extract_foreground_bbox(fg)
    if bbox is None:
        return None

    rmin, cmin, rmax, cmax = bbox
    fg_crop = fg[rmin:rmax, cmin:cmax]
    if fg_crop.size == 0:
        return None

    templates, tmpl_features = get_digit_templates()
    regions = split_digit_regions(fg_crop)

    if len(regions) == 1:
        digit, score = match_digit(fg_crop, templates, tmpl_features)
        if score > UNCERTAIN_THRESHOLD:
            return None
        return digit

    # Two-digit: d0 = left region, d1 = right region → d0*10 + d1
    digits = []
    for x1, x2 in regions:
        region = fg_crop[:, x1:x2]
        if region.size == 0:
            return None
        d, _ = match_digit(region, templates, tmpl_features)
        digits.append(d)
    return digits[0] * 10 + digits[1]


# ---------------------------------------------------------------------------
# 4h. Annotation helper
# ---------------------------------------------------------------------------

def annotate_recognition(cell_img: np.ndarray, value) -> np.ndarray:
    """
    Draw the recognised value on a copy of cell_img.

    Green text  → digit recognised.
    Gray  text  → None / empty.
    White shadow for readability over any cell background.
    """
    if cell_img is not None and cell_img.size > 0:
        vis = cell_img.copy()
    else:
        vis = np.full((TMPL_CANONICAL_H * 2, TMPL_CANONICAL_W * 2, 3),
                      200, dtype=np.uint8)

    label = str(value) if value is not None else "—"
    color = (0, 160, 0) if value is not None else (160, 160, 160)
    y     = max(vis.shape[0] - 4, 10)

    cv2.putText(vis, label, (2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    cv2.putText(vis, label, (2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


# ---------------------------------------------------------------------------
# 4h. Per-cell recognition with diagnostics
# ---------------------------------------------------------------------------

def _recognize_and_diagnose(cell_img: np.ndarray) -> tuple:
    """
    Run recognition on one cell and collect diagnostic data for debug JSON.

    Returns (value, diag) where:
        value : None | int
        diag  : dict with keys fg_pixels, max_component_area, fg_bbox,
                component_count, empty_decision, match_score, recognized_value
    """
    diag = {
        "fg_pixels":         0,
        "max_component_area": 0,
        "fg_bbox":           None,
        "component_count":   0,
        "empty_decision":    True,
        "match_score":       None,
        "recognized_value":  None,
    }

    if cell_img is None or cell_img.size == 0:
        return None, diag

    fg = cell_to_fg(cell_img)
    if fg is None or not fg.any():
        return None, diag

    # Foreground pixel count and component stats
    diag["fg_pixels"] = int((fg > 0).sum())

    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    n_comp = n - 1
    diag["component_count"] = n_comp
    if n_comp > 0:
        diag["max_component_area"] = int(
            max(stats[i, cv2.CC_STAT_AREA] for i in range(1, n)))

    bbox = extract_foreground_bbox(fg)
    if bbox is not None:
        rmin, cmin, rmax, cmax = bbox
        diag["fg_bbox"] = [cmin, rmin, cmax - cmin, rmax - rmin]  # [x,y,w,h]

    # Empty decision
    if is_empty_cell(cell_img):
        diag["empty_decision"] = True
        return None, diag

    diag["empty_decision"] = False

    # Recognition
    if bbox is None:
        return None, diag

    rmin, cmin, rmax, cmax = bbox
    fg_crop = fg[rmin:rmax, cmin:cmax]
    if fg_crop.size == 0:
        return None, diag

    templates, tmpl_features = get_digit_templates()
    regions = split_digit_regions(fg_crop)

    if len(regions) == 1:
        digit, score = match_digit(fg_crop, templates, tmpl_features)
        diag["match_score"] = round(float(score), 4)
        if score > UNCERTAIN_THRESHOLD:
            return None, diag
        diag["recognized_value"] = digit
        return digit, diag

    # Two-digit
    digits = []
    scores = []
    for x1, x2 in regions:
        region = fg_crop[:, x1:x2]
        if region.size == 0:
            return None, diag
        d, s = match_digit(region, templates, tmpl_features)
        digits.append(d)
        scores.append(s)
    diag["match_score"] = round(float(sum(scores) / len(scores)), 4)
    value = digits[0] * 10 + digits[1]
    diag["recognized_value"] = value
    return value, diag


# ---------------------------------------------------------------------------
# 4j. Batch recogniser
# ---------------------------------------------------------------------------

def recognize_cells_batch(cleaned_grid: list, out_dir: str,
                           area_label: str = "") -> tuple:
    """
    Run recognition on every cell and save annotated images + diagnostics JSON.

    cleaned_grid : 2-D list of (cleaned_bgr, is_empty) tuples
    out_dir      : directory for annotated debug images
    area_label   : "top" or "left" — used in diagnostics JSON

    Returns
    -------
    results     : 2-D list of None | int, same shape as cleaned_grid
    stats       : dict {recognized, total, empty, two_digit}
    """
    os.makedirs(out_dir, exist_ok=True)
    results      = []
    two_digit    = 0
    cell_records = []   # for per-cell diagnostics JSON

    for r, row in enumerate(cleaned_grid):
        result_row = []
        for c, (cleaned, _) in enumerate(row):
            value, diag = _recognize_and_diagnose(cleaned)
            result_row.append(value)
            if value is not None and value >= 10:
                two_digit += 1

            # Annotated debug image
            vis   = annotate_recognition(cleaned, value)
            fname = f"r{r:02d}_c{c:02d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), vis)

            # Optionally save tight fg crop for inspection
            fg = cell_to_fg(cleaned) if cleaned is not None else None
            if fg is not None and diag["fg_bbox"] is not None:
                cx, cy, cw, ch = diag["fg_bbox"]
                fg_crop_bgr = cv2.cvtColor(
                    255 - fg[cy:cy+ch, cx:cx+cw], cv2.COLOR_GRAY2BGR)
                safe_save_image(
                    os.path.join(out_dir, f"r{r:02d}_c{c:02d}_fg.png"),
                    fg_crop_bgr)

            record = {"area": area_label, "row": r, "col": c}
            record.update(diag)
            cell_records.append(record)

        results.append(result_row)

    # Save per-cell diagnostics JSON alongside the annotated images
    diag_path = os.path.join(out_dir, "diagnostics.json")
    try:
        with open(diag_path, "w") as fh:
            json.dump(cell_records, fh, indent=2)
    except Exception as exc:
        log_warn(f"Could not save {diag_path}: {exc}")

    recognized = sum(1 for row in results for v in row if v is not None)
    total      = sum(len(row) for row in results)
    empty      = total - recognized
    stats = {"recognized": recognized, "total": total,
             "empty": empty, "two_digit": two_digit}
    print(f"  {recognized}/{total} recognized  "
          f"({empty} empty, {two_digit} two-digit)  → {out_dir}/")
    return results, stats


# ===========================================================================
# Stage 5 — Clue-matrix assembly
# ===========================================================================

def build_row_clues(left_recognized: list) -> list:
    """
    Build row_clues from the left clue area recognition results.

    left_recognized : 2-D list [ROWS][N_left_cols] of None | int
                      produced by recognize_cells_batch on left_clues_area.

    For each row, read cells left → right, drop None, keep integers.

    Example:
        row [None, 3, 1] → [3, 1]
        row [None, None, 5] → [5]
        row [None, None, None] → []  (empty clue row — no filled cells)

    Returns list of ROWS inner lists, each containing only integers.
    """
    row_clues = []
    for row in left_recognized:
        clues = [v for v in row if v is not None]
        row_clues.append(clues)
    return row_clues


def build_col_clues(top_recognized: list) -> list:
    """
    Build col_clues from the top clue area recognition results.

    top_recognized : 2-D list [N_top_rows][COLS] of None | int
                     produced by recognize_cells_batch on top_clues_area.

    For each column, read cells top → bottom, drop None, keep integers.

    Example:
        column [None, 2, 1] → [2, 1]
        column [None, None, 6] → [6]
        column [None, None, None] → []

    Returns list of COLS inner lists, each containing only integers.
    """
    if not top_recognized:
        return []
    n_cols = len(top_recognized[0])
    col_clues = []
    for c in range(n_cols):
        clues = [top_recognized[r][c]
                 for r in range(len(top_recognized))
                 if top_recognized[r][c] is not None]
        col_clues.append(clues)
    return col_clues


def validate_clue_matrix(matrix: list, expected_len: int,
                         label: str) -> bool:
    """
    Run lightweight sanity checks on a finished clue matrix.

    Checks performed:
        • outer length == expected_len
        • every entry is a list
        • every value inside is an int (no None, no float)
        • no negative values
        • warn on suspiciously large values (> 99)
        • warn on completely empty inner lists

    Returns True if all hard checks pass.  Warnings are printed but do
    not cause failure.
    """
    ok = True

    if len(matrix) != expected_len:
        print(f"  [ERROR] {label}: expected {expected_len} entries, "
              f"got {len(matrix)}")
        ok = False

    for i, inner in enumerate(matrix):
        if not isinstance(inner, list):
            print(f"  [ERROR] {label}[{i}] is not a list: {inner!r}")
            ok = False
            continue
        for j, v in enumerate(inner):
            if not isinstance(v, int):
                print(f"  [ERROR] {label}[{i}][{j}] is not int: {v!r}")
                ok = False
            elif v < 0:
                print(f"  [ERROR] {label}[{i}][{j}] is negative: {v}")
                ok = False
            elif v > 99:
                print(f"  [WARN]  {label}[{i}][{j}] = {v}  (unusually large)")
        if len(inner) == 0:
            print(f"  [WARN]  {label}[{i}] is empty  "
                  f"(no filled cells detected for this clue)")

    return ok


def save_result_json(path: str,
                     row_clues: list,
                     col_clues: list) -> None:
    """Save the final row_clues / col_clues to a JSON file."""
    with open(path, "w") as fh:
        json.dump({"row_clues": row_clues, "col_clues": col_clues},
                  fh, indent=2)
    print(f"  saved  {path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # ======================================================================
    # CLI
    # ======================================================================
    img_path, n_rows, n_cols = parse_args()
    ensure_dir(DEBUG_DIR)

    # ======================================================================
    # Stage 1 — Geometry
    # ======================================================================
    log_info("Loading image")
    try:
        img  = load_image(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        log_error(f"Could not load image: {exc}")
        sys.exit(1)

    save_debug("original.png", img)
    log_info(f"Image size: {img.shape[1]} x {img.shape[0]} (W x H)")
    log_info(f"Board size from CLI: {n_rows} rows x {n_cols} cols")

    log_info("Detecting puzzle region")
    try:
        layout = detect_puzzle_layout(gray, n_rows, n_cols)
    except Exception as exc:
        log_error(f"Could not detect puzzle region: {exc}")
        sys.exit(1)

    validate_geometry(layout, img.shape)

    log_info("Splitting puzzle into top / left / board")
    try:
        regions = split_puzzle_regions(img, layout)
    except Exception as exc:
        log_error(f"Could not split puzzle regions: {exc}")
        sys.exit(1)

    top_clues_area  = regions["top_clues_area"]   # → col_clues
    left_clues_area = regions["left_clues_area"]  # → row_clues
    board_area      = regions["board_area"]

    # Sanity: each region must be non-empty
    for region_name, region in (("top_clues_area", top_clues_area),
                                ("left_clues_area", left_clues_area),
                                ("board_area", board_area)):
        if region is None or region.size == 0:
            log_error(f"{region_name} crop is empty — cannot continue")
            sys.exit(1)

    save_debug("puzzle.png",            crop_bbox(img, layout["puzzle_bbox"]))
    save_debug("top.png",               top_clues_area)
    save_debug("left.png",              left_clues_area)
    save_debug("board.png",             board_area)
    save_debug("puzzle_with_lines.png", draw_puzzle_overlay(img, layout))

    def fmt(b):
        return f"({b[0]},{b[1]}) to ({b[2]},{b[3]})"

    print()
    print("=== Stage 1 — Geometry ===")
    print(f"  puzzle_bbox      : {fmt(layout['puzzle_bbox'])}")
    print(f"  top_clues_bbox   : {fmt(layout['top_clues_bbox'])}"
          f"  {top_clues_area.shape[1]}w x {top_clues_area.shape[0]}h")
    print(f"  left_clues_bbox  : {fmt(layout['left_clues_bbox'])}"
          f"  {left_clues_area.shape[1]}w x {left_clues_area.shape[0]}h")
    print(f"  board_bbox       : {fmt(layout['board_bbox'])}"
          f"  {board_area.shape[1]}w x {board_area.shape[0]}h")
    print(f"  board cell size  : {layout['board_cell_w']:.1f} px wide"
          f" x {layout['board_cell_h']:.1f} px tall")

    # ======================================================================
    # Stage 2 — Clue-cell extraction
    # (top_clues_area and left_clues_area processed INDEPENDENTLY)
    # ======================================================================
    log_info("Extracting clue cells")

    try:
        top_cells, top_h_bands, top_col_bands = split_top_clue_area(
            top_clues_area, layout
        )
    except Exception as exc:
        log_error(f"Could not extract top clue cells: {exc}")
        sys.exit(1)

    try:
        left_cells, left_row_bands, left_col_bands = split_left_clue_area(
            left_clues_area, layout
        )
    except Exception as exc:
        log_error(f"Could not extract left clue cells: {exc}")
        sys.exit(1)

    n_top_rows  = len(top_cells)
    n_left_cols = len(left_cells[0]) if left_cells else 0

    # Sanity: top grid must have exactly COLS columns
    if top_cells and len(top_cells[0]) != n_cols:
        log_warn(f"top grid has {len(top_cells[0])} cols, expected {n_cols}")
    # Sanity: left grid must have exactly ROWS rows
    if len(left_cells) != n_rows:
        log_warn(f"left grid has {len(left_cells)} rows, expected {n_rows}")

    top_dir  = os.path.join(DEBUG_DIR, "top_cells")
    left_dir = os.path.join(DEBUG_DIR, "left_cells")
    save_cell_crops(top_cells,  top_dir)
    save_cell_crops(left_cells, left_dir)

    save_debug("top_grid_overlay.png", draw_grid_overlay(
        top_clues_area, top_h_bands, top_col_bands,
        label=f"top_clues_area  {n_top_rows}rows x {n_cols}cols"))
    save_debug("left_grid_overlay.png", draw_grid_overlay(
        left_clues_area, left_row_bands, left_col_bands,
        label=f"left_clues_area  {n_rows}rows x {n_left_cols}cols"))

    top_nonnull  = sum(1 for row in top_cells  for c in row if c is not None)
    left_nonnull = sum(1 for row in left_cells for c in row if c is not None)

    print()
    print("=== Stage 2 — Cell extraction ===")
    print(f"  top_clues_area  : {n_top_rows} clue rows × {n_cols} cols"
          f"  = {n_top_rows * n_cols} slots  ({top_nonnull} non-null crops)")
    print(f"  left_clues_area : {n_rows} rows × {n_left_cols} clue cols"
          f"  = {n_rows * n_left_cols} slots  ({left_nonnull} non-null crops)")
    print(f"  board cell size : {layout['board_cell_w']:.1f} px wide"
          f" x {layout['board_cell_h']:.1f} px tall")

    # ======================================================================
    # Stage 3 — Cell cleaning
    # top_cells and left_cells are cleaned INDEPENDENTLY
    # ======================================================================
    log_info("Cleaning clue cells")

    top_clean_dir  = os.path.join(DEBUG_DIR, "top_cells_clean")
    left_clean_dir = os.path.join(DEBUG_DIR, "left_cells_clean")

    try:
        top_stats,  top_cells_clean  = clean_cells_batch(top_cells,  top_clean_dir)
        left_stats, left_cells_clean = clean_cells_batch(left_cells, left_clean_dir)
    except Exception as exc:
        log_error(f"Cell cleaning failed: {exc}")
        sys.exit(1)

    all_sizes   = top_stats["sizes"] + left_stats["sizes"]
    avg_h = float(np.mean([s[0] for s in all_sizes])) if all_sizes else 0.0
    avg_w = float(np.mean([s[1] for s in all_sizes])) if all_sizes else 0.0
    total_clean_empty = top_stats["empty_count"] + left_stats["empty_count"]
    total_clean       = top_stats["total_cleaned"] + left_stats["total_cleaned"]

    # Warn if almost all cells came out empty (likely a cleaning failure)
    if total_clean > 0 and total_clean_empty / total_clean > 0.9:
        log_warn(f"{total_clean_empty}/{total_clean} cells detected as empty "
                 f"after cleaning — check DARK_THRESH / cell size")

    print()
    print("=== Stage 3 — Cell cleaning ===")
    print(f"  top  cells : {top_stats['total_cleaned']} cleaned"
          f"  ({top_stats['empty_count']} detected empty)")
    print(f"  left cells : {left_stats['total_cleaned']} cleaned"
          f"  ({left_stats['empty_count']} detected empty)")
    print(f"  avg cleaned size    : {avg_w:.0f} w × {avg_h:.0f} h px"
          f"  (upscale {CELL_UPSCALE}×)")

    # ======================================================================
    # Stage 4 — Cell recognition
    # top_cells_clean and left_cells_clean processed INDEPENDENTLY
    # ======================================================================
    log_info("Recognizing clue cells")

    get_digit_templates()   # pre-build once; cached for both batches

    top_recog_dir  = os.path.join(DEBUG_DIR, "top_cells_recognized")
    left_recog_dir = os.path.join(DEBUG_DIR, "left_cells_recognized")

    try:
        top_results,  top_recog_stats  = recognize_cells_batch(
            top_cells_clean,  top_recog_dir,  area_label="top")
        left_results, left_recog_stats = recognize_cells_batch(
            left_cells_clean, left_recog_dir, area_label="left")
    except Exception as exc:
        log_error(f"Recognition failed: {exc}")
        sys.exit(1)

    # Save per-cell JSON
    recog_json_path = os.path.join(DEBUG_DIR, "cell_recognition.json")
    try:
        with open(recog_json_path, "w") as fh:
            json.dump({"top": top_results, "left": left_results}, fh, indent=2)
        print(f"  saved  {recog_json_path}")
    except Exception as exc:
        log_warn(f"Could not save cell_recognition.json: {exc}")

    total_recog     = top_recog_stats["recognized"] + left_recog_stats["recognized"]
    total_recog_empty = top_recog_stats["empty"]    + left_recog_stats["empty"]
    total_two_digit = top_recog_stats["two_digit"]  + left_recog_stats["two_digit"]
    grand_total     = top_recog_stats["total"]      + left_recog_stats["total"]

    # Warn on suspicious recognition ratios
    if grand_total > 0 and total_recog / grand_total < 0.1:
        log_warn("Fewer than 10% of cells were recognized — "
                 "templates may not match puzzle font")
    if grand_total > 0 and total_two_digit / max(total_recog, 1) > 0.5:
        log_warn(f"More than half of recognized cells are two-digit "
                 f"({total_two_digit}/{total_recog}) — verify splitting logic")

    print()
    print("=== Stage 4 — Recognition ===")
    print(f"  top  cells : {top_recog_stats['recognized']}/{top_recog_stats['total']}"
          f"  recognized  ({top_recog_stats['two_digit']} two-digit)")
    print(f"  left cells : {left_recog_stats['recognized']}/{left_recog_stats['total']}"
          f"  recognized  ({left_recog_stats['two_digit']} two-digit)")
    print(f"  total      : {total_recog}/{grand_total} recognized"
          f"  |  {total_recog_empty} empty  |  {total_two_digit} two-digit")

    # ======================================================================
    # Stage 5 — Clue-matrix assembly
    # Uses only in-memory top_results / left_results.  No re-recognition.
    # ======================================================================
    log_info("Building final clue matrices")

    row_clues = build_row_clues(left_results)
    col_clues = build_col_clues(top_results)

    log_info("Validating clue matrices")
    row_ok = validate_clue_matrix(row_clues, n_rows, "row_clues")
    col_ok = validate_clue_matrix(col_clues, n_cols, "col_clues")
    if row_ok and col_ok:
        log_info("Validation passed")

    # ======================================================================
    # Save all outputs
    # ======================================================================
    log_info("Saving debug outputs")

    result_json_path  = os.path.join(DEBUG_DIR, "result.json")
    cells_used_path   = os.path.join(DEBUG_DIR, "final_cells_used.json")
    metadata_path     = os.path.join(DEBUG_DIR, "metadata.json")

    save_result_json(result_json_path, row_clues, col_clues)

    try:
        with open(cells_used_path, "w") as fh:
            json.dump({"left_recognized": left_results,
                       "top_recognized":  top_results}, fh, indent=2)
        print(f"  saved  {cells_used_path}")
    except Exception as exc:
        log_warn(f"Could not save final_cells_used.json: {exc}")

    save_metadata_json(metadata_path, img_path, n_rows, n_cols,
                       layout, n_top_rows, n_left_cols)

    # ======================================================================
    # Final summary + output
    # ======================================================================
    total_clue_ints = (sum(len(r) for r in row_clues) +
                       sum(len(c) for c in col_clues))

    print()
    print("=== Stage 5 — Clue assembly ===")
    print(f"  row clue lists   : {len(row_clues)}  (expected {n_rows})")
    print(f"  col clue lists   : {len(col_clues)}  (expected {n_cols})")
    print(f"  total clue ints  : {total_clue_ints}")

    print()
    print(f"row_clues = {row_clues}")
    print(f"col_clues = {col_clues}")


if __name__ == "__main__":
    main()
