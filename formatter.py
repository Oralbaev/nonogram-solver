"""Output formatting for the nonogram solver."""

import os
import struct
import zlib

from solver import FILLED, EMPTY, Board

CELL_SYMBOLS = {
    FILLED: "■",
    EMPTY:  " ",
    -1:     "?",  # UNKNOWN
}

_CELL_W = 2  # display chars per cell: symbol + 1 space
_GROUP  = 5  # cells per visual group (classic nonogram grouping)


def _cell_str(cell: int) -> str:
    return CELL_SYMBOLS[cell] + " "


def _grid_content(items: list[str]) -> str:
    """Join CELL_W-char items in groups of _GROUP separated by │."""
    groups = [
        "".join(items[i : i + _GROUP])
        for i in range(0, len(items), _GROUP)
    ]
    return "│".join(groups)


def _border_line(n_cols: int, left: str, mid: str, right: str) -> str:
    seg = "─" * (_GROUP * _CELL_W)
    n_groups = (n_cols + _GROUP - 1) // _GROUP
    return left + mid.join([seg] * n_groups) + right


def format_board(
    board: Board,
    row_clues: list[list[int]],
    col_clues: list[list[int]],
) -> str:
    """Format the board as a proper nonogram grid with row and column clues.

    Layout:
      - Column clues stacked above the grid, bottom-aligned.
      - Row clues to the left of each row, right-aligned.
      - Cells grouped in blocks of 5 with │ separators.
      - Horizontal separator line every 5 rows.

    Symbols:  ■ = filled   (space) = empty   ? = unknown
    """
    n_rows = len(board)
    n_cols = len(board[0]) if board else 0

    # ── Row clue strings ────────────────────────────────────────────────────
    row_clue_strs = [" ".join(str(n) for n in clue) for clue in row_clues]
    rcw = max(len(s) for s in row_clue_strs) if row_clue_strs else 0

    # ── Column clue header grid (bottom-aligned) ────────────────────────────
    # Each cell is _CELL_W chars wide; numbers are right-justified inside it.
    cch = max(len(c) for c in col_clues) if col_clues else 0
    col_header: list[list[str]] = []
    for row_idx in range(cch):
        items: list[str] = []
        for clue in col_clues:
            offset = row_idx - (cch - len(clue))
            if offset >= 0:
                items.append(str(clue[offset]).rjust(_CELL_W))
            else:
                items.append(" " * _CELL_W)
        col_header.append(items)

    # ── Border lines ────────────────────────────────────────────────────────
    top    = _border_line(n_cols, "┌", "┬", "┐")
    mid    = _border_line(n_cols, "├", "┼", "┤")
    bottom = _border_line(n_cols, "└", "┴", "┘")

    # Padding on the left for lines that carry no row clue
    pad = " " * rcw

    lines: list[str] = []

    # Column clue rows (float above the top border, aligned to grid columns)
    for items in col_header:
        lines.append(pad + "  " + _grid_content(items))

    # Top border
    lines.append(pad + " " + top)

    # Grid rows
    for r in range(n_rows):
        cell_items = [_cell_str(board[r][c]) for c in range(n_cols)]
        row_clue  = row_clue_strs[r].rjust(rcw)
        lines.append(f"{row_clue} │{_grid_content(cell_items)}│")

        # Thin horizontal separator every _GROUP rows (except after the last)
        if (r + 1) % _GROUP == 0 and r + 1 < n_rows:
            lines.append(pad + " " + mid)

    # Bottom border
    lines.append(pad + " " + bottom)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PNG rendering (stdlib only: struct + zlib)
# ---------------------------------------------------------------------------

_CELL_PX   = 20   # pixels per cell
_PADDING   = 20   # white border around the grid
_GRID_PX   = 1    # grid line thickness between cells (0 = no lines)
_GRID_GRAY = 210  # grid line brightness  (0 = black, 255 = white)
_FILL_GRAY = 0    # filled cell: pure black
_BG_GRAY   = 255  # empty cell / background: pure white


def _make_png(width: int, height: int, rows: list[bytearray]) -> bytes:
    """Encode a grayscale image as a minimal PNG using only stdlib."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    raw  = b"".join(b"\x00" + bytes(r) for r in rows)
    idat = chunk(b"IDAT", zlib.compress(raw, 9))
    iend = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def render_png(board: Board, filepath: str = "solution.png") -> str:
    """Render the solved board as a minimalist PNG image.

    White background, black filled cells, subtle gray grid lines.
    No numbers or annotations.
    Returns the absolute path of the saved file.
    """
    n_rows = len(board)
    n_cols = len(board[0]) if board else 0

    img_w = _PADDING * 2 + n_cols * _CELL_PX + max(0, n_cols - 1) * _GRID_PX
    img_h = _PADDING * 2 + n_rows * _CELL_PX + max(0, n_rows - 1) * _GRID_PX

    # Start with grid-line color everywhere; padding and cells will overwrite.
    canvas = [bytearray([_GRID_GRAY] * img_w) for _ in range(img_h)]

    # Paint padding margins white.
    for y in range(img_h):
        for x in range(img_w):
            if (x < _PADDING or x >= img_w - _PADDING
                    or y < _PADDING or y >= img_h - _PADDING):
                canvas[y][x] = _BG_GRAY

    # Paint each cell.
    for r in range(n_rows):
        for c in range(n_cols):
            color = _FILL_GRAY if board[r][c] == FILLED else _BG_GRAY
            py = _PADDING + r * (_CELL_PX + _GRID_PX)
            px = _PADDING + c * (_CELL_PX + _GRID_PX)
            for dy in range(_CELL_PX):
                row = canvas[py + dy]
                for dx in range(_CELL_PX):
                    row[px + dx] = color

    abs_path = os.path.abspath(filepath)
    with open(abs_path, "wb") as f:
        f.write(_make_png(img_w, img_h, canvas))
    return abs_path


# ---------------------------------------------------------------------------
# Plain-text clue summary (used for debugging)
# ---------------------------------------------------------------------------

def format_clues(row_clues: list[list[int]], col_clues: list[list[int]]) -> str:
    """Plain-text summary of row and column clues."""
    lines: list[str] = ["Row clues:"]
    for i, clue in enumerate(row_clues):
        lines.append(f"  Row {i + 1}: {' '.join(str(n) for n in clue)}")
    lines += ["", "Column clues:"]
    for i, clue in enumerate(col_clues):
        lines.append(f"  Col {i + 1}: {' '.join(str(n) for n in clue)}")
    return "\n".join(lines)
