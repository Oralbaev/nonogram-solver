import asyncio
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

load_dotenv()

from src.formatter import render_png
from src.parser import parse_nonogram_image
from src.solver import UNKNOWN, solve, validate

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ASK_COLS, ASK_ROWS = range(2)

# user_id -> {"image_path": str, "n_cols": int}
sessions: dict[int, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)

# Always relative to this file so the bot works regardless of CWD.
TEMP_DIR = Path(__file__).parent / "tmp_bot"
TEMP_DIR.mkdir(exist_ok=True)

# Compressed photos from Telegram are saved here and never deleted,
# so they can be inspected manually with: python src/parser.py debug_photos/file.jpg ROWS COLS
DEBUG_DIR = Path(__file__).parent / "debug_photos"
DEBUG_DIR.mkdir(exist_ok=True)

_HELP = (
    "Nonogram Solver Bot\n\n"
    "Send me a screenshot of a nonogram puzzle and I will solve it.\n"
    "I will ask for the number of columns and rows, then send back the solved image.\n\n"
    "/start — show this help\n"
    "/cancel — cancel the current session"
)


async def cmd_start(update: Update, context) -> int:
    user_id = update.effective_user.id
    if user_id in sessions:
        _cleanup_session(user_id)
    await update.message.reply_text(_HELP)
    return ConversationHandler.END


async def cmd_cancel(update: Update, context) -> int:
    user_id = update.effective_user.id
    if user_id in sessions:
        _cleanup_session(user_id)
        await update.message.reply_text("Session cancelled.")
    else:
        await update.message.reply_text("No active session to cancel.")
    return ConversationHandler.END


def _enhance_photo(src_path: str) -> str:
    """Preprocess a Telegram-compressed JPEG to improve grid-line detection.

    Telegram re-encodes photos as JPEG which introduces two problems:
      1. 8×8 block artifacts that the grid detector misreads as extra lines.
      2. Blurry edges on grid lines, reducing the dark-pixel fraction below
         the LINE_FRAC threshold so real lines are missed.

    Fix: median blur removes block artifacts, then unsharp mask restores
    crispness.  Result is saved as PNG (lossless) next to the original.
    """
    img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return src_path
    # 3×3 median removes JPEG block noise while keeping sharp edges intact.
    denoised = cv2.medianBlur(img, 3)
    # Unsharp mask: make grid lines and digit edges crisper again.
    blur = cv2.GaussianBlur(denoised, (0, 0), 1.5)
    sharpened = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    out_path = str(Path(src_path).with_suffix("")) + "_enhanced.png"
    cv2.imencode(".png", sharpened)[1].tofile(out_path)
    return out_path


async def handle_image(update: Update, context) -> int:
    user_id = update.effective_user.id
    message = update.message

    if message.photo:
        file_obj = await context.bot.get_file(message.photo[-1].file_id)
        ext = ".jpg"
        is_photo = True
    elif message.document:
        mime = message.document.mime_type or ""
        if not mime.startswith("image/"):
            await message.reply_text("Please send a PNG or JPG image of a nonogram puzzle.")
            return ConversationHandler.END
        file_obj = await context.bot.get_file(message.document.file_id)
        raw_name = message.document.file_name or "image.jpg"
        ext = Path(raw_name).suffix.lower() or ".jpg"
        is_photo = False
    else:
        return ConversationHandler.END

    stem = f"{user_id}_{uuid.uuid4().hex[:8]}"
    image_path = TEMP_DIR / f"{stem}{ext}"
    await file_obj.download_to_drive(str(image_path))

    if is_photo:
        # Keep the raw download in debug_photos/ so it can be inspected manually:
        #   python src/parser.py debug_photos/<file>.jpg ROWS COLS
        debug_copy = DEBUG_DIR / f"{stem}{ext}"
        shutil.copy2(str(image_path), str(debug_copy))
        logger.info("Debug copy of Telegram photo saved: %s", debug_copy)

        # Apply preprocessing and use the enhanced image for parsing.
        enhanced_path = await asyncio.get_running_loop().run_in_executor(
            _executor, _enhance_photo, str(image_path)
        )
        parse_path = enhanced_path
    else:
        parse_path = str(image_path)

    sessions[user_id] = {"image_path": str(image_path), "parse_path": parse_path}
    await message.reply_text(
        "Image received! How many columns does the puzzle have?\n"
        "(Send /cancel to stop.)"
    )
    return ASK_COLS


async def handle_cols(update: Update, context) -> int:
    user_id = update.effective_user.id

    try:
        n_cols = int(update.message.text.strip())
        if n_cols < 1:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Please send a positive integer for the number of columns.")
        return ASK_COLS

    sessions[user_id]["n_cols"] = n_cols
    await update.message.reply_text("How many rows does the puzzle have?")
    return ASK_ROWS


async def handle_rows(update: Update, context) -> int:
    user_id = update.effective_user.id

    try:
        n_rows = int(update.message.text.strip())
        if n_rows < 1:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Please send a positive integer for the number of rows.")
        return ASK_ROWS

    session = sessions[user_id]
    n_cols: int = session["n_cols"]
    image_path: str = session["parse_path"]

    await update.message.reply_text(
        f"Solving a {n_rows}×{n_cols} nonogram… This may take a moment."
    )

    loop = asyncio.get_running_loop()
    result_path: str | None = None
    try:
        result_path, row_clues, col_clues = await loop.run_in_executor(
            _executor,
            _run_solver,
            image_path,
            n_rows,
            n_cols,
            user_id,
        )
    except FileNotFoundError as exc:
        await update.message.reply_text(f"Image file not found: {exc}")
        _cleanup_session(user_id)
        return ConversationHandler.END
    except ValueError as exc:
        await update.message.reply_text(f"Could not parse puzzle clues: {exc}")
        _cleanup_session(user_id)
        return ConversationHandler.END
    except RuntimeError as exc:
        await update.message.reply_text(f"Solver error: {exc}")
        _cleanup_session(user_id)
        return ConversationHandler.END
    except Exception:
        logger.exception("Unexpected error for user %d", user_id)
        await update.message.reply_text("An unexpected error occurred. Please try again.")
        _cleanup_session(user_id)
        return ConversationHandler.END

    # Always show what was parsed so the user can verify the OCR result.
    clue_preview = (
        f"Parsed clues:\n"
        f"row_clues = {row_clues}\n"
        f"col_clues = {col_clues}\n\n"
        "If these look wrong, resend the image as a File (not a photo)."
    )
    await update.message.reply_text(clue_preview)

    if result_path is None:
        await update.message.reply_text(
            "No solution found. The parsed clues above may be incorrect — "
            "try resending the image as a File for better OCR quality."
        )
    else:
        try:
            with open(result_path, "rb") as f:
                await update.message.reply_photo(f, caption="Here is your solved nonogram!")
        except FileNotFoundError:
            await update.message.reply_text(
                "Solver finished but the output image was not created."
            )
    _cleanup_session(user_id, result_path)

    return ConversationHandler.END


def _run_solver(
    image_path: str, n_rows: int, n_cols: int, user_id: int
) -> tuple[str | None, list[list[int]], list[list[int]]]:
    """Blocking: parse clues → solve → render PNG. Runs in thread pool."""
    row_clues, col_clues = parse_nonogram_image(image_path, n_rows, n_cols)

    board = [[UNKNOWN] * n_cols for _ in range(n_rows)]
    solution = solve(board, row_clues, col_clues)

    if solution is None:
        return None, row_clues, col_clues

    if not validate(solution, row_clues, col_clues):
        raise RuntimeError("Solver returned an invalid solution.")

    out_path = str(TEMP_DIR / f"solution_{user_id}.png")
    render_png(solution, filepath=out_path)
    return out_path, row_clues, col_clues


def _cleanup_session(user_id: int, result_path: str | None = None) -> None:
    session = sessions.pop(user_id, {})
    for key in ("image_path", "parse_path"):
        p = session.get(key)
        if p:
            Path(p).unlink(missing_ok=True)
    if result_path:
        Path(result_path).unlink(missing_ok=True)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "Create a .env file (see .env.example) or set the variable in your environment."
        )

    app = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image),
        ],
        states={
            ASK_COLS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_cols)],
            ASK_ROWS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_rows)],
        },
        fallbacks=[
            CommandHandler("cancel", cmd_cancel),
            CommandHandler("start", cmd_start),
        ],
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(conv)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
