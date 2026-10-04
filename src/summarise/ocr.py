"""Tesseract OCR over Putra screenshots (S2). Returns text and wall seconds."""
from __future__ import annotations

import re
import time
from pathlib import Path


def ocr_image(path: str | Path, lang: str = "eng") -> tuple[str, float, str | None]:
    """Returns (text, seconds, error). Never raises: OCR failures are logged
    per site and counted in the coverage figure."""
    t0 = time.perf_counter()
    try:
        import pytesseract
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("L")
            text = pytesseract.image_to_string(im, lang=lang)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text).strip()
        return text, time.perf_counter() - t0, None
    except Exception as e:  # noqa: BLE001
        return "", time.perf_counter() - t0, f"{type(e).__name__}: {e}"
