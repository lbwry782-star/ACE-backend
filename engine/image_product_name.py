"""
Image / Builder1 pipeline: product name presence checks for marketing copy.

Separated from engine.video_product_name so image code does not depend on video modules.
"""

from __future__ import annotations

import unicodedata


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def product_name_reused_in_copy(canonical: str, copy: str) -> bool:
    c = _nfc((canonical or "").strip())
    if not c:
        return True
    return c in _nfc(copy or "")
