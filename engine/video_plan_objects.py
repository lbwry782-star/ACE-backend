"""
Video plan: reject objects whose primary role is to carry visible text or graphic communication.

Server-side validation only — no silent rewriting; planner must pick valid physical objects.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# (log_label, regex) — first match wins; labels are for logs only.
_VIDEO_PLAN_GRAPHIC_CONTENT_RULES: List[Tuple[str, re.Pattern[str]]] = [
    ("poster", re.compile(r"\bposters?\b", re.I)),
    ("flyer", re.compile(r"\bflyers?\b", re.I)),
    ("magazine", re.compile(r"\bmagazines?\b", re.I)),
    ("newspaper", re.compile(r"\bnewspapers?\b", re.I)),
    ("brochure", re.compile(r"\bbrochures?\b", re.I)),
    ("leaflet", re.compile(r"\bleaflets?\b", re.I)),
    ("pamphlet", re.compile(r"\bpamphlets?\b", re.I)),
    ("infographic", re.compile(r"\binfographics?\b", re.I)),
    ("manuscript", re.compile(r"\bmanuscripts?\b", re.I)),
    ("comic_book", re.compile(r"\bcomic\s+(book|strip)\b", re.I)),
    ("printed_display_piece", re.compile(
        r"\bprinted\s+(poster|image|photograph|photo|graphic|advertisement|ad|sheet|page|picture)\b",
        re.I,
    )),
    ("vibrant_printed", re.compile(r"\bvibrant\s+printed\b", re.I)),
    ("photograph", re.compile(r"\bphotographs?\b", re.I)),
    ("photo_of", re.compile(r"\bphoto\s+of\b", re.I)),
    ("image_of", re.compile(r"\bimage\s+of\b", re.I)),
    ("family_photo", re.compile(r"\bfamily\s+photos?\b", re.I)),
    ("oil_painting", re.compile(r"\boil\s+painting\b", re.I)),
    ("painting_of", re.compile(r"\bpainting\s+of\b", re.I)),
    ("watercolor_art", re.compile(r"\bwatercolor\s+(painting|on\s+paper|canvas)\b", re.I)),
    ("canvas_print", re.compile(r"\bcanvas\s+(print|painting|art)\b", re.I)),
    ("art_print", re.compile(r"\bart\s+print\b", re.I)),
    ("artwork", re.compile(r"\bartworks?\b", re.I)),
    ("wall_chart", re.compile(r"\b(wall\s+)?charts?\b", re.I)),
    ("diagram", re.compile(r"\bdiagrams?\b", re.I)),
    ("map_object", re.compile(r"\b(maps?|globes?)\b.*\b(showing|labeled|political|route)\b", re.I)),
    ("calendar", re.compile(r"\bwall\s+calendars?\b", re.I)),
    ("scoreboard", re.compile(r"\bscoreboards?\b", re.I)),
    ("screen_showing", re.compile(
        r"\b(screens?|monitors?|displays?)\s+(showing|displaying)\b",
        re.I,
    )),
    ("showing_on_screen", re.compile(
        r"\b(showing|displaying)\s+.*\b(on\s+)?(screen|monitor|display)\b",
        re.I,
    )),
    ("tv_showing", re.compile(r"\b(tv|television)\s+(showing|displaying|playing)\b", re.I)),
    ("e_reader", re.compile(r"\b(e-reader|ereader|kindle)\b", re.I)),
    ("open_book", re.compile(
        r"\b(open|opened)\s+books?\b|\bbooks?\s+(open|opened)\b|\bread(ing|able)\s+books?\b",
        re.I,
    )),
    ("book_with_text", re.compile(r"\bbooks?\s+with\s+(text|writing|print)\b", re.I)),
    ("sign_with", re.compile(r"\b(sign|signage|signpost)\s+(with|showing|bearing|reading)\b", re.I)),
    ("neon_sign", re.compile(r"\bneon\s+signs?\b", re.I)),
    ("billboard_with", re.compile(
        r"\bbillboards?\b.*\b(with|showing|displaying|advertisement|ad|text|image|graphic)\b",
        re.I,
    )),
    ("ad_on_billboard", re.compile(r"\b(advertisement|ad)\s+on\s+billboard\b", re.I)),
    ("barcode", re.compile(r"\bbarcodes?\b", re.I)),
    ("price_tag", re.compile(r"\bprice\s+tags?\b", re.I)),
    ("product_label", re.compile(r"\bproduct\s+labels?\b", re.I)),
    ("packaging_graphic", re.compile(
        r"\b(packaging|package|carton|box)\s+.*\b(logo|text|brand|graphic|printed\s+design)\b",
        re.I,
    )),
    ("branded_packaging", re.compile(r"\bbranded\s+(packaging|box|carton)\b", re.I)),
    ("whiteboard_text", re.compile(
        r"\b(chalk|white)boards?\b.*\b(with|showing|writing|text)\b",
        re.I,
    )),
    ("greeting_card", re.compile(r"\bgreeting\s+cards?\b", re.I)),
    ("postcard_scene", re.compile(r"\bpostcards?\b.*\b(with|showing|scene)\b", re.I)),
    ("certificate", re.compile(r"\b(certificates?|diplomas?)\b", re.I)),
    ("led_message", re.compile(r"\bled\s+(display|screen|sign|message|ticker)\b", re.I)),
    ("logo_graphic", re.compile(r"\blogo\s+(on|printed|visible|graphic)\b", re.I)),
    ("sticker", re.compile(r"\bstickers?\b", re.I)),
]


def video_plan_object_blob_implies_graphic_text_content(blob: str) -> Optional[str]:
    """
    If any rule matches the combined object/interaction text, return the rule label for logging.
    Otherwise return None (acceptable).
    """
    t = unicodedata.normalize("NFC", blob or "")
    if not t.strip():
        return None
    for label, rx in _VIDEO_PLAN_GRAPHIC_CONTENT_RULES:
        if rx.search(t):
            return label
    return None
