"""
Clean Builder1 rebuild scaffold.
Not wired into production yet.
"""

from dataclasses import dataclass

SIMILARITY_THRESHOLD_REPLACEMENT = 85
MODE_SIDE_BY_SIDE = "SIDE_BY_SIDE"
MODE_REPLACEMENT = "REPLACEMENT"


def decide_mode(similarity: float) -> str:
    if similarity >= SIMILARITY_THRESHOLD_REPLACEMENT:
        return MODE_REPLACEMENT
    return MODE_SIDE_BY_SIDE


@dataclass
class Builder1Input:
    product_name: str
    product_description: str


@dataclass
class ObjectCandidate:
    name: str
    similarity_to_a: float


@dataclass
class Builder1Plan:
    product_name: str
    product_description: str
    advertising_promise: str
    object_a: str
    object_b: str
    similarity_score: float
    mode: str


def normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def detect_language(text: str) -> str:
    value = normalize_text(text)
    if not value:
        return "en"
    hebrew_count = sum(1 for ch in value if "\u0590" <= ch <= "\u05FF")
    latin_count = sum(1 for ch in value if ("a" <= ch.lower() <= "z"))
    if hebrew_count > latin_count:
        return "he"
    return "en"


def derive_object_a_placeholder(description: str) -> str:
    value = normalize_text(description)
    if not value:
        return ""
    return value.split()[0]


def build_builder1_scaffold_plan(user_input: Builder1Input) -> Builder1Plan:
    name = normalize_text(user_input.product_name)
    description = normalize_text(user_input.product_description)

    language = detect_language(description)

    object_a = derive_object_a_placeholder(description)
    object_b = ""
    similarity_score = 0.0

    advertising_promise = ""

    mode = decide_mode(similarity_score)

    return Builder1Plan(
        product_name=name,
        product_description=description,
        advertising_promise=advertising_promise,
        object_a=object_a,
        object_b=object_b,
        similarity_score=similarity_score,
        mode=mode,
    )
