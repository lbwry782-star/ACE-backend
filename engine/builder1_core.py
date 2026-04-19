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
    is_concrete_physical: bool
    has_forbidden_textual_markings: bool


@dataclass
class Builder1Plan:
    product_name: str
    product_description: str
    language: str
    advertising_promise: str
    object_a: str
    secondary_object_a: str
    object_b: str
    similarity_score: float
    mode: str
    composition: str


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


def derive_secondary_object_a_placeholder(object_a: str) -> str:
    value = normalize_text(object_a)
    if not value:
        return ""
    return ""


def is_concrete_physical_object_candidate(name: str) -> bool:
    value = normalize_text(name)
    return bool(value)


def maybe_has_forbidden_textual_markings(name: str) -> bool:
    value = normalize_text(name)
    return any(ch.isdigit() for ch in value)


def build_object_candidate(name: str, similarity_to_a: float) -> ObjectCandidate:
    clean_name = normalize_text(name)
    return ObjectCandidate(
        name=clean_name,
        similarity_to_a=similarity_to_a,
        is_concrete_physical=is_concrete_physical_object_candidate(clean_name),
        has_forbidden_textual_markings=maybe_has_forbidden_textual_markings(clean_name),
    )


def build_builder1_scaffold_plan(user_input: Builder1Input) -> Builder1Plan:
    name = normalize_text(user_input.product_name)
    description = normalize_text(user_input.product_description)
    language = detect_language(description)

    object_a_candidate = build_object_candidate(
        derive_object_a_placeholder(description),
        0.0,
    )
    object_b_candidate = build_object_candidate("", 0.0)

    secondary_object_a = derive_secondary_object_a_placeholder(object_a_candidate.name)
    similarity_score = object_b_candidate.similarity_to_a
    advertising_promise = ""
    mode = decide_mode(similarity_score)

    if mode == MODE_REPLACEMENT:
        composition = "replacement"
    else:
        composition = "partial_overlap"

    return Builder1Plan(
        product_name=name,
        product_description=description,
        language=language,
        advertising_promise=advertising_promise,
        object_a=object_a_candidate.name,
        secondary_object_a=secondary_object_a,
        object_b=object_b_candidate.name,
        similarity_score=similarity_score,
        mode=mode,
        composition=composition,
    )
