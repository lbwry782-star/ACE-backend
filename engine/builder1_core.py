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
    morphological_match_notes: str
    advertising_reason: str


@dataclass
class Builder1ModelOutput:
    resolved_product_name: str
    language: str
    object_a: str
    secondary_object_a: str
    object_b: str
    similarity_score: float
    advertising_promise: str
    headline: str
    headline_placement: str
    marketing_text: str


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
    headline: str
    headline_placement: str
    marketing_text: str
    image_validation_passed: bool


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


def build_object_candidate(
    name: str,
    similarity_to_a: float,
    morphological_match_notes: str = "",
    advertising_reason: str = "",
) -> ObjectCandidate:
    clean_name = normalize_text(name)
    return ObjectCandidate(
        name=clean_name,
        similarity_to_a=similarity_to_a,
        morphological_match_notes=normalize_text(morphological_match_notes),
        advertising_reason=normalize_text(advertising_reason),
    )


def build_object_candidate_from_model(
    name: str,
    similarity_to_a: float,
    morphological_match_notes: str = "",
    advertising_reason: str = "",
) -> ObjectCandidate:
    return build_object_candidate(
        name=name,
        similarity_to_a=similarity_to_a,
        morphological_match_notes=morphological_match_notes,
        advertising_reason=advertising_reason,
    )


def empty_builder1_model_output() -> Builder1ModelOutput:
    return Builder1ModelOutput(
        resolved_product_name="",
        language="",
        object_a="",
        secondary_object_a="",
        object_b="",
        similarity_score=0.0,
        advertising_promise="",
        headline="",
        headline_placement="",
        marketing_text="",
    )


def builder1_model_output_from_dict(data: dict) -> Builder1ModelOutput:
    data = data or {}
    return Builder1ModelOutput(
        resolved_product_name=normalize_text(data.get("resolved_product_name", "")),
        language=normalize_text(data.get("language", "")),
        object_a=normalize_text(data.get("object_a", "")),
        secondary_object_a=normalize_text(data.get("secondary_object_a", "")),
        object_b=normalize_text(data.get("object_b", "")),
        similarity_score=float(data.get("similarity_score", 0.0) or 0.0),
        advertising_promise=normalize_text(data.get("advertising_promise", "")),
        headline=normalize_text(data.get("headline", "")),
        headline_placement=normalize_text(data.get("headline_placement", "")),
        marketing_text=normalize_text(data.get("marketing_text", "")),
    )


def builder1_model_output_to_dict(model_output: Builder1ModelOutput) -> dict:
    return {
        "resolved_product_name": normalize_text(model_output.resolved_product_name),
        "language": normalize_text(model_output.language),
        "object_a": normalize_text(model_output.object_a),
        "secondary_object_a": normalize_text(model_output.secondary_object_a),
        "object_b": normalize_text(model_output.object_b),
        "similarity_score": model_output.similarity_score,
        "advertising_promise": normalize_text(model_output.advertising_promise),
        "headline": normalize_text(model_output.headline),
        "headline_placement": normalize_text(model_output.headline_placement),
        "marketing_text": normalize_text(model_output.marketing_text),
    }


def build_builder1_scaffold_plan(
    user_input: Builder1Input,
    model_output: Builder1ModelOutput,
) -> Builder1Plan:
    input_product_name = normalize_text(user_input.product_name)
    input_description = normalize_text(user_input.product_description)

    resolved_product_name = normalize_text(model_output.resolved_product_name)
    language = normalize_text(model_output.language) or detect_language(input_description)

    object_a_candidate = build_object_candidate_from_model(
        model_output.object_a,
        0.0,
        morphological_match_notes="",
        advertising_reason="",
    )

    object_b_candidate = build_object_candidate_from_model(
        model_output.object_b,
        model_output.similarity_score,
        morphological_match_notes="",
        advertising_reason=model_output.advertising_promise,
    )

    similarity_score = object_b_candidate.similarity_to_a
    advertising_promise = normalize_text(model_output.advertising_promise)
    mode = decide_mode(similarity_score)

    if mode == MODE_REPLACEMENT:
        composition = "replacement"
    else:
        composition = "partial_overlap"

    final_product_name = resolved_product_name or input_product_name
    headline = normalize_text(model_output.headline)
    headline_placement = normalize_text(model_output.headline_placement)
    marketing_text = normalize_text(model_output.marketing_text)
    image_validation_passed = False

    return Builder1Plan(
        product_name=final_product_name,
        product_description=input_description,
        language=language,
        advertising_promise=advertising_promise,
        object_a=object_a_candidate.name,
        secondary_object_a=normalize_text(model_output.secondary_object_a),
        object_b=object_b_candidate.name,
        similarity_score=similarity_score,
        mode=mode,
        composition=composition,
        headline=headline,
        headline_placement=headline_placement,
        marketing_text=marketing_text,
        image_validation_passed=image_validation_passed,
    )
