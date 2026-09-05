"""
Builder2 essential fact fusion / selection preservation — methodology + structural gates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from engine.builder2_product_semantic_brief import _clean, get_product_semantic_brief

BUILDER2_ESSENTIAL_FACT_FUSION = """
ESSENTIAL FACT FUSION — when essentialFacts contain both product/category identity and relative-advantage
identity, fuse them into ONE integrated visual mechanism.

PRODUCT / CATEGORY + RELATIVE ADVANTAGE → ONE INTEGRATED VISUAL MECHANISM

Reject expressing the relative advantage alone while product/category identity disappears from the mechanism.
Analogy is allowed when it remains an analogy OF the product/category — not a generic visualization of
the advantage alone.
""".strip()

BUILDER2_CULTURAL_CONTEXT = """
CULTURAL CONTEXT — evaluate symbolism in the intended market context.

For Israeli-market advertising: military / IDF-associated everyday objects, shapes, equipment, or
familiar references must NOT automatically be classified as militant, aggressive, extremist, or inappropriate
merely because they are military-associated. Assess actual cultural meaning, tone, execution, and context.

Still reject genuinely violent, threatening, hateful, extremist, or unsafe executions.
ASSESS CULTURAL MEANING IN CONTEXT — not MILITARY ASSOCIATION = AUTOMATIC REJECTION.
""".strip()

BUILDER2_ESSENTIAL_FACT_FUSION_METHODOLOGY = "\n\n".join(
    [BUILDER2_ESSENTIAL_FACT_FUSION, BUILDER2_CULTURAL_CONTEXT]
)

_CATEGORY_MARKERS = frozenset(
    {
        "perfume", "fragrance", "cologne", "men", "women", "product", "category",
        "בושם", "ניחוח", "ריח", "לגברים", "לנשים", "גברים", "נשים", "מוצר", "קטגור",
    }
)
_ADVANTAGE_MARKERS = frozenset(
    {
        "israel", "israeli", "local", "locally", "made", "origin", "produced", "alternative",
        "ישראל", "ישראלי", "מקומי", "מיוצר", "תוצרת", "חלופה",
    }
)

CREATOR_FUSION_EVIDENCE_FIELDS: Tuple[str, ...] = (
    "productCategoryEssentialFact",
    "relativeAdvantageEssentialFact",
    "productCategoryInVisualMechanism",
    "relativeAdvantageInVisualMechanism",
    "fusionOrCausalLinkExplanation",
    "meaningfulConnectionWithoutProductNameCopy",
    "usedFactOutsideSelectedBrief",
)

JUDGE_FUSION_ASSESSMENT_FIELDS: Tuple[str, ...] = (
    "productCategoryEssentialFactPreserved",
    "relativeAdvantageEssentialFactPreserved",
    "productCategoryAppliedInVisualMechanism",
    "relativeAdvantageAppliedInVisualMechanism",
    "factsIntegratedIntoOneMechanism",
    "advantageVisualizedWithoutProductApplication",
    "unselectedFactIntroduced",
    "fusionEligible",
    "fusionRequired",
)


def _contains_marker(text: str, markers: frozenset[str]) -> bool:
    lowered = _clean(text).casefold()
    return any(marker.casefold() in lowered for marker in markers)


def classify_essential_fact_text(text: str) -> str:
    category = _contains_marker(text, _CATEGORY_MARKERS)
    advantage = _contains_marker(text, _ADVANTAGE_MARKERS)
    if category and not advantage:
        return "category_identity"
    if advantage and not category:
        return "advantage"
    if category and advantage:
        return "category_identity"
    return "general"


def partition_essential_facts(brief: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    category: List[str] = []
    advantage: List[str] = []
    general: List[str] = []
    for item in brief.get("essentialFacts") or []:
        text = _clean(item.get("text") if isinstance(item, dict) else item)
        if not text:
            continue
        kind = classify_essential_fact_text(text)
        if kind == "category_identity":
            category.append(text)
        elif kind == "advantage":
            advantage.append(text)
        else:
            general.append(text)
    return category, advantage, general


def fusion_required_for_brief(brief: Dict[str, Any]) -> bool:
    category, advantage, _ = partition_essential_facts(brief)
    return bool(category and advantage)


def build_default_creator_essential_fact_fusion_evidence(
    *,
    product_category_fact: str = "",
    relative_advantage_fact: str = "",
    fusion_explanation: str = "",
) -> Dict[str, Any]:
    return {
        "productCategoryEssentialFact": product_category_fact or "...",
        "relativeAdvantageEssentialFact": relative_advantage_fact or "...",
        "productCategoryInVisualMechanism": "...",
        "relativeAdvantageInVisualMechanism": "...",
        "fusionOrCausalLinkExplanation": fusion_explanation or "...",
        "meaningfulConnectionWithoutProductNameCopy": True,
        "usedFactOutsideSelectedBrief": False,
    }


def build_default_judge_essential_fact_fusion_assessment(
    *,
    fusion_required: bool = False,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "productCategoryEssentialFactPreserved": True,
        "relativeAdvantageEssentialFactPreserved": True,
        "productCategoryAppliedInVisualMechanism": True,
        "relativeAdvantageAppliedInVisualMechanism": True,
        "factsIntegratedIntoOneMechanism": True,
        "advantageVisualizedWithoutProductApplication": False,
        "unselectedFactIntroduced": False,
        "fusionRequired": bool(fusion_required),
        "fusionEligible": True,
        "notes": notes or "Essential facts are preserved and integrated in the visual mechanism.",
    }


def build_judge_essential_fact_fusion_prompt_text() -> str:
    fields = ", ".join(JUDGE_FUSION_ASSESSMENT_FIELDS)
    return (
        "essentialFactFusionAssessment is mandatory when productSemanticBrief.essentialFacts require fusion "
        "(both product/category identity and relative-advantage identity).\n"
        f"Return all fields ({fields}) plus notes.\n"
        "fusionRequired=true only when Strategy selected essential facts that require product/category + advantage fusion.\n"
        "fusionEligible=true only when required essential facts survive, product/category is represented in the mechanism, "
        "advantage is represented, facts integrate into one mechanism, and no unselected creative fact was introduced.\n"
        "If advantageVisualizedWithoutProductApplication=true, fusionEligible must be false and eligible must be false.\n"
        "Assess cultural meaning in context — military-associated everyday objects in Israeli-market advertising are not "
        "automatic rejection reasons unless execution is genuinely violent, threatening, hateful, extremist, or unsafe.\n"
        "Compare claims against the selected post-Strategy semantic brief only — not raw productDescription."
    )


def compute_fusion_eligible(assessment: Mapping[str, Any]) -> bool:
    if assessment.get("fusionRequired") is not True:
        return assessment.get("fusionEligible") is not False
    if assessment.get("productCategoryEssentialFactPreserved") is not True:
        return False
    if assessment.get("relativeAdvantageEssentialFactPreserved") is not True:
        return False
    if assessment.get("productCategoryAppliedInVisualMechanism") is not True:
        return False
    if assessment.get("relativeAdvantageAppliedInVisualMechanism") is not True:
        return False
    if assessment.get("factsIntegratedIntoOneMechanism") is not True:
        return False
    if assessment.get("advantageVisualizedWithoutProductApplication") is True:
        return False
    if assessment.get("unselectedFactIntroduced") is True:
        return False
    return True


def apply_fusion_eligibility_rules(judgment: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(judgment)
    assessment = out.get("essentialFactFusionAssessment")
    if not isinstance(assessment, dict):
        return out
    normalized = dict(assessment)
    if normalized.get("fusionRequired") is True:
        normalized["fusionEligible"] = compute_fusion_eligible(normalized)
    out["essentialFactFusionAssessment"] = normalized
    if judgment_rejects_essential_fact_fusion(out):
        out["eligible"] = False
        if normalized.get("advantageVisualizedWithoutProductApplication") is True:
            out.setdefault("disqualifiers", []).append("advantage_visualized_without_product")
        elif normalized.get("unselectedFactIntroduced") is True:
            out.setdefault("disqualifiers", []).append("unselected_fact_introduced")
        elif normalized.get("productCategoryAppliedInVisualMechanism") is False:
            out.setdefault("disqualifiers", []).append("product_category_not_applied_in_mechanism")
        elif normalized.get("factsIntegratedIntoOneMechanism") is False:
            out.setdefault("disqualifiers", []).append("essential_facts_not_integrated")
        elif normalized.get("fusionEligible") is False:
            out.setdefault("disqualifiers", []).append("essential_fact_fusion_ineligible")
    return out


def judgment_rejects_essential_fact_fusion(judgment: Mapping[str, Any]) -> bool:
    assessment = judgment.get("essentialFactFusionAssessment")
    if not isinstance(assessment, dict):
        return False
    if assessment.get("advantageVisualizedWithoutProductApplication") is True:
        return True
    if assessment.get("unselectedFactIntroduced") is True:
        return True
    if assessment.get("fusionEligible") is False:
        return True
    if assessment.get("fusionRequired") is True:
        if assessment.get("productCategoryAppliedInVisualMechanism") is False:
            return True
        if assessment.get("factsIntegratedIntoOneMechanism") is False:
            return True
    return False


def collect_creator_fusion_structural_errors(
    *,
    strategy_foundation: Dict[str, Any],
    candidate: Mapping[str, Any],
    compatibility_mode: bool = False,
) -> List[str]:
    if compatibility_mode:
        return []
    brief = get_product_semantic_brief(strategy_foundation)
    if not fusion_required_for_brief(brief):
        return []
    errors: List[str] = []
    report = candidate.get("creatorReport") if isinstance(candidate.get("creatorReport"), dict) else {}
    evidence = report.get("essentialFactFusionEvidence")
    if not isinstance(evidence, dict):
        errors.append("builder2_creator_validation_failed:creatorReport.essentialFactFusionEvidence")
        return errors
    for field in CREATOR_FUSION_EVIDENCE_FIELDS:
        if field not in evidence:
            errors.append(f"builder2_creator_validation_failed:creatorReport.essentialFactFusionEvidence.{field}")
    if evidence.get("usedFactOutsideSelectedBrief") is True:
        errors.append("builder2_creator_validation_failed:creatorReport.essentialFactFusionEvidence.usedFactOutsideSelectedBrief")
    return list(dict.fromkeys(errors))


def collect_judge_fusion_structural_errors(
    *,
    strategy_foundation: Dict[str, Any],
    judgment: Mapping[str, Any],
    compatibility_mode: bool = False,
) -> List[str]:
    if compatibility_mode:
        return []
    brief = get_product_semantic_brief(strategy_foundation)
    if not fusion_required_for_brief(brief):
        return []
    errors: List[str] = []
    assessment = judgment.get("essentialFactFusionAssessment")
    if not isinstance(assessment, dict):
        errors.append("builder2_judge_validation_failed:essentialFactFusionAssessment")
        return errors
    for field in JUDGE_FUSION_ASSESSMENT_FIELDS:
        if field not in assessment:
            errors.append(f"builder2_judge_validation_failed:essentialFactFusionAssessment.{field}")
    if not str(assessment.get("notes") or "").strip():
        errors.append("builder2_judge_validation_failed:essentialFactFusionAssessment.notes")
    if judgment_rejects_essential_fact_fusion(judgment):
        errors.append("builder2_judge_validation_failed:essentialFactFusionAssessment.ineligible")
    return list(dict.fromkeys(errors))


def collect_fusion_structural_errors(
    *,
    strategy_foundation: Dict[str, Any],
    candidate: Mapping[str, Any],
    judgment: Mapping[str, Any],
    compatibility_mode: bool = False,
) -> List[str]:
    errors = collect_creator_fusion_structural_errors(
        strategy_foundation=strategy_foundation,
        candidate=candidate,
        compatibility_mode=compatibility_mode,
    )
    errors.extend(
        collect_judge_fusion_structural_errors(
            strategy_foundation=strategy_foundation,
            judgment=judgment,
            compatibility_mode=compatibility_mode,
        )
    )
    return list(dict.fromkeys(errors))
