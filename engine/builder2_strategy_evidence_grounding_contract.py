"""
Builder2 Strategy evidence grounding — evidence-bounded Strategy, Creator, Judge, Winner contract.

Distinguishes explicit product facts, safe strategic interpretation, category convention,
and unsupported product claims. Applies to new Strategy generation; legacy jobs remain
inspectable under compatibility mode.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError

BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION = "builder2_strategy_evidence_grounding_v1"

VALID_PRODUCT_MARKET_STATUSES = frozenset(
    {
        "existing_verified",
        "existing_limited_information",
        "prelaunch",
        "concept_stage",
        "unknown",
    }
)
VALID_PRODUCT_INFORMATION_DENSITIES = frozenset({"sufficient", "sparse", "minimal"})
VALID_RELATIVE_ADVANTAGE_INFERENCE_LEVELS = frozenset(
    {"explicit", "direct_derivation", "category_context_only", "speculative", "insufficient_evidence"}
)
VALID_RELATIVE_ADVANTAGE_TYPES = frozenset(
    {
        "comparative_verified",
        "differentiated_verified",
        "functional_direct",
        "access_advantage",
        "explicit_novelty",
        "insufficient_evidence",
    }
)
CANONICAL_RELATIVE_ADVANTAGE_INFERENCE_LEVELS = frozenset({"explicit", "direct_derivation"})

DISPUTED_CAPABILITY_KEYS: Tuple[str, ...] = (
    "feedback",
    "revision",
    "optimization",
    "performance_learning",
    "improvement_loop",
    "campaign_measurement",
    "collaborative_iteration",
)

_CAPABILITY_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "feedback": (
        re.compile(r"\bfeedback\b", re.I),
        re.compile(r"\buser\s+feedback\b", re.I),
        re.compile(r"משוב"),
        re.compile(r"פיד\s*בק"),
        re.compile(r"קבל(?:ת|ים)?\s+משוב"),
    ),
    "revision": (
        re.compile(r"\brevis(?:e|es|ed|ing|ion|ions)\b", re.I),
        re.compile(r"\b(?:re)?draft(?:s|ed|ing)?\b", re.I),
        re.compile(r"עריכ(?:ה|ות)\s+מחדש"),
        re.compile(r"גרס(?:ה|ות)\s+(?:מעודכנ(?:ת|ים)|שני(?:ה|ות)|משופר(?:ת|ים))"),
        re.compile(r"\bround\s+of\s+changes\b", re.I),
    ),
    "optimization": (
        re.compile(r"\boptimi[sz](?:e|es|ed|ing|ation|ations)\b", re.I),
        re.compile(r"אופטימ"),
        re.compile(r"שיפור\s+(?:מתמשך|שוטף|לאורך\s+זמן|על\s+בסיס)"),
        re.compile(r"\btune(?:s|d|ing)?\s+(?:the\s+)?(?:ad|advertisement|campaign)\b", re.I),
    ),
    "performance_learning": (
        re.compile(r"\blearn(?:s|ed|ing)?\s+from\s+(?:campaign|performance|results|feedback)\b", re.I),
        re.compile(r"\blearns?\s+from\s+results\b", re.I),
        re.compile(r"לומד\s+מ(?:ה)?(?:תוצאות|ביצועים|משוב)"),
        re.compile(r"\badaptive\s+learning\b", re.I),
    ),
    "improvement_loop": (
        re.compile(r"\bimprov(?:e|es|ed|ing)\s+(?:over\s+time|with|based\s+on|after)\b", re.I),
        re.compile(r"\bcontinuous(?:ly)?\s+improv", re.I),
        re.compile(r"משתפר\s+(?:עם|מ|לאורך)"),
        re.compile(r"\brefine(?:s|d|ment)?\s+(?:the\s+)?(?:ad|advertisement)\b", re.I),
        re.compile(r"\badapt(?:s|ed|ing)?\s+(?:the\s+)?(?:ad|advertisement)\b", re.I),
    ),
    "campaign_measurement": (
        re.compile(r"\bcampaign\s+(?:performance|results|metrics|measurement)\b", re.I),
        re.compile(r"\bmeasure(?:s|d|ing)?\s+(?:campaign|ad)\s+performance\b", re.I),
        re.compile(r"מדיד(?:ת|ה)\s+(?:קמפיין|ביצועים)"),
    ),
    "collaborative_iteration": (
        re.compile(
            r"\b(?:work(?:s|ing)?\s+with\s+(?:the\s+)?(?:client|user)\s+(?:to|on)\s+(?:revise|refine|improve))\b",
            re.I,
        ),
        re.compile(r"\bongoing\s+(?:account|client)\s+management\b", re.I),
        re.compile(r"ניהול\s+לקוח\s+שוטף"),
        re.compile(r"ליווי\s+שוטף"),
    ),
}

CAPABILITY_OCCURRENCE_POSITIVE_ASSERTION = "positive_assertion"
CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION = "explicit_negation"
CAPABILITY_OCCURRENCE_UNCERTAINTY = "uncertainty_or_no_evidence"
CAPABILITY_OCCURRENCE_CATEGORY_CONVENTION = "category_convention_only"
CAPABILITY_OCCURRENCE_VISUAL_METAPHOR = "visual_metaphor"
CAPABILITY_OCCURRENCE_UNKNOWN = "unknown"

_NEGATION_WINDOW_CHARS = 120
_SCOPE_BOUNDARY = re.compile(r"[.!?;]\s*")
_CONTRAST_RESET = re.compile(r"(?:^|[.!?;]\s*|\s+)(?:אך|אולם|אבל|however|but|instead)\s+", re.I)
_COORDINATED_ATTRIBUTION_NEGATION = re.compile(r"אינ[והםן]\s+מייחס")
_POSITIVE_ACTION_BEFORE_MATCH = re.compile(
    r"(?:"
    r"מבצע|משפר(?:ת)?|כולל(?:ת)?|מספק(?:ת)?|מנהל(?:ת)?|לומד(?:ת)?|"
    r"provides|performs|includes|improves|optimizes|learns|accepts|delivers"
    r")\b[\s\S]{0,48}$",
    re.I,
)
_NEGATION_PREFIX_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"אינ[והםן]\s+מייחס\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"אינ[והםן]\s+מבטיח\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"לא\s+נטען\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"לא\s+כולל\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"(?:^|[\s,])בלי\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"(?:^|[\s,])ללא\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"\bwithout\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"\bdo\s+not\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"\bdoes\s+not\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"\bdon't\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"(?:^|[\s,])לא\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
    (re.compile(r"(?:^|[\s,])אין\b"), CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION),
)
_UNCERTAINTY_PREFIX_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"אין\s+מידע\s+שמוכיח"), CAPABILITY_OCCURRENCE_UNCERTAINTY),
    (re.compile(r"no\s+evidence\s+(?:that|to)\b", re.I), CAPABILITY_OCCURRENCE_UNCERTAINTY),
)

_GENERIC_UNGROUNDED_ADVANTAGE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsmarter\s+advertis", re.I),
    re.compile(r"\bbetter\s+results\b", re.I),
    re.compile(r"\bcontinuous\s+optimization\b", re.I),
    re.compile(r"\badvertising\s+that\s+learns\b", re.I),
    re.compile(r"\bshaped\s+by\s+feedback\b", re.I),
)

_CONSERVATIVE_MARKET_STATUSES = frozenset({"prelaunch", "concept_stage", "unknown"})
_CONSERVATIVE_DENSITIES = frozenset({"sparse", "minimal"})

_CREATOR_TEXT_FIELDS: Tuple[str, ...] = (
    "coreCreativeMechanism",
    "visualMechanism",
    "conceptSummary",
    "creatorReport.problemPerception",
    "creatorReport.relativeAdvantage",
    "creatorReport.mechanismScanSummary",
    "creatorReport.whyParallelExpressesAdvantage",
    "advertisingClosure.sloganText",
    "advertisingSloganFormulation.finalSloganText",
    "advertisingSloganFormulation.whyThisIsAdvertisingCopy",
    "sevenSecondStructure.beginning",
    "sevenSecondStructure.development",
    "sevenSecondStructure.resolution",
    "visualAnchor.description",
    "visualAnchor.whyEssential",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _raise(code: str, *, field: str) -> None:
    raise Builder2TournamentError(f"{code}:{field}")


def contract_version(*, state: Optional[Dict[str, Any]] = None, strategy: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(strategy, dict):
        block = strategy.get("strategyEvidenceGrounding")
        if isinstance(block, dict) and _clean(block.get("contractVersion")):
            return _clean(block.get("contractVersion"))
    if isinstance(state, dict):
        version = _clean(state.get("strategyEvidenceGroundingContractVersion"))
        if version:
            return version
    return ""


def requires_strategy_evidence_grounding(
    *,
    state: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> bool:
    if compatibility_mode:
        return False
    return contract_version(state=state, strategy=strategy) == BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION


def _input_blob(*, product_name: str, product_description: str, target_audience: str = "") -> str:
    return " ".join(part for part in (product_name, product_description, target_audience) if part)


def detect_capabilities_in_text(text: str) -> List[str]:
    blob = _clean(text)
    if not blob:
        return []
    found: List[str] = []
    for key, patterns in _CAPABILITY_PATTERNS.items():
        if any(pattern.search(blob) for pattern in patterns):
            found.append(key)
    return found


def derive_product_information_status(
    *,
    product_name: str,
    product_description: str,
    target_audience: str = "",
) -> Dict[str, str]:
    blob = _input_blob(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    ).lower()
    market_status = "unknown"
    if re.search(r"\b(pre[- ]?launch|not yet (?:launched|available|on the market)|coming soon|beta)\b", blob, re.I):
        market_status = "prelaunch"
    elif re.search(r"\b(concept|prototype|in development|not available yet)\b", blob, re.I):
        market_status = "concept_stage"
    elif re.search(r"\b(launched|available now|on the market|in production|live product)\b", blob, re.I):
        market_status = "existing_verified"
    elif len(blob) > 240:
        market_status = "existing_limited_information"

    word_count = len([part for part in re.split(r"\s+", blob) if part])
    if word_count <= 18:
        density = "minimal"
    elif word_count <= 60:
        density = "sparse"
    else:
        density = "sufficient"
    return {
        "productMarketStatus": market_status,
        "productInformationDensity": density,
    }


def build_product_input_audit(
    *,
    product_name: str,
    product_description: str,
    target_audience: str = "",
) -> Dict[str, Any]:
    blob = _input_blob(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    )
    explicit_capabilities = detect_capabilities_in_text(blob)
    status = derive_product_information_status(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    )
    explicit_claims = [
        line.strip()
        for line in re.split(r"[\n.;]+", product_description)
        if line.strip()
    ]
    return {
        "productName": _clean(product_name),
        "productDescription": _clean(product_description),
        "targetAudience": _clean(target_audience),
        "explicitCapabilitiesSupplied": explicit_capabilities,
        "explicitClaimsSupplied": explicit_claims,
        "launchStatusSupplied": status["productMarketStatus"] != "unknown",
        "feedbackCapabilitySupplied": "feedback" in explicit_capabilities,
        "revisionCapabilitySupplied": "revision" in explicit_capabilities,
        "optimizationCapabilitySupplied": "optimization" in explicit_capabilities,
        "performanceLearningCapabilitySupplied": "performance_learning" in explicit_capabilities,
        "improvementLoopCapabilitySupplied": "improvement_loop" in explicit_capabilities,
        **status,
    }


def build_explicit_product_facts(
    *,
    product_name: str,
    product_description: str,
    target_audience: str = "",
) -> List[str]:
    facts: List[str] = []
    if _clean(product_name):
        facts.append(f"Product name: {_clean(product_name)}")
    for claim in build_product_input_audit(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    ).get("explicitClaimsSupplied") or []:
        facts.append(claim)
    if _clean(target_audience):
        facts.append(f"Target audience: {_clean(target_audience)}")
    blob = _input_blob(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    ).lower()
    if re.search(r"\bai\b|artificial intelligence|בינה\s+מלאכותית", blob, re.I):
        facts.append("The product is an AI application.")
    if re.search(r"\b(create|creates|creating|generate|generates|generating)\b.*\b(ad|advertis|campaign|marketing)", blob, re.I):
        facts.append("The product creates advertising or advertising ideas.")
    if re.search(r"(?:יוצר|ייצר|מייצר|מייצרת).*(?:פרסום|פרסומ|מודע)", blob):
        facts.append("The product creates advertising or advertising ideas.")
    return list(dict.fromkeys(fact for fact in facts if fact))


def _collect_strategy_text(strategy: Dict[str, Any]) -> List[Tuple[str, str]]:
    texts: List[Tuple[str, str]] = []
    pp = strategy.get("problemPerception") if isinstance(strategy.get("problemPerception"), dict) else {}
    ra = strategy.get("relativeAdvantage") if isinstance(strategy.get("relativeAdvantage"), dict) else {}
    ms = strategy.get("mechanismScan") if isinstance(strategy.get("mechanismScan"), dict) else {}
    for field, value in (
        ("problemPerception.statement", pp.get("statement")),
        ("problemPerception.whyItMatters", pp.get("whyItMatters")),
        ("relativeAdvantage.statement", ra.get("statement")),
        ("relativeAdvantage.derivationFromProblem", ra.get("derivationFromProblem")),
        ("relativeAdvantage.truthBoundary", ra.get("truthBoundary")),
        ("mechanismScan.discoveredMechanism", ms.get("discoveredMechanism")),
        ("mechanismScan.creativeOpportunity", ms.get("creativeOpportunity")),
        ("mechanismScan.depthEvidence", ms.get("depthEvidence")),
    ):
        text = _clean(value)
        if text:
            texts.append((field, text))
    for idx, fact in enumerate(ms.get("domainFacts") or []):
        text = _clean(fact)
        if text:
            texts.append((f"mechanismScan.domainFacts[{idx}]", text))
    for idx, item in enumerate(pp.get("groundingEvidence") or []):
        text = _clean(item)
        if text:
            texts.append((f"problemPerception.groundingEvidence[{idx}]", text))
    return texts


def _absolute_clause_bounds(text: str, pos: int) -> Tuple[int, int]:
    start = 0
    for match in _SCOPE_BOUNDARY.finditer(text[:pos]):
        start = match.end()
    end = len(text)
    for match in _SCOPE_BOUNDARY.finditer(text[pos:]):
        end = pos + match.start()
        break
    return start, end


def _last_scope_reset_pos(prefix: str) -> int:
    last = -1
    for match in _SCOPE_BOUNDARY.finditer(prefix):
        last = match.end() - 1
    for match in _CONTRAST_RESET.finditer(prefix):
        if match.start() > last:
            last = match.end() - 1
    return last


def _iter_capability_matches(text: str, capability: str) -> List[Tuple[int, int, str]]:
    patterns = _CAPABILITY_PATTERNS.get(capability) or ()
    matches: List[Tuple[int, int, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), match.group(0)))
    matches.sort(key=lambda item: item[0])
    return matches


def _is_coordinated_attribution_negation(clause: str, rel_pos: int) -> bool:
    for match in _COORDINATED_ATTRIBUTION_NEGATION.finditer(clause):
        list_start = match.end()
        if rel_pos < list_start:
            continue
        between = clause[list_start:rel_pos]
        if _CONTRAST_RESET.search(between):
            continue
        if _SCOPE_BOUNDARY.search(between):
            continue
        return True
    return False


def classify_capability_occurrence(
    text: str,
    capability: str,
    *,
    match_start: int,
    match_end: int,
    field_path: str = "",
) -> str:
    del field_path, match_end
    clause_start, clause_end = _absolute_clause_bounds(text, match_start)
    clause = text[clause_start:clause_end]
    rel_start = match_start - clause_start
    prefix = clause[:rel_start]
    reset_pos = _last_scope_reset_pos(prefix)
    governed_prefix = prefix[reset_pos + 1 :] if reset_pos >= 0 else prefix

    if _is_coordinated_attribution_negation(clause, rel_start):
        return CAPABILITY_OCCURRENCE_EXPLICIT_NEGATION

    for pattern, classification in _UNCERTAINTY_PREFIX_PATTERNS:
        if pattern.search(governed_prefix):
            return classification

    for pattern, classification in _NEGATION_PREFIX_PATTERNS:
        if pattern.search(governed_prefix):
            return classification

    if _POSITIVE_ACTION_BEFORE_MATCH.search(governed_prefix):
        return CAPABILITY_OCCURRENCE_POSITIVE_ASSERTION

    return CAPABILITY_OCCURRENCE_POSITIVE_ASSERTION


def has_positive_capability_claim(
    text: str,
    capability: str,
    *,
    field_path: str = "",
) -> bool:
    if capability not in detect_capabilities_in_text(text):
        return False
    for start, end, _matched in _iter_capability_matches(text, capability):
        if (
            classify_capability_occurrence(
                text,
                capability,
                match_start=start,
                match_end=end,
                field_path=field_path,
            )
            == CAPABILITY_OCCURRENCE_POSITIVE_ASSERTION
        ):
            return True
    return False


def scan_capability_occurrences(
    text: str,
    *,
    allowed_capabilities: Sequence[str],
    field_path: str = "",
) -> List[Dict[str, Any]]:
    allowed = set(allowed_capabilities)
    occurrences: List[Dict[str, Any]] = []
    for capability in DISPUTED_CAPABILITY_KEYS:
        if capability in allowed:
            continue
        for start, end, matched in _iter_capability_matches(text, capability):
            classification = classify_capability_occurrence(
                text,
                capability,
                match_start=start,
                match_end=end,
                field_path=field_path,
            )
            occurrences.append(
                {
                    "fieldPath": field_path,
                    "capability": capability,
                    "matchedText": text,
                    "matchedSpan": matched,
                    "matchStart": start,
                    "matchEnd": end,
                    "occurrenceClassification": classification,
                    "productClaimEmitted": classification == CAPABILITY_OCCURRENCE_POSITIVE_ASSERTION,
                }
            )
    return occurrences


def find_unsupported_capability_claims(
    text: str,
    *,
    allowed_capabilities: Sequence[str],
    field_path: str = "",
) -> List[str]:
    allowed = set(allowed_capabilities)
    claims: List[str] = []
    for capability in detect_capabilities_in_text(text):
        if capability in allowed:
            continue
        if has_positive_capability_claim(text, capability, field_path=field_path):
            claims.append(capability)
    return list(dict.fromkeys(claims))


def scan_texts_for_unsupported_capabilities(
    texts: Sequence[Tuple[str, str]],
    *,
    allowed_capabilities: Sequence[str],
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for field_path, text in texts:
        unsupported = find_unsupported_capability_claims(
            text,
            allowed_capabilities=allowed_capabilities,
            field_path=field_path,
        )
        for capability in unsupported:
            positive_matches = [
                item
                for item in scan_capability_occurrences(
                    text,
                    allowed_capabilities=allowed_capabilities,
                    field_path=field_path,
                )
                if item["capability"] == capability and item["productClaimEmitted"]
            ]
            primary = positive_matches[0] if positive_matches else {}
            hits.append(
                {
                    "fieldPath": field_path,
                    "capability": capability,
                    "matchedText": text,
                    "matchedSpan": primary.get("matchedSpan"),
                    "occurrenceClassification": primary.get("occurrenceClassification")
                    or CAPABILITY_OCCURRENCE_POSITIVE_ASSERTION,
                    "productClaimEmitted": True,
                }
            )
    return hits


def apply_strategy_evidence_grounding(
    strategy: Dict[str, Any],
    *,
    product_name: str,
    product_description: str,
    target_audience: str = "",
) -> Dict[str, Any]:
    out = dict(strategy)
    audit = build_product_input_audit(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    )
    explicit_facts = build_explicit_product_facts(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    )
    allowed_capabilities = list(audit.get("explicitCapabilitiesSupplied") or [])
    unsupported_hits = scan_texts_for_unsupported_capabilities(
        _collect_strategy_text(out),
        allowed_capabilities=allowed_capabilities,
    )
    category_conventions: List[str] = []
    if not allowed_capabilities:
        category_conventions.extend(
            [
                "Advertising agencies often offer revision rounds.",
                "Campaign performance is commonly measured after launch.",
                "Creative work may iterate collaboratively with clients.",
            ]
        )
    ra = dict(out.get("relativeAdvantage") or {})
    inference_level = _clean(ra.get("relativeAdvantageInferenceLevel")) or "direct_derivation"
    if unsupported_hits:
        inference_level = "speculative"
    elif audit.get("productInformationDensity") in _CONSERVATIVE_DENSITIES or audit.get("productMarketStatus") in _CONSERVATIVE_MARKET_STATUSES:
        if inference_level not in CANONICAL_RELATIVE_ADVANTAGE_INFERENCE_LEVELS:
            inference_level = "direct_derivation"
    ra.setdefault("relativeAdvantageType", "functional_direct")
    ra.setdefault("relativeAdvantageEvidence", explicit_facts[:3])
    ra.setdefault("relativeAdvantageEvidenceSourcePaths", ["product_input"])
    ra.setdefault("relativeAdvantageInferenceLevel", inference_level)
    ra.setdefault("categoryConventionDependencies", category_conventions)
    ra.setdefault("unsupportedAssumptions", [hit["capability"] for hit in unsupported_hits])
    ra.setdefault("relativeAdvantageFactuallyGrounded", not unsupported_hits)
    out["relativeAdvantage"] = ra
    out["strategyEvidenceGrounding"] = {
        "contractVersion": BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION,
        "productMarketStatus": audit["productMarketStatus"],
        "productInformationDensity": audit["productInformationDensity"],
        "explicitProductFacts": explicit_facts,
        "safeStrategicInterpretations": [],
        "categoryConventions": category_conventions,
        "unsupportedAssumptions": [hit["capability"] for hit in unsupported_hits],
        "allowedCapabilities": allowed_capabilities,
        "productInputAudit": audit,
    }
    return out


def validate_strategy_evidence_grounding(
    strategy: Dict[str, Any],
    *,
    product_name: str = "",
    product_description: str = "",
    target_audience: str = "",
    compatibility_mode: bool = False,
) -> None:
    if compatibility_mode:
        return
    block = strategy.get("strategyEvidenceGrounding")
    if not isinstance(block, dict) or _clean(block.get("contractVersion")) != BUILDER2_STRATEGY_EVIDENCE_GROUNDING_CONTRACT_VERSION:
        _raise("builder2_strategy_validation_failed", field="strategyEvidenceGrounding.contractVersion")

    audit = build_product_input_audit(
        product_name=product_name or _clean(strategy.get("productNameResolved")),
        product_description=product_description,
        target_audience=target_audience,
    )
    allowed_capabilities = list(audit.get("explicitCapabilitiesSupplied") or [])
    hits = scan_texts_for_unsupported_capabilities(
        _collect_strategy_text(strategy),
        allowed_capabilities=allowed_capabilities,
    )
    if hits:
        _raise("builder2_strategy_validation_failed", field=f"{hits[0]['fieldPath']}.unsupported_product_claim")

    ra = strategy.get("relativeAdvantage") if isinstance(strategy.get("relativeAdvantage"), dict) else {}
    inference = _clean(ra.get("relativeAdvantageInferenceLevel"))
    if inference not in CANONICAL_RELATIVE_ADVANTAGE_INFERENCE_LEVELS:
        _raise("builder2_strategy_validation_failed", field="relativeAdvantage.relativeAdvantageInferenceLevel")
    if ra.get("relativeAdvantageFactuallyGrounded") is not True:
        _raise("builder2_strategy_validation_failed", field="relativeAdvantage.relativeAdvantageFactuallyGrounded")

    statement = _clean(ra.get("statement"))
    for pattern in _GENERIC_UNGROUNDED_ADVANTAGE_PATTERNS:
        if pattern.search(statement):
            _raise("builder2_strategy_validation_failed", field="relativeAdvantage.statement.ungrounded_generic_claim")


def _get_nested_text(candidate: Dict[str, Any], field_path: str) -> str:
    current: Any = candidate
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
    return _clean(current)


def collect_creator_text_fields(candidate: Dict[str, Any]) -> List[Tuple[str, str]]:
    texts: List[Tuple[str, str]] = []
    for field_path in _CREATOR_TEXT_FIELDS:
        text = _get_nested_text(candidate, field_path)
        if text:
            texts.append((field_path, text))
    return texts


def stamp_creator_evidence_inheritance(
    candidate: Dict[str, Any],
    *,
    strategy_foundation: Dict[str, Any],
) -> None:
    block = strategy_foundation.get("strategyEvidenceGrounding")
    if not isinstance(block, dict):
        return
    ra = strategy_foundation.get("relativeAdvantage") if isinstance(strategy_foundation.get("relativeAdvantage"), dict) else {}
    candidate["inheritedProductFacts"] = list(block.get("explicitProductFacts") or [])
    candidate["inheritedRelativeAdvantageEvidence"] = list(ra.get("relativeAdvantageEvidence") or [])
    unsupported = scan_texts_for_unsupported_capabilities(
        collect_creator_text_fields(candidate),
        allowed_capabilities=block.get("allowedCapabilities") or [],
    )
    introduced = sorted({hit["capability"] for hit in unsupported})
    candidate["newProductClaimsIntroduced"] = introduced
    candidate["creatorFactuallyGrounded"] = not introduced


def validate_creator_evidence_grounding(
    candidate: Dict[str, Any],
    *,
    strategy_foundation: Dict[str, Any],
    compatibility_mode: bool = False,
) -> None:
    if not requires_strategy_evidence_grounding(strategy=strategy_foundation, compatibility_mode=compatibility_mode):
        return
    stamp_creator_evidence_inheritance(candidate, strategy_foundation=strategy_foundation)
    introduced = candidate.get("newProductClaimsIntroduced")
    if isinstance(introduced, list) and introduced:
        _raise("builder2_creator_validation_failed", field="newProductClaimsIntroduced")


def build_default_judge_factual_grounding_assessment(*, notes: str = "") -> Dict[str, Any]:
    return {
        "productClaimFactuallyGrounded": True,
        "noUnsupportedFeatureClaim": True,
        "noCategoryConventionPresentedAsProductFact": True,
        "viewerWouldNotInferUnsupportedCapability": True,
        "relativeAdvantageEvidenceAccepted": True,
        "comparedAgainstOriginalProductInput": True,
        "notes": notes or "Candidate product claims remain within the supplied product evidence.",
    }


JUDGE_FACTUAL_GROUNDING_GATE_FIELDS: Tuple[str, ...] = (
    "productClaimFactuallyGrounded",
    "noUnsupportedFeatureClaim",
    "noCategoryConventionPresentedAsProductFact",
    "viewerWouldNotInferUnsupportedCapability",
    "relativeAdvantageEvidenceAccepted",
)


def collect_judge_factual_grounding_structural_errors(
    judgment: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> List[str]:
    if not requires_strategy_evidence_grounding(strategy=strategy_foundation, compatibility_mode=compatibility_mode):
        return []
    errors: List[str] = []
    assessment = judgment.get("factualGroundingAssessment")
    if not isinstance(assessment, dict):
        errors.append("builder2_judge_validation_failed:factualGroundingAssessment")
        return errors
    if not assessment:
        errors.append("builder2_judge_validation_failed:factualGroundingAssessment")
        return errors
    for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS:
        if not isinstance(assessment.get(key), bool):
            errors.append(f"builder2_judge_validation_failed:factualGroundingAssessment.{key}")
    if not str(assessment.get("notes") or "").strip():
        errors.append("builder2_judge_validation_failed:factualGroundingAssessment.notes")
    return list(dict.fromkeys(errors))


def validate_judge_factual_grounding_structure(
    judgment: Dict[str, Any],
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> None:
    errors = collect_judge_factual_grounding_structural_errors(
        judgment,
        strategy_foundation=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    if errors:
        first = errors[0]
        if ":" in first:
            code, field = first.split(":", 1)
            _raise(code, field=field)
        _raise("builder2_judge_validation_failed", field=first)


def collect_failed_factual_grounding_gates(assessment: Dict[str, Any]) -> List[str]:
    return [key for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS if assessment.get(key) is not True]


def apply_judge_factual_grounding_server_corrections(
    judgment: Dict[str, Any],
    *,
    candidate: Optional[Dict[str, Any]] = None,
    strategy_foundation: Optional[Dict[str, Any]] = None,
) -> None:
    if not isinstance(strategy_foundation, dict) or not requires_strategy_evidence_grounding(strategy=strategy_foundation):
        return
    assessment = judgment.get("factualGroundingAssessment")
    if not isinstance(assessment, dict):
        return
    introduced: List[str] = []
    if candidate:
        stamp_creator_evidence_inheritance(candidate, strategy_foundation=strategy_foundation)
        introduced = list(candidate.get("newProductClaimsIntroduced") or [])
    if introduced:
        if assessment.get("productClaimFactuallyGrounded") is True:
            assessment["productClaimFactuallyGrounded"] = False
        if assessment.get("noUnsupportedFeatureClaim") is True:
            assessment["noUnsupportedFeatureClaim"] = False
        judgment["eligible"] = False
        judgment["eligibilityFailureReason"] = "builder2_judge_factual_grounding_failed"
        judgment["factualGroundingFailedGates"] = sorted(
            set(collect_failed_factual_grounding_gates(assessment))
            | {"productClaimFactuallyGrounded", "noUnsupportedFeatureClaim"}
        )
        disqualifiers = list(judgment.get("disqualifiers") or [])
        reason = "unsupported_product_capability"
        if reason not in disqualifiers:
            disqualifiers.append(reason)
        judgment["disqualifiers"] = disqualifiers
    apply_factual_grounding_eligibility_rules(judgment)
    failed = collect_failed_factual_grounding_gates(assessment)
    if failed:
        judgment["factualGroundingFailedGates"] = sorted(set(judgment.get("factualGroundingFailedGates") or []) | set(failed))
        if judgment.get("eligible") is True:
            judgment["eligible"] = False
            judgment["eligibilityFailureReason"] = "builder2_judge_factual_grounding_failed"


def validate_judge_factual_grounding_assessment(
    judgment: Dict[str, Any],
    *,
    candidate: Optional[Dict[str, Any]] = None,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    product_input: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> None:
    if not requires_strategy_evidence_grounding(strategy=strategy_foundation, compatibility_mode=compatibility_mode):
        return
    validate_judge_factual_grounding_structure(
        judgment,
        strategy_foundation=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    apply_judge_factual_grounding_server_corrections(
        judgment,
        candidate=candidate,
        strategy_foundation=strategy_foundation,
    )


def apply_factual_grounding_eligibility_rules(judgment: Dict[str, Any]) -> None:
    assessment = judgment.get("factualGroundingAssessment")
    if not isinstance(assessment, dict):
        return
    if judgment.get("eligible") is not True:
        return
    required = (
        "productClaimFactuallyGrounded",
        "noUnsupportedFeatureClaim",
        "noCategoryConventionPresentedAsProductFact",
        "viewerWouldNotInferUnsupportedCapability",
        "relativeAdvantageEvidenceAccepted",
    )
    if any(assessment.get(key) is not True for key in required):
        judgment["eligible"] = False
        judgment["eligibilityFailureReason"] = "builder2_judge_factual_grounding_failed"


def validate_winner_evidence_grounding(
    winner_plan: Dict[str, Any],
    *,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    product_input: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> None:
    if not requires_strategy_evidence_grounding(strategy=strategy_foundation, compatibility_mode=compatibility_mode):
        return
    block = strategy_foundation.get("strategyEvidenceGrounding")
    if not isinstance(block, dict):
        return
    allowed = block.get("allowedCapabilities") or []
    texts: List[Tuple[str, str]] = []
    for field_path in (
        "problemPerception",
        "relativeAdvantage",
        "coreCreativeMechanism",
        "advertisingClosure.sloganText",
        "advertisingSloganEvidence.finalSloganText",
        "advertisingSloganEvidence.whyAdvertising",
    ):
        text = _get_nested_text(winner_plan, field_path)
        if text:
            texts.append((field_path, text))
    hits = scan_texts_for_unsupported_capabilities(texts, allowed_capabilities=allowed)
    if hits:
        _raise("builder2_winner_validation_failed", field=f"{hits[0]['fieldPath']}.unsupported_product_claim")
    stamp_creator_evidence_inheritance(winning_candidate, strategy_foundation=strategy_foundation)
    introduced = winning_candidate.get("newProductClaimsIntroduced") or []
    if introduced:
        _raise("builder2_winner_validation_failed", field="winningCandidate.newProductClaimsIntroduced")


def strategy_fingerprint(strategy: Dict[str, Any]) -> str:
    payload = json.dumps(strategy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_creator_relative_advantage_to_strategy(
    candidate: Dict[str, Any],
    *,
    strategy_foundation: Dict[str, Any],
) -> str:
    report = candidate.get("creatorReport") if isinstance(candidate.get("creatorReport"), dict) else {}
    strategy_pp = _clean((strategy_foundation.get("problemPerception") or {}).get("statement"))
    strategy_ra = _clean((strategy_foundation.get("relativeAdvantage") or {}).get("statement"))
    creator_pp = _clean(report.get("problemPerception"))
    creator_ra = _clean(report.get("relativeAdvantage"))
    if creator_pp == strategy_pp and creator_ra == strategy_ra:
        return "identical_to_strategy"
    if creator_pp and creator_ra:
        return "semantically_inherited"
    if creator_pp or creator_ra:
        return "materially_changed"
    return "independently_invented"


def inspect_disputed_capability_introduction(
    *,
    product_input: Dict[str, Any],
    strategy: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    allowed = list(product_input.get("explicitCapabilitiesSupplied") or [])
    stages = ["product_input", "strategy", "creator", "judge", "winner", "not_found"]
    introduced_at = "not_found"
    for stage_name, texts in (
        ("strategy", _collect_strategy_text(strategy)),
        (
            "creator",
            [
                item
                for candidate in candidates
                for item in collect_creator_text_fields(candidate if isinstance(candidate, dict) else {})
            ],
        ),
    ):
        hits = scan_texts_for_unsupported_capabilities(texts, allowed_capabilities=allowed)
        if hits:
            introduced_at = stage_name
            break
    return {
        "disputedCapabilityIntroducedAtStage": introduced_at,
        "disputedCapabilityExplicitlyProvided": bool(allowed),
        "disputedCapabilityCategoryConventionOnly": not bool(allowed) and introduced_at != "not_found",
    }
