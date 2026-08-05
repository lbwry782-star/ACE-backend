"""
Builder2 audience knowledge-gap inspector — read-only audit of preserved model outputs.

Checks whether Strategy, Creator, Judge, or Winner Development explicitly considered
the viewer's prior knowledge gap about pack quantity (e.g. one-of-ten product logic).

Run:
  BUILDER2_AUDIENCE_KNOWLEDGE_GAP_INSPECT_JOB_ID=<jobId> python -m engine.builder2_audience_knowledge_gap_inspect

Optional:
  BUILDER2_AUDIENCE_KNOWLEDGE_GAP_INSPECT_CANDIDATE_ID=<candidateId>
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from engine.builder2_judge_response_ledger import ledger_entries as judge_ledger_entries
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_single_slogan_contract import resolve_canonical_slogan_text
from engine.builder2_strategy_evidence_grounding_contract import build_product_input_audit
from engine.builder2_tournament_store import load_tournament_state
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    load_revalidatable_parsed_winner_response,
)
from engine.video_jobs_redis import redis_configured, video_job_get

logger = logging.getLogger(__name__)

CLASSIFICATION_POSITIVE_GAP = "explicitly_considered_positive_knowledge_gap"
CLASSIFICATION_IMMEDIATE = "explicitly_assumed_immediate_understanding"
CLASSIFICATION_RISK = "explicitly_considered_gap_as_risk"
CLASSIFICATION_MIXED = "mixed_or_ambiguous_evidence"
CLASSIFICATION_NONE = "no_explicit_evidence"

LIMITS = [
    "The inspector examines preserved outputs only.",
    "It does not expose or reconstruct private chain of thought.",
    "Absence of explicit evidence is not proof that the model never considered the issue.",
]

NO_EVIDENCE_REASON = (
    "The stored outputs do not document that the model explicitly considered the audience "
    "knowledge gap. This does not prove that the consideration never occurred; it only "
    "means it was not expressed in the preserved outputs."
)

_VIEWER_TERMS = r"(?:viewer|audience|consumer|spectator|watch(?:er)?|צופ(?:ה|ים)|קהל|צרכ(?:ן|נים)|משתמש(?:ים)?|ציבור)"
_PACKAGE_TERMS = r"(?:pack(?:age)?|box|bundle|carton|אריז(?:ה|ת)|חביל(?:ה|ת)|מארז|עשר(?:ה)?|10|ten|תשע(?:ה)?|9|nine|יחיד(?:ות|ה)?|מסטיק(?:ים)?|sticks?)"
_RISK_TERMS = r"(?:confus(?:e|ed|ion)|misunderstand|ambiguous|unclear|misleading|risk|problem|danger|הטע(?:יה|יה)|בלבול|מעורפ(?:ל|ל)|סכנ(?:ה|ת)|עלול\s+לה|לא\s+יבין|לא\s+יובן)"

_BOOLEAN_ONLY_SUFFIXES = (
    "Accepted",
    "Present",
    "Required",
    "Satisfied",
    "Enabled",
)


@dataclass(frozen=True)
class CategoryPattern:
    category: str
    pattern: re.Pattern[str]
    reason: str
    requires_viewer_context: bool = False
    requires_risk_context: bool = False


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


CATEGORY_PATTERNS: Tuple[CategoryPattern, ...] = (
    CategoryPattern(
        "audiencePriorKnowledgeRecognized",
        _compile(
            rf"(?:{_VIEWER_TERMS}.{{0,80}}(?:does\s+not|do\s+not|cannot|can't|won't|will\s+not|אינ[oa]\s+יודע|לא\s+יודע|לא\s+ידוע\s+ל).{{0,80}}(?:{_PACKAGE_TERMS}))"
            rf"|(?:{_VIEWER_TERMS}.{{0,80}}(?:without|before|prior\s+to|מראש|בלי\s+לראות|ללא\s+ידיע).{{0,80}}(?:{_PACKAGE_TERMS}))"
            rf"|(?:without\s+seeing.{{0,60}}(?:{_PACKAGE_TERMS}))"
            rf"|(?:before\s+(?:seeing|viewing).{{0,60}}(?:{_PACKAGE_TERMS}))"
            rf"|(?:הצופ(?:ה|ים)\s+לא\s+(?:רוא(?:ה|ים)|יודע(?:ים)?).{{0,60}}(?:{_PACKAGE_TERMS}))"
        ),
        "Explicit statement that the viewer lacks prior knowledge of pack quantity or packaging.",
        requires_viewer_context=True,
    ),
    CategoryPattern(
        "missingInformationRecognizedByViewer",
        _compile(
            rf"(?:{_VIEWER_TERMS}.{{0,80}}(?:missing\s+information|information\s+gap|lack(?:s)?\s+(?:the\s+)?fact|does\s+not\s+have\s+the\s+fact))"
            rf"|(?:{_VIEWER_TERMS}.{{0,80}}(?:realiz(?:e|es)|recogniz(?:e|es)|understand(?:s)?).{{0,80}}(?:missing|gap|incomplete|חסר))"
            rf"|(?:(?:why|what).{{0,20}}(?:one|1|אחד).{{0,20}}(?:nine|9|תשע))"
            rf"|(?:שואל(?:ים)?\s+(?:למה|מדוע).{{0,40}}(?:אחד|1|תשע|9))"
            rf"|(?:מזהה(?:ים)?\s+(?:פער|חוסר\s+מידע))"
            rf"|(?:{_VIEWER_TERMS}.{{0,80}}(?:פער|חוסר\s+מידע|מידע\s+חסר))"
        ),
        "Explicit statement that the viewer notices missing product information or an open question.",
        requires_viewer_context=True,
    ),
    CategoryPattern(
        "curiosityGapIntended",
        _compile(
            r"(?:curiosity\s+gap|open\s+loop|teaser|delayed\s+(?:reveal|understanding|comprehension))"
            r"|(?:creat(?:e|es)\s+curiosity|invite(?:s)?\s+curiosity|spark(?:s)?\s+curiosity)"
            r"|(?:סקרנות|לולא(?:ה|ת)\s+פתוח(?:ה|ות)|גילוי\s+מאוחר|delayed\s+discovery)"
            r"|(?:intentionally\s+leave(?:s)?\s+(?:a\s+)?gap|designed\s+to\s+make\s+the\s+viewer\s+wonder)"
        ),
        "Explicit description of an intentional curiosity gap or open loop.",
    ),
    CategoryPattern(
        "laterNaturalResolutionExpected",
        _compile(
            rf"(?:when\s+(?:the\s+)?(?:{_VIEWER_TERMS}|they|he|she).{{0,60}}(?:see(?:s)?|encounter(?:s)?|find(?:s)?).{{0,60}}(?:{_PACKAGE_TERMS}|shelf|store|shop|package|אריז(?:ה|ת)|מדף|חנות))"
            rf"|(?:later\s+(?:when|upon|after).{{0,60}}(?:see(?:s)?|view(?:s)?|encounter(?:s)?).{{0,60}}(?:{_PACKAGE_TERMS}|package|אריז(?:ה|ת)))"
            rf"|(?:כש(?:ה)?(?:צופ(?:ה|ים)|הקהל).{{0,60}}(?:ירא(?:ו|ה)|ייתקל(?:ו|ו)|יגיע(?:ו|ים)).{{0,60}}(?:{_PACKAGE_TERMS}|אריז(?:ה|ת)|מדף|חנות))"
            rf"|(?:הבנה\s+(?:ש)?(?:תושלם|תיגמר|תיסגר)\s+מאוחר\s+יותר)"
        ),
        "Explicit expectation that understanding completes later via packaging or product encounter.",
    ),
    CategoryPattern(
        "immediateUnderstandingAssumed",
        _compile(
            rf"(?:{_VIEWER_TERMS}.{{0,80}}(?:immediately|right\s+away|at\s+once|already|from\s+the\s+start|מיד|כבר|מההתחלה).{{0,80}}(?:understand(?:s)?|know(?:s)?|see(?:s)?|gets?))"
            rf"|(?:{_VIEWER_TERMS}.{{0,80}}(?:see(?:s)?|sees).{{0,40}}(?:all\s+)?(?:ten|10|עשר(?:ה)?).{{0,40}}(?:sticks?|units?|מסטיק(?:ים)?|יחיד(?:ות|ה)?))"
            rf"|(?:{_VIEWER_TERMS}.{{0,80}}(?:already\s+know(?:s)?|already\s+understand(?:s)?).{{0,80}}(?:{_PACKAGE_TERMS}))"
            rf"|(?:הצופ(?:ה|ים)\s+(?:רוא(?:ה|ים)|מבין(?:ים)?)\s+מיד.{{0,60}}(?:{_PACKAGE_TERMS}|עשר(?:ה)?|תשע(?:ה)?))"
        ),
        "Explicit assumption that the viewer immediately sees or understands pack quantity.",
        requires_viewer_context=True,
    ),
    CategoryPattern(
        "alternativeMeaningOrConfusionConsidered",
        _compile(
            rf"(?:without\s+(?:knowing|seeing).{{0,60}}(?:{_PACKAGE_TERMS}|pack\s+size|quantity).{{0,80}}(?:{_RISK_TERMS}|different\s+meaning|another\s+meaning))"
            rf"|(?:if\s+(?:the\s+)?(?:{_VIEWER_TERMS}|they).{{0,60}}(?:do\s+not|does\s+not|don't|doesn't).{{0,60}}(?:{_PACKAGE_TERMS}).{{0,80}}(?:{_RISK_TERMS}|misunderstand|confus))"
            rf"|(?:{_VIEWER_TERMS}.{{0,80}}(?:{_RISK_TERMS}).{{0,80}}(?:slogan|line|copy|סלוגן|משפט))"
            rf"|(?:בלי\s+ידיע(?:ת|ה).{{0,40}}(?:{_PACKAGE_TERMS}|כמות|גודל\s+החבילה).{{0,80}}(?:{_RISK_TERMS}|משמעות\s+אחרת|יוטע(?:ו|ה)))"
        ),
        "Explicit consideration of alternative meaning, ambiguity, or confusion without pack knowledge.",
        requires_risk_context=True,
    ),
    CategoryPattern(
        "visualExplanationReliedUpon",
        _compile(
            rf"(?:visual(?:ly)?|scene|video|on\s+screen|בוויז(?:ואל|ואלי)|בסצ(?:נה|ינה)).{{0,80}}(?:show(?:s)?|teach(?:es)?|explain(?:s)?|reveal(?:s)?|demonstrate(?:s)?).{{0,80}}(?:one|1|אחד).{{0,40}}(?:nine|9|remaining|left|נשאר(?:ים|ו|ה)?|תשע)"
            rf"|(?:(?:one|1|אחד).{{0,40}}(?:taken|removed|picked|נלק(?:ח|חו|חה)).{{0,40}}(?:nine|9|remaining|left|נשאר(?:ים|ו|ה)?|תשע))"
            rf"|(?:(?:ten|10|עשר(?:ה)?).{{0,40}}(?:units?|sticks?|items?|יחיד(?:ות|ה)?|מסטיק(?:ים)?).{{0,40}}(?:visible|shown|seen|נרא(?:ה|ים)|מוצג(?:ים|)))"
            rf"|(?:no\s+need\s+for\s+prior\s+knowledge\s+because\s+(?:the\s+)?visual)"
        ),
        "Explicit claim that the visual itself teaches one-vs-nine or total pack quantity.",
    ),
)

PRIORITY_FIELD_HINTS = (
    "viewer",
    "audience",
    "consumer",
    "curiosity",
    "discovery",
    "package",
    "pack",
    "slogan",
    "closure",
    "notes",
    "why",
    "bridge",
    "problem",
    "advantage",
    "mechanism",
    "sequence",
    "videoPrompt",
    "truthBoundary",
    "unsupportedAssumption",
    "explicitProductFact",
    "grounding",
    "persuasion",
    "silent",
    "participation",
    "formulation",
    "assessment",
    "evidence",
    "headline",
    "scene",
    "visual",
    "צופ",
    "קהל",
    "סקרנ",
    "אריז",
    "חביל",
    "תשע",
    "עשר",
    "מסטיק",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _is_boolean_only_field(field_path: str, text: str) -> bool:
    normalized = _clean(text).lower()
    if normalized in {"true", "false", "1", "0", "yes", "no"}:
        return True
    leaf = field_path.rsplit(".", 1)[-1]
    return any(leaf.endswith(suffix) for suffix in _BOOLEAN_ONLY_SUFFIXES)


def _truncate(text: str, *, limit: int = 320) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _match_categories(text: str) -> List[Tuple[str, str]]:
    blob = _clean(text)
    if not blob or _is_boolean_only_field("", blob):
        return []
    hits: List[Tuple[str, str]] = []
    for spec in CATEGORY_PATTERNS:
        match = spec.pattern.search(blob)
        if not match:
            continue
        if spec.requires_risk_context and not re.search(_RISK_TERMS, blob, re.I):
            continue
        hits.append((spec.category, spec.reason))
    return hits


def _iter_text_nodes(
    value: Any,
    *,
    field_path: str = "",
    stage: str,
    candidate_id: str = "",
    judgment_id: str = "",
    source_type: str,
) -> Iterator[Dict[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{field_path}.{key}" if field_path else str(key)
            yield from _iter_text_nodes(
                child,
                field_path=path,
                stage=stage,
                candidate_id=candidate_id,
                judgment_id=judgment_id,
                source_type=source_type,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{field_path}[{index}]"
            if isinstance(child, str) and _clean(child):
                yield {
                    "stage": stage,
                    "candidateId": candidate_id or None,
                    "judgmentId": judgment_id or None,
                    "fieldPath": path,
                    "text": _clean(child),
                    "sourceType": source_type,
                }
            else:
                yield from _iter_text_nodes(
                    child,
                    field_path=path,
                    stage=stage,
                    candidate_id=candidate_id,
                    judgment_id=judgment_id,
                    source_type=source_type,
                )
        return
    if isinstance(value, str):
        text = _clean(value)
        if not text or _is_boolean_only_field(field_path, text):
            return
        yield {
            "stage": stage,
            "candidateId": candidate_id or None,
            "judgmentId": judgment_id or None,
            "fieldPath": field_path,
            "text": text,
            "sourceType": source_type,
        }


def _empty_evidence_block() -> Dict[str, Any]:
    return {spec.category: {"found": False, "excerpts": []} for spec in CATEGORY_PATTERNS}


def _resolve_winner_candidate_id(state: Dict[str, Any], candidate_id: str = "") -> str:
    explicit = _clean(candidate_id)
    if explicit:
        return explicit
    for key in ("winnerDevelopmentCandidateId", "winnerCandidateId", "provisionalWinnerCandidateId"):
        resolved = _clean(state.get(key))
        if resolved:
            return resolved
    return ""


def _load_product_input(state: Dict[str, Any], job_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    product_name = _clean(state.get("productName") or state.get("product_name"))
    product_description = _clean(state.get("productDescription") or state.get("product_description"))
    target_audience = _clean(state.get("targetAudience") or state.get("target_audience"))
    if isinstance(job_record, dict):
        if not product_name:
            product_name = _clean(job_record.get("productName") or job_record.get("product_name"))
        if not product_description:
            product_description = _clean(job_record.get("productDescription") or job_record.get("product_description"))
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    if not product_name:
        product_name = _clean(strategy.get("productNameResolved"))
    audit = build_product_input_audit(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
    )
    explicit_facts = list(audit.get("explicitProductFacts") or [])
    grounding = strategy.get("strategyEvidenceGrounding") if isinstance(strategy.get("strategyEvidenceGrounding"), dict) else {}
    explicit_facts.extend(list(grounding.get("explicitProductFacts") or []))
    return {
        "productName": product_name,
        "productDescription": product_description,
        "targetAudience": target_audience,
        "explicitProductFacts": list(dict.fromkeys(item for item in explicit_facts if _clean(item))),
    }


def _collect_strategy_texts(strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not strategy:
        return []
    return list(
        _iter_text_nodes(strategy, stage="strategy", source_type="persisted_report")
    )


def _collect_creator_texts(state: Dict[str, Any], candidate_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    creator = record.get("creatorOutput") if isinstance(record.get("creatorOutput"), dict) else {}
    if not creator and isinstance(record.get("creatorSnapshot"), dict):
        creator = record["creatorSnapshot"]
    accepted = ((state.get("acceptedCreatorCandidates") or {}).get(candidate_id) or {}).get("creatorOutput")
    if isinstance(accepted, dict) and accepted:
        creator = accepted
    texts: List[Dict[str, Any]] = []
    if creator:
        texts.extend(_iter_text_nodes(creator, stage="creator", candidate_id=candidate_id, source_type="persisted_report"))
    parsed_payload = None
    from engine.builder2_complete_ad_creator_recovery import load_rejected_creator_parsed_response

    rejected = load_rejected_creator_parsed_response(state, candidate_id)
    if isinstance(rejected, dict):
        parsed_payload = rejected.get("parsed") if isinstance(rejected.get("parsed"), dict) else rejected
    if isinstance(parsed_payload, dict):
        texts.extend(
            _iter_text_nodes(
                parsed_payload,
                stage="creator",
                candidate_id=candidate_id,
                source_type="parsed_response",
            )
        )
    diagnostics = (state.get("creatorDiagnosticsByCandidate") or {}).get(candidate_id) or record.get("creatorDiagnostics") or {}
    raw_text = ""
    if isinstance(diagnostics, dict):
        raw_text = _clean(diagnostics.get("responseText") or diagnostics.get("rawResponseText"))
        if raw_text:
            texts.append(
                {
                    "stage": "creator",
                    "candidateId": candidate_id,
                    "judgmentId": None,
                    "fieldPath": "creatorDiagnostics.responseText",
                    "text": raw_text,
                    "sourceType": "raw_response",
                }
            )
    meta = {
        "found": bool(creator),
        "prototypeId": _clean(record.get("prototypeId") or creator.get("prototypeId")),
        "parsedResponseFound": isinstance(parsed_payload, dict) and bool(parsed_payload),
        "rawResponseFound": bool(raw_text),
    }
    return texts, meta


def _collect_judge_texts(state: Dict[str, Any], candidate_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    judgment_id = _clean(record.get("judgmentId"))
    judgment = {}
    if judgment_id:
        judgment = ((state.get("judgments") or {}).get(judgment_id) or {}).get("judgment") or {}
    texts: List[Dict[str, Any]] = []
    if isinstance(judgment, dict) and judgment:
        texts.extend(
            _iter_text_nodes(
                judgment,
                stage="judge",
                candidate_id=candidate_id,
                judgment_id=judgment_id,
                source_type="persisted_report",
            )
        )
    raw_found = False
    parsed_found = False
    for entry in judge_ledger_entries(state, candidate_id):
        parsed = entry.get("parsedResponse") if isinstance(entry.get("parsedResponse"), dict) else {}
        if parsed:
            parsed_found = True
            texts.extend(
                _iter_text_nodes(
                    parsed,
                    stage="judge",
                    candidate_id=candidate_id,
                    judgment_id=_clean(entry.get("judgmentId")) or judgment_id,
                    source_type="parsed_response",
                )
            )
        raw_text = _clean(entry.get("responseText"))
        if entry.get("rawResponseAvailable"):
            raw_found = True
            if raw_text:
                texts.append(
                    {
                        "stage": "judge",
                        "candidateId": candidate_id,
                        "judgmentId": _clean(entry.get("judgmentId")) or judgment_id,
                        "fieldPath": "judgeResponseLedger.responseText",
                        "text": raw_text,
                        "sourceType": "raw_response",
                    }
                )
    meta = {
        "found": bool(judgment),
        "judgmentId": judgment_id or None,
        "parsedResponseFound": parsed_found,
        "rawResponseFound": raw_found,
    }
    return texts, meta


def _collect_winner_texts(state: Dict[str, Any], candidate_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    texts: List[Dict[str, Any]] = []
    plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
    if plan and _clean(plan.get("candidateId") or state.get("winnerDevelopmentCandidateId") or candidate_id):
        texts.extend(_iter_text_nodes(plan, stage="winner", candidate_id=candidate_id, source_type="persisted_report"))
    payload = load_revalidatable_parsed_winner_response(state)
    parsed = dict((payload or {}).get("parsed") or {})
    parsed_found = bool(parsed)
    raw_found = False
    if parsed:
        texts.extend(
            _iter_text_nodes(parsed, stage="winner", candidate_id=candidate_id, source_type="parsed_response")
        )
    if isinstance(payload, dict):
        raw_text = _clean(payload.get("rawResponseText"))
        if payload.get("rawResponseAvailable") and raw_text:
            raw_found = True
            texts.append(
                {
                    "stage": "winner",
                    "candidateId": candidate_id,
                    "judgmentId": None,
                    "fieldPath": f"{PARSED_WINNER_RESPONSE_KEY}.rawResponseText",
                    "text": raw_text,
                    "sourceType": "raw_response",
                }
            )
    return texts, {
        "found": bool(plan) or parsed_found,
        "parsedResponseFound": parsed_found,
        "rawResponseFound": raw_found,
        "responseLocation": PARSED_WINNER_RESPONSE_KEY if payload else ("winnerDevelopmentPlan" if plan else None),
    }


def _scan_texts(texts: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    evidence = _empty_evidence_block()
    relevant: List[Dict[str, Any]] = []
    seen_excerpt_keys: set[str] = set()

    for item in texts:
        text = _clean(item.get("text"))
        field_path = _clean(item.get("fieldPath"))
        if not text or _is_boolean_only_field(field_path, text):
            continue
        categories = _match_categories(text)
        if not categories:
            if any(hint.lower() in field_path.lower() or hint.lower() in text.lower() for hint in PRIORITY_FIELD_HINTS):
                relevant.append(
                    {
                        "stage": item.get("stage"),
                        "fieldPath": field_path,
                        "text": _truncate(text),
                        "matchedCategories": [],
                        "sourceType": item.get("sourceType"),
                    }
                )
            continue
        matched_names = [name for name, _reason in categories]
        relevant.append(
            {
                "stage": item.get("stage"),
                "fieldPath": field_path,
                "text": _truncate(text),
                "matchedCategories": matched_names,
                "sourceType": item.get("sourceType"),
            }
        )
        for category, reason in categories:
            evidence[category]["found"] = True
            excerpt_key = f"{category}|{item.get('stage')}|{field_path}|{text[:80]}"
            if excerpt_key in seen_excerpt_keys:
                continue
            seen_excerpt_keys.add(excerpt_key)
            evidence[category]["excerpts"].append(
                {
                    "stage": item.get("stage"),
                    "candidateId": item.get("candidateId"),
                    "judgmentId": item.get("judgmentId"),
                    "fieldPath": field_path,
                    "exactText": _truncate(text, limit=480),
                    "sourceType": item.get("sourceType"),
                    "classificationCategory": category,
                    "reasonForMatch": reason,
                }
            )
    return evidence, relevant


def _classify(evidence: Dict[str, Any]) -> Tuple[str, str]:
    a = evidence["audiencePriorKnowledgeRecognized"]["found"]
    b = evidence["missingInformationRecognizedByViewer"]["found"]
    c = evidence["curiosityGapIntended"]["found"]
    d = evidence["laterNaturalResolutionExpected"]["found"]
    e = evidence["immediateUnderstandingAssumed"]["found"]
    f = evidence["alternativeMeaningOrConfusionConsidered"]["found"]

    positive = a and b and (c or d)
    if positive:
        return (
            CLASSIFICATION_POSITIVE_GAP,
            "Preserved outputs explicitly state that the viewer lacks prior pack knowledge, "
            "recognizes missing information, and the gap is framed as curiosity or later completion.",
        )

    if e and (a or b or c or d or f):
        return (
            CLASSIFICATION_MIXED,
            "Preserved outputs contain both immediate-understanding assumptions and partial knowledge-gap language.",
        )

    if e:
        return (
            CLASSIFICATION_IMMEDIATE,
            "Preserved outputs explicitly assume the viewer immediately sees or understands pack quantity.",
        )

    if f and not positive:
        return (
            CLASSIFICATION_RISK,
            "Preserved outputs explicitly treat the knowledge gap as ambiguity, confusion, or risk rather than a positive teaser.",
        )

    if a or b or c or d:
        return (
            CLASSIFICATION_MIXED,
            "Preserved outputs mention related ideas, but not the full explicit positive knowledge-gap chain required for a confident positive classification.",
        )

    return CLASSIFICATION_NONE, NO_EVIDENCE_REASON


def inspect_audience_knowledge_gap(
    state: Dict[str, Any],
    *,
    candidate_id: str = "",
    job_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    winner_candidate_id = _resolve_winner_candidate_id(state, candidate_id)
    strategy = state.get("strategyFoundation") if isinstance(state.get("strategyFoundation"), dict) else {}
    original_input = _load_product_input(state, job_record)

    strategy_texts = _collect_strategy_texts(strategy)
    creator_texts, creator_meta = _collect_creator_texts(state, winner_candidate_id) if winner_candidate_id else ([], {"found": False})
    judge_texts, judge_meta = _collect_judge_texts(state, winner_candidate_id) if winner_candidate_id else ([], {"found": False})
    winner_texts, winner_meta = _collect_winner_texts(state, winner_candidate_id) if winner_candidate_id else ([], {"found": False})

    all_texts = strategy_texts + creator_texts + judge_texts + winner_texts
    evidence, relevant_stored_texts = _scan_texts(all_texts)
    classification, classification_reason = _classify(evidence)

    slogan = ""
    if winner_candidate_id:
        creator = ((state.get("candidates") or {}).get(winner_candidate_id) or {}).get("creatorOutput") or {}
        if isinstance(creator, dict):
            slogan = _clean(resolve_canonical_slogan_text(plan=creator, state=state) or (creator.get("advertisingClosure") or {}).get("sloganText"))
    if not slogan and isinstance(state.get("winnerDevelopmentPlan"), dict):
        plan = state["winnerDevelopmentPlan"]
        slogan = _clean(resolve_canonical_slogan_text(plan=plan, state=state) or (plan.get("advertisingClosure") or {}).get("sloganText"))

    return {
        "ok": True,
        "jobId": _clean(state.get("jobId")) or None,
        "tournamentId": _clean(state.get("tournamentId")) or None,
        "winnerCandidateId": winner_candidate_id or None,
        "winnerPrototypeId": _clean(creator_meta.get("prototypeId") or (state.get("candidates") or {}).get(winner_candidate_id, {}).get("prototypeId")) or None,
        "originalInput": original_input,
        "slogan": slogan or None,
        "sourceAvailability": {
            "strategyFound": bool(strategy),
            "creatorFound": bool(creator_meta.get("found")),
            "judgeFound": bool(judge_meta.get("found")),
            "winnerDevelopmentFound": bool(winner_meta.get("found")),
            "creatorRawResponseFound": bool(creator_meta.get("rawResponseFound")),
            "judgeRawResponseFound": bool(judge_meta.get("rawResponseFound")),
            "winnerRawResponseFound": bool(winner_meta.get("rawResponseFound")),
            "parsedResponsesFound": bool(
                creator_meta.get("parsedResponseFound")
                or judge_meta.get("parsedResponseFound")
                or winner_meta.get("parsedResponseFound")
            ),
            "creatorParsedResponseFound": bool(creator_meta.get("parsedResponseFound")),
            "judgeParsedResponseFound": bool(judge_meta.get("parsedResponseFound")),
            "winnerParsedResponseFound": bool(winner_meta.get("parsedResponseFound")),
            "winnerResponseLocation": winner_meta.get("responseLocation"),
            "judgmentId": judge_meta.get("judgmentId"),
        },
        "evidence": evidence,
        "relevantStoredTexts": relevant_stored_texts[:40],
        "classification": classification,
        "classificationReason": classification_reason,
        "limits": list(LIMITS),
        "readOnly": True,
        "stateMutated": False,
        "openAICalls": 0,
        "paidCalls": 0,
        "runwayCalls": 0,
        "imageCalls": 0,
        "ffmpegCalls": 0,
    }


def inspect_audience_knowledge_gap_for_job(
    job_id: str,
    *,
    candidate_id: str = "",
) -> Dict[str, Any]:
    if not _clean(job_id):
        return {
            "ok": False,
            "failureReason": "BUILDER2_AUDIENCE_KNOWLEDGE_GAP_INSPECT_JOB_ID_missing",
            "readOnly": True,
            "stateMutated": False,
            "openAICalls": 0,
            "paidCalls": 0,
            "runwayCalls": 0,
            "imageCalls": 0,
            "ffmpegCalls": 0,
        }
    with read_only_builder2_inspection():
        state = load_tournament_state(job_id, read_only=True)
        if not state:
            return {
                "ok": False,
                "jobId": job_id,
                "failureReason": "job_not_found",
                "readOnly": True,
                "stateMutated": False,
                "openAICalls": 0,
                "paidCalls": 0,
                "runwayCalls": 0,
                "imageCalls": 0,
                "ffmpegCalls": 0,
            }
        job_record = None
        try:
            if redis_configured():
                job_record = video_job_get(job_id)
        except Exception:
            job_record = None
        return inspect_audience_knowledge_gap(
            state,
            candidate_id=candidate_id,
            job_record=job_record if isinstance(job_record, dict) else None,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    job_id = _clean(os.environ.get("BUILDER2_AUDIENCE_KNOWLEDGE_GAP_INSPECT_JOB_ID"))
    candidate_id = _clean(os.environ.get("BUILDER2_AUDIENCE_KNOWLEDGE_GAP_INSPECT_CANDIDATE_ID"))
    report = inspect_audience_knowledge_gap_for_job(job_id, candidate_id=candidate_id)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
