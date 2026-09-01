"""
Builder1 planning stage checkpoints — durable paid-reasoning resume for the same job/request.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

from engine.builder1_job_ownership import _hash_token
from engine.builder1_jobs_store import JOB_TTL_SECONDS
from engine.builder1_methodology_reasons import STAGE_METHODOLOGY_BLOCKS
from engine.builder1_request_idempotency import fingerprint_initial_generate

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "builder1_planning_checkpoint_v2"
LEGACY_CHECKPOINT_VERSION = "builder1_planning_checkpoint_v1"
PLANNING_CONTRACT_VERSION = "builder1_production_v1"
CHECKPOINT_KEY_PREFIX = "builder1:planning_checkpoint:"

CHECKPOINT_IDENTITY_FIELDS: Tuple[str, ...] = (
    "jobId",
    "campaignId",
    "requestFingerprint",
    "planningContractVersion",
)

STAGE_ORDER: Tuple[str, ...] = (
    "strategy_slogan_stage",
    "conceptual_stage",
    "brand_physical",
    "graphic_system",
    "series_ads",
)

STAGE_CHECKPOINT_CONTRACT_VERSIONS: Dict[str, str] = {
    "strategy_slogan_stage": "1",
    "conceptual_stage": "1",
    "brand_physical": "2",
    "graphic_system": "1",
    "series_ads": "1",
}

T = TypeVar("T")

_memory_lock = threading.Lock()
_memory_checkpoints: Dict[str, Dict[str, Any]] = {}

_checkpoint_session_ctx: ContextVar[Optional["PlanningCheckpointSession"]] = ContextVar(
    "builder1_planning_checkpoint_session",
    default=None,
)


class PlanningCheckpointPersistError(RuntimeError):
    """Raised when a validated stage cannot be durably checkpointed before the next paid stage."""


class PlanningCheckpointIdentityError(RuntimeError):
    """Raised when checkpoint ownership or request identity does not match the current execution."""


def _redis_configured() -> bool:
    return bool((os.environ.get("REDIS_URL") or "").strip())


def _get_redis():
    from engine.video_jobs_redis import get_redis

    return get_redis()


def _checkpoint_key(job_id: str) -> str:
    return f"{CHECKPOINT_KEY_PREFIX}{(job_id or '').strip()}"


def stage_methodology_fingerprint(stage: str) -> str:
    block = STAGE_METHODOLOGY_BLOCKS.get(stage, "")
    return _hash_token(json.dumps({stage: block}, sort_keys=True, ensure_ascii=False))


def planning_methodology_fingerprint() -> str:
    """Legacy global fingerprint — retained for migration detection only."""
    payload = {stage: STAGE_METHODOLOGY_BLOCKS.get(stage, "") for stage in STAGE_ORDER}
    return _hash_token(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _canonical_brand_guidelines(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): raw[k] for k in sorted(raw.keys(), key=str)}


def build_request_fingerprint(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    brand_guidelines: Optional[Mapping[str, Any]],
) -> str:
    return fingerprint_initial_generate(
        {
            "productName": product_name,
            "productDescription": product_description,
            "format": format_value,
            "brandGuidelines": brand_guidelines,
        },
        ad_count=ad_count,
    )


@dataclass(frozen=True)
class PlanningCheckpointIdentity:
    job_id: str
    campaign_id: str
    request_fingerprint: str
    planning_contract_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jobId": self.job_id,
            "campaignId": self.campaign_id,
            "requestFingerprint": self.request_fingerprint,
            "planningContractVersion": self.planning_contract_version,
        }


def build_planning_checkpoint_identity(
    *,
    job_id: str,
    campaign_id: str,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    brand_guidelines: Optional[Mapping[str, Any]],
) -> PlanningCheckpointIdentity:
    return PlanningCheckpointIdentity(
        job_id=(job_id or "").strip(),
        campaign_id=(campaign_id or "").strip(),
        request_fingerprint=build_request_fingerprint(
            product_name=product_name,
            product_description=product_description,
            format_value=format_value,
            ad_count=ad_count,
            brand_guidelines=brand_guidelines,
        ),
        planning_contract_version=PLANNING_CONTRACT_VERSION,
    )


def _load_checkpoint_record(job_id: str) -> Optional[Dict[str, Any]]:
    jid = (job_id or "").strip()
    if not jid:
        return None
    if _redis_configured():
        try:
            raw = _get_redis().get(_checkpoint_key(jid))
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.error("BUILDER1_PLANNING_CHECKPOINT_LOAD_ERR jobId=%s err=%s", jid, exc)
            return None
    with _memory_lock:
        stored = _memory_checkpoints.get(jid)
        return dict(stored) if stored is not None else None


def _save_checkpoint_record(job_id: str, record: Dict[str, Any]) -> None:
    jid = (job_id or "").strip()
    if not jid:
        raise PlanningCheckpointPersistError("missing_job_id")
    payload = json.dumps(record, ensure_ascii=False)
    if _redis_configured():
        try:
            _get_redis().set(_checkpoint_key(jid), payload, ex=JOB_TTL_SECONDS)
            return
        except Exception as exc:
            logger.error("BUILDER1_PLANNING_CHECKPOINT_SAVE_ERR jobId=%s err=%s", jid, exc)
            raise PlanningCheckpointPersistError("builder1_planning_checkpoint_unavailable") from exc
    with _memory_lock:
        _memory_checkpoints[jid] = dict(record)


def load_planning_checkpoint_record(job_id: str) -> Optional[Dict[str, Any]]:
    return _load_checkpoint_record(job_id)


def save_planning_checkpoint_record(job_id: str, record: Dict[str, Any]) -> None:
    _save_checkpoint_record(job_id, record)


def delete_planning_checkpoint(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _redis_configured():
        try:
            _get_redis().delete(_checkpoint_key(jid))
        except Exception:
            pass
        return
    with _memory_lock:
        _memory_checkpoints.pop(jid, None)


def get_planning_checkpoint_session() -> Optional["PlanningCheckpointSession"]:
    return _checkpoint_session_ctx.get()


def set_planning_checkpoint_session(session: Optional["PlanningCheckpointSession"]):
    return _checkpoint_session_ctx.set(session)


def reset_planning_checkpoint_session(token) -> None:
    _checkpoint_session_ctx.reset(token)


def load_planning_execution_context(job_id: str) -> Optional[Dict[str, Any]]:
    record = _load_checkpoint_record(job_id)
    if not record:
        return None
    return {
        "explorationSeed": str(record.get("explorationSeed") or ""),
        "lensOrder": list(record.get("lensOrder") or []),
        "productNameResolved": str(record.get("productNameResolved") or ""),
    }


def _output_fingerprint(payload: Mapping[str, Any]) -> str:
    return _hash_token(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _downstream_stages(stage: str) -> List[str]:
    try:
        idx = STAGE_ORDER.index(stage)
    except ValueError:
        return []
    return list(STAGE_ORDER[idx + 1 :])


def strategy_slogan_dependency_fingerprint(
    *,
    identity: PlanningCheckpointIdentity,
    product_description: str,
    product_name_resolved: str,
    server_mandatory_constraints: Sequence[str],
    visibility_policy: str,
    exploration_seed: str,
    lens_order: Sequence[str],
) -> str:
    payload = {
        "requestFingerprint": identity.request_fingerprint,
        "stageMethodologyFingerprint": stage_methodology_fingerprint("strategy_slogan_stage"),
        "productDescription": product_description,
        "productNameResolved": product_name_resolved,
        "mandatoryConstraints": sorted(str(item) for item in server_mandatory_constraints),
        "visibilityPolicy": visibility_policy,
        "explorationSeed": exploration_seed,
        "lensOrder": list(lens_order),
    }
    return _hash_token(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def conceptual_dependency_fingerprint(
    *,
    identity: PlanningCheckpointIdentity,
    exploration_seed: str,
    selected_creative_brief: Mapping[str, Any],
    strategy_output_fingerprint: str,
) -> str:
    payload = {
        "requestFingerprint": identity.request_fingerprint,
        "stageMethodologyFingerprint": stage_methodology_fingerprint("conceptual_stage"),
        "explorationSeed": exploration_seed,
        "selectedCreativeBrief": dict(selected_creative_brief),
        "strategyOutputFingerprint": strategy_output_fingerprint,
    }
    return _hash_token(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def brand_physical_dependency_fingerprint(
    *,
    identity: PlanningCheckpointIdentity,
    visibility_policy: str,
    selected_creative_brief: Mapping[str, Any],
    strategy_output_fingerprint: str,
    conceptual_output_fingerprint: str,
) -> str:
    payload = {
        "requestFingerprint": identity.request_fingerprint,
        "stageMethodologyFingerprint": stage_methodology_fingerprint("brand_physical"),
        "visibilityPolicy": visibility_policy,
        "selectedCreativeBrief": dict(selected_creative_brief),
        "strategyOutputFingerprint": strategy_output_fingerprint,
        "conceptualOutputFingerprint": conceptual_output_fingerprint,
    }
    return _hash_token(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def graphic_system_dependency_fingerprint(
    *,
    identity: PlanningCheckpointIdentity,
    selected_creative_brief: Mapping[str, Any],
    strategy_output_fingerprint: str,
    conceptual_output_fingerprint: str,
    brand_physical_output_fingerprint: str,
) -> str:
    payload = {
        "requestFingerprint": identity.request_fingerprint,
        "stageMethodologyFingerprint": stage_methodology_fingerprint("graphic_system"),
        "selectedCreativeBrief": dict(selected_creative_brief),
        "strategyOutputFingerprint": strategy_output_fingerprint,
        "conceptualOutputFingerprint": conceptual_output_fingerprint,
        "brandPhysicalOutputFingerprint": brand_physical_output_fingerprint,
    }
    return _hash_token(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def series_ads_dependency_fingerprint(
    *,
    identity: PlanningCheckpointIdentity,
    ad_count: int,
    format_value: str,
    visibility_policy: str,
    effective_mandatory_constraints: Sequence[str],
    strategy_output_fingerprint: str,
    conceptual_output_fingerprint: str,
    brand_physical_output_fingerprint: str,
    graphic_output_fingerprint: str,
) -> str:
    payload = {
        "requestFingerprint": identity.request_fingerprint,
        "stageMethodologyFingerprint": stage_methodology_fingerprint("series_ads"),
        "adCount": int(ad_count),
        "format": format_value,
        "visibilityPolicy": visibility_policy,
        "effectiveMandatoryConstraints": sorted(str(item) for item in effective_mandatory_constraints),
        "strategyOutputFingerprint": strategy_output_fingerprint,
        "conceptualOutputFingerprint": conceptual_output_fingerprint,
        "brandPhysicalOutputFingerprint": brand_physical_output_fingerprint,
        "graphicOutputFingerprint": graphic_output_fingerprint,
    }
    return _hash_token(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def serialize_strategy_slogan_stage_output(
    *,
    strategy_selection: Any,
    selected_strategy: Any,
    strategy_candidates: Sequence[Any],
    strategy_reviews: Mapping[str, Any],
    slogan_selection: Any,
    selected_slogan: Any,
    slogan_candidates: Sequence[Any],
    selected_creative_brief: Any,
) -> Dict[str, Any]:
    from engine.builder1_slogan_stage import slogan_candidate_to_dict

    def _strategy_candidate_dict(candidate: Any) -> Dict[str, Any]:
        return {
            "id": candidate.id,
            "lens": candidate.lens,
            "strategicProblem": candidate.strategic_problem,
            "relativeAdvantage": candidate.relative_advantage,
            "briefSupport": candidate.brief_support,
            "advantageSource": candidate.advantage_source,
            "claimRisk": candidate.claim_risk,
            "campaignExecutableNow": candidate.campaign_executable_now,
            "requiresClientConsultation": candidate.requires_client_consultation,
            "clientActionLevel": candidate.client_action_level,
            "implementationCostLevel": candidate.implementation_cost_level,
            "simpleStrategicAction": candidate.simple_strategic_action,
        }

    return {
        "strategySelection": {
            "selectedCandidateId": strategy_selection.selected_candidate_id,
            "selectionReason": strategy_selection.selection_reason,
            "strategyFamily": strategy_selection.strategy_family,
            "scores": dict(strategy_selection.scores),
        },
        "selectedStrategy": _strategy_candidate_dict(selected_strategy),
        "strategyCandidates": [_strategy_candidate_dict(item) for item in strategy_candidates],
        "strategyReviews": {
            key: {
                "candidateId": review.candidate_id,
                "groundedInBrief": review.grounded_in_brief,
                "advantageCurrentlyTrue": review.advantage_currently_true,
                "executableNow": review.executable_now,
                "requiresMaterialInvestment": review.requires_material_investment,
                "requiresClientConsultation": review.requires_client_consultation,
                "requiresBusinessTransformation": review.requires_business_transformation,
                "brandOwnable": review.brand_ownable,
                "categoryRelevant": review.category_relevant,
                "eligible": review.eligible,
                "rejectionCodes": list(review.rejection_codes),
            }
            for key, review in strategy_reviews.items()
        },
        "sloganSelection": {
            "selectedCandidateId": slogan_selection.selected_candidate_id,
            "selectionReason": slogan_selection.selection_reason,
            "scores": dict(slogan_selection.scores),
        },
        "selectedSlogan": slogan_candidate_to_dict(selected_slogan),
        "sloganCandidates": [slogan_candidate_to_dict(item) for item in slogan_candidates],
        "selectedCreativeBrief": selected_creative_brief.to_dict(),
    }


def deserialize_strategy_slogan_stage_output(payload: Mapping[str, Any]) -> Tuple[Any, ...]:
    from engine.builder1_selected_creative_brief import SelectedCreativeBrief
    from engine.builder1_slogan_stage import SloganCandidate, SloganSelection
    from engine.builder1_staged_parsers import StrategyCandidate, StrategyCandidateReview, StrategySelection

    strategy_selection_raw = payload.get("strategySelection") or {}
    selected_strategy_raw = payload.get("selectedStrategy") or {}
    strategy_selection = StrategySelection(
        selected_candidate_id=str(strategy_selection_raw.get("selectedCandidateId") or ""),
        selection_reason=str(strategy_selection_raw.get("selectionReason") or ""),
        strategy_family=str(strategy_selection_raw.get("strategyFamily") or ""),
        scores=dict(strategy_selection_raw.get("scores") or {}),
    )
    selected_strategy = StrategyCandidate(
        id=str(selected_strategy_raw.get("id") or ""),
        lens=str(selected_strategy_raw.get("lens") or ""),
        strategic_problem=str(selected_strategy_raw.get("strategicProblem") or ""),
        relative_advantage=str(selected_strategy_raw.get("relativeAdvantage") or ""),
        brief_support=str(selected_strategy_raw.get("briefSupport") or ""),
        advantage_source=str(selected_strategy_raw.get("advantageSource") or ""),
        claim_risk=str(selected_strategy_raw.get("claimRisk") or ""),
        campaign_executable_now=bool(selected_strategy_raw.get("campaignExecutableNow", True)),
        requires_client_consultation=bool(selected_strategy_raw.get("requiresClientConsultation", False)),
        client_action_level=str(selected_strategy_raw.get("clientActionLevel") or "none"),
        implementation_cost_level=str(selected_strategy_raw.get("implementationCostLevel") or "none"),
        simple_strategic_action=selected_strategy_raw.get("simpleStrategicAction"),
    )
    strategy_candidates = [
        StrategyCandidate(
            id=str(item.get("id") or ""),
            lens=str(item.get("lens") or ""),
            strategic_problem=str(item.get("strategicProblem") or ""),
            relative_advantage=str(item.get("relativeAdvantage") or ""),
            brief_support=str(item.get("briefSupport") or ""),
            advantage_source=str(item.get("advantageSource") or ""),
            claim_risk=str(item.get("claimRisk") or ""),
            campaign_executable_now=bool(item.get("campaignExecutableNow", True)),
            requires_client_consultation=bool(item.get("requiresClientConsultation", False)),
            client_action_level=str(item.get("clientActionLevel") or "none"),
            implementation_cost_level=str(item.get("implementationCostLevel") or "none"),
            simple_strategic_action=item.get("simpleStrategicAction"),
        )
        for item in (payload.get("strategyCandidates") or [])
    ]
    strategy_reviews: Dict[str, StrategyCandidateReview] = {}
    for key, review in dict(payload.get("strategyReviews") or {}).items():
        strategy_reviews[str(key)] = StrategyCandidateReview(
            candidate_id=str(review.get("candidateId") or ""),
            grounded_in_brief=bool(review.get("groundedInBrief")),
            advantage_currently_true=bool(review.get("advantageCurrentlyTrue")),
            executable_now=bool(review.get("executableNow")),
            requires_material_investment=bool(review.get("requiresMaterialInvestment")),
            requires_client_consultation=bool(review.get("requiresClientConsultation")),
            requires_business_transformation=bool(review.get("requiresBusinessTransformation")),
            brand_ownable=bool(review.get("brandOwnable")),
            category_relevant=bool(review.get("categoryRelevant")),
            eligible=bool(review.get("eligible")),
            rejection_codes=list(review.get("rejectionCodes") or []),
        )
    slogan_selection_raw = payload.get("sloganSelection") or {}
    slogan_selection = SloganSelection(
        selected_candidate_id=str(slogan_selection_raw.get("selectedCandidateId") or ""),
        selection_reason=str(slogan_selection_raw.get("selectionReason") or ""),
        scores=dict(slogan_selection_raw.get("scores") or {}),
    )
    selected_slogan_raw = payload.get("selectedSlogan") or {}
    selected_slogan = SloganCandidate(
        id=str(selected_slogan_raw.get("id") or ""),
        brand_slogan=str(selected_slogan_raw.get("brandSlogan") or ""),
        derivation_from_advantage=str(selected_slogan_raw.get("derivationFromAdvantage") or ""),
        implied_action=str(selected_slogan_raw.get("impliedAction") or ""),
        why_ownable=str(selected_slogan_raw.get("whyOwnable") or ""),
        why_natural_in_language=str(selected_slogan_raw.get("whyNaturalInLanguage") or ""),
        competitor_transfer_risk=str(selected_slogan_raw.get("competitorTransferRisk") or ""),
        campaign_generative_power=str(selected_slogan_raw.get("campaignGenerativePower") or ""),
    )
    slogan_candidates = [
        SloganCandidate(
            id=str(item.get("id") or ""),
            brand_slogan=str(item.get("brandSlogan") or ""),
            derivation_from_advantage=str(item.get("derivationFromAdvantage") or ""),
            implied_action=str(item.get("impliedAction") or ""),
            why_ownable=str(item.get("whyOwnable") or ""),
            why_natural_in_language=str(item.get("whyNaturalInLanguage") or ""),
            competitor_transfer_risk=str(item.get("competitorTransferRisk") or ""),
            campaign_generative_power=str(item.get("campaignGenerativePower") or ""),
        )
        for item in (payload.get("sloganCandidates") or [])
    ]
    selected_brief = SelectedCreativeBrief.from_dict(payload.get("selectedCreativeBrief") or {})
    return (
        strategy_selection,
        selected_strategy,
        strategy_candidates or [selected_strategy],
        strategy_reviews,
        slogan_selection,
        selected_slogan,
        slogan_candidates or [selected_slogan],
        selected_brief,
    )


def serialize_conceptual_stage_output(
    *,
    conceptual_selection: Any,
    selected_conceptual: Any,
    conceptual_candidates: Sequence[Any],
) -> Dict[str, Any]:
    def _candidate_dict(candidate: Any) -> Dict[str, Any]:
        return {
            "id": candidate.id,
            "generator": candidate.generator,
            "action": candidate.action,
            "input": candidate.input,
            "transformation": candidate.transformation,
            "result": candidate.result,
            "perceptionToCreate": candidate.perception_to_create,
            "impliedPhysicalLaw": candidate.implied_physical_law,
            "whyItExpressesSlogan": candidate.why_it_expresses_slogan,
            "whyItExpressesAdvantage": candidate.why_it_expresses_advantage,
            "seriesPotential": candidate.series_potential,
            "brandOwnershipPotential": candidate.brand_ownership_potential,
        }

    return {
        "conceptualSelection": {
            "selectedCandidateId": conceptual_selection.selected_candidate_id,
            "selectionReason": conceptual_selection.selection_reason,
            "scores": dict(getattr(conceptual_selection, "scores", {}) or {}),
        },
        "selectedConceptual": _candidate_dict(selected_conceptual),
        "conceptualCandidates": [_candidate_dict(item) for item in conceptual_candidates],
    }


def deserialize_conceptual_stage_output(payload: Mapping[str, Any]) -> Tuple[Any, Any, List[Any]]:
    from engine.builder1_consolidated_stages import ConceptualSelection
    from engine.builder1_staged_parsers import ConceptualCandidate

    selection_raw = payload.get("conceptualSelection") or {}
    conceptual_selection = ConceptualSelection(
        selected_candidate_id=str(selection_raw.get("selectedCandidateId") or ""),
        selection_reason=str(selection_raw.get("selectionReason") or ""),
    )

    def _candidate_from_dict(item: Mapping[str, Any]) -> ConceptualCandidate:
        return ConceptualCandidate(
            id=str(item.get("id") or ""),
            generator=str(item.get("generator") or ""),
            action=str(item.get("action") or ""),
            input=str(item.get("input") or ""),
            transformation=str(item.get("transformation") or ""),
            result=str(item.get("result") or ""),
            perception_to_create=str(item.get("perceptionToCreate") or ""),
            implied_physical_law=str(item.get("impliedPhysicalLaw") or ""),
            why_it_expresses_slogan=str(item.get("whyItExpressesSlogan") or ""),
            why_it_expresses_advantage=str(item.get("whyItExpressesAdvantage") or ""),
            series_potential=str(item.get("seriesPotential") or ""),
            brand_ownership_potential=str(item.get("brandOwnershipPotential") or ""),
        )

    selected_conceptual = _candidate_from_dict(payload.get("selectedConceptual") or {})
    conceptual_candidates = [
        _candidate_from_dict(item)
        for item in (payload.get("conceptualCandidates") or [])
    ]
    return conceptual_selection, selected_conceptual, conceptual_candidates or [selected_conceptual]


class PlanningCheckpointSession:
    def __init__(self, identity: PlanningCheckpointIdentity):
        self.identity = identity
        self._record: Optional[Dict[str, Any]] = None

    @classmethod
    def open(cls, identity: PlanningCheckpointIdentity) -> "PlanningCheckpointSession":
        session = cls(identity)
        session._record = _load_checkpoint_record(identity.job_id) or session._empty_record()
        return session

    def _empty_record(self) -> Dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "identity": self.identity.to_dict(),
            "explorationSeed": "",
            "lensOrder": [],
            "productNameResolved": "",
            "completedStages": {},
        }

    def _ensure_record(self) -> Dict[str, Any]:
        if self._record is None:
            self._record = _load_checkpoint_record(self.identity.job_id) or self._empty_record()
        return self._record

    def _verify_identity(self, record: Mapping[str, Any]) -> bool:
        identity_raw = record.get("identity")
        if not isinstance(identity_raw, dict):
            return False
        expected = self.identity.to_dict()
        for key in CHECKPOINT_IDENTITY_FIELDS:
            if str(identity_raw.get(key) or "") != str(expected.get(key) or ""):
                logger.info(
                    "BUILDER1_PLANNING_CHECKPOINT_IDENTITY_MISMATCH jobId=%s field=%s",
                    self.identity.job_id,
                    key,
                )
                return False
        return True

    def _is_legacy_record(self, record: Mapping[str, Any]) -> bool:
        version = str(record.get("version") or "")
        if version == LEGACY_CHECKPOINT_VERSION:
            return True
        identity_raw = record.get("identity")
        if isinstance(identity_raw, dict) and identity_raw.get("methodologyFingerprint"):
            return True
        return False

    def bind_execution_context(
        self,
        *,
        exploration_seed: str,
        lens_order: Sequence[str],
        product_name_resolved: str = "",
    ) -> None:
        record = self._ensure_record()
        if not self._verify_identity(record):
            if record.get("completedStages"):
                logger.info(
                    "BUILDER1_PLANNING_CHECKPOINT_BIND_IDENTITY_MISMATCH jobId=%s preserveExisting=true",
                    self.identity.job_id,
                )
                return
            record = self._empty_record()
            self._record = record
        if exploration_seed and not record.get("explorationSeed"):
            record["explorationSeed"] = exploration_seed
        if lens_order and not record.get("lensOrder"):
            record["lensOrder"] = list(lens_order)
        if product_name_resolved and not record.get("productNameResolved"):
            record["productNameResolved"] = product_name_resolved
        self._persist_record(record)

    def get_execution_context(self) -> Optional[Dict[str, Any]]:
        record = self._ensure_record()
        if not self._verify_identity(record):
            return None
        seed = str(record.get("explorationSeed") or "")
        lens_order = list(record.get("lensOrder") or [])
        if not seed or not lens_order:
            return None
        return {
            "explorationSeed": seed,
            "lensOrder": lens_order,
            "productNameResolved": str(record.get("productNameResolved") or ""),
        }

    def get_stage_output_fingerprint(self, stage: str) -> str:
        record = self._ensure_record()
        entry = dict(record.get("completedStages") or {}).get(stage) or {}
        return str(entry.get("outputFingerprint") or "")

    def _invalidate_downstream(self, record: Dict[str, Any], stage: str) -> None:
        stages = dict(record.get("completedStages") or {})
        changed = False
        for downstream in _downstream_stages(stage):
            if downstream in stages:
                stages.pop(downstream, None)
                changed = True
        if changed:
            record["completedStages"] = stages

    def _persist_record(self, record: Dict[str, Any]) -> None:
        record["version"] = CHECKPOINT_VERSION
        record["identity"] = self.identity.to_dict()
        _save_checkpoint_record(self.identity.job_id, record)
        self._record = record

    def persist_stage(
        self,
        stage: str,
        *,
        output: Mapping[str, Any],
        dependency_fingerprint: str,
    ) -> str:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown_planning_stage:{stage}")
        record = self._ensure_record()
        if not self._verify_identity(record):
            record = self._empty_record()
        output_fp = _output_fingerprint(output)
        record.setdefault("completedStages", {})
        record["completedStages"][stage] = {
            "status": "succeeded",
            "stageContractVersion": STAGE_CHECKPOINT_CONTRACT_VERSIONS[stage],
            "stageMethodologyFingerprint": stage_methodology_fingerprint(stage),
            "dependencyFingerprint": dependency_fingerprint,
            "outputFingerprint": output_fp,
            "output": dict(output),
            "completedAt": time.time(),
        }
        self._invalidate_downstream(record, stage)
        self._persist_record(record)
        logger.info(
            "BUILDER1_PLANNING_CHECKPOINT_SAVED jobId=%s stage=%s outputFingerprint=%s",
            self.identity.job_id,
            stage,
            output_fp[:12],
        )
        return output_fp

    def try_restore_stage(
        self,
        stage: str,
        *,
        dependency_fingerprint: str,
        deserialize: Callable[[Mapping[str, Any]], T],
        revalidate: Optional[Callable[[T], None]] = None,
    ) -> Optional[T]:
        if stage not in STAGE_ORDER:
            return None
        record = self._ensure_record()
        if not self._verify_identity(record):
            return None
        entry = dict(dict(record.get("completedStages") or {}).get(stage) or {})
        if entry.get("status") != "succeeded":
            return None
        if entry.get("stageContractVersion") != STAGE_CHECKPOINT_CONTRACT_VERSIONS.get(stage):
            logger.info(
                "BUILDER1_PLANNING_CHECKPOINT_CONTRACT_MISMATCH jobId=%s stage=%s stored=%s expected=%s",
                self.identity.job_id,
                stage,
                entry.get("stageContractVersion"),
                STAGE_CHECKPOINT_CONTRACT_VERSIONS.get(stage),
            )
            return None
        stored_stage_methodology = str(entry.get("stageMethodologyFingerprint") or "")
        expected_stage_methodology = stage_methodology_fingerprint(stage)
        if not stored_stage_methodology:
            if self._is_legacy_record(record):
                logger.info(
                    "BUILDER1_PLANNING_CHECKPOINT_LEGACY_STAGE jobId=%s stage=%s",
                    self.identity.job_id,
                    stage,
                )
                return None
            logger.info(
                "BUILDER1_PLANNING_CHECKPOINT_METHODOLOGY_MISSING jobId=%s stage=%s",
                self.identity.job_id,
                stage,
            )
            return None
        if stored_stage_methodology != expected_stage_methodology:
            logger.info(
                "BUILDER1_PLANNING_CHECKPOINT_METHODOLOGY_MISMATCH jobId=%s stage=%s",
                self.identity.job_id,
                stage,
            )
            return None
        if str(entry.get("dependencyFingerprint") or "") != dependency_fingerprint:
            logger.info(
                "BUILDER1_PLANNING_CHECKPOINT_DEPENDENCY_MISMATCH jobId=%s stage=%s",
                self.identity.job_id,
                stage,
            )
            return None
        output_raw = entry.get("output")
        if not isinstance(output_raw, dict):
            logger.warning(
                "BUILDER1_PLANNING_CHECKPOINT_MALFORMED jobId=%s stage=%s",
                self.identity.job_id,
                stage,
            )
            return None
        stored_output_fp = str(entry.get("outputFingerprint") or "")
        if stored_output_fp and stored_output_fp != _output_fingerprint(output_raw):
            logger.info(
                "BUILDER1_PLANNING_CHECKPOINT_OUTPUT_MISMATCH jobId=%s stage=%s",
                self.identity.job_id,
                stage,
            )
            return None
        try:
            restored = deserialize(output_raw)
            if revalidate is not None:
                revalidate(restored)
        except Exception as exc:
            logger.warning(
                "BUILDER1_PLANNING_CHECKPOINT_RESTORE_FAILED jobId=%s stage=%s err=%s",
                self.identity.job_id,
                stage,
                exc,
            )
            return None
        from engine.builder1_planning_metrics import get_planning_metrics

        metrics = get_planning_metrics()
        if metrics is not None:
            metrics.record_stage_checkpoint_hit(stage)
        logger.info(
            "BUILDER1_PLANNING_CHECKPOINT_HIT jobId=%s stage=%s",
            self.identity.job_id,
            stage,
        )
        return restored


def serialize_brand_physical_checkpoint_output(brand_physical: Any) -> Dict[str, Any]:
    from engine.builder1_planner import _brand_physical_to_dict

    payload = _brand_physical_to_dict(brand_physical)
    assessment = getattr(brand_physical, "direct_product_route_assessment", None)
    if assessment is not None and hasattr(assessment, "to_dict"):
        payload["directProductRouteAssessment"] = assessment.to_dict()
    return payload


def deserialize_brand_physical_checkpoint_output(payload: Mapping[str, Any]) -> Any:
    from engine.builder1_direct_product_route import (
        AdditionalTranslationCost,
        DirectProductRouteAssessment,
        RecommendedVisualRoute,
    )
    from engine.builder1_final_stages import BrandPhysicalOutput

    assessment_raw = payload.get("directProductRouteAssessment")
    assessment = None
    if isinstance(assessment_raw, dict):
        assessment = DirectProductRouteAssessment(
            product_or_category_immediately_readable=bool(
                assessment_raw.get("productOrCategoryImmediatelyReadable")
            ),
            relative_advantage_directly_expressible_with_product=bool(
                assessment_raw.get("relativeAdvantageDirectlyExpressibleWithProduct")
            ),
            product_led_advertising_mechanism_available=bool(
                assessment_raw.get("productLedAdvertisingMechanismAvailable")
            ),
            product_led_mechanism_summary=str(assessment_raw.get("productLedMechanismSummary") or ""),
            external_analogy_adds_unique_persuasive_gain=bool(
                assessment_raw.get("externalAnalogyAddsUniquePersuasiveGain")
            ),
            external_analogy_unique_gain=str(assessment_raw.get("externalAnalogyUniqueGain") or ""),
            additional_translation_cost=AdditionalTranslationCost(
                str(assessment_raw.get("additionalTranslationCost") or AdditionalTranslationCost.NONE.value)
            ),
            recommended_route=RecommendedVisualRoute(
                str(assessment_raw.get("recommendedRoute") or RecommendedVisualRoute.ANALOGY_LED.value)
            ),
            route_decision_reason=str(assessment_raw.get("routeDecisionReason") or ""),
        )
    return BrandPhysicalOutput(
        product_name_resolved=str(payload.get("productNameResolved") or ""),
        physical_generator=str(payload.get("physicalGenerator") or ""),
        physical_generator_natural_purpose=str(payload.get("physicalGeneratorNaturalPurpose") or ""),
        physical_generator_campaign_role=str(payload.get("physicalGeneratorCampaignRole") or ""),
        physical_generator_is_product=bool(payload.get("physicalGeneratorIsProduct")),
        physical_generator_is_packaging=bool(payload.get("physicalGeneratorIsPackaging")),
        works_without_product_visible=bool(payload.get("worksWithoutProductVisible")),
        transferred_object=str(payload.get("transferredObject") or ""),
        transferred_object_action=str(payload.get("transferredObjectAction") or ""),
        why_clearer_than_showing_product=str(payload.get("whyClearerThanShowingProduct") or ""),
        medium_participates=bool(payload.get("mediumParticipates")),
        medium_role=str(payload.get("mediumRole") or ""),
        campaign_rationale=str(payload.get("campaignRationale") or ""),
        product_evidence_required=bool(payload.get("productEvidenceRequired", False)),
        product_evidence_reason=str(payload.get("productEvidenceReason") or ""),
        clearer_than_conventional_product_shot=bool(
            payload.get("clearerThanConventionalProductShot", True)
        ),
        survives_product_removal=bool(payload.get("survivesProductRemoval", True)),
        direct_product_route_assessment=assessment,
    )


def serialize_graphic_system_checkpoint_output(graphic: Any) -> Dict[str, Any]:
    from engine.builder1_planner import _graphic_to_dict

    return _graphic_to_dict(graphic)


def deserialize_graphic_system_checkpoint_output(payload: Mapping[str, Any]) -> Any:
    from engine.builder1_final_stages import parse_graphic_system_output

    return parse_graphic_system_output(dict(payload))


def serialize_series_ads_checkpoint_output(series_ads: Any) -> Dict[str, Any]:
    return {
        "seriesGenerator": dict(getattr(series_ads, "series_generator", {}) or {}),
        "ads": list(getattr(series_ads, "ads", []) or []),
    }


def deserialize_series_ads_checkpoint_output(payload: Mapping[str, Any]) -> Any:
    from engine.builder1_final_stages import SeriesAdsOutput

    return SeriesAdsOutput(
        series_generator=dict(payload.get("seriesGenerator") or {}),
        ads=list(payload.get("ads") or []),
    )


def run_checkpointed_planning_stage(
    stage: str,
    runner: Callable[[], T],
    *,
    session: Optional[PlanningCheckpointSession],
    dependency_fingerprint: str,
    serialize: Callable[[T], Mapping[str, Any]],
    deserialize: Callable[[Mapping[str, Any]], T],
    revalidate: Optional[Callable[[T], None]] = None,
) -> T:
    if session is not None:
        restored = session.try_restore_stage(
            stage,
            dependency_fingerprint=dependency_fingerprint,
            deserialize=deserialize,
            revalidate=revalidate,
        )
        if restored is not None:
            return restored
    result = runner()
    if session is not None:
        try:
            session.persist_stage(
                stage,
                output=serialize(result),
                dependency_fingerprint=dependency_fingerprint,
            )
        except PlanningCheckpointPersistError:
            raise
        except Exception as exc:
            raise PlanningCheckpointPersistError(f"{stage}_checkpoint_persist_failed") from exc
    return result
