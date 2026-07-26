"""
Builder2 server-owned creative-order contract — non-authoritative legacy attestation handling.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.builder2_methodology_contract import CREATIVE_STAGE_ORDER

logger = logging.getLogger(__name__)

CREATIVE_ORDER_CONTRACT_VERSION = "builder2_creative_order_v1"

_ATTESTATION_KEYS = (
    "visualCameBeforeKeyword",
    "runwayCheckCameBeforeKeyword",
    "headlineWasNotStartingPoint",
)


def build_creative_order_contract() -> Dict[str, Any]:
    return {
        "version": CREATIVE_ORDER_CONTRACT_VERSION,
        "stageOrder": list(CREATIVE_STAGE_ORDER),
        "enforcedByCreatorPrompt": True,
        "runwayFeasibilityPrecedesVerbalMechanism": True,
        "finalHeadlineIsWinnerOnly": True,
    }


def _attestation_boolean_values(attestation: Any) -> Dict[str, Any]:
    if not isinstance(attestation, dict):
        return {}
    return {key: attestation.get(key) for key in _ATTESTATION_KEYS if key in attestation}


def _build_creator_attestations(attestation: Any) -> Dict[str, Any]:
    received = attestation is not None
    values = _attestation_boolean_values(attestation) if received else {}
    return {
        "creativeOrderConfirmationReceived": received,
        "creativeOrderConfirmation": values,
        "authoritative": False,
    }


def log_creator_order_attestation_ignored(
    attestation: Any,
    *,
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
    prototype_id: str = "",
) -> None:
    if attestation is None:
        return
    values = _attestation_boolean_values(attestation)
    logger.info(
        "BUILDER2_CREATOR_ORDER_ATTESTATION_IGNORED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
        "present=true visualCameBeforeKeyword=%s runwayCheckCameBeforeKeyword=%s headlineWasNotStartingPoint=%s authoritative=false",
        job_id or "(none)",
        tournament_id or "(none)",
        candidate_id or "(none)",
        prototype_id or "(none)",
        values.get("visualCameBeforeKeyword", "(absent)"),
        values.get("runwayCheckCameBeforeKeyword", "(absent)"),
        values.get("headlineWasNotStartingPoint", "(absent)"),
    )


def log_creative_order_contract_attached(
    *,
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
    prototype_id: str = "",
) -> None:
    logger.info(
        "BUILDER2_CREATIVE_ORDER_CONTRACT_ATTACHED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s version=%s",
        job_id or "(none)",
        tournament_id or "(none)",
        candidate_id or "(none)",
        prototype_id or "(none)",
        CREATIVE_ORDER_CONTRACT_VERSION,
    )


def finalize_creator_order_metadata(
    candidate: Dict[str, Any],
    *,
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
    prototype_id: str = "",
) -> Dict[str, Any]:
    out = dict(candidate)
    legacy_attestation = out.pop("creativeOrderConfirmation", None)
    if legacy_attestation is not None:
        out["creatorAttestations"] = _build_creator_attestations(legacy_attestation)
        log_creator_order_attestation_ignored(
            legacy_attestation,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
        )
    out["creativeOrderContract"] = build_creative_order_contract()
    log_creative_order_contract_attached(
        job_id=job_id,
        tournament_id=tournament_id,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
    )
    return out
