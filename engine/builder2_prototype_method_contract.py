"""
Builder2 server-owned prototype method contract — non-authoritative legacy attestation handling.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from engine.builder2_prototypes import require_prototype

logger = logging.getLogger(__name__)

PROTOTYPE_METHOD_CONTRACT_VERSION = "builder2_prototype_method_v1"

_APPLICATION_EVIDENCE_PATHS: Dict[str, List[str]] = {
    "winning_card": [
        "winningCardApplication",
        "visualMechanism",
        "creatorReport.whyParallelExpressesAdvantage",
        "creatorReport",
    ],
    "summer_fan": [
        "summerFanApplication",
        "visualMechanism",
        "creatorReport.whyParallelExpressesAdvantage",
        "creatorReport",
    ],
    "forgot": [
        "forgotApplication",
        "visualMechanism",
        "creatorReport.whyParallelExpressesAdvantage",
        "creatorReport",
    ],
    "greenpeace_essential_pairing": [
        "essentialPairingApplication",
        "visualMechanism",
        "creatorReport.whyParallelExpressesAdvantage",
        "creatorReport",
    ],
    "closest": [
        "closestApplication",
        "visualMechanism",
        "creatorReport.whyParallelExpressesAdvantage",
        "creatorReport",
    ],
    "think_small": [
        "thinkSmallApplication",
        "visualMechanism",
        "creatorReport.whyParallelExpressesAdvantage",
        "creatorReport",
    ],
}


def build_prototype_method_contract(prototype_id: str) -> Dict[str, Any]:
    prototype = require_prototype(prototype_id)
    return {
        "prototypeId": prototype.prototype_id,
        "methodVersion": PROTOTYPE_METHOD_CONTRACT_VERSION,
        "canonicalMethodSummary": prototype.reusable_method,
        "displayName": prototype.display_name,
        "mustNotCopy": prototype.must_not_copy,
        "assignedByServer": True,
        "applicationEvidencePaths": list(
            _APPLICATION_EVIDENCE_PATHS.get(prototype_id, [])
        ),
    }


def _legacy_prototype_attestation_payload(attestation: Any) -> Dict[str, Any]:
    if not isinstance(attestation, dict):
        return {}
    payload: Dict[str, Any] = {}
    for key in ("methodSummary", "applicationToCurrentProblem", "whyThisIsNotLiteralImitation"):
        if key in attestation:
            payload[key] = attestation.get(key)
    copied = attestation.get("surfaceElementsCopied")
    if copied is not None:
        payload["surfaceElementsCopied"] = copied
    return payload


def log_legacy_prototype_attestation_ignored(
    attestation: Any,
    *,
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
    prototype_id: str = "",
) -> None:
    if attestation is None:
        return
    keys = sorted(_legacy_prototype_attestation_payload(attestation).keys())
    logger.info(
        "BUILDER2_LEGACY_PROTOTYPE_ATTESTATION_IGNORED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s "
        "present=true keys=%s authoritative=false",
        job_id or "(none)",
        tournament_id or "(none)",
        candidate_id or "(none)",
        prototype_id or "(none)",
        ",".join(keys) if keys else "(none)",
    )


def log_prototype_method_contract_attached(
    *,
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
    prototype_id: str = "",
) -> None:
    logger.info(
        "BUILDER2_PROTOTYPE_METHOD_CONTRACT_ATTACHED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s version=%s",
        job_id or "(none)",
        tournament_id or "(none)",
        candidate_id or "(none)",
        prototype_id or "(none)",
        PROTOTYPE_METHOD_CONTRACT_VERSION,
    )


def finalize_prototype_method_metadata(
    candidate: Dict[str, Any],
    *,
    prototype_id: str,
    job_id: str = "",
    tournament_id: str = "",
    candidate_id: str = "",
) -> Dict[str, Any]:
    out = dict(candidate)
    legacy_attestation = out.pop("prototypeMethodApplication", None)
    attestations = dict(out.get("creatorAttestations") or {})
    if legacy_attestation is not None:
        payload = _legacy_prototype_attestation_payload(legacy_attestation)
        attestations["prototypeMethodApplicationReceived"] = True
        attestations["prototypeMethodApplication"] = payload
        attestations["authoritative"] = False
        log_legacy_prototype_attestation_ignored(
            legacy_attestation,
            job_id=job_id,
            tournament_id=tournament_id,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
        )
    if attestations:
        out["creatorAttestations"] = attestations
    contract = build_prototype_method_contract(prototype_id)
    if out.get("prototypeMethodContract") != contract:
        out["prototypeMethodContract"] = contract
    log_prototype_method_contract_attached(
        job_id=job_id,
        tournament_id=tournament_id,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
    )
    return out
