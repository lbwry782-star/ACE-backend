"""
Recovery periodic save background service — activation contract (phase 5.6).

Not started from Start-AceLocal in this phase; used by tests and future wiring.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_AUTO_RECOVERY_ALLOWED,
    COORDINATOR_START_ALLOWED,
    COORDINATOR_START_BLOCKED,
    COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
)

SERVICE_SCHEMA_VERSION = 1
ALLOWED_COORDINATOR_OUTCOMES = frozenset(
    {COORDINATOR_START_ALLOWED, COORDINATOR_AUTO_RECOVERY_ALLOWED}
)
BLOCKED_WHEN_UNRESOLVED = frozenset(
    {
        COORDINATOR_START_BLOCKED,
        COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
        "RECOVERY_UNSAFE",
        "GATE_CHECK_FAILED",
    }
)
BLOCKED_COORDINATOR_OUTCOMES = frozenset(
    {
        COORDINATOR_START_BLOCKED,
        COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
        "RECOVERY_UNSAFE",
        "GATE_CHECK_FAILED",
    }
)


def evaluate_periodic_save_background_activation(
    *,
    coordinator_outcome: str,
    allow_periodic_save_service: bool,
    launcher_session: str,
    recovery_unresolved: bool = False,
    isolated_operations: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Returns whether a hidden background periodic-save process may start (future Start wiring).
    Does not spawn processes.
    """
    session = (launcher_session or "").strip()
    outcome = (coordinator_outcome or "").strip()
    isolated = list(isolated_operations or [])
    reasons: list[str] = []

    if recovery_unresolved:
        reasons.append("recovery_unresolved")
    if outcome in BLOCKED_COORDINATOR_OUTCOMES:
        reasons.append(f"coordinator_outcome:{outcome or 'empty'}")
    elif outcome not in ALLOWED_COORDINATOR_OUTCOMES:
        reasons.append(f"coordinator_outcome:{outcome or 'empty'}")
    if not allow_periodic_save_service:
        reasons.append("periodic_save_not_allowed_by_coordinator")
    if not session:
        reasons.append("launcher_session_missing")
    if isolated:
        reasons.append("isolated_operations")

    allowed = not reasons
    return {
        "schemaVersion": SERVICE_SCHEMA_VERSION,
        "allowed": allowed,
        "reasons": reasons,
        "launcherSession": session,
        "coordinatorOutcome": outcome,
        "mustRunHidden": True,
        "mustNotRunTwice": True,
        "controlledStopRequired": True,
        "mustNotTouchGuiProgress": True,
        "note": "Background service activation contract for LOCAL periodic save.",
    }


def _cli_main(argv: Optional[list] = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    payload_path = ""
    for idx, token in enumerate(args):
        if token == "--json-file" and idx + 1 < len(args):
            payload_path = args[idx + 1]
    if not payload_path:
        sys.stderr.write("usage: --json-file <path>\n")
        return 1
    raw = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    result = evaluate_periodic_save_background_activation(
        coordinator_outcome=str(raw.get("coordinatorOutcome") or ""),
        allow_periodic_save_service=bool(raw.get("allowPeriodicSaveService")),
        launcher_session=str(raw.get("launcherSession") or ""),
        recovery_unresolved=bool(raw.get("recoveryUnresolved")),
        isolated_operations=list(raw.get("isolatedOperations") or []),
    )
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0 if result.get("allowed") else 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
