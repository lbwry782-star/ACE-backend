#!/usr/bin/env python3
"""
Operator tool: revalidate and optionally resume Builder1 jobs rejected by campaign integrity.

Usage:
  python scripts/recover_builder1_integrity_rejected_campaign.py --job-id <uuid>
  python scripts/recover_builder1_integrity_rejected_campaign.py --job-id <uuid> --apply
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Revalidate persisted integrityFailureDiagnostic.rejectedPlan and optionally resume image generation.",
    )
    parser.add_argument("--job-id", required=True, help="Builder1 job UUID")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create campaign session and enqueue image pipeline (default: dry-run only)",
    )
    args = parser.parse_args()

    from engine.builder1_integrity_recovery import (
        apply_builder1_integrity_recovery,
        assess_builder1_integrity_recovery,
    )

    if args.apply:
        report = apply_builder1_integrity_recovery(
            args.job_id,
            enqueue_image_pipeline=True,
        )
    else:
        report = assess_builder1_integrity_recovery(args.job_id)
        report["dryRun"] = True

    print(json.dumps({"ok": bool(report.get("recoveryEligible") or report.get("applied")), **report}, ensure_ascii=False, indent=2))
    if not report.get("recoveryEligible") and not report.get("applied"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
