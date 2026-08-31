#!/usr/bin/env python3
"""
Operator tool: deterministic rain-gutter graphic-device cleanup (zero paid calls).

Usage (Render WEB shell):
  DRY_RUN=true python scripts/repair_builder1_graphic_device_campaign.py
  DRY_RUN=false python scripts/repair_builder1_graphic_device_campaign.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair redundant copper-frame graphic device in stored Builder1 campaign.")
    parser.add_argument(
        "--campaign-id",
        default=os.environ.get("CAMPAIGN_ID", "b59781f3-a4fa-4352-9f27-fa9ca326b1f3"),
        help="Target campaign UUID",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist the repaired plan (default: dry-run only)",
    )
    args = parser.parse_args()

    dry_run = not args.apply
    env_dry = os.environ.get("DRY_RUN", "").strip().lower()
    if env_dry in {"1", "true", "yes"}:
        dry_run = True
    if env_dry in {"0", "false", "no"} and args.apply:
        dry_run = False

    from engine.builder1_campaign_store import CampaignStoreError
    from engine.builder1_graphic_device_campaign_repair import run_graphic_device_campaign_cleanup

    try:
        report = run_graphic_device_campaign_cleanup(args.campaign_id, dry_run=dry_run)
    except CampaignStoreError as exc:
        payload = {"ok": False, "error": exc.code, "message": exc.message, "campaignId": args.campaign_id}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"ok": True, **report}, ensure_ascii=False, indent=2))
    if report.get("validationErrors"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
