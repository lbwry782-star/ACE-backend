"""
Builder2 final-output inspector — read-only completed delivery diagnostics.

Run:
  BUILDER2_FINAL_OUTPUT_INSPECT_JOB_ID=<jobId> python -m engine.builder2_final_output_inspect
"""
from __future__ import annotations

import json
import os
import sys

from engine.builder2_final_output_diagnostics import inspect_builder2_final_output
from engine.builder2_tournament_store import load_tournament_state


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def main() -> int:
    job_id = _env("BUILDER2_FINAL_OUTPUT_INSPECT_JOB_ID")
    if not job_id:
        print("BUILDER2_FINAL_OUTPUT_INSPECT_JOB_ID is required", file=sys.stderr)
        return 2
    state = load_tournament_state(job_id, read_only=True)
    if not isinstance(state, dict) or not state:
        print(json.dumps({"jobId": job_id, "error": "tournament_state_missing"}))
        return 1
    report = inspect_builder2_final_output(state)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
