"""CLI entry for external PUBLISH simulation runner (not invoked by Flask LOCAL)."""
from __future__ import annotations

import argparse
import sys

from engine.ace_local_dev_publish_external_runner import (
    PUBLISH_RUNNER_MODE_SIMULATION,
    PublishExternalRunnerError,
    assert_simulation_only_mode,
    run_external_publish_simulation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ACE external PUBLISH runner (SIMULATION only — no real deploy).",
    )
    parser.add_argument("--session-id", required=True, help="Open dev cycle session id")
    parser.add_argument("--operation-id", required=True, help="32-char hex operation id")
    parser.add_argument(
        "--publish-uri-preview-fingerprint",
        required=True,
        help="64-char preview fingerprint from prepare/preview",
    )
    parser.add_argument(
        "--confirm-execute",
        default="PUBLISH",
        help='Must be "PUBLISH" to proceed',
    )
    parser.add_argument(
        "--mode",
        default=PUBLISH_RUNNER_MODE_SIMULATION,
        help=f"Fixed to {PUBLISH_RUNNER_MODE_SIMULATION} only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        assert_simulation_only_mode(str(args.mode))
        if str(args.mode).strip().upper() not in ("", PUBLISH_RUNNER_MODE_SIMULATION):
            raise PublishExternalRunnerError("ace_publish_external_runner_real_mode_forbidden")
        run_external_publish_simulation(
            session_id=str(args.session_id).strip(),
            operation_id=str(args.operation_id).strip(),
            publish_uri_preview_fingerprint=str(args.publish_uri_preview_fingerprint).strip(),
            confirm_execute=str(args.confirm_execute).strip(),
        )
    except PublishExternalRunnerError as exc:
        print(exc.code, file=sys.stderr)
        if exc.message and exc.message != exc.code:
            print(exc.message, file=sys.stderr)
        return 1
    except Exception as exc:
        print("ace_publish_external_runner_failed", file=sys.stderr)
        print(str(exc)[:240], file=sys.stderr)
        return 1
    print("ace_publish_external_runner_simulation_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
