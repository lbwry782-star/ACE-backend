#!/usr/bin/env python3
"""
Manual opt-in smoke test for Builder1 image compliance Responses API requests.

Uses the same production request builder as runtime review. Never runs during
startup or unit tests.

Usage (from repo root, with OPENAI_API_KEY set):

  python scripts/smoke_builder1_image_compliance.py --live

Optional:
  --model gpt-4o
  --schema strict|plain
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.builder1_image_compliance import (  # noqa: E402
    IMAGE_COMPLIANCE_SYSTEM_PROMPT,
    _extract_response_text,
    compliance_model_name,
    parse_image_compliance_response,
)
from engine.builder1_image_compliance_contract import (  # noqa: E402
    COMPLIANCE_SCHEMA_VERSION,
    build_compliance_responses_request_kwargs,
)
from engine.builder1_strict_schema import (  # noqa: E402
    classify_compliance_api_error,
    extract_openai_api_error_details,
)


def _tiny_test_image() -> bytes:
    # Minimal valid 1x1 JPEG (fixed fixture, no external files required).
    return bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "070909080a0c141d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c"
        "1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b0c180d"
        "0d1832211c2132323232323232323232323232323232323232323232323232323232"
        "323232323232323232323232323232323232ffc0001108000100010301110002110003"
        "110001ffc4001500010100000000000000000000000000000008ffc4001410010000"
        "00000000000000000000000000ffda0008010100003f00d2cf20ffd9"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Builder1 compliance Responses API smoke test")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Perform one real OpenAI Responses API call (required)",
    )
    parser.add_argument("--model", default="", help="Override compliance model")
    parser.add_argument(
        "--schema",
        choices=("strict", "plain"),
        default="strict",
        help="Request schema mode (default: strict)",
    )
    args = parser.parse_args()

    if not args.live:
        print("Refusing to run without --live (opt-in manual smoke test only).")
        return 2

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not set.")
        return 2

    model = (args.model or compliance_model_name()).strip()
    image_bytes = _tiny_test_image()
    kwargs = build_compliance_responses_request_kwargs(
        model=model,
        image_bytes=image_bytes,
        system_prompt=IMAGE_COMPLIANCE_SYSTEM_PROMPT,
        product_name="SmokeTestBrand",
        product_description="Smoke test product",
        visibility_policy="FORBIDDEN",
        transferred_object="",
        schema_mode=args.schema,
    )

    print(f"model={model}")
    print(f"schemaVersion={COMPLIANCE_SCHEMA_VERSION}")
    print(f"schemaMode={args.schema}")

    try:
        from openai import OpenAI
        import httpx
    except Exception as exc:
        print(f"http=client_import_failed error={exc}")
        return 1

    client = OpenAI(api_key=api_key, timeout=httpx.Timeout(120.0), max_retries=0)
    try:
        response = client.responses.create(**kwargs)
    except Exception as exc:
        details = extract_openai_api_error_details(exc)
        reason = classify_compliance_api_error(exc)
        print("http=failure")
        print(f"reasonCode={reason}")
        print(f"statusCode={details.get('statusCode')}")
        print(f"errorCode={details.get('errorCode')}")
        print(f"errorParam={details.get('errorParam')}")
        print(f"errorMessage={details.get('errorMessage')}")
        print(f"requestId={details.get('requestId')}")
        return 1

    print("http=success")
    print(f"responseClass={type(response).__name__}")
    out_text = _extract_response_text(response)
    try:
        parsed = parse_image_compliance_response(out_text)
        print(f"canonicalParse=pass passed={parsed.passed}")
    except Exception as exc:
        print(f"canonicalParse=fail error={exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
