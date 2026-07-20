"""
Production-shaped Builder1 image compliance Responses API envelope tests.

Uses repository SDK-shaped test fixtures (NOT actual openai.types.responses.Response).
For real OpenAI SDK model tests see tests.test_builder1_openai_sdk_compliance.

Run: python -m unittest tests.test_builder1_image_compliance_responses_envelope -v
"""
from __future__ import annotations

import base64
import json
import os
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder1_image_compliance import (
    ImageComplianceResponseError,
    ImageComplianceUnavailableError,
    _extract_response_text,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_compliance_contract import (
    COMPLIANCE_RESPONSE_JSON_SCHEMA,
    coerce_review_dict,
    normalize_compliance_payload,
)
from tests.builder1_responses_api_fixtures import (
    PRODUCTION_FAILURE_INNER,
    SdkShapedFixtureResponse,
    build_envelope_empty_output_text_nested,
    build_envelope_fenced_json_text,
    build_envelope_nested_content,
    build_envelope_output_text_property,
    build_envelope_production_failure_inner,
    build_envelope_strict_parsed_content,
    canonical_compliance_inner,
    image_bytes_from_create_kwargs,
    legacy_compliance_inner,
    legacy_pass_gate_would_reject,
)

TEST_IMAGE = b"production-envelope-image-bytes"


def _install_openai_modules() -> None:
    import sys
    from unittest.mock import MagicMock

    if "openai" not in sys.modules or not hasattr(sys.modules["openai"], "OpenAI"):
        openai_module = MagicMock()
        openai_module.OpenAI = MagicMock()
        sys.modules["openai"] = openai_module
    if "httpx" not in sys.modules or not hasattr(sys.modules["httpx"], "Timeout"):
        httpx_module = MagicMock()
        httpx_module.Timeout = MagicMock(return_value=MagicMock())
        sys.modules["httpx"] = httpx_module


def _run_through_openai_client(
    response_object: object,
    *,
    side_effect: list | None = None,
    **review_kwargs: Any,
):
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.responses.create.side_effect = side_effect
    else:
        mock_client.responses.create.return_value = response_object

    with patch("openai.OpenAI", return_value=mock_client), patch(
        "httpx.Timeout",
        return_value=MagicMock(),
    ):
        result = review_builder1_ad_image_compliance(
            TEST_IMAGE,
            product_name="TestBrand",
            ad_index=1,
            campaign_id="cmp-envelope",
            job_id="job-envelope",
            **review_kwargs,
        )
    return result, mock_client


class TestResponsesApiEnvelopeCompliance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_openai_modules()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_image_compliance_parses_real_responses_api_envelope(self) -> None:
        inner = canonical_compliance_inner()
        envelope = build_envelope_nested_content(inner)

        with patch(
            "engine.builder1_image_compliance._extract_response_text",
            wraps=_extract_response_text,
        ) as mock_extract:
            with patch(
                "engine.builder1_image_compliance._log_compliance_response_structure",
            ) as mock_structure_log:
                result, mock_client = _run_through_openai_client(envelope)

        self.assertTrue(result.passed)
        self.assertEqual(result.hard_violations, [])
        mock_extract.assert_called_once()
        mock_structure_log.assert_called_once()
        mock_client.responses.create.assert_called_once()

        extracted = _extract_response_text(envelope)
        decoded = json.loads(extracted)
        self.assertEqual(decoded["reviewStatus"], "completed")
        normalized = normalize_compliance_payload(decoded)
        self.assertFalse(normalized.legacy_normalized)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_nested_content_succeeds_when_output_text_empty(self) -> None:
        inner = canonical_compliance_inner()
        envelope = build_envelope_empty_output_text_nested(inner)
        self.assertEqual(getattr(envelope, "output_text"), "")

        result, _mock_client = _run_through_openai_client(envelope)
        self.assertTrue(result.passed)
        self.assertTrue(_extract_response_text(envelope))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_sdk_shaped_fixture_response_object_succeeds(self) -> None:
        envelope = build_envelope_output_text_property(canonical_compliance_inner())
        self.assertIsInstance(envelope, SdkShapedFixtureResponse)
        self.assertNotEqual(type(envelope).__name__, "Response")

        result, _mock_client = _run_through_openai_client(envelope)
        self.assertTrue(result.passed)
        self.assertIsInstance(envelope.output_text, str)
        self.assertIn("reviewStatus", envelope.output_text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_production_extraction_function_used_end_to_end(self) -> None:
        envelope = build_envelope_nested_content(canonical_compliance_inner())
        with patch(
            "engine.builder1_image_compliance._extract_response_text",
            wraps=_extract_response_text,
        ) as mock_extract:
            _run_through_openai_client(envelope)
        mock_extract.assert_any_call(envelope)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance")
    def test_strict_schema_response_path_matches_parser_fields(
        self,
        mock_text_format: MagicMock,
    ) -> None:
        mock_text_format.return_value = {
            "format": {
                "type": "json_schema",
                "name": "builder1_image_compliance_v2",
                "schema": COMPLIANCE_RESPONSE_JSON_SCHEMA,
                "strict": True,
            }
        }
        inner = canonical_compliance_inner()
        envelope = build_envelope_strict_parsed_content(inner)

        result, mock_client = _run_through_openai_client(envelope)
        self.assertTrue(result.passed)

        create_kwargs = mock_client.responses.create.call_args.kwargs
        self.assertIn("text", create_kwargs)
        schema_required = set(COMPLIANCE_RESPONSE_JSON_SCHEMA["required"])
        for field in schema_required:
            self.assertIn(field, json.dumps(inner))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_fenced_json_in_content_item_succeeds(self) -> None:
        result, _mock_client = _run_through_openai_client(
            build_envelope_fenced_json_text(canonical_compliance_inner()),
        )
        self.assertTrue(result.passed)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_legacy_response_normalized_only_via_adapter(self) -> None:
        with patch("engine.builder1_image_compliance_contract.logger") as mock_logger:
            result, _mock_client = _run_through_openai_client(
                build_envelope_nested_content(legacy_compliance_inner()),
            )
        self.assertTrue(result.passed)
        logged = " ".join(str(call) for call in mock_logger.info.call_args_list)
        self.assertIn("BUILDER1_IMAGE_COMPLIANCE_LEGACY_NORMALIZED", logged)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_invalid_inner_json_triggers_contract_repair_preserving_image(self) -> None:
        bad_envelope = build_envelope_nested_content({"unexpected": True})
        bad_envelope.output[0].content[0].text = "not-json-at-all"
        good_envelope = build_envelope_nested_content(canonical_compliance_inner())

        result, mock_client = _run_through_openai_client(
            bad_envelope,
            side_effect=[bad_envelope, good_envelope],
        )
        self.assertTrue(result.passed)
        self.assertEqual(mock_client.responses.create.call_count, 2)

        first_kwargs = mock_client.responses.create.call_args_list[0].kwargs
        second_kwargs = mock_client.responses.create.call_args_list[1].kwargs
        first_image = image_bytes_from_create_kwargs(first_kwargs)
        second_image = image_bytes_from_create_kwargs(second_kwargs)
        self.assertEqual(first_image, second_image)
        self.assertEqual(base64.b64decode(first_image or ""), TEST_IMAGE)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_second_malformed_result_enters_review_only_unavailable(self) -> None:
        bad_envelope = build_envelope_nested_content(PRODUCTION_FAILURE_INNER)
        bad_envelope.output[0].content[0].text = "still-not-json"

        with self.assertRaises(ImageComplianceUnavailableError) as ctx:
            _run_through_openai_client(
                bad_envelope,
                side_effect=[bad_envelope, bad_envelope],
            )
        self.assertEqual(ctx.exception.reason_code, "malformed_response")
        self.assertTrue(ctx.exception.contract_repair_attempted)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_production_malformed_response_fixture_no_longer_fails(self) -> None:
        production_envelope = build_envelope_production_failure_inner()
        extracted = _extract_response_text(production_envelope)
        decoded = coerce_review_dict(extracted)
        self.assertTrue(legacy_pass_gate_would_reject(decoded))

        with self.assertRaises(ImageComplianceResponseError):
            normalize_compliance_payload(decoded)

        repair_envelope = build_envelope_nested_content(canonical_compliance_inner())
        result, mock_client = _run_through_openai_client(
            production_envelope,
            side_effect=[production_envelope, repair_envelope],
        )
        self.assertTrue(result.passed)
        self.assertEqual(mock_client.responses.create.call_count, 2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_canonical_without_legacy_pass_parses_in_single_client_call(self) -> None:
        inner = canonical_compliance_inner()
        self.assertNotIn("pass", inner)
        envelope = build_envelope_nested_content(inner)

        decoded = json.loads(_extract_response_text(envelope))
        self.assertTrue(legacy_pass_gate_would_reject(decoded))

        result, mock_client = _run_through_openai_client(envelope)
        self.assertTrue(result.passed)
        mock_client.responses.create.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_no_direct_parser_bypass_in_production_shaped_tests(self) -> None:
        with patch(
            "engine.builder1_image_compliance.parse_image_compliance_response",
        ) as mock_direct_parser:
            _run_through_openai_client(
                build_envelope_nested_content(canonical_compliance_inner()),
            )
        mock_direct_parser.assert_not_called()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    def test_malformed_envelope_never_fails_open(self) -> None:
        envelope = build_envelope_nested_content({"reviewStatus": "completed"})
        with self.assertRaises(ImageComplianceUnavailableError):
            _run_through_openai_client(
                envelope,
                side_effect=[envelope, envelope],
            )


class TestPreFixRegressionProof(unittest.TestCase):
    def test_legacy_pass_gate_rejects_production_inner_shape(self) -> None:
        self.assertTrue(legacy_pass_gate_would_reject(PRODUCTION_FAILURE_INNER))

    def test_legacy_pass_gate_rejects_canonical_without_pass(self) -> None:
        self.assertTrue(legacy_pass_gate_would_reject(canonical_compliance_inner()))

    def test_current_contract_accepts_canonical_without_pass(self) -> None:
        normalized = normalize_compliance_payload(canonical_compliance_inner())
        self.assertTrue(normalized.reviewer_pass)
        self.assertEqual(normalized.legacy_shape, "canonical")


if __name__ == "__main__":
    unittest.main()
