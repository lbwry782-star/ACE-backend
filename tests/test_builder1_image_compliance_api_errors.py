"""
Builder1 image compliance API request failure, schema fallback, and strict schema tests.

Run: python -m unittest tests.test_builder1_image_compliance_api_errors -v
"""
from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder1_image_compliance import (
    IMAGE_COMPLIANCE_SYSTEM_PROMPT,
    ImageComplianceResponseError,
    ImageComplianceUnavailableError,
    _log_api_rejected,
    _openai_compliance_responses_call,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_compliance_contract import (
    COMPLIANCE_RESPONSE_JSON_SCHEMA,
    COMPLIANCE_SCHEMA_VERSION,
    build_compliance_responses_request_kwargs,
    build_text_format_for_compliance,
    compliance_strict_json_schema,
    normalize_compliance_payload,
)
from engine.builder1_strict_schema import (
    classify_compliance_api_error,
    find_strict_schema_errors,
    is_invalid_json_schema_api_error,
)
from tests.builder1_responses_api_fixtures import (
    canonical_compliance_inner,
    image_bytes_from_create_kwargs,
)
from tests.builder1_test_helpers import pass_compliance_reviewer
from tests.openai_sdk_test_helpers import build_actual_openai_sdk_response

TEST_IMAGE = b"api-error-test-image-bytes"


class _FakeBadRequestError(Exception):
    status_code = 400
    request_id = "req-test-400"

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_json_schema",
        param: str = "text.format.schema",
    ):
        self.code = code
        self.param = param
        self.body = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
                "param": param,
            }
        }
        super().__init__(message)


class _FakeAuthError(Exception):
    status_code = 401
    request_id = "req-test-401"

    def __init__(self, message: str = "Incorrect API key"):
        self.code = "invalid_api_key"
        self.param = None
        self.body = {"error": {"message": message, "type": "invalid_request_error", "code": self.code}}
        super().__init__(message)


class _FakeRateLimitError(Exception):
    status_code = 429
    request_id = "req-test-429"

    def __init__(self, message: str = "Rate limit exceeded"):
        self.code = "rate_limit_exceeded"
        self.param = None
        self.body = {"error": {"message": message, "type": "rate_limit_error", "code": self.code}}
        super().__init__(message)


def _mock_openai_client(create_side_effect=None, create_return=None) -> MagicMock:
    mock_client = MagicMock()
    if create_side_effect is not None:
        mock_client.responses.create.side_effect = create_side_effect
    else:
        mock_client.responses.create.return_value = create_return
    return mock_client


class TestApiErrorLogging(unittest.TestCase):
    def test_bad_request_logs_status_code_param_message_and_request_id(self) -> None:
        exc = _FakeBadRequestError(
            "Invalid schema for response_format 'builder1_image_compliance_v2': "
            "evidence.items is missing required fields.",
            code="invalid_json_schema",
            param="text.format.schema",
        )
        with self.assertLogs("engine.builder1_image_compliance", level="ERROR") as logs:
            details = _log_api_rejected(
                exc=exc,
                campaign_id="cmp-log",
                job_id="job-log",
                ad_index=2,
                schema_mode="strict",
            )
        joined = "\n".join(logs.output)
        self.assertIn("BUILDER1_IMAGE_COMPLIANCE_API_REJECTED", joined)
        self.assertIn("statusCode=400", joined)
        self.assertIn("errorCode=invalid_json_schema", joined)
        self.assertIn("errorParam=text.format.schema", joined)
        self.assertIn("Invalid schema for response_format", joined)
        self.assertIn("requestId=req-test-400", joined)
        self.assertIn(f"schemaVersion={COMPLIANCE_SCHEMA_VERSION}", joined)
        self.assertEqual(details["errorCode"], "invalid_json_schema")
        self.assertEqual(details["errorParam"], "text.format.schema")


class TestRequestFailureClassification(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance", side_effect=lambda: _strict_text_format())
    @patch("engine.builder1_planning_model.strict_json_schema_available", return_value=True)
    @patch("openai.OpenAI")
    def test_http_400_is_request_rejected_not_malformed_response(self, mock_openai_cls: MagicMock, *_mocks: Any) -> None:
        mock_openai_cls.return_value = _mock_openai_client(
            create_side_effect=_FakeBadRequestError(
                "Unknown parameter: foo",
                code="unknown_parameter",
                param="foo",
            )
        )
        with self.assertRaises(ImageComplianceUnavailableError) as ctx:
            _openai_compliance_responses_call(
                image_bytes=TEST_IMAGE,
                product_name="TestBrand",
                product_description="desc",
                visibility_policy="FORBIDDEN",
                transferred_object="",
                ad_index=1,
                allow_schema_fallback=False,
            )
        self.assertEqual(ctx.exception.reason_code, "request_rejected")
        self.assertNotEqual(ctx.exception.reason_code, "malformed_response")

    def test_classify_auth_and_rate_limit(self) -> None:
        self.assertEqual(classify_compliance_api_error(_FakeAuthError()), "review_auth_error")
        self.assertEqual(classify_compliance_api_error(_FakeRateLimitError()), "review_rate_limited")


def _strict_text_format() -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": COMPLIANCE_SCHEMA_VERSION,
            "schema": compliance_strict_json_schema(),
            "strict": True,
        }
    }


class TestSchemaFallback(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance", side_effect=lambda: _strict_text_format())
    @patch("openai.OpenAI")
    def test_schema_related_400_triggers_one_plain_json_fallback(self, mock_openai_cls: MagicMock, *_mocks: Any) -> None:
        sdk_response = build_actual_openai_sdk_response(canonical_compliance_inner())
        calls: List[Dict[str, Any]] = []

        def create_side_effect(**kwargs: Any) -> object:
            calls.append(kwargs)
            if len(calls) == 1:
                raise _FakeBadRequestError(
                    "Invalid schema for response_format 'builder1_image_compliance_v2'",
                    code="invalid_json_schema",
                    param="text.format.schema",
                )
            return sdk_response

        mock_openai_cls.return_value = _mock_openai_client(create_side_effect=create_side_effect)

        with self.assertLogs("engine.builder1_image_compliance", level="INFO") as logs:
            out_text = _openai_compliance_responses_call(
                image_bytes=TEST_IMAGE,
                product_name="TestBrand",
                product_description="desc",
                visibility_policy="FORBIDDEN",
                transferred_object="",
                campaign_id="cmp-fallback",
                ad_index=1,
            )
        self.assertIn("reviewStatus", out_text)
        self.assertEqual(len(calls), 2)
        self.assertIn("text", calls[0])
        self.assertNotIn("text", calls[1])
        joined = "\n".join(logs.output)
        self.assertIn("BUILDER1_IMAGE_COMPLIANCE_SCHEMA_FALLBACK", joined)
        self.assertIn("fromMode=strict_json_schema", joined)
        self.assertIn("toMode=plain_json", joined)
        self.assertIn("openaiErrorCode=invalid_json_schema", joined)
        self.assertIn("openaiErrorParam=text.format.schema", joined)

        first_image = image_bytes_from_create_kwargs(calls[0])
        second_image = image_bytes_from_create_kwargs(calls[1])
        self.assertEqual(first_image, second_image)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance", side_effect=lambda: _strict_text_format())
    @patch("openai.OpenAI")
    def test_successful_fallback_proceeds_to_adjudication(self, mock_openai_cls: MagicMock, *_mocks: Any) -> None:
        sdk_response = build_actual_openai_sdk_response(canonical_compliance_inner())

        def create_side_effect(**kwargs: Any) -> object:
            if "text" in kwargs:
                raise _FakeBadRequestError(
                    "Invalid schema for response_format",
                    code="invalid_json_schema",
                    param="text.format.schema",
                )
            return sdk_response

        mock_openai_cls.return_value = _mock_openai_client(create_side_effect=create_side_effect)
        result = review_builder1_ad_image_compliance(
            TEST_IMAGE,
            product_name="TestBrand",
            ad_index=1,
        )
        self.assertTrue(result.passed)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance", side_effect=lambda: _strict_text_format())
    @patch("openai.OpenAI")
    def test_second_failure_preserves_review_only_reason(self, mock_openai_cls: MagicMock, *_mocks: Any) -> None:
        def create_side_effect(**kwargs: Any) -> object:
            raise _FakeBadRequestError(
                "Invalid schema for response_format",
                code="invalid_json_schema",
                param="text.format.schema",
            )

        mock_openai_cls.return_value = _mock_openai_client(create_side_effect=create_side_effect)
        with self.assertRaises(ImageComplianceUnavailableError) as ctx:
            review_builder1_ad_image_compliance(
                TEST_IMAGE,
                product_name="TestBrand",
                ad_index=1,
            )
        self.assertEqual(ctx.exception.reason_code, "request_rejected")
        self.assertIsNotNone(ctx.exception.image_bytes)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance", side_effect=lambda: _strict_text_format())
    @patch("openai.OpenAI")
    def test_non_schema_400_does_not_trigger_schema_fallback(self, mock_openai_cls: MagicMock, *_mocks: Any) -> None:
        mock_openai_cls.return_value = _mock_openai_client(
            create_side_effect=_FakeBadRequestError(
                "Invalid image data",
                code="invalid_value",
                param="input[0].content[1].image_url",
            )
        )
        with self.assertRaises(ImageComplianceUnavailableError):
            _openai_compliance_responses_call(
                image_bytes=TEST_IMAGE,
                product_name="TestBrand",
                product_description="desc",
                visibility_policy="FORBIDDEN",
                transferred_object="",
                ad_index=1,
            )
        self.assertEqual(mock_openai_cls.return_value.responses.create.call_count, 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance", side_effect=lambda: _strict_text_format())
    @patch("openai.OpenAI")
    def test_authentication_failure_does_not_trigger_schema_fallback(self, mock_openai_cls: MagicMock, *_mocks: Any) -> None:
        mock_openai_cls.return_value = _mock_openai_client(create_side_effect=_FakeAuthError())
        with self.assertRaises(ImageComplianceUnavailableError) as ctx:
            _openai_compliance_responses_call(
                image_bytes=TEST_IMAGE,
                product_name="TestBrand",
                product_description="desc",
                visibility_policy="FORBIDDEN",
                transferred_object="",
                ad_index=1,
            )
        self.assertEqual(ctx.exception.reason_code, "review_auth_error")
        self.assertEqual(mock_openai_cls.return_value.responses.create.call_count, 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance.build_text_format_for_compliance", side_effect=lambda: _strict_text_format())
    @patch("openai.OpenAI")
    def test_rate_limiting_does_not_trigger_schema_fallback(self, mock_openai_cls: MagicMock, *_mocks: Any) -> None:
        mock_openai_cls.return_value = _mock_openai_client(create_side_effect=_FakeRateLimitError())
        with self.assertRaises(ImageComplianceUnavailableError) as ctx:
            _openai_compliance_responses_call(
                image_bytes=TEST_IMAGE,
                product_name="TestBrand",
                product_description="desc",
                visibility_policy="FORBIDDEN",
                transferred_object="",
                ad_index=1,
            )
        self.assertEqual(ctx.exception.reason_code, "review_rate_limited")
        self.assertEqual(mock_openai_cls.return_value.responses.create.call_count, 1)


class TestStrictComplianceSchema(unittest.TestCase):
    def test_every_strict_object_has_required_and_additional_properties_false(self) -> None:
        prepared = compliance_strict_json_schema()

        def walk(node: Any, path: str = "$") -> None:
            if not isinstance(node, dict):
                return
            raw_type = node.get("type")
            types = [raw_type] if isinstance(raw_type, str) else list(raw_type or [])
            if "object" in types:
                self.assertFalse(node.get("additionalProperties"), msg=path)
                props = node.get("properties") or {}
                required = node.get("required") or []
                self.assertEqual(set(required), set(props.keys()), msg=path)
            if node.get("type") == "array" and "items" in node:
                walk(node["items"], f"{path}.items")
            for name, sub in (node.get("properties") or {}).items():
                walk(sub, f"{path}.properties.{name}")

        walk(prepared)

    def test_evidence_items_have_valid_strict_structure(self) -> None:
        prepared = compliance_strict_json_schema()
        evidence_items = prepared["properties"]["evidence"]["items"]
        self.assertEqual(len(evidence_items["required"]), len(evidence_items["properties"]))
        self.assertIn("code", evidence_items["required"])
        self.assertFalse(evidence_items.get("additionalProperties"))

    def test_obsolete_pass_not_in_schema(self) -> None:
        schema_text = json.dumps(compliance_strict_json_schema())
        self.assertNotIn('"pass"', schema_text)
        self.assertNotIn('"violations"', schema_text)

    def test_local_precheck_has_no_strict_schema_errors(self) -> None:
        errors = find_strict_schema_errors(compliance_strict_json_schema())
        self.assertEqual(errors, [])

    def test_authoritative_request_builder_matches_text_format(self) -> None:
        with patch("engine.builder1_planning_model.strict_json_schema_available", return_value=True):
            kwargs = build_compliance_responses_request_kwargs(
                model="gpt-4o",
                image_bytes=TEST_IMAGE,
                system_prompt=IMAGE_COMPLIANCE_SYSTEM_PROMPT,
                product_name="TestBrand",
                product_description="desc",
                visibility_policy="FORBIDDEN",
                transferred_object="",
                schema_mode="strict",
            )
            text_format = build_text_format_for_compliance()
        self.assertEqual(kwargs["text"], text_format)
        for field in COMPLIANCE_RESPONSE_JSON_SCHEMA["required"]:
            self.assertIn(field, kwargs["text"]["format"]["schema"]["properties"])


class TestCanonicalParserFailClosed(unittest.TestCase):
    def test_malformed_payload_still_fail_closed(self) -> None:
        with self.assertRaises(ImageComplianceResponseError):
            normalize_compliance_payload({"reviewStatus": "completed"})


class TestReviewOnlyUsesCorrectedBuilder(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_planning_model.strict_json_schema_available", return_value=True)
    @patch("engine.builder1_image_compliance.build_compliance_responses_request_kwargs", wraps=build_compliance_responses_request_kwargs)
    @patch("openai.OpenAI")
    def test_review_only_path_uses_request_builder(
        self,
        mock_openai_cls: MagicMock,
        mock_builder: MagicMock,
        *_mocks: Any,
    ) -> None:
        sdk_response = build_actual_openai_sdk_response(canonical_compliance_inner())
        mock_openai_cls.return_value = _mock_openai_client(create_return=sdk_response)
        review_builder1_ad_image_compliance(
            TEST_IMAGE,
            product_name="TestBrand",
            ad_index=1,
        )
        self.assertGreaterEqual(mock_builder.call_count, 1)


class TestQualityPlanningUnchanged(unittest.TestCase):
    def test_quality_planning_call_budget_unchanged(self) -> None:
        from engine.builder1_planning_metrics import (
            NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
            NORMAL_PLANNING_CALLS_WITH_NAME,
        )

        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME, 6)


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_has_no_compliance_request_builder(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("build_compliance_responses_request_kwargs", text)


class TestZeroPlanningAndImageOnRequestFailure(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_planning_model.strict_json_schema_available", return_value=True)
    @patch("engine.builder1_image_generator.generate_builder1_ad_image")
    @patch("engine.builder1_planner.plan_builder1")
    @patch("openai.OpenAI")
    def test_request_failure_makes_zero_planning_and_image_calls(
        self,
        mock_openai_cls: MagicMock,
        mock_plan: MagicMock,
        mock_image: MagicMock,
        *_mocks: Any,
    ) -> None:
        mock_openai_cls.return_value = _mock_openai_client(
            create_side_effect=_FakeBadRequestError(
                "Invalid image data",
                code="invalid_value",
                param="input[0].content[1].image_url",
            )
        )
        with self.assertRaises(ImageComplianceUnavailableError):
            review_builder1_ad_image_compliance(
                TEST_IMAGE,
                product_name="TestBrand",
                ad_index=1,
            )
        mock_plan.assert_not_called()
        mock_image.assert_not_called()


class TestIsInvalidJsonSchemaApiError(unittest.TestCase):
    def test_detects_schema_response_format_errors(self) -> None:
        exc = _FakeBadRequestError(
            "Invalid schema for response_format 'builder1_image_compliance_v2'",
            code="invalid_json_schema",
            param="text.format.schema",
        )
        self.assertTrue(is_invalid_json_schema_api_error(exc))

    def test_rejects_unrelated_bad_request(self) -> None:
        exc = _FakeBadRequestError(
            "Invalid image data",
            code="invalid_value",
            param="input[0].content[1].image_url",
        )
        self.assertFalse(is_invalid_json_schema_api_error(exc))


if __name__ == "__main__":
    unittest.main()
