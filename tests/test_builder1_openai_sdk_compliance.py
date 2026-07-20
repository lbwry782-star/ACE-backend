"""
Tests that use actual openai==2.30.0 SDK response models at the production API boundary.

Run: python -m unittest tests.test_builder1_openai_sdk_compliance -v
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from engine.builder1_image_compliance import review_builder1_ad_image_compliance
from engine.builder1_image_compliance_contract import (
    COMPLIANCE_SCHEMA_VERSION,
    COMPLIANCE_RESPONSE_JSON_SCHEMA,
)
from tests.builder1_responses_api_fixtures import canonical_compliance_inner, image_bytes_from_create_kwargs
from tests.openai_sdk_test_helpers import Response, build_actual_openai_sdk_response

TEST_IMAGE = b"actual-openai-sdk-image-bytes"


class TestActualOpenAiSdkCompliance(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BUILDER1_DISABLE_STRICT_SCHEMA": "1"}, clear=False)
    @patch("httpx.Timeout", return_value=MagicMock())
    @patch("openai.OpenAI")
    def test_image_compliance_parses_actual_openai_sdk_response_object(
        self,
        mock_openai_cls: MagicMock,
        _mock_timeout: MagicMock,
    ) -> None:
        inner = canonical_compliance_inner()
        sdk_response = build_actual_openai_sdk_response(inner)
        self.assertIsInstance(sdk_response, Response)
        self.assertEqual(sdk_response.output[0].type, "message")
        self.assertEqual(sdk_response.output[0].content[0].type, "output_text")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.return_value = sdk_response

        result = review_builder1_ad_image_compliance(
            TEST_IMAGE,
            product_name="TestBrand",
            ad_index=1,
            campaign_id="cmp-actual-sdk",
            job_id="job-actual-sdk",
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.hard_violations, [])
        mock_client.responses.create.assert_called_once()
        returned = mock_client.responses.create.return_value
        self.assertIsInstance(returned, Response)
        self.assertIn("reviewStatus", returned.output_text)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_planning_model.strict_json_schema_available", return_value=True)
    @patch("engine.builder1_planning_model._responses_create_supports_text_parameter", return_value=True)
    @patch("httpx.Timeout", return_value=MagicMock())
    @patch("openai.OpenAI")
    def test_compliance_responses_create_receives_canonical_strict_schema(
        self,
        mock_openai_cls: MagicMock,
        _mock_timeout: MagicMock,
        _mock_supports_text: MagicMock,
        _mock_strict_available: MagicMock,
    ) -> None:
        sdk_response = build_actual_openai_sdk_response(canonical_compliance_inner())
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.return_value = sdk_response

        review_builder1_ad_image_compliance(
            TEST_IMAGE,
            product_name="TestBrand",
            ad_index=1,
        )

        mock_client.responses.create.assert_called_once()
        kwargs = mock_client.responses.create.call_args.kwargs

        self.assertEqual(kwargs.get("model"), "gpt-4o")
        self.assertIn("input", kwargs)
        content = kwargs["input"][0]["content"]
        self.assertTrue(any(item.get("type") == "input_image" for item in content))
        self.assertTrue(any(item.get("type") == "input_text" for item in content))

        text_config = kwargs.get("text") or {}
        fmt = text_config.get("format") or {}
        self.assertEqual(fmt.get("type"), "json_schema")
        self.assertEqual(fmt.get("name"), COMPLIANCE_SCHEMA_VERSION)
        self.assertTrue(fmt.get("strict"))

        schema = fmt.get("schema") or {}
        for field in COMPLIANCE_RESPONSE_JSON_SCHEMA["required"]:
            self.assertIn(field, schema.get("properties", {}))

        prompt_text = next(item["text"] for item in content if item.get("type") == "input_text")
        self.assertIn("reviewStatus", prompt_text)
        self.assertIn("hardViolations", prompt_text)
        self.assertNotIn('"pass": true', prompt_text)
        self.assertNotIn('"pass": false', prompt_text)
        self.assertNotIn('"violations":', prompt_text.split("Return JSON only")[-1][:200])

        image_b64 = image_bytes_from_create_kwargs(kwargs)
        self.assertEqual(image_b64, __import__("base64").b64encode(TEST_IMAGE).decode("ascii"))


if __name__ == "__main__":
    unittest.main()
