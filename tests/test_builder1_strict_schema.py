"""
Builder1 strict JSON schema normalization and validation tests.

Run: python -m unittest tests.test_builder1_strict_schema -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder1_final_stages import parse_series_ads_output
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_marketing_copy import validate_marketing_text_50_words
from engine.builder1_planning_contract import STAGE_SERIES_ADS_SYSTEM
from engine.builder1_planning_model import (
    SERIES_ADS_JSON_SCHEMA,
    STAGE_JSON_SCHEMAS,
    build_text_format_for_stage,
    call_planning_model,
)
from engine.builder1_planner import Builder1PlannerError, _run_stage, plan_builder1
from engine.builder1_strategy_judge import BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT
from engine.builder1_strict_schema import (
    StrictSchemaConfigurationError,
    find_strict_schema_errors,
    is_invalid_json_schema_api_error,
    normalize_strict_json_schema,
    prepare_strict_json_schema,
)
from tests.builder1_test_helpers import marketing_text_words
from tests.test_builder1_staged_planning import (
    _full_final_responses,
    _series_ads,
)


def _node_at_path(schema: Any, path: str) -> Any:
    node = schema
    for segment in path.replace("$.", "").split("."):
        if not segment or segment == "$":
            continue
        if segment == "items":
            node = node["items"]
        elif segment == "properties":
            continue
        else:
            node = node["properties"][segment]
    return node


def _object_paths(schema: Any, path: str = "$") -> List[str]:
    paths: List[str] = []
    if not isinstance(schema, dict):
        return paths
    raw_type = schema.get("type")
    types = [raw_type] if isinstance(raw_type, str) else list(raw_type or [])
    if "object" in types:
        paths.append(path)
    if isinstance(schema.get("properties"), dict):
        for name, subschema in schema["properties"].items():
            paths.extend(_object_paths(subschema, f"{path}.properties.{name}"))
    if schema.get("type") == "array" and "items" in schema:
        paths.extend(_object_paths(schema["items"], f"{path}.items"))
    return paths


class TestSeriesAdsSchemaObjects(unittest.TestCase):
    def test_every_series_ads_object_has_additional_properties_false(self) -> None:
        prepared = prepare_strict_json_schema(SERIES_ADS_JSON_SCHEMA)
        for path in _object_paths(prepared):
            node = _node_at_path(prepared, path)
            self.assertFalse(
                node.get("additionalProperties"),
                msg=f"{path} missing additionalProperties:false",
            )

    def test_ads_items_additional_properties_false(self) -> None:
        prepared = prepare_strict_json_schema(SERIES_ADS_JSON_SCHEMA)
        self.assertFalse(prepared["properties"]["ads"]["items"]["additionalProperties"])

    def test_series_generator_additional_properties_false(self) -> None:
        prepared = prepare_strict_json_schema(SERIES_ADS_JSON_SCHEMA)
        self.assertFalse(prepared["properties"]["seriesGenerator"]["additionalProperties"])

    def test_graphic_system_palette_additional_properties_false(self) -> None:
        prepared = prepare_strict_json_schema(STAGE_JSON_SCHEMAS["graphic_system"])
        self.assertFalse(prepared["properties"]["palette"]["additionalProperties"])


class TestStrictSchemaNormalizer(unittest.TestCase):
    def test_normalizer_fixes_deeply_nested_objects(self) -> None:
        raw = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "object", "properties": {"value": {"type": "string"}}},
                    },
                },
            },
        }
        normalized = normalize_strict_json_schema(raw)
        self.assertFalse(normalized["additionalProperties"])
        self.assertFalse(normalized["properties"]["outer"]["additionalProperties"])
        self.assertFalse(normalized["properties"]["outer"]["properties"]["inner"]["additionalProperties"])

    def test_normalizer_recurses_through_arrays(self) -> None:
        raw = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"name": {"type": "string"}}},
                },
            },
        }
        normalized = normalize_strict_json_schema(raw)
        self.assertFalse(normalized["properties"]["items"]["items"]["additionalProperties"])

    def test_normalizer_preserves_enums_and_nullable_fields(self) -> None:
        raw = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "closed"]},
                "headline": {"type": ["string", "null"]},
            },
            "required": ["status", "headline"],
        }
        normalized = normalize_strict_json_schema(raw)
        self.assertEqual(normalized["properties"]["status"]["enum"], ["open", "closed"])
        self.assertEqual(normalized["properties"]["headline"]["type"], ["string", "null"])
        self.assertEqual(normalized["required"], ["status", "headline"])


class TestStrictSchemaPrecheck(unittest.TestCase):
    def test_precheck_reports_exact_missing_object_path(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ads": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"index": {"type": "integer"}}},
                },
            },
        }
        errors = find_strict_schema_errors(schema)
        self.assertIn("$.properties.ads.items:missing_additionalProperties_false", errors)

    def test_invalid_schema_rejected_before_api_call(self) -> None:
        client = MagicMock()
        bad_schema = copy.deepcopy(SERIES_ADS_JSON_SCHEMA)
        bad_schema["required"] = ["seriesGenerator", "ads", "unexpectedField"]
        with patch.dict(STAGE_JSON_SCHEMAS, {"series_ads": bad_schema}, clear=False):
            with patch(
                "engine.builder1_planning_model.strict_json_schema_available",
                return_value=True,
            ):
                with self.assertRaises(StrictSchemaConfigurationError) as ctx:
                    call_planning_model(
                        client,
                        model="gpt-test",
                        system_prompt="sys",
                        user_prompt="user",
                        stage="series_ads",
                        parse_json_text=lambda text: text,
                    )
        self.assertTrue(any("unexpectedField" in err for err in ctx.exception.errors))
        client.responses.create.assert_not_called()


class TestInvalidJsonSchemaRetry(unittest.TestCase):
    def test_invalid_json_schema_does_not_retry(self) -> None:
        calls: List[int] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            calls.append(1)
            raise StrictSchemaConfigurationError(["properties.ads.items:missing_additionalProperties_false"])

        with self.assertRaises(Builder1PlannerError) as ctx:
            _run_stage(
                "series_ads",
                model_caller,
                "system",
                "user",
                lambda raw: raw,
            )
        self.assertIn("series_ads_failed", str(ctx.exception))
        self.assertEqual(len(calls), 1)

    def test_call_planning_model_maps_api_invalid_schema(self) -> None:
        client = MagicMock()
        client.responses.create.side_effect = Exception("Invalid schema for response_format 'builder1_series_ads'")

        with patch(
            "engine.builder1_planning_model.strict_json_schema_available",
            return_value=True,
        ):
            with self.assertRaises(StrictSchemaConfigurationError):
                call_planning_model(
                    client,
                    model="gpt-test",
                    system_prompt="sys",
                    user_prompt="user",
                    stage="series_ads",
                    parse_json_text=lambda text: text,
                )
        self.assertEqual(client.responses.create.call_count, 1)

    def test_is_invalid_json_schema_api_error(self) -> None:
        class ApiError(Exception):
            code = "invalid_json_schema"

        self.assertTrue(is_invalid_json_schema_api_error(ApiError("Invalid schema for response_format")))


class TestStrictSchemaIntegration(unittest.TestCase):
    BRIEF = "Reinforced shell product for daily carry"

    def test_valid_strict_series_ads_response_parses(self) -> None:
        payload = _series_ads(2)
        result = parse_series_ads_output(payload, expected_ad_count=2)
        self.assertEqual(len(result.ads), 2)

    def test_build_text_format_for_series_ads_is_strict(self) -> None:
        with patch(
            "engine.builder1_planning_model.strict_json_schema_available",
            return_value=True,
        ):
            text_format = build_text_format_for_stage("series_ads")
        self.assertIsNotNone(text_format)
        schema = text_format["format"]["schema"]
        self.assertTrue(text_format["format"]["strict"])
        self.assertFalse(schema["properties"]["ads"]["items"]["additionalProperties"])

    def test_ad_count_four_produces_four_planned_ads(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(4).get(system, {"pass": True}))

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=4,
        )
        self.assertEqual(plan.ad_count, 4)
        self.assertEqual(len(plan.ads), 4)

    def test_successful_series_ads_reaches_judge(self) -> None:
        judge_calls: List[int] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT:
                judge_calls.append(1)
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True}))

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertEqual(len(judge_calls), 1)

    def test_successful_judge_reaches_image_generation(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True}))

        plan = plan_builder1(
            product_name="",
            product_description=self.BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        image_calls: List[int] = []

        def image_caller(_p: str, _f: str) -> bytes:
            image_calls.append(1)
            return b"jpeg"

        generate_builder1_ad_image(plan, 1, image_caller)
        self.assertEqual(len(image_calls), 1)

    def test_exactly_fifty_marketing_words_remain_valid(self) -> None:
        text = marketing_text_words(50)
        validate_marketing_text_50_words(text)
        payload = _series_ads(2)
        for ad in payload["ads"]:
            validate_marketing_text_50_words(ad["marketingText"])


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_has_no_strict_schema_module(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_strict_schema", text)


if __name__ == "__main__":
    unittest.main()
