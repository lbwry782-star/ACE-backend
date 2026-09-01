"""
Builder1 stage parser ↔ API schema contract tests.

Run: python -m unittest tests.test_builder1_stage_contract_preflight -v
"""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import MagicMock, patch

from engine.builder1_direct_product_route import (
    DIRECT_PRODUCT_ROUTE_ASSESSMENT_NESTED_FIELDS,
    direct_product_route_assessment_json_schema,
)
from engine.builder1_final_stages import parse_brand_physical_output
from engine.builder1_planning_contract import STAGE_BRAND_PHYSICAL_SYSTEM
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planning_model import (
    BRAND_PHYSICAL_JSON_SCHEMA,
    STAGE_JSON_SCHEMAS,
    build_text_format_for_stage,
    call_planning_model,
)
from engine.builder1_planner import Builder1PlannerError, _run_stage
from engine.builder1_stage_contract_preflight import (
    STAGE_PARSER_API_CONTRACTS,
    verify_all_registered_stage_contracts,
    verify_stage_parser_api_contract,
)
from engine.builder1_strict_schema import StrictSchemaConfigurationError, prepare_strict_json_schema
from tests.builder1_test_helpers import direct_product_route_assessment
from tests.test_builder1_staged_planning import _brand_physical


class TestBrandPhysicalStrictSchema(unittest.TestCase):
    def test_effective_schema_contains_direct_product_route_assessment(self) -> None:
        prepared = prepare_strict_json_schema(BRAND_PHYSICAL_JSON_SCHEMA)
        self.assertIn("directProductRouteAssessment", prepared["properties"])
        self.assertIn("directProductRouteAssessment", prepared["required"])

    def test_nested_assessment_fields_required(self) -> None:
        prepared = prepare_strict_json_schema(BRAND_PHYSICAL_JSON_SCHEMA)
        assessment = prepared["properties"]["directProductRouteAssessment"]
        self.assertFalse(assessment.get("additionalProperties"))
        for field in DIRECT_PRODUCT_ROUTE_ASSESSMENT_NESTED_FIELDS:
            self.assertIn(field, assessment["properties"], msg=field)
            self.assertIn(field, assessment["required"], msg=field)

    def test_canonical_assessment_schema_matches_brand_physical_embed(self) -> None:
        canonical = direct_product_route_assessment_json_schema()
        embedded = BRAND_PHYSICAL_JSON_SCHEMA["properties"]["directProductRouteAssessment"]
        self.assertEqual(set(canonical["required"]), set(embedded["required"]))
        self.assertEqual(set(canonical["properties"].keys()), set(embedded["properties"].keys()))

    def test_build_text_format_for_brand_physical_includes_assessment(self) -> None:
        with patch(
            "engine.builder1_planning_model.strict_json_schema_available",
            return_value=True,
        ):
            text_format = build_text_format_for_stage("brand_physical")
        assert text_format is not None
        schema = text_format["format"]["schema"]
        self.assertIn("directProductRouteAssessment", schema["required"])
        assessment = schema["properties"]["directProductRouteAssessment"]
        self.assertIn("recommendedRoute", assessment["required"])

    def test_offline_effective_schema_snapshot(self) -> None:
        """Regression anchor: serialized effective schema must include assessment."""
        prepared = prepare_strict_json_schema(STAGE_JSON_SCHEMAS["brand_physical"])
        serialized = json.dumps(prepared, sort_keys=True)
        self.assertIn("directProductRouteAssessment", serialized)
        self.assertIn("productLedMechanismSummary", serialized)


class TestParserSchemaAlignment(unittest.TestCase):
    def test_registered_brand_physical_contract_passes(self) -> None:
        prepared = prepare_strict_json_schema(STAGE_JSON_SCHEMAS["brand_physical"])
        errors = verify_stage_parser_api_contract("brand_physical", prepared)
        self.assertEqual(errors, [])

    def test_all_registered_contracts_pass(self) -> None:
        prepared = {
            stage: prepare_strict_json_schema(schema)
            for stage, schema in STAGE_JSON_SCHEMAS.items()
            if stage in STAGE_PARSER_API_CONTRACTS
        }
        report = verify_all_registered_stage_contracts(prepared)
        self.assertEqual(report, {})

    def test_schema_valid_response_passes_parser(self) -> None:
        payload = _brand_physical()
        result = parse_brand_physical_output(payload, visibility_policy="CREATIVE_DECISION")
        self.assertIsNotNone(result.direct_product_route_assessment)

    def test_missing_assessment_fails_parser(self) -> None:
        payload = _brand_physical()
        payload.pop("directProductRouteAssessment", None)
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("physical_route_assessment_missing", str(ctx.exception))

    def test_legacy_plan_without_assessment_still_loadable(self) -> None:
        from tests.test_builder1_series import _base_campaign, _parse

        data = copy.deepcopy(_base_campaign(2))
        plan = _parse(data, 2)
        self.assertNotIn("directProductRouteAssessment", (plan.planning_internals or {}))


class TestPreflightBlocksPaidCalls(unittest.TestCase):
    def test_drift_detected_before_model_submission(self) -> None:
        bad_schema = copy.deepcopy(BRAND_PHYSICAL_JSON_SCHEMA)
        bad_schema["properties"].pop("directProductRouteAssessment", None)
        bad_schema["required"] = [
            field for field in bad_schema["required"] if field != "directProductRouteAssessment"
        ]
        with patch.dict(STAGE_JSON_SCHEMAS, {"brand_physical": bad_schema}, clear=False):
            with patch(
                "engine.builder1_planning_model.strict_json_schema_available",
                return_value=True,
            ):
                with self.assertRaises(StrictSchemaConfigurationError) as ctx:
                    build_text_format_for_stage("brand_physical")
        self.assertTrue(
            any("directProductRouteAssessment" in err for err in ctx.exception.errors),
            ctx.exception.errors,
        )

    def test_call_planning_model_zero_calls_on_contract_mismatch(self) -> None:
        client = MagicMock()
        bad_schema = copy.deepcopy(BRAND_PHYSICAL_JSON_SCHEMA)
        bad_schema["properties"].pop("directProductRouteAssessment", None)
        with patch.dict(STAGE_JSON_SCHEMAS, {"brand_physical": bad_schema}, clear=False):
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
                        stage="brand_physical",
                        parse_json_text=lambda text: text,
                    )
        client.responses.create.assert_not_called()

    def test_repair_retry_use_same_stage_schema(self) -> None:
        calls: list[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            calls.append(stage or "")
            with patch(
                "engine.builder1_planning_model.strict_json_schema_available",
                return_value=True,
            ):
                text_format = build_text_format_for_stage(stage or "brand_physical")
            assert text_format is not None
            self.assertIn("directProductRouteAssessment", text_format["format"]["schema"]["required"])
            raise StrictSchemaConfigurationError(["forced_stop"])

        with self.assertRaises(Builder1PlannerError):
            _run_stage(
                "brand_physical",
                model_caller,
                STAGE_BRAND_PHYSICAL_SYSTEM,
                "user",
                lambda raw: raw,
                repair_builder=lambda broken, reasons: "repair",
            )
        self.assertEqual(calls, ["brand_physical"])


class TestCallCountsUnchanged(unittest.TestCase):
    def test_supplied_name_calls(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)

    def test_generated_name_calls(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME, 6)


if __name__ == "__main__":
    unittest.main()
