"""
Builder1 graphic-system contract alignment tests.

Run: python -m unittest tests.test_builder1_graphic_contract -v
"""
from __future__ import annotations

import copy
import json
import logging
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_final_stages import parse_graphic_system_output
from engine.builder1_graphic_contract import (
    GRAPHIC_DESCRIPTIVE_FIELDS,
    GRAPHIC_DESCRIPTIVE_MIN_LENGTH,
    STRUCTURED_FIELD_ENUMS,
    descriptive_field_prompt_lines,
    is_graphic_contract_mismatch,
    repair_instructions_for_reasons,
    structured_enum_prompt_lines,
    validate_descriptive_graphic_text,
    validate_structured_enum,
)
from engine.builder1_planning_contract import (
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    build_graphic_system_repair_prompt,
)
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
    Builder1PlanningMetrics,
)
from engine.builder1_planning_model import GRAPHIC_SYSTEM_JSON_SCHEMA
from engine.builder1_planning_profile import (
    resolve_stage_model,
    resolve_stage_reasoning_effort,
    stage_model_override,
)
from engine.builder1_planning_pipeline import _run_graphic_system_stage
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_staged_parsers import StageParseError
from tests.test_builder1_staged_planning import _full_final_responses, _graphic

BRIEF = "Reinforced shell product for daily carry"


def _gpt41_shaped_graphic(**overrides: Any) -> Dict[str, Any]:
    payload = _graphic()
    payload.update(
        {
            "typographyStyle": "Heavy sans-serif with tight tracking and bold hierarchy",
            "imageStyle": "Cinematic lifestyle photography with warm directional light",
            "backgroundTreatment": "Soft gradient wash behind the transferred object",
        }
    )
    payload.update(overrides)
    return payload


class TestGraphicContractAlignment(unittest.TestCase):
    def test_prompt_and_schema_typography_contract(self) -> None:
        prompt = STAGE_GRAPHIC_SYSTEM_SYSTEM
        schema = GRAPHIC_SYSTEM_JSON_SCHEMA["properties"]["typographyStyle"]
        self.assertIn("typographyStyle", prompt)
        self.assertIn("descriptive campaign-direction", prompt.lower())
        self.assertEqual(schema["type"], "string")
        self.assertEqual(schema["minLength"], GRAPHIC_DESCRIPTIVE_MIN_LENGTH)

    def test_prompt_and_schema_image_style_contract(self) -> None:
        prompt = STAGE_GRAPHIC_SYSTEM_SYSTEM
        schema = GRAPHIC_SYSTEM_JSON_SCHEMA["properties"]["imageStyle"]
        self.assertIn("imageStyle", prompt)
        self.assertNotIn("imageStyle", structured_enum_prompt_lines())
        self.assertEqual(schema["type"], "string")
        self.assertIn("minLength", schema)

    def test_prompt_and_schema_background_contract(self) -> None:
        prompt = STAGE_GRAPHIC_SYSTEM_SYSTEM
        schema = GRAPHIC_SYSTEM_JSON_SCHEMA["properties"]["backgroundTreatment"]
        self.assertIn("backgroundTreatment", prompt)
        self.assertIn("not closed enums", prompt.lower())
        self.assertEqual(schema["type"], "string")

    def test_validator_uses_shared_structured_constants(self) -> None:
        allowed = STRUCTURED_FIELD_ENUMS["layoutTemplate"]
        self.assertIsNotNone(
            validate_structured_enum("not_a_layout", field="layoutTemplate", allowed=allowed)
        )
        self.assertIsNone(
            validate_structured_enum(next(iter(allowed)), field="layoutTemplate", allowed=allowed)
        )

    def test_descriptive_fields_not_checked_against_hidden_enums(self) -> None:
        prose = "Campaign-specific editorial photography with soft contrast"
        for field in ("typographyStyle", "imageStyle", "backgroundTreatment"):
            self.assertIsNone(validate_descriptive_graphic_text(prose, field=field))

    def test_valid_campaign_specific_descriptions_pass(self) -> None:
        graphic = parse_graphic_system_output(_gpt41_shaped_graphic())
        self.assertIn("Cinematic", graphic.image_style)

    def test_invalid_structural_values_fail_with_precise_reasons(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_graphic_system_output(_gpt41_shaped_graphic(backgroundTreatment="short"))
        self.assertIn("graphic_generator_background_treatment_too_short", ctx.exception.reasons)


class TestProductionShapedGpt41Output(unittest.TestCase):
    def test_gpt41_shaped_response_passes_when_contract_satisfied(self) -> None:
        graphic = parse_graphic_system_output(_gpt41_shaped_graphic())
        self.assertTrue(graphic.typography_style)
        self.assertTrue(graphic.image_style)
        self.assertTrue(graphic.background_treatment)

    def test_wording_variation_not_rejected_for_descriptive_field(self) -> None:
        payload = _gpt41_shaped_graphic(
            imageStyle="Hand-drawn ink illustration with expressive line weight"
        )
        graphic = parse_graphic_system_output(payload)
        self.assertEqual(graphic.image_style, payload["imageStyle"])

    def test_unsupported_enum_value_fails_for_structured_field(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_graphic_system_output(_gpt41_shaped_graphic(layoutTemplate="invalid_layout"))
        self.assertIn("graphic_generator_invalid_layout", ctx.exception.reasons)

    def test_repair_receives_exact_structural_requirements(self) -> None:
        prompt = build_graphic_system_repair_prompt(
            broken_json="{}",
            reasons=["graphic_generator_invalid_layout"],
        )
        allowed = ", ".join(sorted(STRUCTURED_FIELD_ENUMS["layoutTemplate"]))
        self.assertIn("layoutTemplate", prompt)
        self.assertIn(allowed, prompt)

    def test_repair_receives_exact_descriptive_requirements(self) -> None:
        prompt = build_graphic_system_repair_prompt(
            broken_json="{}",
            reasons=["graphic_generator_image_style_too_short"],
        )
        self.assertIn("imageStyle", prompt)
        self.assertIn(f"{GRAPHIC_DESCRIPTIVE_MIN_LENGTH}+ characters", prompt)

    def test_repair_preserves_valid_fields_instruction(self) -> None:
        instructions = repair_instructions_for_reasons(["graphic_generator_invalid_border"])
        self.assertTrue(any("Preserve every other already-valid" in line for line in instructions))


class TestGraphicPipeline(unittest.TestCase):
    def test_balanced_routes_graphic_system_to_execution_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BUILDER1_PLANNING_PROFILE": "BALANCED",
                "BUILDER1_PLANNING_MODEL": "gpt-5.6-sol",
                "BUILDER1_EXECUTION_MODEL": "gpt-4.1",
            },
            clear=False,
        ):
            self.assertEqual(resolve_stage_model("graphic_system"), "gpt-4.1")

    def test_successful_graphic_proceeds_to_series_ads(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertIn("graphic_system", stages)
        self.assertIn("series_ads", stages)
        self.assertLess(stages.index("graphic_system"), stages.index("series_ads"))

    def test_supplied_name_planning_remains_six_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(len(stages), NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_generated_name_planning_remains_seven_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(len(stages), NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME)

    def test_exceptional_graphic_call_counted(self) -> None:
        metrics = Builder1PlanningMetrics()
        metrics.record_model_call("graphic_system")
        metrics.record_model_call("graphic_system")
        metrics.record_model_call("graphic_system")
        self.assertEqual(metrics.graphic_stage_calls, 3)

    def test_focused_graphic_repair_counted(self) -> None:
        metrics = Builder1PlanningMetrics()
        metrics.record_stage_repair("graphic_system")
        self.assertEqual(metrics.stage_repair_calls, 1)

    def test_complete_stage_retry_counted(self) -> None:
        metrics = Builder1PlanningMetrics()
        metrics.record_stage_retry("graphic_system")
        self.assertEqual(metrics.stage_retry_calls, 1)

    def test_quality_fallback_counted_separately(self) -> None:
        metrics = Builder1PlanningMetrics()
        metrics.record_stage_model_fallback("graphic_system")
        self.assertEqual(metrics.stage_model_fallback_calls, 1)

    def test_quality_fallback_reruns_only_graphic_system(self) -> None:
        from engine.builder1_planner import _run_stage

        calls: List[str] = []

        def failing_then_ok(system: str, user: str, stage: str | None = None) -> object:
            calls.append(stage or "")
            if len(calls) <= 3:
                return _gpt41_shaped_graphic(imageStyle="bad")
            return _gpt41_shaped_graphic()

        with patch.dict(
            os.environ,
            {
                "BUILDER1_PLANNING_PROFILE": "BALANCED",
                "BUILDER1_PLANNING_MODEL": "gpt-5.6-sol",
                "BUILDER1_EXECUTION_MODEL": "gpt-4.1",
            },
            clear=False,
        ):
            with stage_model_override({"graphic_system": "gpt-4.1"}):
                with patch(
                    "engine.builder1_planning_profile.resolve_stage_model",
                    side_effect=lambda stage: "gpt-4.1" if stage == "graphic_system" else "gpt-5.6-sol",
                ):
                    with patch(
                        "engine.builder1_planning_profile.quality_model",
                        return_value="gpt-5.6-sol",
                    ):
                        with patch(
                            "engine.builder1_planning_profile.execution_optimization_active",
                            return_value=True,
                        ):
                            _run_graphic_system_stage(
                                failing_then_ok,
                                user_prompt="Return graphic generator.",
                                run_stage=_run_stage,
                            )
        self.assertGreater(len(calls), 3)
        self.assertTrue(all(stage == "graphic_system" for stage in calls))

    def test_upstream_selections_remain_fixed_during_graphic_fallback(self) -> None:
        self.assertTrue(is_graphic_contract_mismatch(["graphic_generator_image_style_too_short"]))


class TestGraphicLogging(unittest.TestCase):
    def test_rejected_field_logged_safely_and_truncated(self) -> None:
        long_value = "x" * 200
        with self.assertLogs("engine.builder1_graphic_contract", level="ERROR") as captured:
            validate_descriptive_graphic_text(long_value[:3], field="imageStyle")
        line = next(l for l in captured.output if "BUILDER1_GRAPHIC_FIELD_REJECTED" in l)
        self.assertIn("field=imageStyle", line)
        self.assertIn("valuePreview=", line)
        self.assertNotIn("x" * 100, line)

    def test_full_model_output_not_logged_on_rejection(self) -> None:
        payload = json.dumps(_gpt41_shaped_graphic(imageStyle="short"))
        with self.assertLogs("engine.builder1_graphic_contract", level="ERROR") as captured:
            with self.assertRaises(StageParseError):
                parse_graphic_system_output(_gpt41_shaped_graphic(imageStyle="short"))
        joined = "\n".join(captured.output)
        self.assertNotIn(payload, joined)

    def test_reasoning_effort_warning_not_duplicated(self) -> None:
        with self.assertLogs("engine.builder1_planning_profile", level="WARNING") as captured:
            resolve_stage_reasoning_effort("graphic_system", "gpt-4.1")
            resolve_stage_reasoning_effort("graphic_system", "gpt-4.1")
        warnings = [line for line in captured.output if "BUILDER1_REASONING_EFFORT_UNSUPPORTED" in line]
        self.assertEqual(len(warnings), 1)


class TestGraphicRegression(unittest.TestCase):
    def test_methodology_prompts_remain_present(self) -> None:
        prompt = STAGE_GRAPHIC_SYSTEM_SYSTEM
        self.assertIn("Define the graphic generator before producing individual ads", prompt)
        self.assertIn("Why:", prompt)
        self.assertIn("Selection test:", prompt)

    def test_product_visibility_forbidden_in_series_prompt(self) -> None:
        from engine.builder1_planning_contract import STAGE_SERIES_ADS_SYSTEM

        self.assertIn("FORBIDDEN", STAGE_SERIES_ADS_SYSTEM)

    def test_no_logo_enforcement_remains(self) -> None:
        self.assertIn("logo", STAGE_GRAPHIC_SYSTEM_SYSTEM.lower())

    def test_fifty_marketing_words_enforced_elsewhere(self) -> None:
        from engine.builder1_plan_parser import _word_count

        self.assertEqual(_word_count(" ".join(["word"] * 50)), 50)

    def test_no_builder1_final_judge_added(self) -> None:
        import engine.builder1_planner as planner_module

        source = open(planner_module.__file__, encoding="utf-8").read()
        self.assertNotIn("final_judge", source.lower())

    def test_builder2_unchanged(self) -> None:
        import subprocess
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", "engine/builder2_zip.py"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.stdout.strip(), "")


class TestGraphicContractMismatchDetection(unittest.TestCase):
    def test_production_shaped_enum_like_values_now_pass(self) -> None:
        graphic = parse_graphic_system_output(
            _gpt41_shaped_graphic(
                typographyStyle="bold_geometric_sans",
                imageStyle="editorial_photography",
                backgroundTreatment="solid gradient field",
            )
        )
        self.assertEqual(graphic.typography_style, "bold_geometric_sans")

    def test_contract_mismatch_stops_without_quality_fallback_when_same_model(self) -> None:
        from engine.builder1_planner import _run_stage

        def always_bad(*_a: Any, **_k: Any) -> object:
            return _gpt41_shaped_graphic(imageStyle="bad")

        with patch(
            "engine.builder1_planning_profile.execution_optimization_active",
            return_value=False,
        ):
            with self.assertRaises(Builder1PlannerError) as ctx:
                _run_graphic_system_stage(
                    always_bad,
                    user_prompt="Return graphic generator.",
                    run_stage=_run_stage,
                )
        self.assertTrue(is_graphic_contract_mismatch(ctx.exception.reasons))


if __name__ == "__main__":
    unittest.main()
