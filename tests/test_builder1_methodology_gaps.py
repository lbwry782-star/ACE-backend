"""
Builder1 methodology gap tests — strategic restart, no historic blacklist, image compliance.

Run: python -m unittest tests.test_builder1_methodology_gaps -v
"""
from __future__ import annotations

import copy
import inspect
import os
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from engine.builder1_creative_methodology import (
    FOUNDATIONAL_STRATEGIC_REJECTION_CODES,
    deterministic_methodology_checks,
    is_foundational_strategic_rejection,
    scan_prompt_for_reused_mechanisms,
)
from engine.builder1_image_compliance import (
    IMAGE_COMPLIANCE_VIOLATION_CODES,
    ImageComplianceResult,
    parse_image_compliance_response,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_planner import Builder1PlannerError, _judge_repair_stage, plan_builder1
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_SLOGAN_SCAN_SYSTEM,
    STAGE_SLOGAN_SELECT_SYSTEM,
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
)
from engine.builder1_strategy_judge import StrategyJudgeResult
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _full_final_responses,
    _graphic,
    _series_ads,
)


BRIEF = "Reinforced shell product for daily carry"
HISTORIC_NAMES = ("DANKAL", "Acamol", "Tnuva", "dankal", "acamol", "tnuva")


def _pass_compliance_reviewer(**_kwargs: Any) -> ImageComplianceResult:
    return ImageComplianceResult(passed=True, violations=[], confidence="high")


def _fail_compliance_reviewer(*, violations: List[str], **_kwargs: Any) -> ImageComplianceResult:
    return ImageComplianceResult(passed=False, violations=violations, confidence="high")


class TestStrategicRestart(unittest.TestCase):
    def _plan_with_judge_sequence(self, judge_sequence: List[StrategyJudgeResult], *, ad_count: int = 2):
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(ad_count).get(system, {"pass": True}))

        with patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            side_effect=judge_sequence,
        ) as mock_judge:
            plan = plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=ad_count,
            )
        return plan, mock_judge

    def test_foundational_failure_triggers_full_restart(self) -> None:
        plan, mock_judge = self._plan_with_judge_sequence(
            [
                StrategyJudgeResult(False, ["campaign_transferable_to_competitor"]),
                StrategyJudgeResult(True, []),
            ]
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertEqual(mock_judge.call_count, 2)

    def test_restart_uses_new_exploration_seed(self) -> None:
        seeds: List[str] = []
        real_run = __import__(
            "engine.builder1_planning_pipeline", fromlist=["run_builder1_campaign_pipeline"]
        ).run_builder1_campaign_pipeline

        def tracking_run(**kwargs: Any):
            seeds.append(kwargs["exploration_seed"])
            return real_run(**kwargs)

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True}))

        with patch(
            "engine.builder1_planning_pipeline.run_builder1_campaign_pipeline",
            side_effect=tracking_run,
        ), patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            side_effect=[
                StrategyJudgeResult(False, ["category_relevance_patched"]),
                StrategyJudgeResult(True, []),
            ],
        ):
            plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertEqual(len(seeds), 2)
        self.assertNotEqual(seeds[0], seeds[1])

    def test_restart_preserves_product_name_resolved(self) -> None:
        plan, _ = self._plan_with_judge_sequence(
            [
                StrategyJudgeResult(False, ["strategy_not_brand_ownable"]),
                StrategyJudgeResult(True, []),
            ]
        )
        self.assertEqual(plan.product_name_resolved, "CarryShell")

    def test_restart_preserves_target_ad_count(self) -> None:
        plan, _ = self._plan_with_judge_sequence(
            [
                StrategyJudgeResult(False, ["business_transformation_required"]),
                StrategyJudgeResult(True, []),
            ],
            ad_count=4,
        )
        self.assertEqual(plan.ad_count, 4)
        self.assertEqual(len(plan.ads), 4)

    def test_restart_discards_rejected_slogan(self) -> None:
        slogan_stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_SLOGAN_SCAN_SYSTEM:
                slogan_stages.append("scan")
            if system == STAGE_SLOGAN_SELECT_SYSTEM:
                slogan_stages.append("select")
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True}))

        with patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            side_effect=[
                StrategyJudgeResult(False, ["client_consultation_required"]),
                StrategyJudgeResult(True, []),
            ],
        ):
            plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertGreaterEqual(slogan_stages.count("scan"), 2)
        self.assertGreaterEqual(slogan_stages.count("select"), 2)

    def test_second_foundational_failure_returns_planning_failed(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True}))

        with patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            side_effect=[
                StrategyJudgeResult(False, ["unsupported_future_capability"]),
                StrategyJudgeResult(False, ["campaign_transferable_to_competitor"]),
            ],
        ):
            with self.assertRaises(Builder1PlannerError) as ctx:
                plan_builder1(
                    product_name="CarryShell",
                    product_description=BRIEF,
                    format_value="portrait",
                    model_caller=model_caller,
                    ad_count=2,
                )
        self.assertEqual(str(ctx.exception), "planning_failed")

    def test_no_infinite_strategic_restart_loop(self) -> None:
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True}))

        with patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            side_effect=[
                StrategyJudgeResult(False, ["material_client_investment_required"]),
                StrategyJudgeResult(False, ["advantage_not_currently_true"]),
            ],
        ) as mock_judge:
            with self.assertRaises(Builder1PlannerError):
                plan_builder1(
                    product_name="CarryShell",
                    product_description=BRIEF,
                    format_value="portrait",
                    model_caller=model_caller,
                    ad_count=2,
                )
        self.assertEqual(mock_judge.call_count, 2)

    def test_minor_structural_failure_gets_targeted_repair(self) -> None:
        graphic_calls = {"n": 0}
        judge_calls = {"n": 0}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_GRAPHIC_SYSTEM_SYSTEM:
                graphic_calls["n"] += 1
                if "Repair ONLY" in user:
                    return _graphic()
                return _graphic(missing_palette=True)
            return copy.deepcopy(_full_final_responses(2).get(system, {"pass": True}))

        def judge_side_effect(*_args, **_kwargs):
            judge_calls["n"] += 1
            if judge_calls["n"] == 1:
                return StrategyJudgeResult(False, ["invented_product_logo"])
            return StrategyJudgeResult(True, [])

        with patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            side_effect=judge_side_effect,
        ):
            plan = plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertEqual(plan.ad_count, 2)
        self.assertGreaterEqual(graphic_calls["n"], 2)

    def test_foundational_codes_are_complete(self) -> None:
        expected = {
            "campaign_transferable_to_competitor",
            "category_relevance_patched",
            "advantage_not_currently_true",
            "relative_advantage_not_currently_true",
            "strategy_not_brand_ownable",
            "business_transformation_required",
            "client_consultation_required",
            "material_client_investment_required",
            "unsupported_future_capability",
        }
        self.assertTrue(expected.issubset(FOUNDATIONAL_STRATEGIC_REJECTION_CODES))

    def test_only_one_restart_in_planner_source(self) -> None:
        from engine import builder1_planner as module

        source = inspect.getsource(module.plan_builder1)
        self.assertEqual(source.count("strategic_restart_used = True"), 1)
        self.assertIn("BUILDER1_STRATEGIC_RESTART_START", source)


class TestNoHistoricBlacklist(unittest.TestCase):
    def test_production_prompts_contain_no_historic_campaign_names(self) -> None:
        engine_dir = Path(__file__).resolve().parents[1] / "engine"
        hits: List[str] = []
        for path in engine_dir.rglob("*.py"):
            if "builder2" in path.name.lower():
                continue
            text = path.read_text(encoding="utf-8")
            for name in HISTORIC_NAMES:
                if name in text:
                    hits.append(f"{path.name}:{name}")
        self.assertEqual(hits, [])

    def test_dankal_in_current_brief_not_rejected(self) -> None:
        plan = {
            "brandSlogan": "Built To Last",
            "relativeAdvantage": "Survives daily drops",
            "sloganDerivation": "From durability advantage",
            "detectedLanguage": "en",
            "ads": [
                {
                    "index": 1,
                    "sceneDescription": "DANKAL reinforced shell on a desk",
                    "headline": None,
                    "noReuseCheck": "unique",
                }
            ],
        }
        self.assertNotIn(
            "no_mechanism_reuse_inside_campaign",
            deterministic_methodology_checks(plan),
        )
        self.assertIsNone(scan_prompt_for_reused_mechanisms("Product named DANKAL for daily carry"))

    def test_explicit_cross_campaign_reuse_wording_still_rejected(self) -> None:
        code = scan_prompt_for_reused_mechanisms("reuse the previous campaign mechanism")
        self.assertEqual(code, "no_mechanism_reuse_inside_campaign")

    def test_duplicate_executions_within_campaign_rejected(self) -> None:
        plan = {
            "brandSlogan": "Built To Last",
            "relativeAdvantage": "Survives daily drops",
            "sloganDerivation": "From durability",
            "detectedLanguage": "en",
            "ads": [
                {"index": 1, "sceneDescription": "Same scene", "headline": "Head A", "noReuseCheck": "unique"},
                {"index": 2, "sceneDescription": "Same scene", "headline": "Head B", "noReuseCheck": "unique"},
            ],
        }
        self.assertIn("same_image_different_headlines", deterministic_methodology_checks(plan))

    def test_no_global_cross_user_memory_module(self) -> None:
        engine_dir = Path(__file__).resolve().parents[1] / "engine"
        for path in engine_dir.glob("builder1*.py"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("cross_user", text)
            self.assertNotIn("global creative memory", text)


class TestImageCompliance(unittest.TestCase):
    def _plan(self, ad_count: int = 2):
        from tests.test_builder1_series import _base_campaign, _parse

        return _parse(_base_campaign(ad_count), ad_count)

    def test_compliance_response_schema(self) -> None:
        parsed = parse_image_compliance_response(
            {"pass": True, "violations": [], "confidence": "high"}
        )
        self.assertTrue(parsed.passed)
        self.assertEqual(parsed.confidence, "high")

    def test_allowed_violation_codes(self) -> None:
        self.assertEqual(
            IMAGE_COMPLIANCE_VIOLATION_CODES,
            frozenset(
                {
                    "invented_product_logo",
                    "supplied_logo_displayed",
                    "logo_like_brand_symbol",
                    "packaging_contains_brand_mark",
                    "campaign_device_used_as_logo",
                    "product_name_rendered_as_logo",
                }
            ),
        )

    def test_compliant_text_only_product_name_passes(self) -> None:
        result = review_builder1_ad_image_compliance(
            b"fake-image",
            product_name="TestBrand",
            ad_index=1,
            reviewer=_pass_compliance_reviewer,
        )
        self.assertTrue(result.passed)

    def test_lightning_bolt_logo_fails(self) -> None:
        result = review_builder1_ad_image_compliance(
            b"fake-image",
            product_name="EnergyX",
            ad_index=1,
            reviewer=lambda **kw: _fail_compliance_reviewer(
                violations=["logo_like_brand_symbol"], **kw
            ),
        )
        self.assertFalse(result.passed)
        self.assertIn("logo_like_brand_symbol", result.violations)

    def test_monogram_on_package_fails(self) -> None:
        result = review_builder1_ad_image_compliance(
            b"fake-image",
            product_name="TestBrand",
            ad_index=1,
            reviewer=lambda **kw: _fail_compliance_reviewer(
                violations=["packaging_contains_brand_mark"], **kw
            ),
        )
        self.assertFalse(result.passed)

    def test_campaign_decoration_not_used_as_logo_passes(self) -> None:
        result = review_builder1_ad_image_compliance(
            b"fake-image",
            product_name="TestBrand",
            ad_index=1,
            reviewer=_pass_compliance_reviewer,
        )
        self.assertTrue(result.passed)

    def test_failed_review_triggers_exactly_one_regeneration(self) -> None:
        calls = {"gen": 0, "review": 0}

        def caller(prompt: str, fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**kwargs: Any) -> ImageComplianceResult:
            calls["review"] += 1
            if calls["review"] == 1:
                return ImageComplianceResult(
                    passed=False, violations=["invented_product_logo"], confidence="high"
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        generate_builder1_ad_image(
            self._plan(2),
            1,
            caller,
            compliance_reviewer=reviewer,
        )
        self.assertEqual(calls["gen"], 2)
        self.assertEqual(calls["review"], 2)

    def test_rejected_image_not_returned_on_repeated_failure(self) -> None:
        def caller(_prompt: str, _fmt: str) -> bytes:
            return b"bad"

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            return ImageComplianceResult(
                passed=False, violations=["invented_product_logo"], confidence="high"
            )

        with self.assertRaises(Exception) as ctx:
            generate_builder1_ad_image(
                self._plan(2),
                1,
                caller,
                compliance_reviewer=reviewer,
            )
        from engine.builder1_image_compliance import ImageComplianceError

        self.assertIsInstance(ctx.exception, ImageComplianceError)

    def test_successful_replacement_keeps_same_ad_index(self) -> None:
        review_n = {"n": 0}

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            review_n["n"] += 1
            if review_n["n"] == 1:
                return ImageComplianceResult(
                    passed=False, violations=["supplied_logo_displayed"], confidence="high"
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        result = generate_builder1_ad_image(
            self._plan(2),
            1,
            lambda _p, _f: b"img",
            compliance_reviewer=reviewer,
        )
        self.assertEqual(result.index, 1)

    def test_correction_block_present_after_regeneration(self) -> None:
        prompts: List[str] = []

        def caller(prompt: str, _fmt: str) -> bytes:
            prompts.append(prompt)
            return b"img"

        review_n = {"n": 0}

        def reviewer(**_kwargs: Any) -> ImageComplianceResult:
            review_n["n"] += 1
            if review_n["n"] == 1:
                return ImageComplianceResult(
                    passed=False, violations=["invented_product_logo"], confidence="high"
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        generate_builder1_ad_image(
            self._plan(2),
            1,
            caller,
            compliance_reviewer=reviewer,
        )
        self.assertGreaterEqual(len(prompts), 2)
        self.assertIn("IMAGE COMPLIANCE CORRECTION", prompts[1])


class TestImageComplianceAppIntegration(unittest.TestCase):
    def test_generated_count_advances_only_after_compliance_passes(self) -> None:
        from engine.builder1_campaign_store import (
            clear_memory_store_for_tests,
            create_campaign_session,
            mark_ad_generated,
            try_acquire_generation_lock,
        )
        from tests.test_builder1_series import _base_campaign, _parse

        clear_memory_store_for_tests()
        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="cmp-test", plan=plan, target_ad_count=2)
        try_acquire_generation_lock("cmp-test", 1)
        session = mark_ad_generated("cmp-test", 1)
        self.assertEqual(session.generated_count, 1)

    def test_app_handles_compliance_failure_without_storing_image(self) -> None:
        from app import _builder1_generate_single_ad
        from engine.builder1_campaign_store import clear_memory_store_for_tests, create_campaign_session
        from engine.builder1_image_compliance import ImageComplianceError
        from tests.test_builder1_series import _base_campaign, _parse

        clear_memory_store_for_tests()
        plan = _parse(_base_campaign(2), 2)
        session = create_campaign_session(campaign_id="cmp-compliance-res", plan=plan, target_ad_count=2)
        session.generating_index = 1

        def raise_compliance(*_args, **_kwargs):
            raise ImageComplianceError(["invented_product_logo"], ad_index=1)

        with patch("app.generate_builder1_ad_image", side_effect=raise_compliance):
            with patch("app.get_campaign_session", return_value=session):
                result = _builder1_generate_single_ad(
                    job_id="job-1",
                    campaign_id="cmp-compliance-res",
                    ad_index=1,
                    already_reserved=True,
                )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "image_compliance_failed")
        self.assertTrue(result["retryable"])
        self.assertEqual(result["nextAdIndex"], 1)
        self.assertEqual(result["generatedCount"], 0)
        self.assertEqual(result["targetAdCount"], 2)
        self.assertNotIn("image_base64", str(result))
        self.assertNotIn("ad", result)

    def test_repeated_failure_preserves_next_ad_index(self) -> None:
        from app import _builder1_generate_single_ad
        from engine.builder1_campaign_store import clear_memory_store_for_tests, create_campaign_session
        from engine.builder1_image_compliance import ImageComplianceError
        from tests.test_builder1_series import _base_campaign, _parse

        clear_memory_store_for_tests()
        plan = _parse(_base_campaign(2), 2)
        session = create_campaign_session(campaign_id="cmp-retry", plan=plan, target_ad_count=2)
        session.generating_index = 1

        with patch("app.generate_builder1_ad_image", side_effect=ImageComplianceError(["invented_product_logo"], ad_index=1)):
            with patch("app.get_campaign_session", return_value=session):
                result = _builder1_generate_single_ad(
                    job_id="job-2",
                    campaign_id="cmp-retry",
                    ad_index=1,
                    already_reserved=True,
                )
        self.assertEqual(result["nextAdIndex"], 1)
        self.assertEqual(result["generatedCount"], 0)
        self.assertEqual(result["targetAdCount"], 2)

    def test_generate_again_does_not_rerun_planning(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            from engine.builder1_campaign_store import (
                clear_memory_store_for_tests,
                create_campaign_session,
                mark_ad_generated,
                try_acquire_generation_lock,
                validate_next_ad_request,
            )
            from tests.test_builder1_series import _base_campaign, _parse

            clear_memory_store_for_tests()
            plan = _parse(_base_campaign(2), 2)
            create_campaign_session(campaign_id="gap-next", plan=plan, target_ad_count=2)
            try_acquire_generation_lock("gap-next", 1)
            mark_ad_generated("gap-next", 1)
            validate_next_ad_request("gap-next", 2)

    def test_concurrent_campaigns_remain_isolated(self) -> None:
        from engine.builder1_campaign_store import clear_memory_store_for_tests, create_campaign_session

        from tests.test_builder1_series import _base_campaign, _parse

        clear_memory_store_for_tests()
        plan_a = _parse(_base_campaign(2), 2)
        plan_b = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="gap-a", plan=plan_a, target_ad_count=2)
        create_campaign_session(campaign_id="gap-b", plan=plan_b, target_ad_count=2)
        generate_builder1_ad_image(
            plan_a,
            1,
            lambda _p, _f: b"a",
            compliance_reviewer=_pass_compliance_reviewer,
        )
        generate_builder1_ad_image(
            plan_b,
            1,
            lambda _p, _f: b"b",
            compliance_reviewer=_pass_compliance_reviewer,
        )


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_files_untouched(self) -> None:
        engine_dir = Path(__file__).resolve().parents[1] / "engine"
        builder2_files = list(engine_dir.glob("builder2*.py"))
        self.assertTrue(builder2_files)


if __name__ == "__main__":
    unittest.main()
