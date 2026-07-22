"""
Builder1 cross-campaign idea memory tests.

Run: python -m unittest tests.test_builder1_idea_memory -v
"""
from __future__ import annotations

import copy
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_idea_memory import (
    BUILDER1_IDEA_MEMORY_MAX_ADS,
    IdeaMemoryScope,
    build_records_from_plan,
    build_stage_memory_block,
    clear_idea_memory_for_tests,
    compute_ad_execution_fingerprint,
    compute_campaign_idea_fingerprint,
    find_historical_duplicate,
    idea_memory_active,
    load_builder1_idea_memory,
    normalize_product_scope_text,
    persist_idea_memory_records,
    resolve_idea_memory_scope,
)
from engine.builder1_idea_memory_pipeline import (
    persist_plan_idea_memory,
    run_strategy_slogan_with_memory_guard,
    stage_memory_block,
)
from engine.builder1_plan_spec import (
    Builder1AdPlan,
    Builder1GraphicGenerator,
    Builder1Palette,
    Builder1SeriesGenerator,
    Builder1SeriesPlan,
    Builder1CopySafeArea,
)
from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_NAME
from engine.builder1_planner import Builder1PlannerError
from engine.builder1_series_distinctness import validate_ad_execution_distinctness


def _graphic() -> Builder1GraphicGenerator:
    palette = Builder1Palette("navy", "white", "gold", "gray", "black")
    safe = Builder1CopySafeArea("left", 30)
    return Builder1GraphicGenerator(
        palette=palette,
        layout_template="hero-right",
        headline_placement="bottom",
        headline_alignment="left",
        headline_max_width_percent=60,
        brand_block_placement="top-left",
        slogan_placement="bottom-left",
        copy_safe_area=safe,
        typography_style="sans",
        headline_scale="large",
        brand_scale="small",
        slogan_scale="medium",
        image_style="photographic",
        background_treatment="clean",
        border_treatment="none",
        recurring_graphic_device="thin frame",
        recurring_graphic_device_rule="always present",
        shape_language="rounded",
        framing_rule="center-weighted",
        spacing_rule="generous",
        slogan_placement_reason="RTL default",
    )


def _plan(*, ad_count: int = 2, campaign_suffix: str = "a") -> Builder1SeriesPlan:
    ads: List[Builder1AdPlan] = []
    ad_internals: Dict[int, Dict[str, str]] = {}
    for idx in range(1, ad_count + 1):
        ads.append(
            Builder1AdPlan(
                index=idx,
                variation_label=f"v{idx}",
                new_contribution=f"contrib {idx}",
                physical_execution=f"physical {idx}",
                visual_execution=f"visual {idx}",
                scene_description=f"scene {idx}",
                conceptual_execution=f"conceptual {idx}",
                conceptual_action_proof=f"proof {idx}",
                headline=None,
                headline_needed_reason="self-explanatory",
                marketing_text="word " * 50,
            )
        )
        ad_internals[idx] = {
            "executionSubject": f"subject-{idx}-{campaign_suffix}",
            "executionAction": f"action-{idx}-{campaign_suffix}",
            "executionObjectState": f"state-{idx}-{campaign_suffix}",
            "executionScene": f"scene-{idx}-{campaign_suffix}",
            "executionPunchline": f"punch-{idx}-{campaign_suffix}",
        }
    return Builder1SeriesPlan(
        product_name="CarryShell",
        product_description="Reinforced shell product",
        format="portrait",
        ad_count=ad_count,
        product_name_resolved="CarryShell",
        detected_language="en",
        strategic_problem=f"Problem {campaign_suffix}",
        strategic_problem_evidence="evidence",
        relative_advantage=f"Advantage {campaign_suffix}",
        relative_advantage_source="brief",
        relative_advantage_brief_support="support",
        relative_advantage_claim_risk="low",
        problem_advantage_link="linked",
        brand_slogan=f"Slogan {campaign_suffix}",
        slogan_derivation="derived",
        slogan_action="acts",
        conceptual_generator=f"Conceptual {campaign_suffix}",
        conceptual_generator_action="transform",
        conceptual_generator_input="input",
        conceptual_generator_transformation="transform",
        conceptual_generator_result="result",
        conceptual_generator_why_it_expresses_advantage="why",
        physical_generator=f"Physical {campaign_suffix}",
        physical_generator_natural_purpose="purpose",
        physical_generator_campaign_role="role",
        transferred_object=f"Object {campaign_suffix}",
        transferred_object_action=f"Action {campaign_suffix}",
        product_visibility_policy="FORBIDDEN",
        graphic_generator=_graphic(),
        series_generator=Builder1SeriesGenerator("progression", "principle", "progression"),
        medium_participates=False,
        medium_role="",
        campaign_rationale="rationale",
        ads=ads,
        planning_internals={"adInternals": ad_internals},
    )


class IdeaMemoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._force_env = patch.dict(os.environ, {"BUILDER1_IDEA_MEMORY_FORCE": "1"}, clear=False)
        self._force_env.start()
        clear_idea_memory_for_tests()

    def tearDown(self) -> None:
        clear_idea_memory_for_tests()
        self._force_env.stop()


class TestIdeaMemoryScope(IdeaMemoryTestCase):
    def test_product_name_normalization(self) -> None:
        a = resolve_idea_memory_scope(user_product_name="  Carry Shell! ", user_product_description="brief")
        b = resolve_idea_memory_scope(user_product_name="carry shell", user_product_description="brief")
        self.assertEqual(a.product_scope_hash, b.product_scope_hash)

    def test_different_products_have_separate_histories(self) -> None:
        scope_a = resolve_idea_memory_scope(user_product_name="Alpha", user_product_description="a")
        scope_b = resolve_idea_memory_scope(user_product_name="Beta", user_product_description="b")
        self.assertNotEqual(scope_a.product_scope_hash, scope_b.product_scope_hash)

    @patch.dict(os.environ, {"BUILDER1_MEMORY_TENANT_SCOPE": "tenant-a"}, clear=False)
    def test_different_tenants_have_separate_histories(self) -> None:
        scope_a = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        with patch.dict(os.environ, {"BUILDER1_MEMORY_TENANT_SCOPE": "tenant-b"}, clear=False):
            scope_b = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        self.assertNotEqual(scope_a.tenant_scope_hash, scope_b.tenant_scope_hash)


class TestIdeaMemoryPersistence(IdeaMemoryTestCase):
    def test_empty_memory_permits_first_campaign(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        snapshot = load_builder1_idea_memory(scope=scope)
        self.assertEqual(snapshot.records, [])

    def test_two_ad_plan_writes_two_records(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        plan = _plan(ad_count=2, campaign_suffix="one")
        records = build_records_from_plan(plan, scope=scope, campaign_id="camp-1")
        self.assertEqual(len(records), 2)
        result = persist_idea_memory_records(records, scope=scope)
        self.assertEqual(result.added_count, 2)
        self.assertEqual(result.count_after, 2)

    def test_four_ad_plan_writes_four_records(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        plan = _plan(ad_count=4, campaign_suffix="four")
        records = build_records_from_plan(plan, scope=scope, campaign_id="camp-4")
        self.assertEqual(len(records), 4)
        result = persist_idea_memory_records(records, scope=scope)
        self.assertEqual(result.added_count, 4)

    def test_later_campaign_receives_prior_memory(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        persist_idea_memory_records(
            build_records_from_plan(_plan(campaign_suffix="prior"), scope=scope, campaign_id="prior"),
            scope=scope,
        )
        snapshot = load_builder1_idea_memory(scope=scope, exclude_campaign_id="new")
        self.assertEqual(len(snapshot.historical_records(exclude_campaign_id="new")), 2)

    def test_different_campaign_id_does_not_bypass_memory(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        persist_idea_memory_records(
            build_records_from_plan(_plan(campaign_suffix="old"), scope=scope, campaign_id="old-id"),
            scope=scope,
        )
        snapshot = load_builder1_idea_memory(scope=scope, exclude_campaign_id="brand-new-id")
        finding = find_historical_duplicate(
            stage="strategy_slogan_stage",
            snapshot=snapshot,
            exclude_campaign_id="brand-new-id",
            strategic_problem="Problem old",
            relative_advantage="Advantage old",
        )
        self.assertIsNotNone(finding)

    def test_idempotent_retry_does_not_consume_fifo_slot(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        records = build_records_from_plan(_plan(campaign_suffix="retry"), scope=scope, campaign_id="retry-camp")
        first = persist_idea_memory_records(records, scope=scope)
        second = persist_idea_memory_records(records, scope=scope)
        self.assertEqual(first.added_count, 2)
        self.assertEqual(second.skipped_idempotent_count, 2)
        self.assertEqual(second.added_count, 0)
        snapshot = load_builder1_idea_memory(scope=scope)
        self.assertEqual(len(snapshot.records), 2)

    def test_fifo_evicts_oldest_on_201st_record(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        oldest = build_records_from_plan(_plan(campaign_suffix="oldest"), scope=scope, campaign_id="camp-0")[0]
        persist_idea_memory_records([oldest], scope=scope)
        for idx in range(1, BUILDER1_IDEA_MEMORY_MAX_ADS):
            record = copy.deepcopy(oldest)
            record.record_id = f"camp-{idx}:1"
            record.campaign_id = f"camp-{idx}"
            record.ad_index = 1
            record.strategic_problem = f"Problem {idx}"
            record.relative_advantage = f"Advantage {idx}"
            persist_idea_memory_records([record], scope=scope)
        overflow = copy.deepcopy(oldest)
        overflow.record_id = "camp-overflow:1"
        overflow.campaign_id = "camp-overflow"
        overflow.strategic_problem = "Problem overflow"
        overflow.relative_advantage = "Advantage overflow"
        result = persist_idea_memory_records([overflow], scope=scope)
        self.assertEqual(result.evicted_count, 1)
        self.assertEqual(result.count_after, BUILDER1_IDEA_MEMORY_MAX_ADS)
        snapshot = load_builder1_idea_memory(scope=scope)
        record_ids = {r.record_id for r in snapshot.records}
        self.assertNotIn("camp-0:1", record_ids)
        self.assertIn("camp-overflow:1", record_ids)

    def test_evicted_record_becomes_eligible_for_reuse(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        oldest = build_records_from_plan(_plan(campaign_suffix="reuse"), scope=scope, campaign_id="reuse-camp")[0]
        persist_idea_memory_records([oldest], scope=scope)
        for idx in range(1, BUILDER1_IDEA_MEMORY_MAX_ADS):
            record = copy.deepcopy(oldest)
            record.record_id = f"fill-{idx}:1"
            record.campaign_id = f"fill-{idx}"
            record.strategic_problem = f"Unique problem {idx}"
            record.relative_advantage = f"Unique advantage {idx}"
            persist_idea_memory_records([record], scope=scope)
        overflow = copy.deepcopy(oldest)
        overflow.record_id = "overflow:1"
        overflow.campaign_id = "overflow"
        overflow.strategic_problem = "Overflow problem"
        overflow.relative_advantage = "Overflow advantage"
        persist_idea_memory_records([overflow], scope=scope)
        snapshot = load_builder1_idea_memory(scope=scope, exclude_campaign_id="new")
        finding = find_historical_duplicate(
            stage="strategy_slogan_stage",
            snapshot=snapshot,
            exclude_campaign_id="new",
            strategic_problem=oldest.strategic_problem,
            relative_advantage=oldest.relative_advantage,
        )
        self.assertIsNone(finding)

    def test_no_image_bytes_stored(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        records = build_records_from_plan(_plan(), scope=scope, campaign_id="no-image")
        payload = records[0].to_dict()
        self.assertFalse(any("base64" in str(value).lower() for value in payload.values()))


class TestHistoricalDuplicateDetection(IdeaMemoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        persist_idea_memory_records(
            build_records_from_plan(_plan(campaign_suffix="hist"), scope=self.scope, campaign_id="hist"),
            scope=self.scope,
        )
        self.snapshot = load_builder1_idea_memory(scope=self.scope, exclude_campaign_id="current")

    def test_strategy_duplicate_rejected(self) -> None:
        finding = find_historical_duplicate(
            stage="strategy_slogan_stage",
            snapshot=self.snapshot,
            exclude_campaign_id="current",
            strategic_problem="Problem hist",
            relative_advantage="Advantage hist",
        )
        self.assertIsNotNone(finding)

    def test_slogan_change_alone_is_not_new(self) -> None:
        finding = find_historical_duplicate(
            stage="strategy_slogan_stage",
            snapshot=self.snapshot,
            exclude_campaign_id="current",
            strategic_problem="Problem hist",
            relative_advantage="Advantage hist",
            brand_slogan="Totally new slogan",
        )
        self.assertIsNotNone(finding)

    def test_conceptual_duplicate_detected(self) -> None:
        finding = find_historical_duplicate(
            stage="conceptual_stage",
            snapshot=self.snapshot,
            exclude_campaign_id="current",
            conceptual_generator="Conceptual hist",
        )
        self.assertIsNotNone(finding)

    def test_reworded_conceptual_generator_still_matches(self) -> None:
        finding = find_historical_duplicate(
            stage="conceptual_stage",
            snapshot=self.snapshot,
            exclude_campaign_id="current",
            conceptual_generator="  conceptual   HIST ",
        )
        self.assertIsNotNone(finding)

    def test_physical_duplicate_detected(self) -> None:
        finding = find_historical_duplicate(
            stage="brand_physical",
            snapshot=self.snapshot,
            exclude_campaign_id="current",
            physical_generator="Physical hist",
            transferred_object="Object hist",
            transferred_object_action="Action hist",
        )
        self.assertIsNotNone(finding)

    def test_different_background_does_not_make_execution_new(self) -> None:
        prior = self.snapshot.records[0]
        ad = {
            "conceptualExecution": prior.conceptual_execution,
            "physicalExecution": prior.physical_execution,
            "visualExecution": prior.visual_execution,
            "sceneDescription": "Different background only",
            "executionSubject": prior.execution_subject,
            "executionAction": prior.execution_action,
            "executionObjectState": prior.execution_object_state,
            "executionScene": prior.execution_scene,
            "executionPunchline": prior.execution_punchline,
        }
        fp = compute_ad_execution_fingerprint(ad)
        finding = find_historical_duplicate(
            stage="series_ads",
            snapshot=self.snapshot,
            exclude_campaign_id="current",
            ad_execution_fingerprint=fp,
        )
        self.assertIsNotNone(finding)

    def test_genuinely_different_campaign_passes(self) -> None:
        finding = find_historical_duplicate(
            stage="strategy_slogan_stage",
            snapshot=self.snapshot,
            exclude_campaign_id="current",
            strategic_problem="Fresh strategic problem",
            relative_advantage="Fresh relative advantage",
        )
        self.assertIsNone(finding)

    def test_current_campaign_excluded_from_history(self) -> None:
        snapshot = load_builder1_idea_memory(scope=self.scope, exclude_campaign_id="hist")
        finding = find_historical_duplicate(
            stage="strategy_slogan_stage",
            snapshot=snapshot,
            exclude_campaign_id="hist",
            strategic_problem="Problem hist",
            relative_advantage="Advantage hist",
        )
        self.assertIsNone(finding)


class TestStageMemoryInjection(IdeaMemoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        persist_idea_memory_records(
            build_records_from_plan(_plan(campaign_suffix="inj"), scope=self.scope, campaign_id="inj"),
            scope=self.scope,
        )
        self.snapshot = load_builder1_idea_memory(scope=self.scope, exclude_campaign_id="next")

    def test_stage_specific_blocks_are_compact(self) -> None:
        strategy_block = build_stage_memory_block(
            "strategy_slogan_stage", self.snapshot, exclude_campaign_id="next"
        )
        series_block = build_stage_memory_block("series_ads", self.snapshot, exclude_campaign_id="next")
        self.assertIn("strategicProblem", strategy_block)
        self.assertNotIn("executionPunchline", strategy_block)
        self.assertIn("executionPunchline", series_block)
        self.assertNotIn("strategicProblem", series_block)

    def test_stage_memory_block_helper(self) -> None:
        block = stage_memory_block("conceptual_stage", self.snapshot, campaign_id="next")
        self.assertIn("conceptualGenerator", block)


class TestStrategyMemoryGuard(IdeaMemoryTestCase):
    def test_duplicate_after_repair_fails(self) -> None:
        scope = resolve_idea_memory_scope(user_product_name="CarryShell", user_product_description="brief")
        persist_idea_memory_records(
            build_records_from_plan(_plan(campaign_suffix="dup"), scope=scope, campaign_id="prior"),
            scope=scope,
        )
        snapshot = load_builder1_idea_memory(scope=scope, exclude_campaign_id="current")

        class Strategy:
            strategic_problem = "Problem dup"
            relative_advantage = "Advantage dup"

        class Slogan:
            brand_slogan = "Slogan dup"

        def fake_run_stage(stage, model_caller, system, user, parse, **kwargs):
            if stage == "strategy_slogan_repair":
                return (None, Strategy(), [], {}, None, Slogan(), [])
            return parse({})

        def fake_run_strategy(*args, **kwargs):
            return (None, Strategy(), [], {}, None, Slogan(), [])

        with patch(
            "engine.builder1_idea_memory_pipeline.run_strategy_slogan_stage",
            side_effect=fake_run_strategy,
        ):
            with self.assertRaises(Builder1PlannerError) as ctx:
                run_strategy_slogan_with_memory_guard(
                    fake_run_stage,
                    None,
                    campaign_id="current",
                    idea_memory=snapshot,
                    product_name="CarryShell",
                    product_name_resolved="CarryShell",
                    product_description="brief",
                    detected_language="en",
                    lens_order=["problem"],
                    exploration_seed="seed",
                    idea_memory_block="",
                )
        self.assertEqual(str(ctx.exception), "builder1_historical_idea_duplicate")


class TestSeriesDistinctnessRegression(unittest.TestCase):
    def test_sibling_ads_may_share_series_generator(self) -> None:
        ads = [
            {
                "index": 1,
                "conceptualExecution": "Shared generator family ad 1",
                "physicalExecution": "Shared physical family",
                "visualExecution": "Visual 1",
                "sceneDescription": "Scene 1",
                "executionSubject": "Subject A",
                "executionAction": "Action A",
                "executionObjectState": "State A",
                "executionScene": "Scene A",
                "executionPunchline": "Punch A",
            },
            {
                "index": 2,
                "conceptualExecution": "Shared generator family ad 2",
                "physicalExecution": "Shared physical family",
                "visualExecution": "Visual 2",
                "sceneDescription": "Scene 2",
                "executionSubject": "Subject B",
                "executionAction": "Action B",
                "executionObjectState": "State B",
                "executionScene": "Scene B",
                "executionPunchline": "Punch B",
            },
        ]
        reasons, _ = validate_ad_execution_distinctness(ads)
        self.assertEqual(reasons, [])


class TestPlanningCallBudget(unittest.TestCase):
    def test_normal_expected_calls_remain_five(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)

    def test_idea_memory_inactive_without_redis_or_force(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(idea_memory_active())
        with patch.dict(os.environ, {"BUILDER1_IDEA_MEMORY_FORCE": "1"}, clear=False):
            self.assertTrue(idea_memory_active())


class TestCampaignFingerprint(unittest.TestCase):
    def test_campaign_fingerprint_ignores_campaign_id(self) -> None:
        from engine.builder1_idea_memory import _campaign_idea_payload

        payload = _campaign_idea_payload(
            strategic_problem="Problem",
            relative_advantage="Advantage",
            slogan="Slogan",
            conceptual_generator="Concept",
            physical_generator="Physical",
            transferred_object="Object",
            transferred_object_action="Action",
        )
        self.assertTrue(compute_campaign_idea_fingerprint(payload))


if __name__ == "__main__":
    unittest.main()
