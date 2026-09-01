"""
Builder1 planning stage checkpoint + paid-reasoning resume tests.

Run: python -m unittest tests.test_builder1_planning_checkpoint -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_planning_checkpoint import (
    CHECKPOINT_VERSION,
    STAGE_CHECKPOINT_CONTRACT_VERSIONS,
    PlanningCheckpointPersistError,
    PlanningCheckpointSession,
    build_planning_checkpoint_identity,
    delete_planning_checkpoint,
    load_planning_checkpoint_record,
    save_planning_checkpoint_record,
    serialize_strategy_slogan_stage_output,
)
from engine.builder1_planning_metrics import get_planning_metrics
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
)
from tests.test_builder1_staged_planning import (
    STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM,
    _brand_physical,
    _full_final_responses,
    _graphic,
    _series_ads,
)

BRIEF = "Reinforced shell product for daily carry"
JOB_ID = "job-checkpoint-test"
CAMPAIGN_ID = "campaign-checkpoint-test"


def _identity(**overrides: Any):
    base = build_planning_checkpoint_identity(
        job_id=JOB_ID,
        campaign_id=CAMPAIGN_ID,
        product_name="CarryShell",
        product_description=BRIEF,
        format_value="portrait",
        ad_count=2,
        brand_guidelines=None,
    )
    if not overrides:
        return base
    data = base.to_dict()
    data.update(overrides)
    return type(base)(**{
        "job_id": data["jobId"],
        "campaign_id": data["campaignId"],
        "request_fingerprint": data["requestFingerprint"],
        "planning_contract_version": data["planningContractVersion"],
    })


def _stage_counter_responses(ad_count: int = 2) -> tuple[Dict[str, Any], Dict[str, int]]:
    responses = _full_final_responses(ad_count)
    counts: Dict[str, int] = {
        "strategy_slogan_stage": 0,
        "conceptual_stage": 0,
        "brand_physical": 0,
        "graphic_system": 0,
        "series_ads": 0,
        "product_name_resolution": 0,
    }

    def model_caller(system: str, user: str, stage: str | None = None) -> object:
        stage_name = stage or ""
        if stage_name in counts:
            counts[stage_name] += 1
        if stage_name == "product_name_resolution":
            return responses[STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM]
        if stage_name == "strategy_slogan_stage":
            return responses[STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM]
        if stage_name == "conceptual_stage":
            return responses[STAGE_CONCEPTUAL_STAGE_SYSTEM]
        if stage_name == "brand_physical":
            return responses[STAGE_BRAND_PHYSICAL_SYSTEM]
        if stage_name == "graphic_system":
            return responses[STAGE_GRAPHIC_SYSTEM_SYSTEM]
        if stage_name == "series_ads":
            return responses[STAGE_SERIES_ADS_SYSTEM]
        return responses.get(system, {})

    return responses, counts  # type: ignore[return-value]


class TestPlanningCheckpointPersistence(unittest.TestCase):
    def setUp(self) -> None:
        delete_planning_checkpoint(JOB_ID)
        self._stage_counts: Dict[str, int] = {
            "strategy_slogan_stage": 0,
            "conceptual_stage": 0,
            "brand_physical": 0,
            "graphic_system": 0,
            "series_ads": 0,
            "product_name_resolution": 0,
        }
        self._responses = _full_final_responses(2)

    def tearDown(self) -> None:
        delete_planning_checkpoint(JOB_ID)

    def _model_caller(self, system: str, user: str, stage: str | None = None) -> object:
        stage_name = stage or ""
        if stage_name in self._stage_counts:
            self._stage_counts[stage_name] += 1
        if stage_name == "product_name_resolution":
            return self._responses[STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM]
        if stage_name == "strategy_slogan_stage":
            return self._responses[STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM]
        if stage_name == "conceptual_stage":
            return self._responses[STAGE_CONCEPTUAL_STAGE_SYSTEM]
        if stage_name == "brand_physical":
            return self._responses[STAGE_BRAND_PHYSICAL_SYSTEM]
        if stage_name == "graphic_system":
            return self._responses[STAGE_GRAPHIC_SYSTEM_SYSTEM]
        if stage_name == "series_ads":
            return self._responses[STAGE_SERIES_ADS_SYSTEM]
        return self._responses.get(system, {})

    def _run_plan(self, *, job_id: str = JOB_ID, campaign_id: str = CAMPAIGN_ID, **kwargs: Any):
        for key in self._stage_counts:
            self._stage_counts[key] = 0
        plan = plan_builder1(
            product_name=kwargs.get("product_name", "CarryShell"),
            product_description=kwargs.get("product_description", BRIEF),
            format_value=kwargs.get("format_value", "portrait"),
            model_caller=self._model_caller,
            ad_count=kwargs.get("ad_count", 2),
            brand_guidelines=kwargs.get("brand_guidelines"),
            campaign_id=campaign_id,
            job_id=job_id,
        )
        return plan, dict(self._stage_counts)

    def test_strategy_checkpoint_before_conceptual(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_conceptual_with_memory_guard",
            side_effect=Builder1PlannerError("conceptual_stage_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        self.assertIsNotNone(record)
        stages = dict(record.get("completedStages") or {})
        self.assertIn("strategy_slogan_stage", stages)
        self.assertNotIn("conceptual_stage", stages)

    def test_conceptual_checkpoint_before_brand_physical(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        stages = dict(record.get("completedStages") or {})
        self.assertIn("strategy_slogan_stage", stages)
        self.assertIn("conceptual_stage", stages)
        self.assertNotIn("brand_physical", stages)

    def test_brand_physical_checkpoint_before_graphic(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline._run_graphic_system_stage",
            side_effect=Builder1PlannerError("graphic_system_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        stages = dict(record.get("completedStages") or {})
        self.assertIn("brand_physical", stages)
        self.assertNotIn("graphic_system", stages)

    def test_graphic_checkpoint_before_series(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline._run_series_stage_with_integrity",
            side_effect=Builder1PlannerError("series_ads_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        stages = dict(record.get("completedStages") or {})
        self.assertIn("graphic_system", stages)
        self.assertNotIn("series_ads", stages)

    def test_full_success_finalizes_without_leaving_partial_series(self) -> None:
        plan, counts = self._run_plan()
        self.assertEqual(plan.ad_count, 2)
        record = load_planning_checkpoint_record(JOB_ID)
        stages = dict(record.get("completedStages") or {})
        self.assertIn("series_ads", stages)
        self.assertEqual(counts.get("strategy_slogan_stage"), 1)

    def test_resume_after_conceptual_reuses_upstream(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        _, counts = self._run_plan()
        self.assertEqual(counts.get("strategy_slogan_stage"), 0)
        self.assertEqual(counts.get("conceptual_stage"), 0)
        self.assertEqual(counts.get("brand_physical"), 1)

    def test_brand_physical_failure_resume_zero_upstream_calls(self) -> None:
        self.test_resume_after_conceptual_reuses_upstream()

    def test_graphic_failure_resume_reuses_three_upstream(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline._run_graphic_system_stage",
            side_effect=Builder1PlannerError("graphic_system_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        _, counts = self._run_plan()
        self.assertEqual(counts.get("strategy_slogan_stage"), 0)
        self.assertEqual(counts.get("conceptual_stage"), 0)
        self.assertEqual(counts.get("brand_physical"), 0)
        self.assertEqual(counts.get("graphic_system"), 1)

    def test_series_failure_resume_reuses_four_upstream(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline._run_series_stage_with_integrity",
            side_effect=Builder1PlannerError("series_ads_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        _, counts = self._run_plan()
        self.assertEqual(counts.get("strategy_slogan_stage"), 0)
        self.assertEqual(counts.get("conceptual_stage"), 0)
        self.assertEqual(counts.get("brand_physical"), 0)
        self.assertEqual(counts.get("graphic_system"), 0)
        self.assertEqual(counts.get("series_ads"), 1)

    def test_request_fingerprint_change_blocks_reuse(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        _, counts = self._run_plan(product_description="Different brief text entirely")
        self.assertEqual(counts.get("strategy_slogan_stage"), 1)
        self.assertEqual(counts.get("conceptual_stage"), 1)

    def test_product_description_change_blocks_unsafe_reuse(self) -> None:
        self.test_request_fingerprint_change_blocks_reuse()

    def test_selected_creative_brief_change_invalidates_downstream(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        strategy_output = dict(record["completedStages"]["strategy_slogan_stage"]["output"])
        strategy_output["selectedCreativeBrief"] = {
            "essentialFacts": ["Changed fact"],
            "supportingEvidence": [],
            "mandatoryConstraints": [],
        }
        record["completedStages"]["strategy_slogan_stage"]["output"] = strategy_output
        save_planning_checkpoint_record(JOB_ID, record)
        _, counts = self._run_plan()
        self.assertEqual(counts.get("conceptual_stage"), 1)

    def test_strategy_output_change_invalidates_downstream(self) -> None:
        from engine.builder1_planning_checkpoint import _output_fingerprint

        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        strategy_output = dict(record["completedStages"]["strategy_slogan_stage"]["output"])
        strategy_output["selectedStrategy"]["relativeAdvantage"] = "Changed advantage"
        record["completedStages"]["strategy_slogan_stage"]["output"] = strategy_output
        record["completedStages"]["strategy_slogan_stage"]["outputFingerprint"] = _output_fingerprint(
            strategy_output
        )
        for downstream in ("conceptual_stage", "brand_physical", "graphic_system", "series_ads"):
            record["completedStages"].pop(downstream, None)
        save_planning_checkpoint_record(JOB_ID, record)
        _, counts = self._run_plan()
        self.assertEqual(counts.get("strategy_slogan_stage"), 0)
        self.assertEqual(counts.get("conceptual_stage"), 1)

    def test_conceptual_output_change_invalidates_physical(self) -> None:
        from engine.builder1_planning_checkpoint import _output_fingerprint

        with patch(
            "engine.builder1_planning_pipeline._run_graphic_system_stage",
            side_effect=Builder1PlannerError("graphic_system_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        conceptual_output = dict(record["completedStages"]["conceptual_stage"]["output"])
        conceptual_output["selectedConceptual"]["generator"] = "Changed generator"
        record["completedStages"]["conceptual_stage"]["output"] = conceptual_output
        record["completedStages"]["conceptual_stage"]["outputFingerprint"] = _output_fingerprint(
            conceptual_output
        )
        for downstream in ("brand_physical", "graphic_system", "series_ads"):
            record["completedStages"].pop(downstream, None)
        save_planning_checkpoint_record(JOB_ID, record)
        _, counts = self._run_plan()
        self.assertEqual(counts.get("conceptual_stage"), 0)
        self.assertEqual(counts.get("brand_physical"), 1)

    def test_stage_contract_version_change_reruns_only_affected_stage(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline._run_graphic_system_stage",
            side_effect=Builder1PlannerError("graphic_system_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        record["completedStages"]["brand_physical"]["stageContractVersion"] = "1"
        save_planning_checkpoint_record(JOB_ID, record)
        _, counts = self._run_plan()
        self.assertEqual(counts.get("strategy_slogan_stage"), 0)
        self.assertEqual(counts.get("conceptual_stage"), 0)
        self.assertEqual(counts.get("brand_physical"), 1)
        self.assertEqual(counts.get("graphic_system"), 1)

    def test_malformed_checkpoint_fails_safe(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        record["completedStages"]["strategy_slogan_stage"]["output"] = "not-a-dict"
        save_planning_checkpoint_record(JOB_ID, record)
        _, counts = self._run_plan()
        self.assertEqual(counts.get("strategy_slogan_stage"), 1)

    def test_wrong_campaign_checkpoint_rejected(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        record["identity"]["campaignId"] = "other-campaign"
        save_planning_checkpoint_record(JOB_ID, record)
        _, counts = self._run_plan()
        self.assertEqual(counts.get("strategy_slogan_stage"), 1)

    def test_checkpoint_persist_failure_blocks_next_stage(self) -> None:
        with patch(
            "engine.builder1_planning_checkpoint._save_checkpoint_record",
            side_effect=PlanningCheckpointPersistError("save_failed"),
        ):
            with self.assertRaises(PlanningCheckpointPersistError):
                self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        self.assertTrue(record is None or "series_ads" not in dict(record.get("completedStages") or {}))

    def test_fresh_supplied_name_five_calls(self) -> None:
        _, counts = self._run_plan()
        planning_calls = sum(
            counts.get(stage, 0)
            for stage in (
                "strategy_slogan_stage",
                "conceptual_stage",
                "brand_physical",
                "graphic_system",
                "series_ads",
            )
        )
        self.assertEqual(planning_calls, 5)

    def test_fresh_generated_name_six_calls(self) -> None:
        _, counts = self._run_plan(product_name="")
        planning_calls = counts.get("product_name_resolution", 0) + sum(
            counts.get(stage, 0)
            for stage in (
                "strategy_slogan_stage",
                "conceptual_stage",
                "brand_physical",
                "graphic_system",
                "series_ads",
            )
        )
        self.assertEqual(planning_calls, 6)

    def test_resumed_campaign_excludes_reused_stages_from_new_calls(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.run_brand_physical_with_memory_guard",
            side_effect=Builder1PlannerError("brand_physical_failed"),
        ):
            with self.assertRaises(Builder1PlannerError):
                self._run_plan()
        with self.assertLogs("engine.builder1_planning_metrics", level="INFO") as captured:
            self._run_plan()
        summary = next(line for line in captured.output if "BUILDER1_PLANNING_CALL_SUMMARY" in line)
        self.assertIn("reusedPlanningStages=2", summary)
        self.assertIn("newPlanningCalls=3", summary)

    def test_checkpoint_record_shape(self) -> None:
        self._run_plan()
        record = load_planning_checkpoint_record(JOB_ID)
        self.assertEqual(record.get("version"), CHECKPOINT_VERSION)
        self.assertEqual(record["identity"]["jobId"], JOB_ID)
        self.assertTrue(record.get("explorationSeed"))
        self.assertTrue(record.get("lensOrder"))


class TestPlanningCheckpointUnit(unittest.TestCase):
    def setUp(self) -> None:
        delete_planning_checkpoint(JOB_ID)

    def tearDown(self) -> None:
        delete_planning_checkpoint(JOB_ID)

    def test_stage_contract_versions_present(self) -> None:
        for stage in (
            "strategy_slogan_stage",
            "conceptual_stage",
            "brand_physical",
            "graphic_system",
            "series_ads",
        ):
            self.assertIn(stage, STAGE_CHECKPOINT_CONTRACT_VERSIONS)

    def test_persist_invalidates_downstream(self) -> None:
        session = PlanningCheckpointSession.open(_identity())
        session.persist_stage("strategy_slogan_stage", output={"a": 1}, dependency_fingerprint="dep1")
        session.persist_stage("conceptual_stage", output={"b": 2}, dependency_fingerprint="dep2")
        session.persist_stage("strategy_slogan_stage", output={"a": 2}, dependency_fingerprint="dep1")
        record = load_planning_checkpoint_record(JOB_ID)
        stages = dict(record.get("completedStages") or {})
        self.assertIn("strategy_slogan_stage", stages)
        self.assertNotIn("conceptual_stage", stages)


if __name__ == "__main__":
    unittest.main()
