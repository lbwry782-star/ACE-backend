"""
Builder2 Winner continuous-event sceneVariations normalization tests.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_headline_repair import validate_and_finalize_repaired_winner_plan
from engine.builder2_winner_plan import _clean_scene_variations, validate_builder2_winner_plan
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
    offline_revalidate_parsed_winner_response,
    process_winner_development_response,
)
from engine.builder2_winner_repair_failure_inspect import inspect_builder2_winner_repair_failure
from engine.builder2_winner_scene_variations_normalization import (
    normalize_continuous_event_scene_variations_for_execution,
)
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan
from tests.test_builder2_winner_headline_repair import (
    _forgot_winner_id,
    _judgment_requiring_headline,
    _parsed_plan_missing_headline,
    _six_six_missing_headline_state,
)
from tests.test_builder2_winner_repair_failure_inspect import _preserved_plan_from_state, _repaired_failure_state


def _processing_context(state: Dict[str, Any]) -> Dict[str, Any]:
    winner_id = _forgot_winner_id(state)
    winner_rec = state["candidates"][winner_id]
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
    winning_judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
    strategy = state["strategyFoundation"]
    source_reference = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=winner_id,
    )
    preservation_snapshot = build_winning_candidate_preservation_snapshot(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=winner_id,
    )
    return {
        "winner_id": winner_id,
        "winning_candidate": winning_candidate,
        "winning_judgment": winning_judgment,
        "source_reference": source_reference,
        "preservation_snapshot": preservation_snapshot,
    }


class TestNormalizeContinuousEventSceneVariationsHelper(unittest.TestCase):
    def _base_plan(self, *, scene_variations: Any) -> Dict[str, Any]:
        plan = _winner_plan()
        plan["structureType"] = "continuous_event"
        if scene_variations is None:
            plan.pop("sceneVariations", None)
        else:
            plan["sceneVariations"] = scene_variations
        return plan

    def test_missing_key_normalizes_to_empty_list(self) -> None:
        plan = self._base_plan(scene_variations=None)
        self.assertTrue(normalize_continuous_event_scene_variations_for_execution(plan))
        self.assertEqual(plan["sceneVariations"], [])

    def test_empty_list_remains_empty(self) -> None:
        plan = self._base_plan(scene_variations=[])
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["sceneVariations"], [])

    def test_four_valid_strings_normalize_to_empty(self) -> None:
        plan = self._base_plan(
            scene_variations=["Beat one.", "Beat two.", "Beat three.", "Beat four."]
        )
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["sceneVariations"], [])

    def test_invalid_dictionary_entries_normalize_to_empty(self) -> None:
        plan = self._base_plan(scene_variations=[{"familyId": "nearness"} for _ in range(4)])
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["sceneVariations"], [])

    def test_variation_only_dictionaries_normalize_to_empty(self) -> None:
        plan = self._base_plan(
            scene_variations=[{"variation": "Visible beat", "familyId": "nearness"} for _ in range(4)]
        )
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["sceneVariations"], [])

    def test_sequence_unchanged(self) -> None:
        plan = self._base_plan(scene_variations=[{"familyId": "nearness"} for _ in range(4)])
        sequence = deepcopy(plan["sequence"])
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["sequence"], sequence)

    def test_video_prompt_unchanged(self) -> None:
        plan = self._base_plan(scene_variations=[{"familyId": "nearness"} for _ in range(4)])
        video_prompt = plan["videoPrompt"]
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["videoPrompt"], video_prompt)

    def test_headline_fields_unchanged(self) -> None:
        plan = self._base_plan(scene_variations=[{"familyId": "nearness"} for _ in range(4)])
        headline = plan["headline"]
        keyword = plan["headlineCoreKeyword"]
        decision = deepcopy(plan["headlineDecision"])
        form = plan["headlineForm"]
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["headline"], headline)
        self.assertEqual(plan["headlineCoreKeyword"], keyword)
        self.assertEqual(plan["headlineDecision"], decision)
        self.assertEqual(plan["headlineForm"], form)

    def test_advertising_closure_unchanged(self) -> None:
        plan = self._base_plan(scene_variations=[{"familyId": "nearness"} for _ in range(4)])
        plan["advertisingClosure"] = {
            "required": True,
            "productNameText": "ACE Product",
            "sloganText": "Quality you can trust",
            "language": "en",
        }
        closure = deepcopy(plan["advertisingClosure"])
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["advertisingClosure"], closure)

    def test_idempotent(self) -> None:
        plan = self._base_plan(scene_variations=[{"familyId": "nearness"} for _ in range(4)])
        normalize_continuous_event_scene_variations_for_execution(plan)
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["sceneVariations"], [])

    def test_variation_montage_not_changed(self) -> None:
        plan = _winner_plan()
        plan["structureType"] = "variation_montage"
        variations = [
            {"description": "Two people stand apart.", "familyId": "nearness"},
            {"description": "One step closes the distance.", "familyId": "nearness"},
        ]
        plan["sceneVariations"] = deepcopy(variations)
        self.assertFalse(normalize_continuous_event_scene_variations_for_execution(plan))
        self.assertEqual(plan["sceneVariations"], variations)

    def test_clean_scene_variations_still_strict_for_montage(self) -> None:
        sequence = {"beginning": "A", "development": "B", "resolution": "C"}
        with self.assertRaises(Builder2TournamentError) as ctx:
            _clean_scene_variations(
                [{"familyId": "nearness"}, {"description": "Valid.", "familyId": "nearness"}],
                structure="variation_montage",
                sequence=sequence,
            )
        self.assertEqual(str(ctx.exception.args[0]), "builder2_winner_development_failed")


class TestNormalizeContinuousEventSceneVariationsProcessingPaths(unittest.TestCase):
    def _state_with_scene_variations(self, scene_variations: Any) -> Dict[str, Any]:
        state = _six_six_missing_headline_state(repair_calls=1)
        winner_id = _forgot_winner_id(state)
        plan = _parsed_plan_missing_headline(candidate_id=winner_id)
        plan["headline"] = "Quality speaks clearly here"
        plan["headlineCoreKeyword"] = "Quality"
        plan["sceneVariations"] = scene_variations
        state[PARSED_WINNER_RESPONSE_KEY] = {
            "parsed": plan,
            "candidateId": winner_id,
            "prototypeId": "forgot",
            "topLevelKeyCount": len(plan),
        }
        return state

    def test_process_winner_development_response_applies_normalization(self) -> None:
        state = self._state_with_scene_variations([{"familyId": "nearness"} for _ in range(4)])
        ctx = _processing_context(state)
        plan = dict(state[PARSED_WINNER_RESPONSE_KEY]["parsed"])
        validated = process_winner_development_response(
            plan,
            source_reference=ctx["source_reference"],
            winning_candidate=ctx["winning_candidate"],
            preservation_snapshot=ctx["preservation_snapshot"],
            winning_judgment=ctx["winning_judgment"],
            compatibility_mode=False,
            job_id="job-process",
            tournament_id="tournament-process",
        )
        self.assertEqual(validated["sceneVariations"], [])

    def test_offline_revalidation_applies_normalization(self) -> None:
        state = self._state_with_scene_variations([{"familyId": "nearness"} for _ in range(4)])
        ctx = _processing_context(state)
        validated = offline_revalidate_parsed_winner_response(
            state,
            source_reference=ctx["source_reference"],
            winning_candidate=ctx["winning_candidate"],
            preservation_snapshot=ctx["preservation_snapshot"],
            winning_judgment=ctx["winning_judgment"],
            compatibility_mode=False,
            job_id="job-offline",
            tournament_id="tournament-offline",
        )
        self.assertEqual(validated["sceneVariations"], [])

    def test_repaired_finalization_applies_normalization(self) -> None:
        state = self._state_with_scene_variations([{"familyId": "nearness"} for _ in range(4)])
        ctx = _processing_context(state)
        validated = validate_and_finalize_repaired_winner_plan(
            dict(state[PARSED_WINNER_RESPONSE_KEY]["parsed"]),
            source_reference=ctx["source_reference"],
            winning_candidate=ctx["winning_candidate"],
            winning_judgment=ctx["winning_judgment"],
            preservation_snapshot=ctx["preservation_snapshot"],
            compatibility_mode=False,
            job_id="job-finalize",
            tournament_id="tournament-finalize",
        )
        self.assertEqual(validated["sceneVariations"], [])
        self.assertTrue(validated.get("headlineText"))

    def test_production_shaped_plan_passes_scene_variations_clean_stage(self) -> None:
        state = self._state_with_scene_variations([{"familyId": "nearness"} for _ in range(4)])
        ctx = _processing_context(state)
        preserved = _preserved_plan_from_state(state)
        with self.assertRaises(Builder2TournamentError):
            validate_builder2_winner_plan(
                deepcopy(preserved),
                winning_candidate=ctx["winning_candidate"],
                preservation_snapshot=ctx["preservation_snapshot"],
                winning_judgment=ctx["winning_judgment"],
                compatibility_mode=False,
            )
        validated = process_winner_development_response(
            dict(state[PARSED_WINNER_RESPONSE_KEY]["parsed"]),
            source_reference=ctx["source_reference"],
            winning_candidate=ctx["winning_candidate"],
            preservation_snapshot=ctx["preservation_snapshot"],
            winning_judgment=ctx["winning_judgment"],
            compatibility_mode=False,
        )
        self.assertEqual(validated["sceneVariations"], [])

    def test_invalid_variation_montage_still_rejected(self) -> None:
        plan = _winner_plan()
        plan["structureType"] = "variation_montage"
        plan["sceneVariations"] = [{"familyId": "nearness"}, {"description": "Valid.", "familyId": "nearness"}]
        with self.assertRaises(Builder2TournamentError):
            validate_builder2_winner_plan(plan)


class TestNormalizeContinuousEventSceneVariationsInspector(unittest.TestCase):
    def _production_shaped_state(self) -> Dict[str, Any]:
        state = _repaired_failure_state(headline="Quality speaks clearly here", keyword="Quality")
        plan = state[PARSED_WINNER_RESPONSE_KEY]["parsed"]
        plan["sceneVariations"] = [{"familyId": "nearness"} for _ in range(4)]
        return state

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_inspector_reports_normalization_and_offline_validity(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(self._production_shaped_state())
        report = inspect_builder2_winner_repair_failure("job-inspect-normalized")
        normalization = report["continuousEventSceneVariationsNormalization"]
        self.assertTrue(normalization["applicable"])
        self.assertEqual(normalization["originalListCount"], 4)
        self.assertEqual(normalization["normalizedListCount"], 0)
        self.assertFalse(normalization["normalizationChangedOtherFields"])
        self.assertTrue(report["offlineWinnerRevalidationAfterNormalizationAttempted"])
        self.assertTrue(report["offlineWinnerRevalidationAfterNormalizationAccepted"])
        self.assertTrue(report["finalWinnerPlanValidOffline"])
        self.assertTrue(report["headlineCompositionAccepted"])
        self.assertTrue(report["headlineTextDerived"])
        self.assertFalse(report["additionalPaidCallRequired"])
        self.assertEqual(report["inspectionOpenAICalls"], 0)
        self.assertEqual(report["redisMutations"], 0)

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_inspector_reports_next_failure_after_normalization(self, read_raw: Any, _redis: Any) -> None:
        state = self._production_shaped_state()
        state[PARSED_WINNER_RESPONSE_KEY]["parsed"]["headline"] = "totally unrelated phrase"
        state[PARSED_WINNER_RESPONSE_KEY]["parsed"]["headlineCoreKeyword"] = "Quality"
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_winner_repair_failure("job-inspect-next-fail")
        self.assertTrue(report["offlineWinnerRevalidationAfterNormalizationAttempted"])
        self.assertFalse(report["offlineWinnerRevalidationAfterNormalizationAccepted"])
        self.assertEqual(report["firstFailureAfterNormalizationStage"], "headlineCompositionValidation")
        self.assertFalse(report["finalWinnerPlanValidOffline"])
        self.assertFalse(report["additionalPaidCallRequired"])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_inspector_read_only_no_redis_mutation(self, read_raw: Any, _redis: Any) -> None:
        original = deepcopy(self._production_shaped_state())
        read_raw.return_value = deepcopy(original)
        inspect_builder2_winner_repair_failure("job-read-only")
        self.assertEqual(read_raw.return_value[PARSED_WINNER_RESPONSE_KEY], original[PARSED_WINNER_RESPONSE_KEY])


class TestReasoningResumeOfflineSceneVariationsRecovery(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _production_state(self) -> Dict[str, Any]:
        state = _six_six_missing_headline_state(repair_calls=1)
        winner_id = _forgot_winner_id(state)
        plan = _parsed_plan_missing_headline(candidate_id=winner_id)
        plan["headline"] = "Quality speaks clearly here"
        plan["headlineCoreKeyword"] = "Quality"
        plan["sceneVariations"] = [{"familyId": "nearness"} for _ in range(4)]
        state[PARSED_WINNER_RESPONSE_KEY] = {
            "parsed": plan,
            "candidateId": winner_id,
            "prototypeId": "forgot",
            "topLevelKeyCount": len(plan),
        }
        state["winnerCandidateId"] = winner_id
        state["failureStage"] = "winner_development"
        state["failureReason"] = "builder2_winner_development_failed"
        ensure_metrics(state)
        state["metrics"]["winnerDevelopmentCalls"] = 2
        state["metrics"]["winnerNormalCalls"] = 1
        state["metrics"]["winnerRepairCalls"] = 1
        state["metrics"]["winnerRetryCalls"] = 0
        return state

    def test_offline_recovery_without_paid_calls(self) -> None:
        state = self._production_state()
        working: Dict[str, Any] = {}
        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate"
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate"
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate"
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
            side_effect=lambda job_id, tournament_state: working.update(deepcopy(tournament_state)),
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
                max_calls=1,
                stop_before_media=True,
            )
        self.assertTrue(report["ok"])
        self.assertTrue(report.get("stoppedBeforeMedia"))
        self.assertEqual(report["winnerCallsThisRun"], 0)
        self.assertEqual(report["creatorCallsThisRun"], 0)
        self.assertEqual(report["judgeCallsThisRun"], 0)
        self.assertTrue(report.get("winnerDevelopmentAccepted"))
        creator_mock.assert_not_called()
        judge_mock.assert_not_called()
        winner_mock.assert_not_called()
        self.assertEqual(working["metrics"]["winnerNormalCalls"], 1)
        self.assertEqual(working["metrics"]["winnerRepairCalls"], 1)
        self.assertEqual(working["metrics"]["winnerRetryCalls"], 0)
        self.assertTrue(is_valid_persisted_winner_development(working))


class TestNormalizeContinuousEventSceneVariationsBuilder1Isolation(unittest.TestCase):
    def test_builder1_unchanged(self) -> None:
        import glob
        import os

        root = os.path.dirname(os.path.dirname(__file__))
        for path in glob.glob(os.path.join(root, "engine", "builder1*.py")):
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("winner_scene_variations_normalization", source)


if __name__ == "__main__":
    unittest.main()
