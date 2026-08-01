"""
Builder2 Winner headline omit dependency and offline salvage — production-shaped tests.
"""
from __future__ import annotations

import copy
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_headline_decision_contract import (
    analyze_headline_omit_textual_dependency,
    validate_headline_decision_methodology,
)
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_single_slogan_contract import stamp_single_slogan_contract
from engine.builder2_tournament_contracts import Builder2TournamentError, TOURNAMENT_STATE_SCHEMA_VERSION
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_offline_salvage import (
    attempt_offline_winner_development_salvage,
    inspect_offline_winner_salvage_preconditions,
    run_offline_winner_salvage_for_job,
)
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    process_winner_development_response,
)
from engine.builder2_winner_scene_variations_normalization import (
    normalize_continuous_event_scene_variations_for_execution,
)
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt


def _summer_fan_winner_id(state: Dict[str, Any]) -> str:
    for candidate_id, record in (state.get("candidates") or {}).items():
        if record.get("prototypeId") == "summer_fan":
            return str(candidate_id)
    return "cand-1-summer_fan-1-7e19ebc6"


def _production_summer_fan_parsed_plan(*, video_prompt: str) -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("summer_fan")
    candidate["verbalPotential"] = {
        "decision": "not_needed",
        "reason": "The waving hand communicates cooling absence without a headline.",
    }
    plan = _winner_plan_from_prompt("")
    plan.update(
        methodology_winner_extras(
            headline_decision="omit",
            winning_candidate=candidate,
            strategy=strategy,
        )
    )
    plan["productNameResolved"] = "אורי לב"
    plan["language"] = "he"
    plan["prototypeId"] = "summer_fan"
    plan["structureType"] = "continuous_event"
    plan["headlineDecision"] = {
        "decision": "omit",
        "reason": "The final Closure slogan completes the business meaning after the silent visual event.",
        "reasonSource": "model",
    }
    plan["headlineForm"] = "none"
    plan["headline"] = ""
    plan["headlineText"] = ""
    plan["headlineCoreKeyword"] = ""
    plan["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
    plan["advertisingClosure"]["productNameText"] = "אורי לב"
    plan["advertisingClosure"]["sloganText"] = "קרוב אליך יותר ממה שחשבת"
    plan["sceneVariations"] = [
        "A hand begins waving quickly.",
        "The motion accelerates into familiar fan rhythm.",
        "The inferred cooling absence lands clearly.",
    ]
    plan["videoPrompt"] = video_prompt
    plan["preservationReference"] = {
        "strategyFoundationId": strategy.get("strategyFoundationId") or "strategy-test",
        "prototypeId": "summer_fan",
        "structureType": candidate.get("structureType"),
        "visualParallelType": candidate.get("visualParallelType"),
        "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
        "sourceCandidateId": "cand-1-summer_fan-1-7e19ebc6",
    }
    return plan


def _production_summer_fan_state(*, video_prompt: str) -> Dict[str, Any]:
    state = _six_prototype_state(judged=6, creators=6)
    state["jobId"] = "eab8a682-67d3-41bb-bf32-00ed05836e22"
    state["tournamentId"] = "a66a05ce-4ee7-4bb9-b1f2-ce1e50218244"
    state["builder2NewFormatVersion"] = BUILDER2_NEW_FORMAT_VERSION
    state["schemaVersion"] = TOURNAMENT_STATE_SCHEMA_VERSION
    stamp_single_slogan_contract(state)
    winner_id = _summer_fan_winner_id(state)
    for candidate_id, record in state["candidates"].items():
        if candidate_id == winner_id:
            record["totalScore"] = 91
        else:
            record["totalScore"] = 70
    state["winnerCandidateId"] = winner_id
    state["winnerDevelopmentPaidCallRecorded"] = True
    judgment = _judgment(winner_id, total_hint=91, eligible=True)
    judgment.update(methodology_judgment_extras(prototype_id="summer_fan"))
    judgment["headlineNecessityAssessment"] = {
        "headlineNeeded": False,
        "visualWouldWorkWithoutHeadline": True,
        "notes": "The waving motion communicates the cooling insight silently.",
    }
    judgment_id = state["candidates"][winner_id]["judgmentId"]
    state["judgments"][judgment_id]["judgment"] = judgment
    parsed = _production_summer_fan_parsed_plan(video_prompt=video_prompt)
    state[PARSED_WINNER_RESPONSE_KEY] = {
        "parsed": parsed,
        "candidateId": winner_id,
        "prototypeId": "summer_fan",
        "topLevelKeys": sorted(parsed.keys()),
        "topLevelKeyCount": len(parsed),
        "responseCharCount": 7459,
    }
    state["winnerDevelopmentFailure"] = {
        "stage": "methodology_validation",
        "failureField": "headlineDecision.omit_with_textual_dependency",
    }
    ensure_metrics(state)
    state["metrics"]["winnerDevelopmentCalls"] = 1
    state["metrics"]["winnerNormalCalls"] = 1
    return state


class TestHeadlineOmitDependencyAnalyzer(unittest.TestCase):
    def test_silent_policy_prohibition_does_not_trigger_dependency(self) -> None:
        plan = _production_summer_fan_parsed_plan(
            video_prompt=(
                "A continuous realistic scene of a hand waving quickly to infer a missing fan. "
                "No on-screen text during the scene. Purely pictorial motion."
            )
        )
        analysis = analyze_headline_omit_textual_dependency(plan)
        self.assertFalse(analysis["dependencyBeforeClosure"])
        self.assertTrue(analysis["dependencyOnlyOnClosureSlogan"])
        self.assertFalse(analysis["videoPromptRequestsRenderedText"])

    def test_reason_mentioning_final_slogan_is_not_scanned(self) -> None:
        plan = _production_summer_fan_parsed_plan(video_prompt="Silent continuous motion only.")
        analysis = analyze_headline_omit_textual_dependency(plan)
        self.assertFalse(analysis["dependencyBeforeClosure"])

    def test_slogan_evidence_only_is_not_pre_closure_dependency(self) -> None:
        plan = _production_summer_fan_parsed_plan(video_prompt="Silent continuous motion only.")
        evidence = plan.get("advertisingSloganEvidence")
        assert isinstance(evidence, dict)
        evidence["whyAdvertising"] = "The closing line reveals the business meaning after the visual works."
        analysis = analyze_headline_omit_textual_dependency(plan)
        self.assertFalse(analysis["dependencyBeforeClosure"])

    def test_required_caption_in_video_prompt_is_rejected(self) -> None:
        plan = _production_summer_fan_parsed_plan(
            video_prompt="The viewer must read the headline text before the scene resolves."
        )
        analysis = analyze_headline_omit_textual_dependency(plan)
        self.assertTrue(analysis["dependencyBeforeClosure"])
        self.assertIn("videoPrompt", analysis["textualDependencySourceFields"])

    def test_sequence_requiring_sign_is_rejected(self) -> None:
        plan = _production_summer_fan_parsed_plan(video_prompt="Silent motion only.")
        plan["sequence"]["resolution"] = "The viewer reads the sign to understand the cooling joke."
        analysis = analyze_headline_omit_textual_dependency(plan)
        self.assertTrue(analysis["dependencyBeforeClosure"])
        self.assertIn("sequence.resolution", analysis["textualDependencySourceFields"])


class TestProductionShapedSummerFanWinner(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_production_shaped_plan_passes_after_dependency_fix(self) -> None:
        state = _production_summer_fan_state(
            video_prompt=(
                "Continuous realistic event: a hand waves quickly, the cooling absence becomes readable. "
                "No on-screen text during the scene. No text overlay. Purely pictorial motion."
            )
        )
        winner_id = _summer_fan_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        candidate = winner_rec["creatorOutput"]
        judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        source = build_server_owned_winner_source_reference(
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            candidate_id=winner_id,
        )
        result = process_winner_development_response(
            deepcopy(state[PARSED_WINNER_RESPONSE_KEY]["parsed"]),
            source_reference=source,
            winning_candidate=candidate,
            winning_judgment=judgment,
            tournament_state=state,
        )
        self.assertEqual(result["headlineDecision"]["decision"], "omit")
        self.assertEqual(result.get("sceneVariations"), [])

    def test_inspector_reports_salvage_possible_for_production_shape(self) -> None:
        state = _production_summer_fan_state(
            video_prompt="Silent continuous motion. No on-screen text during the scene."
        )
        report = inspect_offline_winner_salvage_preconditions(state)
        self.assertTrue(report["wouldPassCorrectedHeadlineContract"])
        self.assertTrue(report["offlineWinnerSalvagePossible"])
        self.assertFalse(report["dependencyBeforeClosure"])
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["paidCalls"], 0)

    def test_offline_salvage_zero_openai_calls_and_idempotent(self) -> None:
        state = _production_summer_fan_state(
            video_prompt="Silent continuous motion. No on-screen text during the scene."
        )
        winner_id = _summer_fan_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        with patch("engine.builder2_winner_development.call_builder2_role_json_with_text") as mock_call:
            first = run_offline_winner_salvage_for_job(
                state["jobId"],
                tournament_state=state,
                save_state=False,
            )
            mock_call.assert_not_called()
        self.assertTrue(first["ok"])
        self.assertTrue(is_valid_persisted_winner_development(state))
        fingerprint_before = first["winnerResponseFingerprint"]
        second = run_offline_winner_salvage_for_job(
            state["jobId"],
            tournament_state=state,
            save_state=False,
        )
        self.assertTrue(second["ok"])
        self.assertTrue(second.get("reusedAccepted"))
        self.assertEqual(second["winnerResponseFingerprint"], fingerprint_before)

    def test_text_dependent_plan_remains_rejected(self) -> None:
        state = _production_summer_fan_state(
            video_prompt="The viewer must read the headline text to understand the scene."
        )
        winner_id = _summer_fan_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        report = inspect_offline_winner_salvage_preconditions(state)
        self.assertFalse(report["wouldPassCorrectedHeadlineContract"])
        self.assertFalse(report["offlineWinnerSalvagePossible"])
        with self.assertRaises(Builder2TournamentError):
            attempt_offline_winner_development_salvage(
                state,
                winner_candidate_id=winner_id,
                prototype_id="summer_fan",
                strategy_foundation=state["strategyFoundation"],
                winning_candidate=winner_rec["creatorOutput"],
                winning_judgment=state["judgments"][winner_rec["judgmentId"]]["judgment"],
            )


class TestContinuousEventNormalizationOrdering(unittest.TestCase):
    def test_normalization_runs_before_validation_and_preserves_provenance(self) -> None:
        plan = _production_summer_fan_parsed_plan(video_prompt="Silent motion only.")
        original_prompt = plan["videoPrompt"]
        original_sequence = deepcopy(plan["sequence"])
        normalize_continuous_event_scene_variations_for_execution(plan)
        self.assertEqual(plan["sceneVariations"], [])
        self.assertEqual(plan["videoPrompt"], original_prompt)
        self.assertEqual(plan["sequence"], original_sequence)
        provenance = plan.get("continuousEventSceneVariationsNormalization")
        assert isinstance(provenance, dict)
        self.assertEqual(provenance.get("originalListCount"), 3)
        self.assertEqual(provenance.get("normalizedListCount"), 0)


class TestHeadlineDecisionContractRegression(unittest.TestCase):
    def test_use_still_requires_headline(self) -> None:
        plan = _production_summer_fan_parsed_plan(video_prompt="Silent motion only.")
        plan["headlineDecision"] = {"decision": "use"}
        plan["headlineForm"] = "direct"
        plan["headline"] = ""
        with self.assertRaises(Builder2TournamentError):
            validate_headline_decision_methodology(plan)

    def test_omit_keeps_empty_headline_fields(self) -> None:
        plan = _production_summer_fan_parsed_plan(
            video_prompt="Silent motion only. No on-screen text during the scene."
        )
        validate_headline_decision_methodology(plan)
        self.assertEqual(plan.get("headline"), "")
        self.assertEqual(plan.get("headlineText"), "")
        self.assertEqual(plan.get("headlineCoreKeyword"), "")


if __name__ == "__main__":
    unittest.main()
