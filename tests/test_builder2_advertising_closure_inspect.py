"""
Builder2 Advertising Closure foundation inspector tests — mocks only.
"""
from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_advertising_closure_inspect import (
    DEFAULT_INSPECT_JOB_ID,
    build_proposal_diagnostics,
    inspect_advertising_closure_foundation,
)
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_preservation_contract import SERVER_OWNED_WINNER_SOURCE_KEY
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_media_resume import HISTORICAL_JOB_ID, _media_ready_state
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt
from tests.test_builder2_winner_resume import _historical_judged_state


HISTORICAL_CANDIDATE_ID = "cand-1-summer_fan-1-57f415ca"
PROPOSED_PRODUCT = "קופי"
PROPOSED_SLOGAN = "הקפה שהופך לחלק מהדרך"


def _foundation_state(*, with_proposal: bool = True, with_judgment_assessment: bool = True) -> Dict[str, Any]:
    state = _media_ready_state(job_id=HISTORICAL_JOB_ID)
    candidate_id = str(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId") or HISTORICAL_CANDIDATE_ID)
    strategy = state.get("strategyFoundation") or _strategy(language="he")
    candidate = state["candidates"][candidate_id]
    candidate["creatorSnapshot"] = candidate.get("creatorSnapshot") or candidate.get("creatorOutput") or _candidate("summer_fan")
    plan = _winner_plan_from_prompt("")
    plan.update(methodology_winner_extras(headline_decision="omit", winning_candidate=candidate["creatorSnapshot"], strategy=strategy))
    plan["productNameResolved"] = "ACE Product"
    plan["language"] = "he"
    plan["headlineDecision"] = {"decision": "omit", "reasonSource": "not_required"}
    plan["headlineForm"] = "none"
    plan["headline"] = ""
    plan["headlineText"] = ""
    plan[SERVER_OWNED_WINNER_SOURCE_KEY] = {
        "sourceCandidateId": candidate_id,
        "sourcePrototypeId": "summer_fan",
        "problemPerception": strategy.get("problemPerception"),
        "relativeAdvantage": strategy.get("relativeAdvantage"),
        "coreCreativeMechanism": candidate["creatorSnapshot"].get("coreCreativeMechanism"),
        "coreVisualIdea": candidate["creatorSnapshot"].get("coreVisualIdea"),
        "visualAnchor": candidate["creatorSnapshot"].get("visualAnchor"),
        "prototypeMethodContract": candidate["creatorSnapshot"].get("prototypeMethodApplication"),
        "visualParallelType": candidate["creatorSnapshot"].get("visualParallelType"),
        "participationMechanism": candidate["creatorSnapshot"].get("participationMechanism"),
    }
    state["winnerDevelopmentPlan"] = plan
    state["winnerDevelopmentCandidateId"] = candidate_id
    state["winnerDevelopmentPrototypeId"] = "summer_fan"
    candidate["totalScore"] = 80
    judgment_id = str(candidate.get("judgmentId") or "judge-summer-fan")
    judgment_payload = methodology_judgment_extras()
    if not with_judgment_assessment:
        judgment_payload.pop("advertisingCompletionAssessment", None)
    state.setdefault("judgments", {})[judgment_id] = {
        "judgmentId": judgment_id,
        "candidateId": candidate_id,
        "totalScore": 80,
        "judgment": judgment_payload,
    }
    state.setdefault("acceptedCreatorCandidates", {})[candidate_id] = {
        "candidateId": candidate_id,
        "prototypeId": "summer_fan",
        "validationStatus": "accepted",
        "creatorOutput": candidate["creatorSnapshot"],
    }
    if with_proposal:
        state["advertisingClosure"] = {
            "required": True,
            "productNameText": PROPOSED_PRODUCT,
            "sloganText": PROPOSED_SLOGAN,
            "language": "he",
            "presentationMode": "end_card",
            "durationSeconds": 1.5,
            "headlineSource": "advertising_closure_role",
            "noLogo": True,
        }
        state["advertisingClosureStatus"] = "proposed"
    return state


class TestAdvertisingClosureFoundationInspect(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_authoritative_winner_fields_loaded(self, load_state: Any, _redis: Any) -> None:
        load_state.return_value = _foundation_state()
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertTrue(report["ok"])
        self.assertTrue(report["foundation"]["problemPerception"]["present"])
        self.assertTrue(report["foundation"]["relativeAdvantage"]["present"])
        self.assertTrue(report["foundation"]["coreCreativeMechanism"]["present"])

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_correct_selected_candidate_and_judge(self, load_state: Any, _redis: Any) -> None:
        state = _foundation_state()
        load_state.return_value = state
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertEqual(report["winnerCandidateId"], state["winnerDevelopmentCandidateId"])
        self.assertEqual(report["winnerPrototypeId"], "summer_fan")
        self.assertEqual(report["winnerScore"], 80)
        self.assertTrue(report["selectedJudgment"]["headlineNecessityAssessment"]["present"])

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_server_owned_fields_marked_authoritative(self, load_state: Any, _redis: Any) -> None:
        load_state.return_value = _foundation_state()
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertTrue(report["foundation"]["problemPerception"]["authoritative"])
        self.assertIn(SERVER_OWNED_WINNER_SOURCE_KEY, report["foundation"]["problemPerception"]["sourcePath"])

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_closure_proposal_non_authoritative(self, load_state: Any, _redis: Any) -> None:
        load_state.return_value = _foundation_state()
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertFalse(report["currentProposal"]["authoritative"])
        self.assertEqual(report["currentProposal"]["productNameText"], PROPOSED_PRODUCT)
        self.assertEqual(report["currentProposal"]["sloganText"], PROPOSED_SLOGAN)

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_missing_fields_reported_without_invention(self, load_state: Any, _redis: Any) -> None:
        state = _foundation_state(with_proposal=False)
        plan = state["winnerDevelopmentPlan"]
        plan.pop("advertisingPromise", None)
        load_state.return_value = state
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertFalse(report["foundation"]["advertisingPromise"]["present"])
        self.assertIsNone(report["foundation"]["advertisingPromise"]["value"])

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_source_paths_included(self, load_state: Any, _redis: Any) -> None:
        load_state.return_value = _foundation_state()
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertTrue(report["foundation"]["productNameResolved"]["sourcePath"])
        self.assertTrue(report["currentProposal"]["sourcePath"])

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.video_jobs_redis.get_redis")
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_inspector_performs_zero_redis_writes(self, load_state: Any, get_redis: Any, _redis: Any) -> None:
        load_state.return_value = _foundation_state()
        redis_client = MagicMock()
        get_redis.return_value = redis_client
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertTrue(report["ok"])
        self.assertEqual(report["redisMutations"], 0)
        for method_name in ("hset", "set", "expire", "lpush", "pipeline"):
            getattr(redis_client, method_name).assert_not_called()

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_zero_external_calls(self, load_state: Any, _redis: Any) -> None:
        load_state.return_value = _foundation_state()
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)

    @patch("engine.builder2_advertising_closure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_advertising_closure_inspect.load_tournament_state")
    def test_sensitive_fields_omitted_from_print(self, load_state: Any, _redis: Any) -> None:
        state = _foundation_state()
        state["mediaResume"] = {"runwayTaskId": "secret-task", "startImageArtifact": "data:image/png;base64,abc"}
        load_state.return_value = state
        report = inspect_advertising_closure_foundation(HISTORICAL_JOB_ID)
        output = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("secret-task", output)
        self.assertNotIn("data:image/png", output)


class TestProposalDiagnostics(unittest.TestCase):
    def test_seven_word_diagnostic(self) -> None:
        diagnostics = build_proposal_diagnostics(
            product_name="ACE Product",
            slogan="one two three four five six seven eight",
            status="proposed",
        )
        self.assertTrue(diagnostics["sloganExceedsSevenWords"])

    def test_generic_journey_phrase_diagnostic(self) -> None:
        diagnostics = build_proposal_diagnostics(
            product_name=PROPOSED_PRODUCT,
            slogan=PROPOSED_SLOGAN,
            status="proposed",
        )
        self.assertTrue(diagnostics["sloganContainsGenericJourneyPhrase"])
        self.assertFalse(diagnostics["proposalApproved"])
        self.assertFalse(diagnostics["proposalRendered"])


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
