"""
Builder2 creative tournament tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder1_planning_profile import quality_model
from engine.builder2_judge import calculate_judge_total, judge_candidate, validate_judge_response
from engine.builder2_prototypes import (
    active_prototype_ids,
    get_prototype,
    reference_only_prototype_ids,
)
from engine.builder2_tournament_config import (
    DEFAULT_ACTIVE_PROTOTYPE_IDS,
    resolve_builder2_active_prototype_ids,
    resolve_builder2_tournament_enabled,
)
from engine.builder2_tournament_contracts import (
    CANDIDATE_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    STRATEGY_SCHEMA_VERSION,
    WINNER_PLAN_SCHEMA_VERSION,
    Builder2TournamentError,
    compare_candidate_rankings,
)
from engine.builder2_creator import (
    generate_creator_candidate,
    validate_creator_candidate,
    validate_creator_purity,
)
from engine.builder2_strategy import generate_strategy_foundation, validate_strategy_foundation
from engine.builder2_tournament_manager import (
    run_builder2_tournament,
    select_global_winner,
)
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from engine.builder2_winner_development import normalize_winner_plan_for_runway, validate_winner_plan
from engine.runway_video import RunwayVideoMVPError, _generate_one_video_mvp_body


def _strategy(language: str = "en") -> Dict[str, Any]:
    return {
        "schemaVersion": STRATEGY_SCHEMA_VERSION,
        "productNameResolved": "ACE Product",
        "language": language,
        "problemPerception": {
            "statement": "Buyers struggle to see why this product beats familiar alternatives.",
            "groundingType": "common_market_behavior",
            "groundingEvidence": ["Customers compare against familiar agencies by default."],
            "whyItMatters": "The product must reframe the comparison.",
        },
        "relativeAdvantage": {
            "statement": "Closeness becomes the advantage.",
            "derivationFromProblem": "Because buyers default to familiar options, being closer is reframed as better fit.",
        },
        "mechanismScan": {
            "domainFacts": ["People choose what feels nearest to their need."],
            "discoveredMechanism": "Physical closeness can express strategic closeness.",
            "creativeOpportunity": "Show closeness as the persuasive proof.",
        },
    }


def _candidate(prototype_id: str, *, structure: str = "continuous_event") -> Dict[str, Any]:
    return {
        "schemaVersion": CANDIDATE_SCHEMA_VERSION,
        "prototypeId": prototype_id,
        "prototypeMethodApplied": "Applied the reusable method without copying the literal source ad.",
        "coreCreativeMechanism": "Closeness shown through a simple human gesture.",
        "conceptSummary": "One clear visual proof of closeness.",
        "visualParallelType": "physical_behavior",
        "visualFamily": "human closeness gestures",
        "structureType": structure,
        "sevenSecondStructure": {
            "beginning": "Two people stand apart.",
            "development": "One step closes the distance.",
            "resolution": "They meet in a clear embrace.",
        },
        "visualAnchor": {
            "description": "The moment the distance closes.",
            "whyEssential": "It proves closeness visually.",
        },
        "silentVerification": {
            "understandableWithoutAudio": True,
            "explanation": "The closing distance is visible without sound.",
        },
        "runwayFeasibility": {
            "mainSubject": "Two people",
            "mainAction": "One person steps forward and they hug",
            "location": "Simple neutral room",
            "openingFrame": "Two people with visible space between them",
            "continuityRisk": "low",
            "generationRisks": [],
            "whyRunwayShouldUnderstand": "Single continuous human action in one room.",
        },
        "editingPlan": {
            "purpose": "Make closeness legible quickly.",
            "reveal": "Distance closes before the embrace.",
            "pacing": "Begin wide, resolve in a steady close.",
        },
        "creatorReport": {
            "problemPerception": "Buyers default to familiar alternatives.",
            "relativeAdvantage": "Closeness becomes the advantage.",
            "mechanismScanSummary": "Physical closeness expresses strategic closeness.",
            "goldPrototypeUsed": prototype_id,
            "visualParallelType": "physical_behavior",
            "whyParallelExpressesAdvantage": "Closing distance makes closeness visible.",
            "whyRunwayShouldUnderstand": "One action in one location.",
        },
    }


def _judgment(candidate_id: str, *, total_hint: int = 80, eligible: bool = True) -> Dict[str, Any]:
    scores = {
        "problemAdvantageIntegrity": min(20, total_hint // 5),
        "mechanismQuality": 12,
        "prototypeMethodApplication": 8,
        "silentVisualClarity": 12,
        "originalityFreshness": 10,
        "eleganceSimplicity": 8,
        "runwayFeasibility": 8,
        "editingContribution": 4,
    }
    return {
        "schemaVersion": JUDGMENT_SCHEMA_VERSION,
        "candidateId": candidate_id,
        "eligible": eligible,
        "disqualifiers": [] if eligible else ["fabricated_problem"],
        "scores": scores,
        "verdict": "Strong silent visual proof.",
        "strengths": ["Clear mechanism"],
        "weaknesses": [],
        "prototypeQualityComparison": "Applies method without copying surface.",
        "confidence": 0.82,
    }


def _winner_plan(language: str = "en") -> Dict[str, Any]:
    return {
        "schemaVersion": WINNER_PLAN_SCHEMA_VERSION,
        "productNameResolved": "ACE Product",
        "language": language,
        "problemPerception": "Buyers default to familiar alternatives.",
        "relativeAdvantage": "Closeness becomes the advantage.",
        "prototypeId": "closest",
        "coreCreativeMechanism": "Distance closes into an embrace.",
        "visualParallelType": "physical_behavior",
        "visualFamily": "human closeness gestures",
        "structureType": "continuous_event",
        "headline": "Closer than you think",
        "headlineCoreKeyword": "Closer",
        "coreVisualIdea": "closing distance",
        "sequence": {
            "beginning": "Two people stand apart.",
            "development": "One step closes the distance.",
            "resolution": "They meet in a clear embrace.",
        },
        "sceneVariations": [],
        "visualAnchor": "The moment the distance closes.",
        "openingFrameDescription": "Two people with visible space between them.",
        "videoPrompt": (
            "A seven-second continuous realistic scene: two people stand apart, one step closes the "
            "distance, they embrace clearly. Natural movement. No text."
        ),
    }


class TournamentMockLLM:
    def __init__(self, *, score_by_prototype: Dict[str, int] | None = None) -> None:
        self.calls: List[str] = []
        self.score_by_prototype = score_by_prototype or {}
        self._candidate_ids: Dict[str, str] = {}

    def __call__(self, *, role: str, model: str, prompt: str) -> Dict[str, Any]:
        self.calls.append(role)
        if role == "builder2_strategy":
            return _strategy()
        if role == "builder2_creator":
            prototype_id = "closest"
            for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                if f"Assigned prototype ID: {pid}" in prompt:
                    prototype_id = pid
                    break
            if prototype_id == "think_small" and "weakness" not in prompt:
                pass
            candidate = _candidate(prototype_id)
            if prototype_id == "think_small":
                candidate["creatorReport"]["problemPerception"] = (
                    "The real weakness is limited size, and the ad inverts it."
                )
            return candidate
        if role == "builder2_judge":
            candidate_id = "unknown"
            for token in prompt.split():
                if token.startswith("cand-"):
                    candidate_id = token.strip()
                    break
            prototype_id = "closest"
            for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                if pid in prompt:
                    prototype_id = pid
                    break
            total = self.score_by_prototype.get(prototype_id, 70)
            return _judgment(candidate_id, total_hint=total)
        if role == "builder2_winner":
            return _winner_plan()
        raise AssertionError(f"unexpected role {role}")


class TestBuilder2TournamentConfig(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_enabled_by_default(self) -> None:
        self.assertTrue(resolve_builder2_tournament_enabled())

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_ENABLED": "false"}, clear=True)
    def test_explicit_false_uses_legacy_flag(self) -> None:
        self.assertFalse(resolve_builder2_tournament_enabled())

    @patch.dict(os.environ, {}, clear=True)
    def test_active_pool_defaults(self) -> None:
        self.assertEqual(resolve_builder2_active_prototype_ids(), list(DEFAULT_ACTIVE_PROTOTYPE_IDS))

    def test_reference_only_not_active(self) -> None:
        self.assertNotIn("context_collision", active_prototype_ids())

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "unknown_proto"}, clear=True)
    def test_unknown_prototype_fails(self) -> None:
        with self.assertRaises(Exception):
            resolve_builder2_active_prototype_ids()


class TestBuilder2TournamentValidation(unittest.TestCase):
    def test_strategy_validation(self) -> None:
        validate_strategy_foundation(_strategy())

    def test_creator_validation_and_purity(self) -> None:
        cand = _candidate("closest")
        validate_creator_candidate(cand, assigned_prototype_id="closest", prototype_display_name="Closest")
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_purity({**cand, "conceptSummary": "This outperforms the previous candidate."})
        self.assertIn("builder2_creator_purity_violation", str(ctx.exception.args[0]))

    def test_judge_score_recalculation(self) -> None:
        judgment = _judgment("cand-1")
        parsed, total, scores = validate_judge_response(judgment, candidate_id="cand-1")
        self.assertEqual(total, calculate_judge_total(scores))
        self.assertNotIn("total", parsed.get("scores", {}))

    def test_model_total_not_authoritative(self) -> None:
        bad = _judgment("cand-1")
        bad["totalScore"] = 999
        with self.assertRaises(Builder2TournamentError):
            validate_judge_response(bad, candidate_id="cand-1")


class TestBuilder2TournamentLogic(unittest.TestCase):
    def test_best_attempt_preservation(self) -> None:
        better = {
            "candidateId": "a",
            "totalScore": 90,
            "tieScores": {"silentVisualClarity": 14, "problemAdvantageIntegrity": 18, "runwayFeasibility": 9},
            "completedAt": "2026-01-02T00:00:00+00:00",
            "eligible": True,
        }
        weaker = {
            "candidateId": "b",
            "totalScore": 70,
            "tieScores": {"silentVisualClarity": 10, "problemAdvantageIntegrity": 12, "runwayFeasibility": 7},
            "completedAt": "2026-01-03T00:00:00+00:00",
            "eligible": True,
        }
        self.assertGreater(compare_candidate_rankings(better, weaker), 0)
        self.assertLess(compare_candidate_rankings(weaker, better), 0)

    def test_global_winner_can_be_eliminated_prototype(self) -> None:
        state = {
            "candidates": {
                "c1": {
                    "eligible": True,
                    "validationStatus": "accepted",
                    "totalScore": 95,
                    "tieScores": {"silentVisualClarity": 15, "problemAdvantageIntegrity": 19, "runwayFeasibility": 9},
                    "completedAt": "2026-01-01T00:00:00+00:00",
                },
                "c2": {
                    "eligible": True,
                    "validationStatus": "accepted",
                    "totalScore": 80,
                    "tieScores": {"silentVisualClarity": 12, "problemAdvantageIntegrity": 15, "runwayFeasibility": 8},
                    "completedAt": "2026-01-02T00:00:00+00:00",
                },
            }
        }
        self.assertEqual(select_global_winner(state), "c1")


class TestBuilder2TournamentEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card,forgot",
            "BUILDER2_TOURNAMENT_ATTEMPTS_PER_PROTOTYPE_PER_ROUND": "1",
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    def test_tournament_end_to_end_mocked(self) -> None:
        llm = TournamentMockLLM(
            score_by_prototype={"closest": 95, "winning_card": 80, "forgot": 60}
        )
        plan = run_builder2_tournament(
            job_id="job-tournament-e2e",
            product_name="ACE Product",
            product_description="A useful product.",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-e2e",
        )
        self.assertIn("videoPrompt", plan)
        self.assertIn("headlineText", plan)
        state = load_tournament_state("job-tournament-e2e")
        assert state is not None
        self.assertEqual(state["winnerCandidateId"], select_global_winner(state))
        self.assertEqual(state.get("completionReason"), "max_rounds_reached")
        self.assertIn("builder2_winner", llm.calls)

    @patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card",
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    def test_resume_does_not_repeat_completed_calls(self) -> None:
        llm = TournamentMockLLM(score_by_prototype={"closest": 90, "winning_card": 70})

        run_builder2_tournament(
            job_id="job-resume",
            product_name="ACE Product",
            product_description="A useful product.",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-resume",
        )
        first_count = len(llm.calls)
        run_builder2_tournament(
            job_id="job-resume",
            product_name="ACE Product",
            product_description="A useful product.",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-resume",
        )
        self.assertEqual(len(llm.calls), first_count)


class TestBuilder2TournamentWorkerIntegration(unittest.TestCase):
    PLAN = _winner_plan()

    @patch("engine.runway_video.load_tournament_state", return_value=None)
    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_TOURNAMENT_ENABLED": "true",
        },
        clear=False,
    )
    @patch("engine.runway_video.postprocess_video_headline", return_value="https://final/video.mp4")
    @patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy")
    @patch("engine.runway_video.record_ad_promise_generation_success")
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once")
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-x")
    @patch("engine.runway_video._create_text_to_video_task")
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    def test_runway_called_once_after_tournament_plan(
        self,
        _product_name,
        tournament_mock,
        _start_image,
        text_task_mock,
        image_task_mock,
        poll_mock,
        _sleep_mock,
        _promise,
        _copy,
        _post,
        _phase,
        _redis_name,
        _load_state,
    ) -> None:
        normalized = normalize_winner_plan_for_runway(
            self.PLAN,
            product_name="Product",
            product_description="A useful product.",
            content_language="en",
        )
        tournament_mock.return_value = normalized
        poll_mock.return_value = {"status": "SUCCEEDED", "output": ["https://runway/video.mp4"]}
        _generate_one_video_mvp_body("Product", "A useful product.", job_id="job-runway-once")
        tournament_mock.assert_called_once()
        image_task_mock.assert_called_once()
        text_task_mock.assert_not_called()
        _start_image.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
            "BUILDER1_PLANNING_PROFILE": "QUALITY",
            "BUILDER2_TOURNAMENT_ENABLED": "true",
        },
        clear=True,
    )
    def test_builder1_env_does_not_change_tournament_config(self) -> None:
        self.assertTrue(resolve_builder2_tournament_enabled())
        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
