"""
Builder2 tournament correction tests — one-round default, recovery, continuous event, metrics.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder2_creator import validate_creator_purity
from engine.builder2_judge import validate_judge_purity, validate_judge_response
from engine.builder2_tournament_config import (
    DEFAULT_ACTIVE_PROTOTYPE_IDS,
    DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS,
    Builder2TournamentConfigError,
    resolve_builder2_tournament_max_rounds,
)
from engine.builder2_tournament_contracts import Builder2TournamentError, compare_candidate_rankings
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_recovery import (
    acquire_job_lease,
    clear_job_queued,
    disable_memory_recovery,
    enable_memory_recovery,
    expire_job_lease,
    has_active_lease,
    is_job_queued,
    mark_job_queued,
    new_worker_token,
    register_recoverable_job,
    requeue_recoverable_job,
    remove_recoverable_job,
    scan_and_requeue_recoverable_jobs,
    set_memory_job_hash,
)
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    load_tournament_state,
    new_tournament_state,
    save_tournament_state,
)
from engine.builder2_winner_plan import (
    builder2_video_plan_struct_ok_for_runway,
    validate_and_normalize_builder2_winner_plan,
    validate_builder2_winner_plan,
)
from engine.runway_video import RunwayVideoMVPError, _generate_one_video_mvp_body
from engine.video_planning import build_runway_prompt_from_plan, validate_and_normalize_plan
from engine.video_start_image import build_ace_start_frame_image_prompt
from tests.test_builder2_tournament import (
    TournamentMockLLM,
    _candidate,
    _judgment,
    _strategy,
    _winner_plan,
)


def _run_one_round(
    job_id: str,
    *,
    prototypes: str = "closest,winning_card,forgot",
    max_rounds: str = "1",
    score_by_prototype: Dict[str, int] | None = None,
) -> tuple[TournamentMockLLM, Dict[str, Any]]:
    llm = TournamentMockLLM(score_by_prototype=score_by_prototype or {})
    with patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": prototypes,
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": max_rounds,
            "BUILDER2_TOURNAMENT_ATTEMPTS_PER_PROTOTYPE_PER_ROUND": "1",
        },
        clear=True,
    ):
        run_builder2_tournament(
            job_id=job_id,
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed=f"seed-{job_id}",
        )
    state = load_tournament_state(job_id)
    assert state is not None
    return llm, state


class TestOneRoundDefaults(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_default_max_rounds_is_one(self) -> None:
        self.assertEqual(DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS, 1)
        self.assertEqual(resolve_builder2_tournament_max_rounds(), 1)

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_MAX_ROUNDS": "-1"}, clear=True)
    def test_negative_max_rounds_fails(self) -> None:
        with self.assertRaises(Builder2TournamentConfigError):
            resolve_builder2_tournament_max_rounds()

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_MAX_ROUNDS": "abc"}, clear=True)
    def test_non_numeric_max_rounds_fails(self) -> None:
        with self.assertRaises(Builder2TournamentConfigError):
            resolve_builder2_tournament_max_rounds()


class TestOneRoundBehavior(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_store()
        disable_memory_recovery()

    def test_one_round_no_elimination(self) -> None:
        _, state = _run_one_round("job-one-round")
        self.assertEqual(state.get("completionReason"), "max_rounds_reached")
        self.assertEqual(len(state.get("eliminatedPrototypeIds") or []), 0)
        self.assertEqual(_count_completed_rounds(state), 1)

    def test_every_active_prototype_gets_attempt(self) -> None:
        _, state = _run_one_round("job-attempts")
        prototype_ids = {c.get("prototypeId") for c in state["candidates"].values()}
        self.assertEqual(prototype_ids, {"closest", "winning_card", "forgot"})

    def test_every_valid_candidate_judged(self) -> None:
        _, state = _run_one_round("job-judged")
        for cand in state["candidates"].values():
            if cand.get("validationStatus") == "accepted":
                self.assertTrue(cand.get("judgmentId"))
        self.assertEqual(len(state["judgments"]), 3)

    def test_no_second_round_created(self) -> None:
        _, state = _run_one_round("job-single-round")
        self.assertEqual(state.get("currentRound"), 1)
        self.assertEqual(_count_completed_rounds(state), 1)

    def test_global_winner_highest_eligible_score(self) -> None:
        _, state = _run_one_round(
            "job-winner",
            score_by_prototype={"closest": 95, "winning_card": 80, "forgot": 60},
        )
        winner_id = state["winnerCandidateId"]
        winner = state["candidates"][winner_id]
        self.assertEqual(winner["prototypeId"], "closest")
        self.assertEqual(winner_id, select_global_winner(state))

    def test_prototype_identity_does_not_affect_winning(self) -> None:
        state = {
            "candidates": {
                "cand-b": {
                    "eligible": True,
                    "validationStatus": "accepted",
                    "totalScore": 80,
                    "tieScores": {"silentVisualClarity": 12, "problemAdvantageIntegrity": 15, "runwayFeasibility": 8},
                    "completedAt": "2026-01-02T00:00:00+00:00",
                },
                "cand-a": {
                    "eligible": True,
                    "validationStatus": "accepted",
                    "totalScore": 80,
                    "tieScores": {"silentVisualClarity": 12, "problemAdvantageIntegrity": 15, "runwayFeasibility": 8},
                    "completedAt": "2026-01-02T00:00:00+00:00",
                },
            }
        }
        self.assertEqual(select_global_winner(state), "cand-a")

    def test_metrics_eight_calls_for_three_prototypes(self) -> None:
        _, state = _run_one_round("job-metrics-3")
        metrics = state["metrics"]
        self.assertEqual(metrics["strategyCalls"], 1)
        self.assertEqual(metrics["creatorCalls"], 3)
        self.assertEqual(metrics["judgeCalls"], 3)
        self.assertEqual(metrics["winnerDevelopmentCalls"], 1)
        self.assertEqual(metrics["totalReasoningCalls"], 8)

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"}, clear=True)
    def test_metrics_fourteen_calls_for_six_prototypes(self) -> None:
        llm = TournamentMockLLM(score_by_prototype={pid: 70 + i for i, pid in enumerate(DEFAULT_ACTIVE_PROTOTYPE_IDS)})
        run_builder2_tournament(
            job_id="job-metrics-6",
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-6",
        )
        state = load_tournament_state("job-metrics-6")
        assert state is not None
        self.assertEqual(state["metrics"]["totalReasoningCalls"], 14)

    def test_only_winner_development_call_once(self) -> None:
        llm, _ = _run_one_round("job-winner-dev")
        self.assertEqual(llm.calls.count("builder2_winner"), 1)

    def test_runway_roles_not_called_before_winner(self) -> None:
        llm, _ = _run_one_round("job-order")
        winner_idx = llm.calls.index("builder2_winner")
        self.assertEqual(set(llm.calls[:winner_idx]), {"builder2_strategy", "builder2_creator", "builder2_judge"})


class TestPurityOneRound(unittest.TestCase):
    def test_creator_purity_rejects_other_candidate(self) -> None:
        cand = _candidate("closest")
        cand["conceptSummary"] = "Better than the previous candidate."
        with self.assertRaises(Builder2TournamentError):
            validate_creator_purity(cand)

    def test_creator_purity_rejects_judge_scores(self) -> None:
        cand = _candidate("closest")
        cand["conceptSummary"] = "Judge score was too low last time."
        with self.assertRaises(Builder2TournamentError):
            validate_creator_purity(cand)

    def test_judge_purity_rejects_unseen_comparison(self) -> None:
        bad = _judgment("c1")
        bad["verdict"] = "Compared to other candidates this is weaker."
        with self.assertRaises(Builder2TournamentError):
            validate_judge_response(bad, candidate_id="c1")

    def test_judge_purity_rejects_redesign(self) -> None:
        bad = _judgment("c1")
        bad["verdict"] = "We should redesign the candidate entirely."
        with self.assertRaises(Builder2TournamentError):
            validate_judge_purity(bad)

    def test_judge_purity_rejects_invented_creator_reasoning(self) -> None:
        bad = _judgment("c1")
        bad["verdict"] = "The creator probably meant a different mechanism."
        with self.assertRaises(Builder2TournamentError):
            validate_judge_purity(bad)

    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
        clear=True,
    )
    @patch("engine.builder2_creator.validate_creator_purity", wraps=validate_creator_purity)
    @patch("engine.builder2_judge.validate_judge_purity", wraps=validate_judge_purity)
    def test_purity_checks_run_in_one_round_mode(self, judge_mock, creator_mock) -> None:
        llm = TournamentMockLLM(score_by_prototype={"closest": 90})
        run_builder2_tournament(
            job_id="job-purity-round",
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-purity",
        )
        self.assertGreaterEqual(creator_mock.call_count, 1)
        self.assertGreaterEqual(judge_mock.call_count, 1)


class TestAlternativeRoundSettings(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_max_rounds_zero_eliminates_until_one_remains(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card",
                "BUILDER2_TOURNAMENT_MAX_ROUNDS": "0",
                "BUILDER2_TOURNAMENT_ELIMINATIONS_PER_ROUND": "1",
            },
            clear=True,
        ):
            llm = TournamentMockLLM(score_by_prototype={"closest": 95, "winning_card": 70})
            run_builder2_tournament(
                job_id="job-multi",
                product_name="ACE Product",
                product_description="desc",
                content_language="en",
                llm_client=llm,
                rng_seed="seed-multi",
            )
        state = load_tournament_state("job-multi")
        assert state is not None
        self.assertEqual(len(state.get("activePrototypeIds") or []), 1)
        self.assertGreater(len(state.get("eliminatedPrototypeIds") or []), 0)

    def test_max_rounds_two_eliminates_only_before_final_round(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card,forgot",
                "BUILDER2_TOURNAMENT_MAX_ROUNDS": "2",
                "BUILDER2_TOURNAMENT_ELIMINATIONS_PER_ROUND": "1",
            },
            clear=True,
        ):
            llm = TournamentMockLLM(
                score_by_prototype={"closest": 95, "winning_card": 85, "forgot": 60}
            )
            run_builder2_tournament(
                job_id="job-two-rounds",
                product_name="ACE Product",
                product_description="desc",
                content_language="en",
                llm_client=llm,
                rng_seed="seed-two",
            )
        state = load_tournament_state("job-two-rounds")
        assert state is not None
        self.assertEqual(state.get("completionReason"), "max_rounds_reached")
        self.assertEqual(_count_completed_rounds(state), 2)
        self.assertEqual(len(state.get("eliminatedPrototypeIds") or []), 1)


class TestContinuousEvent(unittest.TestCase):
    def test_continuous_event_validation(self) -> None:
        plan = _winner_plan()
        plan["structureType"] = "continuous_event"
        plan["sceneVariations"] = []
        validated = validate_builder2_winner_plan(plan)
        self.assertEqual(validated.get("sceneSequenceSemantics"), "temporal_beats")

    def test_continuous_event_does_not_require_montage_variations(self) -> None:
        plan = _winner_plan()
        plan["structureType"] = "continuous_event"
        plan["sceneVariations"] = []
        validated = validate_builder2_winner_plan(plan)
        self.assertEqual(validated["sceneVariations"], [])

    def test_continuous_event_preserves_sequence(self) -> None:
        plan = _winner_plan()
        normalized = validate_and_normalize_builder2_winner_plan(
            plan,
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        seq = normalized["sequence"]
        self.assertIn("beginning", seq)
        self.assertIn("development", seq)
        self.assertIn("resolution", seq)

    def test_continuous_runway_prompt_has_no_montage_language(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _winner_plan(),
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        prompt = build_runway_prompt_from_plan(plan)
        self.assertIn("one continuous", prompt.lower())
        event_part = prompt.lower().split("continuous event (follow exactly):", 1)[-1]
        self.assertNotRegex(event_part, r"\bmontage of\b|\bmultiple clips\b|\bquick cuts\b")

    def test_continuous_start_image_uses_beginning(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _winner_plan(),
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        prompt = build_ace_start_frame_image_prompt(plan)
        self.assertIn("continuous event", prompt.lower())
        self.assertIn("Two people with visible space", prompt)

    def test_continuous_downstream_fields_present(self) -> None:
        plan = validate_and_normalize_builder2_winner_plan(
            _winner_plan(),
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        ok, reason = builder2_video_plan_struct_ok_for_runway(plan)
        self.assertTrue(ok, reason)
        for key in (
            "productNameResolved",
            "headlineText",
            "headlineCoreKeyword",
            "coreVisualIdea",
            "sceneConcept",
            "videoPromptCore",
            "openingFrameDescription",
            "structureType",
        ):
            self.assertTrue(str(plan.get(key) or "").strip(), key)

    def test_variation_montage_still_passes(self) -> None:
        plan = _winner_plan()
        plan["structureType"] = "variation_montage"
        plan["sceneVariations"] = [
            "Two people stand apart.",
            "One step closes the distance.",
            "They meet in a clear embrace.",
        ]
        validated = validate_builder2_winner_plan(plan)
        self.assertEqual(validated.get("sceneSequenceSemantics"), "montage_variations")

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_ENABLED": "false"}, clear=True)
    def test_legacy_planner_unchanged_when_tournament_disabled(self) -> None:
        legacy = {
            "productNameResolved": "ACE Product",
            "headline": "Closer than you think",
            "headlineCoreKeyword": "Closer",
            "coreVisualIdea": "closing distance",
            "sceneVariations": ["Beat one.", "Beat two.", "Beat three."],
            "videoPrompt": "Three related beats in seven seconds.",
            "language": "en",
        }
        normalized, reason = validate_and_normalize_plan(
            legacy,
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
        )
        self.assertIsNotNone(normalized, reason)
        self.assertNotIn("builder2_tournament_winner_v1", normalized.get("planInferenceMode", ""))


class TestRecovery(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_store()
        disable_memory_recovery()

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_ENABLED": "true"}, clear=True)
    def test_recoverable_job_requeued_once(self) -> None:
        job_id = "job-recover"
        set_memory_job_hash(job_id, {"status": "interrupted", "error": "worker_shutdown_during_job"})
        register_recoverable_job(job_id)
        save_tournament_state(
            job_id,
            new_tournament_state(
                job_id=job_id,
                language="en",
                active_prototype_ids=["closest"],
                random_seed="seed",
            ),
        )
        self.assertTrue(requeue_recoverable_job(job_id))
        self.assertTrue(is_job_queued(job_id))
        self.assertFalse(requeue_recoverable_job(job_id))

    def test_same_job_id_preserved_on_requeue(self) -> None:
        job_id = "job-same-id"
        set_memory_job_hash(job_id, {"status": "interrupted"})
        register_recoverable_job(job_id)
        save_tournament_state(
            job_id,
            new_tournament_state(
                job_id=job_id,
                language="en",
                active_prototype_ids=["closest"],
                random_seed="preserve-seed",
            ),
        )
        requeue_recoverable_job(job_id)
        self.assertTrue(is_job_queued(job_id))

    def test_random_seed_preserved_in_state(self) -> None:
        job_id = "job-seed"
        state = new_tournament_state(
            job_id=job_id,
            language="en",
            active_prototype_ids=["closest"],
            random_seed="fixed-seed-123",
        )
        save_tournament_state(job_id, state)
        loaded = load_tournament_state(job_id)
        assert loaded is not None
        self.assertEqual(loaded["randomSeed"], "fixed-seed-123")

    def test_round_deck_preserved_after_save(self) -> None:
        job_id = "job-deck"
        llm, state = _run_one_round("job-deck", prototypes="closest,winning_card")
        deck = state["rounds"][0]["shuffledPrototypeOrder"]
        loaded = load_tournament_state(job_id)
        assert loaded is not None
        self.assertEqual(loaded["rounds"][0]["shuffledPrototypeOrder"], deck)

    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
        clear=True,
    )
    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
        clear=True,
    )
    def test_resume_skips_completed_strategy_and_partial_candidates(self) -> None:
        llm = TournamentMockLLM(score_by_prototype={"closest": 90, "winning_card": 70})
        state = new_tournament_state(
            job_id="job-partial-resume",
            language="en",
            active_prototype_ids=["closest", "winning_card"],
            random_seed="seed-partial",
        )
        state["strategyFoundation"] = _strategy()
        state["status"] = "round_generating"
        state["lastCompletedStep"] = "strategy_complete"
        state["currentRound"] = 1
        state["rounds"] = [
            {
                "roundIndex": 1,
                "shuffledPrototypeOrder": ["closest", "winning_card"],
                "attemptsRequested": 1,
                "attemptsCompleted": 0,
                "judgmentsCompleted": 0,
                "bestCandidateByPrototype": {},
                "eliminatedPrototypeId": None,
                "eliminationReason": None,
            }
        ]
        candidate_id = "cand-closest-1-1"
        state["candidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": "closest",
            "roundIndex": 1,
            "attemptNumber": 1,
            "creatorOutput": _candidate("closest"),
            "validationStatus": "accepted",
            "judgmentId": "jud-1",
            "eligible": True,
            "totalScore": 90,
            "tieScores": {"silentVisualClarity": 12, "problemAdvantageIntegrity": 15, "runwayFeasibility": 8},
            "completedAt": "2026-01-01T00:00:00+00:00",
        }
        state["judgments"]["jud-1"] = {
            "judgmentId": "jud-1",
            "candidateId": candidate_id,
            "judgment": _judgment(candidate_id, total_hint=90),
            "totalScore": 90,
            "scores": _judgment(candidate_id)["scores"],
            "eligible": True,
            "completedAt": "2026-01-01T00:00:00+00:00",
        }
        state["bestCandidateByPrototype"]["closest"] = candidate_id
        save_tournament_state("job-partial-resume", state)
        run_builder2_tournament(
            job_id="job-partial-resume",
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-partial",
        )
        self.assertNotIn("builder2_strategy", llm.calls)
        self.assertIn("builder2_creator", llm.calls)
        self.assertIn("builder2_judge", llm.calls)

    def test_valid_lease_prevents_duplicate_workers(self) -> None:
        job_id = "job-lease"
        token_a = new_worker_token()
        token_b = new_worker_token()
        self.assertTrue(acquire_job_lease(job_id, token_a))
        self.assertFalse(acquire_job_lease(job_id, token_b))
        self.assertTrue(has_active_lease(job_id))

    def test_expired_lease_permits_recovery(self) -> None:
        job_id = "job-expired"
        token = new_worker_token()
        with patch("engine.builder2_tournament_recovery._lease_seconds", return_value=1):
            self.assertTrue(acquire_job_lease(job_id, token))
            expire_job_lease(job_id)
            self.assertFalse(has_active_lease(job_id))
            self.assertTrue(acquire_job_lease(job_id, new_worker_token()))

    def test_completed_job_removed_from_recoverable(self) -> None:
        job_id = "job-done"
        register_recoverable_job(job_id)
        remove_recoverable_job(job_id)
        set_memory_job_hash(job_id, {"status": "done"})
        save_tournament_state(
            job_id,
            new_tournament_state(
                job_id=job_id,
                language="en",
                active_prototype_ids=["closest"],
                random_seed="seed",
            ),
        )
        self.assertFalse(requeue_recoverable_job(job_id))

    def test_permanently_failed_job_not_recovered(self) -> None:
        job_id = "job-failed"
        register_recoverable_job(job_id)
        set_memory_job_hash(job_id, {"status": "error", "error": "permanent_failure"})
        save_tournament_state(
            job_id,
            new_tournament_state(
                job_id=job_id,
                language="en",
                active_prototype_ids=["closest"],
                random_seed="seed",
            ),
        )
        self.assertFalse(requeue_recoverable_job(job_id))

    @patch.dict(os.environ, {"BUILDER2_TOURNAMENT_ENABLED": "true"}, clear=True)
    def test_startup_scan_requeues_recoverable_jobs(self) -> None:
        job_id = "job-scan"
        set_memory_job_hash(job_id, {"status": "interrupted"})
        register_recoverable_job(job_id)
        save_tournament_state(
            job_id,
            new_tournament_state(
                job_id=job_id,
                language="en",
                active_prototype_ids=["closest"],
                random_seed="seed",
            ),
        )
        requeued = scan_and_requeue_recoverable_jobs()
        self.assertIn(job_id, requeued)


class TestRunwayRecovery(unittest.TestCase):
    PLAN = validate_and_normalize_builder2_winner_plan(
        _winner_plan(),
        product_name="Product",
        product_description="desc",
        content_language="en",
    )

    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "BUILDER2_TOURNAMENT_ENABLED": "true"},
        clear=False,
    )
    @patch("engine.runway_video.postprocess_video_headline", return_value="https://final/video.mp4")
    @patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy")
    @patch("engine.runway_video.record_ad_promise_generation_success")
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once")
    @patch("engine.runway_video._create_image_to_video_task")
    @patch("engine.runway_video._create_text_to_video_task")
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    def test_stored_runway_task_id_prevents_duplicate_submission(
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
    ) -> None:
        tournament_mock.return_value = self.PLAN
        poll_mock.return_value = {"status": "SUCCEEDED", "output": ["https://runway/video.mp4"]}
        job_id = "job-runway-resume"
        state = new_tournament_state(
            job_id=job_id,
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        state["runway"] = {"taskId": "task-existing", "submissionState": "submitted", "startImageCompleted": True}
        save_tournament_state(job_id, state)
        _generate_one_video_mvp_body("Product", "desc", job_id=job_id)
        image_task_mock.assert_not_called()
        text_task_mock.assert_not_called()

    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    @patch("engine.runway_video.video_job_set_phase")
    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch.dict(
        os.environ,
        {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "BUILDER2_TOURNAMENT_ENABLED": "true"},
        clear=False,
    )
    def test_ambiguous_runway_submission_raises(
        self,
        _redis_name,
        _phase,
        _product_name,
        tournament_mock,
    ) -> None:
        tournament_mock.return_value = self.PLAN
        job_id = "job-runway-ambiguous"
        state = new_tournament_state(
            job_id=job_id,
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )
        state["runway"] = {
            "taskId": None,
            "submissionState": "pending",
            "startImageCompleted": True,
            "startImageDataUri": "data:image/png;base64,x",
        }
        save_tournament_state(job_id, state)
        with self.assertRaises(RunwayVideoMVPError) as ctx:
            _generate_one_video_mvp_body("Product", "desc", job_id=job_id)
        self.assertEqual(ctx.exception.args[0], "builder2_runway_resume_ambiguous")


class TestIsolation(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_store()
        disable_memory_recovery()

    def test_two_jobs_have_separate_tournament_state(self) -> None:
        _, state_a = _run_one_round("job-a", prototypes="closest")
        _, state_b = _run_one_round("job-b", prototypes="winning_card")
        self.assertNotEqual(state_a["tournamentId"], state_b["tournamentId"])
        self.assertNotEqual(state_a["randomSeed"], state_b["randomSeed"])

    @patch.dict(
        os.environ,
        {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "99"},
        clear=True,
    )
    def test_builder1_env_does_not_change_builder2_defaults(self) -> None:
        from engine.builder2_tournament_config import resolve_builder2_tournament_enabled

        self.assertTrue(resolve_builder2_tournament_enabled())
        self.assertEqual(resolve_builder2_tournament_max_rounds(), 99)

    def test_builder2_recovery_keys_are_builder2_specific(self) -> None:
        from engine.builder2_tournament_recovery import LEASE_KEY_PREFIX, QUEUED_KEY_PREFIX, RECOVERABLE_JOBS_KEY

        self.assertTrue(RECOVERABLE_JOBS_KEY.startswith("ace:builder2:"))
        self.assertTrue(QUEUED_KEY_PREFIX.startswith("ace:builder2:"))
        self.assertTrue(LEASE_KEY_PREFIX.startswith("ace:builder2:"))


class TestOneRoundEndToEndPipeline(unittest.TestCase):
    PLAN = validate_and_normalize_builder2_winner_plan(
        _winner_plan(),
        product_name="Product",
        product_description="desc",
        content_language="en",
    )

    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_TOURNAMENT_ENABLED": "true",
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card,forgot",
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    @patch("engine.runway_video.postprocess_video_headline", return_value="https://final/video.mp4")
    @patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy")
    @patch("engine.runway_video.record_ad_promise_generation_success")
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once")
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-e2e")
    @patch("engine.runway_video._create_text_to_video_task")
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    def test_one_round_through_runway(
        self,
        _product_name,
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
    ) -> None:
        poll_mock.return_value = {"status": "SUCCEEDED", "output": ["https://runway/video.mp4"]}
        llm = TournamentMockLLM(score_by_prototype={"closest": 95, "winning_card": 80, "forgot": 60})

        def _run_tournament(**kwargs: Any) -> Dict[str, Any]:
            kwargs["llm_client"] = llm
            return run_builder2_tournament(**kwargs)

        with patch("engine.runway_video.run_builder2_tournament", side_effect=_run_tournament):
            _generate_one_video_mvp_body("Product", "desc", job_id="job-e2e-one-round")
        state = load_tournament_state("job-e2e-one-round")
        assert state is not None
        self.assertEqual(state.get("completionReason"), "max_rounds_reached")
        self.assertEqual(state["metrics"]["totalReasoningCalls"], 8)
        image_task_mock.assert_called_once()
        text_task_mock.assert_not_called()
        stored = load_tournament_state("job-e2e-one-round")
        assert stored is not None
        self.assertEqual((stored.get("runway") or {}).get("taskId"), "task-e2e")


class TestRestartRecoveryFlow(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_store()
        disable_memory_recovery()

    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest,winning_card", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
        clear=True,
    )
    def test_restart_resumes_without_duplicate_strategy(self) -> None:
        llm = TournamentMockLLM(score_by_prototype={"closest": 90, "winning_card": 70})
        run_builder2_tournament(
            job_id="job-restart",
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-restart",
        )
        state = load_tournament_state("job-restart")
        assert state is not None
        state = deepcopy(state)
        state["winnerDevelopmentPlan"] = None
        state["winnerCandidateId"] = None
        state["status"] = "round_complete"
        state["lastCompletedStep"] = "round_1_complete"
        save_tournament_state("job-restart", state)
        set_memory_job_hash("job-restart", {"status": "interrupted", "error": "worker_shutdown_during_job"})
        register_recoverable_job("job-restart")
        requeue_recoverable_job("job-restart")
        llm.calls.clear()
        run_builder2_tournament(
            job_id="job-restart",
            product_name="ACE Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-restart",
        )
        self.assertNotIn("builder2_strategy", llm.calls)
        self.assertNotIn("builder2_creator", llm.calls)
        self.assertNotIn("builder2_judge", llm.calls)
        self.assertIn("builder2_winner", llm.calls)


def _count_completed_rounds(state: Dict[str, Any]) -> int:
    return len([r for r in state.get("rounds", []) if r.get("roundComplete")])


if __name__ == "__main__":
    unittest.main()
