"""
Builder2 complete-ad resume inspect and stage resolution tests — mocks only.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_complete_ad_resume_plan import (
    plan_complete_ad_reasoning_roles,
    resolve_complete_ad_resume_stage,
)
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import _winner_plan_from_prompt


def _production_like_missing_greenpeace_state() -> Dict[str, Any]:
    state = _six_prototype_state(judged=5, creators=5)
    state["jobId"] = "d6425b71-c612-4fcd-a3cf-8c30db88ca52"
    state["provisionalWinnerCandidateId"] = "cand-1-forgot-1-test"
    state["winnerCandidateId"] = None
    state[PARSED_WINNER_RESPONSE_KEY] = {
        "parsed": _winner_plan_from_prompt(""),
        "candidateId": "cand-1-forgot-1-2ab7377c",
    }
    return state


class TestCompleteAdResumeStageResolution(unittest.TestCase):
    def test_five_creators_five_judges_missing_creator_resolves_creator_generation(self) -> None:
        state = _production_like_missing_greenpeace_state()
        self.assertEqual(resolve_complete_ad_resume_stage(state, read_only=True), "creator_generation")

    def test_six_creators_five_judges_resolves_judge_generation(self) -> None:
        state = _six_prototype_state(judged=5, creators=6)
        self.assertEqual(resolve_complete_ad_resume_stage(state, read_only=True), "judge_generation")

    def test_five_creators_cannot_resolve_judge_generation(self) -> None:
        state = _six_prototype_state(judged=5, creators=5)
        self.assertNotEqual(resolve_complete_ad_resume_stage(state, read_only=True), "judge_generation")

    def test_resolver_matches_complete_ad_stage_for_missing_creator(self) -> None:
        state = _production_like_missing_greenpeace_state()
        resolved = resolve_builder2_resume_stage({}, state, read_only=True)
        self.assertEqual(resolved.get("resumeFromStage"), "creator_generation")


class TestCompleteAdRolePlanning(unittest.TestCase):
    def test_missing_creator_requires_creator_and_judge_roles(self) -> None:
        plan = plan_complete_ad_reasoning_roles(_production_like_missing_greenpeace_state(), read_only=True)
        self.assertEqual(plan["requiredNextReasoningRoles"], ["builder2_creator", "builder2_judge"])

    def test_conditional_winner_role_is_separate(self) -> None:
        plan = plan_complete_ad_reasoning_roles(_production_like_missing_greenpeace_state(), read_only=True)
        self.assertEqual(plan["conditionalNextReasoningRoles"], ["builder2_winner"])
        self.assertEqual(
            plan["expectedNextReasoningRoles"],
            ["builder2_creator", "builder2_judge", "builder2_winner_if_winner_changes"],
        )

    def test_minimum_and_maximum_call_estimates(self) -> None:
        plan = plan_complete_ad_reasoning_roles(_production_like_missing_greenpeace_state(), read_only=True)
        self.assertEqual(plan["minimumAdditionalReasoningCalls"], 2)
        self.assertEqual(plan["maximumAdditionalReasoningCalls"], 3)

    def test_parsed_winner_does_not_remove_missing_judge(self) -> None:
        plan = plan_complete_ad_reasoning_roles(_production_like_missing_greenpeace_state(), read_only=True)
        self.assertIn("builder2_judge", plan["requiredNextReasoningRoles"])

    def test_six_creators_five_judges_requires_only_judge(self) -> None:
        plan = plan_complete_ad_reasoning_roles(_six_prototype_state(judged=5, creators=6), read_only=True)
        self.assertEqual(plan["requiredNextReasoningRoles"], ["builder2_judge"])
        self.assertEqual(plan["minimumAdditionalReasoningCalls"], 1)


class TestCompleteAdResumeInspectReadOnly(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    @patch("engine.builder2_tournament_store.save_tournament_state")
    @patch("engine.builder2_tournament_recovery.mark_job_queued")
    @patch("engine.builder2_execution_lease.acquire_job_lease", return_value=False)
    def test_inspector_zero_redis_writes(
        self,
        _lease: Any,
        _queue: Any,
        save_state: Any,
        _raw_job: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        original = _production_like_missing_greenpeace_state()
        read_raw.return_value = deepcopy(original)
        before = deepcopy(original)
        report = inspect_builder2_complete_ad_resume("d6425b71-c612-4fcd-a3cf-8c30db88ca52")
        self.assertTrue(report["ok"])
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["resolvedResumeStage"], "creator_generation")
        self.assertEqual(report["requiredNextReasoningRoles"], ["builder2_creator", "builder2_judge"])
        self.assertEqual(report["conditionalNextReasoningRoles"], ["builder2_winner"])
        self.assertEqual(report["minimumAdditionalReasoningCalls"], 2)
        self.assertEqual(report["maximumAdditionalReasoningCalls"], 3)
        save_state.assert_not_called()
        _queue.assert_not_called()
        _lease.assert_not_called()
        self.assertEqual(read_raw.return_value.get("provisionalWinnerCandidateId"), before["provisionalWinnerCandidateId"])

    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    def test_inspector_derives_judgment_index_without_backfill_log(
        self,
        _raw_job: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        state = _six_prototype_state(judged=5, creators=5)
        state["acceptedJudgments"] = {}
        for rec in state["candidates"].values():
            if rec.get("judgmentId"):
                rec["judgeStatus"] = "accepted"
        read_raw.return_value = deepcopy(state)
        with self.assertLogs("engine.builder2_tournament_completion_gate", level="INFO") as logs:
            report = inspect_builder2_complete_ad_resume("job-read-only-derive")
        self.assertTrue(report["ok"])
        joined = "\n".join(logs.output)
        self.assertIn("BUILDER2_ACCEPTED_JUDGMENT_INDEX_DERIVED_READ_ONLY", joined)
        self.assertNotIn("BUILDER2_ACCEPTED_JUDGMENT_INDEX_BACKFILLED", joined)

    def test_read_only_guard_blocks_save(self) -> None:
        from engine.builder2_tournament_contracts import TOURNAMENT_STATE_SCHEMA_VERSION

        state = _six_prototype_state(judged=6, creators=6)
        state["schemaVersion"] = TOURNAMENT_STATE_SCHEMA_VERSION
        save_tournament_state("job-read-only-block", state)
        with read_only_builder2_inspection() as counter:
            save_tournament_state("job-read-only-block", state)
            self.assertEqual(counter.redis_mutations, 0)


if __name__ == "__main__":
    unittest.main()
