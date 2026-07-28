"""
Builder2 Winner headline repair tests — mocks only.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume
from engine.builder2_headline_decision_contract import get_normalized_headline_decision
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_tournament_contracts import Builder2TournamentError, TOURNAMENT_STATE_SCHEMA_VERSION
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from engine.builder2_winner_headline_repair import (
    REPAIR_ALREADY_ATTEMPTED,
    assess_winner_headline_repair_eligibility,
    attempt_winner_headline_repair_after_offline_failure,
    classify_headline_only_offline_failure,
    merge_headline_repair_into_parsed_plan,
    parse_winner_headline_repair_partial,
    repair_builder2_winner_headline_from_parsed,
)
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
    offline_revalidate_parsed_winner_response,
)
from engine.builder2_winner_resume import run_one_winner_resume
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _record_repair_submitted(callback: Any) -> bool:
    if callable(callback):
        callback()
    return False


def _forgot_winner_id(state: Dict[str, Any]) -> str:
    for candidate_id, record in (state.get("candidates") or {}).items():
        if _clean(record.get("prototypeId")) == "forgot":
            return str(candidate_id)
    return "cand-1-forgot-1-test"


def _judgment_requiring_headline(candidate_id: str) -> Dict[str, Any]:
    judgment = _judgment(candidate_id, total_hint=88, eligible=True)
    judgment.update(methodology_judgment_extras())
    judgment["headlineNecessityAssessment"] = {
        "headlineNeeded": True,
        "visualWouldWorkWithoutHeadline": False,
        "headlineRecommended": True,
        "notes": "Headline required for this concept.",
    }
    return judgment


def _parsed_plan_missing_headline(*, candidate_id: str, prototype_id: str = "forgot") -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate(prototype_id)
    plan = _winner_plan_from_prompt("")
    plan.update(
        methodology_winner_extras(
            headline_decision="include",
            winning_candidate=candidate,
            strategy=strategy,
        )
    )
    plan["headlineDecision"] = {
        "decision": "use",
        "reason": "Headline complements the visual mechanism.",
        "reasonSource": "judge",
    }
    plan["headlineForm"] = "direct"
    plan["prototypeId"] = prototype_id
    plan["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
    for key in ("headline", "headlineText", "headlineCoreKeyword"):
        plan.pop(key, None)
    return plan


def _six_six_missing_headline_state(*, repair_calls: int = 0) -> Dict[str, Any]:
    state = _six_prototype_state(judged=6, creators=6)
    winner_id = _forgot_winner_id(state)
    state["jobId"] = "job-headline-repair"
    state["tournamentId"] = "tournament-headline-repair"
    state["builder2ResumeContractVersion"] = BUILDER2_RESUME_CONTRACT_VERSION
    state.pop("copyContractVersion", None)
    state["schemaVersion"] = TOURNAMENT_STATE_SCHEMA_VERSION
    state["productDescription"] = "Repair product"
    state["contentLanguage"] = "he"
    state["winnerCandidateId"] = winner_id
    state["winnerDevelopmentPaidCallRecorded"] = True
    for candidate_id, record in state["candidates"].items():
        if candidate_id == winner_id:
            record["totalScore"] = 88
        elif _clean(record.get("prototypeId")) == "greenpeace_essential_pairing":
            record["totalScore"] = 67
    judgment = _judgment_requiring_headline(winner_id)
    judgment_id = state["candidates"][winner_id]["judgmentId"]
    state["judgments"][judgment_id]["judgment"] = judgment
    parsed = _parsed_plan_missing_headline(candidate_id=winner_id)
    state[PARSED_WINNER_RESPONSE_KEY] = {
        "parsed": parsed,
        "candidateId": winner_id,
        "prototypeId": "forgot",
        "topLevelKeys": sorted(parsed.keys()),
    }
    ensure_metrics(state)
    state["metrics"]["winnerDevelopmentCalls"] = 1
    state["metrics"]["winnerNormalCalls"] = 1
    state["metrics"]["winnerRepairCalls"] = repair_calls
    state["metrics"]["winnerRetryCalls"] = 0
    return state


def _make_repair_llm(*, headline: str = "איכות שמדברת", keyword: str = "איכות") -> Any:
    def _llm(role: str, model: str, prompt: str) -> Dict[str, Any]:
        return {"headline": headline, "headlineCoreKeyword": keyword}

    return _llm


class TestWinnerHeadlineRepairEligibility(unittest.TestCase):
    def test_authorization_defaults_false(self) -> None:
        state = _six_six_missing_headline_state()
        eligibility = assess_winner_headline_repair_eligibility(
            state,
            winner_candidate_id=_forgot_winner_id(state),
            offline_failure_reason="builder2_tournament_invalid_field:headline",
            allow_repair=False,
            remaining_call_budget=1,
        )
        self.assertFalse(eligibility["eligible"])
        self.assertEqual(eligibility["reason"], "builder2_winner_headline_repair_ineligible:authorization_disabled")

    def test_classifies_headline_only_failures(self) -> None:
        self.assertEqual(classify_headline_only_offline_failure("builder2_tournament_invalid_field:headline"), "headline")
        self.assertIsNone(classify_headline_only_offline_failure("builder2_tournament_invalid_field:videoPrompt"))

    def test_use_decision_required(self) -> None:
        state = _six_six_missing_headline_state()
        parsed = state[PARSED_WINNER_RESPONSE_KEY]["parsed"]
        parsed["headlineDecision"] = {"decision": "omit", "reasonSource": "not_required"}
        eligibility = assess_winner_headline_repair_eligibility(
            state,
            winner_candidate_id=_forgot_winner_id(state),
            offline_failure_reason="builder2_tournament_invalid_field:headline",
            allow_repair=True,
            remaining_call_budget=1,
        )
        self.assertFalse(eligibility["eligible"])

    def test_judge_required_headline(self) -> None:
        state = _six_six_missing_headline_state()
        winner_id = _forgot_winner_id(state)
        judgment_id = state["candidates"][winner_id]["judgmentId"]
        state["judgments"][judgment_id]["judgment"]["headlineNecessityAssessment"] = {
            "headlineNeeded": False,
            "visualWouldWorkWithoutHeadline": True,
            "notes": "No headline needed.",
        }
        eligibility = assess_winner_headline_repair_eligibility(
            state,
            winner_candidate_id=winner_id,
            offline_failure_reason="builder2_tournament_invalid_field:headline",
            allow_repair=True,
            remaining_call_budget=1,
        )
        self.assertFalse(eligibility["eligible"])

    def test_repair_already_attempted(self) -> None:
        state = _six_six_missing_headline_state(repair_calls=1)
        eligibility = assess_winner_headline_repair_eligibility(
            state,
            winner_candidate_id=_forgot_winner_id(state),
            offline_failure_reason="builder2_tournament_invalid_field:headline",
            allow_repair=True,
            remaining_call_budget=1,
        )
        self.assertFalse(eligibility["eligible"])
        self.assertEqual(eligibility["reason"], REPAIR_ALREADY_ATTEMPTED)


class TestWinnerHeadlineRepairMergeAndValidation(unittest.TestCase):
    def test_partial_response_schema(self) -> None:
        parsed = parse_winner_headline_repair_partial({"headline": "Quality speaks", "headlineCoreKeyword": "Quality"})
        self.assertEqual(parsed["headline"], "Quality speaks")
        with self.assertRaises(Builder2TournamentError):
            parse_winner_headline_repair_partial({"headline": "x", "headlineCoreKeyword": "a b", "videoPrompt": "nope"})

    def test_merge_preserves_other_fields(self) -> None:
        original = _parsed_plan_missing_headline(candidate_id="cand-1-forgot-1-test")
        original["videoPrompt"] = "PRESERVE_ME"
        original["sequence"] = {"beginning": "A", "development": "B", "resolution": "C"}
        merged, _count = merge_headline_repair_into_parsed_plan(
            original,
            partial={"headline": "Quality speaks", "headlineCoreKeyword": "Quality"},
        )
        self.assertEqual(merged["videoPrompt"], "PRESERVE_ME")
        self.assertEqual(merged["sequence"], original["sequence"])
        self.assertEqual(merged["headlineForm"], "direct")
        self.assertEqual(get_normalized_headline_decision(merged), "use")
        self.assertNotIn("headlineText", merged)

    def test_offline_revalidation_still_fails_before_repair(self) -> None:
        state = _six_six_missing_headline_state()
        winner_id = _forgot_winner_id(state)
        candidate = state["candidates"][winner_id]["creatorOutput"]
        source = build_server_owned_winner_source_reference(
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            candidate_id=winner_id,
        )
        with self.assertRaises(Builder2TournamentError) as ctx:
            offline_revalidate_parsed_winner_response(
                state,
                source_reference=source,
                winning_candidate=candidate,
                winning_judgment=state["judgments"][state["candidates"][winner_id]["judgmentId"]]["judgment"],
            )
        self.assertIn("headline", str(ctx.exception))


class TestWinnerHeadlineRepairPaidCall(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_repair_records_repair_metrics(self) -> None:
        state = _six_six_missing_headline_state()
        winner_id = _forgot_winner_id(state)
        candidate = state["candidates"][winner_id]["creatorOutput"]
        judgment = state["judgments"][state["candidates"][winner_id]["judgmentId"]]["judgment"]
        repair_builder2_winner_headline_from_parsed(
            state,
            candidate_id=winner_id,
            prototype_id="forgot",
            product_name="ACE Product",
            language="he",
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            winning_judgment=judgment,
            validation_failures=["builder2_tournament_invalid_field:headline"],
            llm_client=_make_repair_llm(),
            job_id=state["jobId"],
            tournament_id=state["tournamentId"],
        )
        metrics = state["metrics"]
        self.assertEqual(metrics["winnerRepairCalls"], 1)
        self.assertEqual(metrics["winnerNormalCalls"], 1)
        self.assertEqual(metrics["winnerDevelopmentCalls"], 2)
        self.assertTrue(is_valid_persisted_winner_development(state))

    def test_headline_text_derived_canonically(self) -> None:
        state = _six_six_missing_headline_state()
        winner_id = _forgot_winner_id(state)
        candidate = state["candidates"][winner_id]["creatorOutput"]
        judgment = state["judgments"][state["candidates"][winner_id]["judgmentId"]]["judgment"]
        plan = repair_builder2_winner_headline_from_parsed(
            state,
            candidate_id=winner_id,
            prototype_id="forgot",
            product_name="ACE Product",
            language="he",
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            winning_judgment=judgment,
            validation_failures=["builder2_tournament_invalid_field:headline"],
            llm_client=_make_repair_llm(headline="איכות שמדברת", keyword="איכות"),
            job_id=state["jobId"],
            tournament_id=state["tournamentId"],
        )
        self.assertTrue(str(plan.get("headlineText") or "").strip())
        self.assertIn("איכות", str(plan.get("headlineText") or ""))

    def test_second_repair_not_eligible(self) -> None:
        state = _six_six_missing_headline_state(repair_calls=1)
        outcome = attempt_winner_headline_repair_after_offline_failure(
            state,
            job_id=state["jobId"],
            winner_candidate_id=_forgot_winner_id(state),
            prototype_id="forgot",
            product_name="ACE Product",
            language="he",
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=state["candidates"][_forgot_winner_id(state)]["creatorOutput"],
            winning_judgment=state["judgments"][state["candidates"][_forgot_winner_id(state)]["judgmentId"]]["judgment"],
            offline_failure_reason="builder2_tournament_invalid_field:headline",
            allow_repair=True,
            remaining_call_budget=1,
            llm_client=_make_repair_llm(),
        )
        self.assertFalse(outcome["accepted"])
        self.assertFalse(outcome["attempted"])


class TestControlledResumeHeadlineRepairIntegration(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_authorization_false_zero_winner_calls(self) -> None:
        state = _six_six_missing_headline_state()
        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate"
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate"
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate"
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state"
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
                max_calls=1,
                stop_before_media=True,
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["winnerCallsThisRun"], 0)
        creator_mock.assert_not_called()
        judge_mock.assert_not_called()
        winner_mock.assert_not_called()

    def test_authorized_repair_success(self) -> None:
        state = _six_six_missing_headline_state()
        working: Dict[str, Any] = {}
        with patch.dict(
            "os.environ",
            {"BUILDER2_COMPLETE_AD_REASONING_RESUME_ALLOW_WINNER_HEADLINE_REPAIR": "true"},
        ), patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease"
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate"
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate"
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate"
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
            side_effect=lambda job_id, tournament_state: working.update(deepcopy(tournament_state)),
        ), patch(
            "engine.builder2_winner_headline_repair.call_builder2_role_json_with_text",
            side_effect=lambda **kwargs: (
                _record_repair_submitted(kwargs.get("on_paid_request_submitted"))
                or (
                    {"headline": "איכות שמדברת", "headlineCoreKeyword": "איכות"},
                    '{"headline":"איכות שמדברת","headlineCoreKeyword":"איכות"}',
                )
            ),
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
                max_calls=1,
                stop_before_media=True,
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["winnerCallsThisRun"], 1)
        self.assertTrue(report.get("winnerHeadlineRepairAccepted"))
        self.assertEqual(report["creatorCallsThisRun"], 0)
        self.assertEqual(report["judgeCallsThisRun"], 0)
        creator_mock.assert_not_called()
        judge_mock.assert_not_called()
        winner_mock.assert_not_called()
        self.assertEqual(working["metrics"]["winnerRepairCalls"], 1)
        self.assertEqual(working["metrics"]["winnerNormalCalls"], 1)
        self.assertTrue(is_valid_persisted_winner_development(working))
        self.assertEqual(working["winnerDevelopmentCandidateId"], _forgot_winner_id(state))

    def test_repair_failure_no_normal_fallback(self) -> None:
        state = _six_six_missing_headline_state()
        with patch.dict(
            "os.environ",
            {"BUILDER2_COMPLETE_AD_REASONING_RESUME_ALLOW_WINNER_HEADLINE_REPAIR": "true"},
        ), patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease"
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate"
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state"
        ), patch(
            "engine.builder2_winner_headline_repair.call_builder2_role_json_with_text",
            side_effect=Builder2TournamentError("builder2_winner_headline_repair_invalid_response:extra_keys"),
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
                max_calls=1,
                stop_before_media=True,
            )
        self.assertFalse(report["ok"])
        self.assertTrue(report.get("winnerHeadlineRepairAttempted"))
        winner_mock.assert_not_called()

    def test_second_resume_zero_calls_after_repair(self) -> None:
        state = _six_six_missing_headline_state(repair_calls=1)
        state["failureStage"] = "winner_development"
        state["failureReason"] = "builder2_winner_headline_repair_failed:test"
        with patch.dict(
            "os.environ",
            {"BUILDER2_COMPLETE_AD_REASONING_RESUME_ALLOW_WINNER_HEADLINE_REPAIR": "true"},
        ), patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease"
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate"
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state"
        ), patch(
            "engine.builder2_winner_headline_repair.call_builder2_role_json_with_text",
        ) as repair_mock:
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
                max_calls=1,
                stop_before_media=True,
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["winnerCallsThisRun"], 0)
        repair_mock.assert_not_called()
        winner_mock.assert_not_called()


class TestWinnerResumeUnchanged(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_winner_resume_module_has_no_headline_repair_flag(self) -> None:
        import inspect
        import engine.builder2_winner_resume as mod

        source = inspect.getsource(mod)
        self.assertNotIn("ALLOW_WINNER_HEADLINE_REPAIR", source)
        self.assertNotIn("builder2_winner_headline_repair", source)

    def test_builder1_unchanged(self) -> None:
        import glob
        import os

        root = os.path.dirname(os.path.dirname(__file__))
        for path in glob.glob(os.path.join(root, "engine", "builder1*.py")):
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("winner_headline_repair", source)


if __name__ == "__main__":
    unittest.main()
