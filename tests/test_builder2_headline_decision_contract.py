"""
Builder2 headline decision contract tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_headline_decision_contract import (
    capture_headline_decision_diagnostic,
    normalize_headline_decision_object,
    validate_headline_decision_methodology,
)
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from engine.builder2_winner_development import develop_builder2_winning_candidate
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    load_revalidatable_parsed_winner_response,
    persist_parsed_winner_response,
    process_winner_development_response,
)
from engine.builder2_winner_revalidate import run_one_winner_revalidate
from engine.builder2_winner_resume import run_one_winner_resume
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_reasoning_resume import (
    _candidate_id_for_prototype,
    _judgment_for_candidate,
    _make_llm,
)
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt
from tests.test_builder2_winner_resume import _historical_judged_state


def _winner_plan(*, reason: Any = None, decision: str = "omit", headline: str = "") -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("summer_fan")
    candidate["verbalPotential"] = {
        "decision": "not_needed",
        "reason": "The visible fan behavior communicates absence without a headline.",
    }
    plan = _winner_plan_from_prompt("")
    plan.update(methodology_winner_extras(headline_decision=decision, winning_candidate=candidate, strategy=strategy))
    for key in ("prototypeId", "structureType", "visualParallelType", "coreCreativeMechanism"):
        if candidate.get(key) is not None:
            plan[key] = candidate[key]
    if isinstance(plan.get("preservationReference"), dict):
        plan["preservationReference"].update(
            {
                "prototypeId": candidate.get("prototypeId"),
                "structureType": candidate.get("structureType"),
                "visualParallelType": candidate.get("visualParallelType"),
                "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
            }
        )
    headline_obj: Dict[str, Any] = {"decision": decision}
    if reason is not None:
        headline_obj["reason"] = reason
    plan["headlineDecision"] = headline_obj
    plan["headline"] = headline
    plan["headlineCoreKeyword"] = headline
    if decision == "omit":
        plan["headlineForm"] = "none"
        plan["headline"] = ""
        plan["headlineCoreKeyword"] = ""
    return plan


def _judgment_for_summer_fan(*, headline_needed: bool = False) -> Dict[str, Any]:
    candidate_id = _candidate_id_for_prototype("summer_fan")
    candidate = _candidate("summer_fan")
    candidate["verbalPotential"] = {
        "decision": "not_needed",
        "reason": "The visible fan behavior communicates absence without a headline.",
    }
    judgment = _judgment_for_candidate(candidate_id, candidate, eligible=True)
    judgment.update(methodology_judgment_extras())
    judgment["headlineNecessityAssessment"] = {
        "headlineNeeded": headline_needed,
        "visualWouldWorkWithoutHeadline": not headline_needed,
        "notes": "Assessment for headline contract tests.",
    }
    return judgment


def _process_plan(plan: Dict[str, Any], *, judgment: Dict[str, Any] | None = None) -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("summer_fan")
    source = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=candidate,
        candidate_id=_candidate_id_for_prototype("summer_fan"),
    )
    return process_winner_development_response(
        plan,
        source_reference=source,
        winning_candidate=candidate,
        winning_judgment=judgment or _judgment_for_summer_fan(),
    )


class TestOptionalHeadlineReason(unittest.TestCase):
    def test_omit_missing_reason_is_valid(self) -> None:
        plan = _winner_plan()
        plan["headlineDecision"] = {"decision": "omit"}
        result = _process_plan(plan)
        self.assertEqual(result["headlineDecision"]["decision"], "omit")
        self.assertIsNone(result["headlineDecision"]["reason"])

    def test_omit_null_reason_is_valid(self) -> None:
        result = _process_plan(_winner_plan(reason=None))
        self.assertIsNone(result["headlineDecision"]["reason"])

    def test_omit_empty_reason_is_valid(self) -> None:
        result = _process_plan(_winner_plan(reason=""))
        self.assertIsNone(result["headlineDecision"]["reason"])

    def test_omit_with_reason_string_is_valid(self) -> None:
        result = _process_plan(_winner_plan(reason="Visual proof is sufficient."))
        self.assertEqual(result["headlineDecision"]["reasonSource"], "model")

    def test_use_with_valid_headline_and_missing_reason_is_valid(self) -> None:
        plan = _winner_plan(decision="include", headline="Cooler together")
        plan["headlineDecision"] = {"decision": "include"}
        result = _process_plan(plan)
        self.assertEqual(result["headlineDecision"]["decision"], "use")
        self.assertIsNone(result["headlineDecision"]["reason"])

    def test_use_without_headline_fails(self) -> None:
        plan = _winner_plan(decision="include", headline="")
        plan["headlineDecision"] = {"decision": "include"}
        with self.assertRaises(Builder2TournamentError) as ctx:
            _process_plan(plan)
        self.assertIn("headline", str(ctx.exception))

    def test_invalid_decision_enum_fails(self) -> None:
        plan = _winner_plan()
        plan["headlineDecision"] = {"decision": "maybe"}
        with self.assertRaises(Builder2TournamentError) as ctx:
            _process_plan(plan)
        self.assertIn("headlineDecision.decision", str(ctx.exception))

    def test_omit_with_stray_headline_text_fails_or_normalizes(self) -> None:
        plan = _winner_plan(reason=None)
        plan["headline"] = "Should not remain"
        result = _process_plan(plan)
        self.assertEqual(result.get("headline"), "")

    def test_omit_with_textual_dependency_fails(self) -> None:
        plan = _winner_plan(reason=None)
        plan["videoPrompt"] = "The viewer must read the headline text to understand the scene."
        with self.assertRaises(Builder2TournamentError) as ctx:
            _process_plan(plan)
        self.assertIn("textual_dependency", str(ctx.exception))

    def test_omit_contradicts_judge_when_headline_required(self) -> None:
        plan = _winner_plan(reason=None)
        judgment = _judgment_for_summer_fan(headline_needed=True)
        with self.assertRaises(Builder2TournamentError) as ctx:
            _process_plan(plan, judgment=judgment)
        self.assertIn("omit_contradicts_judge", str(ctx.exception))

    def test_judge_supports_omit_without_copying_prose(self) -> None:
        plan = _winner_plan(reason=None)
        judgment = _judgment_for_summer_fan(headline_needed=False)
        result = _process_plan(plan, judgment=judgment)
        self.assertEqual(result["headlineDecision"]["reasonSource"], "judge")
        self.assertIsNone(result["headlineDecision"]["reason"])
        notes = (judgment.get("headlineNecessityAssessment") or {}).get("notes")
        self.assertNotIn(str(notes or ""), str(result.get("headlineDecision")))


class TestNormalizationAndDiagnostics(unittest.TestCase):
    def test_include_alias_normalizes_to_use(self) -> None:
        normalized = normalize_headline_decision_object({"decision": "include", "reason": None})
        self.assertEqual(normalized["decision"], "use")

    def test_missing_reason_sets_not_required(self) -> None:
        normalized = normalize_headline_decision_object({"decision": "omit"})
        self.assertEqual(normalized["reasonSource"], "not_required")

    def test_diagnostic_reports_reason_absence(self) -> None:
        diagnostic = capture_headline_decision_diagnostic({"decision": "omit"})
        self.assertFalse(diagnostic["reasonPresent"])


class TestPersistenceRevalidationAndResume(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_parsed_response_persisted_before_methodology_validation(self) -> None:
        state = _historical_judged_state(job_id="job-headline-persist")
        plan = _winner_plan(reason=None)
        persist_parsed_winner_response(
            state,
            parsed=plan,
            candidate_id=_candidate_id_for_prototype("summer_fan"),
            prototype_id="summer_fan",
        )
        payload = load_revalidatable_parsed_winner_response(state)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["topLevelKeyCount"], len(plan))

    def test_offline_revalidation_zero_calls(self) -> None:
        state = _historical_judged_state(job_id="job-headline-revalidate")
        plan = _winner_plan(reason=None)
        persist_parsed_winner_response(
            state,
            parsed=plan,
            candidate_id=_candidate_id_for_prototype("summer_fan"),
            prototype_id="summer_fan",
            top_level_keys=sorted(plan.keys()),
        )
        report = run_one_winner_revalidate(job_id="job-headline-revalidate", tournament_state=deepcopy(state))
        self.assertTrue(report["ok"])
        self.assertTrue(report["winnerRevalidated"])
        self.assertEqual(report["winnerNormalCalls"], 0)
        self.assertEqual(report["headlineDecision"], "omit")
        self.assertFalse(report["headlineReasonPresent"])

    def test_offline_revalidation_persists_winner(self) -> None:
        state = _historical_judged_state(job_id="job-headline-revalidate-persist")
        plan = _winner_plan(reason=None)
        persist_parsed_winner_response(
            state,
            parsed=plan,
            candidate_id=_candidate_id_for_prototype("summer_fan"),
            prototype_id="summer_fan",
        )
        report = run_one_winner_revalidate(job_id="job-headline-revalidate-persist", tournament_state=deepcopy(state))
        self.assertTrue(report["winnerDevelopmentAccepted"])
        saved = load_tournament_state("job-headline-revalidate-persist")
        assert saved is not None
        self.assertTrue(saved.get("winnerDevelopmentAccepted"))

    def test_persisted_winner_reused_with_zero_calls(self) -> None:
        state = _historical_judged_state(job_id="job-headline-reuse")
        plan = _winner_plan(reason=None)
        persist_parsed_winner_response(
            state,
            parsed=plan,
            candidate_id=_candidate_id_for_prototype("summer_fan"),
            prototype_id="summer_fan",
        )
        first = run_one_winner_revalidate(job_id="job-headline-reuse", tournament_state=deepcopy(state))
        self.assertTrue(first["ok"])
        second = run_one_winner_revalidate(
            job_id="job-headline-reuse",
            tournament_state=load_tournament_state("job-headline-reuse"),
        )
        self.assertTrue(second["winnerReused"])
        self.assertEqual(second["winnerNormalCalls"], 0)

    def test_winner_resume_fallback_one_call(self) -> None:
        state = _historical_judged_state(job_id="job-headline-resume")
        report = run_one_winner_resume(job_id="job-headline-resume", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["winnerNormalCalls"], 1)
        self.assertEqual(report["strategyCalls"], 0)
        self.assertEqual(report["creatorCalls"], 0)
        self.assertEqual(report["judgeCalls"], 0)

    def test_no_media_calls_on_resume(self) -> None:
        state = _historical_judged_state(job_id="job-headline-media")
        report = run_one_winner_resume(job_id="job-headline-media", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["startImageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)

    def test_six_judgments_not_regenerated(self) -> None:
        state = _historical_judged_state(job_id="job-headline-judgments")
        report = run_one_winner_resume(job_id="job-headline-judgments", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["reusedJudgmentCount"], 6)
        self.assertEqual(report["judgeCalls"], 0)


class TestPreservationAndRegression(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_model_preservation_null_remains_non_authoritative(self) -> None:
        plan = _winner_plan(reason=None)
        plan["winnerPreservationCheck"] = {"problemPreserved": None}
        result = _process_plan(plan)
        self.assertEqual(result.get("serverPreservationCheck", {}).get("source"), "server_owned_contract")

    def test_full_production_remains_14_reasoning_calls(self) -> None:
        def llm(**kwargs: Any) -> Dict[str, Any]:
            role = kwargs.get("role", "")
            prompt = kwargs.get("prompt", "")
            if role == "builder2_strategy":
                return _strategy()
            if role == "builder2_creator":
                prototype_id = "closest"
                for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                    if pid in prompt:
                        prototype_id = pid
                        break
                return _candidate(prototype_id, prompt=prompt)
            if role == "builder2_judge":
                candidate_id = next(token.strip().strip(",") for token in prompt.split() if token.startswith("cand-"))
                prototype_id = "closest"
                for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                    if pid in prompt:
                        prototype_id = pid
                        break
                return _judgment_for_candidate(candidate_id, _candidate(prototype_id), eligible=True)
            if role == "builder2_winner":
                return _winner_plan_from_prompt(prompt)
            raise AssertionError(role)

        run_builder2_tournament(
            job_id="job-headline-regression-14",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-headline-regression",
        )
        state = load_tournament_state("job-headline-regression-14")
        assert state is not None
        self.assertEqual((state.get("metrics") or {}).get("totalReasoningCalls"), 14)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


class TestDevelopPersistenceHook(unittest.TestCase):
    def test_develop_persists_parsed_before_methodology_validation_failure(self) -> None:
        enable_memory_store()
        try:
            state = _historical_judged_state(job_id="job-headline-develop-persist")
            winner_id = _candidate_id_for_prototype("summer_fan")
            winner_rec = state["candidates"][winner_id]
            bad_plan = _winner_plan(reason=None)
            bad_plan["preservationReference"]["prototypeId"] = "think_small"
            with patch(
                "engine.builder2_winner_development.call_builder2_role_json_with_text",
                return_value=(bad_plan, "{}"),
            ):
                with self.assertRaises(Builder2TournamentError):
                    develop_builder2_winning_candidate(
                        product_name="Product",
                        product_description="desc",
                        language="he",
                        strategy_foundation=state["strategyFoundation"],
                        winning_candidate=winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {},
                        winning_judgment=_judgment_for_summer_fan(),
                        prototype_id="summer_fan",
                        runway_mode="standard",
                        llm_client=None,
                        state=state,
                        candidate_id=winner_id,
                    )
            self.assertIsNotNone(load_revalidatable_parsed_winner_response(state))
        finally:
            disable_memory_store()


if __name__ == "__main__":
    unittest.main()
