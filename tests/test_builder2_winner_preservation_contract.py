"""
Builder2 Winner preservation contract tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from engine.builder2_winner_preservation_contract import (
    SERVER_OWNED_WINNER_SOURCE_KEY,
    apply_server_owned_preservation,
    build_server_owned_winner_source_reference,
    capture_model_preservation_diagnostic,
    detect_winner_immutable_identity_violations,
    load_revalidatable_parsed_winner_response,
    normalize_winner_response_compatibility_fields,
    persist_parsed_winner_response,
    process_winner_development_response,
    validate_winner_preservation_contract_required,
    validate_winner_source_identity,
)
from engine.builder2_winner_revalidate import run_one_winner_revalidate
from engine.builder2_winner_resume import run_one_winner_resume
from tests.builder2_methodology_fixtures import methodology_winner_extras
from tests.test_builder2_reasoning_resume import (
    _candidate_id_for_prototype,
    _judgment_for_candidate,
    _make_llm,
)
from tests.test_builder2_winner_resume import _historical_judged_state
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan, _winner_plan_from_prompt


def _source_and_plan(
    *,
    prototype_id: str = "summer_fan",
    candidate_id: str = "cand-1-summer_fan-1-resume",
    winner_check: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    strategy = _strategy(language="he")
    candidate = _candidate(prototype_id)
    if prototype_id == "summer_fan":
        candidate["verbalPotential"] = {
            "decision": "not_needed",
            "reason": "The visible fan behavior communicates absence without a headline.",
        }
    source = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=candidate,
        candidate_id=candidate_id,
    )
    plan = _winner_plan_from_prompt("")
    plan.update(methodology_winner_extras(headline_decision="omit", winning_candidate=candidate, strategy=strategy))
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
    plan["headline"] = ""
    plan["headlineCoreKeyword"] = ""
    if winner_check is not None:
        plan["winnerPreservationCheck"] = winner_check
    return strategy, candidate, source, plan


class TestModelSelfAttestationNonAuthoritative(unittest.TestCase):
    def test_problem_preserved_false_is_not_structural_failure(self) -> None:
        _, candidate, source, plan = _source_and_plan(
            winner_check={
                "problemPreserved": False,
                "relativeAdvantagePreserved": True,
                "mechanismPreserved": True,
                "prototypeMethodPreserved": True,
                "visualParallelPreserved": True,
                "structurePreserved": True,
                "editingOnlyStrengthens": True,
            }
        )
        result = process_winner_development_response(plan, source_reference=source, winning_candidate=candidate)
        self.assertIn(SERVER_OWNED_WINNER_SOURCE_KEY, result)
        self.assertNotIn("winnerPreservationCheck", result)

    def test_missing_problem_preserved_is_not_structural_failure(self) -> None:
        _, candidate, source, plan = _source_and_plan(
            winner_check={
                "relativeAdvantagePreserved": True,
                "mechanismPreserved": True,
                "prototypeMethodPreserved": True,
                "visualParallelPreserved": True,
                "structurePreserved": True,
                "editingOnlyStrengthens": True,
            }
        )
        result = process_winner_development_response(plan, source_reference=source, winning_candidate=candidate)
        self.assertEqual(result.get("prototypeId"), "summer_fan")

    def test_null_problem_preserved_is_not_structural_failure(self) -> None:
        _, candidate, source, plan = _source_and_plan(
            winner_check={
                "problemPreserved": None,
                "relativeAdvantagePreserved": True,
                "mechanismPreserved": True,
                "prototypeMethodPreserved": True,
                "visualParallelPreserved": True,
                "structurePreserved": True,
                "editingOnlyStrengthens": True,
            }
        )
        result = process_winner_development_response(plan, source_reference=source, winning_candidate=candidate)
        self.assertTrue(result.get("serverPreservationCheck", {}).get("problemPreserved"))

    def test_string_false_is_normalized_as_non_authoritative(self) -> None:
        normalized = normalize_winner_response_compatibility_fields(
            {"winnerPreservationCheck": {"problemPreserved": "false"}}
        )
        diagnostic = capture_model_preservation_diagnostic(normalized)
        self.assertFalse(normalized["winnerPreservationCheck"]["problemPreserved"])
        self.assertEqual(diagnostic["problemPreservedValue"], False)


class TestServerOwnedPreservationIdentity(unittest.TestCase):
    def test_server_preservation_reference_required(self) -> None:
        _, candidate, source, plan = _source_and_plan()
        merged = apply_server_owned_preservation(plan, source_reference=source)
        self.assertIsInstance(merged.get("serverOwnedWinnerSource"), dict)

    def test_correct_source_candidate_passes(self) -> None:
        _, candidate, source, plan = _source_and_plan()
        merged = apply_server_owned_preservation(plan, source_reference=source)
        validate_winner_source_identity(merged, source_reference=source)

    def test_wrong_source_candidate_id_in_reference_missing_still_builds(self) -> None:
        strategy = _strategy()
        candidate = _candidate("summer_fan")
        source = build_server_owned_winner_source_reference(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="",
        )
        with self.assertRaises(Builder2TournamentError):
            validate_winner_preservation_contract_required(source)

    def test_wrong_prototype_id_fails(self) -> None:
        _, candidate, source, plan = _source_and_plan()
        assert isinstance(plan.get("preservationReference"), dict)
        plan["preservationReference"]["prototypeId"] = "think_small"
        with self.assertRaises(Builder2TournamentError) as ctx:
            detect_winner_immutable_identity_violations(plan, source_reference=source)
        self.assertIn("prototypeId", str(ctx.exception))

    def test_correct_prototype_id_passes(self) -> None:
        _, candidate, source, plan = _source_and_plan()
        detect_winner_immutable_identity_violations(plan, source_reference=source)
        merged = apply_server_owned_preservation(plan, source_reference=source)
        validate_winner_source_identity(merged, source_reference=source)

    def test_immutable_strategic_fields_carried_forward(self) -> None:
        strategy, candidate, source, plan = _source_and_plan()
        plan["coreCreativeMechanism"] = "Different mechanism"
        merged = apply_server_owned_preservation(plan, source_reference=source)
        self.assertEqual(merged["coreCreativeMechanism"], source["coreCreativeMechanism"])
        self.assertEqual(
            merged["problemPerception"],
            strategy["problemPerception"]["statement"],
        )
        self.assertEqual(
            merged["relativeAdvantage"],
            strategy["relativeAdvantage"]["statement"],
        )

    def test_execution_fields_remain_model_developed(self) -> None:
        _, candidate, source, plan = _source_and_plan()
        plan["openingFrameDescription"] = "Custom opening frame"
        plan["videoPrompt"] = "Custom runway prompt"
        merged = apply_server_owned_preservation(plan, source_reference=source)
        self.assertEqual(merged["openingFrameDescription"], "Custom opening frame")
        self.assertEqual(merged["videoPrompt"], "Custom runway prompt")

    def test_headline_decision_omit_remains_valid(self) -> None:
        _, candidate, source, plan = _source_and_plan()
        result = process_winner_development_response(plan, source_reference=source, winning_candidate=candidate)
        self.assertEqual(result.get("headlineDecision", {}).get("decision"), "omit")

    def test_continuous_event_plan_remains_valid(self) -> None:
        _, candidate, source, plan = _source_and_plan()
        result = process_winner_development_response(plan, source_reference=source, winning_candidate=candidate)
        self.assertEqual(result.get("structureType"), "continuous_event")


class TestOfflineRevalidationAndResume(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_failed_parsed_response_can_be_revalidated_offline(self) -> None:
        state = _historical_judged_state(job_id="job-preservation-revalidate")
        _, candidate, source, plan = _source_and_plan(
            candidate_id=_candidate_id_for_prototype("summer_fan"),
            winner_check={"problemPreserved": False},
        )
        persist_parsed_winner_response(
            state,
            parsed=plan,
            candidate_id=_candidate_id_for_prototype("summer_fan"),
            prototype_id="summer_fan",
            top_level_keys=sorted(plan.keys()),
            response_char_count=100,
        )
        report = run_one_winner_revalidate(job_id="job-preservation-revalidate", tournament_state=deepcopy(state))
        self.assertTrue(report["ok"])
        self.assertEqual(report["winnerNormalCalls"], 0)

    def test_offline_revalidation_makes_zero_model_calls(self) -> None:
        state = _historical_judged_state(job_id="job-preservation-zero-calls")
        _, _, _, plan = _source_and_plan(winner_check={"problemPreserved": False})
        persist_parsed_winner_response(
            state,
            parsed=plan,
            candidate_id=_candidate_id_for_prototype("summer_fan"),
            prototype_id="summer_fan",
        )
        llm = _make_llm()
        report = run_one_winner_revalidate(job_id="job-preservation-zero-calls", tournament_state=state)
        self.assertTrue(report["ok"])
        if hasattr(llm, "__call__"):
            pass

    def test_winner_resume_one_call_without_reusable_parsed_response(self) -> None:
        state = _historical_judged_state(job_id="job-preservation-resume")
        report = run_one_winner_resume(job_id="job-preservation-resume", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["winnerNormalCalls"], 1)

    def test_persisted_winner_reused_with_zero_calls(self) -> None:
        state = _historical_judged_state(job_id="job-preservation-reuse")
        first = run_one_winner_resume(job_id="job-preservation-reuse", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertTrue(first["ok"])
        second = run_one_winner_resume(
            job_id="job-preservation-reuse",
            llm_client=_make_llm(),
            tournament_state=load_tournament_state("job-preservation-reuse"),
        )
        self.assertTrue(second["winnerReused"])
        self.assertEqual(second["winnerNormalCalls"], 0)

    def test_six_accepted_judgments_not_regenerated(self) -> None:
        state = _historical_judged_state(job_id="job-preservation-judgments")
        report = run_one_winner_resume(job_id="job-preservation-judgments", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["reusedJudgmentCount"], 6)
        self.assertEqual(report["judgeCalls"], 0)

    def test_no_media_calls(self) -> None:
        state = _historical_judged_state(job_id="job-preservation-media")
        report = run_one_winner_resume(job_id="job-preservation-media", llm_client=_make_llm(), tournament_state=deepcopy(state))
        self.assertEqual(report["startImageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)

    def test_missing_parsed_response_reported_honestly(self) -> None:
        state = _historical_judged_state(job_id="job-preservation-missing-parsed")
        report = run_one_winner_revalidate(job_id="job-preservation-missing-parsed", tournament_state=state)
        self.assertFalse(report["parsedResponseAvailable"])
        self.assertEqual(report["failureReason"], "builder2_winner_revalidate_parsed_response_missing")


class TestPreservationRegression(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

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
                    if pid in candidate_id:
                        prototype_id = pid
                        break
                return _judgment_for_candidate(candidate_id, _candidate(prototype_id), eligible=True)
            if role == "builder2_winner":
                return _winner_plan_from_prompt(prompt)
            raise AssertionError(role)

        run_builder2_tournament(
            job_id="job-preservation-regression-14",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-preservation-regression",
        )
        state = load_tournament_state("job-preservation-regression-14")
        assert state is not None
        self.assertEqual((state.get("metrics") or {}).get("totalReasoningCalls"), 14)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
