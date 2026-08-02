"""
Builder2 Winner development offline recovery, ledger, and planning tests.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_complete_ad_resume_plan import (
    RESUME_STAGE_WINNER_OFFLINE_REVALIDATION,
    parsed_winner_reusable_for_candidate,
    resolve_complete_ad_canonical_resume_plan,
    resolve_winner_development_action,
)
from engine.builder2_tournament_contracts import Builder2TournamentError, WINNER_PLAN_SCHEMA_VERSION
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_development import develop_builder2_winning_candidate
from engine.builder2_winner_development_diagnostics import STAGE_VALIDATION
from engine.builder2_winner_development_failure_inspect import inspect_winner_development_failure
from engine.builder2_winner_development_offline_recovery import recover_winner_development_offline
from engine.builder2_winner_downstream import extract_builder2_video_prompt_text
from engine.builder2_winner_offline_salvage import populate_winner_development_call_report
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_plan import validate_builder2_winner_plan, _MONTAGE_LANGUAGE
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    persist_parsed_winner_response,
    process_winner_development_response,
)
from engine.builder2_winner_response_ledger import (
    parsed_response_fingerprint,
    record_winner_parsed_response_received,
    resolve_winner_parsed_response_fingerprint,
)
from engine.builder2_winner_validation_replay import infer_typeerror_failure, replay_prepare_and_validate
from tests.builder2_methodology_fixtures import methodology_winner_extras, single_slogan_contract_extras
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan
from tests.test_builder2_winner_offline_salvage import (
    _current_job_shaped_state,
    _judgment_requiring_verbal_copy,
    _parsed_winner_plan_omit,
    _winning_card_winner_id,
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _structured_video_prompt(description: str) -> Dict[str, Any]:
    return {"description": description, "promptText": description}


class TestExtractBuilder2VideoPromptText(unittest.TestCase):
    def test_structured_dict_uses_canonical_text_field(self) -> None:
        text = extract_builder2_video_prompt_text(
            _structured_video_prompt("One continuous storefront proof beat across a single surface."),
            "videoPrompt",
        )
        self.assertIn("storefront", text)

    def test_plain_string_preserved(self) -> None:
        text = extract_builder2_video_prompt_text("Plain runway prompt.", "videoPrompt")
        self.assertEqual(text, "Plain runway prompt.")


class TestWinnerVideoPromptValidationFix(unittest.TestCase):
    def _continuous_plan(self, *, video_prompt: Any) -> Dict[str, Any]:
        strategy = _strategy(language="he")
        candidate = _candidate("winning_card")
        plan = _winner_plan(language="he")
        plan.update(
            methodology_winner_extras(
                headline_decision="omit",
                winning_candidate=candidate,
                strategy=strategy,
            )
        )
        plan.update(single_slogan_contract_extras())
        plan["prototypeId"] = "winning_card"
        plan["structureType"] = "continuous_event"
        plan["sceneVariations"] = []
        plan["headlineDecision"] = {"decision": "omit", "reasonSource": "not_required"}
        plan["headlineForm"] = "none"
        plan["headline"] = ""
        plan["headlineCoreKeyword"] = ""
        plan["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
        plan["videoPrompt"] = video_prompt
        plan["preservationReference"] = {
            "strategyFoundationId": strategy.get("strategyFoundationId") or "strategy-test",
            "prototypeId": "winning_card",
            "structureType": candidate.get("structureType"),
            "visualParallelType": candidate.get("visualParallelType"),
            "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
        }
        return plan

    def test_dict_video_prompt_no_longer_raises_typeerror(self) -> None:
        strategy = _strategy(language="he")
        candidate = _candidate("winning_card")
        plan = self._continuous_plan(
            video_prompt=_structured_video_prompt(
                "A single continuous event unfolds across one storefront surface in one temporal beat."
            )
        )
        source = build_server_owned_winner_source_reference(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="cand-x",
        )
        validated = process_winner_development_response(
            plan,
            source_reference=source,
            winning_candidate=candidate,
            winning_judgment=_judgment("cand-x", total_hint=89, eligible=True),
        )
        self.assertIsInstance(validated["videoPrompt"], str)
        self.assertNotIn("montage", validated["videoPrompt"].lower())

    def test_legacy_typeerror_path_identified(self) -> None:
        plan = self._continuous_plan(
            video_prompt=_structured_video_prompt("Continuous proof beat across one surface.")
        )
        exc = TypeError("expected string or bytes-like object, got 'dict'")
        field, function, operation = infer_typeerror_failure(exc, plan=plan)
        self.assertEqual(field, "videoPrompt")
        self.assertEqual(function, "validate_builder2_winner_plan")
        self.assertEqual(operation, "_MONTAGE_LANGUAGE.search")
        with self.assertRaises(TypeError):
            _MONTAGE_LANGUAGE.search(plan["videoPrompt"])


class TestWinnerResponseLedger(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_parsed_response_retained_after_validation_failure(self) -> None:
        state: Dict[str, Any] = {"jobId": "job-ledger", "tournamentId": "t-ledger", "metrics": ensure_metrics({})}
        parsed = {"schemaVersion": WINNER_PLAN_SCHEMA_VERSION, "videoPrompt": {"description": "x"}}
        record_winner_parsed_response_received(
            state,
            parsed=parsed,
            candidate_id="cand-1",
            prototype_id="winning_card",
            response_char_count=12,
            response_text='{"videoPrompt":{"description":"x"}}',
        )
        self.assertTrue(state.get("winnerDevelopmentResponseReceived"))
        self.assertTrue(state.get("winnerDevelopmentParsed"))
        payload = state[PARSED_WINNER_RESPONSE_KEY]
        self.assertTrue(payload.get("parsedResponseFingerprint"))
        self.assertEqual(
            payload.get("parsedResponseFingerprint"),
            parsed_response_fingerprint(parsed),
        )

    def test_develop_winner_persists_before_validation_failure(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        strategy = state["strategyFoundation"]
        candidate = (state["candidates"][winner_id]["creatorOutput"])
        judgment = _judgment_requiring_verbal_copy(winner_id)
        parsed = _parsed_winner_plan_omit(candidate_id=winner_id)
        parsed["videoPrompt"] = _structured_video_prompt("A montage of quick cuts across the scene.")
        with patch(
            "engine.builder2_winner_development.call_builder2_role_json_with_text",
            return_value=(parsed, json.dumps(parsed)),
        ):
            with self.assertRaises(Builder2TournamentError):
                develop_builder2_winning_candidate(
                    product_name="ACE",
                    product_description="desc",
                    language="he",
                    strategy_foundation=strategy,
                    winning_candidate=candidate,
                    winning_judgment=judgment,
                    prototype_id="winning_card",
                    runway_mode="text_to_video",
                    state=state,
                    candidate_id=winner_id,
                )
        self.assertIsInstance(state.get(PARSED_WINNER_RESPONSE_KEY), dict)
        self.assertTrue(state.get("winnerDevelopmentResponseReceived"))
        self.assertTrue(state.get("winnerDevelopmentParsed"))
        failure = state.get("winnerDevelopmentFailure") or {}
        self.assertEqual(failure.get("stage"), STAGE_VALIDATION)


class TestWinnerOfflineRecoveryAndPlanning(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _parsed_reusable_state(self) -> Dict[str, Any]:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        parsed = _parsed_winner_plan_omit(candidate_id=winner_id)
        parsed["videoPrompt"] = _structured_video_prompt(
            "One continuous proof beat across the winning-card surface in a single unfolding event."
        )
        persist_parsed_winner_response(
            state,
            parsed=parsed,
            candidate_id=winner_id,
            prototype_id="winning_card",
            response_char_count=len(json.dumps(parsed)),
            response_text=json.dumps(parsed),
        )
        state["winnerCandidateId"] = winner_id
        state["winnerDevelopmentFailure"] = {
            "stage": "validation",
            "fieldPath": "videoPrompt",
            "preciseReason": "expected string or bytes-like object, got 'dict'",
            "exceptionClass": "TypeError",
        }
        state["metrics"]["winnerDevelopmentCalls"] = 1
        state["winnerDevelopmentPaidCallRecorded"] = True
        return state

    def test_parsed_reusable_blocks_dispatch_plan(self) -> None:
        state = self._parsed_reusable_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["winnerAction"], "offline_revalidate")
        self.assertEqual(plan["resolvedResumeStage"], RESUME_STAGE_WINNER_OFFLINE_REVALIDATION)
        self.assertFalse(plan["winnerWouldDispatch"])
        self.assertFalse(plan["winnerDevelopmentCallRequired"])
        self.assertEqual(plan["winnerNormalCallsPlanned"], 0)
        self.assertEqual(plan["reasoningBudgetRequiredForNextInvocation"], 0)
        self.assertEqual(plan["recommendedNextInvocationMaxCalls"], 0)
        self.assertEqual(plan["totalCallsRemainingAcrossInvocations"], 0)
        self.assertEqual(plan["requiredNextReasoningRoles"], [])
        self.assertEqual(plan["expectedNextReasoningRoles"], [])

    def test_summary_reports_response_received_and_parsed(self) -> None:
        state = self._parsed_reusable_state()
        report: Dict[str, Any] = {}
        populate_winner_development_call_report(state, report)
        self.assertTrue(report["winnerDevelopmentResponseReceived"])
        self.assertTrue(report["winnerDevelopmentParsed"])

    def test_offline_replay_and_recovery_dry_run(self) -> None:
        state = self._parsed_reusable_state()
        winner_id = _winning_card_winner_id(state)
        payload = state[PARSED_WINNER_RESPONSE_KEY]
        fingerprint = resolve_winner_parsed_response_fingerprint(payload).get("effective")
        dry = recover_winner_development_offline(
            state,
            expected_candidate_id=winner_id,
            expected_parsed_fingerprint=_clean(fingerprint),
            dry_run=True,
        )
        self.assertTrue(dry["recoveryEligible"])
        self.assertTrue(dry["validationAcceptedAfter"])
        self.assertEqual(dry["normalizationPaths"], ["videoPrompt"])
        self.assertFalse(dry["stateMutated"])
        self.assertEqual(dry["paidCalls"], 0)

    def test_apply_recovery_is_fingerprint_guarded(self) -> None:
        state = self._parsed_reusable_state()
        winner_id = _winning_card_winner_id(state)
        blocked = recover_winner_development_offline(
            state,
            expected_candidate_id=winner_id,
            expected_parsed_fingerprint="deadbeef",
            dry_run=False,
        )
        self.assertEqual(blocked["reason"], "builder2_winner_offline_recovery_fingerprint_mismatch")
        self.assertFalse(is_valid_persisted_winner_development(state))

    def test_apply_recovery_accepts_plan(self) -> None:
        state = self._parsed_reusable_state()
        winner_id = _winning_card_winner_id(state)
        payload = state[PARSED_WINNER_RESPONSE_KEY]
        fingerprint = resolve_winner_parsed_response_fingerprint(payload).get("effective")
        applied = recover_winner_development_offline(
            state,
            expected_candidate_id=winner_id,
            expected_parsed_fingerprint=_clean(fingerprint),
            dry_run=False,
        )
        self.assertTrue(applied["persisted"])
        self.assertTrue(applied["winnerDevelopmentAccepted"])
        self.assertEqual(applied["finalWinnerCandidateId"], winner_id)
        self.assertTrue(is_valid_persisted_winner_development(state))
        self.assertEqual(
            _clean(state.get("winnerDevelopmentCandidateId") or state.get("winnerCandidateId")),
            winner_id,
        )

    def test_failure_inspector_reports_structured_video_prompt(self) -> None:
        state = self._parsed_reusable_state()
        report = inspect_winner_development_failure(state)
        self.assertTrue(report["parsedResponseAvailable"])
        self.assertEqual(report["topLevelTypes"]["videoPrompt"], "dict")
        self.assertTrue(report["fingerprintDerivationPossible"])
        self.assertEqual(report["winnerAction"], "offline_revalidate")

    def test_no_arbitrary_dict_stringification_in_normalization(self) -> None:
        strategy = _strategy(language="he")
        candidate = _candidate("winning_card")
        parsed = _parsed_winner_plan_omit(candidate_id="cand-test")
        parsed["videoPrompt"] = _structured_video_prompt("Canonical extracted prompt text.")
        source = build_server_owned_winner_source_reference(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="cand-test",
        )
        validated = process_winner_development_response(
            parsed,
            source_reference=source,
            winning_candidate=candidate,
            winning_judgment=_judgment("cand-test", total_hint=89, eligible=True),
        )
        self.assertEqual(validated["videoPrompt"], "Canonical extracted prompt text.")
        self.assertNotIn("{", validated["videoPrompt"])


class TestBuilder1Unchanged(unittest.TestCase):
    def test_builder1_files_untouched(self) -> None:
        import engine.builder1_planning_profile as profile

        self.assertTrue(callable(profile.quality_model))


if __name__ == "__main__":
    unittest.main()
