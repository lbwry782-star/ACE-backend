"""
Builder2 Winner repair failure inspect — read-only replay and exception-chain tests.
"""
from __future__ import annotations

import io
import json
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_winner_repair_failure_inspect import (
    _build_headline_text_timing_assessment,
    _build_required_winner_field_audit,
    _prepare_preserved_plan,
    _replay_low_level_winner_plan_validation,
    _build_exception_chain,
    inspect_builder2_winner_repair_failure,
    main,
)
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    build_winning_candidate_preservation_snapshot,
)
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt
from tests.test_builder2_winner_headline_repair import (
    _forgot_winner_id,
    _judgment_requiring_headline,
    _parsed_plan_missing_headline,
    _six_six_missing_headline_state,
)


def _repaired_failure_state(*, headline: str, keyword: str) -> Dict[str, Any]:
    state = _six_six_missing_headline_state(repair_calls=1)
    winner_id = _forgot_winner_id(state)
    plan = _parsed_plan_missing_headline(candidate_id=winner_id)
    plan["headline"] = headline
    plan["headlineCoreKeyword"] = keyword
    state[PARSED_WINNER_RESPONSE_KEY] = {
        "parsed": plan,
        "candidateId": winner_id,
        "prototypeId": "forgot",
        "topLevelKeys": sorted(plan.keys()),
        "topLevelKeyCount": len(plan),
        "responseCharCount": 73,
    }
    state["metrics"]["winnerDevelopmentCalls"] = 2
    state["metrics"]["winnerRepairCalls"] = 1
    state["failureStage"] = "winner_development"
    state["failureReason"] = "builder2_winner_development_failed"
    return state


def _preserved_plan_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    winner_id = _forgot_winner_id(state)
    parsed = dict(state[PARSED_WINNER_RESPONSE_KEY]["parsed"])
    winner_rec = state["candidates"][winner_id]
    winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
    judgment_id = winner_rec["judgmentId"]
    winning_judgment = state["judgments"][judgment_id]["judgment"]
    strategy = state["strategyFoundation"]
    source_reference = build_server_owned_winner_source_reference(
        strategy_foundation=strategy,
        winning_candidate=winning_candidate,
        candidate_id=winner_id,
    )
    return _prepare_preserved_plan(
        parsed,
        source_reference=source_reference,
        winning_candidate=winning_candidate,
        winning_judgment=winning_judgment,
        compatibility_mode=bool(state.get("methodologyCompatibilityMode")),
    )


class TestWinnerRepairFailureInspectDiagnostics(unittest.TestCase):
    def test_low_level_identifies_missing_sequence_development(self) -> None:
        state = _repaired_failure_state(headline="valid words here", keyword="valid")
        winner_id = _forgot_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        winning_judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        preserved = _preserved_plan_from_state(state)
        preserved["sequence"] = dict(preserved.get("sequence") or {})
        preserved["sequence"].pop("development", None)
        stages = _replay_low_level_winner_plan_validation(
            preserved,
            winning_candidate=winning_candidate,
            preservation_snapshot=build_winning_candidate_preservation_snapshot(
                strategy_foundation=state["strategyFoundation"],
                winning_candidate=winning_candidate,
                candidate_id=winner_id,
            ),
            winning_judgment=winning_judgment,
            compatibility_mode=False,
        )
        first = next(stage for stage in stages if stage.get("firstFailure"))
        self.assertEqual(first["exactFieldPath"], "sequence.development")
        self.assertEqual(first["exactSafeErrorCode"], "builder2_tournament_invalid_field:sequence.development")

    def test_wrong_type_distinguished_from_missing(self) -> None:
        state = _repaired_failure_state(headline="valid words here", keyword="valid")
        winner_id = _forgot_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        winning_judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        preserved = _preserved_plan_from_state(state)
        preserved["sequence"] = "not-a-dict"
        audit = _build_required_winner_field_audit(
            preserved,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
        )
        sequence_entry = next(entry for entry in audit if entry["fieldPath"] == "sequence")
        self.assertEqual(sequence_entry["structuralStatus"], "wrong_type")
        missing_entry = next(entry for entry in audit if entry["fieldPath"] == "sequence.development")
        self.assertIn(missing_entry["structuralStatus"], {"missing", "null"})

    def test_invalid_enum_reported_safely(self) -> None:
        state = _repaired_failure_state(headline="valid words here", keyword="valid")
        winner_id = _forgot_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        winning_candidate = winner_rec.get("creatorSnapshot") or winner_rec.get("creatorOutput") or {}
        winning_judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        preserved = _preserved_plan_from_state(state)
        preserved["structureType"] = "invalid_structure"
        audit = _build_required_winner_field_audit(
            preserved,
            winning_candidate=winning_candidate,
            winning_judgment=winning_judgment,
        )
        structure_entry = next(entry for entry in audit if entry["fieldPath"] == "structureType")
        self.assertEqual(structure_entry["structuralStatus"], "invalid_enum")
        stages = _replay_low_level_winner_plan_validation(
            preserved,
            winning_candidate=winning_candidate,
            preservation_snapshot={},
            winning_judgment=winning_judgment,
            compatibility_mode=False,
        )
        first = next(stage for stage in stages if stage.get("firstFailure"))
        self.assertEqual(first["stageName"], "structureType_unrecognized")
        self.assertEqual(first["exactFieldPath"], "structureType")

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_generic_wrapper_origin_for_montage_language(self, read_raw: Any, _redis: Any) -> None:
        state = _repaired_failure_state(headline="valid words here", keyword="valid")
        plan = state[PARSED_WINNER_RESPONSE_KEY]["parsed"]
        plan["videoPrompt"] = "A montage of quick cuts across the scene"
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_winner_repair_failure("job-montage-language")
        self.assertTrue(report["genericWrapperLostInnerError"])
        origin = report["genericWrapperOrigin"]
        self.assertEqual(origin["file"], "engine/builder2_winner_plan.py")
        self.assertEqual(origin["function"], "validate_builder2_winner_plan")
        self.assertEqual(report["firstConcreteFailureField"], "videoPrompt")
        self.assertEqual(report["firstConcreteFailingStage"], "continuous_event_videoPrompt_montage_language")
        self.assertTrue(report["offlineDataSufficientForDiagnosis"])

    def test_headline_text_timing_assessment(self) -> None:
        timing = _build_headline_text_timing_assessment()
        self.assertEqual(timing["canonicalHelper"], "compose_builder2_headline_text")
        self.assertFalse(timing["requiredByBaseWinnerPlanValidation"])
        self.assertTrue(timing["requiredByHeadlineCompositionValidation"])
        self.assertFalse(timing["missingHeadlineTextCanExplainBaseWinnerPlanFailure"])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_audit_never_emits_non_empty_creative_text(self, read_raw: Any, _redis: Any) -> None:
        secret = "TOP SECRET CREATIVE COPY"
        state = _repaired_failure_state(headline=secret, keyword="SECRET")
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_winner_repair_failure("job-audit-redact")
        payload = json.dumps(report)
        self.assertNotIn(secret, payload)
        for entry in report["requiredWinnerFieldAudit"]:
            self.assertNotIn("value", entry)


class TestWinnerRepairFailureInspectReadOnly(unittest.TestCase):
    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    @patch("engine.builder2_tournament_store.load_tournament_state")
    @patch("engine.builder2_tournament_store.save_tournament_state")
    @patch("engine.builder2_tournament_store._write_raw")
    @patch("engine.builder2_execution_lease.acquire_job_lease", return_value=False)
    @patch("engine.builder2_winner_persistence.persist_winner_development_atomically")
    @patch("engine.builder2_winner_preservation_contract.persist_parsed_winner_response")
    def test_read_raw_only_zero_mutations(
        self,
        persist_parsed: Any,
        persist_plan: Any,
        _lease: Any,
        write_raw: Any,
        save_state: Any,
        load_state: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        original = _repaired_failure_state(headline="wrong words", keyword="Quality")
        read_raw.return_value = deepcopy(original)
        before = deepcopy(original)
        report = inspect_builder2_winner_repair_failure("job-repair-failure-inspect")
        self.assertTrue(report["ok"])
        self.assertEqual(report["redisMutations"], 0)
        read_raw.assert_called_once_with("job-repair-failure-inspect")
        load_state.assert_not_called()
        save_state.assert_not_called()
        write_raw.assert_not_called()
        persist_plan.assert_not_called()
        persist_parsed.assert_not_called()
        self.assertEqual(read_raw.return_value[PARSED_WINNER_RESPONSE_KEY], before[PARSED_WINNER_RESPONSE_KEY])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_non_empty_creative_text_never_emitted(self, read_raw: Any, _redis: Any) -> None:
        secret = "SECRET HEADLINE TEXT MUST NOT LEAK"
        state = _repaired_failure_state(headline=secret, keyword="SECRET")
        read_raw.return_value = deepcopy(state)
        payload = json.dumps(inspect_builder2_winner_repair_failure("job-redact"))
        self.assertNotIn(secret, payload)
        self.assertTrue(json.loads(payload)["repairedFieldMetadata"]["headline"]["valuePresent"])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_inner_error_captured_from_wrapped_chain(self, read_raw: Any, _redis: Any) -> None:
        state = _repaired_failure_state(headline="totally unrelated phrase", keyword="Quality")
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_winner_repair_failure("job-inner-error")
        self.assertTrue(report["ok"])
        chain = report["exceptionChain"]
        self.assertTrue(chain)
        codes = [entry["safeErrorCode"] for entry in chain]
        self.assertTrue(any("headline" in code.lower() or "keyword" in code.lower() for code in codes))
        self.assertIsNotNone(report["firstFailingStage"])
        self.assertTrue(report["failureIsHeadlineRelated"])

    def test_exception_chain_reports_cause_and_context(self) -> None:
        inner = Builder2TournamentError("builder2_headline_composition_invalid:keyword_not_in_headline")
        outer = Builder2TournamentError("builder2_winner_development_failed")
        outer.__cause__ = inner
        chain = _build_exception_chain(outer)
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["safeErrorCode"], "builder2_winner_development_failed")
        self.assertEqual(chain[1]["safeErrorCode"], "builder2_headline_composition_invalid")
        self.assertEqual(chain[0]["causeClass"], "Builder2TournamentError")
        self.assertTrue(chain[1]["wrapped"])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_validation_stages_identified(self, read_raw: Any, _redis: Any) -> None:
        state = _repaired_failure_state(headline="wrong words", keyword="Quality")
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_winner_repair_failure("job-stages")
        stages = report["validationStages"]
        self.assertTrue(stages["preservationApplied"]["accepted"])
        self.assertFalse(stages["finalOfflineProcessing"]["accepted"])
        self.assertIn("headlineCompositionValidation", report["firstFailingStage"])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_repair_metrics_reported(self, read_raw: Any, _redis: Any) -> None:
        state = _repaired_failure_state(headline="wrong", keyword="Quality")
        read_raw.return_value = deepcopy(state)
        metrics = inspect_builder2_winner_repair_failure("job-metrics")["repairMetrics"]
        self.assertEqual(metrics["winnerRepairCalls"], 1)
        self.assertEqual(metrics["winnerNormalCalls"], 1)
        self.assertEqual(metrics["winnerDevelopmentCalls"], 2)

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_missing_tournament(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = None
        report = inspect_builder2_winner_repair_failure("missing")
        self.assertFalse(report["ok"])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_inspection_succeeds_when_validation_fails(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_repaired_failure_state(headline="bad", keyword="Quality"))
        report = inspect_builder2_winner_repair_failure("job-expected-fail")
        self.assertTrue(report["ok"])
        self.assertTrue(report["inspectionCompleted"])
        self.assertFalse(report["offlineRepairableWithoutModelCall"])

    @patch("engine.builder2_winner_repair_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_repair_failure_inspect._read_raw")
    def test_main_exit_codes(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_repaired_failure_state(headline="bad", keyword="Quality"))
        with patch.dict("os.environ", {"BUILDER2_WINNER_REPAIR_FAILURE_INSPECT_JOB_ID": "job-main"}):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main()
            self.assertEqual(code, 0)

        read_raw.return_value = None
        with patch.dict("os.environ", {"BUILDER2_WINNER_REPAIR_FAILURE_INSPECT_JOB_ID": "missing"}):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main()
            self.assertEqual(code, 1)

    def test_inspect_module_has_no_paid_or_media_imports(self) -> None:
        import ast
        import inspect as inspect_module
        import engine.builder2_winner_repair_failure_inspect as mod

        source = inspect_module.getsource(mod)
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in (
            "openai",
            "engine.runway_video",
            "engine.builder2_image",
            "engine.ffmpeg",
        ):
            self.assertFalse(any(name == forbidden or name.startswith(forbidden + ".") for name in imported))
        self.assertNotIn("load_tournament_state", source)
        self.assertNotIn("persist_winner_development_atomically", source)

    def test_builder1_unchanged(self) -> None:
        import glob
        import os

        root = os.path.dirname(os.path.dirname(__file__))
        for path in glob.glob(os.path.join(root, "engine", "builder1*.py")):
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("winner_repair_failure_inspect", source)


if __name__ == "__main__":
    unittest.main()
