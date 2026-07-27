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
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY
from engine.builder2_winner_repair_failure_inspect import (
    _build_exception_chain,
    inspect_builder2_winner_repair_failure,
    main,
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
