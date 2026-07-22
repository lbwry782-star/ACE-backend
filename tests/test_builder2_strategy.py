"""
Builder2 strategy foundation tests — parsing, validation, repair, diagnostics.
"""
from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_strategy import (
    generate_strategy_foundation,
    normalize_strategy_raw,
    validate_strategy_foundation,
)
from engine.builder2_tournament_contracts import (
    STRATEGY_SCHEMA_VERSION,
    VALID_GROUNDING_TYPES,
    Builder2TournamentError,
)
from engine.builder2_tournament_llm import extract_responses_output_text, parse_json_object
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, new_tournament_state
from engine.builder2_tournament_manager import run_builder2_tournament
from tests.test_builder2_tournament import TournamentMockLLM, _strategy


def _valid_strategy(*, language: str = "en", grounding_type: str = "common_market_behavior") -> Dict[str, Any]:
    data = _strategy(language=language)
    data["problemPerception"]["groundingType"] = grounding_type
    return data


class TestStrategyValidation(unittest.TestCase):
    def test_valid_english_strategy_passes(self) -> None:
        validate_strategy_foundation(_valid_strategy(language="en"))

    def test_valid_hebrew_strategy_passes(self) -> None:
        validate_strategy_foundation(_valid_strategy(language="he"))

    def test_every_allowed_grounding_type_passes(self) -> None:
        for gt in VALID_GROUNDING_TYPES:
            validate_strategy_foundation(_valid_strategy(grounding_type=gt))

    def test_valid_perceptual_problem_passes(self) -> None:
        data = _valid_strategy()
        data["problemPerception"]["statement"] = (
            "Buyers often fail to notice how much closer this option feels to their actual need."
        )
        data["problemPerception"]["groundingType"] = "observable_practice"
        validate_strategy_foundation(data)

    def test_valid_physical_problem_passes(self) -> None:
        data = _valid_strategy()
        data["problemPerception"]["groundingType"] = "physical_reality"
        validate_strategy_foundation(data)

    def test_malformed_json_code(self) -> None:
        data = _valid_strategy()
        data["schemaVersion"] = "wrong"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation(data)
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_schema_invalid:schemaVersion")
        with self.assertRaises(ValueError):
            parse_json_object("not json at all")

    def test_wrong_schema_version(self) -> None:
        data = _valid_strategy()
        data["schemaVersion"] = "wrong"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation(data)
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_schema_invalid:schemaVersion")

    def test_empty_evidence_validation_failure(self) -> None:
        data = _valid_strategy()
        data["problemPerception"]["groundingEvidence"] = []
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation(data)
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_validation_failed:problemPerception.groundingEvidence")

    def test_unsupported_grounding_enum(self) -> None:
        data = _valid_strategy()
        data["problemPerception"]["groundingType"] = "fabricated_study"
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation(data)
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_schema_invalid:problemPerception.groundingType")

    def test_model_planning_failure_is_not_grounded(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_strategy_foundation({"planningFailure": "builder2_strategy_not_grounded"})
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_not_grounded")

    def test_string_evidence_is_normalized(self) -> None:
        data = _valid_strategy()
        data["problemPerception"]["groundingEvidence"] = "Customers compare against familiar agencies by default."
        normalized = normalize_strategy_raw(data)
        self.assertEqual(len(normalized["problemPerception"]["groundingEvidence"]), 1)
        validate_strategy_foundation(data)

    def test_language_alias_normalized(self) -> None:
        data = _valid_strategy()
        data["language"] = "Hebrew"
        normalized = normalize_strategy_raw(data)
        self.assertEqual(normalized["language"], "he")
        validate_strategy_foundation(data)


class TestResponsesExtraction(unittest.TestCase):
    def test_extracts_output_text_attribute(self) -> None:
        class Response:
            output_text = '{"schemaVersion":"builder2_strategy_v1"}'

        self.assertIn("builder2_strategy_v1", extract_responses_output_text(Response()))

    def test_extracts_output_text_parts(self) -> None:
        class Content:
            type = "output_text"
            text = '{"schemaVersion":"builder2_strategy_v1","language":"en"}'

        class Block:
            content = [Content()]

        class Response:
            output_text = ""
            output = [Block()]

        parsed = parse_json_object(extract_responses_output_text(Response()))
        self.assertEqual(parsed["language"], "en")


class TestStrategyGenerationFlow(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _state(self, job_id: str = "job-strategy") -> Dict[str, Any]:
        return new_tournament_state(
            job_id=job_id,
            language="en",
            active_prototype_ids=["closest"],
            random_seed="seed",
        )

    def test_empty_response_code(self) -> None:
        state = self._state("job-empty")
        with self.assertRaises(Builder2TournamentError) as ctx:
            generate_strategy_foundation(
                product_name="ACE Product",
                product_description="desc",
                language="en",
                llm_client=lambda **kwargs: "",
                state=state,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_empty_response")

    def test_malformed_response_produces_malformed_code(self) -> None:
        state = self._state()
        llm = lambda **kwargs: "not-json"

        with self.assertRaises(Builder2TournamentError) as ctx:
            generate_strategy_foundation(
                product_name="ACE Product",
                product_description="desc",
                language="en",
                llm_client=llm,
                state=state,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_strategy_malformed_response")
        self.assertEqual(state["strategyDiagnostics"]["parseStatus"], "failed")

    def test_repair_attempted_for_schema_invalid_output(self) -> None:
        calls: List[str] = []

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs.get("role", ""))
            if len(calls) == 1:
                bad = _valid_strategy()
                bad["schemaVersion"] = "builder2_strategy_v0"
                return bad
            return _valid_strategy()

        state = self._state("job-repair")
        foundation = generate_strategy_foundation(
            product_name="ACE Product",
            product_description="desc",
            language="en",
            llm_client=llm,
            state=state,
        )
        self.assertIn("problemPerception", foundation)
        self.assertEqual(len(calls), 2)
        self.assertTrue(state["strategyDiagnostics"]["repairAttempted"])
        self.assertEqual(state["metrics"]["strategyRepairCalls"], 1)

    def test_failed_repair_preserves_exact_reason(self) -> None:
        def llm(**kwargs: Any) -> Dict[str, Any]:
            bad = _valid_strategy()
            bad["schemaVersion"] = "builder2_strategy_v0"
            return bad

        state = self._state("job-repair-fail")
        with self.assertRaises(Builder2TournamentError) as ctx:
            generate_strategy_foundation(
                product_name="ACE Product",
                product_description="desc",
                language="en",
                llm_client=llm,
                state=state,
            )
        self.assertTrue(str(ctx.exception.args[0]).startswith("builder2_strategy_schema_invalid:schemaVersion"))
        self.assertTrue(state["strategyDiagnostics"]["repairAttempted"])

    def test_no_creator_after_final_strategy_failure(self) -> None:
        calls: List[str] = []

        def failing_strategy(**kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs.get("role", ""))
            if kwargs.get("role") == "builder2_strategy":
                return {"planningFailure": "builder2_strategy_not_grounded"}
            return _strategy()

        with patch.dict(
            os.environ,
            {"BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": "closest", "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1"},
            clear=True,
        ):
            with self.assertRaises(Builder2TournamentError):
                run_builder2_tournament(
                    job_id="job-no-creator",
                    product_name="ACE Product",
                    product_description="desc",
                    content_language="en",
                    llm_client=failing_strategy,
                    rng_seed="seed",
                )
        self.assertEqual(calls, ["builder2_strategy"])
        state = load_tournament_state("job-no-creator")
        assert state is not None
        self.assertEqual(state["error"], "builder2_strategy_not_grounded")


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_env_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")
        self.assertEqual(STRATEGY_SCHEMA_VERSION, "builder2_strategy_v1")


if __name__ == "__main__":
    unittest.main()
