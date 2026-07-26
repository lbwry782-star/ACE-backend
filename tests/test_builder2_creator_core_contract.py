"""
Builder2 Creator core contract tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_creator import (
    collect_creator_structural_errors,
    validate_creator_candidate,
)
from engine.builder2_creator_circuit_breaker import (
    SYSTEMIC_FAILURE_CODE,
    assert_creator_contract_available,
    is_creator_contract_circuit_breaker_tripped,
    record_creator_contract_failure,
)
from engine.builder2_creator_core_contract import (
    CREATOR_OWNERSHIP_CREATOR_CORE,
    CREATOR_OWNERSHIP_SERVER_DERIVED,
    build_creator_required_keys_prompt_text,
    creator_model_required_field_paths,
    filter_creator_owned_structural_errors,
)
from engine.builder2_creator_normalization import normalize_creator_candidate
from engine.builder2_methodology_validation import validate_creator_methodology
from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError, CANDIDATE_SCHEMA_VERSION
from engine.builder2_tournament_manager import run_builder2_creator_preflight, run_builder2_tournament
from engine.builder2_tournament_prompts import build_creator_prompt
from engine.builder2_tournament_recovery import (
    clear_job_queued,
    disable_memory_recovery,
    enable_memory_recovery,
    requeue_recoverable_job,
    register_recoverable_job,
    remove_recoverable_job,
    set_memory_job_hash,
)
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, new_tournament_state, save_tournament_state
from tests.builder2_methodology_fixtures import methodology_strategy_extras, realistic_core_candidate_extras
from tests.test_builder2_tournament import TournamentMockLLM, _candidate, _deep_merge, _strategy


def _realistic_candidate(prototype_id: str = "closest") -> Dict[str, Any]:
    strategy = _strategy()
    base = {
        "schemaVersion": CANDIDATE_SCHEMA_VERSION,
        "prototypeId": prototype_id,
        "coreCreativeMechanism": "Closeness shown through a simple human gesture.",
        "visualParallelType": "physical_behavior",
        "structureType": "continuous_event",
        "sevenSecondStructure": {
            "beginning": "Two people stand apart.",
            "development": "One step closes the distance.",
            "resolution": "They meet in a clear embrace.",
        },
        "visualAnchor": {
            "description": "The moment the distance closes.",
            "whyEssential": "It proves closeness visually.",
            "visualAnchorTiming": "development",
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
        "creatorReport": {
            "problemPerception": "Buyers default to familiar alternatives.",
            "relativeAdvantage": "Closeness becomes the advantage.",
            "mechanismScanSummary": "Physical closeness expresses strategic closeness.",
            "goldPrototypeUsed": prototype_id,
            "visualParallelType": "physical_behavior",
            "whyParallelExpressesAdvantage": "Closing distance makes closeness visible.",
            "whyRunwayShouldUnderstand": "One action in one location.",
            "silentVerification": "The closing distance is visible without sound.",
        },
    }
    return _deep_merge(_deep_merge(base, realistic_core_candidate_extras(prototype_id, strategy=strategy)), {})


class TestCreatorContractParity(unittest.TestCase):
    def test_prompt_contains_all_core_required_fields(self) -> None:
        prompt = build_creator_prompt(
            product_name="Product",
            product_description="desc",
            language="en",
            strategy_foundation=_strategy(),
            prototype=require_prototype("closest"),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="image_to_video",
        )
        required_text = build_creator_required_keys_prompt_text(prototype_id="closest")
        for token in ("coreCreativeMechanism", "visualMechanism", "closestApplication", "verbalPotential"):
            self.assertIn(token, required_text)
            self.assertIn(token, prompt)
        self.assertNotIn("essenceExtreme,", required_text.split("Required keys:")[1].split("Do NOT output")[0])

    def test_validator_paths_subset_of_contract(self) -> None:
        paths = creator_model_required_field_paths(prototype_id="closest")
        self.assertIn("coreCreativeMechanism", paths)
        self.assertIn("closestApplication", paths)


class TestServerDerivedNormalization(unittest.TestCase):
    def test_missing_analytical_objects_accepted(self) -> None:
        cand = _realistic_candidate("closest")
        out = validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertTrue(out["essenceExtreme"]["derivedByServer"])
        self.assertTrue(out["participationMechanism"]["derivedByServer"])
        self.assertTrue(out["visualFamilyConsistency"]["derivedByServer"])

    def test_continuous_event_not_required_montage_family(self) -> None:
        cand = _realistic_candidate("closest")
        cand.pop("visualFamilyConsistency", None)
        normalized, resolved = normalize_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        self.assertIn("visualFamilyConsistency", resolved)
        self.assertEqual(normalized["visualFamilyConsistency"]["structureType"], "continuous_event")

    def test_verbal_not_needed_accepted(self) -> None:
        cand = _realistic_candidate("closest")
        validate_creator_candidate(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )


class TestPrototypeRealisticFixtures(unittest.TestCase):
    PROTOTYPES = DEFAULT_ACTIVE_PROTOTYPE_IDS

    def test_all_prototypes_accept_realistic_core(self) -> None:
        for prototype_id in self.PROTOTYPES:
            with self.subTest(prototype_id=prototype_id):
                validate_creator_candidate(
                    _realistic_candidate(prototype_id),
                    assigned_prototype_id=prototype_id,
                    prototype_display_name=require_prototype(prototype_id).display_name,
                    strategy_foundation=_strategy(),
                )


class TestStructuralRepairAggregation(unittest.TestCase):
    def test_server_derived_fields_excluded_from_repair_list(self) -> None:
        cand = _realistic_candidate("closest")
        cand.pop("closestApplication", None)
        errors = collect_creator_structural_errors(
            cand,
            assigned_prototype_id="closest",
            prototype_display_name="Closest",
            strategy_foundation=_strategy(),
        )
        joined = " ".join(errors)
        self.assertIn("closestApplication", joined)
        self.assertNotIn("essenceExtreme", joined)

    def test_filter_removes_derived_paths(self) -> None:
        raw = [
            "builder2_creator_validation_failed:essenceExtreme.advantageEssence",
            "builder2_creator_validation_failed:closestApplication",
        ]
        filtered = filter_creator_owned_structural_errors(raw)
        self.assertEqual(len(filtered), 1)
        self.assertIn("closestApplication", filtered[0])


class RealisticMockLLM:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def __call__(self, *, role: str, model: str, prompt: str) -> Dict[str, Any]:
        self.calls.append(role)
        if role == "builder2_strategy":
            return _strategy()
        if role == "builder2_creator":
            prototype_id = DEFAULT_ACTIVE_PROTOTYPE_IDS[0]
            for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                if f"Assigned prototype ID: {pid}" in prompt:
                    prototype_id = pid
                    break
            cand = deepcopy(_realistic_candidate(prototype_id))
            marker = "strategyFoundationId (return exactly): "
            if marker in prompt:
                start = prompt.index(marker) + len(marker)
                cand["strategyFoundationId"] = prompt[start:].split("\n", 1)[0].strip().strip('"').strip("'")
            return cand
        if role == "builder2_judge":
            from tests.test_builder2_tournament import _judgment

            candidate_id = "unknown"
            for token in prompt.split():
                if token.startswith("cand-"):
                    candidate_id = token.strip()
                    break
            return _judgment(candidate_id, total_hint=80)
        if role == "builder2_winner":
            from tests.test_builder2_tournament import _winner_plan_from_prompt

            return _winner_plan_from_prompt(prompt)
        raise AssertionError(role)


class TestHappyPathRealistic(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS),
            "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
        },
        clear=True,
    )
    def test_fourteen_call_realistic_happy_path(self) -> None:
        llm = RealisticMockLLM()
        run_builder2_tournament(
            job_id="job-realistic-core",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            rng_seed="seed-realistic",
        )
        state = load_tournament_state("job-realistic-core")
        assert state is not None
        metrics = state.get("metrics") or {}
        self.assertEqual(metrics.get("creatorCalls"), 6)
        self.assertEqual(metrics.get("creatorRepairCalls"), 0)
        self.assertEqual(metrics.get("judgeCalls"), 6)
        self.assertEqual(metrics.get("totalReasoningCalls"), 14)


class TestCircuitBreaker(unittest.TestCase):
    def test_two_post_repair_shared_field_trips_breaker(self) -> None:
        state: Dict[str, Any] = {"jobId": "job-breaker"}
        record_creator_contract_failure(
            state,
            prototype_id="closest",
            error_paths=["strategyFoundationId"],
            after_repair=True,
        )
        record_creator_contract_failure(
            state,
            prototype_id="think_small",
            error_paths=["strategyFoundationId"],
            after_repair=True,
        )
        self.assertTrue(is_creator_contract_circuit_breaker_tripped(state))
        with self.assertRaises(Builder2TournamentError) as ctx:
            assert_creator_contract_available(state)
        self.assertIn(SYSTEMIC_FAILURE_CODE, str(ctx.exception))

    def test_single_prototype_failure_does_not_trip(self) -> None:
        state: Dict[str, Any] = {}
        record_creator_contract_failure(
            state,
            prototype_id="closest",
            error_paths=["closestApplication.admittedGap"],
            after_repair=True,
        )
        self.assertFalse(is_creator_contract_circuit_breaker_tripped(state))


class TestRecoveryCostSafety(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_recovery()
        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ENABLED": "true",
            "BUILDER2_RECOVERY_MAX_AUTOMATIC_ATTEMPTS": "2",
        },
        clear=True,
    )
    def test_third_recovery_blocked(self) -> None:
        job_id = "job-recovery-limit"
        set_memory_job_hash(job_id, {"status": "interrupted", "error": "worker_shutdown_during_job"})
        register_recoverable_job(job_id)
        save_tournament_state(
            job_id,
            new_tournament_state(
                job_id=job_id,
                language="en",
                active_prototype_ids=["closest"],
                random_seed="s",
            ),
        )
        self.assertTrue(requeue_recoverable_job(job_id))
        clear_job_queued(job_id)
        self.assertTrue(requeue_recoverable_job(job_id))
        clear_job_queued(job_id)
        self.assertFalse(requeue_recoverable_job(job_id))

    def test_terminal_failed_job_not_requeued(self) -> None:
        job_id = "job-terminal"
        state = new_tournament_state(job_id=job_id, language="en", active_prototype_ids=["closest"], random_seed="s")
        state["status"] = "failed"
        save_tournament_state(job_id, state)
        set_memory_job_hash(job_id, {"status": "failed"})
        register_recoverable_job(job_id)
        self.assertFalse(requeue_recoverable_job(job_id))


class TestPreflightMode(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_preflight_never_calls_judge(self) -> None:
        llm = RealisticMockLLM()
        report = run_builder2_creator_preflight(
            job_id="job-preflight",
            product_name="Product",
            product_description="desc",
            content_language="en",
            llm_client=llm,
            prototype_id="closest",
        )
        self.assertTrue(report.get("preflight"))
        self.assertEqual(report.get("validationStatus"), "accepted")
        self.assertNotIn("builder2_judge", llm.calls)


class TestFieldOwnershipAudit(unittest.TestCase):
    def test_key_fields_classified(self) -> None:
        from engine.builder2_creator_core_contract import FIELD_OWNERSHIP

        self.assertEqual(FIELD_OWNERSHIP["visualMechanism"], CREATOR_OWNERSHIP_CREATOR_CORE)
        self.assertEqual(FIELD_OWNERSHIP["essenceExtreme"], CREATOR_OWNERSHIP_SERVER_DERIVED)
        self.assertEqual(FIELD_OWNERSHIP["closestApplication"], CREATOR_OWNERSHIP_CREATOR_CORE)


if __name__ == "__main__":
    unittest.main()
