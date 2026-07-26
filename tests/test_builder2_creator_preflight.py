"""
Builder2 Creator preflight isolation and prototype schema parity tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder2_creator import validate_creator_candidate
from engine.builder2_creator_core_contract import (
    PROTOTYPE_APPLICATION_CHILD_FIELDS,
    build_creator_required_keys_prompt_text,
    prototype_application_child_fields,
)
from engine.builder2_creator_normalization import normalize_creator_candidate
from engine.builder2_creator_preflight import (
    PREFLIGHT_ISOLATION_ERROR,
    PreflightIsolationGuard,
    creator_preflight_only_enabled,
    run_one_isolated_creator_preflight,
)
from engine.builder2_methodology_validation import validate_creator_methodology
from engine.builder2_prototypes import require_prototype
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_prompts import build_creator_prompt
from engine.builder2_tournament_recovery import (
    clear_job_queued,
    disable_memory_recovery,
    enable_memory_recovery,
    requeue_recoverable_job,
    register_recoverable_job,
    scan_and_requeue_recoverable_jobs,
    set_memory_job_hash,
    _load_recovery_meta,
    _migrate_legacy_recovery_meta,
)
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, new_tournament_state, save_tournament_state
from tests.test_builder2_tournament import _candidate, _strategy


HISTORICAL_JOB_ID = "beff536b-8605-4fde-a326-66bdf18b7acc"


class PreflightMockLLM:
    calls: List[str] = []

    def __call__(self, *, role: str, model: str, prompt: str) -> Dict[str, Any]:
        PreflightMockLLM.calls.append(role)
        if role == "builder2_strategy":
            return _strategy(language="he")
        if role == "builder2_creator":
            cand = _candidate("think_small")
            cand.pop("thinkSmallApplication", None)
            cand["thinkSmallApplication"] = {
                "realWeakness": "Limited physical footprint",
                "realWeaknessEvidence": "Buyers notice the small size first",
                "acceptance": "The ad accepts the small size openly",
                "reframe": "Small size becomes maneuverability",
                "advantageCreated": "Agility becomes the strategic payoff",
            }
            marker = "strategyFoundationId (return exactly): "
            if marker in prompt:
                start = prompt.index(marker) + len(marker)
                cand["strategyFoundationId"] = prompt[start:].split("\n", 1)[0].strip().strip('"').strip("'")
            return cand
        if role == "builder2_judge":
            raise AssertionError("judge must not run in preflight")
        if role == "builder2_winner":
            raise AssertionError("winner must not run in preflight")
        raise AssertionError(role)


class TestPreflightIsolation(unittest.TestCase):
    def setUp(self) -> None:
        PreflightMockLLM.calls = []
        PreflightIsolationGuard.begin()

    def tearDown(self) -> None:
        PreflightIsolationGuard.end()

    @patch.dict(os.environ, {"BUILDER2_CREATOR_PREFLIGHT_ONLY": "true"}, clear=True)
    def test_preflight_mode_skips_recovery_scan(self) -> None:
        enable_memory_recovery()
        register_recoverable_job(HISTORICAL_JOB_ID)
        try:
            self.assertEqual(scan_and_requeue_recoverable_jobs(), [])
        finally:
            disable_memory_recovery()

    def test_isolation_assertion_blocks_paid_call(self) -> None:
        PreflightIsolationGuard.recovery_scan_performed = True
        with self.assertRaises(Builder2TournamentError) as ctx:
            PreflightIsolationGuard.assert_safe_before_paid_call()
        self.assertIn(PREFLIGHT_ISOLATION_ERROR, str(ctx.exception))

    def test_one_strategy_one_creator_only(self) -> None:
        report = run_one_isolated_creator_preflight(
            product_name="Product",
            product_description="desc",
            content_language="he",
            prototype_id="think_small",
            llm_client=PreflightMockLLM(),
        )
        self.assertTrue(report.get("strategyAccepted"))
        self.assertTrue(report.get("creatorAccepted"))
        self.assertEqual(PreflightMockLLM.calls.count("builder2_strategy"), 1)
        self.assertEqual(PreflightMockLLM.calls.count("builder2_creator"), 1)
        self.assertEqual(report.get("creatorNormalCalls"), 1)
        self.assertEqual(report.get("creatorRepairCalls"), 0)
        self.assertEqual(report.get("judgeCalls"), 0)
        self.assertEqual(report.get("winnerCalls"), 0)
        self.assertEqual(report.get("runwayCalls"), 0)
        self.assertEqual(report.get("ffmpegCalls"), 0)

    def test_preflight_report_fields_only(self) -> None:
        report = run_one_isolated_creator_preflight(
            product_name="Product",
            product_description="desc",
            content_language="he",
            prototype_id="think_small",
            llm_client=PreflightMockLLM(),
        )
        allowed = {
            "preflightId",
            "prototypeId",
            "strategyAccepted",
            "creatorAccepted",
            "creatorNormalCalls",
            "creatorRepairCalls",
            "creatorRetryCalls",
            "validationFailurePaths",
            "serverDerivedFieldPaths",
            "judgeCalls",
            "winnerCalls",
            "runwayCalls",
            "ffmpegCalls",
            "failureReason",
            "ok",
            "candidateId",
        }
        self.assertTrue(set(report.keys()).issubset(allowed))


class TestThinkSmallParity(unittest.TestCase):
    def test_canonical_fields_in_prompt(self) -> None:
        prompt = build_creator_prompt(
            product_name="Product",
            product_description="desc",
            language="he",
            strategy_foundation=_strategy(),
            prototype=require_prototype("think_small"),
            candidate_id="cand-1",
            attempt_number=1,
            runway_mode="image_to_video",
        )
        for field in prototype_application_child_fields("think_small"):
            self.assertIn(field, prompt)

    def test_alias_normalization(self) -> None:
        cand = _candidate("think_small")
        cand["thinkSmallApplication"] = {
            "realWeakness": "Limited physical footprint",
            "realWeaknessEvidence": "Buyers notice the small size first",
            "acceptance": "The ad accepts the small size openly",
            "reframe": "Small size becomes maneuverability",
            "advantageCreated": "Agility becomes the strategic payoff",
        }
        normalized, resolved = normalize_creator_candidate(
            cand,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=_strategy(),
        )
        app = normalized["thinkSmallApplication"]
        self.assertTrue(app.get("evidenceTheWeaknessIsReal"))
        self.assertTrue(app.get("acceptanceRatherThanDenial"))
        self.assertTrue(app.get("reframing"))
        self.assertTrue(app.get("relativeAdvantageCreated"))
        self.assertIn("thinkSmallApplication.evidenceTheWeaknessIsReal", resolved)
        validate_creator_candidate(
            normalized,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=_strategy(),
        )

    def test_missing_semantic_evidence_still_fails(self) -> None:
        cand = _candidate("think_small")
        cand["thinkSmallApplication"] = {
            "realWeakness": "Limited physical footprint",
            "evidenceTheWeaknessIsReal": "",
            "acceptanceRatherThanDenial": "accepted",
            "reframing": "reframed",
            "relativeAdvantageCreated": "advantage",
        }
        with self.assertRaises(Builder2TournamentError):
            validate_creator_methodology(cand, assigned_prototype_id="think_small", strategy_foundation=_strategy())


class TestAllPrototypeParity(unittest.TestCase):
    PROTOTYPES = DEFAULT_ACTIVE_PROTOTYPE_IDS

    def test_validator_children_in_prompt(self) -> None:
        for prototype_id in self.PROTOTYPES:
            with self.subTest(prototype_id=prototype_id):
                required_text = build_creator_required_keys_prompt_text(prototype_id=prototype_id)
                for child in PROTOTYPE_APPLICATION_CHILD_FIELDS.get(prototype_id, ()):
                    self.assertIn(child, required_text)


class TestHistoricalRecoveryMigration(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        enable_memory_recovery()

    def tearDown(self) -> None:
        disable_memory_recovery()
        disable_memory_store()

    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_ENABLED": "true", "BUILDER2_RECOVERY_MAX_AUTOMATIC_ATTEMPTS": "2"},
        clear=True,
    )
    def test_legacy_registry_gets_persisted_attempt_floor(self) -> None:
        from engine.builder2_tournament_recovery import _memory_recovery_meta

        job_id = HISTORICAL_JOB_ID
        set_memory_job_hash(job_id, {"status": "interrupted", "error": "worker_shutdown_during_job"})
        save_tournament_state(
            job_id,
            new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="legacy"),
        )
        register_recoverable_job(job_id)
        _memory_recovery_meta.pop(job_id, None)
        meta = _migrate_legacy_recovery_meta(job_id)
        self.assertGreaterEqual(int(meta.get("recoveryAttemptCount") or 0), 1)
        self.assertTrue(meta.get("legacyMigrated"))
        self.assertFalse(requeue_recoverable_job(job_id))

    @patch.dict(
        os.environ,
        {
            "BUILDER2_TOURNAMENT_ENABLED": "true",
            "BUILDER2_RECOVERY_MAX_AUTOMATIC_ATTEMPTS": "1",
            "BUILDER2_CREATOR_PREFLIGHT_ONLY": "true",
        },
        clear=True,
    )
    def test_preflight_does_not_consume_recovery_job(self) -> None:
        job_id = HISTORICAL_JOB_ID
        set_memory_job_hash(job_id, {"status": "interrupted", "error": "worker_shutdown_during_job"})
        save_tournament_state(
            job_id,
            new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="legacy"),
        )
        register_recoverable_job(job_id)
        self.assertEqual(scan_and_requeue_recoverable_jobs(), [])
        self.assertFalse(requeue_recoverable_job(job_id))

    @patch.dict(
        os.environ,
        {"BUILDER2_TOURNAMENT_ENABLED": "true", "BUILDER2_RECOVERY_MAX_AUTOMATIC_ATTEMPTS": "2"},
        clear=True,
    )
    def test_exhausted_historical_record_removed(self) -> None:
        job_id = "job-historical-exhaust"
        set_memory_job_hash(job_id, {"status": "interrupted"})
        save_tournament_state(
            job_id,
            new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["closest"], random_seed="legacy"),
        )
        register_recoverable_job(job_id)
        self.assertTrue(requeue_recoverable_job(job_id))
        clear_job_queued(job_id)
        self.assertTrue(requeue_recoverable_job(job_id))
        clear_job_queued(job_id)
        self.assertFalse(requeue_recoverable_job(job_id))
        meta = _load_recovery_meta(job_id)
        self.assertEqual(meta.get("recoveryTerminalReason"), "recovery_exhausted")


if __name__ == "__main__":
    unittest.main()
