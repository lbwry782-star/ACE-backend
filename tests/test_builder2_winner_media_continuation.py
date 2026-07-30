"""
Builder2 Winner → media continuation contract tests — mocks only.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume
from engine.builder2_incomplete_tournament_resume import run_incomplete_tournament_resume
from engine.builder2_media_resume import collect_media_resume_missing_paths, run_one_media_resume
from engine.builder2_media_resume_contract_inspect import inspect_media_resume_contract
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_single_slogan_contract import BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION, stamp_single_slogan_contract
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, save_tournament_state
from engine.builder2_winner_persistence import (
    WINNER_DEVELOPMENT_SOURCE_NORMAL,
    WINNER_DEVELOPMENT_SOURCE_OFFLINE_SALVAGE,
    collect_winner_media_continuation_missing_fields,
    is_valid_persisted_winner_development,
    is_winner_media_continuation_ready,
    persist_accepted_winner_development_for_media,
    reload_verified_winner_media_state,
)
from engine.builder2_winner_offline_salvage import attempt_offline_winner_development_salvage
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY, process_winner_development_response
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt
from tests.test_builder2_winner_offline_salvage import (
    _current_job_shaped_state,
    _judgment_requiring_verbal_copy,
    _parsed_winner_plan_omit,
    _winning_card_winner_id,
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _think_small_ledgers(state: Dict[str, Any]) -> None:
    from engine.builder2_creator_slogan_repair_patch import reconcile_slogan_repair_call_ledger
    from engine.builder2_creator_semantic_bridge_repair_patch import reconcile_semantic_bridge_repair_call_ledger

    ensure_metrics(state)
    state["metrics"]["creatorCalls"] = 1
    state["metrics"]["creatorRepairCalls"] = 1
    state["metrics"]["creatorSemanticBridgeRepairCalls"] = 1
    state["metrics"]["judgeCalls"] = 6
    slogan = reconcile_slogan_repair_call_ledger(state, prototype_id="think_small")
    slogan["persistedCreatorNormalCalls"] = 1
    slogan["persistedCreatorRepairCalls"] = 1
    slogan["canonicalCreatorNormalCalls"] = 1
    slogan["canonicalCreatorRepairCalls"] = 1
    semantic = reconcile_semantic_bridge_repair_call_ledger(state, prototype_id="think_small")
    semantic["paidDispatchCount"] = 1
    semantic["accepted"] = True
    for _judgment_id, record in (state.get("judgments") or {}).items():
        candidate_id = _clean(record.get("candidateId"))
        candidate = (state.get("candidates") or {}).get(candidate_id) or {}
        if _clean(candidate.get("prototypeId")) == "think_small":
            record["accepted"] = True


class TestCanonicalWinnerMediaPersistence(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _accepted_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        candidate = winner_rec["creatorOutput"]
        judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        from engine.builder2_winner_preservation_contract import build_server_owned_winner_source_reference

        source = build_server_owned_winner_source_reference(
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            candidate_id=winner_id,
        )
        return process_winner_development_response(
            deepcopy(state[PARSED_WINNER_RESPONSE_KEY]["parsed"]),
            source_reference=source,
            winning_candidate=candidate,
            winning_judgment=judgment,
            tournament_state=state,
        )

    def test_offline_salvage_writes_media_required_fields(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        save_tournament_state(state["jobId"], state)
        attempt_offline_winner_development_salvage(
            state,
            winner_candidate_id=winner_id,
            prototype_id="winning_card",
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=winner_rec["creatorOutput"],
            winning_judgment=state["judgments"][winner_rec["judgmentId"]]["judgment"],
            job_id=state["jobId"],
            tournament_id=state["tournamentId"],
        )
        save_tournament_state(state["jobId"], state)
        reloaded = reload_verified_winner_media_state(state["jobId"])
        self.assertTrue(reloaded.get("mediaContinuationRequired"))
        self.assertTrue(reloaded.get("winnerDevelopmentAccepted"))
        self.assertEqual(reloaded.get("winnerDevelopmentSource"), WINNER_DEVELOPMENT_SOURCE_OFFLINE_SALVAGE)
        self.assertEqual(reloaded.get("winnerDevelopmentPrototypeId"), "winning_card")
        self.assertEqual(reloaded.get("winnerDevelopmentCandidateId"), winner_id)
        self.assertEqual(collect_media_resume_missing_paths(reloaded), [])
        self.assertTrue(reloaded.get("winnerDevelopmentFailureResolved"))
        self.assertIsInstance(reloaded.get("winnerDevelopmentFailure"), dict)

    def test_normal_and_offline_use_same_helper(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        plan = self._accepted_plan(state)
        persist_accepted_winner_development_for_media(
            state,
            candidate_id=winner_id,
            prototype_id="winning_card",
            winner_plan=plan,
            source=WINNER_DEVELOPMENT_SOURCE_NORMAL,
            job_id=state["jobId"],
            save=True,
        )
        normal = reload_verified_winner_media_state(state["jobId"])
        self.assertEqual(normal.get("winnerDevelopmentSource"), WINNER_DEVELOPMENT_SOURCE_NORMAL)

        state2 = _current_job_shaped_state()
        state2["jobId"] = "job-media-handoff-offline"
        save_tournament_state(state2["jobId"], state2)
        winner_id2 = _winning_card_winner_id(state2)
        winner_rec2 = state2["candidates"][winner_id2]
        attempt_offline_winner_development_salvage(
            state2,
            winner_candidate_id=winner_id2,
            prototype_id="winning_card",
            strategy_foundation=state2["strategyFoundation"],
            winning_candidate=winner_rec2["creatorOutput"],
            winning_judgment=state2["judgments"][winner_rec2["judgmentId"]]["judgment"],
            job_id=state2["jobId"],
        )
        save_tournament_state(state2["jobId"], state2)
        offline = reload_verified_winner_media_state(state2["jobId"])
        for key in (
            "mediaContinuationRequired",
            "winnerDevelopmentAccepted",
            "winnerDevelopmentPrototypeId",
            "winnerDevelopmentCandidateId",
            "winnerDevelopmentPlanFingerprint",
        ):
            self.assertIn(key, offline)
            self.assertIn(key, normal)

    def test_media_preflight_passes_for_production_shaped_fixture(self) -> None:
        state = _current_job_shaped_state()
        _think_small_ledgers(state)
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        save_tournament_state(state["jobId"], state)
        attempt_offline_winner_development_salvage(
            state,
            winner_candidate_id=winner_id,
            prototype_id="winning_card",
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=winner_rec["creatorOutput"],
            winning_judgment=state["judgments"][winner_rec["judgmentId"]]["judgment"],
            job_id=state["jobId"],
        )
        save_tournament_state(state["jobId"], state)
        reloaded = reload_verified_winner_media_state(state["jobId"])
        inspect_report = inspect_media_resume_contract(reloaded)
        self.assertTrue(inspect_report["mediaResumeReady"])
        self.assertTrue(inspect_report["winnerDevelopmentAccepted"])
        self.assertEqual(inspect_report["acceptedCreatorsCount"], 6)
        self.assertEqual(inspect_report["acceptedJudgmentsCount"], 6)
        self.assertEqual(inspect_report["missingPrototypeIds"], [])

    @patch("engine.builder2_media_resume.run_one_media_resume")
    def test_incomplete_resume_reloads_before_media(self, media_mock) -> None:
        state = _current_job_shaped_state()
        _think_small_ledgers(state)
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        save_tournament_state(state["jobId"], state)
        attempt_offline_winner_development_salvage(
            state,
            winner_candidate_id=winner_id,
            prototype_id="winning_card",
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=winner_rec["creatorOutput"],
            winning_judgment=state["judgments"][winner_rec["judgmentId"]]["judgment"],
            job_id=state["jobId"],
        )
        save_tournament_state(state["jobId"], state)

        def _media(**kwargs: Any) -> Dict[str, Any]:
            passed_state = kwargs.get("tournament_state") or {}
            self.assertTrue(is_winner_media_continuation_ready(passed_state))
            self.assertEqual(collect_winner_media_continuation_missing_fields(passed_state), [])
            return {"ok": True, "runwaySubmissionCalls": 0, "jobCompleted": False}

        media_mock.side_effect = _media

        with patch("engine.builder2_incomplete_tournament_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_incomplete_tournament_resume.run_controlled_complete_ad_reasoning_resume"
        ) as reasoning_mock:
            reasoning_mock.return_value = {
                "ok": True,
                "finalWinnerCandidateId": winner_id,
                "winnerDevelopmentOfflineSalvageAccepted": True,
                "winnerDevelopmentAccepted": True,
            }
            report = run_incomplete_tournament_resume(
                job_id=state["jobId"],
                tournament_state=load_tournament_state(state["jobId"]),
                run_media=True,
            )
        self.assertTrue(report["winnerDevelopmentAccepted"])
        self.assertTrue(report["winnerDevelopmentOfflineSalvageAccepted"])
        self.assertEqual(report["thinkSmallNormalCreatorCalls"], 1)
        self.assertEqual(report["thinkSmallRepairCalls"], 1)
        self.assertEqual(report["totalSemanticBridgeRepairCalls"], 1)
        self.assertEqual(report["thinkSmallJudgeCalls"], 1)

    def test_inspector_is_read_only(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        plan = self._accepted_plan(state)
        persist_accepted_winner_development_for_media(
            state,
            candidate_id=winner_id,
            prototype_id="winning_card",
            winner_plan=plan,
            job_id=state["jobId"],
            save=True,
        )
        before = load_tournament_state(state["jobId"])
        report = inspect_media_resume_contract(before or {})
        after = load_tournament_state(state["jobId"])
        self.assertEqual(before, after)
        self.assertFalse(report["stateMutated"])
        self.assertEqual(report["paidCalls"], 0)


if __name__ == "__main__":
    unittest.main()
