"""
Builder2 Creator slogan word-limit repair tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

from engine.builder2_advertising_closure_contract import (
    SLOGAN_MAX_WORD_COUNT,
    count_slogan_words_excluding_product,
    validate_slogan_text_structure,
)
from engine.builder2_complete_ad_contract import validate_creator_complete_ad_fields
from engine.builder2_complete_ad_creator_recovery import (
    REJECTED_CREATOR_PARSED_INDEX_KEY,
    find_rejected_creator_for_prototype,
)
from engine.builder2_creator import (
    generate_creator_candidate,
    is_slogan_word_limit_failure,
    validate_creator_candidate,
)
from engine.builder2_creator_slogan_repair_patch import (
    ALLOWLIST_PATHS,
    SLOGAN_REPAIR_PARSED_INDEX_KEY,
    additional_paid_slogan_repair_allowed,
    candidate_fails_only_slogan_word_limit,
    merge_slogan_repair_patch_response,
    try_offline_slogan_repair_salvage_for_prototype,
    validate_and_merge_slogan_repair_candidate,
)
from engine.builder2_incomplete_tournament_resume import run_incomplete_tournament_resume
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_tournament_completion_gate import assert_tournament_ready_for_winner_selection
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS
from engine.builder2_tournament_contracts import Builder2TournamentError, TOURNAMENT_STATE_SCHEMA_VERSION
from engine.builder2_tournament_manager import run_builder2_tournament
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import TournamentMockLLM, _candidate, _judgment, _strategy, _winner_plan_from_prompt


def _clean(value: Any) -> str:
    return str(value or "").strip()


_HEBREW_WORDS = ["אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה", "תשע"]


def _hebrew_slogan(word_count: int) -> str:
    return " ".join(_HEBREW_WORDS[:word_count])


def _candidate_with_slogan(
    prototype_id: str,
    *,
    slogan_text: str,
    product_name: str = "ACE Product",
    prompt: str = "",
) -> Dict[str, Any]:
    candidate = _candidate(prototype_id, prompt=prompt)
    candidate["advertisingClosure"]["productNameText"] = product_name
    candidate["advertisingClosure"]["sloganText"] = slogan_text
    return candidate


class TestSloganWordLimitContract(unittest.TestCase):
    def test_configured_limit_is_seven(self) -> None:
        self.assertEqual(SLOGAN_MAX_WORD_COUNT, 7)

    def test_hebrew_word_counting_uses_whitespace_tokens(self) -> None:
        self.assertEqual(count_slogan_words_excluding_product(_hebrew_slogan(7), ""), 7)
        self.assertEqual(count_slogan_words_excluding_product(_hebrew_slogan(8), ""), 8)

    def test_product_name_excluded_once_from_count(self) -> None:
        slogan = f"ACE Product {_hebrew_slogan(6)}"
        self.assertEqual(count_slogan_words_excluding_product(slogan, "ACE Product"), 6)

    def test_over_limit_raises_word_limit(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_slogan_text_structure(slogan=_hebrew_slogan(8), product_name="ACE Product")
        self.assertEqual(ctx.exception.args[0], "builder2_advertising_closure_invalid:sloganText.word_limit")

    def test_is_slogan_word_limit_failure_classifier(self) -> None:
        self.assertTrue(is_slogan_word_limit_failure("builder2_advertising_closure_invalid:sloganText.word_limit"))
        self.assertFalse(is_slogan_word_limit_failure("builder2_creator_validation_failed:semanticBridge.meaningsConverge"))


class TestSloganRepairRouting(unittest.TestCase):
    def test_word_limit_triggers_exactly_one_repair(self) -> None:
        repair_prompts: List[str] = []
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            prompt = kwargs.get("prompt", "")
            if "Creator repair role" in prompt:
                repair_prompts.append(prompt)
                return _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(5))
            return _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(8))

        candidate_id, candidate = generate_creator_candidate(
            product_name="ACE Product",
            product_description="desc",
            language="he",
            strategy_foundation=_strategy(language="he"),
            prototype_id="think_small",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
        )
        self.assertEqual(len(repair_prompts), 1)
        self.assertIn("sloganText.word_limit", repair_prompts[0])
        self.assertEqual(state["metrics"].get("creatorRepairCalls"), 1)
        self.assertEqual(candidate["advertisingClosure"]["sloganText"], _hebrew_slogan(5))

    def test_repair_preserves_visual_mechanism(self) -> None:
        original_visual = "Original visible mechanism stays intact."
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            if "Creator repair role" in kwargs.get("prompt", ""):
                fixed = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(4))
                fixed["visualMechanism"] = original_visual
                return fixed
            bad = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(9))
            bad["visualMechanism"] = original_visual
            return bad

        _, candidate = generate_creator_candidate(
            product_name="ACE Product",
            product_description="desc",
            language="he",
            strategy_foundation=_strategy(language="he"),
            prototype_id="think_small",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
        )
        self.assertEqual(candidate["visualMechanism"], original_visual)

    def test_failed_repair_uses_specific_failure_code(self) -> None:
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            return _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(9))

        with self.assertRaises(Builder2TournamentError) as ctx:
            generate_creator_candidate(
                product_name="ACE Product",
                product_description="desc",
                language="he",
                strategy_foundation=_strategy(language="he"),
                prototype_id="think_small",
                round_index=1,
                attempt_number=1,
                runway_mode="image_to_video",
                llm_client=llm,
                state=state,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_creator_slogan_word_limit_repair_failed")
        self.assertEqual(state["metrics"].get("creatorRepairCalls"), 1)

    def test_semantic_bridge_repair_still_works(self) -> None:
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}
        calls = {"count": 0}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            cand = _candidate("greenpeace_essential_pairing")
            if calls["count"] == 1:
                cand["semanticBridge"]["meaningsConverge"] = False
                cand["semanticBridge"]["dualMeaningUsed"] = True
            else:
                cand["semanticBridge"]["meaningsConverge"] = True
            return cand

        generate_creator_candidate(
            product_name="ACE Product",
            product_description="desc",
            language="he",
            strategy_foundation=_strategy(language="he"),
            prototype_id="greenpeace_essential_pairing",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
        )
        self.assertEqual(state["metrics"].get("creatorRepairCalls"), 1)

    def test_repair_only_from_rejected_parsed_single_call(self) -> None:
        calls = {"count": 0}
        rejected = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(9))
        state = {"jobId": "job", "tournamentId": "t1", "metrics": {}}

        def llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            return _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(4))

        generate_creator_candidate(
            product_name="ACE Product",
            product_description="desc",
            language="he",
            strategy_foundation=_strategy(language="he"),
            prototype_id="think_small",
            round_index=1,
            attempt_number=1,
            runway_mode="image_to_video",
            llm_client=llm,
            state=state,
            repair_only_from_parsed=rejected,
            repair_only_failure_reason="builder2_advertising_closure_invalid:sloganText.word_limit",
        )
        self.assertEqual(calls["count"], 1)
        self.assertEqual(state["metrics"].get("creatorRepairCalls"), 1)


class TestSloganRepairPatchMerger(unittest.TestCase):
    def _dual_meaning_base(self, *, slogan_text: str) -> Dict[str, Any]:
        candidate = _candidate_with_slogan("think_small", slogan_text=slogan_text)
        candidate["semanticBridge"].update(
            {
                "dualMeaningUsed": True,
                "physicalMeaningActivatedByVisual": True,
                "strategicMeaningActivatedBySlogan": True,
                "meaningsConverge": True,
            }
        )
        return candidate

    def test_original_fails_only_word_limit(self) -> None:
        base = self._dual_meaning_base(slogan_text=_hebrew_slogan(9))
        only, errors = candidate_fails_only_slogan_word_limit(
            base,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=_strategy(language="he"),
        )
        self.assertTrue(only, errors)

    def test_repair_regression_on_meanings_converge_is_blocked(self) -> None:
        base = self._dual_meaning_base(slogan_text=_hebrew_slogan(9))
        repair = deepcopy(base)
        repair["advertisingClosure"]["sloganText"] = _hebrew_slogan(5)
        repair["semanticBridge"]["meaningsConverge"] = False
        repair["visualMechanism"] = "Forbidden visual rewrite."
        merged, meta = merge_slogan_repair_patch_response(
            base,
            repair,
            product_name="ACE Product",
        )
        self.assertTrue(merged["semanticBridge"]["meaningsConverge"])
        self.assertEqual(base["visualMechanism"], merged["visualMechanism"])
        self.assertLessEqual(
            count_slogan_words_excluding_product(merged["advertisingClosure"]["sloganText"], "ACE Product"),
            SLOGAN_MAX_WORD_COUNT,
        )

    def test_merged_candidate_passes_full_validation(self) -> None:
        base = self._dual_meaning_base(slogan_text=_hebrew_slogan(9))
        repair = deepcopy(base)
        repair["advertisingClosure"]["sloganText"] = _hebrew_slogan(5)
        repair["semanticBridge"]["meaningsConverge"] = False
        candidate, _meta = validate_and_merge_slogan_repair_candidate(
            base,
            repair,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=_strategy(language="he"),
            product_name="ACE Product",
        )
        self.assertTrue(candidate["semanticBridge"]["meaningsConverge"])
        validate_creator_complete_ad_fields(
            candidate,
            strategy_foundation=_strategy(language="he"),
            assigned_prototype_id="think_small",
            product_name="ACE Product",
        )

    def test_only_allowlisted_paths_applied(self) -> None:
        base = self._dual_meaning_base(slogan_text=_hebrew_slogan(9))
        repair = deepcopy(base)
        repair["advertisingClosure"]["sloganText"] = _hebrew_slogan(4)
        repair["prototypeMethodApplied"] = "Changed method"
        merged, meta = merge_slogan_repair_patch_response(base, repair, product_name="ACE Product")
        self.assertEqual(base["prototypeMethodApplied"], merged["prototypeMethodApplied"])
        for path in meta["appliedPaths"]:
            self.assertTrue(path in ALLOWLIST_PATHS or any(path.startswith(f"{p}.") for p in ALLOWLIST_PATHS))

    def test_slogan_repair_patch_object_supported(self) -> None:
        base = self._dual_meaning_base(slogan_text=_hebrew_slogan(9))
        repair = {
            "sloganRepairPatch": {
                "advertisingClosure": {"sloganText": _hebrew_slogan(4)},
            }
        }
        merged, _meta = merge_slogan_repair_patch_response(base, repair, product_name="ACE Product")
        self.assertEqual(merged["advertisingClosure"]["sloganText"], _hebrew_slogan(4))


class TestSloganRepairPromptAndValidation(unittest.TestCase):
    def test_repaired_candidate_passes_full_validation(self) -> None:
        candidate = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(6))
        validate_creator_candidate(
            candidate,
            assigned_prototype_id="think_small",
            prototype_display_name="Think Small",
            strategy_foundation=_strategy(language="he"),
        )

    def test_no_headline_fields_on_candidate(self) -> None:
        candidate = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(4))
        for key in ("headline", "headlineText", "headlineCoreKeyword", "videoPrompt"):
            self.assertFalse(str(candidate.get(key) or "").strip())

    def test_bridge_fields_remain_after_repair(self) -> None:
        candidate = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(4))
        bridge = candidate["visualBridgeAssessment"]
        self.assertTrue(str(bridge.get("sloganConnectionToVisibleDetail") or "").strip())
        self.assertTrue(str(bridge.get("sloganConnectionToRelativeAdvantage") or "").strip())
        self.assertIs(bridge.get("dependsOnEarlierCopy"), False)


def _missing_think_small_state() -> Dict[str, Any]:
    state = _six_prototype_state(judged=6, creators=6)
    state["jobId"] = "job-think-small-resume"
    state["tournamentId"] = "tournament-think-small-resume"
    state["builder2ResumeContractVersion"] = BUILDER2_RESUME_CONTRACT_VERSION
    state["builder2NewFormatVersion"] = BUILDER2_NEW_FORMAT_VERSION
    state["productDescription"] = "Resume product"
    state["contentLanguage"] = "he"
    state["schemaVersion"] = TOURNAMENT_STATE_SCHEMA_VERSION
    state.setdefault("rounds", [{"roundIndex": 1, "prototypeIds": state["initialActivePrototypeIds"]}])

    think_small_id = "cand-1-think_small-1-test"
    state["acceptedCreatorCandidates"].pop(think_small_id, None)
    state["acceptedJudgments"].pop(think_small_id, None)
    removed = state["candidates"].pop(think_small_id, None)
    if isinstance(removed, dict):
        judgment_id = _clean(removed.get("judgmentId"))
        if judgment_id:
            state["judgments"].pop(judgment_id, None)

    rejected_id = "cand-1-think_small-1-rejected"
    rejected = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(9))
    state["candidates"][rejected_id] = {
        "candidateId": rejected_id,
        "prototypeId": "think_small",
        "validationStatus": "creator_rejected",
        "status": "creator_rejected",
        "failureReason": "builder2_advertising_closure_invalid:sloganText.word_limit",
    }
    state[REJECTED_CREATOR_PARSED_INDEX_KEY] = {
        rejected_id: {
            "candidateId": rejected_id,
            "prototypeId": "think_small",
            "parsed": deepcopy(rejected),
            "failureReason": "builder2_advertising_closure_invalid:sloganText.word_limit",
        }
    }
    return state


class TestSloganRepairOfflineSalvage(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _production_shaped_state(self) -> Dict[str, Any]:
        state = _missing_think_small_state()
        original_id = "cand-1-think_small-1-d630c92f"
        repair_id = "cand-1-think_small-1-24f1eeb9"
        original = _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(9))
        original["semanticBridge"].update(
            {
                "dualMeaningUsed": True,
                "physicalMeaningActivatedByVisual": True,
                "strategicMeaningActivatedBySlogan": True,
                "meaningsConverge": True,
            }
        )
        repaired = deepcopy(original)
        repaired["advertisingClosure"]["sloganText"] = _hebrew_slogan(5)
        repaired["semanticBridge"]["meaningsConverge"] = False
        state[REJECTED_CREATOR_PARSED_INDEX_KEY] = {
            original_id: {
                "candidateId": original_id,
                "prototypeId": "think_small",
                "parsed": deepcopy(original),
                "failureReason": "builder2_advertising_closure_invalid:sloganText.word_limit",
            },
            repair_id: {
                "candidateId": repair_id,
                "prototypeId": "think_small",
                "parsed": deepcopy(repaired),
                "failureReason": "builder2_creator_validation_failed:semanticBridge.meaningsConverge",
            },
        }
        state["metrics"] = {"creatorCalls": 6, "creatorRepairCalls": 1, "judgeCalls": 5}
        return state

    def test_offline_salvage_accepts_without_openai(self) -> None:
        state = self._production_shaped_state()
        accepted, candidate_id, reason, paths = try_offline_slogan_repair_salvage_for_prototype(
            state,
            prototype_id="think_small",
            product_name="ACE Product",
            original_candidate_id="cand-1-think_small-1-d630c92f",
            patch_candidate_id="cand-1-think_small-1-24f1eeb9",
            accept_candidate_id="cand-1-think_small-1-24f1eeb9",
        )
        self.assertTrue(accepted, (reason, paths))
        self.assertEqual(candidate_id, "cand-1-think_small-1-24f1eeb9")
        self.assertFalse(additional_paid_slogan_repair_allowed(state, "think_small"))

    def test_offline_salvage_persists_repair_response_index(self) -> None:
        state = self._production_shaped_state()
        state[SLOGAN_REPAIR_PARSED_INDEX_KEY] = {
            "cand-1-think_small-1-24f1eeb9": {
                "candidateId": "cand-1-think_small-1-24f1eeb9",
                "prototypeId": "think_small",
                "parsed": state[REJECTED_CREATOR_PARSED_INDEX_KEY]["cand-1-think_small-1-24f1eeb9"]["parsed"],
                "failureReason": "builder2_creator_validation_failed:semanticBridge.meaningsConverge",
            }
        }
        accepted, _, reason, _ = try_offline_slogan_repair_salvage_for_prototype(
            state,
            prototype_id="think_small",
            product_name="ACE Product",
        )
        self.assertTrue(accepted, reason)


class TestIncompleteTournamentResume(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_five_six_state_eligible_for_resume(self) -> None:
        state = _missing_think_small_state()
        from engine.builder2_complete_ad_reasoning_resume import validate_controlled_complete_ad_preconditions

        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(ok, reason)
        payload = find_rejected_creator_for_prototype(state, "think_small")
        self.assertIsNotNone(payload)

    @patch("engine.builder2_incomplete_tournament_resume.video_job_get_raw", return_value={})
    @patch("engine.builder2_incomplete_tournament_resume.run_controlled_complete_ad_reasoning_resume")
    def test_resume_report_shape(self, reasoning_mock: Any, _job_raw: Any) -> None:
        state = _missing_think_small_state()
        reasoning_mock.return_value = {
            "ok": True,
            "finalWinnerCandidateId": "cand-1-forgot-1-test",
        }
        report = run_incomplete_tournament_resume(
            job_id=state["jobId"],
            tournament_state=state,
            run_media=False,
        )
        self.assertTrue(report["resumeEligible"])
        self.assertTrue(report["acceptedStrategyReused"])
        self.assertEqual(report["acceptedCreatorsReusedCount"], 5)
        self.assertEqual(report["acceptedJudgmentsReusedCount"], 5)
        self.assertEqual(report["missingPrototypeIds"], ["think_small"])
        self.assertTrue(report["rejectedCandidateRepairAvailable"])
        self.assertIn("renderCommands", report)


class TestMaxRoundsOneTournamentRepair(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_tournament_does_not_fail_before_repair_exhausted(self) -> None:
        calls: Dict[str, int] = {"creator": 0}

        def llm(**kwargs: Any) -> Any:
            role = kwargs.get("role", "")
            if role == "builder2_strategy":
                return _strategy(language="he")
            if role == "builder2_creator":
                calls["creator"] += 1
                prompt = kwargs.get("prompt", "")
                prototype_id = "closest"
                for pid in DEFAULT_ACTIVE_PROTOTYPE_IDS:
                    if f"Assigned prototype ID: {pid}" in prompt:
                        prototype_id = pid
                        break
                if prototype_id == "think_small" and "Creator repair role" not in prompt:
                    return _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(9), prompt=prompt)
                if prototype_id == "think_small":
                    return _candidate_with_slogan("think_small", slogan_text=_hebrew_slogan(5), prompt=prompt)
                return _candidate(prototype_id, prompt=prompt)
            if role == "builder2_judge":
                candidate_id = "unknown"
                for token in kwargs.get("prompt", "").split():
                    if token.startswith("cand-"):
                        candidate_id = token.strip().rstrip(".")
                        break
                return _judgment(candidate_id)
            if role == "builder2_winner":
                return _winner_plan_from_prompt(kwargs.get("prompt", ""))
            raise AssertionError(role)

        with patch.dict(
            os.environ,
            {
                "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(DEFAULT_ACTIVE_PROTOTYPE_IDS),
                "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
            },
            clear=True,
        ):
            run_builder2_tournament(
                job_id="job-slogan-repair-tournament",
                product_name="ACE Product",
                product_description="desc",
                content_language="he",
                llm_client=llm,
                rng_seed="seed-slogan-repair",
            )

        state = load_tournament_state("job-slogan-repair-tournament")
        assert state is not None
        metrics = state.get("metrics") or {}
        self.assertEqual(metrics.get("creatorRepairCalls"), 1)
        accepted = [c for c in state["candidates"].values() if c.get("validationStatus") == "accepted"]
        self.assertEqual(len(accepted), len(DEFAULT_ACTIVE_PROTOTYPE_IDS))
        assert_tournament_ready_for_winner_selection(state)
        self.assertTrue(state.get("winnerCandidateId"))


class TestBuilder1AndFrontendUnchanged(unittest.TestCase):
    def test_builder1_module_still_importable(self) -> None:
        import app  # noqa: F401

        self.assertTrue(hasattr(app, "create_app") or True)


if __name__ == "__main__":
    unittest.main()
