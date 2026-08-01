"""
Builder2 advertising-slogan quality — six-prototype Creator/Judge coverage matrix (mocks only).
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

from engine.builder2_advertising_slogan_quality_contract import (
    BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
    CREATOR_SLOGAN_FORMULATION_FIELDS,
    JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS,
    apply_advertising_slogan_eligibility_rules,
    build_default_creator_slogan_formulation,
    validate_creator_advertising_slogan_formulation,
    validate_judge_advertising_slogan_assessment,
    validate_slogan_advertising_quality_deterministic,
)
from engine.builder2_complete_ad_contract import apply_semantic_eligibility_rules
from engine.builder2_creator import validate_creator_candidate
from engine.builder2_creator_core_contract import (
    build_creator_required_keys_prompt_text,
    prototype_application_field,
)
from engine.builder2_creator_slogan_repair_patch import (
    additional_paid_slogan_repair_allowed,
    validate_and_merge_slogan_repair_candidate,
)
from engine.builder2_judge import validate_judge_response
from engine.builder2_judge_core_contract import build_judge_required_keys_prompt_text
from engine.builder2_methodology_validation import validate_judge_methodology
from engine.builder2_new_format_config import NORMAL_REASONING_CALL_BUDGET
from engine.builder2_prototypes import require_prototype
from engine.builder2_single_slogan_contract import builder2_requires_headline_overlay
from engine.builder2_tournament_completion_gate import assert_tournament_ready_for_winner_selection
from engine.builder2_tournament_config import (
    DEFAULT_ACTIVE_PROTOTYPE_IDS,
    DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS,
    resolve_builder2_active_prototype_ids,
    resolve_builder2_tournament_max_rounds,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import run_builder2_tournament, select_global_winner
from engine.builder2_tournament_prompts import build_creator_prompt, build_judge_prompt
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, new_tournament_state
from tests.builder2_methodology_fixtures import (
    logo_policy_creator_extras,
    methodology_judgment_extras,
    methodology_strategy_extras,
    methodology_strategy_identity_for,
)
from tests.test_builder2_creator_slogan_repair import _candidate_with_slogan, _hebrew_slogan
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt

HEBREW_RELATIVE_ADVANTAGE = "שוקולד בסגנון דובאי שמיוצר בישראל"
WEAK_STRATEGIC_PROSE = "סגנון דובאי ממקור ישראלי גלוי"
VALID_ADVERTISING_SLOGAN = "שוקולד דובאי תוצרת ישראל"
PRODUCT_NAME = "דובי"

SIX_PROTOTYPE_IDS: Tuple[str, ...] = DEFAULT_ACTIVE_PROTOTYPE_IDS


def _tournament_state(*, prototype_id: str = "closest") -> Dict[str, Any]:
    state = new_tournament_state(
        job_id=f"job-slogan-{prototype_id}",
        language="he",
        active_prototype_ids=list(SIX_PROTOTYPE_IDS),
        random_seed="seed",
    )
    state["strategyFoundation"] = methodology_strategy_extras()
    return state


def _prototype_candidate(
    prototype_id: str,
    *,
    slogan_text: str,
    relative_advantage: str,
    strategy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    strategy_obj = strategy or methodology_strategy_extras()
    candidate = _candidate(prototype_id)
    candidate.update(methodology_strategy_identity_for(strategy_obj))
    candidate["advertisingClosure"]["productNameText"] = PRODUCT_NAME
    candidate["advertisingClosure"]["sloganText"] = slogan_text
    candidate.update(logo_policy_creator_extras(advertised_entity_name=PRODUCT_NAME))
    candidate["advertisingSloganFormulation"] = build_default_creator_slogan_formulation(
        relative_advantage_source=relative_advantage,
        final_slogan_text=slogan_text,
        transformation_type="contrast" if slogan_text == VALID_ADVERTISING_SLOGAN else "direct_distillation",
        why_advertising="ניסוח פרסומי קצר ולא הסבר אסטרטגי.",
    )
    return candidate


def _hebrew_strategy(*, relative_advantage: str = HEBREW_RELATIVE_ADVANTAGE) -> Dict[str, Any]:
    strategy = methodology_strategy_extras()
    strategy["productNameResolved"] = PRODUCT_NAME
    strategy["language"] = "he"
    strategy["relativeAdvantage"] = {
        "statement": relative_advantage,
        "derivationFromProblem": "הקונה מחפש שוקולד דובאי אמיתי שמיוצר מקומית.",
        "truthBoundary": "לא טוען ייצור בדובאי עצמו.",
        "admitsRelevantGap": True,
    }
    return strategy


class TestSixPrototypePromptCoverage(unittest.TestCase):
    def test_creator_prompt_requires_formulation_for_each_prototype(self) -> None:
        results: Dict[str, bool] = {}
        strategy = _hebrew_strategy()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                prototype = require_prototype(prototype_id)
                prompt = build_creator_prompt(
                    product_name=PRODUCT_NAME,
                    product_description="premium chocolate",
                    language="he",
                    strategy_foundation=strategy,
                    prototype=prototype,
                    candidate_id=f"cand-{prototype_id}",
                    attempt_number=1,
                    runway_mode="text_to_video",
                )
                contract = build_creator_required_keys_prompt_text(prototype_id=prototype_id)
                self.assertIn("advertisingSloganFormulation", contract)
                self.assertIn("advertisingSloganFormulation", prompt)
                self.assertIn(prototype_application_field(prototype_id), prompt)
                results[prototype_id] = True
        self.assertEqual(set(results.keys()), set(SIX_PROTOTYPE_IDS))

    def test_judge_prompt_requires_assessment_for_each_prototype(self) -> None:
        results: Dict[str, bool] = {}
        strategy = _hebrew_strategy()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                prototype = require_prototype(prototype_id)
                candidate = _prototype_candidate(
                    prototype_id,
                    slogan_text=VALID_ADVERTISING_SLOGAN,
                    relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
                    strategy=strategy,
                )
                prompt = build_judge_prompt(
                    product_name=PRODUCT_NAME,
                    product_description="premium chocolate",
                    language="he",
                    strategy_foundation=strategy,
                    prototype=prototype,
                    candidate=candidate,
                    candidate_id=f"cand-{prototype_id}",
                )
                contract = build_judge_required_keys_prompt_text(
                    creator_verbal_decision="available",
                    candidate_id=f"cand-{prototype_id}",
                )
                self.assertIn("advertisingSloganAssessment", contract)
                self.assertIn("advertisingSloganAssessment", prompt)
                self.assertIn("merelyDescriptive", prompt)
                results[prototype_id] = True
        self.assertEqual(set(results.keys()), set(SIX_PROTOTYPE_IDS))


class TestSixPrototypeCreatorValidation(unittest.TestCase):
    def test_creator_formulation_fields_required_for_each_prototype(self) -> None:
        strategy = _hebrew_strategy()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                candidate = _prototype_candidate(
                    prototype_id,
                    slogan_text=VALID_ADVERTISING_SLOGAN,
                    relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
                    strategy=strategy,
                )
                for field in CREATOR_SLOGAN_FORMULATION_FIELDS:
                    self.assertIn(field, candidate["advertisingSloganFormulation"])
                validate_creator_advertising_slogan_formulation(
                    candidate,
                    strategy_foundation=strategy,
                    product_name=PRODUCT_NAME,
                )

    def test_creator_rejects_descriptive_slogan_for_each_prototype(self) -> None:
        strategy = _hebrew_strategy()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                with self.assertRaises(Builder2TournamentError) as ctx:
                    validate_slogan_advertising_quality_deterministic(
                        slogan=WEAK_STRATEGIC_PROSE,
                        product_name=PRODUCT_NAME,
                        relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
                    )
                self.assertIn("strategic_description_markers", ctx.exception.args[0])

    def test_creator_accepts_advertising_slogan_for_each_prototype(self) -> None:
        strategy = _hebrew_strategy()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                candidate = _prototype_candidate(
                    prototype_id,
                    slogan_text=VALID_ADVERTISING_SLOGAN,
                    relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
                    strategy=strategy,
                )
                state = _tournament_state(prototype_id=prototype_id)
                validate_creator_candidate(
                    candidate,
                    assigned_prototype_id=prototype_id,
                    prototype_display_name=require_prototype(prototype_id).display_name,
                    strategy_foundation=strategy,
                    tournament_state=state,
                )
                self.assertEqual(
                    candidate["advertisingSloganFormulation"]["finalSloganText"],
                    VALID_ADVERTISING_SLOGAN,
                )

    def test_prototype_application_fields_remain_required(self) -> None:
        strategy = methodology_strategy_extras()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                candidate = _prototype_candidate(
                    prototype_id,
                    slogan_text="קרוב יותר ממה שחשבת",
                    relative_advantage=strategy["relativeAdvantage"]["statement"],
                    strategy=strategy,
                )
                app_field = prototype_application_field(prototype_id)
                candidate.pop(app_field, None)
                state = _tournament_state(prototype_id=prototype_id)
                with self.assertRaises(Builder2TournamentError):
                    validate_creator_candidate(
                        candidate,
                        assigned_prototype_id=prototype_id,
                        prototype_display_name=require_prototype(prototype_id).display_name,
                        strategy_foundation=strategy,
                        tournament_state=state,
                    )


class TestSixPrototypeJudgeValidation(unittest.TestCase):
    def _judgment_for(self, prototype_id: str, candidate_id: str) -> Dict[str, Any]:
        judgment = _judgment(candidate_id, eligible=True, total_hint=85)
        judgment.update(methodology_judgment_extras(prototype_id=prototype_id))
        return judgment

    def test_judge_assessment_required_for_each_prototype(self) -> None:
        strategy = methodology_strategy_extras()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                candidate = _prototype_candidate(
                    prototype_id,
                    slogan_text=VALID_ADVERTISING_SLOGAN,
                    relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
                    strategy=_hebrew_strategy(),
                )
                judgment = self._judgment_for(prototype_id, f"cand-{prototype_id}")
                for field in JUDGE_SLOGAN_ASSESSMENT_BOOLEAN_FIELDS:
                    self.assertIn(field, judgment["advertisingSloganAssessment"])
                validate_judge_methodology(judgment, candidate=candidate)

    def test_judge_merely_descriptive_ineligible_for_each_prototype(self) -> None:
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                judgment = self._judgment_for(prototype_id, f"cand-{prototype_id}")
                judgment["eligible"] = True
                judgment["advertisingSloganAssessment"]["merelyDescriptive"] = True
                with self.assertRaises(Builder2TournamentError):
                    validate_judge_advertising_slogan_assessment(judgment)
                adjusted = apply_semantic_eligibility_rules(apply_advertising_slogan_eligibility_rules(judgment))
                self.assertFalse(adjusted["eligible"])
                self.assertIn("slogan_merely_descriptive", adjusted["disqualifiers"])

    def test_judge_not_advertising_ineligible_for_each_prototype(self) -> None:
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                judgment = self._judgment_for(prototype_id, f"cand-{prototype_id}")
                judgment["eligible"] = True
                judgment["advertisingSloganAssessment"]["merelyDescriptive"] = False
                judgment["advertisingSloganAssessment"]["soundsLikeAdvertising"] = False
                adjusted = apply_semantic_eligibility_rules(apply_advertising_slogan_eligibility_rules(judgment))
                self.assertFalse(adjusted["eligible"])
                self.assertIn("slogan_not_advertising_copy", adjusted["disqualifiers"])

    def test_judge_valid_advertising_can_remain_eligible_for_each_prototype(self) -> None:
        strategy = _hebrew_strategy()
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                candidate = _prototype_candidate(
                    prototype_id,
                    slogan_text=VALID_ADVERTISING_SLOGAN,
                    relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
                    strategy=strategy,
                )
                judgment = self._judgment_for(prototype_id, f"cand-{prototype_id}")
                parsed, _total, _scores = validate_judge_response(
                    judgment,
                    candidate_id=f"cand-{prototype_id}",
                    candidate=candidate,
                )
                self.assertTrue(parsed["eligible"])


class TestSixPrototypeRepairSynchronization(unittest.TestCase):
    def test_repair_syncs_formulation_for_each_prototype(self) -> None:
        strategy = methodology_strategy_extras()
        repaired_slogan = _hebrew_slogan(6)
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                base = _candidate_with_slogan(prototype_id, slogan_text=_hebrew_slogan(8))
                repair = {
                    "advertisingClosure": {"sloganText": repaired_slogan},
                    "sloganRepairPatch": {"advertisingClosure.sloganText": repaired_slogan},
                }
                candidate, meta = validate_and_merge_slogan_repair_candidate(
                    base,
                    repair,
                    assigned_prototype_id=prototype_id,
                    prototype_display_name=require_prototype(prototype_id).display_name,
                    strategy_foundation=strategy,
                    product_name="ACE Product",
                    candidate_id=f"cand-repair-{prototype_id}",
                    tournament_state=_tournament_state(prototype_id=prototype_id),
                )
                self.assertIn("advertisingClosure.sloganText", meta["appliedPaths"])
                self.assertEqual(candidate["advertisingClosure"]["sloganText"], repaired_slogan)
                self.assertEqual(
                    candidate["advertisingSloganFormulation"]["finalSloganText"],
                    repaired_slogan,
                )

    def test_one_repair_per_prototype_limit_unchanged(self) -> None:
        for prototype_id in SIX_PROTOTYPE_IDS:
            with self.subTest(prototype_id=prototype_id):
                state: Dict[str, Any] = {"metrics": {"creatorRepairCalls": 1}}
                self.assertFalse(additional_paid_slogan_repair_allowed(state, prototype_id))


class TestSixPrototypeTournamentRegression(unittest.TestCase):
    DESCRIPTIVE_PROTOTYPE = "think_small"

    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_six_way_tournament_slogan_quality_and_winner_selection(self) -> None:
        creator_outputs: Dict[str, Dict[str, Any]] = {}
        judge_outputs: Dict[str, Dict[str, Any]] = {}
        llm_calls: List[str] = []

        def llm(**kwargs: Any) -> Dict[str, Any]:
            role = kwargs.get("role")
            prompt = kwargs.get("prompt", "")
            llm_calls.append(str(role))
            if role == "builder2_strategy":
                return _strategy()
            if role == "builder2_creator":
                prototype_id = SIX_PROTOTYPE_IDS[0]
                for pid in SIX_PROTOTYPE_IDS:
                    if f"Assigned prototype ID: {pid}" in prompt:
                        prototype_id = pid
                        break
                candidate = _candidate(prototype_id, prompt=prompt)
                creator_outputs[prototype_id] = candidate
                return candidate
            if role == "builder2_judge":
                candidate_id = "unknown"
                for token in prompt.split():
                    if token.startswith("cand-"):
                        candidate_id = token.strip().strip(",")
                        break
                prototype_id = SIX_PROTOTYPE_IDS[0]
                for pid in SIX_PROTOTYPE_IDS:
                    marker = f'"prototypeId": "{pid}"'
                    if marker in prompt or f'"prototypeId":"{pid}"' in prompt.replace(" ", ""):
                        prototype_id = pid
                        break
                judgment = _judgment(
                    candidate_id,
                    eligible=True,
                    total_hint=90 if prototype_id != self.DESCRIPTIVE_PROTOTYPE else 95,
                )
                judgment["candidateId"] = candidate_id
                judgment.update(methodology_judgment_extras(prototype_id=prototype_id))
                if prototype_id == self.DESCRIPTIVE_PROTOTYPE:
                    judgment["eligible"] = False
                    judgment["disqualifiers"] = ["slogan_merely_descriptive"]
                    judgment["advertisingSloganAssessment"]["merelyDescriptive"] = True
                    judgment["advertisingSloganAssessment"]["soundsLikeAdvertising"] = False
                judge_outputs[prototype_id] = judgment
                return judgment
            if role == "builder2_winner":
                return _winner_plan_from_prompt(prompt)
            raise AssertionError(role)

        custom_prototypes = list(SIX_PROTOTYPE_IDS)
        with patch.dict(
            os.environ,
            {
                "BUILDER2_TOURNAMENT_ACTIVE_PROTOTYPES": ",".join(custom_prototypes),
                "BUILDER2_TOURNAMENT_MAX_ROUNDS": "1",
            },
            clear=True,
        ):
            self.assertEqual(resolve_builder2_active_prototype_ids(), custom_prototypes)
            self.assertEqual(NORMAL_REASONING_CALL_BUDGET, 14)
            with patch.dict(os.environ, {"BUILDER2_TOURNAMENT_MAX_ROUNDS": "2"}, clear=False):
                self.assertEqual(resolve_builder2_tournament_max_rounds(), 2)

            run_builder2_tournament(
                job_id="job-six-slogan-tournament",
                product_name="Product",
                product_description="desc",
                content_language="he",
                llm_client=llm,
                rng_seed="seed-six-slogan",
            )

        state = load_tournament_state("job-six-slogan-tournament")
        assert state is not None
        assert_tournament_ready_for_winner_selection(state)
        self.assertEqual(state.get("advertisingSloganQualityContractVersion"), BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION)

        for prototype_id in custom_prototypes:
            with self.subTest(prototype_id=prototype_id):
                candidate = creator_outputs[prototype_id]
                formulation = candidate.get("advertisingSloganFormulation")
                self.assertIsInstance(formulation, dict)
                self.assertFalse(formulation.get("merelyDescriptive"))
                self.assertTrue(formulation.get("factualGroundingPreserved"))
                self.assertEqual(
                    formulation.get("finalSloganText"),
                    candidate["advertisingClosure"]["sloganText"],
                )
                judgment = judge_outputs[prototype_id]
                self.assertIn("advertisingSloganAssessment", judgment)

        descriptive_records = [
            cand
            for cand in state["candidates"].values()
            if cand.get("prototypeId") == self.DESCRIPTIVE_PROTOTYPE
        ]
        self.assertTrue(descriptive_records)
        self.assertFalse(any(cand.get("eligible") for cand in descriptive_records))

        winner_id = select_global_winner(state)
        winner = state["candidates"][winner_id]
        self.assertNotEqual(winner.get("prototypeId"), self.DESCRIPTIVE_PROTOTYPE)
        self.assertTrue(winner.get("eligible"))

        plan = _winner_plan_from_prompt("")
        self.assertFalse(builder2_requires_headline_overlay(plan=plan, state=state))

        metrics = state.get("metrics") or {}
        self.assertLessEqual(metrics.get("creatorRepairCalls", 0), len(custom_prototypes))
        self.assertEqual(metrics.get("totalReasoningCalls"), 14)
        self.assertEqual(len([role for role in llm_calls if role == "builder2_creator"]), len(custom_prototypes))
        self.assertEqual(len([role for role in llm_calls if role == "builder2_judge"]), len(custom_prototypes))


class TestClosureOverridePreserved(unittest.TestCase):
    def test_closure_only_override_still_zero_reasoning(self) -> None:
        from engine.builder2_closure_copy import resolve_trusted_closure_copy

        job_id = "edb3136e-21d3-419e-86cd-c5d5bda18012"
        corrected = "שוקולד דובאי תוצרת ישראל"
        state: Dict[str, Any] = {
            "jobId": job_id,
            "winnerDevelopmentPlan": {
                "productNameResolved": PRODUCT_NAME,
                "advertisingClosure": {
                    "required": True,
                    "productNameText": PRODUCT_NAME,
                    "sloganText": WEAK_STRATEGIC_PROSE,
                    "language": "he",
                    "presentationMode": "end_card",
                    "durationSeconds": 3.5,
                    "noLogo": True,
                },
            },
            "mediaResume": {"closureSloganOverride": corrected},
        }
        with patch.dict(os.environ, {"BUILDER2_CLOSURE_ONLY_RERENDER_SLOGAN_TEXT": corrected}, clear=False):
            product_name, slogan, language = resolve_trusted_closure_copy(state)
        self.assertEqual(slogan, corrected)
        self.assertEqual(product_name, PRODUCT_NAME)
        self.assertEqual(language, "he")


if __name__ == "__main__":
    unittest.main()
