"""
Builder2 Winner sequence-stage normalization — structured dict stages and Runway handoff.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_winner_development import normalize_winner_plan_for_runway
from engine.builder2_winner_downstream import (
    Builder2WinnerDownstreamError,
    extract_builder2_sequence_stage_text,
    normalize_builder2_winner_downstream,
)
from engine.builder2_winner_persistence import (
    compute_winner_development_plan_fingerprint,
    persist_accepted_winner_development_for_media,
)
from engine.builder2_winner_plan import validate_and_normalize_builder2_winner_plan, validate_builder2_winner_plan
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    process_winner_development_response,
)
from tests.builder2_methodology_fixtures import methodology_winner_extras, single_slogan_contract_extras
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan
from tests.test_builder2_winner_offline_salvage import (
    _current_job_shaped_state,
    _judgment_requiring_verbal_copy,
    _winning_card_winner_id,
)


def _structured_stage(*, description: str, action: str = "", visual: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {"description": description}
    if action:
        payload["action"] = action
    if visual:
        payload["visual"] = visual
    return payload


def _structured_sequence_plan(*, prototype_id: str = "winning_card") -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate(prototype_id)
    plan = _winner_plan(language="he")
    plan.update(
        methodology_winner_extras(
            headline_decision="omit",
            winning_candidate=candidate,
            strategy=strategy,
        )
    )
    plan.update(single_slogan_contract_extras())
    plan["builder2NewFormatVersion"] = BUILDER2_NEW_FORMAT_VERSION
    plan["prototypeId"] = prototype_id
    plan["structureType"] = "continuous_event"
    plan["sceneVariations"] = []
    plan["headlineDecision"] = {"decision": "omit", "reasonSource": "not_required"}
    plan["headlineForm"] = "none"
    plan["headline"] = ""
    plan["headlineCoreKeyword"] = ""
    plan["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
    plan["sequence"] = {
        "beginning": _structured_stage(
            description="The storefront window stays neutral before the proof appears.",
            action="A passerby slows near the display.",
        ),
        "development": _structured_stage(
            description="The window surface becomes the persuasive medium.",
            visual="The display format visibly transforms.",
        ),
        "resolution": _structured_stage(
            description="The transformed medium completes the strategic proof.",
            action="The proof lands in one clear final beat.",
        ),
    }
    plan["preservationReference"] = {
        "strategyFoundationId": strategy.get("strategyFoundationId") or "strategy-test",
        "prototypeId": prototype_id,
        "structureType": candidate.get("structureType"),
        "visualParallelType": candidate.get("visualParallelType"),
        "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
    }
    return plan


class TestExtractBuilder2SequenceStageText(unittest.TestCase):
    def test_plain_string_stage(self) -> None:
        text = extract_builder2_sequence_stage_text("Two people stand apart.", "sequence.beginning")
        self.assertEqual(text, "Two people stand apart.")

    def test_structured_dict_prefers_description(self) -> None:
        stage = {
            "description": "Canonical beat text.",
            "action": "Secondary action detail.",
            "visual": "Secondary visual detail.",
        }
        self.assertEqual(
            extract_builder2_sequence_stage_text(stage, "sequence.development"),
            "Canonical beat text.",
        )

    def test_structured_dict_falls_back_to_scene(self) -> None:
        stage = {"scene": "Visible storefront beat.", "action": "Ignored when scene exists after description miss."}
        self.assertEqual(extract_builder2_sequence_stage_text(stage, "sequence.beginning"), "Visible storefront beat.")

    def test_mixed_historical_and_current_representations(self) -> None:
        sequence = {
            "beginning": "Plain opening beat.",
            "development": {"description": "Structured middle beat."},
            "resolution": {"scene": "Plain-scene resolution beat."},
        }
        texts = [
            extract_builder2_sequence_stage_text(sequence["beginning"], "sequence.beginning"),
            extract_builder2_sequence_stage_text(sequence["development"], "sequence.development"),
            extract_builder2_sequence_stage_text(sequence["resolution"], "sequence.resolution"),
        ]
        self.assertEqual(
            texts,
            ["Plain opening beat.", "Structured middle beat.", "Plain-scene resolution beat."],
        )

    def test_missing_canonical_text_in_dict_rejected(self) -> None:
        with self.assertRaises(Builder2WinnerDownstreamError):
            extract_builder2_sequence_stage_text({"familyId": "nearness"}, "sequence.beginning")

    def test_empty_structured_stage_rejected(self) -> None:
        with self.assertRaises(Builder2WinnerDownstreamError):
            extract_builder2_sequence_stage_text({}, "sequence.resolution")

    def test_malformed_stage_type_rejected(self) -> None:
        with self.assertRaises(Builder2WinnerDownstreamError):
            extract_builder2_sequence_stage_text(["not", "a", "stage"], "sequence.beginning")


class TestWinnerSequenceNormalizationLifecycle(unittest.TestCase):
    def _processing_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        winning_candidate = winner_rec["creatorOutput"]
        winning_judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        return {
            "winner_id": winner_id,
            "winning_candidate": winning_candidate,
            "winning_judgment": winning_judgment,
            "source_reference": build_server_owned_winner_source_reference(
                strategy_foundation=state["strategyFoundation"],
                winning_candidate=winning_candidate,
                candidate_id=winner_id,
            ),
        }

    def test_validate_persist_and_normalize_without_crash(self) -> None:
        state = _current_job_shaped_state()
        ctx = self._processing_context(state)
        raw = _structured_sequence_plan()
        validated = process_winner_development_response(
            raw,
            source_reference=ctx["source_reference"],
            winning_candidate=ctx["winning_candidate"],
            winning_judgment=ctx["winning_judgment"],
            tournament_state=state,
        )
        for key in ("beginning", "development", "resolution"):
            self.assertIsInstance(validated["sequence"][key], str)

        fingerprint_before = compute_winner_development_plan_fingerprint(validated)
        persisted = persist_accepted_winner_development_for_media(
            state,
            candidate_id=ctx["winner_id"],
            prototype_id="winning_card",
            winner_plan=validated,
            winning_candidate=ctx["winning_candidate"],
            winning_judgment=ctx["winning_judgment"],
        )
        self.assertTrue(state.get("mediaContinuationRequired"))
        self.assertEqual(
            compute_winner_development_plan_fingerprint(persisted),
            fingerprint_before,
        )

        normalized = normalize_winner_plan_for_runway(
            persisted,
            product_name="ACE Product",
            product_description="desc",
            content_language="he",
        )
        scene = normalized["sceneConcept"]
        beginning = "The storefront window stays neutral before the proof appears."
        development = "The window surface becomes the persuasive medium."
        resolution = "The transformed medium completes the strategic proof."
        self.assertIn(beginning, scene)
        self.assertIn(development, scene)
        self.assertIn(resolution, scene)
        self.assertLess(scene.index(beginning), scene.index(development))
        self.assertLess(scene.index(development), scene.index(resolution))
        self.assertNotIn("{'description'", scene)
        self.assertNotIn("dict", scene.lower())

    def test_string_stages_still_work(self) -> None:
        plan = _winner_plan(language="he")
        normalized = validate_and_normalize_builder2_winner_plan(
            plan,
            product_name="ACE Product",
            product_description="desc",
            content_language="he",
        )
        self.assertIn("Two people stand apart.", normalized["sceneConcept"])

    def test_invalid_sequence_rejected_at_validation(self) -> None:
        plan = _structured_sequence_plan()
        plan["sequence"]["development"] = {"familyId": "nearness"}
        with self.assertRaises(Builder2TournamentError):
            validate_builder2_winner_plan(plan)

    def test_media_resume_downstream_accepts_persisted_structured_sequence(self) -> None:
        plan = _structured_sequence_plan()
        raw_sequence = deepcopy(plan["sequence"])
        normalized = normalize_builder2_winner_downstream(plan, compatibility_mode=False)
        self.assertEqual(normalized["sequence"]["beginning"], raw_sequence["beginning"]["description"])
        self.assertEqual(normalized["sequence"]["development"], raw_sequence["development"]["description"])
        self.assertEqual(normalized["sequence"]["resolution"], raw_sequence["resolution"]["description"])

    @patch("engine.builder2_tournament_llm.call_builder2_role_json_with_text")
    def test_no_openai_repair_for_structural_normalization(self, llm_mock) -> None:
        plan = _structured_sequence_plan()
        validate_and_normalize_builder2_winner_plan(
            plan,
            product_name="ACE Product",
            product_description="desc",
            content_language="he",
        )
        llm_mock.assert_not_called()

    def test_winner_identity_preserved_through_normalization(self) -> None:
        plan = _structured_sequence_plan()
        validated = validate_builder2_winner_plan(plan)
        fingerprint = compute_winner_development_plan_fingerprint(validated)
        normalized = normalize_winner_plan_for_runway(
            validated,
            product_name="ACE Product",
            product_description="desc",
            content_language="he",
        )
        self.assertEqual(validated.get("prototypeId"), "winning_card")
        self.assertEqual(normalized.get("prototypeId"), "winning_card")
        self.assertEqual(compute_winner_development_plan_fingerprint(validated), fingerprint)
        self.assertEqual(normalized["sequence"]["beginning"], validated["sequence"]["beginning"])


class TestWinnerSequenceSceneVariationsFallback(unittest.TestCase):
    def test_invalid_scene_variation_count_uses_normalized_sequence_text(self) -> None:
        plan = _structured_sequence_plan()
        plan["sceneVariations"] = ["Beat one.", "Beat two.", "Beat three.", "Beat four."]
        validated = validate_builder2_winner_plan(plan)
        self.assertEqual(len(validated["sceneVariations"]), 3)
        self.assertEqual(
            validated["sceneVariations"][0],
            "The storefront window stays neutral before the proof appears.",
        )
