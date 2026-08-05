"""
Builder2 audience knowledge-gap inspector tests — no network, no Redis writes.
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_audience_knowledge_gap_inspect import (
    CLASSIFICATION_IMMEDIATE,
    CLASSIFICATION_MIXED,
    CLASSIFICATION_NONE,
    CLASSIFICATION_POSITIVE_GAP,
    CLASSIFICATION_RISK,
    inspect_audience_knowledge_gap,
    inspect_audience_knowledge_gap_for_job,
)
from engine.builder2_judge_response_ledger import append_judge_response_attempt
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.builder2_tournament_contracts import TOURNAMENT_STATE_SCHEMA_VERSION
from tests.test_builder2_tournament import _candidate, _judgment, _strategy


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _forgot_state(*, creator_report_note: str = "", judge_notes: str = "", winner_note: str = "") -> Dict[str, Any]:
    strategy = _strategy(language="he")
    strategy["productNameResolved"] = "גומיה"
    strategy["problemPerception"] = {
        "statement": "הצופה לא תמיד מבין למה המוצר שונה מממתק רגיל.",
        "groundingType": "common_market_behavior",
        "groundingEvidence": ["ממתקים נתפסים כמוצר חד-פעמי."],
        "whyItMatters": "צריך לבנות הבנה פרסומית.",
    }
    candidate_id = "cand-1-forgot-1-b28eccbc"
    candidate = _candidate("forgot")
    candidate["candidateId"] = candidate_id
    candidate["advertisingClosure"] = {
        "required": True,
        "productNameText": "גומיה",
        "sloganText": "אחד לעכשיו תשע לפעמים הבאות",
        "language": "he",
        "presentationMode": "end_card",
        "durationSeconds": 3.5,
        "headlineSource": "creator",
        "noLogo": True,
    }
    if creator_report_note:
        candidate.setdefault("creatorReport", {})
        candidate["creatorReport"]["relativeAdvantage"] = creator_report_note
    judgment = _judgment(candidate_id, total_hint=89, eligible=True)
    if judge_notes:
        judgment.setdefault("advertisingSloganAssessment", {})
        judgment["advertisingSloganAssessment"]["notes"] = judge_notes
    state: Dict[str, Any] = {
        "schemaVersion": TOURNAMENT_STATE_SCHEMA_VERSION,
        "jobId": "13922a7b-b22e-4ef7-9a27-89abbef085bd",
        "tournamentId": "454136f1-00dc-440a-b492-3784d1601ef2",
        "productName": "גומיה",
        "productDescription": "מסטיק בטעם פירות. בא בחבילות של 10 מסטיקים.",
        "strategyFoundation": strategy,
        "winnerCandidateId": candidate_id,
        "candidates": {
            candidate_id: {
                "candidateId": candidate_id,
                "prototypeId": "forgot",
                "validationStatus": "accepted",
                "creatorOutput": candidate,
                "judgmentId": f"judgment-{candidate_id}",
                "totalScore": 89,
            }
        },
        "judgments": {
            f"judgment-{candidate_id}": {
                "judgmentId": f"judgment-{candidate_id}",
                "candidateId": candidate_id,
                "judgment": judgment,
            }
        },
    }
    if winner_note:
        state["winnerDevelopmentPlan"] = {
            "schemaVersion": "builder2_winner_video_plan_v1",
            "prototypeId": "forgot",
            "advertisingClosure": candidate["advertisingClosure"],
            "advertisingSloganEvidence": {"whyAdvertising": winner_note},
        }
    return state


class TestAudienceKnowledgeGapClassification(unittest.TestCase):
    def test_explicit_positive_knowledge_gap(self) -> None:
        note = (
            "The viewer does not know in advance that the product comes in a pack of ten sticks. "
            "The viewer realizes missing information and asks why one and nine. "
            "The gap is designed to create curiosity and complete later when the viewer sees the package on the shelf."
        )
        report = inspect_audience_knowledge_gap(
            _forgot_state(creator_report_note=note),
            candidate_id="cand-1-forgot-1-b28eccbc",
        )
        self.assertEqual(report["classification"], CLASSIFICATION_POSITIVE_GAP)
        self.assertTrue(report["evidence"]["audiencePriorKnowledgeRecognized"]["found"])
        self.assertTrue(report["evidence"]["missingInformationRecognizedByViewer"]["found"])
        self.assertTrue(report["evidence"]["curiosityGapIntended"]["found"] or report["evidence"]["laterNaturalResolutionExpected"]["found"])

    def test_immediate_understanding_assumed(self) -> None:
        note = "The viewer immediately sees all ten gum sticks and already understands one now and nine later."
        report = inspect_audience_knowledge_gap(
            _forgot_state(judge_notes=note),
            candidate_id="cand-1-forgot-1-b28eccbc",
        )
        self.assertEqual(report["classification"], CLASSIFICATION_IMMEDIATE)

    def test_gap_considered_as_risk(self) -> None:
        note = "Without knowing the pack size, the viewer may be confused by the slogan and misunderstand the line."
        report = inspect_audience_knowledge_gap(
            _forgot_state(judge_notes=note),
            candidate_id="cand-1-forgot-1-b28eccbc",
        )
        self.assertEqual(report["classification"], CLASSIFICATION_RISK)

    def test_only_ten_units_no_viewer_reference(self) -> None:
        note = "Ten gum sticks sit in the pack and nine remain after one is taken."
        report = inspect_audience_knowledge_gap(
            _forgot_state(creator_report_note=note, winner_note="Nine remain after one is removed from ten sticks."),
            candidate_id="cand-1-forgot-1-b28eccbc",
        )
        self.assertEqual(report["classification"], CLASSIFICATION_NONE)
        self.assertFalse(report["evidence"]["audiencePriorKnowledgeRecognized"]["found"])

    def test_visual_bridge_boolean_only_is_not_evidence(self) -> None:
        state = _forgot_state()
        state["candidates"]["cand-1-forgot-1-b28eccbc"]["creatorOutput"]["sloganVisualBridgeAccepted"] = True
        report = inspect_audience_knowledge_gap(state, candidate_id="cand-1-forgot-1-b28eccbc")
        self.assertEqual(report["classification"], CLASSIFICATION_NONE)

    def test_mixed_evidence(self) -> None:
        note = (
            "The viewer does not know the pack size in advance, but the viewer immediately sees all ten sticks "
            "and understands one and nine right away."
        )
        report = inspect_audience_knowledge_gap(
            _forgot_state(creator_report_note=note),
            candidate_id="cand-1-forgot-1-b28eccbc",
        )
        self.assertEqual(report["classification"], CLASSIFICATION_MIXED)

    def test_safety_fields_present(self) -> None:
        report = inspect_audience_knowledge_gap(_forgot_state(), candidate_id="cand-1-forgot-1-b28eccbc")
        self.assertTrue(report["readOnly"])
        self.assertFalse(report["stateMutated"])
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["paidCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)


class TestAudienceKnowledgeGapReadOnly(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.builder2_audience_knowledge_gap_inspect.redis_configured", return_value=False)
    def test_inspector_does_not_mutate_state(self, _redis: Any) -> None:
        from engine.builder2_tournament_store import load_tournament_state

        state = _forgot_state(creator_report_note="The viewer does not know the pack size in advance.")
        job_id = _clean(state["jobId"])
        save_tournament_state(job_id, state)
        before = copy.deepcopy(state)
        with read_only_builder2_inspection() as counter:
            report = inspect_audience_knowledge_gap_for_job(job_id)
            self.assertTrue(report["ok"])
            self.assertEqual(counter.redis_mutations, 0)
        reloaded = load_tournament_state(job_id, read_only=True)
        self.assertEqual(reloaded, before)
        self.assertFalse(report["stateMutated"])

    @patch("engine.builder2_tournament_store.save_tournament_state")
    @patch("engine.builder2_audience_knowledge_gap_inspect.redis_configured", return_value=False)
    def test_job_wrapper_never_calls_save(self, _redis: Any, save_mock: Any) -> None:
        enable_memory_store()
        state = _forgot_state()
        save_tournament_state(state["jobId"], state)
        report = inspect_audience_knowledge_gap_for_job(state["jobId"])
        self.assertTrue(report["ok"])
        save_mock.assert_not_called()


class TestAudienceKnowledgeGapSources(unittest.TestCase):
    def test_collects_parsed_judge_response(self) -> None:
        state = _forgot_state()
        candidate_id = "cand-1-forgot-1-b28eccbc"
        parsed = {"advertisingSloganAssessment": {"notes": "The viewer does not know the pack size in advance."}}
        append_judge_response_attempt(
            state,
            candidate_id=candidate_id,
            judgment_id=f"judgment-{candidate_id}",
            call_type="normal",
            response_text='{"notes":"raw"}',
            parsed=parsed,
        )
        report = inspect_audience_knowledge_gap(state, candidate_id=candidate_id)
        self.assertTrue(report["sourceAvailability"]["judgeParsedResponseFound"])
        self.assertTrue(report["sourceAvailability"]["judgeRawResponseFound"])


if __name__ == "__main__":
    unittest.main()
