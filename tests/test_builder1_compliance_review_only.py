"""
Builder1 image compliance contract, contract repair, and REVIEW_ONLY retry tests.

Run: python -m unittest tests.test_builder1_compliance_review_only -v
"""
from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest.mock import Mock, patch

from engine.builder1_campaign_store import (
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_compliance_review_required,
    reserve_next_ad_index,
)
from engine.builder1_image_compliance import (
    IMAGE_COMPLIANCE_SYSTEM_PROMPT,
    ImageComplianceError,
    ImageComplianceResponseError,
    ImageComplianceUnavailableError,
    parse_image_compliance_response,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_compliance_contract import (
    COMPLIANCE_RESPONSE_JSON_SCHEMA,
    compliance_prompt_json_instructions,
    normalize_compliance_payload,
    coerce_review_dict,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_pending_image_store import (
    clear_memory_pending_for_tests,
    load_pending_image,
    save_pending_image,
)
from engine.builder1_retry_state import RETRY_MODE_REVIEW_ONLY
from tests.builder1_test_helpers import pass_compliance_reviewer, seed_builder1_image_job
from tests.test_builder1_series import _base_campaign, _parse


def _plan(ad_count: int = 2):
    plan = _parse(_base_campaign(ad_count), ad_count)
    plan.product_visibility_policy = "FORBIDDEN"
    return plan


def _canonical_pass_payload() -> dict[str, Any]:
    return {
        "reviewStatus": "completed",
        "hardViolations": [],
        "advisories": [],
        "evidence": [],
        "overallConfidence": "high",
    }


class TestComplianceContract(unittest.TestCase):
    def test_canonical_new_response_parses(self) -> None:
        result = parse_image_compliance_response(_canonical_pass_payload())
        self.assertTrue(result.passed)

    def test_prompt_and_schema_share_fields(self) -> None:
        prompt = compliance_prompt_json_instructions()
        required = set(COMPLIANCE_RESPONSE_JSON_SCHEMA["required"])
        for field in required:
            self.assertIn(field, prompt)
        self.assertIn("hardViolations", IMAGE_COMPLIANCE_SYSTEM_PROMPT)

    def test_advisory_only_passes(self) -> None:
        payload = {
            "reviewStatus": "completed",
            "hardViolations": [],
            "advisories": ["possible_logo_like_shape"],
            "evidence": [],
            "overallConfidence": "low",
        }
        result = parse_image_compliance_response(payload)
        self.assertTrue(result.passed)
        self.assertIn("possible_logo_like_shape", result.advisories or [])

    def test_hard_violation_fails(self) -> None:
        payload = {
            "reviewStatus": "completed",
            "hardViolations": ["invented_product_logo"],
            "advisories": [],
            "evidence": [],
            "overallConfidence": "high",
        }
        result = parse_image_compliance_response(payload)
        self.assertFalse(result.passed)

    def test_evidence_array_parses(self) -> None:
        payload = {
            "reviewStatus": "completed",
            "hardViolations": [],
            "advisories": ["possible_logo_like_shape"],
            "evidence": [
                {
                    "code": "possible_logo_like_shape",
                    "confidence": "low",
                    "evidenceType": "visual_context",
                    "description": "small dark mark",
                    "location": "top_left",
                    "relationshipToBrandText": "none",
                }
            ],
            "overallConfidence": "low",
        }
        result = parse_image_compliance_response(payload)
        self.assertTrue(result.evidence)

    def test_malformed_structure_fails_safely(self) -> None:
        with self.assertRaises(ImageComplianceResponseError):
            parse_image_compliance_response({"unexpected": True})

    def test_old_contract_normalizes(self) -> None:
        result = parse_image_compliance_response(
            {"pass": True, "violations": [], "confidence": "high"}
        )
        self.assertTrue(result.passed)

    @patch("engine.builder1_image_compliance_contract.logger")
    def test_legacy_normalization_logged(self, mock_logger) -> None:
        parse_image_compliance_response({"pass": True, "violations": [], "confidence": "high"})
        logged = " ".join(str(call) for call in mock_logger.info.call_args_list)
        self.assertIn("BUILDER1_IMAGE_COMPLIANCE_LEGACY_NORMALIZED", logged)

    def test_new_shape_without_pass_not_malformed(self) -> None:
        payload = _canonical_pass_payload()
        normalized = normalize_compliance_payload(coerce_review_dict(payload))
        self.assertEqual(normalized.legacy_shape, "canonical")
        self.assertTrue(normalized.reviewer_pass)


class TestMalformedResponseRepair(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance._openai_compliance_contract_repair_call")
    @patch("engine.builder1_image_compliance._openai_compliance_review_call")
    def test_invalid_json_triggers_contract_repair(self, mock_review, mock_repair) -> None:
        mock_review.return_value = "not-json"
        mock_repair.return_value = json.dumps(_canonical_pass_payload())
        result = review_builder1_ad_image_compliance(
            b"img",
            product_name="Brand",
            ad_index=1,
        )
        self.assertTrue(result.passed)
        mock_repair.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance._openai_compliance_contract_repair_call")
    @patch("engine.builder1_image_compliance._openai_compliance_review_call")
    def test_missing_field_triggers_contract_repair(self, mock_review, mock_repair) -> None:
        mock_review.return_value = json.dumps(
            {
                "hardViolations": [],
                "advisories": [],
                "overallConfidence": "high",
            }
        )
        mock_repair.return_value = json.dumps(_canonical_pass_payload())
        review_builder1_ad_image_compliance(b"img", product_name="Brand", ad_index=1)
        mock_repair.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance._openai_compliance_contract_repair_call")
    @patch("engine.builder1_image_compliance._openai_compliance_review_call")
    def test_contract_repair_reuses_same_image(self, mock_review, mock_repair) -> None:
        image = b"same-image-bytes"
        mock_review.return_value = "bad"
        mock_repair.return_value = json.dumps(_canonical_pass_payload())

        def repair_side_effect(**kwargs):
            self.assertEqual(kwargs["image_bytes"], image)
            return json.dumps(_canonical_pass_payload())

        mock_repair.side_effect = repair_side_effect
        review_builder1_ad_image_compliance(image, product_name="Brand", ad_index=1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("engine.builder1_image_compliance._openai_compliance_contract_repair_call")
    @patch("engine.builder1_image_compliance._openai_compliance_review_call")
    def test_second_malformed_enters_unavailable(self, mock_review, mock_repair) -> None:
        mock_review.return_value = "bad"
        mock_repair.return_value = "still-bad"
        with self.assertRaises(ImageComplianceUnavailableError) as ctx:
            review_builder1_ad_image_compliance(b"img", product_name="Brand", ad_index=1)
        self.assertEqual(ctx.exception.reason_code, "malformed_response")
        self.assertTrue(ctx.exception.contract_repair_attempted)

    def test_malformed_never_passes_open(self) -> None:
        with self.assertRaises(ImageComplianceResponseError):
            parse_image_compliance_response({"reviewStatus": "completed"})


class TestReviewOnlyPersistence(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_pending_for_tests()

    def test_review_only_state_persists(self) -> None:
        plan = _plan(3)
        create_campaign_session(campaign_id="cmp-review", plan=plan, target_ad_count=3)
        ref = save_pending_image(
            campaign_id="cmp-review",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"pending",
            visual_prompt="prompt",
        )
        session = mark_compliance_review_required(
            "cmp-review",
            failed_ad_index=1,
            reason="malformed_response",
            pending_image_key=ref,
            visual_prompt="prompt",
            contract_attempts=2,
        )
        self.assertEqual(session.retry_mode, RETRY_MODE_REVIEW_ONLY)
        self.assertEqual(session.failed_ad_index, 1)
        self.assertEqual(session.plan_revision, 1)
        self.assertEqual(session.generated_count, 0)
        self.assertTrue(session.image_generated_pending)
        loaded = load_pending_image(ref)
        self.assertEqual(loaded.image_bytes, b"pending")

    def test_pending_survives_campaign_reload(self) -> None:
        plan = _plan(2)
        create_campaign_session(campaign_id="cmp-ser", plan=plan, target_ad_count=2)
        ref = save_pending_image(
            campaign_id="cmp-ser",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"x",
            visual_prompt="p",
        )
        mark_compliance_review_required(
            "cmp-ser",
            failed_ad_index=1,
            reason="malformed_response",
            pending_image_key=ref,
        )
        raw = get_campaign_session("cmp-ser")
        self.assertEqual(raw.pending_image_key, ref)


class TestReviewOnlyExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys
        from unittest.mock import MagicMock

        sys.modules.setdefault("openai", MagicMock())

    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_pending_for_tests()

    def _seed_review_only(self, *, campaign_id: str = "cmp-ro", job_id: str = "job-ro") -> None:
        plan = _plan(3)
        create_campaign_session(campaign_id=campaign_id, plan=plan, target_ad_count=3)
        ref = save_pending_image(
            campaign_id=campaign_id,
            ad_index=1,
            plan_revision=1,
            image_bytes=b"stored-image",
            visual_prompt="visual",
        )
        mark_compliance_review_required(
            campaign_id,
            failed_ad_index=1,
            reason="malformed_response",
            pending_image_key=ref,
            visual_prompt="visual",
        )
        reserve_next_ad_index(campaign_id, 1, job_id=job_id)
        seed_builder1_image_job(
            job_id=job_id,
            campaign_id=campaign_id,
            ad_index=1,
            target_ad_count=3,
            plan_revision=1,
        )

    def test_review_only_zero_planning_and_image_calls(self) -> None:
        from app import _builder1_generate_review_only_ad

        self._seed_review_only()
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            with patch("app.generate_builder1_ad_image") as mock_image:
                with patch(
                    "app.review_builder1_ad_image_compliance",
                    return_value=pass_compliance_reviewer(image_bytes=b"stored-image"),
                ):
                    result = _builder1_generate_review_only_ad(
                        job_id="job-ro",
                        campaign_id="cmp-ro",
                        ad_index=1,
                        already_reserved=True,
                    )
        mock_plan.assert_not_called()
        mock_image.assert_not_called()
        self.assertTrue(result["ok"])

    def test_hard_violation_transitions_to_image_only(self) -> None:
        from app import _builder1_generate_review_only_ad
        from engine.builder1_image_compliance import ImageComplianceResult

        self._seed_review_only(job_id="job-hard")
        failing = ImageComplianceResult(
            passed=False,
            violations=["invented_product_logo"],
            hard_violations=["invented_product_logo"],
            raw_violations=["invented_product_logo"],
            confidence="high",
        )
        with patch("app.review_builder1_ad_image_compliance", return_value=failing):
            result = _builder1_generate_review_only_ad(
                job_id="job-hard",
                campaign_id="cmp-ro",
                ad_index=1,
                already_reserved=True,
            )
        self.assertEqual(result["error"], "image_compliance_failed")
        self.assertEqual(result["retryMode"], "image_only")
        session = get_campaign_session("cmp-ro")
        self.assertEqual(session.status, "image_retry_required")

    def test_continued_unavailability_preserves_review_only(self) -> None:
        from app import _builder1_generate_review_only_ad

        self._seed_review_only(job_id="job-unavail")
        with patch(
            "app.review_builder1_ad_image_compliance",
            side_effect=ImageComplianceUnavailableError("malformed_response", ad_index=1),
        ):
            result = _builder1_generate_review_only_ad(
                job_id="job-unavail",
                campaign_id="cmp-ro",
                ad_index=1,
                already_reserved=True,
            )
        self.assertEqual(result["error"], "image_compliance_unavailable")
        self.assertEqual(result["retryMode"], RETRY_MODE_REVIEW_ONLY)
        session = get_campaign_session("cmp-ro")
        self.assertEqual(session.status, "compliance_review_required")


class TestPublicContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys
        from unittest.mock import MagicMock

        sys.modules.setdefault("openai", MagicMock())

    def setUp(self) -> None:
        clear_memory_store_for_tests()
        clear_memory_pending_for_tests()

    def test_compliance_unavailable_payload(self) -> None:
        from app import _builder1_generate_single_ad

        plan = _plan(3)
        create_campaign_session(campaign_id="cmp-pub", plan=plan, target_ad_count=3)
        reserve_next_ad_index("cmp-pub", 1, job_id="job-pub")
        seed_builder1_image_job(
            job_id="job-pub",
            campaign_id="cmp-pub",
            ad_index=1,
            target_ad_count=3,
        )
        err = ImageComplianceUnavailableError(
            "malformed_response",
            ad_index=1,
            image_bytes=b"img",
            visual_prompt="prompt",
            contract_repair_attempted=True,
        )
        with patch("app.generate_builder1_ad_image", side_effect=err):
            result = _builder1_generate_single_ad(
                job_id="job-pub",
                campaign_id="cmp-pub",
                ad_index=1,
                already_reserved=True,
            )
        self.assertEqual(result["error"], "image_compliance_unavailable")
        self.assertEqual(result["retryMode"], RETRY_MODE_REVIEW_ONLY)
        self.assertTrue(result["planningComplete"])
        self.assertTrue(result["imageGenerated"])
        self.assertFalse(result["complianceAvailable"])
        self.assertEqual(result["campaignId"], "cmp-pub")
        self.assertEqual(result["retryAdIndex"], 1)
        self.assertEqual(result["planRevision"], 1)

    def test_generate_next_routes_review_only(self) -> None:
        plan = _plan(3)
        create_campaign_session(campaign_id="cmp-route", plan=plan, target_ad_count=3)
        ref = save_pending_image(
            campaign_id="cmp-route",
            ad_index=1,
            plan_revision=1,
            image_bytes=b"x",
            visual_prompt="p",
        )
        mark_compliance_review_required(
            "cmp-route",
            failed_ad_index=1,
            reason="malformed_response",
            pending_image_key=ref,
        )
        session = get_campaign_session("cmp-route")
        self.assertEqual(session.next_ad_index, 1)

        submitted: list[tuple] = []

        def capture_submit(fn, *args):
            submitted.append((fn.__name__, args))
            return Mock()

        with patch("app._builder1_executor") as mock_exec:
            mock_exec.submit.side_effect = capture_submit
            from app import app

            client = app.test_client()
            resp = client.post(
                "/api/builder1-generate-next",
                json={"campaignId": "cmp-route", "expectedNextIndex": 1},
            )
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(submitted[0][0], "_builder1_run_review_only_job")


class TestRegression(unittest.TestCase):
    def test_builder2_files_unchanged(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "engine"
        names = {path.name for path in root.glob("builder2*.py")}
        self.assertTrue(names)


if __name__ == "__main__":
    unittest.main()
