"""Builder2 closure duration contract tests."""
from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_closure_duration_contract import (
    BUILDER2_CLOSURE_V2_SEGMENT_DURATION_SECONDS,
    BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS,
    ENV_BUILDER2_END_CARD_DURATION_SECONDS,
    build_closure_duration_inspector_fields,
    enforce_v3_closure_duration_contract,
    resolve_configured_closure_segment_duration_seconds,
    resolve_expected_final_video_duration_seconds,
)
from engine.builder2_closure_rerender_inspect import inspect_builder2_closure_rerender
from engine.builder2_closure_typography import (
    BUILDER2_CLOSURE_TYPOGRAPHY_V2,
    BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
)
from engine.builder2_new_format_config import resolve_builder2_effective_closure_segment_duration_seconds
from engine.builder2_tournament_contracts import Builder2TournamentError


def _completed_v2_state() -> Dict[str, Any]:
    return {
        "jobId": "8b34c172-2b8b-404a-885d-ca41a07513a7",
        "tournamentId": "t-1",
        "status": "completed",
        "winnerDevelopmentPlan": {
            "productNameResolved": "Product",
            "advertisingClosure": {
                "productNameText": "Product",
                "sloganText": "Slogan",
                "language": "he",
            },
        },
        "advertisingClosure": {
            "required": True,
            "productNameText": "Product",
            "sloganText": "Slogan",
            "durationSeconds": 2.0,
            "presentationMode": "end_card",
            "language": "he",
        },
        "mediaResume": {
            "mediaResumeStatus": "completed",
            "closureTypographyContractVersion": BUILDER2_CLOSURE_TYPOGRAPHY_V2,
            "finalVideoWithClosureUrl": "https://example.com/final.mp4",
            "finalPublicUrl": "https://example.com/durable-final.mp4",
            "rawRunwayVideoUrl": "https://example.com/raw-runway.mp4",
            "actualFinalVideoDurationSeconds": 12.033,
            "measuredRawRunwayDurationSeconds": 10.042,
        },
    }


class TestBuilder2ClosureDurationContract(unittest.TestCase):
    def test_v3_resolves_closure_duration_to_three_point_five(self) -> None:
        resolved = resolve_configured_closure_segment_duration_seconds(
            typography_contract_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
        )
        self.assertAlmostEqual(resolved, 3.5, places=3)

    def test_historical_two_seconds_does_not_override_requested_v3(self) -> None:
        resolved = resolve_builder2_effective_closure_segment_duration_seconds(
            2.0,
            typography_contract_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
        )
        self.assertAlmostEqual(resolved, 3.5, places=3)

    @patch.dict("os.environ", {ENV_BUILDER2_END_CARD_DURATION_SECONDS: "2.0"}, clear=False)
    def test_environment_two_seconds_raises_for_v3(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            enforce_v3_closure_duration_contract()
        self.assertEqual(ctx.exception.args[0], "builder2_closure_duration_contract_mismatch")

    @patch.dict("os.environ", {ENV_BUILDER2_END_CARD_DURATION_SECONDS: "2.0"}, clear=False)
    def test_environment_two_seconds_cannot_resolve_v3_effective_duration(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            resolve_builder2_effective_closure_segment_duration_seconds(
                typography_contract_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
            )

    def test_expected_final_uses_measured_raw_plus_closure(self) -> None:
        expected = resolve_expected_final_video_duration_seconds(
            raw_video_duration_seconds=10.042,
            typography_contract_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
        )
        self.assertAlmostEqual(expected, 13.542, places=3)

    def test_non_ten_second_raw_derives_non_thirteen_point_five_total(self) -> None:
        expected = resolve_expected_final_video_duration_seconds(
            raw_video_duration_seconds=8.25,
            typography_contract_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
        )
        self.assertAlmostEqual(expected, 11.75, places=3)
        self.assertNotAlmostEqual(expected, 13.5, places=2)

    def test_inspector_distinguishes_historical_and_requested_duration(self) -> None:
        fields = build_closure_duration_inspector_fields(
            _completed_v2_state(),
            requested_typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
        )
        self.assertAlmostEqual(fields["currentArtifactClosureDurationSeconds"], 2.0, places=3)
        self.assertAlmostEqual(fields["requestedClosureDurationSeconds"], 3.5, places=3)
        self.assertAlmostEqual(fields["measuredRawVideoDurationSeconds"], 10.042, places=3)
        self.assertAlmostEqual(fields["requestedExpectedFinalDurationSeconds"], 13.542, places=3)
        self.assertTrue(fields["durationUpgradeNeeded"])
        self.assertAlmostEqual(fields["configuredClosureSegmentDurationSeconds"], 3.5, places=3)
        self.assertAlmostEqual(fields["configuredFinalVideoDurationSeconds"], 13.542, places=3)
        self.assertNotAlmostEqual(fields["configuredClosureSegmentDurationSeconds"], 2.0, places=2)
        self.assertNotAlmostEqual(fields["configuredFinalVideoDurationSeconds"], 12.0, places=2)

    @patch.dict("os.environ", {ENV_BUILDER2_END_CARD_DURATION_SECONDS: "2.0"}, clear=False)
    def test_inspector_reports_contract_unsatisfied_with_legacy_env(self) -> None:
        fields = build_closure_duration_inspector_fields(
            _completed_v2_state(),
            requested_typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
        )
        self.assertFalse(fields["closureDurationContractSatisfied"])
        self.assertAlmostEqual(fields["environmentEndCardDurationSeconds"], 2.0, places=3)

    def test_inspector_full_report_for_v2_job(self) -> None:
        report = inspect_builder2_closure_rerender(_completed_v2_state())
        self.assertEqual(report["currentTypographyContractVersion"], BUILDER2_CLOSURE_TYPOGRAPHY_V2)
        self.assertEqual(report["requestedTypographyContractVersion"], BUILDER2_CLOSURE_TYPOGRAPHY_VERSION)
        self.assertAlmostEqual(report["currentArtifactClosureDurationSeconds"], 2.0, places=3)
        self.assertAlmostEqual(report["requestedClosureDurationSeconds"], BUILDER2_CLOSURE_V3_SEGMENT_DURATION_SECONDS, places=3)
        self.assertTrue(report["durationUpgradeNeeded"])
        self.assertTrue(report["typographyUpgradeNeeded"])


class TestBuilder2ClosureOnlyRerenderDurationGuard(unittest.TestCase):
    @patch.dict("os.environ", {ENV_BUILDER2_END_CARD_DURATION_SECONDS: "2.0"}, clear=False)
    def test_rerender_refuses_when_env_contract_incompatible(self) -> None:
        from engine.builder2_closure_only_rerender import run_builder2_closure_only_rerender

        report = run_builder2_closure_only_rerender(
            job_id="8b34c172-2b8b-404a-885d-ca41a07513a7",
            tournament_state=_completed_v2_state(),
            expected_typography_version=BUILDER2_CLOSURE_TYPOGRAPHY_VERSION,
            public_base_url="https://ace.example.com",
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureReason"], "builder2_closure_duration_contract_mismatch")


if __name__ == "__main__":
    unittest.main()
