"""
Public-base URL resolver tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from engine.builder2_media_resume_config import build_media_resume_configuration
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.public_base_url import normalize_public_base_url, require_public_base_url, resolve_public_base_url


class TestPublicBaseUrlResolver(unittest.TestCase):
    def test_ace_public_base_url_resolves(self) -> None:
        with patch.dict(os.environ, {"ACE_PUBLIC_BASE_URL": "https://ace-backend-k1p6.onrender.com/"}, clear=True):
            resolution = resolve_public_base_url()
        self.assertTrue(resolution.configured)
        self.assertEqual(resolution.source, "ACE_PUBLIC_BASE_URL")
        self.assertEqual(resolution.value, "https://ace-backend-k1p6.onrender.com")

    def test_public_base_url_env_resolves(self) -> None:
        with patch.dict(os.environ, {"PUBLIC_BASE_URL": "https://example.com"}, clear=True):
            resolution = resolve_public_base_url()
        self.assertTrue(resolution.configured)
        self.assertEqual(resolution.source, "PUBLIC_BASE_URL")

    def test_ace_wins_when_both_env_vars_exist(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ACE_PUBLIC_BASE_URL": "https://ace-backend-k1p6.onrender.com",
                "PUBLIC_BASE_URL": "https://example.com",
            },
            clear=True,
        ):
            resolution = resolve_public_base_url()
        self.assertEqual(resolution.source, "ACE_PUBLIC_BASE_URL")

    def test_job_public_base_url_wins_over_env(self) -> None:
        with patch.dict(os.environ, {"ACE_PUBLIC_BASE_URL": "https://ace-backend-k1p6.onrender.com"}, clear=True):
            resolution = resolve_public_base_url(job_data={"public_base_url": "https://job.example.com/"})
        self.assertEqual(resolution.source, "job_public_base_url")
        self.assertEqual(resolution.value, "https://job.example.com")

    def test_job_public_base_url_camel_case_supported(self) -> None:
        resolution = resolve_public_base_url(job_data={"publicBaseUrl": "https://job-camel.example.com/"})
        self.assertEqual(resolution.source, "job_publicBaseUrl")
        self.assertEqual(resolution.value, "https://job-camel.example.com")

    def test_trailing_slash_normalized(self) -> None:
        self.assertEqual(normalize_public_base_url("https://example.com/"), "https://example.com")

    def test_invalid_url_not_configured(self) -> None:
        resolution = resolve_public_base_url(job_data={"public_base_url": "not-a-url"})
        self.assertFalse(resolution.configured)

    def test_missing_url_raises_on_require(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Builder2TournamentError) as ctx:
                require_public_base_url()
        self.assertIn("publicBaseUrl", str(ctx.exception))


class TestMediaResumeConfigurationParity(unittest.TestCase):
    def test_dry_run_and_actual_use_same_configuration_object(self) -> None:
        env = {
            "ACE_PUBLIC_BASE_URL": "https://ace-backend-k1p6.onrender.com",
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_media_resume_configuration(
                job_id="job-config",
                job_data=None,
                tournament_state=None,
                start_image_required=True,
                ffmpeg_required=False,
            )
        self.assertEqual(config.publicBaseUrl, "https://ace-backend-k1p6.onrender.com")
        self.assertEqual(config.public_base_url.source, "ACE_PUBLIC_BASE_URL")
        self.assertEqual(config.runwayModel, "gen4_turbo")
        self.assertEqual(config.durationSeconds, 7)

    def test_missing_public_url_fails_configuration(self) -> None:
        with patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}, clear=True):
            with self.assertRaises(Builder2TournamentError) as ctx:
                build_media_resume_configuration(
                    job_id="job-missing-url",
                    job_data=None,
                    tournament_state=None,
                    start_image_required=True,
                    ffmpeg_required=False,
                )
        self.assertIn("publicBaseUrl", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
