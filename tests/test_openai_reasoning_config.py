"""Tests for shared OpenAI reasoning model configuration."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from engine.openai_reasoning import (
    DEFAULT_OPENAI_REASONING_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_MODE,
    build_reasoning_payload,
    normalize_legacy_text_model,
    resolve_openai_reasoning_model,
)


class TestOpenAIReasoningConfig(unittest.TestCase):
    def test_default_model_is_gpt_5_6_sol(self) -> None:
        self.assertEqual(DEFAULT_OPENAI_REASONING_MODEL, "gpt-5.6-sol")

    def test_o3_pro_maps_to_gpt_5_6_sol(self) -> None:
        self.assertEqual(normalize_legacy_text_model("o3-pro"), "gpt-5.6-sol")

    def test_o4_mini_is_not_mapped_to_gpt_5_6_sol(self) -> None:
        self.assertEqual(normalize_legacy_text_model("o4-mini"), "o4-mini")

    @patch.dict("os.environ", {"OPENAI_TEXT_MODEL": "o4-mini"}, clear=True)
    def test_o4_mini_env_preserves_literal_model(self) -> None:
        self.assertEqual(resolve_openai_reasoning_model(), "o4-mini")

    @patch.dict("os.environ", {}, clear=True)
    def test_resolve_model_default(self) -> None:
        self.assertEqual(resolve_openai_reasoning_model(), "gpt-5.6-sol")

    def test_reasoning_payload_defaults(self) -> None:
        self.assertEqual(
            build_reasoning_payload(),
            {"mode": DEFAULT_REASONING_MODE, "effort": DEFAULT_REASONING_EFFORT},
        )
        self.assertEqual(DEFAULT_REASONING_MODE, "standard")
        self.assertEqual(DEFAULT_REASONING_EFFORT, "high")


if __name__ == "__main__":
    unittest.main()
