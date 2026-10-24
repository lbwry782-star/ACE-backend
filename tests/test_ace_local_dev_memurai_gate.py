"""Tests for LOCAL Memurai availability gate (file-based, no Redis)."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from engine.ace_local_dev_config import reset_local_dev_bind_confirmation_for_tests
from engine.ace_local_dev_memurai_gate import (
    ERROR_MEMURAI_UNAVAILABLE,
    load_memurai_availability,
    memurai_blocks_new_redis_work,
    public_memurai_availability_payload,
)


class MemuraiGateTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_local_dev_bind_confirmation_for_tests()
        self._prev_local = os.environ.pop("ACE_LOCAL_DEV_ENABLED", None)
        os.environ["ACE_LOCAL_DEV_ENABLED"] = "true"

    def tearDown(self) -> None:
        if self._prev_local is None:
            os.environ.pop("ACE_LOCAL_DEV_ENABLED", None)
        else:
            os.environ["ACE_LOCAL_DEV_ENABLED"] = self._prev_local
        reset_local_dev_bind_confirmation_for_tests()

    def test_missing_file_does_not_block(self) -> None:
        ace = Path(self._mkdtemp())
        self.assertIsNone(memurai_blocks_new_redis_work(ace_root=ace))

    def test_ping_ok_does_not_block(self) -> None:
        ace = Path(self._mkdtemp())
        self._write_avail(ace, {"pingOk": True, "blocksNewRedisWork": False})
        self.assertIsNone(memurai_blocks_new_redis_work(ace_root=ace))

    def test_down_blocks(self) -> None:
        ace = Path(self._mkdtemp())
        self._write_avail(
            ace,
            {
                "pingOk": False,
                "blocksNewRedisWork": True,
                "scenario": "ServiceStopped",
            },
        )
        self.assertEqual(
            memurai_blocks_new_redis_work(ace_root=ace),
            ERROR_MEMURAI_UNAVAILABLE,
        )

    def test_public_payload(self) -> None:
        ace = Path(self._mkdtemp())
        self._write_avail(
            ace,
            {
                "pingOk": False,
                "blocksNewRedisWork": True,
                "scenario": "ServiceStopped",
                "windowsServiceStatus": "Stopped",
                "updatedAt": "2026-01-01T00:00:00+00:00",
            },
        )
        payload = public_memurai_availability_payload(ace_root=ace)
        self.assertFalse(payload["available"])
        self.assertTrue(payload["blocksNewRedisWork"])
        self.assertEqual(payload["scenario"], "ServiceStopped")

    def _mkdtemp(self) -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="ace-memurai-gate-")

    def _write_avail(self, ace: Path, body: dict) -> None:
        d = ace / "_backups" / "local-dev"
        d.mkdir(parents=True)
        path = d / "memurai-availability.json"
        payload = {"schemaVersion": 1, **body}
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        loaded = load_memurai_availability(ace_root=ace)
        assert loaded is not None


if __name__ == "__main__":
    unittest.main()
