"""
ACE local dev guards — offline only (no Redis, no paid APIs).

Run: python -m unittest tests.test_ace_local_dev -v
"""
from __future__ import annotations

import os
import unittest

from engine.ace_local_dev_config import (
    assert_local_dev_startup_safe,
    classify_public_base_url_for_local_dev,
    classify_redis_url_for_local_dev,
    confirm_local_dev_loopback_bind,
    enable_local_dev_security_bypass_for_tests,
    is_local_dev_security_bypass_allowed,
    is_loopback_bind_host,
    is_loopback_redis_host,
    reset_local_dev_config_for_tests,
    resolve_local_dev_bind_host,
)
from engine.ace_security_config import (
    SECURITY_STATE_DISABLED,
    assert_protected_route_security_config,
    is_security_explicitly_disabled,
    resolve_security_config_state,
)


class AceLocalDevRedisHostTests(unittest.TestCase):
    def test_loopback_hosts(self) -> None:
        for host in ("127.0.0.1", "localhost", "::1"):
            self.assertTrue(is_loopback_redis_host(host))
            self.assertTrue(is_loopback_bind_host(host))

    def test_remote_redis_rejected_for_local_dev(self) -> None:
        self.assertEqual(
            classify_redis_url_for_local_dev("redis://user:pass@redis.example.com:6379/0"),
            "ace_local_dev_redis_not_loopback",
        )

    def test_loopback_redis_allowed(self) -> None:
        self.assertIsNone(classify_redis_url_for_local_dev("redis://127.0.0.1:6379/0"))
        self.assertIsNone(classify_redis_url_for_local_dev(""))

    def test_production_public_base_url_blocked(self) -> None:
        self.assertEqual(
            classify_public_base_url_for_local_dev("https://ace-backend.onrender.com"),
            "ace_local_dev_public_base_url_production",
        )
        self.assertIsNone(classify_public_base_url_for_local_dev("http://127.0.0.1:5000"))


class AceLocalDevSecurityBypassTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_PUBLIC_BASE_URL", None)

    def test_security_off_without_local_dev_not_bypassed(self) -> None:
        from app import app

        os.environ["ACE_SECURITY_ENABLED"] = "false"
        self.assertEqual(resolve_security_config_state(), SECURITY_STATE_DISABLED)
        self.assertFalse(is_security_explicitly_disabled())
        with app.app_context():
            err = assert_protected_route_security_config()
        self.assertIsNotNone(err)
        self.assertEqual(err[1], 503)

    def test_security_off_with_local_dev_and_loopback_bypassed(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        os.environ["ACE_SECURITY_ENABLED"] = "false"
        self.assertTrue(is_local_dev_security_bypass_allowed())
        self.assertTrue(is_security_explicitly_disabled())

    def test_local_dev_bind_host_is_loopback(self) -> None:
        os.environ["ACE_LOCAL_DEV_ENABLED"] = "true"
        self.assertEqual(resolve_local_dev_bind_host(), "127.0.0.1")
        reset_local_dev_config_for_tests()

    def test_startup_blocks_non_loopback_redis(self) -> None:
        os.environ["ACE_LOCAL_DEV_ENABLED"] = "true"
        confirm_local_dev_loopback_bind("127.0.0.1")
        os.environ["REDIS_URL"] = "redis://redis-cloud.example.com:6379"
        with self.assertRaises(RuntimeError) as ctx:
            assert_local_dev_startup_safe()
        self.assertEqual(str(ctx.exception), "ace_local_dev_redis_not_loopback")

    def test_startup_blocks_production_public_base_url(self) -> None:
        os.environ.pop("REDIS_URL", None)
        os.environ["ACE_LOCAL_DEV_ENABLED"] = "true"
        confirm_local_dev_loopback_bind("127.0.0.1")
        os.environ["ACE_PUBLIC_BASE_URL"] = "https://www.ace-advertising.agency"
        with self.assertRaises(RuntimeError) as ctx:
            assert_local_dev_startup_safe()
        self.assertEqual(str(ctx.exception), "ace_local_dev_public_base_url_production")

if __name__ == "__main__":
    unittest.main()
