"""
Builder1 ZIP route security — offline tests (no provider calls).

Run: python -m unittest tests.test_ace_payment_builder1_zip_security -v
"""
from __future__ import annotations

import base64
import os
import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import unittest

from engine.ace_payment_checkout_store import (
    bind_checkout_to_resource,
    clear_memory_checkout_store_for_tests,
    configure_checkout_memory_store,
    create_checkout,
    mark_checkout_paid,
)
from engine.ace_payment_icount_ipn import process_icount_ipn
from engine.ace_payment_offers import get_offer
from engine.ace_payment_tokens import sha256_hex
from tests.builder1_test_helpers import marketing_text_words

_IMG = base64.b64encode(b"fakejpeg").decode()
_TEXT50 = marketing_text_words(50)


def _checkout_headers(checkout_id: str, browser_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {browser_token}",
        "X-ACE-Checkout-Id": checkout_id,
    }


def _bound_builder1_checkout(campaign_id: str) -> Dict[str, str]:
    offer = get_offer("b1-2")
    record, browser_token, payment_ref = create_checkout(offer)
    mark_checkout_paid(
        checkout_id=record.checkout_id,
        doctype="invrec",
        docnum="7101",
        sum_value="45",
        currency_code="ILS",
    )
    ok, code = bind_checkout_to_resource(
        checkout_id=record.checkout_id,
        resource_id=campaign_id,
        resource_type="builder1_campaign",
    )
    assert ok and code in {"bound", "idempotent"}
    return {
        "checkoutId": record.checkout_id,
        "browserToken": browser_token,
    }


class Builder1ZipSecurityOnTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        os.environ["REDIS_URL"] = "redis://test"
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = sha256_hex("b1-zip-secret")
        from app import app

        self.client = app.test_client()
        self.campaign_id = str(uuid.uuid4())

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)

    def _single_ad_payload(self, *, campaign_id: str | None = None) -> Dict[str, Any]:
        return {
            "scope": "single_ad",
            "campaignId": self.campaign_id if campaign_id is None else campaign_id,
            "campaign": {
                "productNameResolved": "TestBrand",
                "brandSlogan": "Built To Last",
            },
            "ad": {
                "index": 1,
                "headline": None,
                "marketingText": _TEXT50,
                "imageBase64": _IMG,
            },
        }

    def _campaign_payload(self, *, campaign_id: str | None = None, scope: str | None = "campaign") -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "campaignId": self.campaign_id if campaign_id is None else campaign_id,
            "productName": "BrandX",
            "brandSlogan": "Stay Sharp",
            "ads": [
                {"index": 1, "imageBase64": _IMG, "headline": None, "marketingText": _TEXT50},
                {"index": 2, "imageBase64": _IMG, "headline": "Hi", "marketingText": marketing_text_words(50, "two")},
            ],
        }
        if scope is not None:
            payload["scope"] = scope
        return payload

    @patch("engine.video_jobs_redis.get_redis")
    def test_single_ad_allowed_with_bound_checkout(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        creds = _bound_builder1_checkout(self.campaign_id)
        resp = self.client.post(
            "/api/builder1-zip",
            json=self._single_ad_payload(),
            headers=_checkout_headers(creds["checkoutId"], creds["browserToken"]),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "application/zip")

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app.build_builder1_zip_from_request")
    def test_single_ad_missing_auth_rejected_before_zip_build(
        self,
        mock_build: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        resp = self.client.post("/api/builder1-zip", json=self._single_ad_payload())
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "ace_checkout_credentials_required")
        mock_build.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app.build_builder1_zip_from_request")
    def test_single_ad_wrong_checkout_rejected(
        self,
        mock_build: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        creds = _bound_builder1_checkout(self.campaign_id)
        resp = self.client.post(
            "/api/builder1-zip",
            json=self._single_ad_payload(),
            headers=_checkout_headers(creds["checkoutId"], "wrong-browser-token"),
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "browser_token_invalid")
        mock_build.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app.build_builder1_zip_from_request")
    def test_single_ad_wrong_campaign_rejected(
        self,
        mock_build: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        creds = _bound_builder1_checkout(self.campaign_id)
        resp = self.client.post(
            "/api/builder1-zip",
            json=self._single_ad_payload(campaign_id=str(uuid.uuid4())),
            headers=_checkout_headers(creds["checkoutId"], creds["browserToken"]),
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "ace_checkout_resource_mismatch")
        mock_build.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    def test_campaign_allowed_with_bound_checkout(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        creds = _bound_builder1_checkout(self.campaign_id)
        resp = self.client.post(
            "/api/builder1-zip",
            json=self._campaign_payload(),
            headers=_checkout_headers(creds["checkoutId"], creds["browserToken"]),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "application/zip")

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app.build_builder1_zip_from_request")
    def test_campaign_missing_auth_rejected(
        self,
        mock_build: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        resp = self.client.post("/api/builder1-zip", json=self._campaign_payload())
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "ace_checkout_credentials_required")
        mock_build.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app.build_builder1_zip_from_request")
    @patch("app._builder1_verify_campaign_access")
    def test_campaign_server_missing_auth_rejected(
        self,
        mock_verify: MagicMock,
        mock_build: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        mock_verify.return_value = (MagicMock(), None)
        resp = self.client.post(
            "/api/builder1-zip",
            json={"scope": "campaign_server", "campaignId": self.campaign_id},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "ace_checkout_credentials_required")
        mock_build.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app.build_builder1_zip_from_request")
    def test_default_scope_requires_auth(
        self,
        mock_build: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        resp = self.client.post("/api/builder1-zip", json=self._campaign_payload(scope=None))
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "ace_checkout_credentials_required")
        mock_build.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    def test_both_url_aliases_identical(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        creds = _bound_builder1_checkout(self.campaign_id)
        headers = _checkout_headers(creds["checkoutId"], creds["browserToken"])
        payload = self._single_ad_payload()
        resp_a = self.client.post("/api/builder1-zip", json=payload, headers=headers)
        resp_b = self.client.post("/api/builder1-download-zip", json=payload, headers=headers)
        self.assertEqual(resp_a.status_code, 200)
        self.assertEqual(resp_b.status_code, 200)
        self.assertEqual(resp_a.data, resp_b.data)


class Builder1ZipSecurityOffTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        os.environ["REDIS_URL"] = "redis://test"
        os.environ["ACE_SECURITY_ENABLED"] = "false"
        from app import app

        self.client = app.test_client()

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)

    def test_legacy_single_ad_without_checkout(self) -> None:
        payload = {
            "scope": "single_ad",
            "campaign": {
                "productNameResolved": "TestBrand",
                "brandSlogan": "Built To Last",
            },
            "ad": {
                "index": 1,
                "headline": None,
                "marketingText": _TEXT50,
                "imageBase64": _IMG,
            },
        }
        resp = self.client.post("/api/builder1-zip", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "application/zip")

    def test_legacy_campaign_without_checkout(self) -> None:
        payload = {
            "productName": "BrandX",
            "brandSlogan": "Stay Sharp",
            "ads": [
                {"index": 1, "imageBase64": _IMG, "headline": None, "marketingText": _TEXT50},
                {"index": 2, "imageBase64": _IMG, "headline": "Hi", "marketingText": marketing_text_words(50, "two")},
            ],
        }
        resp = self.client.post("/api/builder1-zip", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "application/zip")

    @patch("app._builder1_verify_campaign_access")
    @patch("app.build_builder1_zip_from_request", return_value=(b"legacy-zip", "builder1-campaign.zip"))
    def test_campaign_server_legacy_unchanged(
        self,
        mock_build: MagicMock,
        mock_verify: MagicMock,
    ) -> None:
        mock_verify.return_value = (MagicMock(), None)
        campaign_id = str(uuid.uuid4())
        resp = self.client.post(
            "/api/builder1-zip",
            json={"scope": "campaign_server", "campaignId": campaign_id},
        )
        self.assertEqual(resp.status_code, 200)
        mock_verify.assert_called_once_with(campaign_id)
        mock_build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
