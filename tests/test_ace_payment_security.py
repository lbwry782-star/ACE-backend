"""
ACE payment security — offline tests (no provider calls).

Run: python -m unittest tests.test_ace_payment_security -v
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import unittest
import uuid
from decimal import Decimal
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

from engine.ace_payment_checkout_store import (
    CHECKOUT_STATE_BOUND,
    CHECKOUT_STATE_PAID,
    CHECKOUT_STATE_PENDING,
    bind_checkout_to_resource,
    clear_memory_checkout_store_for_tests,
    configure_checkout_memory_store,
    create_checkout,
    get_checkout_record,
    mark_checkout_paid,
    verify_checkout_browser_token,
)
from engine.ace_payment_icount_ipn import (
    extract_icount_form_fields,
    process_icount_ipn,
    validate_icount_ipn_payload,
    verify_icount_ipn_secret_header,
)
from engine.ace_payment_offers import append_payment_reference_to_url, get_offer, list_offer_codes
from engine.ace_payment_tokens import (
    generate_browser_token,
    generate_payment_reference,
    load_icount_ipn_secret_sha256_from_env,
    sha256_hex,
    verify_browser_token,
)
from engine.ace_local_dev_config import (
    enable_local_dev_security_bypass_for_tests,
    reset_local_dev_config_for_tests,
)
from engine.ace_security_config import (
    SECURITY_STATE_DISABLED,
    SECURITY_STATE_ENABLED,
    SECURITY_STATE_INVALID,
    is_security_config_invalid,
    is_security_enabled,
    is_security_explicitly_disabled,
    resolve_security_config_state,
    security_config_payload,
)


def _checkout_headers(checkout_id: str, browser_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {browser_token}",
        "X-ACE-Checkout-Id": checkout_id,
    }


class AceSecurityConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        reset_local_dev_config_for_tests()

    def test_explicit_true_enables(self) -> None:
        for val in ("true", "TRUE", "  true  ", "1", "yes", "on"):
            os.environ["ACE_SECURITY_ENABLED"] = val
            self.assertEqual(resolve_security_config_state(), SECURITY_STATE_ENABLED)
            self.assertTrue(is_security_enabled())
            self.assertFalse(is_security_explicitly_disabled())
            self.assertFalse(is_security_config_invalid())

    def test_explicit_false_parses_disabled_env(self) -> None:
        for val in ("false", "FALSE", "  false  ", "0", "no", "off"):
            os.environ["ACE_SECURITY_ENABLED"] = val
            self.assertEqual(resolve_security_config_state(), SECURITY_STATE_DISABLED)
            self.assertFalse(is_security_enabled())
            self.assertFalse(is_security_explicitly_disabled())
            self.assertFalse(is_security_config_invalid())

    def test_explicit_false_bypass_only_with_local_dev(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        os.environ["ACE_SECURITY_ENABLED"] = "false"
        self.assertTrue(is_security_explicitly_disabled())
        self.assertEqual(security_config_payload(), {"securityEnabled": False})

    def test_missing_empty_typo_fail_closed(self) -> None:
        for val in ("", "  ", "treu", "maybe", "TRUEE", "enabled"):
            os.environ["ACE_SECURITY_ENABLED"] = val
            self.assertEqual(resolve_security_config_state(), SECURITY_STATE_INVALID)
            self.assertFalse(is_security_enabled())
            self.assertFalse(is_security_explicitly_disabled())
            self.assertTrue(is_security_config_invalid())
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        self.assertEqual(resolve_security_config_state(), SECURITY_STATE_INVALID)


class AcePaymentOfferTests(unittest.TestCase):
    def test_all_offers_registered(self) -> None:
        codes = set(list_offer_codes())
        self.assertEqual(codes, {"b1-2", "b1-3", "b1-4", "b2-1", "b2-2"})

    def test_offer_amounts_and_builders(self) -> None:
        self.assertEqual(get_offer("b1-2").quantity, 2)
        self.assertEqual(get_offer("b1-2").amount, Decimal("45"))
        self.assertEqual(get_offer("b1-2").builder, "builder1")
        self.assertEqual(get_offer("b2-2").quantity, 2)
        self.assertEqual(get_offer("b2-2").amount, Decimal("140"))
        self.assertEqual(get_offer("b2-2").builder, "builder2")

    def test_payment_url_preserves_utm_and_appends_correlation(self) -> None:
        offer = get_offer("b2-1")
        ref = "pay-ref-abc123"
        url = append_payment_reference_to_url(offer.payment_page_url, ref)
        self.assertIn("utm_source=iCount", url)
        self.assertIn("m__ace_checkout_id=pay-ref-abc123", url)


class AceCheckoutStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()

    def test_browser_token_not_stored_raw(self) -> None:
        offer = get_offer("b1-2")
        record, browser_token, payment_reference = create_checkout(offer)
        stored = get_checkout_record(record.checkout_id)
        self.assertIsNotNone(stored)
        raw_dump = json.dumps(stored.__dict__ if hasattr(stored, "__dict__") else stored)
        self.assertNotIn(browser_token, raw_dump)
        self.assertEqual(stored.browser_token_hash, sha256_hex(browser_token))
        self.assertNotEqual(browser_token, payment_reference)
        self.assertEqual(stored.payment_reference_hash, sha256_hex(payment_reference))

    def test_tokens_are_random(self) -> None:
        t1 = generate_browser_token()
        t2 = generate_browser_token()
        self.assertNotEqual(t1, t2)
        self.assertEqual(len(t1), 64)

    def test_simultaneous_checkouts_do_not_collide(self) -> None:
        offer = get_offer("b1-3")
        ids = set()
        for _ in range(20):
            record, _, _ = create_checkout(offer)
            ids.add(record.checkout_id)
        self.assertEqual(len(ids), 20)

    def test_mark_paid_idempotent_same_document(self) -> None:
        offer = get_offer("b1-2")
        record, _, payment_ref = create_checkout(offer)
        ok1, code1 = mark_checkout_paid(
            checkout_id=record.checkout_id,
            doctype="invrec",
            docnum="1001",
            sum_value="45",
            currency_code="ILS",
        )
        ok2, code2 = mark_checkout_paid(
            checkout_id=record.checkout_id,
            doctype="invrec",
            docnum="1001",
            sum_value="45",
            currency_code="ILS",
        )
        self.assertTrue(ok1)
        self.assertEqual(code1, "paid")
        self.assertTrue(ok2)
        self.assertEqual(code2, "idempotent")

    def test_same_document_cannot_pay_two_checkouts(self) -> None:
        offer = get_offer("b1-2")
        r1, _, _ = create_checkout(offer)
        r2, _, _ = create_checkout(offer)
        ok1, _ = mark_checkout_paid(
            checkout_id=r1.checkout_id,
            doctype="invrec",
            docnum="2002",
            sum_value="45",
            currency_code="ILS",
        )
        ok2, code2 = mark_checkout_paid(
            checkout_id=r2.checkout_id,
            doctype="invrec",
            docnum="2002",
            sum_value="45",
            currency_code="ILS",
        )
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertEqual(code2, "document_already_claimed")


class AceIcountIpnTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        self.raw_secret = "test-icount-shared-secret-value"
        os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = sha256_hex(self.raw_secret)

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)

    def test_secret_header_verification(self) -> None:
        present, verified, code = verify_icount_ipn_secret_header(
            {"X-Icount-Secret": self.raw_secret}
        )
        self.assertTrue(present)
        self.assertTrue(verified)
        self.assertIsNone(code)

        present2, verified2, code2 = verify_icount_ipn_secret_header({})
        self.assertFalse(present2)
        self.assertFalse(verified2)
        self.assertEqual(code2, "ipn_secret_missing")

        present3, verified3, code3 = verify_icount_ipn_secret_header(
            {"X-Icount-Secret": "wrong-secret"}
        )
        self.assertTrue(present3)
        self.assertFalse(verified3)
        self.assertEqual(code3, "ipn_secret_invalid")

    def test_raw_secret_never_logged(self) -> None:
        with self.assertLogs("engine.ace_payment_icount_ipn", level="WARNING") as captured:
            process_icount_ipn(form_fields={}, headers={})
        blob = "\n".join(captured.output)
        self.assertNotIn(self.raw_secret, blob)

    def _paid_checkout(self, offer_code: str = "b1-2") -> Tuple[str, str]:
        offer = get_offer(offer_code)
        record, _, payment_ref = create_checkout(offer)
        status, _, _ = process_icount_ipn(
            form_fields={
                "ace_checkout_id": payment_ref,
                "sum": format(offer.amount, "f"),
                "currency_code": "ILS",
                "doctype": "invrec",
                "docnum": str(uuid.uuid4().int % 10_000_000),
                "cp": "3",
            },
            headers={"X-Icount-Secret": self.raw_secret},
        )
        self.assertEqual(status, 200)
        return record.checkout_id, payment_ref

    def test_valid_ipn_marks_paid(self) -> None:
        checkout_id, _ = self._paid_checkout()
        record = get_checkout_record(checkout_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.state, CHECKOUT_STATE_PAID)

    def test_unknown_payment_reference_rejected(self) -> None:
        status, _, reason = process_icount_ipn(
            form_fields={
                "ace_checkout_id": "unknown-ref-value",
                "sum": "45",
                "currency_code": "ILS",
                "doctype": "invrec",
                "docnum": "1",
            },
            headers={"X-Icount-Secret": self.raw_secret},
        )
        self.assertEqual(status, 400)
        self.assertEqual(reason, "payment_reference_unknown")

    def test_wrong_amount_and_currency_rejected(self) -> None:
        offer = get_offer("b1-2")
        record, _, payment_ref = create_checkout(offer)
        for fields, reason in (
            (
                {
                    "ace_checkout_id": payment_ref,
                    "sum": "99",
                    "currency_code": "ILS",
                    "doctype": "invrec",
                    "docnum": "9",
                },
                "amount_mismatch",
            ),
            (
                {
                    "ace_checkout_id": payment_ref,
                    "sum": "45",
                    "currency_code": "USD",
                    "doctype": "invrec",
                    "docnum": "9",
                },
                "currency_mismatch",
            ),
            (
                {
                    "ace_checkout_id": payment_ref,
                    "sum": "45",
                    "currency_code": "ILS",
                    "doctype": "",
                    "docnum": "",
                },
                "document_identity_missing",
            ),
        ):
            status, _, got = process_icount_ipn(
                form_fields=fields,
                headers={"X-Icount-Secret": self.raw_secret},
            )
            self.assertEqual(status, 400, msg=reason)
            self.assertEqual(got, reason)

    def test_form_parsing(self) -> None:
        fields = extract_icount_form_fields(
            {"sum": ["45"], "currency_code": "ILS", "ace_checkout_id": "abc"}
        )
        self.assertEqual(fields["sum"], "45")
        self.assertEqual(fields["ace_checkout_id"], "abc")


class AcePaymentRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        os.environ["REDIS_URL"] = "redis://test"
        os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = sha256_hex("route-test-secret")
        from app import app

        self.app = app
        self.client = app.test_client()

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)
        reset_local_dev_config_for_tests()

    @patch("engine.video_jobs_redis.get_redis")
    def test_security_config_invalid_returns_503(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        resp = self.client.get("/api/security/config")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json(), {"error": "ace_security_config_invalid"})
        self.assertEqual(resp.headers.get("Cache-Control"), "no-store")
        self.assertNotIn("securityEnabled", resp.get_json())

    @patch("engine.video_jobs_redis.get_redis")
    def test_security_config_explicit_false_returns_200(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        enable_local_dev_security_bypass_for_tests()
        os.environ["ACE_SECURITY_ENABLED"] = "false"
        resp = self.client.get("/api/security/config")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"securityEnabled": False})

    @patch("engine.video_jobs_redis.get_redis")
    def test_checkout_disabled_when_security_off(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        enable_local_dev_security_bypass_for_tests()
        os.environ["ACE_SECURITY_ENABLED"] = "false"
        resp = self.client.post("/api/payments/checkout", json={"offerCode": "b1-2"})
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["error"], "ace_security_disabled")

    @patch("engine.video_jobs_redis.get_redis")
    def test_checkout_create_and_status(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        resp = self.client.post("/api/payments/checkout", json={"offerCode": "b1-4"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("checkoutId", data)
        self.assertIn("browserToken", data)
        self.assertIn("m__ace_checkout_id=", data["paymentUrl"])
        self.assertEqual(data["offer"]["quantity"], 4)

        status_resp = self.client.post(
            "/api/payments/checkout/status",
            json={"checkoutId": data["checkoutId"]},
            headers=_checkout_headers(data["checkoutId"], data["browserToken"]),
        )
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.get_json()["state"], CHECKOUT_STATE_PENDING)

    @patch("engine.video_jobs_redis.get_redis")
    def test_wrong_browser_token_rejected(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        resp = self.client.post("/api/payments/checkout", json={"offerCode": "b2-1"})
        data = resp.get_json()
        bad = self.client.post(
            "/api/payments/checkout/status",
            json={"checkoutId": data["checkoutId"]},
            headers=_checkout_headers(data["checkoutId"], "wrong-token"),
        )
        self.assertEqual(bad.status_code, 401)

    @patch("engine.video_jobs_redis.get_redis")
    def test_ipn_endpoint_disabled_when_security_off(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        enable_local_dev_security_bypass_for_tests()
        os.environ["ACE_SECURITY_ENABLED"] = "false"
        resp = self.client.post("/api/payments/icount/ipn", data={"sum": "45"})
        self.assertEqual(resp.status_code, 503)


class AceBuilderGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        os.environ["REDIS_URL"] = "redis://test"
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = sha256_hex("builder-guard-secret")
        from app import app

        self.app = app
        self.client = app.test_client()

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)

    def _paid_checkout(self, offer_code: str) -> Dict[str, Any]:
        offer = get_offer(offer_code)
        record, browser_token, payment_ref = create_checkout(offer)
        status, _, _ = process_icount_ipn(
            form_fields={
                "ace_checkout_id": payment_ref,
                "sum": format(offer.amount, "f"),
                "currency_code": "ILS",
                "doctype": "invrec",
                "docnum": str(uuid.uuid4().int % 10_000_000),
            },
            headers={"X-Icount-Secret": "builder-guard-secret"},
        )
        self.assertEqual(status, 200)
        return {
            "checkoutId": record.checkout_id,
            "browserToken": browser_token,
            "offer": offer,
        }

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app.create_builder1_job")
    @patch("app._builder1_executor")
    @patch("app._builder1_idempotency_begin")
    @patch("app._builder1_enforce_create_ownership", return_value=None)
    @patch("app._builder1_require_request_id", return_value=None)
    @patch("app._builder1_require_production_ready", return_value=None)
    def test_invalid_config_blocks_builder1_before_job(
        self,
        _prod: MagicMock,
        _req: MagicMock,
        _own: MagicMock,
        mock_idem: MagicMock,
        mock_executor: MagicMock,
        mock_create_job: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        mock_idem.return_value = (None, "key", {}, str(uuid.uuid4()), str(uuid.uuid4()))
        resp = self.client.post(
            "/api/builder1-generate",
            json={"productDescription": "Valid product description for testing.", "adCount": 2},
            headers={"X-ACE-Batch-State": "batch-1"},
        )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["error"], "ace_security_config_invalid")
        mock_create_job.assert_not_called()
        mock_executor.submit.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app._builder1_require_production_ready", return_value=None)
    @patch("app._builder1_require_request_id", return_value=None)
    @patch("app._builder1_enforce_create_ownership", return_value=None)
    @patch("app._builder1_idempotency_begin")
    @patch("app.create_builder1_job")
    @patch("app._builder1_executor")
    def test_builder1_unpaid_rejected_before_job(
        self,
        mock_executor: MagicMock,
        mock_create_job: MagicMock,
        mock_idem: MagicMock,
        _ownership: MagicMock,
        _req_id: MagicMock,
        _prod: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        mock_idem.return_value = (None, "key", {}, str(uuid.uuid4()), str(uuid.uuid4()))
        resp = self.client.post(
            "/api/builder1-generate",
            json={
                "productDescription": "Valid product description for testing.",
                "adCount": 2,
            },
            headers={"X-ACE-Batch-State": "batch-1", "Authorization": "Bearer owner"},
        )
        self.assertEqual(resp.status_code, 403)
        mock_create_job.assert_not_called()
        mock_executor.submit.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app._builder1_require_production_ready", return_value=None)
    @patch("app._builder1_require_request_id", return_value=None)
    @patch("app._builder1_enforce_create_ownership", return_value=None)
    @patch("app._builder1_idempotency_begin")
    @patch("app._builder1_idempotency_finish")
    @patch("app.create_builder1_job")
    @patch("app._builder1_executor")
    def test_builder1_paid_entitlement_sets_ad_count(
        self,
        mock_executor: MagicMock,
        mock_create_job: MagicMock,
        mock_finish: MagicMock,
        mock_idem: MagicMock,
        _ownership: MagicMock,
        _req_id: MagicMock,
        _prod: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        paid = self._paid_checkout("b1-3")
        job_id = str(uuid.uuid4())
        campaign_id = str(uuid.uuid4())
        mock_idem.return_value = (None, "key", {}, job_id, campaign_id)

        def _finish(**kwargs: Any) -> Any:
            return self.client.application.response_class(
                response=json.dumps(kwargs["response_body"]),
                status=200,
                mimetype="application/json",
            )

        mock_finish.side_effect = _finish

        headers = _checkout_headers(paid["checkoutId"], paid["browserToken"])
        headers["X-ACE-Batch-State"] = "batch-1"

        resp = self.client.post(
            "/api/builder1-generate",
            json={
                "productDescription": "Valid product description for testing.",
                "adCount": 3,
            },
            headers=headers,
        )
        self.assertEqual(resp.status_code, 200, msg=resp.get_json())
        body = resp.get_json()
        self.assertEqual(body["targetAdCount"], 3)
        mock_create_job.assert_called_once()
        args, kwargs = mock_create_job.call_args
        self.assertEqual(kwargs.get("target_ad_count"), 3)
        record = get_checkout_record(paid["checkoutId"])
        self.assertEqual(record.state, CHECKOUT_STATE_BOUND)
        self.assertEqual(record.bound_resource_type, "builder1_campaign")
        self.assertTrue(record.bound_resource_id)

    @patch("engine.video_jobs_redis.get_redis")
    @patch("engine.video_jobs_redis.redis_configured", return_value=True)
    @patch("engine.video_jobs_redis.video_job_create")
    def test_builder2_unpaid_rejected(
        self,
        mock_enqueue: MagicMock,
        _redis_ok: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"
        resp = self.client.post(
            "/api/generate-video",
            json={
                "productName": "Product",
                "productDescription": "Valid product description.",
                "targetVideoCount": 1,
            },
            headers={"X-ACE-Batch-State": "batch-b2", "Authorization": "Bearer owner"},
        )
        self.assertEqual(resp.status_code, 403)
        mock_enqueue.assert_not_called()


class AceConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()

    def test_duplicate_ipn_race_idempotent(self) -> None:
        offer = get_offer("b1-2")
        record, _, _ = create_checkout(offer)
        results: list[tuple[bool, str]] = []
        lock = threading.Lock()

        def _pay() -> None:
            ok, code = mark_checkout_paid(
                checkout_id=record.checkout_id,
                doctype="invrec",
                docnum="9001",
                sum_value="45",
                currency_code="ILS",
            )
            with lock:
                results.append((ok, code))

        threads = [threading.Thread(target=_pay) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(any(ok for ok, _ in results))
        paid_codes = [code for ok, code in results if ok]
        self.assertTrue(all(c in {"paid", "idempotent"} for c in paid_codes))
        record_after = get_checkout_record(record.checkout_id)
        self.assertEqual(record_after.state, CHECKOUT_STATE_PAID)

    def test_bind_idempotent(self) -> None:
        offer = get_offer("b2-1")
        record, _, _ = create_checkout(offer)
        mark_checkout_paid(
            checkout_id=record.checkout_id,
            doctype="invrec",
            docnum="7001",
            sum_value="80",
            currency_code="ILS",
        )
        resource = str(uuid.uuid4())
        ok1, c1 = bind_checkout_to_resource(
            checkout_id=record.checkout_id,
            resource_id=resource,
            resource_type="builder2_allowance",
        )
        ok2, c2 = bind_checkout_to_resource(
            checkout_id=record.checkout_id,
            resource_id=resource,
            resource_type="builder2_allowance",
        )
        self.assertTrue(ok1)
        self.assertEqual(c1, "bound")
        self.assertTrue(ok2)
        self.assertEqual(c2, "idempotent")

    def test_concurrent_bind_competing_resources_one_winner(self) -> None:
        offer = get_offer("b1-2")
        record, _, _ = create_checkout(offer)
        mark_checkout_paid(
            checkout_id=record.checkout_id,
            doctype="invrec",
            docnum="8801",
            sum_value="45",
            currency_code="ILS",
        )
        resource_a = str(uuid.uuid4())
        resource_b = str(uuid.uuid4())
        results: list[tuple[bool, str, str]] = []
        barrier = threading.Barrier(2)

        def _bind(resource_id: str) -> None:
            barrier.wait(timeout=5)
            ok, code = bind_checkout_to_resource(
                checkout_id=record.checkout_id,
                resource_id=resource_id,
                resource_type="builder1_campaign",
            )
            results.append((ok, code, resource_id))

        threads = [
            threading.Thread(target=_bind, args=(resource_a,)),
            threading.Thread(target=_bind, args=(resource_b,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 2)
        successes = [r for r in results if r[0]]
        failures = [r for r in results if not r[0]]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(successes[0][1], "bound")
        self.assertEqual(failures[0][1], "checkout_already_bound")
        bound = get_checkout_record(record.checkout_id)
        self.assertIsNotNone(bound)
        self.assertEqual(bound.state, CHECKOUT_STATE_BOUND)
        self.assertIn(bound.bound_resource_id, {resource_a, resource_b})
        self.assertTrue(bound.bound_at)


class AceDownloadVideoZipTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        os.environ["REDIS_URL"] = "redis://test"
        os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = sha256_hex("zip-test-secret")
        from app import app

        self.app = app
        self.client = app.test_client()

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)
        reset_local_dev_config_for_tests()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("requests.get")
    def test_security_on_remote_url_rejected_before_fetch(
        self,
        mock_requests_get: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        remote_url = "https://cdn.example.com/videos/remote-ad.mp4"
        resp = self.client.get(f"/api/download-video-zip?videoUrl={remote_url}")
        self.assertEqual(resp.status_code, 403)
        mock_requests_get.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("requests.get")
    def test_security_off_remote_url_legacy_fetch(
        self,
        mock_requests_get: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        enable_local_dev_security_bypass_for_tests()
        os.environ["ACE_SECURITY_ENABLED"] = "false"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "video/mp4"}
        mock_response.iter_content.return_value = [b"fake-video-bytes"]
        mock_response.text = ""
        mock_requests_get.return_value = mock_response
        remote_url = "https://cdn.example.com/videos/legacy-ad.mp4"
        resp = self.client.get(f"/api/download-video-zip?videoUrl={remote_url}")
        self.assertEqual(resp.status_code, 200)
        mock_requests_get.assert_called_once()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("requests.get")
    def test_security_on_bound_checkout_allows_matching_remote_url(
        self,
        mock_requests_get: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        from engine.builder2_video_allowance_store import (
            clear_memory_allowance_store_for_tests,
            create_video_allowance,
            enable_memory_allowance_store,
        )
        from engine.video_jobs_redis import enable_memory_jobs, video_job_create

        enable_memory_jobs()
        enable_memory_allowance_store()
        clear_memory_allowance_store_for_tests()
        mock_get_redis.return_value.ping.return_value = True
        os.environ["ACE_SECURITY_ENABLED"] = "true"

        offer = get_offer("b2-1")
        record, browser_token, payment_ref = create_checkout(offer)
        process_icount_ipn(
            form_fields={
                "ace_checkout_id": payment_ref,
                "sum": "80",
                "currency_code": "ILS",
                "doctype": "invrec",
                "docnum": "99001",
            },
            headers={"X-Icount-Secret": "zip-test-secret"},
        )
        allowance_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        remote_url = "https://cdn.example.com/videos/paid-ad.mp4"
        bind_checkout_to_resource(
            checkout_id=record.checkout_id,
            resource_id=allowance_id,
            resource_type="builder2_allowance",
        )
        create_video_allowance(
            video_allowance_id=allowance_id,
            owner_context_ref="owner-zip",
            target_video_count=1,
            product_name="Product",
            product_description="Desc",
            first_job_id=job_id,
        )
        video_job_create(
            job_id,
            "Product",
            "Desc",
            "https://example.com",
            extra_fields={
                "videoAllowanceId": allowance_id,
                "video_url": remote_url,
            },
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "video/mp4"}
        mock_response.iter_content.return_value = [b"paid-video-bytes"]
        mock_response.text = ""
        mock_requests_get.return_value = mock_response

        headers = _checkout_headers(record.checkout_id, browser_token)
        resp = self.client.get(
            f"/api/download-video-zip?videoUrl={remote_url}",
            headers=headers,
        )
        self.assertEqual(resp.status_code, 200, msg=resp.get_json())
        mock_requests_get.assert_called_once()

    @patch("engine.video_jobs_redis.get_redis")
    @patch("requests.get")
    def test_security_on_wrong_checkout_rejected_before_fetch(
        self,
        mock_requests_get: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        offer = get_offer("b2-1")
        record, browser_token, payment_ref = create_checkout(offer)
        process_icount_ipn(
            form_fields={
                "ace_checkout_id": payment_ref,
                "sum": "80",
                "currency_code": "ILS",
                "doctype": "invrec",
                "docnum": "99002",
            },
            headers={"X-Icount-Secret": "zip-test-secret"},
        )
        bind_checkout_to_resource(
            checkout_id=record.checkout_id,
            resource_id=str(uuid.uuid4()),
            resource_type="builder2_allowance",
        )
        remote_url = "https://cdn.example.com/videos/other-ad.mp4"
        headers = _checkout_headers(record.checkout_id, browser_token)
        resp = self.client.get(
            f"/api/download-video-zip?videoUrl={remote_url}",
            headers=headers,
        )
        self.assertEqual(resp.status_code, 403)
        mock_requests_get.assert_not_called()


class AceTokenTests(unittest.TestCase):
    def test_malformed_env_secret_rejected(self) -> None:
        os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = "not-a-valid-hex"
        self.assertIsNone(load_icount_ipn_secret_sha256_from_env())
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)

    def test_browser_token_verify(self) -> None:
        raw = generate_browser_token()
        stored = sha256_hex(raw)
        self.assertTrue(verify_browser_token(raw, stored))
        self.assertFalse(verify_browser_token("other", stored))


if __name__ == "__main__":
    unittest.main()
