"""
ACE payment replay / refresh purchase-limit hardening tests — zero provider calls.

Run: python -m unittest tests.test_ace_payment_replay_limits -v
"""
from __future__ import annotations

import copy
import os
import threading
import unittest
import uuid
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

from engine.ace_payment_authorization import (
    authorize_builder_create,
    authorize_builder1_campaign_access,
    authorize_builder2_allowance_access,
    peek_builder_checkout_entitlement,
    verify_checkout_browser_token,
)
from engine.ace_payment_checkout_store import (
    CHECKOUT_STATE_BOUND,
    bind_checkout_to_resource,
    clear_memory_checkout_store_for_tests,
    configure_checkout_memory_store,
    create_checkout,
    get_checkout_record,
    mark_checkout_paid,
)
from engine.ace_payment_icount_ipn import process_icount_ipn
from engine.ace_payment_offers import get_offer
from engine.ace_payment_tokens import generate_payment_reference, sha256_hex
from engine.builder1_campaign_store import (
    CampaignStoreError,
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_ad_generated,
    release_generation_lock,
    reserve_next_ad_index,
    validate_next_ad_request,
)
from engine.builder2_video_allowance_store import (
    clear_memory_allowance_store_for_tests,
    create_video_allowance,
    enable_memory_allowance_store,
    get_video_allowance,
    reserve_video_two_slot,
)
from engine.builder2_video_allowance import request_generate_video_next
from tests.test_builder1_series import _base_campaign, _parse


def _checkout_headers(checkout_id: str, browser_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {browser_token}",
        "X-ACE-Checkout-Id": checkout_id,
    }


def _mock_request(headers: Dict[str, str]) -> MagicMock:
    req = MagicMock()
    req.headers = headers
    req.is_json = False
    return req


def _paid_checkout(offer_code: str, *, clear: bool = True) -> Tuple[str, str, str, int]:
    configure_checkout_memory_store(True)
    if clear:
        clear_memory_checkout_store_for_tests()
    os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = sha256_hex("replay-test-secret")
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
        headers={"X-Icount-Secret": "replay-test-secret"},
    )
    assert status == 200
    return record.checkout_id, browser_token, offer_code, offer.quantity


def _plan(ad_count: int) -> Any:
    return _parse(_base_campaign(ad_count), ad_count)


class AceCheckoutReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        os.environ["REDIS_URL"] = "redis://test"
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        from app import app

        self.app = app
        self.app_context = app.app_context()
        self.app_context.push()

    def tearDown(self) -> None:
        self.app_context.pop()
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)

    @patch("engine.video_jobs_redis.get_redis")
    def test_bound_checkout_cannot_start_second_campaign(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        checkout_id, browser_token, _, _ = _paid_checkout("b1-2")
        campaign_a = str(uuid.uuid4())
        headers = _checkout_headers(checkout_id, browser_token)
        req = _mock_request(headers)

        _, _, _, deny_bind = authorize_builder_create(
            request=req,
            builder="builder1",
            client_quantity=2,
            resource_id=campaign_a,
            resource_type="builder1_campaign",
            initial_create=True,
        )
        self.assertIsNone(deny_bind)

        campaign_b = str(uuid.uuid4())
        _, _, _, deny_second = authorize_builder_create(
            request=req,
            builder="builder1",
            client_quantity=2,
            resource_id=campaign_b,
            resource_type="builder1_campaign",
            initial_create=True,
        )
        self.assertIsNotNone(deny_second)
        self.assertEqual(deny_second[0].get_json()["error"], "ace_checkout_already_bound")

    @patch("engine.video_jobs_redis.get_redis")
    def test_bound_checkout_rejects_repeat_initial_on_same_campaign(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        checkout_id, browser_token, _, _ = _paid_checkout("b1-3")
        campaign_id = str(uuid.uuid4())
        headers = _checkout_headers(checkout_id, browser_token)
        req = _mock_request(headers)

        authorize_builder_create(
            request=req,
            builder="builder1",
            client_quantity=3,
            resource_id=campaign_id,
            resource_type="builder1_campaign",
            initial_create=True,
        )
        _, _, _, deny_repeat = authorize_builder_create(
            request=req,
            builder="builder1",
            client_quantity=3,
            resource_id=campaign_id,
            resource_type="builder1_campaign",
            initial_create=True,
        )
        self.assertIsNotNone(deny_repeat)
        self.assertEqual(deny_repeat[0].get_json()["error"], "ace_checkout_initial_already_started")

    @patch("engine.video_jobs_redis.get_redis")
    def test_payment_reference_cannot_authenticate_as_browser_token(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        offer = get_offer("b1-2")
        record, _, payment_ref = create_checkout(offer)
        mark_checkout_paid(
            checkout_id=record.checkout_id,
            doctype="invrec",
            docnum="123",
            sum_value="45",
            currency_code="ILS",
        )
        _, err = verify_checkout_browser_token(record.checkout_id, payment_ref)
        self.assertEqual(err, "browser_token_invalid")

    @patch("engine.video_jobs_redis.get_redis")
    def test_wrong_checkout_token_pair_rejected(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        c1, t1, _, _ = _paid_checkout("b1-2", clear=False)
        c2, t2, _, _ = _paid_checkout("b1-2", clear=False)
        campaign = str(uuid.uuid4())
        bind_checkout_to_resource(
            checkout_id=c1,
            resource_id=campaign,
            resource_type="builder1_campaign",
        )
        req = _mock_request(_checkout_headers(c1, t2))
        deny = authorize_builder1_campaign_access(req, campaign)
        self.assertIsNotNone(deny)
        self.assertEqual(deny[0].get_json()["error"], "browser_token_invalid")

        req2 = _mock_request(_checkout_headers(c2, t1))
        deny2 = authorize_builder1_campaign_access(req2, campaign)
        self.assertIsNotNone(deny2)

    @patch("engine.video_jobs_redis.get_redis")
    def test_bound_checkout_cannot_bind_second_allowance(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        checkout_id, browser_token, _, _ = _paid_checkout("b2-2", clear=False)
        allowance_a = str(uuid.uuid4())
        req = _mock_request(_checkout_headers(checkout_id, browser_token))
        _, _, _, deny1 = authorize_builder_create(
            request=req,
            builder="builder2",
            client_quantity=2,
            resource_id=allowance_a,
            resource_type="builder2_allowance",
            initial_create=True,
        )
        self.assertIsNone(deny1)
        allowance_b = str(uuid.uuid4())
        _, _, _, deny2 = authorize_builder_create(
            request=req,
            builder="builder2",
            client_quantity=2,
            resource_id=allowance_b,
            resource_type="builder2_allowance",
            initial_create=True,
        )
        self.assertIsNotNone(deny2)
        self.assertEqual(deny2[0].get_json()["error"], "ace_checkout_already_bound")


class AceBuilder1OutputLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def tearDown(self) -> None:
        clear_memory_store_for_tests()

    def test_b1_2_never_exceeds_two_ads(self) -> None:
        cid = "camp-b1-2"
        create_campaign_session(campaign_id=cid, plan=_plan(2), target_ad_count=2)
        reserve_next_ad_index(cid, 1, job_id="job-1")
        mark_ad_generated(cid, 1)
        reserve_next_ad_index(cid, 2, job_id="job-2")
        mark_ad_generated(cid, 2)
        session = get_campaign_session(cid)
        self.assertTrue(session.complete)
        self.assertEqual(session.generated_count, 2)
        with self.assertRaises(CampaignStoreError) as ctx:
            validate_next_ad_request(cid, 3)
        self.assertEqual(ctx.exception.code, "campaign_complete")

    def test_b1_4_cap_enforced_on_generate_next(self) -> None:
        cid = "camp-b1-4"
        create_campaign_session(campaign_id=cid, plan=_plan(4), target_ad_count=4)
        for idx in range(1, 5):
            reserve_next_ad_index(cid, idx, job_id=f"job-{idx}")
            mark_ad_generated(cid, idx)
        with self.assertRaises(CampaignStoreError) as ctx:
            reserve_next_ad_index(cid, 5, job_id="job-5")
        self.assertIn(ctx.exception.code, {"campaign_complete", "invalid_next_index"})

    def test_concurrent_final_ad_slot_b1_2(self) -> None:
        cid = "camp-concurrent-final"
        create_campaign_session(campaign_id=cid, plan=_plan(2), target_ad_count=2)
        reserve_next_ad_index(cid, 1, job_id="job-1")
        mark_ad_generated(cid, 1)

        results: List[Tuple[bool, str]] = []
        barrier = threading.Barrier(2)

        def _attempt(job_id: str) -> None:
            barrier.wait(timeout=5)
            try:
                reserve_next_ad_index(cid, 2, job_id=job_id)
                results.append((True, job_id))
            except CampaignStoreError as exc:
                results.append((False, exc.code))

        t1 = threading.Thread(target=_attempt, args=("job-a",))
        t2 = threading.Thread(target=_attempt, args=("job-b",))
        for t in (t1, t2):
            t.start()
        for t in (t1, t2):
            t.join()

        successes = [r for r in results if r[0]]
        failures = [r for r in results if not r[0]]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        session = get_campaign_session(cid)
        self.assertEqual(session.generated_count, 1)
        self.assertEqual(session.next_ad_index, 2)

    def test_cancel_does_not_restore_purchased_slots(self) -> None:
        cid = "camp-cancel"
        create_campaign_session(campaign_id=cid, plan=_plan(2), target_ad_count=2)
        reserve_next_ad_index(cid, 1, job_id="job-cancel")
        mark_ad_generated(cid, 1)
        release_generation_lock(cid, job_id="job-cancel")
        session = get_campaign_session(cid)
        self.assertEqual(session.target_ad_count, 2)
        self.assertEqual(session.generated_count, 1)
        reserve_next_ad_index(cid, 2, job_id="job-two")
        mark_ad_generated(cid, 2)
        self.assertTrue(get_campaign_session(cid).complete)


class AceBuilder2OutputLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_allowance_store()
        clear_memory_allowance_store_for_tests()

    def tearDown(self) -> None:
        clear_memory_allowance_store_for_tests()

    def test_b2_1_second_video_slot_rejected(self) -> None:
        aid = str(uuid.uuid4())
        job1 = str(uuid.uuid4())
        create_video_allowance(
            video_allowance_id=aid,
            owner_context_ref="owner-b2",
            target_video_count=1,
            product_name="P",
            product_description="D",
            first_job_id=job1,
        )
        result = reserve_video_two_slot(aid, owner_context_ref="owner-b2", job_id=str(uuid.uuid4()))
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "target_video_count_not_two")

    def test_b2_2_concurrent_video_two_exactly_one_winner(self) -> None:
        aid = str(uuid.uuid4())
        job1 = str(uuid.uuid4())
        create_video_allowance(
            video_allowance_id=aid,
            owner_context_ref="owner-b2",
            target_video_count=2,
            product_name="P",
            product_description="D",
            first_job_id=job1,
        )
        allowance = get_video_allowance(aid)
        assert allowance is not None
        videos = list(allowance.get("videos") or [])
        videos[0]["videoIndex"] = 1
        allowance["videos"] = videos
        from engine.builder2_video_allowance_store import _write_raw

        _write_raw(aid, allowance)

        results: List[Tuple[bool, str]] = []
        barrier = threading.Barrier(2)

        def _reserve(job_id: str) -> None:
            barrier.wait(timeout=5)
            outcome = reserve_video_two_slot(aid, owner_context_ref="owner-b2", job_id=job_id)
            results.append((outcome.ok, outcome.code or ""))

        j2a = str(uuid.uuid4())
        j2b = str(uuid.uuid4())
        threads = [
            threading.Thread(target=_reserve, args=(j2a,)),
            threading.Thread(target=_reserve, args=(j2b,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ok_count = sum(1 for ok, _ in results if ok)
        self.assertEqual(ok_count, 2)
        final = get_video_allowance(aid)
        assert final is not None
        video_two_jobs = [
            e.get("jobId")
            for e in (final.get("videos") or [])
            if int(e.get("videoIndex") or 0) == 2
        ]
        self.assertEqual(len(video_two_jobs), 1)


class AceBuilder1HttpReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_checkout_memory_store(True)
        clear_memory_checkout_store_for_tests()
        os.environ["REDIS_URL"] = "redis://test"
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        os.environ["ACE_ICOUNT_IPN_SECRET_SHA256"] = sha256_hex("http-replay-secret")
        from app import app

        self.app = app
        self.client = app.test_client()

    def tearDown(self) -> None:
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ACE_ICOUNT_IPN_SECRET_SHA256", None)

    def _pay(self, offer_code: str) -> Dict[str, str]:
        offer = get_offer(offer_code)
        record, browser_token, payment_ref = create_checkout(offer)
        process_icount_ipn(
            form_fields={
                "ace_checkout_id": payment_ref,
                "sum": format(offer.amount, "f"),
                "currency_code": "ILS",
                "doctype": "invrec",
                "docnum": str(uuid.uuid4().int % 10_000_000),
            },
            headers={"X-Icount-Secret": "http-replay-secret"},
        )
        return {"checkoutId": record.checkout_id, "browserToken": browser_token}

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app._builder1_require_production_ready", return_value=None)
    @patch("app._builder1_require_request_id", return_value=None)
    @patch("app._builder1_enforce_create_ownership", return_value=None)
    @patch("app._builder1_idempotency_begin")
    @patch("app._builder1_idempotency_finish")
    @patch("app.create_builder1_job")
    @patch("app._builder1_executor")
    def test_new_request_id_after_bound_rejected(
        self,
        mock_executor: MagicMock,
        mock_create_job: MagicMock,
        mock_finish: MagicMock,
        mock_idem: MagicMock,
        _own: MagicMock,
        _req: MagicMock,
        _prod: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        paid = self._pay("b1-2")
        headers = _checkout_headers(paid["checkoutId"], paid["browserToken"])
        headers["X-ACE-Batch-State"] = "batch-1"

        job_id = str(uuid.uuid4())
        campaign_id = str(uuid.uuid4())
        mock_idem.return_value = (None, "idem-key", {}, job_id, campaign_id)

        def _finish(**kwargs: Any) -> Any:
            import json

            return self.client.application.response_class(
                response=json.dumps(kwargs["response_body"]),
                status=202,
                mimetype="application/json",
            )

        mock_finish.side_effect = _finish

        first = self.client.post(
            "/api/builder1-generate",
            json={"productDescription": "Valid product description.", "adCount": 2},
            headers=headers,
        )
        self.assertEqual(first.status_code, 202, msg=first.get_data(as_text=True))

        headers["X-ACE-Request-Id"] = str(uuid.uuid4())
        second = self.client.post(
            "/api/builder1-generate",
            json={"productDescription": "Valid product description.", "adCount": 2},
            headers=headers,
        )
        self.assertEqual(second.status_code, 403)
        self.assertEqual(second.get_json()["error"], "ace_checkout_initial_already_started")
        self.assertEqual(mock_create_job.call_count, 1)

    @patch("engine.video_jobs_redis.get_redis")
    @patch("app._builder1_require_production_ready", return_value=None)
    @patch("app._builder1_require_request_id", return_value=None)
    @patch("app._builder1_enforce_create_ownership", return_value=None)
    @patch("app._builder1_idempotency_finish")
    @patch("app.create_builder1_job")
    @patch("app._builder1_executor")
    def test_client_ad_count_upgrade_rejected(
        self,
        mock_executor: MagicMock,
        mock_create_job: MagicMock,
        mock_finish: MagicMock,
        _own: MagicMock,
        _req: MagicMock,
        _prod: MagicMock,
        mock_get_redis: MagicMock,
    ) -> None:
        mock_get_redis.return_value.ping.return_value = True
        paid = self._pay("b1-2")
        headers = _checkout_headers(paid["checkoutId"], paid["browserToken"])
        resp = self.client.post(
            "/api/builder1-generate",
            json={"productDescription": "Valid product description.", "adCount": 4},
            headers=headers,
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "ace_purchased_quantity_mismatch")
        mock_create_job.assert_not_called()
        mock_executor.submit.assert_not_called()

    @patch("engine.video_jobs_redis.get_redis")
    def test_missing_browser_token_rejected(self, mock_get_redis: MagicMock) -> None:
        mock_get_redis.return_value.ping.return_value = True
        paid = self._pay("b1-2")
        resp = self.client.post(
            "/api/builder1-generate",
            json={"productDescription": "Valid product description.", "adCount": 2},
            headers={"X-ACE-Checkout-Id": paid["checkoutId"], "X-ACE-Batch-State": "batch-1"},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.get_json()["error"], "ace_checkout_credentials_required")


class AcePeekEntitlementTests(unittest.TestCase):
    @patch("engine.video_jobs_redis.get_redis")
    def test_peek_returns_bound_campaign_without_consuming(self, mock_get_redis: MagicMock) -> None:
        from app import app

        mock_get_redis.return_value.ping.return_value = True
        os.environ["ACE_SECURITY_ENABLED"] = "true"
        os.environ["REDIS_URL"] = "redis://test"
        configure_checkout_memory_store(True)
        with app.app_context():
            checkout_id, browser_token, _, qty = _paid_checkout("b1-3")
            campaign_id = str(uuid.uuid4())
            bind_checkout_to_resource(
                checkout_id=checkout_id,
                resource_id=campaign_id,
                resource_type="builder1_campaign",
            )
            req = _mock_request(_checkout_headers(checkout_id, browser_token))
            quantity, bound_id, _, err = peek_builder_checkout_entitlement(
                request=req,
                builder="builder1",
                client_quantity=3,
            )
            self.assertIsNone(err)
            self.assertEqual(quantity, qty)
            self.assertEqual(bound_id, campaign_id)
            record = get_checkout_record(checkout_id)
            assert record is not None
            self.assertEqual(record.purchased_quantity, 3)
            self.assertEqual(record.state, CHECKOUT_STATE_BOUND)
        os.environ.pop("ACE_SECURITY_ENABLED", None)
        os.environ.pop("REDIS_URL", None)
        configure_checkout_memory_store(False)
        clear_memory_checkout_store_for_tests()


if __name__ == "__main__":
    unittest.main()
