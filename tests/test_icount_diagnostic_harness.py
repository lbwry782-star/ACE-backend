"""
iCount diagnostic harness tests — sanitization and route behavior only.

Run: python -m unittest tests.test_icount_diagnostic_harness -v
"""
from __future__ import annotations

import json
import unittest

from engine.icount_diagnostic_harness import (
    SYNTHETIC_MARKER_PREFIX,
    build_correlation_hints,
    build_observation_report,
    sanitize_field_map,
    sanitize_headers,
    sanitize_scalar_value,
    validate_diagnostic_nonce,
)
from engine.icount_diagnostic_routes import icount_diagnostic_bp


class TestSanitization(unittest.TestCase):
    def test_synthetic_marker_logged_in_full(self) -> None:
        marker = f"{SYNTHETIC_MARKER_PREFIX}abc123"
        self.assertEqual(sanitize_scalar_value("ace_checkout_ref", marker), marker)

    def test_sensitive_card_field_redacted_by_name(self) -> None:
        sanitized = sanitize_field_map({"card_number": "4111111111111111"})
        self.assertEqual(sanitized["card_number"]["redacted"], True)

    def test_email_field_redacted_by_name(self) -> None:
        sanitized = sanitize_field_map({"customer_email": "user@example.com"})
        self.assertEqual(sanitized["customer_email"]["redacted"], True)

    def test_amount_and_currency_may_log_full(self) -> None:
        self.assertEqual(sanitize_scalar_value("amount", "49.90"), "49.90")
        self.assertEqual(sanitize_scalar_value("currency", "ILS"), "ILS")

    def test_unknown_field_logs_digest_only(self) -> None:
        out = sanitize_scalar_value("unknown_field", "secret-ish-value")
        self.assertIn("sha256", out)
        self.assertNotIn("secret-ish-value", json.dumps(out))

    def test_authorization_header_value_not_logged_in_full(self) -> None:
        _, headers = sanitize_headers({"Authorization": "Bearer super-secret-token"})
        value = headers[0]["value"]
        self.assertIn("sha256", value)
        self.assertNotIn("super-secret-token", json.dumps(headers))

    def test_cookie_header_redacted(self) -> None:
        _, headers = sanitize_headers({"Cookie": "session=abc"})
        self.assertEqual(headers[0]["value"]["redacted"], True)


class TestCorrelationHints(unittest.TestCase):
    def test_hints_list_present_field_names_only(self) -> None:
        hints = build_correlation_hints(["docnum", "amount", "currency", "status", "ace_checkout_ref"])
        self.assertIn("docnum", hints["documentIdFieldNames"])
        self.assertIn("amount", hints["amountFieldNames"])
        self.assertIn("currency", hints["currencyFieldNames"])
        self.assertIn("status", hints["statusFieldNames"])
        self.assertIn("ace_checkout_ref", hints["customReferenceFieldNames"])


class TestObservationReport(unittest.TestCase):
    def test_report_includes_field_names_and_synthetic_marker(self) -> None:
        marker = f"{SYNTHETIC_MARKER_PREFIX}nonce-1"
        report = build_observation_report(
            channel="IPN",
            diagnostic_nonce="testnonce01",
            method="POST",
            content_type="application/x-www-form-urlencoded",
            query_fields={"cp": "123"},
            form_fields={"ace_checkout_ref": marker, "amount": "10"},
            json_payload=None,
            header_map={"Content-Type": "application/x-www-form-urlencoded"},
        )
        self.assertIn("cp", report["queryFieldNames"])
        self.assertIn("ace_checkout_ref", report["formFieldNames"])
        self.assertIn(marker, report["syntheticMarkers"])
        self.assertEqual(report["sanitizedForm"]["ace_checkout_ref"], marker)


class TestDiagnosticRoutes(unittest.TestCase):
    def setUp(self) -> None:
        from flask import Flask

        self.app = Flask(__name__)
        self.app.register_blueprint(icount_diagnostic_bp)
        self.client = self.app.test_client()

    def test_ipn_post_returns_ok_without_mutation(self) -> None:
        marker = f"{SYNTHETIC_MARKER_PREFIX}route-post"
        response = self.client.post(
            "/api/diagnostics/icount/ipn/testnonce01",
            data={"ace_checkout_ref": marker, "amount": "19.90", "currency": "ILS", "status": "paid"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "OK")

    def test_ipn_get_supported(self) -> None:
        response = self.client.get(
            "/api/diagnostics/icount/ipn/testnonce01?docnum=999&amount=1&currency=ILS"
        )
        self.assertEqual(response.status_code, 200)

    def test_return_redirects(self) -> None:
        response = self.client.get(
            f"/api/diagnostics/icount/return/testnonce01?ace_checkout_ref={SYNTHETIC_MARKER_PREFIX}ret1",
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers.get("Location"))

    def test_invalid_nonce_rejected(self) -> None:
        response = self.client.post("/api/diagnostics/icount/ipn/bad")
        self.assertEqual(response.status_code, 400)


class TestIsolation(unittest.TestCase):
    def test_harness_does_not_import_checkout_or_entitlement_modules(self) -> None:
        import engine.icount_diagnostic_harness as harness
        import engine.icount_diagnostic_routes as routes

        for module in (harness, routes):
            source_path = module.__file__ or ""
            self.assertNotIn("builder1", source_path)
            self.assertNotIn("builder2", source_path)

    def test_nonce_validation(self) -> None:
        self.assertTrue(validate_diagnostic_nonce("testnonce01"))
        self.assertFalse(validate_diagnostic_nonce("short"))


if __name__ == "__main__":
    unittest.main()
