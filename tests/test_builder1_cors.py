"""
Builder1 CORS preflight compatibility — X-ACE-Request-Id and existing mutation headers.
"""
from __future__ import annotations

import os
import unittest
import uuid
from typing import Any
from unittest.mock import patch

import app as app_module


class TestBuilder1CorsPreflight(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_builder1_generate_options_allows_request_id_header(self) -> None:
        resp = self.client.open(
            "/api/builder1-generate",
            method="OPTIONS",
            headers={
                "Origin": "https://ace-advertising.agency",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "content-type,x-ace-request-id,x-ace-batch-state,authorization"
                ),
            },
        )
        self.assertEqual(resp.status_code, 200)
        allow_headers = (resp.headers.get("Access-Control-Allow-Headers") or "").lower()
        self.assertIn("x-ace-request-id", allow_headers)
        self.assertIn("x-ace-batch-state", allow_headers)
        self.assertIn("authorization", allow_headers)
        self.assertIn("content-type", allow_headers)
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "https://ace-advertising.agency")
        self.assertIn("POST", resp.headers.get("Access-Control-Allow-Methods") or "")

    def test_builder1_generate_post_includes_cors_origin(self) -> None:
        os.environ.pop("BUILDER1_PRODUCTION_MODE", None)
        os.environ.pop("BUILDER1_REQUEST_ID_REQUIRED", None)
        rid = str(uuid.uuid4())
        with patch.object(app_module._builder1_executor, "submit", return_value=None):
            with patch.object(app_module, "_builder1_run_initial_job", return_value=None):
                resp = self.client.post(
                    "/api/builder1-generate",
                    json={"productDescription": "desc", "productName": "P", "adCount": 2},
                    headers={
                        "Origin": "https://ace-advertising.agency",
                        "X-ACE-Batch-State": "cors-batch",
                        "Authorization": "Bearer cors-token",
                        "X-ACE-Request-Id": rid,
                    },
                )
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(
            resp.headers.get("Access-Control-Allow-Origin"),
            "https://ace-advertising.agency",
        )
        allow_headers = (resp.headers.get("Access-Control-Allow-Headers") or "").lower()
        self.assertIn("x-ace-request-id", allow_headers)
        self.assertIn("x-ace-batch-state", allow_headers)
        self.assertIn("authorization", allow_headers)


if __name__ == "__main__":
    unittest.main()
