"""Strict JSON bodies for local dev mutating endpoints."""
from __future__ import annotations

from typing import Any, Dict, Set, Tuple

from flask import Request


def parse_strict_json_object(request: Request, *, allowed_keys: Set[str]) -> Tuple[Dict[str, Any], str | None]:
    if not request.is_json:
        return {}, "ace_local_dev_json_required"
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return {}, "ace_local_dev_json_object_required"
    extra = set(data.keys()) - allowed_keys
    if extra:
        return {}, "ace_local_dev_unexpected_json_fields"
    return data, None
