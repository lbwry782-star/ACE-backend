"""
Reject client-forged LOCAL/environment binding fields on Builder job records.

Product-owned: does not import untracked LOCAL modules.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

CLIENT_FORGED_BINDING_FIELDS = frozenset(
    {
        "localDevInstanceId",
        "localDevBound",
        "localDevStartGeneration",
        "environment",
        "aceEnvironment",
        "localScope",
    }
)


def strip_client_binding_fields(fields: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(fields, Mapping):
        return out
    for key, value in fields.items():
        if not key or value is None:
            continue
        if str(key) in CLIENT_FORGED_BINDING_FIELDS:
            continue
        out[str(key)] = str(value)
    return out
