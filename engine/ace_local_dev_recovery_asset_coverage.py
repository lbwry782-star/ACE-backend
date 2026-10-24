"""
Recovery asset coverage map (phase 5.3 — policy reference, no mutations).

Each asset: backed_up | verified | auto_restore | preserve_readonly
"""
from __future__ import annotations

from typing import Any, Dict, List

COVERAGE_BACKED_UP = "backed_up"
COVERAGE_VERIFIED = "verified"
COVERAGE_AUTO_RESTORE = "auto_restore"
COVERAGE_PRESERVE_READONLY = "preserve_readonly"
COVERAGE_RECONCILE_READONLY = "reconcile_readonly"
COVERAGE_NOT_IN_SCOPE = "not_in_recovery_engine"


def recovery_asset_coverage_table() -> List[Dict[str, Any]]:
    return [
        {
            "asset": "backend_worktree_files",
            "description": "קבצי BACKEND (mirror / deleted-from-HEAD)",
            "backedUp": True,
            "verified": True,
            "autoRestore": True,
            "onConflict": "preserve_workspace",
            "coverage": COVERAGE_AUTO_RESTORE,
        },
        {
            "asset": "frontend_worktree_files",
            "description": "קבצי FRONTEND",
            "backedUp": True,
            "verified": True,
            "autoRestore": True,
            "onConflict": "preserve_workspace",
            "coverage": COVERAGE_AUTO_RESTORE,
        },
        {
            "asset": "env_dpapi_blobs",
            "description": "גיבוי ENV מוצפן DPAPI ב-checkpoint",
            "backedUp": True,
            "verified": True,
            "autoRestore": "missing_file_dpapi_decrypt_to_original_path",
            "onConflict": "no_auto_restore",
            "coverage": COVERAGE_AUTO_RESTORE,
            "secretsInLogs": False,
        },
        {
            "asset": "git_head_branch_metadata",
            "description": "Git HEAD/branch.txt ב-checkpoint (מידע בלבד)",
            "backedUp": True,
            "verified": True,
            "autoRestore": False,
            "coverage": COVERAGE_PRESERVE_READONLY,
            "engineGitCommands": False,
        },
        {
            "asset": "uncommitted_new_files",
            "description": "קבצים חדשים שלא ב-commit",
            "backedUp": "if_in_manifest",
            "verified": True,
            "autoRestore": "if_missing_only",
            "coverage": COVERAGE_AUTO_RESTORE,
        },
        {
            "asset": "uncommitted_modified_files",
            "description": "שינויים לא committed",
            "backedUp": True,
            "verified": True,
            "autoRestore": "never_overwrite",
            "coverage": COVERAGE_AUTO_RESTORE,
        },
        {
            "asset": "deleted_files",
            "description": "מחיקות ב-workspace",
            "backedUp": True,
            "verified": True,
            "autoRestore": "delete_if_safe_only",
            "coverage": COVERAGE_AUTO_RESTORE,
        },
        {
            "asset": "recovery_ledger",
            "description": "recovery-ledger.json",
            "backedUp": False,
            "verified": "read_on_startup",
            "autoRestore": False,
            "coverage": COVERAGE_PRESERVE_READONLY,
        },
        {
            "asset": "open_incident",
            "description": "openIncident ב-Ledger",
            "backedUp": False,
            "autoRestore": False,
            "coverage": COVERAGE_PRESERVE_READONLY,
        },
        {
            "asset": "new_cancel_publish_state",
            "description": "מצב NEW/CANCEL/PUBLISH (progress/journal)",
            "backedUp": False,
            "autoRestore": False,
            "coverage": COVERAGE_RECONCILE_READONLY,
        },
        {
            "asset": "open_active_cycle",
            "description": "OPEN/ACTIVE (session cycle)",
            "backedUp": False,
            "autoRestore": False,
            "coverage": COVERAGE_PRESERVE_READONLY,
        },
        {
            "asset": "production_registry",
            "description": "production-registry.json",
            "backedUp": False,
            "autoRestore": False,
            "coverage": COVERAGE_PRESERVE_READONLY,
        },
        {
            "asset": "publish_candidate_registry",
            "description": "publish-candidate-registry.json",
            "backedUp": False,
            "autoRestore": False,
            "coverage": COVERAGE_PRESERVE_READONLY,
        },
        {
            "asset": "progress_and_external_journals",
            "description": "progress / external effects journals",
            "backedUp": False,
            "autoRestore": False,
            "coverage": COVERAGE_RECONCILE_READONLY,
        },
    ]


def asset_coverage_to_dict() -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "phase": "5.3",
        "assets": recovery_asset_coverage_table(),
    }
