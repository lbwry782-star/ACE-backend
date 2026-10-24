"""GitBackendPushPort — controlled git CLI push to an explicit remote (tests / future wiring only)."""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from engine.ace_local_dev_publish_candidate_registry import normalize_full_sha
from engine.ace_local_dev_publish_real_execute_ports import GitBackendPushOutcome
from engine.ace_local_dev_publish_uri_candidate_prepare import FORBIDDEN_MIXED_COMMIT_SHA

logger = logging.getLogger(__name__)

_TEST_PATH_TOKEN = "ace-publish-git-test"
_TEST_FIXTURE_NAME = ".ace-publish-git-test-fixture"
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_REMOTE_SCHEMES = ("https://", "http://", "git@", "ssh://")
_LOCAL_PATH_LEAK_TOKENS = (
    "local-dev-infrastructure",
    "scripts/ori",
    "scripts/start-ace",
    "scripts/stop-ace",
    "ace_local_dev_",
)


class GitBackendPushPortError(Exception):
    def __init__(self, code: str, *, message: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


@dataclass(frozen=True)
class GitBackendPushPortConfig:
    """All targets explicit — no default origin/main/production URL."""

    repo_path: Path
    remote_name: str
    remote_url: str
    branch: str


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    cmd = ["git", *args]
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitBackendPushPortError(
            "ace_publish_real_git_command_failed",
            message=f"Git command failed: {exc}",
        ) from exc


def _git_ok(proc: subprocess.CompletedProcess[str], *, code: str, message: str) -> None:
    if proc.returncode == 0:
        return
    detail = (proc.stderr or proc.stdout or "").strip()
    if len(detail) > 240:
        detail = detail[:237] + "..."
    raise GitBackendPushPortError(code, message=f"{message} ({detail})" if detail else message)


def is_allowed_git_push_repo_path(repo_path: Path) -> bool:
    resolved = repo_path.resolve()
    if (resolved / _TEST_FIXTURE_NAME).is_file():
        return True
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        resolved.relative_to(temp_root)
    except ValueError:
        return False
    return _TEST_PATH_TOKEN in {p.lower() for p in resolved.parts}


def is_allowed_git_push_remote_url(remote_url: str) -> bool:
    url = (remote_url or "").strip()
    if not url:
        return False
    lower = url.lower()
    for prefix in _FORBIDDEN_REMOTE_SCHEMES:
        if lower.startswith(prefix):
            return False
    if lower.startswith("file:"):
        return _TEST_PATH_TOKEN in lower.replace("\\", "/")
    path = Path(url)
    if path.is_absolute() or url.startswith("."):
        try:
            resolved = path.resolve()
        except OSError:
            return False
        temp_root = Path(tempfile.gettempdir()).resolve()
        try:
            resolved.relative_to(temp_root)
        except ValueError:
            return False
        return _TEST_PATH_TOKEN in {p.lower() for p in resolved.parts}
    return False


def assert_git_backend_push_config_allowed(config: GitBackendPushPortConfig) -> None:
    repo = config.repo_path.resolve()
    if not repo.is_dir():
        raise GitBackendPushPortError(
            "ace_publish_real_git_repo_missing",
            message="Configured backend repository path does not exist.",
        )
    if not is_allowed_git_push_repo_path(repo):
        raise GitBackendPushPortError(
            "ace_publish_real_git_repo_not_allowed",
            message="Repository path is not an approved isolated test layout.",
        )
    remote_name = (config.remote_name or "").strip()
    branch = (config.branch or "").strip()
    if not remote_name or not branch:
        raise GitBackendPushPortError(
            "ace_publish_real_git_config_incomplete",
            message="remote_name and branch are required.",
        )
    if remote_name.lower() in ("origin", "upstream") and not is_allowed_git_push_repo_path(repo):
        raise GitBackendPushPortError(
            "ace_publish_real_git_remote_name_reserved",
            message="Reserved remote name requires isolated test repository.",
        )
    if not is_allowed_git_push_remote_url(config.remote_url):
        raise GitBackendPushPortError(
            "ace_publish_real_git_remote_url_not_allowed",
            message="Remote URL must be an isolated file:// or temp test path.",
        )


def _normalize_candidate(candidate_sha: str) -> str:
    normalized = normalize_full_sha(candidate_sha)
    if not normalized:
        raise GitBackendPushPortError(
            "ace_publish_real_git_candidate_invalid",
            message="Candidate SHA must be a full 40-character commit hash.",
        )
    return normalized


def _plan_candidate_sha(execution_plan: Dict[str, Any]) -> str:
    verified = execution_plan.get("verifiedSources") if isinstance(execution_plan.get("verifiedSources"), dict) else {}
    return str(verified.get("candidateBackendSha") or "").strip().lower()


def _validate_plan_alignment(candidate: str, execution_plan: Dict[str, Any]) -> None:
    plan_sha = _plan_candidate_sha(execution_plan)
    if not plan_sha or not _FULL_SHA_RE.match(plan_sha):
        raise GitBackendPushPortError(
            "ace_publish_real_git_plan_candidate_missing",
            message="Execution plan verifiedSources.candidateBackendSha is missing.",
        )
    if plan_sha != candidate:
        raise GitBackendPushPortError(
            "ace_publish_real_git_candidate_plan_mismatch",
            message="Candidate SHA does not match execution plan.",
        )


def _validate_forbidden_publish_content(execution_plan: Dict[str, Any], candidate: str) -> None:
    forbidden = FORBIDDEN_MIXED_COMMIT_SHA.lower()
    commits = execution_plan.get("publishCommitShas")
    if isinstance(commits, list):
        for item in commits:
            sha = str(item or "").strip().lower()
            if sha == forbidden:
                raise GitBackendPushPortError(
                    "ace_publish_real_git_forbidden_commit",
                    message="Forbidden mixed local-infra commit in publish set.",
                )
    backend_files = execution_plan.get("allowedProductFiles")
    paths: List[str] = []
    if isinstance(backend_files, dict):
        paths = [str(p) for p in (backend_files.get("backend") or [])]
    for path in paths:
        token = path.lower().replace("\\", "/")
        for leak in _LOCAL_PATH_LEAK_TOKENS:
            if leak in token:
                raise GitBackendPushPortError(
                    "ace_publish_real_git_local_path_leak",
                    message="Local-only path in publish execution plan.",
                )
    if candidate == forbidden:
        raise GitBackendPushPortError(
            "ace_publish_real_git_forbidden_commit",
            message="Candidate is the forbidden mixed commit.",
        )


def _commit_exists(repo: Path, sha: str) -> bool:
    proc = _run_git(["rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=repo)
    return proc.returncode == 0


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    proc = _run_git(["merge-base", "--is-ancestor", ancestor, descendant], cwd=repo)
    return proc.returncode == 0


def _resolve_remote_tip(repo: Path, config: GitBackendPushPortConfig) -> Optional[str]:
    proc = _run_git(
        ["ls-remote", "--heads", config.remote_url, f"refs/heads/{config.branch}"],
        cwd=repo,
    )
    if proc.returncode != 0:
        raise GitBackendPushPortError(
            "ace_publish_real_git_ls_remote_failed",
            message="Could not read remote branch state.",
        )
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = line[-1].split()
    if not parts:
        return None
    tip = normalize_full_sha(parts[0])
    return tip


def _ensure_remote_configured(repo: Path, config: GitBackendPushPortConfig) -> None:
    proc = _run_git(["remote", "get-url", config.remote_name], cwd=repo)
    if proc.returncode == 0:
        current = (proc.stdout or "").strip()
        if current != config.remote_url:
            raise GitBackendPushPortError(
                "ace_publish_real_git_remote_mismatch",
                message="Configured remote URL does not match local git remote.",
            )
        return
    add = _run_git(["remote", "add", config.remote_name, config.remote_url], cwd=repo)
    _git_ok(add, code="ace_publish_real_git_remote_add_failed", message="Failed to add git remote.")


def _analyze_remote_state(
    repo: Path,
    *,
    candidate: str,
    remote_tip: Optional[str],
    execution_plan: Dict[str, Any],
) -> GitBackendPushOutcome:
    if remote_tip and remote_tip == candidate:
        return GitBackendPushOutcome(
            ok=True,
            code="ace_publish_real_git_already_on_remote",
            message="Candidate commit is already the remote branch tip.",
            commit_sha=candidate,
            already_on_remote=True,
            idempotent=True,
        )
    verified = execution_plan.get("verifiedSources") if isinstance(execution_plan.get("verifiedSources"), dict) else {}
    baseline = normalize_full_sha(str(verified.get("productionBackendWeb") or "")) or ""
    if remote_tip:
        if _is_ancestor(repo, candidate, remote_tip) and remote_tip != candidate:
            return GitBackendPushOutcome(
                ok=False,
                code="ace_publish_real_git_remote_advanced",
                message="Remote branch advanced beyond candidate; push refused.",
                commit_sha=candidate,
            )
        if not _is_ancestor(repo, remote_tip, candidate):
            return GitBackendPushOutcome(
                ok=False,
                code="ace_publish_real_git_remote_diverged",
                message="Remote branch diverged from candidate; push refused (no force).",
                commit_sha=candidate,
            )
    elif baseline and not _is_ancestor(repo, baseline, candidate):
        return GitBackendPushOutcome(
            ok=False,
            code="ace_publish_real_git_candidate_not_after_baseline",
            message="Candidate is not a descendant of production baseline.",
            commit_sha=candidate,
        )
    return GitBackendPushOutcome(
        ok=True,
        code="ace_publish_real_git_push_ready",
        message="Remote accepts fast-forward push.",
        commit_sha=candidate,
    )


def _push_fast_forward_only(repo: Path, config: GitBackendPushPortConfig, candidate: str) -> GitBackendPushOutcome:
    refspec = f"{candidate}:refs/heads/{config.branch}"
    proc = _run_git(["push", config.remote_name, refspec], cwd=repo)
    if proc.returncode != 0:
        combined = f"{proc.stderr or ''}\n{proc.stdout or ''}".lower()
        if "non-fast-forward" in combined or "rejected" in combined:
            return GitBackendPushOutcome(
                ok=False,
                code="ace_publish_real_git_push_rejected",
                message="Push rejected (non-fast-forward). Force push is not allowed.",
                commit_sha=candidate,
            )
        return GitBackendPushOutcome(
            ok=False,
            code="ace_publish_real_git_push_failed",
            message="Git push failed.",
            commit_sha=candidate,
        )
    return GitBackendPushOutcome(
        ok=True,
        code="ace_publish_real_git_pushed",
        message="Candidate pushed to remote branch.",
        commit_sha=candidate,
        pushed=True,
    )


class CliGitBackendPushPort:
    """GitBackendPushPort using git subprocess (shell=False)."""

    def __init__(self, config: GitBackendPushPortConfig) -> None:
        assert_git_backend_push_config_allowed(config)
        self._config = config

    @property
    def config(self) -> GitBackendPushPortConfig:
        return self._config

    def push_backend_to_main(self, *, candidate_sha: str, execution_plan: Dict[str, Any]) -> GitBackendPushOutcome:
        try:
            candidate = _normalize_candidate(candidate_sha)
            _validate_plan_alignment(candidate, execution_plan)
            _validate_forbidden_publish_content(execution_plan, candidate)
            repo = self._config.repo_path.resolve()
            if not _commit_exists(repo, candidate):
                return GitBackendPushOutcome(
                    ok=False,
                    code="ace_publish_real_git_commit_missing",
                    message="Candidate commit does not exist in the configured repository.",
                    commit_sha=candidate,
                )
            _ensure_remote_configured(repo, self._config)
            remote_tip = _resolve_remote_tip(repo, self._config)
            decision = _analyze_remote_state(
                repo,
                candidate=candidate,
                remote_tip=remote_tip,
                execution_plan=execution_plan,
            )
            if not decision.ok:
                return decision
            if decision.already_on_remote or decision.idempotent:
                return decision
            return _push_fast_forward_only(repo, self._config, candidate)
        except GitBackendPushPortError as exc:
            return GitBackendPushOutcome(ok=False, code=exc.code, message=exc.message)


def build_cli_git_backend_push_port(config: GitBackendPushPortConfig) -> CliGitBackendPushPort:
    return CliGitBackendPushPort(config)
