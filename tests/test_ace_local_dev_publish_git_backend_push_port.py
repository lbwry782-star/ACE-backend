"""
GitBackendPushPort — isolated temp bare repositories only.

Run: python -m unittest tests.test_ace_local_dev_publish_git_backend_push_port -v
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

from engine.ace_local_dev_publish_execute_adapters import PublishExecuteContext
from engine.ace_local_dev_publish_git_backend_push_port import (
    CliGitBackendPushPort,
    GitBackendPushPortConfig,
    GitBackendPushPortError,
    assert_git_backend_push_config_allowed,
    build_cli_git_backend_push_port,
    is_allowed_git_push_remote_url,
)
from engine.ace_local_dev_publish_real_execute_adapters import RealPublishExecuteAdapters
from engine.ace_local_dev_publish_real_execute_ports import RealPublishExecutePorts
from engine.ace_local_dev_publish_uri_candidate_prepare import FORBIDDEN_MIXED_COMMIT_SHA
from tests.test_ace_local_dev_publish_real_execute_adapters import MockPorts, _bundle

GIT = ["git"]


def _git(cwd: Path, *args: str) -> None:
    proc = subprocess.run(
        GIT + list(args),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr or proc.stdout}")


def _plan(candidate: str, *, baseline: str = "0" * 40, extra_paths=None) -> dict:
    return {
        "verifiedSources": {
            "candidateBackendSha": candidate,
            "productionBackendWeb": baseline,
        },
        "allowedProductFiles": {"backend": list(extra_paths or ["engine/product.py"])},
        "publishCommitShas": [candidate],
    }


class TempGitPushLayout:
    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ace-publish-git-test-layout-"))
        self.bare = self.root / "remote.git"
        self.repo = self.root / "backend"
        self.baseline_sha = ""
        self.candidate_sha = ""
        self.remote_name = "testpublish"
        self.branch = "release"

    def setup(self) -> None:
        _git(self.root, "init", "--bare", str(self.bare))
        self.repo.mkdir()
        _git(self.repo, "init")
        (self.repo / ".ace-publish-git-test-fixture").write_text("ok\n", encoding="utf-8")
        (self.repo / "engine").mkdir()
        (self.repo / "engine" / "product.py").write_text("v1\n", encoding="utf-8")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-m", "baseline")
        self.baseline_sha = subprocess.check_output(
            GIT + ["rev-parse", "HEAD"], cwd=self.repo, text=True, shell=False
        ).strip()
        (self.repo / "engine" / "product.py").write_text("v2\n", encoding="utf-8")
        _git(self.repo, "add", "engine/product.py")
        _git(self.repo, "commit", "-m", "candidate")
        self.candidate_sha = subprocess.check_output(
            GIT + ["rev-parse", "HEAD"], cwd=self.repo, text=True, shell=False
        ).strip()
        remote_url = self.bare.as_uri()
        _git(self.repo, "remote", "add", self.remote_name, remote_url)

    def config(self) -> GitBackendPushPortConfig:
        return GitBackendPushPortConfig(
            repo_path=self.repo,
            remote_name=self.remote_name,
            remote_url=self.bare.as_uri(),
            branch=self.branch,
        )

    def remote_tip(self) -> str:
        out = subprocess.check_output(
            GIT + ["ls-remote", self.bare.as_uri(), f"refs/heads/{self.branch}"],
            cwd=self.repo,
            text=True,
            shell=False,
        ).strip()
        if not out:
            return ""
        return out.split()[0]

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class GitBackendPushPortTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = TempGitPushLayout()
        self.layout.setup()
        self.port = build_cli_git_backend_push_port(self.layout.config())

    def tearDown(self) -> None:
        self.layout.cleanup()

    def test_successful_push_to_bare_remote(self) -> None:
        plan = _plan(self.layout.candidate_sha, baseline=self.layout.baseline_sha)
        out = self.port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=plan,
        )
        self.assertTrue(out.ok)
        self.assertTrue(out.pushed)
        self.assertEqual(self.layout.remote_tip(), self.layout.candidate_sha)

    def test_idempotent_when_sha_already_published(self) -> None:
        plan = _plan(self.layout.candidate_sha, baseline=self.layout.baseline_sha)
        first = self.port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=plan,
        )
        self.assertTrue(first.ok)
        second = self.port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=plan,
        )
        self.assertTrue(second.ok)
        self.assertTrue(second.already_on_remote)
        self.assertTrue(second.idempotent)

    def test_missing_commit_sha(self) -> None:
        missing = "f" * 40
        out = self.port.push_backend_to_main(
            candidate_sha=missing,
            execution_plan=_plan(missing, baseline=self.layout.baseline_sha),
        )
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_git_commit_missing")

    def test_candidate_plan_mismatch(self) -> None:
        out = self.port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=_plan("a" * 40, baseline=self.layout.baseline_sha),
        )
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_git_candidate_plan_mismatch")

    def test_remote_advanced_rejected(self) -> None:
        plan = _plan(self.layout.candidate_sha, baseline=self.layout.baseline_sha)
        _git(self.layout.repo, "push", self.layout.remote_name, f"{self.layout.candidate_sha}:refs/heads/{self.layout.branch}")
        (self.layout.repo / "engine" / "product.py").write_text("v3\n", encoding="utf-8")
        _git(self.layout.repo, "add", "engine/product.py")
        _git(self.layout.repo, "commit", "-m", "ahead on remote")
        ahead = subprocess.check_output(GIT + ["rev-parse", "HEAD"], cwd=self.layout.repo, text=True, shell=False).strip()
        _git(self.layout.repo, "push", self.layout.remote_name, f"{ahead}:refs/heads/{self.layout.branch}")
        out = self.port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=plan,
        )
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_git_remote_advanced")

    def test_no_force_push_in_source(self) -> None:
        mod_path = Path(importlib.import_module("engine.ace_local_dev_publish_git_backend_push_port").__file__)
        text = mod_path.read_text(encoding="utf-8")
        self.assertNotIn("--force", text)
        self.assertNotRegex(text, r'\["push"[^\]]*"-f"')
        self.assertNotRegex(text, r'\["push"[^\]]*"--force"')

    def test_external_remote_url_blocked(self) -> None:
        cfg = GitBackendPushPortConfig(
            repo_path=self.layout.repo,
            remote_name="testpublish",
            remote_url="https://github.com/example/repo.git",
            branch="main",
        )
        with self.assertRaises(GitBackendPushPortError) as ctx:
            assert_git_backend_push_config_allowed(cfg)
        self.assertEqual(ctx.exception.code, "ace_publish_real_git_remote_url_not_allowed")

    def test_fixture_cannot_point_to_github(self) -> None:
        self.assertFalse(is_allowed_git_push_remote_url("https://github.com/org/repo.git"))

    def test_local_infra_path_leak_blocked(self) -> None:
        plan = _plan(
            self.layout.candidate_sha,
            baseline=self.layout.baseline_sha,
            extra_paths=["scripts/Start-AceLocal.ps1"],
        )
        out = self.port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=plan,
        )
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_git_local_path_leak")

    def test_forbidden_mixed_commit_blocked(self) -> None:
        forbidden = FORBIDDEN_MIXED_COMMIT_SHA
        plan = _plan(forbidden, baseline=self.layout.baseline_sha)
        plan["publishCommitShas"] = [forbidden]
        out = self.port.push_backend_to_main(candidate_sha=forbidden, execution_plan=plan)
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_git_forbidden_commit")

    def test_git_push_failure_non_fast_forward(self) -> None:
        plan = _plan(self.layout.candidate_sha, baseline=self.layout.baseline_sha)
        _git(self.layout.repo, "push", self.layout.remote_name, f"{self.layout.baseline_sha}:refs/heads/{self.layout.branch}")
        (self.layout.repo / "engine" / "product.py").write_text("side\n", encoding="utf-8")
        _git(self.layout.repo, "add", "engine/product.py")
        _git(self.layout.repo, "commit", "-m", "side")
        side = subprocess.check_output(GIT + ["rev-parse", "HEAD"], cwd=self.layout.repo, text=True, shell=False).strip()
        _git(self.layout.repo, "push", self.layout.remote_name, f"{side}:refs/heads/{self.layout.branch}")
        out = self.port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=plan,
        )
        self.assertFalse(out.ok)
        self.assertIn(
            out.code,
            ("ace_publish_real_git_remote_advanced", "ace_publish_real_git_remote_diverged", "ace_publish_real_git_push_rejected"),
        )

    def test_module_has_no_registry_or_render_imports(self) -> None:
        mod = importlib.import_module("engine.ace_local_dev_publish_git_backend_push_port")
        text = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("production_registry", text)
        self.assertNotIn("RenderApi", text)
        self.assertNotIn("github.com", text.lower())

    def test_real_repo_path_blocked_without_fixture(self) -> None:
        backend_root = Path(__file__).resolve().parent.parent
        cfg = GitBackendPushPortConfig(
            repo_path=backend_root,
            remote_name="testpublish",
            remote_url=self.layout.bare.as_uri(),
            branch="main",
        )
        with self.assertRaises(GitBackendPushPortError):
            assert_git_backend_push_config_allowed(cfg)

    def test_integration_with_real_publish_adapters(self) -> None:
        mocks = MockPorts()
        ports = mocks.build()
        ports = RealPublishExecutePorts(
            git_backend_push=self.port,
            render_web=ports.render_web,
            render_worker=ports.render_worker,
            github_pages=ports.github_pages,
            production_verify=ports.production_verify,
            registry_update=ports.registry_update,
        )
        adapters = RealPublishExecuteAdapters(ports)
        pre = _bundle()
        pre.candidate_backend_sha = self.layout.candidate_sha
        pre.execution_plan = _plan(self.layout.candidate_sha, baseline=self.layout.baseline_sha)
        ctx = PublishExecuteContext(precheck=pre, operation_id="git-port-op")
        self.assertTrue(adapters.prepare_isolated_release(ctx).ok)
        res = adapters.git_push_backend(ctx)
        self.assertTrue(res.ok)
        self.assertEqual(mocks.web_calls, 0)
        self.assertEqual(mocks.registry_calls, 0)

    def test_git_failure_mid_operation_ls_remote(self) -> None:
        bad_dir = self.layout.root / "ace-publish-git-test-not-a-repo"
        bad_dir.mkdir()
        cfg = GitBackendPushPortConfig(
            repo_path=self.layout.repo,
            remote_name="brokenremote",
            remote_url=bad_dir.as_uri(),
            branch=self.layout.branch,
        )
        port = build_cli_git_backend_push_port(cfg)
        out = port.push_backend_to_main(
            candidate_sha=self.layout.candidate_sha,
            execution_plan=_plan(self.layout.candidate_sha, baseline=self.layout.baseline_sha),
        )
        self.assertFalse(out.ok)
        self.assertEqual(out.code, "ace_publish_real_git_ls_remote_failed")

    def test_routes_still_not_wired(self) -> None:
        routes = Path(importlib.import_module("engine.ace_local_dev_routes").__file__).read_text(encoding="utf-8")
        self.assertNotIn("ace_local_dev_publish_git_backend_push_port", routes)
        self.assertNotIn("CliGitBackendPushPort", routes)


if __name__ == "__main__":
    unittest.main()
