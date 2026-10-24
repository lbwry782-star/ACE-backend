#Requires -Version 5.1
<#
  Isolated tests for LOCAL shutdown snapshot (temp git repos only; never snapshots live ACE repos).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function Initialize-TempGitRepo {
    param([Parameter(Mandatory = $true)][string]$Root)
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Push-Location -LiteralPath $Root
    try {
        git init -q 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'git_init_failed' }
        git config user.email 'shutdown-test@ace.local' 2>$null | Out-Null
        git config user.name 'Shutdown Test' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'README.md') -Value 'base' -Encoding UTF8
        git add README.md 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'git_commit_failed' }
    }
    finally {
        Pop-Location
    }
}

function Get-RepoGitFingerprint {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)
    Push-Location -LiteralPath $RepoRoot
    try {
        $head = (git rev-parse HEAD).Trim()
        $status = (git status -u --porcelain=v1 | Out-String).Trim()
        return @{ Head = $head; Status = $status }
    }
    finally {
        Pop-Location
    }
}

function New-TempAceLayout {
    $root = Join-Path $env:TEMP ("ace-shutdown-snap-{0}" -f [guid]::NewGuid().ToString('N'))
    $aceRoot = Join-Path $root 'ACE'
    $backend = Join-Path $aceRoot 'ACE-Backend'
    $frontend = Join-Path $aceRoot 'ACE-Frontend'
    $sessions = Join-Path $aceRoot '_backups\sessions'
    $registry = Join-Path $aceRoot '_registry'
    $localDev = Join-Path $aceRoot '_backups\local-dev'
    $snapRoot = Join-Path $localDev 'shutdown-snapshots'
    foreach ($d in @($backend, $frontend, $sessions, $registry, $localDev, $snapRoot)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    Set-Content -LiteralPath (Join-Path $localDev 'managed-processes.json') -Value '{"schemaVersion":2}' -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $sessions 'active-dev-cycle.json') -Value '{"schemaVersion":1,"sessionId":"ori-20260101-120000-abcdef01","cycleStatus":"OPEN"}' -Encoding UTF8
    $sessionDir = Join-Path $sessions 'ori-20260101-120000-abcdef01'
    New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $sessionDir 'cancel-dev-cycle.state.json') -Value '{"status":"OPEN"}' -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $registry 'production-registry.json') -Value '{"schemaVersion":1}' -Encoding UTF8
    'abc123  production-registry.json' | Set-Content -LiteralPath (Join-Path $registry 'production-registry.json.sha256') -Encoding ASCII
    return @{
        TempRoot     = $root
        AceRoot      = $aceRoot
        BackendRoot  = $backend
        FrontendRoot = $frontend
        SessionsRoot = $sessions
        SnapshotsRoot = $snapRoot
    }
}

Write-Host 'Test-AceLocalShutdownSnapshot (isolated temp repos)...' -ForegroundColor Cyan

$startLocalText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startLocalText -notmatch 'AceLocalShutdownSnapshot') 'not_wired_to_start_ace_local'
$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($stopText -notmatch 'AceLocalShutdownSnapshot') 'not_wired_to_stop_ace_local'

$layout = New-TempAceLayout
try {
    $be = $layout.BackendRoot
    $fe = $layout.FrontendRoot

    # modified
    'modified-content' | Set-Content -LiteralPath (Join-Path $be 'tracked-edit.txt') -Encoding UTF8
    git -C $be add tracked-edit.txt 2>$null | Out-Null

    # untracked
    'new-untracked' | Set-Content -LiteralPath (Join-Path $be 'new-untracked.txt') -Encoding UTF8

    # deleted
    Remove-Item -LiteralPath (Join-Path $be 'README.md') -Force

    # ignored not included
    '*ignored*' | Set-Content -LiteralPath (Join-Path $be '.gitignore') -Encoding UTF8
    'skip-me' | Set-Content -LiteralPath (Join-Path $be 'ignored-not-included.txt') -Encoding UTF8

    # ignored included via allowlist (*.local.json)
    '{"dev":true}' | Set-Content -LiteralPath (Join-Path $be 'settings.local.json') -Encoding UTF8

    # ENV (DPAPI)
    "ACE_TEST=secret-value`n" | Set-Content -LiteralPath (Join-Path $be '.env.local') -Encoding UTF8
    'VITE_BACKEND_URL=http://127.0.0.1:5000' | Set-Content -LiteralPath (Join-Path $fe '.env.development.local') -Encoding UTF8

    $fpBeforeBe = Get-RepoGitFingerprint -RepoRoot $be
    $fpBeforeFe = Get-RepoGitFingerprint -RepoRoot $fe

    $snap1 = New-AceLocalShutdownSnapshot -BackendRoot $be -FrontendRoot $fe -AceRoot $layout.AceRoot `
        -SessionsRoot $layout.SessionsRoot -SnapshotsRootOverride $layout.SnapshotsRoot
    Assert-True $snap1.Ok 'snapshot_verified_ok'
    Assert-True ($snap1.Status -eq 'VERIFIED') 'snapshot_status_verified'

    $root1 = $snap1.SnapshotRoot
    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'BACKEND\mirror\tracked-edit.txt')) 'modified_in_mirror'
    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'BACKEND\mirror\new-untracked.txt')) 'untracked_in_mirror'
    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'BACKEND\deleted-from-HEAD\README.md')) 'deleted_preserved'
    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'BACKEND\mirror\settings.local.json')) 'ignored_included_in_mirror'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $root1 'BACKEND\mirror\ignored-not-included.txt'))) 'ignored_not_included_absent'

    $meta1 = Get-Content -LiteralPath (Join-Path $root1 'snapshot-metadata.json') -Raw | ConvertFrom-Json
    $review = @($meta1.ignoredUserReview)
    Assert-True (@($review | Where-Object { $_.relPath -match 'ignored-not-included' }).Count -ge 1) 'ignored_not_included_reported'

    $envPlainPath = Join-Path $root1 'ENV\backend.env.local'
    Assert-True (-not (Test-Path -LiteralPath $envPlainPath)) 'env_not_plain_text_file'
    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'ENV\backend.env.local.dpapi')) 'env_dpapi_present'
    $roundEnv = Read-AceLocalShutdownEnvPlain -DpapiPath (Join-Path $root1 'ENV\backend.env.local.dpapi')
    Assert-True ($roundEnv -match 'ACE_TEST=secret-value') 'env_dpapi_roundtrip'

    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'local-state\active-dev-cycle.json')) 'local_state_active_copy'
    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'local-state\cancel-dev-cycle.state.json')) 'local_state_open_copy'
    Assert-True (Test-Path -LiteralPath (Join-Path $root1 'local-state\production-registry.json')) 'local_state_registry_copy'

    $integrity1 = Test-AceLocalShutdownSnapshotIntegrity -SnapshotRoot $root1
    Assert-True $integrity1.Ok 'integrity_ok'

    $fpAfterBe = Get-RepoGitFingerprint -RepoRoot $be
    $fpAfterFe = Get-RepoGitFingerprint -RepoRoot $fe
    Assert-True ($fpBeforeBe.Head -eq $fpAfterBe.Head) 'source_backend_head_unchanged'
    Assert-True ($fpBeforeFe.Head -eq $fpAfterFe.Head) 'source_frontend_head_unchanged'
    Assert-True ($fpBeforeBe.Status -eq $fpAfterBe.Status) 'source_backend_status_unchanged'

    # restore to temp clone
    $restoreRoot = Join-Path $layout.TempRoot 'restore-backend'
    New-Item -ItemType Directory -Path $restoreRoot -Force | Out-Null
    git clone -q $be $restoreRoot 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'git_clone_for_restore_failed'
    }
    Restore-AceLocalShutdownRepoToTempRoot -SnapshotRoot $root1 -Label 'BACKEND' -TargetRepoRoot $restoreRoot
    Assert-True (Test-Path -LiteralPath (Join-Path $restoreRoot 'new-untracked.txt')) 'restore_untracked'
    Assert-True (Test-Path -LiteralPath (Join-Path $restoreRoot 'settings.local.json')) 'restore_ignored_included'
    Assert-True (Test-Path -LiteralPath (Join-Path $restoreRoot 'tracked-edit.txt')) 'restore_modified'

    # concurrent drift during backend copy
    $layout2 = New-TempAceLayout
    try {
        $be2 = $layout2.BackendRoot
        'drift-me' | Set-Content -LiteralPath (Join-Path $be2 'drift.txt') -Encoding UTF8
        $hook = {
            Set-Content -LiteralPath (Join-Path $be2 'drift.txt') -Value 'changed-during-backup' -Encoding UTF8
        }
        $snapDrift = New-AceLocalShutdownSnapshot -BackendRoot $be2 -FrontendRoot $layout2.FrontendRoot `
            -AceRoot $layout2.AceRoot -SessionsRoot $layout2.SessionsRoot -SnapshotsRootOverride $layout2.SnapshotsRoot `
            -DuringBackendCopyHook $hook
        Assert-True (-not $snapDrift.Ok) 'concurrent_drift_fails_snapshot'
        $metaDrift = Get-Content -LiteralPath (Join-Path $snapDrift.SnapshotRoot 'snapshot-metadata.json') -Raw | ConvertFrom-Json
        Assert-True ([string]$metaDrift.status -eq 'INCOMPLETE') 'concurrent_drift_incomplete_status'
    }
    finally {
        Remove-Item -LiteralPath $layout2.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }

    # fail after backend — previous verified preserved
    $ptrBefore = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $layout.AceRoot
    Assert-True ($null -ne $ptrBefore) 'last_verified_pointer_exists'
    Assert-True ([string]$ptrBefore.snapshotId -eq $snap1.SnapshotId) 'last_verified_points_to_first'

    $snapFail = New-AceLocalShutdownSnapshot -BackendRoot $be -FrontendRoot $fe -AceRoot $layout.AceRoot `
        -SessionsRoot $layout.SessionsRoot -SnapshotsRootOverride $layout.SnapshotsRoot -SimulateFailAfterBackend
    Assert-True (-not $snapFail.Ok) 'simulated_fail_not_ok'
    $ptrAfter = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $layout.AceRoot
    Assert-True ([string]$ptrAfter.snapshotId -eq $snap1.SnapshotId) 'last_verified_unchanged_after_fail'

    # manifest hash mismatch detection
    $badIntegrity = Test-AceLocalShutdownSnapshotIntegrity -SnapshotRoot $root1 -SimulateManifestHashMismatch
    Assert-True (-not $badIntegrity.Ok) 'manifest_hash_mismatch_detected'

    # partial / incomplete folder still has metadata
    Assert-True (Test-Path -LiteralPath (Join-Path $snapFail.SnapshotRoot 'snapshot-metadata.json')) 'failed_snapshot_metadata_present'
    $metaFail = Get-Content -LiteralPath (Join-Path $snapFail.SnapshotRoot 'snapshot-metadata.json') -Raw | ConvertFrom-Json
    Assert-True ([string]$metaFail.status -eq 'INCOMPLETE') 'failed_snapshot_incomplete'
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (@($failures).Count -gt 0) {
    Write-Host ("`nFailures: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalShutdownSnapshot tests passed.' -ForegroundColor Green
exit 0
