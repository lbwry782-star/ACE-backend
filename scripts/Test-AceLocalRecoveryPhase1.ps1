#Requires -Version 5.1
<#
  Phase 1 recovery tests — temp git repos only; no live ACE, no checkpoints, no restore.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryDirtyDetectLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryCheckpointPolicyLib.ps1')

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
        git config user.email 'recovery-test@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery Test' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally {
        Pop-Location
    }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-p1-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    New-Item -ItemType Directory -Path $backend, $frontend -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    return @{ TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend }
}

Write-Host 'Test-AceLocalRecoveryPhase1 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -notmatch 'Initialize-AceLocalRecoveryLedger') 'start_does_not_init_recovery_ledger'
Assert-True ($startText -notmatch 'run_conditional_auto_recovery') 'start_does_not_run_restore_engine'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend

    $ledger = Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260101-120000'
    Assert-True ($ledger.schemaVersion -eq 1) 'ledger_new_schema'
    Assert-True ($ledger.launcherSession -eq '20260101-120000') 'ledger_launcher_session'

    $read = Read-AceLocalRecoveryLedger -AceRoot $ace -AllowCreateStateDir
    Assert-True ($null -ne $read) 'ledger_existing_read'
    Assert-True ((Test-AceLocalRecoveryLedgerValid -Ledger $read).Ok) 'ledger_valid'

    $path = Get-AceLocalRecoveryLedgerPath -AceRoot $ace
    $hashBefore = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint ('a' * 64)
    $hashAfter = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    Assert-True ($hashBefore -ne $hashAfter) 'ledger_atomic_write_changed'
    $tmpLeft = @(Get-ChildItem -LiteralPath (Split-Path -Parent $path) -Filter 'recovery-session-ledger.json.tmp.*' -ErrorAction SilentlyContinue)
    Assert-True ($tmpLeft.Count -eq 0) 'ledger_no_tmp_leftover'

    Set-Content -LiteralPath $path -Value '{ broken' -Encoding UTF8
    Assert-True ($null -eq (Read-AceLocalRecoveryLedger -AceRoot $ace)) 'ledger_corrupt_returns_null'

    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260101-120000'
    Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId 'cp-test-1' -CheckpointPath 'D:\fake\path'
    $scan0 = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    $fp0 = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scan0
    Assert-True (-not (Test-AceLocalRecoveryWorkspaceHasWorkingTreeChanges -WorkspaceScan $scan0)) 'no_changes_initially'

    'modified' | Set-Content -LiteralPath (Join-Path $be 'tracked.txt') -Encoding UTF8
    git -C $be add tracked.txt 2>$null | Out-Null
    $scan1 = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True (@($scan1.Backend.Entries | Where-Object { $_.changeType -ne 'ignored_included' }).Count -ge 1) 'modified_detected'
    $fp1 = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scan1
    Assert-True (Test-AceLocalRecoveryWorkspaceDirtyChanged -PreviousFingerprint $fp0 -CurrentFingerprint $fp1) 'modified_changes_fingerprint'

    'new' | Set-Content -LiteralPath (Join-Path $be 'brand-new.txt') -Encoding UTF8
    $scan2 = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True (@($scan2.Backend.Entries | Where-Object { $_.path -match 'brand-new' }).Count -ge 1) 'untracked_detected'

    Remove-Item -LiteralPath (Join-Path $be 'base.txt') -Force
    $scan3 = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True (@($scan3.Backend.Entries | Where-Object { $_.changeType -eq 'deleted' }).Count -ge 1) 'deleted_detected'

    @(
        'prefs.local.json'
        'skip-me.txt'
    ) | Set-Content -LiteralPath (Join-Path $be '.gitignore') -Encoding UTF8
    '{"x":1}' | Set-Content -LiteralPath (Join-Path $be 'prefs.local.json') -Encoding UTF8
    'skip' | Set-Content -LiteralPath (Join-Path $be 'skip-me.txt') -Encoding UTF8
    $scan4 = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True (@($scan4.Backend.IgnoredIncluded | Where-Object { $_.RelPath -match 'prefs.local.json' }).Count -eq 1) 'ignored_allowlist_included'
    Assert-True (@($scan4.Backend.IgnoredExcluded | Where-Object { $_.RelPath -match 'skip-me' }).Count -ge 1) 'ignored_not_in_allowlist_reported'

    git -C $be commit -q -m 'second' --allow-empty 2>$null | Out-Null
    $headBefore = (git -C $be rev-parse HEAD).Trim()
    $scanHead = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True ($scanHead.Backend.Head -eq $headBefore) 'head_captured'
    $fpHead1 = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scanHead
    git -C $be commit -q -m 'third' --allow-empty 2>$null | Out-Null
    $scanHead2 = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    $fpHead2 = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scanHead2
    Assert-True (Test-AceLocalRecoveryWorkspaceDirtyChanged -PreviousFingerprint $fpHead1 -CurrentFingerprint $fpHead2) 'head_change_fingerprint'

    git -C $be checkout -q -b feature-recovery-test 2>$null | Out-Null
    $scanBranch = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True ($scanBranch.Backend.Branch -eq 'feature-recovery-test') 'branch_captured'

    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260101-120000'
    $ledgerForTiming = Read-AceLocalRecoveryLedger -AceRoot $ace -AllowCreateStateDir
    $now = [DateTime]::UtcNow
    $decNone = Get-AceLocalRecoveryCheckpointTimingDecision -Ledger $ledgerForTiming -NowUtc $now `
        -WorkspaceDirtyChanged $false -HasWorkingTreeChanges $false
    Assert-True ($decNone.Action -eq 'dirty_scan') 'timing_dirty_scan_due_first'

    $ledgerForTiming.lastDirtyScanAt = $now.AddSeconds(-130).ToString('o')
    $decLight = Get-AceLocalRecoveryCheckpointTimingDecision -Ledger $ledgerForTiming -NowUtc $now `
        -WorkspaceDirtyChanged $true -HasWorkingTreeChanges $true
    Assert-True ($decLight.Action -in @('dirty_scan', 'light_checkpoint')) 'timing_light_or_scan_when_dirty'

    if (-not $ledgerForTiming.timingState) {
        $ledgerForTiming | Add-Member -NotePropertyName 'timingState' -NotePropertyValue ([PSCustomObject]@{}) -Force
    }
    $ledgerForTiming.timingState | Add-Member -NotePropertyName 'lastFullCheckpointDueAt' -NotePropertyValue ($now.AddSeconds(-1900).ToString('o')) -Force
    $ledgerForTiming.timingState | Add-Member -NotePropertyName 'lastLightCheckpointDueAt' -NotePropertyValue ($now.AddSeconds(-5).ToString('o')) -Force
    $ledgerForTiming.lastDirtyScanAt = $now.AddSeconds(-5).ToString('o')
    $decFull = Get-AceLocalRecoveryCheckpointTimingDecision -Ledger $ledgerForTiming -NowUtc $now `
        -WorkspaceDirtyChanged $false -HasWorkingTreeChanges $true
    Assert-True ($decFull.Action -eq 'full_checkpoint') 'timing_full_checkpoint_due'

    $policy = Get-AceLocalRecoveryCheckpointPolicy
    Assert-True ($policy.dirtyScanIntervalSeconds -eq 120) 'policy_dirty_interval'
    Assert-True ($policy.minimumVerifiedCheckpointRetention -eq 5) 'policy_retention'

    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession 'sess-clean'
    Register-AceLocalRecoveryCleanShutdownComplete -AceRoot $ace -LauncherSession 'sess-clean' -ShutdownVerified $true
    Assert-True (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $ace -LauncherSession 'sess-clean') 'clean_shutdown_recorded'

    $interrupted = $false
    try {
        Register-AceLocalRecoveryCleanShutdownComplete -AceRoot $ace -LauncherSession 'sess-clean' -ShutdownVerified $false
    }
    catch {
        $interrupted = $true
    }
    Assert-True $interrupted 'clean_shutdown_not_verified_rejected'

    $mismatch = $false
    try {
        Register-AceLocalRecoveryCleanShutdownComplete -AceRoot $ace -LauncherSession 'wrong-session' -ShutdownVerified $true
    }
    catch {
        $mismatch = $true
    }
    Assert-True $mismatch 'clean_shutdown_session_mismatch'

    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260101-999999'
    Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId 'preserve-me' -CheckpointPath 'C:\temp\cp'
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260101-888888'
    $preserved = Read-AceLocalRecoveryLedger -AceRoot $ace -AllowCreateStateDir
    Assert-True ($preserved.lastVerifiedCheckpoint.checkpointId -eq 'preserve-me') 'checkpoint_preserved_across_reinit'

    $fpBeforeRepo = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)
    $null = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    $fpAfterRepo = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)
    Assert-True ($fpBeforeRepo -eq $fpAfterRepo) 'source_repo_unchanged_by_scan'
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (@($failures).Count -gt 0) {
    Write-Host ("`nFailures: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalRecovery phase 1 tests passed.' -ForegroundColor Green
exit 0
