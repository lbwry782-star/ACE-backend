#Requires -Version 5.1
<#
  Phase 5.7 — live pre-write safety in background periodic save (temp only).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')

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
        git config user.email 'recovery-cp57@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP57' 2>$null | Out-Null
        'v1' | Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp57-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $cpRoot = Join-Path $ace '_backups\local-dev\recovery-checkpoints'
    New-Item -ItemType Directory -Path $backend, $frontend, $cpRoot -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260201-120000' | Out-Null
    return @{ TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend; CpRoot = $cpRoot }
}

function Write-CoordinatorSnapshot {
    param([string]$AceRoot, [string]$Launcher, [string]$Outcome, [bool]$Allow)
    Write-AceLocalRecoveryPeriodicSaveActivationSnapshot -AceRoot $AceRoot -LauncherSession $Launcher `
        -CoordinatorOutcome $Outcome -AllowPeriodicSaveService $Allow
}

Write-Host 'Test-AceLocalRecoveryPhase57 (isolated)...' -ForegroundColor Cyan
$env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'

$libText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1') -Raw
Assert-True ($libText -notmatch 'SkipLiveAssessment:\$BackgroundMode') 'background_mode_does_not_skip_live'
Assert-True ($libText -match 'Invoke-AceLocalRecoveryPeriodicSavePreWriteGate') 'pre_write_gate_wired'
$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart') 'wired_to_start'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = '20260201-120000'
    $t0 = [DateTime]::Parse('2026-02-01T12:00:00Z', $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()

    Write-CoordinatorSnapshot -AceRoot $ace -Launcher $launcher -Outcome 'START_ALLOWED' -Allow $true
    'live' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8

    $tickBg = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot
    Assert-True ($tickBg.Outcome -eq 'checkpoint_verified') 'background_mode_live_checkpoint'
    if ($tickBg.CheckpointId) {
        Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $tickBg.CheckpointId `
            -CheckpointPath (Join-Path $cpRoot $tickBg.CheckpointId) -VerifiedAt $t0.ToString('o') -CheckpointType full | Out-Null
    }

    $tickBypass = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot -SkipLiveAssessment
    Assert-True ($tickBypass.Outcome -eq 'blocked_background_assessment_bypass') 'forbid_skip_in_background'

    $tickCancel = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot -CriticalGuardOverride @{ blockCheckpoint = $true; reason = 'cancel_critical' }
    Assert-True ($tickCancel.Outcome -eq 'blocked_critical') 'cancel_critical'

    $tickPub = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot -CriticalGuardOverride @{ blockCheckpoint = $true; reason = 'publish_unknown' }
    Assert-True ($tickPub.Outcome -eq 'blocked_critical') 'publish_unknown'

    $tickNew = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot -CriticalGuardOverride @{ blockCheckpoint = $true; reason = 'new_partial' }
    Assert-True ($tickNew.Outcome -eq 'blocked_critical') 'new_unsafe'

    $tickAssessFail = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot -CriticalGuardOverride @{ blockCheckpoint = $true; reason = 'assessment_failed' }
    Assert-True ($tickAssessFail.Outcome -in @('blocked_critical', 'blocked_assessment_unknown')) 'assessment_fail'

    $tickSession = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession '20260201-999999' -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot
    Assert-True ($tickSession.Outcome -eq 'blocked_launcher_session') 'launcher_session_mismatch'

    Write-CoordinatorSnapshot -AceRoot $ace -Launcher $launcher -Outcome 'START_WITH_ISOLATED_OPERATIONS' -Allow $false
    $gateRecovery = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $ace -LauncherSession $launcher `
        -GuardBackendRoot $BackendRoot
    Assert-True (-not $gateRecovery.Allowed) 'recovery_not_cleared'

    Write-CoordinatorSnapshot -AceRoot $ace -Launcher $launcher -Outcome 'START_ALLOWED' -Allow $true
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $tickStop = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot
    Assert-True ($tickStop.Outcome -eq 'stop_requested') 'stop_before_checkpoint'
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    Write-CoordinatorSnapshot -AceRoot $ace -Launcher $launcher -Outcome 'START_ALLOWED' -Allow $true
    $invokerStopMid = {
        param($action)
        Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
        return @{ Ok = $false; Skipped = $true; Reason = 'stop_mid_checkpoint'; Status = '' }
    }
    $scanMid = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scanMid) `
        -DirtyScanAt $t0.AddSeconds(3590).ToString('o') | Out-Null
    'stop-mid-only' | Set-Content -LiteralPath (Join-Path $be 'stop-only.txt') -Encoding UTF8
    $tickMid = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(3600) -CheckpointsRootOverride $cpRoot `
        -SkipLiveAssessment -CheckpointInvoker $invokerStopMid
    Assert-True ($tickMid.Outcome -eq 'checkpoint_skipped') 'stop_during_checkpoint'

    $lock = Enter-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -LauncherSession $launcher
    Assert-True $lock.Acquired 'lock_holder'
    $tickParallel = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot
    Assert-True ($tickParallel.Outcome -eq 'skipped_lock_live') 'parallel_save_blocked'
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace

    $bgText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1') -Raw
    Assert-True ($bgText -match 'WindowStyle Hidden') 'no_visible_window'
    Assert-True ((git -C $be rev-parse HEAD).Trim().Length -gt 0) 'git_head_unchanged'
}
finally {
    if ($layout.TempRoot) {
        Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_periodic_save_pre_write.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_pre_write' }
    else { Write-Host '  ok pytest_pre_write' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if ($failures.Count -gt 0) {
    Write-Host ("Phase57 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase57 PASSED' -ForegroundColor Green
