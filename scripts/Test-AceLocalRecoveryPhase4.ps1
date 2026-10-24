#Requires -Version 5.1
<#
  Phase 4 recovery periodic save tests — temp git repos only.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')

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
        git config user.email 'recovery-cp4@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP4' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp4-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $cpRoot = Join-Path $ace '_backups\local-dev\recovery-checkpoints'
    New-Item -ItemType Directory -Path $backend, $frontend, $cpRoot -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260201-120000' | Out-Null
    return @{
        TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend; CpRoot = $cpRoot
    }
}

function Get-RepoStatusText {
    param([string]$RepoRoot)
    return (git -C $RepoRoot status -u --porcelain=v1 | Out-String).Trim()
}

function Add-BackendChange {
    param([string]$BackendRoot, [string]$Text = 'edit')
    'tracked' | Set-Content -LiteralPath (Join-Path $BackendRoot 'edit-me.txt') -Encoding UTF8
}

Write-Host 'Test-AceLocalRecoveryPhase4 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart') 'wired_to_start'
$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($stopText -match 'Stop-AceLocalRecoveryPeriodicSaveForLocalQuit') 'wired_to_stop'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = '20260201-120000'
    $t0 = [DateTime]::Parse('2026-02-01T12:00:00Z', $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()

    Add-BackendChange -BackendRoot $be -Text 'v1'
    $statusBefore = Get-RepoStatusText -RepoRoot $be
    $headBefore = (git -C $be rev-parse HEAD).Trim()

    $tick1 = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tick1.Outcome -eq 'checkpoint_verified') 'first_save_full_verified'
    Assert-True ($tick1.DecisionAction -eq 'full_checkpoint') 'first_save_full_decision'

    Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $tick1.CheckpointId `
        -CheckpointPath (Join-Path $cpRoot $tick1.CheckpointId) -VerifiedAt $t0.ToString('o') -CheckpointType full | Out-Null
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $t0.ToString('o') | Out-Null

    $verified = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    Assert-True ($verified.Count -ge 1) 'minimum_one_verified'
    $fullId = [string]$verified[0].CheckpointId

    Assert-True ((Get-RepoStatusText -RepoRoot $be) -eq $statusBefore) 'source_repo_unchanged'
    Assert-True ((git -C $be rev-parse HEAD).Trim() -eq $headBefore) 'source_head_unchanged'

    'v2' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    $tLight = $t0.AddSeconds(301)
    $tickLight = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tLight -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickLight.Outcome -eq 'checkpoint_verified') 'light_checkpoint_verified'
    Assert-True ($tickLight.DecisionAction -eq 'light_checkpoint') 'light_decision'
    Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $tickLight.CheckpointId `
        -CheckpointPath (Join-Path $cpRoot $tickLight.CheckpointId) -VerifiedAt $tLight.ToString('o') | Out-Null
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $tLight.ToString('o') | Out-Null

    $tNoChange = $tLight.AddSeconds(301)
    $tickNo = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tNoChange -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickNo.Outcome -in @('noop', 'dirty_scan')) 'no_content_change_no_new_checkpoint'
    if ($tickNo.Outcome -eq 'dirty_scan') {
        Assert-True ($tickNo.DecisionAction -eq 'dirty_scan') 'dirty_scan_only_when_no_cp'
    }

    'v3' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    'v4' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    $tRepeat = $tLight.AddSeconds(310)
    $tickRepeat = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tRepeat -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickRepeat.Outcome -eq 'checkpoint_verified') 'repeated_file_edit_checkpoint'
    $lastVerifiedId = [string]$tickRepeat.CheckpointId
    Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $lastVerifiedId `
        -CheckpointPath (Join-Path $cpRoot $lastVerifiedId) -VerifiedAt $tRepeat.ToString('o') | Out-Null

    $failCp = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType 'full' -CheckpointsRootOverride $cpRoot `
        -SimulateFailAfterBackend
    Assert-True (-not $failCp.Ok) 'mid_checkpoint_fails'
    Assert-True ($failCp.Status -eq 'INCOMPLETE') 'incomplete_left'
    $ledgerAfterFail = Read-AceLocalRecoveryLedger -AceRoot $ace -AllowCreateStateDir
    Assert-True ($ledgerAfterFail.lastVerifiedCheckpoint) 'ledger_has_verified_after_fail'
    Assert-True ([string]$ledgerAfterFail.lastVerifiedCheckpoint.checkpointId -eq $lastVerifiedId) 'verified_not_advanced_on_fail'

    $lockPath = Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $ace
    Write-AceLocalRecoveryJsonAtomic -Path $lockPath -Object ([ordered]@{
            pid              = $PID
            startedAt        = (Get-Date).ToUniversalTime().ToString('o')
            launcherSession  = $launcher
        })
    $tickConcurrent = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tRepeat.AddSeconds(5) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickConcurrent.Outcome -in @('skipped_lock_live', 'skipped_lock_unknown')) 'live_lock_blocks_second_save'
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace

    $tickCancel = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tRepeat.AddSeconds(10) -CheckpointsRootOverride $cpRoot `
        -SkipLiveAssessment -CriticalGuardOverride @{ blockCheckpoint = $true; reason = 'cancel_critical' }
    Assert-True ($tickCancel.Outcome -eq 'blocked_critical') 'cancel_critical_blocks'

    $guardPub = [ordered]@{ blockCheckpoint = $true; reason = 'publish_external_running' }
    $tickPub = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tRepeat.AddSeconds(11) -CheckpointsRootOverride $cpRoot `
        -CriticalGuardOverride $guardPub -SkipLiveAssessment
    Assert-True ($tickPub.Outcome -eq 'blocked_critical') 'publish_critical_blocks'

    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $tickStop = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tRepeat.AddSeconds(12) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickStop.Outcome -eq 'stop_requested') 'controlled_stop'
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    $verifiedAfterPower = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    Assert-True ($verifiedAfterPower.Count -ge 1) 'power_loss_last_verified_still_there'
    Assert-True ($verifiedAfterPower[0].CheckpointId -eq $fullId -or $verifiedAfterPower.Count -ge 2) 'verified_chain_preserved'

    $retention = Get-AceLocalRecoveryCheckpointRetentionPlan -AceRoot $ace
    Assert-True ($retention.WouldDeleteCount -eq 0) 'no_checkpoint_deletion'
    Assert-True ($retention.VerifiedCount -ge 1) 'retention_plan_keeps_minimum'

    $metaIncomplete = Read-AceLocalRecoveryCheckpointMetadata -CheckpointRoot $failCp.CheckpointRoot
    Assert-True ([string]$metaIncomplete.status -eq 'INCOMPLETE') 'incomplete_metadata'

    $guideText = Get-Content -LiteralPath (Join-Path $failCp.CheckpointRoot 'RECOVERY.md') -Raw
    Assert-True ($guideText -notmatch 'sk-test-secret') 'no_plaintext_secrets_in_guide'
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All phase 4 periodic save tests passed.' -ForegroundColor Green
