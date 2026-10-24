#Requires -Version 5.1
<#
  Phase 4.1 safety fixes — temp git repos + timing/lock/guard unit checks.
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
        git config user.email 'recovery-cp41@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP41' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp41-{0}" -f [guid]::NewGuid().ToString('N'))
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

Write-Host 'Test-AceLocalRecoveryPhase41...' -ForegroundColor Cyan

$t0 = [DateTime]::Parse('2026-02-01T12:00:00Z', $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
$ledgerShell = New-AceLocalRecoveryLedgerShell -LauncherSession '20260201-120000'
$ledgerShell.lastFullVerifiedCheckpoint = [ordered]@{
    checkpointId   = 'cp-full-1'
    checkpointPath = 'C:\fake\full'
    verifiedAt     = $t0.ToString('o')
}
$ledgerShell.lastVerifiedCheckpoint = [ordered]@{
    checkpointId   = 'cp-light-1'
    checkpointPath = 'C:\fake\light'
    verifiedAt     = $t0.AddMinutes(5).ToString('o')
}
$dLightOnly = Get-AceLocalRecoveryPeriodicSaveTimingDecision -Ledger $ledgerShell -NowUtc $t0.AddMinutes(6) `
    -WorkspaceDirtyChanged $true -HasWorkingTreeChanges $true
Assert-True ($dLightOnly.Action -ne 'full_checkpoint') 'light_does_not_reset_full_timer'
Assert-True ($dLightOnly.SecondsSinceFullVerified -lt 1800) 'full_clock_still_running'

$dFullDue = Get-AceLocalRecoveryPeriodicSaveTimingDecision -Ledger $ledgerShell -NowUtc $t0.AddMinutes(31) `
    -WorkspaceDirtyChanged $true -HasWorkingTreeChanges $true
Assert-True ($dFullDue.Action -eq 'full_checkpoint') 'full_due_after_30min_from_last_full'

$ledgerNoFull = New-AceLocalRecoveryLedgerShell -LauncherSession '20260201-120000'
$dFirst = Get-AceLocalRecoveryPeriodicSaveTimingDecision -Ledger $ledgerNoFull -NowUtc $t0 `
    -WorkspaceDirtyChanged $true -HasWorkingTreeChanges $true
Assert-True ($dFirst.Action -eq 'full_checkpoint') 'first_checkpoint_is_full'
Assert-True ($dFirst.Reasons -contains 'first_full_required_no_prior_full_verified') 'first_full_reason'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = '20260201-120000'

    'tracked' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    $tickFull = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickFull.DecisionAction -eq 'full_checkpoint') 'integration_first_full'

    Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $tickFull.CheckpointId `
        -CheckpointPath (Join-Path $cpRoot $tickFull.CheckpointId) -VerifiedAt $t0.ToString('o') -CheckpointType full | Out-Null
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $t0.ToString('o') | Out-Null

    $fullVerifiedAt = $t0
    for ($i = 1; $i -le 5; $i++) {
        "light-$i" | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
        $tLight = $fullVerifiedAt.AddSeconds(300 * $i)
        $tickL = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
            -LauncherSession $launcher -NowUtc $tLight -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
        Assert-True ($tickL.DecisionAction -eq 'light_checkpoint') ("light_interval_$i")
        Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $tickL.CheckpointId `
            -CheckpointPath (Join-Path $cpRoot $tickL.CheckpointId) -VerifiedAt $tLight.ToString('o') -CheckpointType light | Out-Null
        Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
            (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
            -DirtyScanAt $tLight.ToString('o') | Out-Null
    }
    'before-full-again' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    $tFullAgain = $fullVerifiedAt.AddSeconds(1801)
    $tickFull2 = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tFullAgain -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickFull2.DecisionAction -eq 'full_checkpoint') 'full_after_lights_every_5min'

    $staleGatePath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'local-shutdown-gate.json'
    Write-AceLocalRecoveryJsonAtomic -Path $staleGatePath -Object ([ordered]@{
            observations = @()
            gateStatus   = 'RUNNING'
            updatedAt    = '2020-01-01T00:00:00Z'
        })
    $tickStale = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $tFullAgain.AddSeconds(10) -CheckpointsRootOverride $cpRoot `
        -SkipLiveAssessment -CriticalGuardOverride @{ blockCheckpoint = $true; reason = 'live_critical' }
    Assert-True ($tickStale.Outcome -eq 'blocked_critical') 'stale_gate_file_not_used_for_allow'

    $lockPath = Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $ace
    Write-AceLocalRecoveryJsonAtomic -Path $lockPath -Object ([ordered]@{
            pid             = 999999
            startedAt       = '2026-01-01T00:00:00Z'
            launcherSession = $launcher
        })
    $assessDead = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $ace -ExpectedLauncherSession $launcher
    Assert-True ($assessDead.Classification -eq 'stale_dead_pid') 'crash_lock_classified_dead_pid'
    $enterAfterCrash = Enter-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -LauncherSession $launcher
    Assert-True $enterAfterCrash.Acquired 'stale_dead_lock_released'
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace

    Write-AceLocalRecoveryJsonAtomic -Path $lockPath -Object ([ordered]@{
            pid             = $PID
            startedAt       = (Get-Date).ToUniversalTime().ToString('o')
            launcherSession = $launcher
        })
    $assessLive = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $ace -ExpectedLauncherSession $launcher
    Assert-True ($assessLive.Classification -eq 'live') 'live_lock_preserved'
    $enterLive = Enter-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -LauncherSession $launcher
    Assert-True (-not $enterLive.Acquired) 'active_lock_not_stolen'
    Assert-True ($enterLive.Reason -eq 'lock_live') 'active_lock_reason'
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace

    $verifiedBefore = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    $firstVerifiedId = [string]$verifiedBefore[0].CheckpointId
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All phase 4.1 safety tests passed.' -ForegroundColor Green
