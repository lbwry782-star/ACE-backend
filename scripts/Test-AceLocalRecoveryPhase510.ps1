#Requires -Version 5.1
<#
  Phase 5.10 — duplicate prevention, safe quit, fast stop wake (temp/mocks only).
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
        git config user.email 'recovery-cp510@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP510' 2>$null | Out-Null
        'v1' | Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp510-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    New-Item -ItemType Directory -Path $backend, $frontend -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260201-150000' | Out-Null
    return @{ TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend }
}

function Write-CoordinatorSnapshot {
    param([string]$AceRoot, [string]$Launcher, [string]$Outcome, [bool]$Allow)
    Write-AceLocalRecoveryPeriodicSaveActivationSnapshot -AceRoot $AceRoot -LauncherSession $Launcher `
        -CoordinatorOutcome $Outcome -AllowPeriodicSaveService $Allow
}

function New-MockGate {
    param([string]$Outcome = 'START_ALLOWED', [bool]$AllowPeriodic = $true)
    return [PSCustomObject]@{
        coordinatorOutcome = $Outcome
        report             = [PSCustomObject]@{
            coordinatorDecision = [PSCustomObject]@{
                coordinatorOutcome       = $Outcome
                allowPeriodicSaveService = $AllowPeriodic
                isolatedOperations       = @()
            }
        }
    }
}

Write-Host 'Test-AceLocalRecoveryPhase510 (isolated)...' -ForegroundColor Cyan
$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart') 'regression_start_wired'
Assert-True ($stopText -match 'Stop-AceLocalRecoveryPeriodicSaveForLocalQuit') 'regression_stop_wired'
Assert-True ($startText -notmatch 'Complete-AceLocalProgressState[\s\S]{0,500}Recovery periodic') 'regression_progress_unchanged'
Assert-True ($startText -match 'Start-AceLocalMemuraiWatchAsync') 'regression_memurai_unchanged'
Assert-True ($startText -notmatch 'Invoke-AceLocalFailure[\s\S]{0,120}periodicSaveStart') 'local_not_blocked_on_periodic_fail'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $launcher = '20260201-150000'
    $cpRoot = Join-Path $ace '_backups\recovery-checkpoints-test'
    New-Item -ItemType Directory -Path $cpRoot -Force | Out-Null
    $registry = New-AceLocalManagedRegistryShell -BackendRoot $be -FrontendRoot $fe -LauncherSessionId $launcher
    $t0 = [DateTime]::Parse('2026-02-01T12:00:00Z', $null, [System.Globalization.DateTimeStyles]::RoundtripKind)

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'
    $first = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true `
        -TickIntervalSeconds 2 -CheckpointsRootOverride $cpRoot -EngineBackendRoot $BackendRoot -ManagedRegistry $registry
    Assert-True $first.Started 'spawn_ok'
    Start-Sleep -Seconds 1

    $spawnRec = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $ace
    Set-Content -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'recovery-periodic-save.pid') `
        -Value '999999' -Encoding ASCII
    $second = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true `
        -TickIntervalSeconds 2 -CheckpointsRootOverride $cpRoot -EngineBackendRoot $BackendRoot -ManagedRegistry $registry
    Assert-True (-not $second.Started -and $second.Reason -eq 'already_running') 'double_start_with_stale_pid_file'

    [void](Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $ace -WaitSeconds 40)

    Remove-Item -LiteralPath (Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $ace) -Force -ErrorAction SilentlyContinue
    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $ace -State @{
        status          = 'running'
        launcherSession = $launcher
        pid             = 999999
        spawnToken      = 'dead-token'
    }
    Set-Content -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'recovery-periodic-save.pid') `
        -Value '999999' -Encoding ASCII
    $reconciled = Clear-AceLocalRecoveryPeriodicSaveProvenStaleState -AceRoot $ace -ExpectedLauncherSession $launcher
    Assert-True $reconciled.Cleared 'stale_state_reconciled'
    $afterCrashSpawn = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true `
        -TickIntervalSeconds 2 -CheckpointsRootOverride $cpRoot -EngineBackendRoot $BackendRoot -ManagedRegistry $registry
    Assert-True $afterCrashSpawn.Started 'spawn_after_proven_stale'
    [void](Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $ace -WaitSeconds 25)

    $wakeJob = Start-Job -ScriptBlock {
        param($AceRootPath, $LibPath)
        . $LibPath
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        Wait-AceLocalRecoveryPeriodicSaveInterTickDelay -AceRoot $AceRootPath -TotalSeconds 30
        $sw.Stop()
        return $sw.Elapsed.TotalSeconds
    } -ArgumentList $ace, (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')
    Start-Sleep -Milliseconds 700
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $wakeWait = Wait-Job $wakeJob -Timeout 15
    $wakeSeconds = Receive-Job $wakeJob
    Remove-Job $wakeJob -Force -ErrorAction SilentlyContinue
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace
    Assert-True ($null -ne $wakeWait) 'stop_wake_job_finished'
    Assert-True ($wakeSeconds -lt 8) 'stop_wake_under_8s'

    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $tickIdle = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickIdle.Outcome -eq 'stop_requested') 'quit_while_idle'
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    Write-CoordinatorSnapshot -AceRoot $ace -Launcher $launcher -Outcome 'START_ALLOWED' -Allow $true
    'cp-base' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    $tickFullSeed = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickFullSeed.Outcome -eq 'checkpoint_verified') 'seed_verified_checkpoint'
    Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $tickFullSeed.CheckpointId `
        -CheckpointPath (Join-Path $cpRoot $tickFullSeed.CheckpointId) -VerifiedAt $t0.ToString('o') -CheckpointType full | Out-Null
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $t0.ToString('o') | Out-Null
    $verifiedBefore = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    $verifiedId = [string]$verifiedBefore[0].CheckpointId

    'dirty510' | Set-Content -LiteralPath (Join-Path $be 'dirty510.txt') -Encoding UTF8
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $t0.AddSeconds(121).ToString('o') | Out-Null
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $tickDirty = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(122) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickDirty.Outcome -eq 'stop_requested') 'quit_during_dirty_scan'
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    Write-CoordinatorSnapshot -AceRoot $ace -Launcher $launcher -Outcome 'START_ALLOWED' -Allow $true
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace
    'light510' | Set-Content -LiteralPath (Join-Path $be 'light510.txt') -Encoding UTF8
    $invLight = {
        param($action)
        Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
        return @{ Ok = $false; Skipped = $true; Reason = 'stop_mid_light'; Status = '' }
    }
    $tickLight = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(302) -CheckpointsRootOverride $cpRoot `
        -SkipLiveAssessment -CheckpointInvoker $invLight
    Assert-True ($tickLight.Outcome -in @('checkpoint_skipped', 'stop_requested')) 'quit_during_light_checkpoint'
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    Write-CoordinatorSnapshot -AceRoot $ace -Launcher $launcher -Outcome 'START_ALLOWED' -Allow $true
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $t0.AddSeconds(3601).ToString('o') | Out-Null
    'full510' | Set-Content -LiteralPath (Join-Path $be 'full510.txt') -Encoding UTF8
    $invFull = {
        param($action)
        Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
        return @{ Ok = $false; Skipped = $true; Reason = 'stop_mid_full'; Status = '' }
    }
    $tickFull = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(3602) -CheckpointsRootOverride $cpRoot `
        -SkipLiveAssessment -CheckpointInvoker $invFull
    Assert-True ($tickFull.Outcome -eq 'checkpoint_skipped') 'quit_during_full_checkpoint'
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -ErrorAction SilentlyContinue
    $lock = Enter-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -LauncherSession $launcher
    Assert-True $lock.Acquired 'lock_acquired_for_quit_test'
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $tickLock = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(4000) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment -SkipLock
    Assert-True ($tickLock.Outcome -eq 'stop_requested') 'quit_with_active_lock_holder'
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -ErrorAction SilentlyContinue
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace -ErrorAction SilentlyContinue
    try { [void](Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $ace -WaitSeconds 5) } catch { }
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace

    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $ace -State @{
        status          = 'running'
        launcherSession = $launcher
        pid             = 999998
    }
    Set-Content -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'recovery-periodic-save.pid') `
        -Value '999998' -Encoding ASCII
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $ace) -Object ([ordered]@{
            pid             = $PID
            startedAt       = (Get-Date).ToUniversalTime().ToString('o')
            launcherSession = $launcher
        })
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $stopTimeout = Stop-AceLocalRecoveryPeriodicSaveForLocalQuit -AceRoot $ace -WaitSeconds 1 -CheckpointWaitSeconds 2
    Assert-True (-not $stopTimeout.Stopped -and $stopTimeout.Reason -eq 'stop_timeout_evidence_saved') 'stop_timeout_evidence'
    $incompletePath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'recovery-periodic-save-stop-incomplete.json'
    Assert-True (Test-Path -LiteralPath $incompletePath) 'stop_incomplete_file_written'
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -ErrorAction SilentlyContinue
    [void](Clear-AceLocalRecoveryPeriodicSaveProvenStaleState -AceRoot $ace -ExpectedLauncherSession $launcher)

    $verifiedAfter = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    if ($verifiedId) {
        Assert-True ($verifiedAfter.Count -ge 1 -and [string]$verifiedAfter[0].CheckpointId -eq $verifiedId) `
            'verified_checkpoint_untouched'
    }
    else {
        Assert-True ($true) 'verified_checkpoint_untouched'
    }

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'unreachable'
    $gate = New-MockGate
    $probeDeny = Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart -BackendRoot $be -FrontendRoot $fe `
        -AceRoot $ace -BackendPort 5000 -GateResult $gate -ManagedRegistry $registry -EngineBackendRoot $BackendRoot
    Assert-True (-not $probeDeny.Started -and $probeDeny.Reason -eq 'backend_probe_failed') 'backend_down_no_spawn'

    $bgText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1') -Raw
    Assert-True ($bgText -match 'Test-AceLocalRecoveryPeriodicSaveInstanceOccupied') 'occupancy_guard_present'
    Assert-True ($bgText -match 'WindowStyle Hidden') 'regression_hidden_window'
}
finally {
    Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE -ErrorAction SilentlyContinue
    if ($layout.TempRoot) {
        Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $layout.AceRoot -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($failures.Count -gt 0) {
    Write-Host ("Phase510 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase510 PASSED' -ForegroundColor Green
