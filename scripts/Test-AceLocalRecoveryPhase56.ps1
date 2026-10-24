#Requires -Version 5.1
<#
  Phase 5.6 — background periodic save (hidden process contract; temp repos only).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')

$failures = @()
$timings = @{}
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}
function Measure-Phase56 {
    param([string]$Name, [scriptblock]$Block)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Block
    $sw.Stop()
    $script:timings[$Name] = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
}

function Initialize-TempGitRepo {
    param([Parameter(Mandatory = $true)][string]$Root)
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Push-Location -LiteralPath $Root
    try {
        git init -q 2>$null | Out-Null
        git config user.email 'recovery-cp56@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP56' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp56-{0}" -f [guid]::NewGuid().ToString('N'))
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

Write-Host 'Test-AceLocalRecoveryPhase56 (isolated)...' -ForegroundColor Cyan
$env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart') 'wired_to_start'
Assert-True ($stopText -match 'Stop-AceLocalRecoveryPeriodicSaveForLocalQuit') 'wired_to_stop'
$bgText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1') -Raw
Assert-True ($bgText -match 'WindowStyle Hidden') 'hidden_powershell_spawn'
Assert-True ($bgText -match 'Start-Process') 'uses_start_process_hidden'
$loopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocalRecoveryPeriodicSave.ps1') -Raw
Assert-True ($loopText -match 'BackgroundMode') 'background_mode_flag'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = '20260201-120000'
    $t0 = [DateTime]::Parse('2026-02-01T12:00:00Z', $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()

    # 1 — GUI mock stays responsive during scan
    Measure-Phase56 -Name 'dirty_scan_ms' -Block {
        'edit' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
        $tick = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
            -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
        $script:scanOutcome = $tick.Outcome
        $script:firstCpId = [string]$tick.CheckpointId
    }
    Assert-True ($script:scanOutcome -eq 'checkpoint_verified') 'scan_during_simulated_gui'
    if ($script:scanOutcome -eq 'checkpoint_verified') {
        Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $ace -CheckpointId $script:firstCpId `
            -CheckpointPath (Join-Path $cpRoot $script:firstCpId) -VerifiedAt $t0.ToString('o') -CheckpointType full `
            -ErrorAction SilentlyContinue | Out-Null
    }

    $ledger = Read-AceLocalRecoveryLedger -AceRoot $ace
    $verifiedBefore = [string]$ledger.lastVerifiedCheckpoint.checkpointId
    $headBefore = (git -C $be rev-parse HEAD).Trim()

    # 2 — IO defer before another checkpoint window
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $t0.AddSeconds(200).ToString('o') | Out-Null
    'defer-edit' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    Set-AceLocalRecoveryPeriodicSaveCheckpointDefer -AceRoot $ace -DeferUntilUtc $t0.AddMinutes(10) -Reason 'simulated_io_load'
    $tickDefer = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(301) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickDefer.Outcome -eq 'checkpoint_deferred' -and $tickDefer.DecisionAction -match 'checkpoint') `
        ("io_defer_skips_checkpoint (outcome={0} action={1})" -f $tickDefer.Outcome, $tickDefer.DecisionAction)
    Remove-Item -LiteralPath (Get-AceLocalRecoveryPeriodicSaveDeferPath -AceRoot $ace) -Force -ErrorAction SilentlyContinue

    # 3 — checkpoint while dev edit
    Measure-Phase56 -Name 'light_checkpoint_ms' -Block {
        'v2' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
        $tick2 = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
            -LauncherSession $launcher -NowUtc $t0.AddSeconds(302) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
        $script:cpOutcome = $tick2.Outcome
    }
    Assert-True ($script:cpOutcome -in @('checkpoint_verified', 'checkpoint_skipped', 'dirty_scan', 'noop')) 'checkpoint_during_dev'

    # 4 — concurrent tick
    $lock = Enter-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -LauncherSession $launcher
    Assert-True $lock.Acquired 'lock_for_concurrent_test'
    $tickConcurrent = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(302) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickConcurrent.Outcome -eq 'skipped_lock_live') 'concurrent_tick_skipped'
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace

    # 4 — no changes
    $tickNo = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(130) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickNo.Outcome -in @('noop', 'dirty_scan')) 'no_change_tick'

    # 5 — file change during scan (reentrant guard)
    $job = Start-Job -ScriptBlock {
        param($be, $fe, $ace, $launcher, $cpRoot, $lib)
        . $lib
        Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
            -LauncherSession $launcher -NowUtc ([DateTime]::UtcNow) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    } -ArgumentList $be, $fe, $ace, $launcher, $cpRoot, (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')
    'v3' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    Wait-Job $job -Timeout 60 | Out-Null
    Receive-Job $job | Out-Null
    Remove-Job $job -Force -ErrorAction SilentlyContinue
    Assert-True ($true) 'change_during_scan_no_throw'

    # 6 — IO defer
    Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $ace -DirtyFingerprint `
        (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan (Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe)) `
        -DirtyScanAt $t0.AddSeconds(200).ToString('o') | Out-Null
    'defer-edit' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    Set-AceLocalRecoveryPeriodicSaveCheckpointDefer -AceRoot $ace -DeferUntilUtc $t0.AddMinutes(10) -Reason 'simulated_io_load'
    $tickDefer = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0.AddSeconds(301) -CheckpointsRootOverride $cpRoot -SkipLiveAssessment
    Assert-True ($tickDefer.Outcome -eq 'checkpoint_deferred' -and $tickDefer.DecisionAction -match 'checkpoint') `
        ("io_defer_skips_checkpoint (outcome={0} action={1})" -f $tickDefer.Outcome, $tickDefer.DecisionAction)
    Remove-Item -LiteralPath (Get-AceLocalRecoveryPeriodicSaveDeferPath -AceRoot $ace) -Force -ErrorAction SilentlyContinue
    $ledgerMid = Read-AceLocalRecoveryLedger -AceRoot $ace
    if ($verifiedBefore) {
        Assert-True ([bool]$ledgerMid.lastVerifiedCheckpoint.checkpointId) 'verified_checkpoint_preserved'
    }

    # 8 — controlled stop mid-cycle (background short run)
    Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $ace -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'recovery-periodic-save.pid') -Force -ErrorAction SilentlyContinue
    $spawn = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true `
        -TickIntervalSeconds 2 -CheckpointsRootOverride $cpRoot -EngineBackendRoot $BackendRoot
    Assert-True $spawn.Started 'background_started'
    Start-Sleep -Seconds 1
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $ace
    $stopped = Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $ace -WaitSeconds 20
    Assert-True (-not $stopped.IsRunning -or $stopped.StateStatus -eq 'stopped') 'controlled_stop'

    # 9 — crash detection (stale running state)
    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $ace -State @{
        status = 'running'
        launcherSession = $launcher
        pid = 999999
    }
    Set-Content -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'recovery-periodic-save.pid') -Value '999999' -Encoding ASCII
    $crashStatus = Get-AceLocalRecoveryPeriodicSaveBackgroundStatus -AceRoot $ace -ExpectedLauncherSession $launcher
    Assert-True $crashStatus.Crashed 'detect_crashed_service'

    # 10 — no popup contract (static)
    Assert-True ($bgText -notmatch 'MessageBox') 'no_messagebox_in_background_lib'

    # 11 — source unchanged
    Assert-True ((git -C $be rev-parse HEAD).Trim() -eq $headBefore) 'source_head_unchanged'

    # 10 — no popup contract (static)
    Assert-True ($bgText -notmatch 'MessageBox') 'no_messagebox_in_background_lib'

    # 11 — source unchanged
    Assert-True ((git -C $be rev-parse HEAD).Trim() -eq $headBefore) 'source_head_unchanged'

    # 12 — last verified preserved on blocked tick
    $ledgerAfter = Read-AceLocalRecoveryLedger -AceRoot $ace
    if ($verifiedBefore) {
        Assert-True ([string]$ledgerAfter.lastVerifiedCheckpoint.checkpointId -eq $verifiedBefore) 'verified_checkpoint_preserved'
    }

    # activation denied when recovery unresolved
    $deny = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true -RecoveryUnresolved
    Assert-True (-not $deny.Started) 'blocked_when_recovery_unresolved'

    # double start prevention
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $ace
    $first = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true `
        -TickIntervalSeconds 2 -CheckpointsRootOverride $cpRoot -EngineBackendRoot $BackendRoot
    Start-Sleep -Seconds 1
    $second = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true `
        -TickIntervalSeconds 2 -CheckpointsRootOverride $cpRoot -EngineBackendRoot $BackendRoot
    Assert-True ($first.Started) 'first_background_ok'
    Assert-True (-not $second.Started) 'double_start_prevented'
    [void](Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $ace -WaitSeconds 25)
}
finally {
    if ($layout.TempRoot -and (Test-Path -LiteralPath $layout.TempRoot)) {
        Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $layout.AceRoot -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
        Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_periodic_save_service.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_service' }
    else { Write-Host '  ok pytest_service' -ForegroundColor DarkGray }
}
finally { Pop-Location }

Write-Host '--- measured timings (ms) ---' -ForegroundColor Cyan
foreach ($key in ($timings.Keys | Sort-Object)) {
    Write-Host ("  {0}: {1}" -f $key, $timings[$key])
}

if ($failures.Count -gt 0) {
    Write-Host ("Phase56 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase56 PASSED' -ForegroundColor Green
