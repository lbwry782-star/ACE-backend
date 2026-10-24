#Requires -Version 5.1
<#
  Isolated PERIODIC_SAVE_QUIESCE tests for Invoke-AceLocalSafeQuit (5.19.4).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Orchestrator = Join-Path $ScriptDir 'Invoke-AceLocalSafeQuit.ps1'
. (Join-Path $ScriptDir 'AceLocalSafeQuitPeriodicSaveLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function New-IsolatedQuiesceLayout {
    param(
        [string]$LauncherSession = '20260101-120000',
        [string]$RegistryLauncherSession = ''
    )
    if (-not $RegistryLauncherSession) { $RegistryLauncherSession = $LauncherSession }
    $root = Join-Path $env:TEMP ("ace-safe-quit-quiesce-{0}" -f [guid]::NewGuid().ToString('N'))
    $localDev = Join-Path $root '_backups\local-dev'
    $sessions = Join-Path $root '_backups\sessions'
    New-Item -ItemType Directory -Path $localDev -Force | Out-Null
    New-Item -ItemType Directory -Path $sessions -Force | Out-Null
    $registry = @{
        schemaVersion   = 1
        launcherSession = $RegistryLauncherSession
        updatedAt       = '2026-01-01T00:00:00.0000000Z'
        components      = @{}
    } | ConvertTo-Json -Depth 6
    $ledger = @{
        schemaVersion   = 1
        launcherSession = $LauncherSession
        cleanShutdown   = @{ completed = $false; completedAt = $null; launcherSession = $null }
        updatedAt       = '2026-01-01T00:00:00.0000000Z'
    } | ConvertTo-Json -Depth 6
    Set-Content -LiteralPath (Join-Path $localDev 'managed-processes.json') -Value $registry -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $localDev 'recovery-session-ledger.json') -Value $ledger -Encoding UTF8
    return @{
        AceRoot         = $root
        LocalDev        = $localDev
        LauncherSession = $LauncherSession
        StopPath        = Join-Path $localDev 'recovery-periodic-save.stop'
        LockPath        = Join-Path $localDev 'recovery-periodic-save.lock'
    }
}

function Invoke-IsolatedQuiesceTest {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$GateWaitSeconds = 15,
        [int]$PeriodicSaveQuiesceWaitSeconds = 15,
        [switch]$SimulateQuiesceWaitThrow
    )
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $Orchestrator,
        '-AceRoot', $AceRoot,
        '-IsolatedQuiesceTest',
        '-LauncherSession', $LauncherSession,
        '-GateWaitSeconds', ([string]$GateWaitSeconds),
        '-PeriodicSaveQuiesceWaitSeconds', ([string]$PeriodicSaveQuiesceWaitSeconds)
    )
    if ($SimulateQuiesceWaitThrow) { $args += '-TestSimulateQuiesceWaitThrow' }
    $p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $args -PassThru -Wait -WindowStyle Hidden
    return $p.ExitCode
}

function Write-LivePeriodicLock {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$LockPid = $PID
    )
    $payload = [ordered]@{
        pid             = $LockPid
        startedAt       = (Get-Date).ToUniversalTime().ToString('o')
        launcherSession = $LauncherSession
        aceRoot         = $AceRoot
    }
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $AceRoot) -Object $payload
}

Write-Host 'Test-Invoke-AceLocalSafeQuit5194 (isolated quiesce)...' -ForegroundColor Cyan

$orchText = Get-Content -LiteralPath $Orchestrator -Raw
Assert-True ($orchText -match 'IsolatedQuiesceTest') 'isolated_quiesce_switch'
Assert-True ($orchText -match 'PERIODIC_SAVE_QUIESCE') 'quiesce_phase_wired'
Assert-True ($orchText -notmatch 'New-AceLocalShutdownSnapshot') 'no_snapshot'
Assert-True ($orchText -notmatch 'Stop-AceLocalRecoveryPeriodicSaveForLocalQuit') 'no_process_kill'

# DryRun / IsolatedGateTest: no stop file
$dryLayout = New-IsolatedQuiesceLayout
try {
    $dryArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Orchestrator, '-AceRoot', $dryLayout.AceRoot, '-DryRun')
    $dry = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $dryArgs -PassThru -Wait -WindowStyle Hidden
    Assert-True ($dry.ExitCode -eq 0) 'dryrun_ok'
    Assert-True (-not (Test-Path -LiteralPath $dryLayout.StopPath)) 'dryrun_no_stop'

    $gateArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Orchestrator, '-AceRoot', $dryLayout.AceRoot,
        '-IsolatedGateTest', '-LauncherSession', $dryLayout.LauncherSession)
    $gateOnly = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $gateArgs -PassThru -Wait -WindowStyle Hidden
    Assert-True ($gateOnly.ExitCode -eq 0) 'gate_only_ok'
    Assert-True (-not (Test-Path -LiteralPath $dryLayout.StopPath)) 'gate_only_no_stop'
}
finally {
    Remove-Item -LiteralPath $dryLayout.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Idle — not active, no lock
$idle = New-IsolatedQuiesceLayout
try {
    $code = Invoke-IsolatedQuiesceTest -AceRoot $idle.AceRoot -LauncherSession $idle.LauncherSession
    Assert-True ($code -eq 0) 'idle_quiesce_ok'
    Assert-True (Test-Path -LiteralPath $idle.StopPath) 'idle_stop_requested'
    $pre = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $idle.AceRoot -LauncherSession $idle.LauncherSession `
        -GuardBackendRoot (Split-Path -Parent $ScriptDir) -SkipLiveAssessment
    Assert-True (-not $pre.Allowed -and ($pre.Reasons -contains 'stop_requested')) 'idle_pre_write_blocked'
    [void](Reset-AceLocalSafeQuitPeriodicSaveQuiesce -AceRoot $idle.AceRoot -ForceIfInitMatchesAttemptOnly)
    Assert-True (-not (Test-Path -LiteralPath $idle.StopPath)) 'idle_reset_clears_stop'
}
finally {
    Remove-Item -LiteralPath $idle.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Active not writing — service state only (no lock)
$activeIdle = New-IsolatedQuiesceLayout
try {
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryPeriodicSaveStatePath -AceRoot $activeIdle.AceRoot) -Object @{
        schemaVersion = 1
        status        = 'running'
        lastDecision  = 'dirty_scan_idle'
        updatedAt     = (Get-Date).ToUniversalTime().ToString('o')
    }
    $codeA = Invoke-IsolatedQuiesceTest -AceRoot $activeIdle.AceRoot -LauncherSession $activeIdle.LauncherSession
    Assert-True ($codeA -eq 0) 'service_idle_no_lock_ok'
}
finally {
    Remove-Item -LiteralPath $activeIdle.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Checkpoint completes during wait
$completing = New-IsolatedQuiesceLayout
try {
    Write-LivePeriodicLock -AceRoot $completing.AceRoot -LauncherSession $completing.LauncherSession
    $lockPath = $completing.LockPath
    $aceCopy = $completing.AceRoot
    $job = Start-Job -ScriptBlock {
        param($lp)
        Start-Sleep -Seconds 2
        Remove-Item -LiteralPath $lp -Force -ErrorAction SilentlyContinue
    } -ArgumentList $lockPath
    $codeC = Invoke-IsolatedQuiesceTest -AceRoot $completing.AceRoot -LauncherSession $completing.LauncherSession `
        -PeriodicSaveQuiesceWaitSeconds 20
    Wait-Job $job | Out-Null
    Remove-Job $job -Force
    Assert-True ($codeC -eq 0) 'checkpoint_completes_ok'
    Assert-True (-not (Test-Path -LiteralPath $lockPath)) 'lock_cleared_after_write'
}
finally {
    Remove-Item -LiteralPath $completing.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Wait timeout — live lock held
$timeout = New-IsolatedQuiesceLayout
try {
    Write-LivePeriodicLock -AceRoot $timeout.AceRoot -LauncherSession $timeout.LauncherSession
    $codeT = Invoke-IsolatedQuiesceTest -AceRoot $timeout.AceRoot -LauncherSession $timeout.LauncherSession `
        -PeriodicSaveQuiesceWaitSeconds 2
    Assert-True ($codeT -eq 9) 'quiesce_wait_timeout_exit_9'
    Assert-True (Test-Path -LiteralPath $timeout.StopPath) 'timeout_leaves_stop_for_reset'
    $initPath = Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath -AceRoot $timeout.AceRoot
    $attempt = [string]((Get-Content -LiteralPath $initPath -Raw | ConvertFrom-Json).safeQuitAttemptId)
    [void](Reset-AceLocalSafeQuitPeriodicSaveQuiesce -AceRoot $timeout.AceRoot -SafeQuitAttemptId $attempt -ForceIfInitMatchesAttemptOnly)
    Assert-True (-not (Test-Path -LiteralPath $timeout.StopPath)) 'timeout_reset_retry_ready'
    Remove-Item -LiteralPath $timeout.LockPath -Force -ErrorAction SilentlyContinue
    $codeRetry = Invoke-IsolatedQuiesceTest -AceRoot $timeout.AceRoot -LauncherSession $timeout.LauncherSession
    Assert-True ($codeRetry -eq 0) 'retry_after_reset_ok'
}
finally {
    Remove-Item -LiteralPath $timeout.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Corrupt lock
$corrupt = New-IsolatedQuiesceLayout
try {
    Set-Content -LiteralPath $corrupt.LockPath -Value '{bad' -Encoding UTF8
    $codeU = Invoke-IsolatedQuiesceTest -AceRoot $corrupt.AceRoot -LauncherSession $corrupt.LauncherSession
    Assert-True ($codeU -eq 8) 'corrupt_lock_exit_8'
    Assert-True (-not (Test-Path -LiteralPath $corrupt.StopPath)) 'corrupt_lock_no_stop'
}
finally {
    Remove-Item -LiteralPath $corrupt.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Session mismatch (registry vs parameter)
$sessionMismatch = New-IsolatedQuiesceLayout -LauncherSession '20260101-120000' -RegistryLauncherSession 'other-session'
try {
    $codeS = Invoke-IsolatedQuiesceTest -AceRoot $sessionMismatch.AceRoot -LauncherSession '20260101-120000'
    Assert-True ($codeS -eq 10) 'registry_session_mismatch_exit_10'
}
finally {
    Remove-Item -LiteralPath $sessionMismatch.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Lock session mismatch (live lock from other session)
$lockSession = New-IsolatedQuiesceLayout
$lockHolder = $null
try {
    $lockHolder = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList @('-NoProfile', '-Command', 'Start-Sleep -Seconds 120') -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 400
    Write-LivePeriodicLock -AceRoot $lockSession.AceRoot -LauncherSession 'old-launcher-999' -LockPid $lockHolder.Id
    $codeL = Invoke-IsolatedQuiesceTest -AceRoot $lockSession.AceRoot -LauncherSession $lockSession.LauncherSession
    Assert-True ($codeL -eq 8) 'lock_old_session_exit_8'
    Assert-True (-not (Test-Path -LiteralPath $lockSession.StopPath)) 'lock_old_session_no_stop'
}
finally {
    if ($lockHolder -and -not $lockHolder.HasExited) {
        Stop-Process -Id $lockHolder.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $lockSession.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Foreign stop (no safe-quit init marker)
$foreignStop = New-IsolatedQuiesceLayout
try {
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $foreignStop.AceRoot
    $codeF = Invoke-IsolatedQuiesceTest -AceRoot $foreignStop.AceRoot -LauncherSession $foreignStop.LauncherSession
    Assert-True ($codeF -eq 8) 'foreign_stop_exit_8'
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $foreignStop.AceRoot
}
finally {
    Remove-Item -LiteralPath $foreignStop.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Exception during wait — cleanup stop via catch path
$throwWait = New-IsolatedQuiesceLayout
try {
    Write-LivePeriodicLock -AceRoot $throwWait.AceRoot -LauncherSession $throwWait.LauncherSession
    $codeX = Invoke-IsolatedQuiesceTest -AceRoot $throwWait.AceRoot -LauncherSession $throwWait.LauncherSession `
        -PeriodicSaveQuiesceWaitSeconds 30 -SimulateQuiesceWaitThrow
    Assert-True ($codeX -eq 1) 'simulated_wait_throw_exit_1'
    $initPath = Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath -AceRoot $throwWait.AceRoot
    if (Test-Path -LiteralPath $initPath) {
        $attempt = [string]((Get-Content -LiteralPath $initPath -Raw | ConvertFrom-Json).safeQuitAttemptId)
        [void](Reset-AceLocalSafeQuitPeriodicSaveQuiesce -AceRoot $throwWait.AceRoot -SafeQuitAttemptId $attempt -ForceIfInitMatchesAttemptOnly)
    }
    Assert-True (-not (Test-Path -LiteralPath $throwWait.StopPath)) 'throw_cleanup_stop_cleared'
}
finally {
    Remove-Item -LiteralPath $throwWait.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Gate blocked — quiesce NOT_EXECUTED (active NEW session)
$gateBlocked = New-IsolatedQuiesceLayout
try {
    $sid = 'ori-20260101-120000-abcdef01'
    $sessionDir = Join-Path (Join-Path $gateBlocked.AceRoot '_backups\sessions') $sid
    New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $sessionDir 'session.json') -Value (@{
            status = 'IN_PROGRESS'; sessionId = $sid
        } | ConvertTo-Json) -Encoding UTF8
    $codeG = Invoke-IsolatedQuiesceTest -AceRoot $gateBlocked.AceRoot -LauncherSession $gateBlocked.LauncherSession -GateWaitSeconds 1
    Assert-True ($codeG -eq 6) 'gate_timeout_no_quiesce_exit_6'
    Assert-True (-not (Test-Path -LiteralPath $gateBlocked.StopPath)) 'gate_fail_no_periodic_stop'
}
finally {
    Remove-Item -LiteralPath $gateBlocked.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host ('FAILED: ' + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All Invoke-AceLocalSafeQuit5194 tests passed.' -ForegroundColor Green
