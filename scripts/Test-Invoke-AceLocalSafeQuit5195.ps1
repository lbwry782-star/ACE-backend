#Requires -Version 5.1
<#
  Isolated SHUTDOWN_SNAPSHOT pipeline tests for Invoke-AceLocalSafeQuit (5.19.5).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Orchestrator = Join-Path $ScriptDir 'Invoke-AceLocalSafeQuit.ps1'
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')

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
        git config user.email 'safe-quit-5195@ace.local' 2>$null | Out-Null
        git config user.name 'SafeQuit5195' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'README.md') -Value 'base' -Encoding UTF8
        git add README.md 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-IsolatedSnapshotLayout {
    param([string]$LauncherSession = '20260101-120000')
    $tempRoot = Join-Path $env:TEMP ("ace-safe-quit-snap5195-{0}" -f [guid]::NewGuid().ToString('N'))
    $aceRoot = Join-Path $tempRoot 'ACE'
    $backend = Join-Path $aceRoot 'ACE-Backend'
    $frontend = Join-Path $aceRoot 'ACE-Frontend'
    $sessions = Join-Path $aceRoot '_backups\sessions'
    $localDev = Join-Path $aceRoot '_backups\local-dev'
    $snapRoot = Join-Path $localDev 'shutdown-snapshots'
    foreach ($d in @($backend, $frontend, $sessions, $localDev, $snapRoot)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    'tracked-change' | Set-Content -LiteralPath (Join-Path $backend 'work.txt') -Encoding UTF8
    $reg = New-AceLocalManagedRegistryShell -BackendRoot $backend -FrontendRoot $frontend -LauncherSessionId $LauncherSession
    Write-AceLocalManagedRegistry -AceRoot $aceRoot -RegistryObject $reg
    $ledger = @{
        schemaVersion   = 1
        launcherSession = $LauncherSession
        cleanShutdown   = @{ completed = $false; completedAt = $null; launcherSession = $null }
        updatedAt       = '2026-01-01T00:00:00.0000000Z'
    } | ConvertTo-Json -Depth 6
    Set-Content -LiteralPath (Join-Path $localDev 'recovery-session-ledger.json') -Value $ledger -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $sessions 'active-dev-cycle.json') -Value '{"sessionId":"ori-test","cycleStatus":"OPEN"}' -Encoding UTF8
    return @{
        TempRoot        = $tempRoot
        AceRoot         = $aceRoot
        Backend         = $backend
        Frontend        = $frontend
        LocalDev        = $localDev
        SnapshotsRoot   = $snapRoot
        LauncherSession = $LauncherSession
        LedgerPath      = Join-Path $localDev 'recovery-session-ledger.json'
    }
}

function Invoke-IsolatedSnapshotTest {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$GateWaitSeconds = 15,
        [switch]$SimulateSnapshotFailAfterBackend,
        [switch]$SimulateSnapshotConcurrentDrift,
        [switch]$SimulateSnapshotVerificationFailure,
        [switch]$SimulateQuiesceWaitThrow
    )
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $Orchestrator,
        '-AceRoot', $AceRoot,
        '-IsolatedSnapshotTest',
        '-LauncherSession', $LauncherSession,
        '-GateWaitSeconds', ([string]$GateWaitSeconds),
        '-PeriodicSaveQuiesceWaitSeconds', '15'
    )
    if ($SimulateSnapshotFailAfterBackend) { $args += '-TestSimulateSnapshotFailAfterBackend' }
    if ($SimulateSnapshotConcurrentDrift) { $args += '-TestSimulateSnapshotConcurrentDrift' }
    if ($SimulateSnapshotVerificationFailure) { $args += '-TestSimulateSnapshotVerificationFailure' }
    if ($SimulateQuiesceWaitThrow) { $args += '-TestSimulateQuiesceWaitThrow' }
    $p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $args -PassThru -Wait -WindowStyle Hidden
    return $p.ExitCode
}

Write-Host 'Test-Invoke-AceLocalSafeQuit5195 (isolated snapshot)...' -ForegroundColor Cyan

$orchText = Get-Content -LiteralPath $Orchestrator -Raw
Assert-True ($orchText -match 'IsolatedSnapshotTest') 'isolated_snapshot_switch'
Assert-True ($orchText -notmatch 'New-AceLocalShutdownSnapshot') 'orchestrator_no_direct_snapshot'
Assert-True ($orchText -notmatch 'Stop-AceLocalRecoveryPeriodicSaveForLocalQuit') 'no_periodic_kill'
Assert-True ($orchText -notmatch 'Register-AceLocalRecoveryCleanShutdownComplete') 'no_clean_shutdown_ledger'
Assert-True ($orchText -notmatch 'Get-ManagedComponentStopActions') 'no_managed_stack_stop'

# DryRun / quiesce-only still no snapshot dirs
$dry = New-IsolatedSnapshotLayout
try {
    $dryArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Orchestrator, '-AceRoot', $dry.AceRoot, '-DryRun')
    $d = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $dryArgs -PassThru -Wait -WindowStyle Hidden
    Assert-True ($d.ExitCode -eq 0) 'dryrun_ok'
    Assert-True ((@(Get-ChildItem -LiteralPath $dry.SnapshotsRoot -ErrorAction SilentlyContinue)).Count -eq 0) 'dryrun_no_snapshot_dirs'

    $qArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Orchestrator, '-AceRoot', $dry.AceRoot,
        '-IsolatedQuiesceTest', '-LauncherSession', $dry.LauncherSession)
    $q = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $qArgs -PassThru -Wait -WindowStyle Hidden
    Assert-True ($q.ExitCode -eq 0) 'quiesce_only_ok'
    Assert-True ((@(Get-ChildItem -LiteralPath $dry.SnapshotsRoot -ErrorAction SilentlyContinue)).Count -eq 0) 'quiesce_no_snapshot_dirs'
}
finally {
    Remove-Item -LiteralPath $dry.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Verified snapshot end-to-end
$ok = New-IsolatedSnapshotLayout
try {
    $ledgerHashBefore = (Get-FileHash -LiteralPath $ok.LedgerPath -Algorithm SHA256).Hash
    $code = Invoke-IsolatedSnapshotTest -AceRoot $ok.AceRoot -LauncherSession $ok.LauncherSession
    Assert-True ($code -eq 0) 'pipeline_verified_exit_zero'
    $ptr = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $ok.AceRoot
    Assert-True ($null -ne $ptr) 'last_verified_pointer_set'
    $meta = Get-Content -LiteralPath (Join-Path $ptr.snapshotRoot 'snapshot-metadata.json') -Raw | ConvertFrom-Json
    Assert-True ([string]$meta.status -eq 'VERIFIED') 'metadata_verified'
    $integrity = Test-AceLocalShutdownSnapshotIntegrity -SnapshotRoot $ptr.snapshotRoot
    Assert-True $integrity.Ok 'post_hoc_integrity_ok'
    $ledgerHashAfter = (Get-FileHash -LiteralPath $ok.LedgerPath -Algorithm SHA256).Hash
    Assert-True ($ledgerHashBefore -eq $ledgerHashAfter) 'ledger_unchanged_no_clean_shutdown'
    [void](Reset-AceLocalSafeQuitFailedAttempt -AceRoot $ok.AceRoot)
}
finally {
    Remove-Item -LiteralPath $ok.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Gate blocked — no snapshot
$blocked = New-IsolatedSnapshotLayout
try {
    $sid = 'ori-20260101-120000-abcdef01'
    $sessionDir = Join-Path (Join-Path $blocked.AceRoot '_backups\sessions') $sid
    New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $sessionDir 'session.json') -Value (@{ status = 'IN_PROGRESS'; sessionId = $sid } | ConvertTo-Json) -Encoding UTF8
    $codeB = Invoke-IsolatedSnapshotTest -AceRoot $blocked.AceRoot -LauncherSession $blocked.LauncherSession -GateWaitSeconds 1
    Assert-True ($codeB -eq 6) 'gate_timeout_exit_6'
    Assert-True ((@(Get-ChildItem -LiteralPath $blocked.SnapshotsRoot -ErrorAction SilentlyContinue)).Count -eq 0) 'gate_blocked_no_snapshots'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $blocked.LocalDev 'recovery-periodic-save.stop'))) 'gate_blocked_stop_cleared'
}
finally {
    Remove-Item -LiteralPath $blocked.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Simulated partial snapshot — prior verified preserved
$partial = New-IsolatedSnapshotLayout
try {
    $first = New-AceLocalShutdownSnapshot -BackendRoot $partial.Backend -FrontendRoot $partial.Frontend -AceRoot $partial.AceRoot `
        -SnapshotsRootOverride $partial.SnapshotsRoot
    Assert-True $first.Ok 'seed_verified_snapshot'
    $ptr1 = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $partial.AceRoot
    $codeF = Invoke-IsolatedSnapshotTest -AceRoot $partial.AceRoot -LauncherSession $partial.LauncherSession `
        -SimulateSnapshotFailAfterBackend
    Assert-True ($codeF -eq 11) 'partial_snapshot_exit_11'
    $ptr2 = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $partial.AceRoot
    Assert-True ([string]$ptr2.snapshotId -eq [string]$ptr1.snapshotId) 'verified_pointer_unchanged_after_fail'
    $dirs = @(Get-ChildItem -LiteralPath $partial.SnapshotsRoot -Directory)
    Assert-True ($dirs.Count -ge 2) 'incomplete_folder_retained'
    $incomplete = @($dirs | Where-Object { $_.Name -ne (Split-Path -Leaf $ptr1.snapshotRoot) } | Select-Object -Last 1)
    if ($incomplete) {
        $metaInc = Get-Content -LiteralPath (Join-Path $incomplete.FullName 'snapshot-metadata.json') -Raw | ConvertFrom-Json
        Assert-True ([string]$metaInc.status -eq 'INCOMPLETE') 'failed_run_incomplete_status'
    }
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $partial.LocalDev 'recovery-periodic-save.stop'))) 'partial_fail_stop_reset'
}
finally {
    Remove-Item -LiteralPath $partial.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Concurrent drift
$drift = New-IsolatedSnapshotLayout
try {
    $codeD = Invoke-IsolatedSnapshotTest -AceRoot $drift.AceRoot -LauncherSession $drift.LauncherSession `
        -SimulateSnapshotConcurrentDrift
    Assert-True ($codeD -eq 11) 'concurrent_drift_exit_11'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $drift.LocalDev 'recovery-periodic-save.stop'))) 'drift_fail_stop_reset'
}
finally {
    Remove-Item -LiteralPath $drift.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Verification phase failure (simulated) — deferred pointer not committed
$verFail = New-IsolatedSnapshotLayout
try {
    $seed = New-AceLocalShutdownSnapshot -BackendRoot $verFail.Backend -FrontendRoot $verFail.Frontend -AceRoot $verFail.AceRoot `
        -SnapshotsRootOverride $verFail.SnapshotsRoot
    Assert-True $seed.Ok 'seed_for_pointer_test'
    $ptrBefore = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $verFail.AceRoot
    $codeV = Invoke-IsolatedSnapshotTest -AceRoot $verFail.AceRoot -LauncherSession $verFail.LauncherSession `
        -SimulateSnapshotVerificationFailure
    Assert-True ($codeV -eq 12) 'verification_fail_exit_12'
    $ptrAfter = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $verFail.AceRoot
    Assert-True ([string]$ptrAfter.snapshotId -eq [string]$ptrBefore.snapshotId) 'verify_fail_pointer_unchanged'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $verFail.LocalDev 'recovery-periodic-save.stop'))) 'verify_fail_stop_reset'
}
finally {
    Remove-Item -LiteralPath $verFail.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Session mismatch
$mis = New-IsolatedSnapshotLayout -LauncherSession '20260101-120000'
try {
    $reg = Get-Content -LiteralPath (Join-Path $mis.LocalDev 'managed-processes.json') -Raw | ConvertFrom-Json
    $reg.launcherSession = 'other-session'
    ($reg | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $mis.LocalDev 'managed-processes.json') -Encoding UTF8
    $codeM = Invoke-IsolatedSnapshotTest -AceRoot $mis.AceRoot -LauncherSession '20260101-120000'
    Assert-True ($codeM -eq 10) 'session_mismatch_exit_10'
}
finally {
    Remove-Item -LiteralPath $mis.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Duplicate invocation (mutex)
$mutex = New-IsolatedSnapshotLayout
try {
    $job = Start-Job -ScriptBlock {
        param($orch, $ace, $ls)
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $orch -AceRoot $ace -IsolatedSnapshotTest -LauncherSession $ls -GateWaitSeconds 30
    } -ArgumentList $Orchestrator, $mutex.AceRoot, $mutex.LauncherSession
    Start-Sleep -Milliseconds 500
    $code2 = Invoke-IsolatedSnapshotTest -AceRoot $mutex.AceRoot -LauncherSession $mutex.LauncherSession
    Assert-True ($code2 -eq 2) 'duplicate_quit_exit_2'
    Wait-Job $job | Out-Null
    Remove-Job $job -Force
}
finally {
    Remove-Item -LiteralPath $mutex.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host ('FAILED: ' + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All Invoke-AceLocalSafeQuit5195 tests passed.' -ForegroundColor Green
