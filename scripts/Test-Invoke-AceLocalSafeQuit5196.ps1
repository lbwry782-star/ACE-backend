#Requires -Version 5.1
<#
  Isolated MANAGED_STACK_STOP tests for Invoke-AceLocalSafeQuit (5.19.6).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Orchestrator = Join-Path $ScriptDir 'Invoke-AceLocalSafeQuit.ps1'
. (Join-Path $ScriptDir 'AceLocalSafeQuitManagedStackStopLib.ps1')
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')
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
        git config user.email 'safe-quit-5196@ace.local' 2>$null | Out-Null
        git config user.name 'SafeQuit5196' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'README.md') -Value 'base' -Encoding UTF8
        git add README.md 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-IsolatedStackStopLayout {
    param([string]$LauncherSession = '20260101-120000')
    $tempRoot = Join-Path $env:TEMP ("ace-safe-quit-stack5196-{0}" -f [guid]::NewGuid().ToString('N'))
    $aceRoot = Join-Path $tempRoot 'ACE'
    $backend = Join-Path $aceRoot 'ACE-Backend'
    $frontend = Join-Path $aceRoot 'ACE-Frontend'
    $localDev = Join-Path $aceRoot '_backups\local-dev'
    $sessions = Join-Path $aceRoot '_backups\sessions'
    $snapRoot = Join-Path $localDev 'shutdown-snapshots'
    foreach ($d in @($backend, $frontend, $localDev, $sessions, $snapRoot)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
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

function Invoke-IsolatedStackStopTest {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [switch]$SimulateSnapshotVerificationFailure,
        [switch]$SimulatePartialStackStop,
        [switch]$SimulateStackStopThrow
    )
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $Orchestrator,
        '-AceRoot', $AceRoot,
        '-IsolatedStackStopTest',
        '-LauncherSession', $LauncherSession,
        '-GateWaitSeconds', '15',
        '-PeriodicSaveQuiesceWaitSeconds', '15'
    )
    if ($SimulateSnapshotVerificationFailure) { $args += '-TestSimulateSnapshotVerificationFailure' }
    if ($SimulatePartialStackStop) { $args += '-TestSimulatePartialStackStop' }
    if ($SimulateStackStopThrow) { $args += '-TestSimulateStackStopThrow' }
    $p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $args -PassThru -Wait -WindowStyle Hidden
    return $p.ExitCode
}

function Get-TestManagedPidsFromRegistry {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $reg = Read-AceLocalManagedRegistry -AceRoot $AceRoot
    $pids = @()
    foreach ($name in @('frontend', 'worker', 'backend')) {
        $rec = $reg.components.$name
        if ($rec -and $rec.rootPid) { $pids += [int]$rec.rootPid }
    }
    return @($pids | Sort-Object -Unique)
}

Write-Host 'Test-Invoke-AceLocalSafeQuit5196 (isolated stack stop)...' -ForegroundColor Cyan

$orchText = Get-Content -LiteralPath $Orchestrator -Raw
Assert-True ($orchText -match 'IsolatedStackStopTest') 'isolated_stack_stop_switch'
Assert-True ($orchText -notmatch 'Stop-AceLocal\.ps1') 'orchestrator_no_stop_ace_local_script'
Assert-True ($orchText -notmatch 'Register-AceLocalRecoveryCleanShutdownComplete') 'no_clean_shutdown'
Assert-True ($orchText -notmatch 'Get-ManagedComponentStopActions') 'orchestrator_uses_isolated_stop_driver'

# Full pipeline stops test-owned stack only
$full = New-IsolatedStackStopLayout
$trackedPids = @()
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $full.AceRoot -LauncherSession $full.LauncherSession `
        -IncludeLegacyBootstrap -IncludeMemuraiWatch -IncludePeriodicSave)
    $trackedPids = Get-TestManagedPidsFromRegistry -AceRoot $full.AceRoot
    $ledgerBefore = (Get-FileHash -LiteralPath $full.LedgerPath -Algorithm SHA256).Hash
    $code = Invoke-IsolatedStackStopTest -AceRoot $full.AceRoot -LauncherSession $full.LauncherSession
    Assert-True ($code -eq 0) 'full_pipeline_exit_zero'
    foreach ($pidVal in $trackedPids) {
        Assert-True (-not (Test-ProcessAlive $pidVal)) ("test_proc_stopped_$pidVal")
    }
    $ledgerAfter = (Get-FileHash -LiteralPath $full.LedgerPath -Algorithm SHA256).Hash
    Assert-True ($ledgerBefore -eq $ledgerAfter) 'ledger_unchanged'
    $ledgerObj = Get-Content -LiteralPath $full.LedgerPath -Raw | ConvertFrom-Json
    Assert-True (-not $ledgerObj.cleanShutdown.completed) 'no_clean_shutdown_flag'
}
finally {
    foreach ($pidVal in $trackedPids) {
        if (Test-ProcessAlive $pidVal) { Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue }
    }
    Remove-Item -LiteralPath $full.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Verification failure — no stack stop, pointer unchanged
$verFail = New-IsolatedStackStopLayout
try {
    $seed = New-AceLocalShutdownSnapshot -BackendRoot $verFail.Backend -FrontendRoot $verFail.Frontend -AceRoot $verFail.AceRoot `
        -SnapshotsRootOverride $verFail.SnapshotsRoot
    Assert-True $seed.Ok 'seed_snapshot'
    $ptrBefore = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $verFail.AceRoot
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $verFail.AceRoot -LauncherSession $verFail.LauncherSession)
    $pidsBefore = Get-TestManagedPidsFromRegistry -AceRoot $verFail.AceRoot
    $codeV = Invoke-IsolatedStackStopTest -AceRoot $verFail.AceRoot -LauncherSession $verFail.LauncherSession `
        -SimulateSnapshotVerificationFailure
    Assert-True ($codeV -eq 12) 'verify_fail_exit_12'
    $ptrAfter = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $verFail.AceRoot
    Assert-True ([string]$ptrAfter.snapshotId -eq [string]$ptrBefore.snapshotId) 'pointer_unchanged_on_verify_fail'
    foreach ($pidVal in $pidsBefore) {
        Assert-True (Test-ProcessAlive $pidVal) ("stack_still_running_after_verify_fail_$pidVal")
        Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
    }
}
finally {
    Remove-Item -LiteralPath $verFail.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Partial stack stop
$partial = New-IsolatedStackStopLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $partial.AceRoot -LauncherSession $partial.LauncherSession)
    $pidsP = Get-TestManagedPidsFromRegistry -AceRoot $partial.AceRoot
    $codeP = Invoke-IsolatedStackStopTest -AceRoot $partial.AceRoot -LauncherSession $partial.LauncherSession -SimulatePartialStackStop
    Assert-True ($codeP -eq 13) 'partial_stack_stop_exit_13'
    $alive = @($pidsP | Where-Object { Test-ProcessAlive $_ })
    Assert-True ($alive.Count -gt 0) 'partial_stop_leaves_some_test_procs'
    foreach ($pidVal in $pidsP) {
        if (Test-ProcessAlive $pidVal) { Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue }
    }
}
finally {
    Remove-Item -LiteralPath $partial.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Memurai watch foreign PID (no test marker) — stack stop fails safely
$foreign = New-IsolatedStackStopLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $foreign.AceRoot -LauncherSession $foreign.LauncherSession)
    $holder = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList @('-NoProfile', '-Command', 'Start-Sleep 120') -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 300
    Set-Content -LiteralPath (Join-Path $foreign.LocalDev 'memurai-watch.pid') -Value ([string]$holder.Id) -Encoding ASCII
    $codeF = Invoke-IsolatedStackStopTest -AceRoot $foreign.AceRoot -LauncherSession $foreign.LauncherSession
    Assert-True ($codeF -eq 13) 'foreign_memurai_watch_exit_13'
    Assert-True (Test-ProcessAlive $holder.Id) 'foreign_process_not_killed'
    Stop-Process -Id $holder.Id -Force -ErrorAction SilentlyContinue
    $pidsF = Get-TestManagedPidsFromRegistry -AceRoot $foreign.AceRoot
    foreach ($pidVal in $pidsF) {
        if (Test-ProcessAlive $pidVal) { Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue }
    }
}
finally {
    Remove-Item -LiteralPath $foreign.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Stop order (library-level)
$orderLayout = New-IsolatedStackStopLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $orderLayout.AceRoot -LauncherSession $orderLayout.LauncherSession `
        -IncludeLegacyBootstrap -IncludeMemuraiWatch -IncludePeriodicSave)
    $gateOk = @{ Ok = $true; GateStatus = 'SAFE_TO_STOP' }
    $quiesceOk = @{ Ok = $true }
    $cap = New-AceLocalShutdownSnapshot -BackendRoot $orderLayout.Backend -FrontendRoot $orderLayout.Frontend `
        -AceRoot $orderLayout.AceRoot -SnapshotsRootOverride $orderLayout.SnapshotsRoot -DeferLastVerifiedPointerUpdate
    $ver = Invoke-AceLocalSafeQuitSnapshotVerification -AceRoot $orderLayout.AceRoot -CaptureResult $cap
    Assert-True $ver.Ok 'order_test_verify'
    $stop = Invoke-AceLocalSafeQuitManagedStackStop -AceRoot $orderLayout.AceRoot -LauncherSession $orderLayout.LauncherSession `
        -VerificationResult $ver -CaptureResult $cap
    Assert-True $stop.Ok 'order_test_stop_ok'
    $stepNames = @($stop.Steps | ForEach-Object { [string]$_.Step })
    $periodicIdx = [array]::IndexOf($stepNames, 'periodic_save')
    $legacyIdx = [array]::IndexOf($stepNames, 'legacy_bootstrap')
    $feIdx = [array]::IndexOf($stepNames, 'frontend')
    $beIdx = [array]::IndexOf($stepNames, 'backend')
    $mwIdx = [array]::IndexOf($stepNames, 'memurai_watch')
    Assert-True ($periodicIdx -ge 0 -and $legacyIdx -gt $periodicIdx) 'order_periodic_before_legacy'
    Assert-True ($feIdx -gt $legacyIdx -and $beIdx -gt $feIdx) 'order_frontend_before_backend'
    Assert-True ($mwIdx -gt $beIdx) 'order_memurai_watch_last'
}
finally {
    Remove-Item -LiteralPath $orderLayout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Duplicate quit mutex
$mutex = New-IsolatedStackStopLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $mutex.AceRoot -LauncherSession $mutex.LauncherSession)
    $job = Start-Job -ScriptBlock {
        param($orch, $ace, $ls)
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $orch -AceRoot $ace -IsolatedStackStopTest -LauncherSession $ls
    } -ArgumentList $Orchestrator, $mutex.AceRoot, $mutex.LauncherSession
    Start-Sleep -Milliseconds 600
    $code2 = Invoke-IsolatedStackStopTest -AceRoot $mutex.AceRoot -LauncherSession $mutex.LauncherSession
    Assert-True ($code2 -eq 2) 'duplicate_quit_exit_2'
    Wait-Job $job | Out-Null
    Remove-Job $job -Force
    $pidsM = Get-TestManagedPidsFromRegistry -AceRoot $mutex.AceRoot
    foreach ($pidVal in $pidsM) {
        if (Test-ProcessAlive $pidVal) { Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue }
    }
}
finally {
    Remove-Item -LiteralPath $mutex.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host ('FAILED: ' + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All Invoke-AceLocalSafeQuit5196 tests passed.' -ForegroundColor Green
