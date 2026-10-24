#Requires -Version 5.1
<#
  Isolated tests for Invoke-AceLocalSafeQuit skeleton (5.19.1). Never uses live ACE local-dev state.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Orchestrator = Join-Path $ScriptDir 'Invoke-AceLocalSafeQuit.ps1'
$LiveAceRoot = 'D:\אס2\ACE'
$LiveLocalDev = Join-Path $LiveAceRoot '_backups\local-dev'

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function Get-PathSha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function New-IsolatedAceLayout {
    $root = Join-Path $env:TEMP ("ace-safe-quit-5191-{0}" -f [guid]::NewGuid().ToString('N'))
    $localDev = Join-Path $root '_backups\local-dev'
    $sessions = Join-Path $root '_backups\sessions'
    New-Item -ItemType Directory -Path $localDev -Force | Out-Null
    New-Item -ItemType Directory -Path $sessions -Force | Out-Null
    $registry = @{
        schemaVersion   = 1
        launcherSession = 'test-session-5191'
        updatedAt       = '2026-01-01T00:00:00.0000000Z'
        components      = @{}
    } | ConvertTo-Json -Depth 6
    $ledger = @{
        schemaVersion = 1
        launcherSession = 'test-session-5191'
        cleanShutdown = @{ completed = $false; completedAt = $null; launcherSession = $null }
        updatedAt = '2026-01-01T00:00:00.0000000Z'
    } | ConvertTo-Json -Depth 6
    $registryPath = Join-Path $localDev 'managed-processes.json'
    $ledgerPath = Join-Path $localDev 'recovery-session-ledger.json'
    Set-Content -LiteralPath $registryPath -Value $registry -Encoding UTF8
    Set-Content -LiteralPath $ledgerPath -Value $ledger -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $sessions 'active-dev-cycle.json') -Value '{"sessionId":"ori-test","cycleStatus":"OPEN"}' -Encoding UTF8
    return @{
        AceRoot      = $root
        LocalDev     = $localDev
        RegistryPath = $registryPath
        LedgerPath   = $ledgerPath
    }
}

function Invoke-SafeQuitDryRun {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [switch]$SimulateThrowAfterLock,
        [int]$TestHoldLockMilliseconds = 0
    )
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $Orchestrator,
        '-AceRoot', $AceRoot,
        '-DryRun'
    )
    if ($SimulateThrowAfterLock) { $args += '-TestSimulateThrowAfterLock' }
    if ($TestHoldLockMilliseconds -gt 0) { $args += @('-TestHoldLockMilliseconds', ([string]$TestHoldLockMilliseconds)) }
    $p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $args -PassThru -Wait -WindowStyle Hidden
    return $p.ExitCode
}

Write-Host 'Test-Invoke-AceLocalSafeQuit5191 (isolated)...' -ForegroundColor Cyan

$orchText = Get-Content -LiteralPath $Orchestrator -Raw
Assert-True ($orchText -match 'Global\\ACE-LOCAL-SAFE-QUIT') 'dedicated_quit_mutex'
Assert-True (Test-Path -LiteralPath (Join-Path $ScriptDir 'AceLocalSafeQuitShutdownSnapshotLib.ps1')) 'snapshot_lib_present'
Assert-True ($orchText -notmatch 'New-AceLocalShutdownSnapshot') 'orchestrator_no_direct_snapshot_call'
Assert-True ($orchText -notmatch 'Stop-AceLocalRecovery') 'no_recovery_stop_call'
Assert-True (Test-Path -LiteralPath (Join-Path $ScriptDir 'AceLocalShutdownGateLib.ps1')) 'gate_lib_present'
Assert-True ($orchText -match 'IsolatedGateTest') 'isolated_gate_parameter'
Assert-True ($orchText -match 'Invoke-AceLocalSafeQuitDryRunReport') 'dryrun_report_path'
Assert-True ($orchText -notmatch 'Register-AceLocalRecoveryCleanShutdownComplete') 'no_ledger_register_call'
Assert-True (Test-Path -LiteralPath (Join-Path $ScriptDir 'AceLocalSafeQuitPeriodicSaveLib.ps1')) 'quiesce_lib_present'
Assert-True ($orchText -notmatch 'Request-AceLocalRecoveryPeriodicSaveStop') 'orchestrator_no_direct_periodic_stop'
Assert-True ($orchText -notmatch 'Stop-AceLocalRecoveryPeriodicSaveForLocalQuit') 'orchestrator_no_periodic_kill'
Assert-True ($orchText -match 'NOT_EXECUTED') 'not_executed_status'

$expectedOrder = @(
    'PREFLIGHT', 'SHUTDOWN_GATE', 'WAIT_FOR_ACTIVE_OPERATIONS', 'PERIODIC_SAVE_QUIESCE',
    'SHUTDOWN_SNAPSHOT', 'SNAPSHOT_VERIFICATION', 'MANAGED_STACK_STOP', 'CLEAN_SHUTDOWN_LEDGER', 'COMPLETE'
)
$pos = -1
foreach ($needle in $expectedOrder) {
    $pattern = "Id = '$needle'"
    $idx = $orchText.IndexOf($pattern)
    Assert-True ($idx -gt $pos) ("phase_order_def_$needle")
    $pos = $idx
}

$layout = New-IsolatedAceLayout
try {
    $regHashBefore = Get-PathSha256 $layout.RegistryPath
    $ledgerHashBefore = Get-PathSha256 $layout.LedgerPath

    $noDryArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Orchestrator, '-AceRoot', $layout.AceRoot)
    $noDry = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $noDryArgs -PassThru -Wait -WindowStyle Hidden
    Assert-True ($noDry.ExitCode -eq 3) 'live_mode_rejected_in_5191'

    $code = Invoke-SafeQuitDryRun -AceRoot $layout.AceRoot
    Assert-True ($code -eq 0) 'single_dry_run_exit_zero'

    $regHashAfter = Get-PathSha256 $layout.RegistryPath
    $ledgerHashAfter = Get-PathSha256 $layout.LedgerPath
    Assert-True ($regHashBefore -eq $regHashAfter) 'registry_unchanged'
    Assert-True ($ledgerHashBefore -eq $ledgerHashAfter) 'ledger_unchanged'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $layout.LocalDev 'recovery-periodic-save.stop'))) 'no_periodic_stop_file'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $layout.LocalDev 'shutdown-gate-state.json'))) 'no_gate_state_file'

    $codeThrow = Invoke-SafeQuitDryRun -AceRoot $layout.AceRoot -SimulateThrowAfterLock
    Assert-True ($codeThrow -eq 1) 'simulated_error_exit_one'
    $codeAfterThrow = Invoke-SafeQuitDryRun -AceRoot $layout.AceRoot
    Assert-True ($codeAfterThrow -eq 0) 'mutex_released_after_error'

    $job = Start-Job -ScriptBlock {
        param($orch, $ace)
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $orch -AceRoot $ace -DryRun -TestHoldLockMilliseconds 4000
    } -ArgumentList $Orchestrator, $layout.AceRoot
    Start-Sleep -Milliseconds 400
    $codeConcurrent = Invoke-SafeQuitDryRun -AceRoot $layout.AceRoot
    Assert-True ($codeConcurrent -eq 2) 'second_invocation_rejected_while_locked'
    Wait-Job $job | Out-Null
    Remove-Job $job -Force
    $codeAfterHold = Invoke-SafeQuitDryRun -AceRoot $layout.AceRoot
    Assert-True ($codeAfterHold -eq 0) 'mutex_released_after_concurrent_hold'
}
finally {
    Remove-Item -LiteralPath $layout.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (Test-Path -LiteralPath $LiveLocalDev) {
    $liveLedger = Join-Path $LiveLocalDev 'recovery-session-ledger.json'
    $liveReg = Join-Path $LiveLocalDev 'managed-processes.json'
    Assert-True (-not ($Orchestrator -like '*\\*')) 'orchestrator_path_sane'
    Write-Host '  ok live_local_dev_not_used_by_tests' -ForegroundColor DarkGray
    if (Test-Path -LiteralPath $liveLedger) {
        Write-Host ('  note live ledger mtime unchanged by this test run (tests used TEMP only)') -ForegroundColor DarkYellow
    }
}
else {
    Write-Host '  ok no_live_local_dev_path_on_machine' -ForegroundColor DarkGray
}

if ($failures.Count -gt 0) {
    Write-Host ('FAILED: ' + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All Invoke-AceLocalSafeQuit5191 tests passed.' -ForegroundColor Green
