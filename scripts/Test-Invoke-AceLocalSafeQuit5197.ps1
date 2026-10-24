#Requires -Version 5.1
<#
  Isolated CLEAN_SHUTDOWN_LEDGER tests for Invoke-AceLocalSafeQuit (5.19.7).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Orchestrator = Join-Path $ScriptDir 'Invoke-AceLocalSafeQuit.ps1'
. (Join-Path $ScriptDir 'AceLocalSafeQuitManagedStackStopLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitCleanShutdownLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')
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
        git config user.email 'safe-quit-5197@ace.local' 2>$null | Out-Null
        git config user.name 'SafeQuit5197' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'README.md') -Value 'base' -Encoding UTF8
        git add README.md 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-IsolatedCleanShutdownLayout {
    param([string]$LauncherSession = '20260101-120000')
    $tempRoot = Join-Path $env:TEMP ("ace-safe-quit-ledger5197-{0}" -f [guid]::NewGuid().ToString('N'))
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
        LauncherSession = $LauncherSession
        LedgerPath      = Join-Path $localDev 'recovery-session-ledger.json'
    }
}

function Invoke-IsolatedCleanShutdownTest {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [switch]$SimulateSnapshotVerificationFailure,
        [switch]$SimulatePartialStackStop,
        [switch]$SimulateLedgerRegisterThrow
    )
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $Orchestrator,
        '-AceRoot', $AceRoot,
        '-IsolatedCleanShutdownTest',
        '-LauncherSession', $LauncherSession,
        '-GateWaitSeconds', '15',
        '-PeriodicSaveQuiesceWaitSeconds', '15'
    )
    if ($SimulateSnapshotVerificationFailure) { $args += '-TestSimulateSnapshotVerificationFailure' }
    if ($SimulatePartialStackStop) { $args += '-TestSimulatePartialStackStop' }
    if ($SimulateLedgerRegisterThrow) { $args += '-TestSimulateLedgerRegisterThrow' }
    $p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $args -PassThru -Wait -WindowStyle Hidden
    return $p.ExitCode
}

Write-Host 'Test-Invoke-AceLocalSafeQuit5197 (isolated clean shutdown)...' -ForegroundColor Cyan

$orchText = Get-Content -LiteralPath $Orchestrator -Raw
Assert-True ($orchText -match 'IsolatedCleanShutdownTest') 'isolated_clean_shutdown_switch'
Assert-True ($orchText -notmatch 'Register-AceLocalRecoveryCleanShutdownComplete') 'orchestrator_no_direct_ledger_register'
Assert-True (Test-Path -LiteralPath (Join-Path $ScriptDir 'AceLocalSafeQuitCleanShutdownLib.ps1')) 'clean_shutdown_lib_present'

# Stack-only mode still skips ledger
$stackOnly = New-IsolatedCleanShutdownLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $stackOnly.AceRoot -LauncherSession $stackOnly.LauncherSession)
    $args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Orchestrator, '-AceRoot', $stackOnly.AceRoot,
        '-IsolatedStackStopTest', '-LauncherSession', $stackOnly.LauncherSession)
    $p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $args -PassThru -Wait -WindowStyle Hidden
    Assert-True ($p.ExitCode -eq 0) 'stack_only_still_no_ledger'
    Assert-True (-not (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $stackOnly.AceRoot)) 'stack_only_ledger_not_recorded'
}
finally {
    Remove-Item -LiteralPath $stackOnly.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Full success
$ok = New-IsolatedCleanShutdownLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $ok.AceRoot -LauncherSession $ok.LauncherSession `
        -IncludeLegacyBootstrap -IncludeMemuraiWatch -IncludePeriodicSave)
    $code = Invoke-IsolatedCleanShutdownTest -AceRoot $ok.AceRoot -LauncherSession $ok.LauncherSession
    Assert-True ($code -eq 0) 'full_pipeline_exit_zero'
    Assert-True (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $ok.AceRoot -LauncherSession $ok.LauncherSession) 'clean_shutdown_recorded'
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $ok.AceRoot -AllowCreateStateDir
    Assert-True ($ledger.PSObject.Properties.Match('openIncident').Count -eq 0 -or $null -eq $ledger.openIncident) 'open_incident_cleared_on_success'
}
finally {
    Remove-Item -LiteralPath $ok.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Duplicate registration attempt
$dup = New-IsolatedCleanShutdownLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $dup.AceRoot -LauncherSession $dup.LauncherSession)
    $c1 = Invoke-IsolatedCleanShutdownTest -AceRoot $dup.AceRoot -LauncherSession $dup.LauncherSession
    Assert-True ($c1 -eq 0) 'duplicate_first_success'
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $dup.AceRoot -LauncherSession $dup.LauncherSession)
    $c2 = Invoke-IsolatedCleanShutdownTest -AceRoot $dup.AceRoot -LauncherSession $dup.LauncherSession
    Assert-True ($c2 -eq 14) 'duplicate_second_exit_14'
}
finally {
    Remove-Item -LiteralPath $dup.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Verification failure — no ledger
$verFail = New-IsolatedCleanShutdownLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $verFail.AceRoot -LauncherSession $verFail.LauncherSession)
    $codeV = Invoke-IsolatedCleanShutdownTest -AceRoot $verFail.AceRoot -LauncherSession $verFail.LauncherSession `
        -SimulateSnapshotVerificationFailure
    Assert-True ($codeV -eq 12) 'verify_fail_exit_12'
    Assert-True (-not (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $verFail.AceRoot)) 'verify_fail_no_ledger'
}
finally {
    Remove-Item -LiteralPath $verFail.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Partial stack stop — no ledger
$partial = New-IsolatedCleanShutdownLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $partial.AceRoot -LauncherSession $partial.LauncherSession)
    $codeP = Invoke-IsolatedCleanShutdownTest -AceRoot $partial.AceRoot -LauncherSession $partial.LauncherSession `
        -SimulatePartialStackStop
    Assert-True ($codeP -eq 13) 'partial_stack_exit_13'
    Assert-True (-not (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $partial.AceRoot)) 'partial_stack_no_ledger'
}
finally {
    Remove-Item -LiteralPath $partial.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Open recovery incident
$incident = New-IsolatedCleanShutdownLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $incident.AceRoot -LauncherSession $incident.LauncherSession)
    [void](Open-AceLocalRecoveryLedgerIncident -AceRoot $incident.AceRoot -Kind 'test_incident' -Detail '5197 open incident')
    $codeI = Invoke-IsolatedCleanShutdownTest -AceRoot $incident.AceRoot -LauncherSession $incident.LauncherSession
    Assert-True ($codeI -eq 14) 'open_incident_exit_14'
    Assert-True (-not (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $incident.AceRoot)) 'incident_no_clean_shutdown'
    $ledgerI = Read-AceLocalRecoveryLedger -AceRoot $incident.AceRoot -AllowCreateStateDir
    Assert-True ($null -ne $ledgerI.openIncident) 'incident_preserved_after_failed_quit'
}
finally {
    Remove-Item -LiteralPath $incident.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Ledger session mismatch — preconditions block register
$sessionMis = New-IsolatedCleanShutdownLayout -LauncherSession '20260101-120000'
try {
    $ledgerWrong = Get-Content -LiteralPath $sessionMis.LedgerPath -Raw | ConvertFrom-Json
    $ledgerWrong.launcherSession = 'other-ledger-session'
    ($ledgerWrong | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $sessionMis.LedgerPath -Encoding UTF8
    $gateOk = @{ Ok = $true; GateStatus = 'SAFE_TO_STOP' }
    $quiesceOk = @{ Ok = $true }
    $snapRootMis = Join-Path $sessionMis.LocalDev 'shutdown-snapshots'
    $cap = New-AceLocalShutdownSnapshot -BackendRoot $sessionMis.Backend -FrontendRoot $sessionMis.Frontend `
        -AceRoot $sessionMis.AceRoot -SnapshotsRootOverride $snapRootMis -DeferLastVerifiedPointerUpdate
    $capWrap = @{ Ok = $cap.Ok; SnapshotId = $cap.SnapshotId; SnapshotRoot = $cap.SnapshotRoot; Status = $cap.Status }
    $ver = Invoke-AceLocalSafeQuitSnapshotVerification -AceRoot $sessionMis.AceRoot -CaptureResult $capWrap
    $stackOk = @{ Ok = $true }
    $pre = Test-AceLocalSafeQuitCleanShutdownPreconditions -AceRoot $sessionMis.AceRoot -LauncherSession '20260101-120000' `
        -GateResult $gateOk -QuiesceResult $quiesceOk -CaptureResult $capWrap -VerificationResult $ver -StackStopResult $stackOk
    Assert-True (-not $pre.Ok -and $pre.Reason -eq 'ledger_session_mismatch') 'ledger_session_mismatch_precondition'
    $regAttempt = Invoke-AceLocalSafeQuitCleanShutdownLedgerRegister -AceRoot $sessionMis.AceRoot -LauncherSession '20260101-120000' `
        -GateResult $gateOk -QuiesceResult $quiesceOk -CaptureResult $capWrap -VerificationResult $ver -StackStopResult $stackOk
    Assert-True (-not $regAttempt.Ok) 'ledger_session_mismatch_register_rejected'
}
finally {
    Remove-Item -LiteralPath $sessionMis.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Simulated register exception
$regThrow = New-IsolatedCleanShutdownLayout
try {
    [void](Register-AceLocalSafeQuitTestManagedStack -AceRoot $regThrow.AceRoot -LauncherSession $regThrow.LauncherSession)
    $codeT = Invoke-IsolatedCleanShutdownTest -AceRoot $regThrow.AceRoot -LauncherSession $regThrow.LauncherSession `
        -SimulateLedgerRegisterThrow
    Assert-True ($codeT -eq 14) 'ledger_register_throw_exit_14'
    Assert-True (-not (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $regThrow.AceRoot)) 'register_throw_no_success_flag'
}
finally {
    Remove-Item -LiteralPath $regThrow.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host ('FAILED: ' + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All Invoke-AceLocalSafeQuit5197 tests passed.' -ForegroundColor Green
