#Requires -Version 5.1
<#
  Isolated Shutdown Gate integration tests for Invoke-AceLocalSafeQuit (5.19.2).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Orchestrator = Join-Path $ScriptDir 'Invoke-AceLocalSafeQuit.ps1'
. (Join-Path $ScriptDir 'AceLocalShutdownGateLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function New-IsolatedGateAceLayout {
    param([string]$LauncherSession = '20260101-120000')
    $root = Join-Path $env:TEMP ("ace-safe-quit-gate-{0}" -f [guid]::NewGuid().ToString('N'))
    $localDev = Join-Path $root '_backups\local-dev'
    $sessions = Join-Path $root '_backups\sessions'
    New-Item -ItemType Directory -Path $localDev -Force | Out-Null
    New-Item -ItemType Directory -Path $sessions -Force | Out-Null
    $registry = @{
        schemaVersion   = 1
        launcherSession = $LauncherSession
        updatedAt       = '2026-01-01T00:00:00.0000000Z'
        components      = @{}
    } | ConvertTo-Json -Depth 6
    Set-Content -LiteralPath (Join-Path $localDev 'managed-processes.json') -Value $registry -Encoding UTF8
    return @{
        AceRoot          = $root
        LocalDev         = $localDev
        SessionsRoot     = $sessions
        LauncherSession  = $LauncherSession
        GatePath         = Join-Path $localDev 'shutdown-gate-state.json'
    }
}

function Invoke-IsolatedGateTest {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$GateWaitSeconds = 15,
        [switch]$SimulateGateThrow
    )
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $Orchestrator,
        '-AceRoot', $AceRoot,
        '-IsolatedGateTest',
        '-LauncherSession', $LauncherSession,
        '-GateWaitSeconds', ([string]$GateWaitSeconds)
    )
    if ($SimulateGateThrow) { $args += '-TestSimulateGateThrow' }
    $p = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $args -PassThru -Wait -WindowStyle Hidden
    return $p.ExitCode
}

Write-Host 'Test-Invoke-AceLocalSafeQuit5192 (isolated gate)...' -ForegroundColor Cyan

$orchText = Get-Content -LiteralPath $Orchestrator -Raw
Assert-True ($orchText -notmatch 'New-AceLocalShutdownSnapshot') 'no_snapshot'
Assert-True ($orchText -notmatch 'Stop-AceLocalRecovery') 'no_periodic_stop'
Assert-True ($orchText -notmatch 'Register-AceLocalRecoveryCleanShutdownComplete') 'no_ledger_register'

# DryRun: no gate file under temp layout
$dryLayout = New-IsolatedGateAceLayout
try {
    $dryArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Orchestrator, '-AceRoot', $dryLayout.AceRoot, '-DryRun')
    $dry = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList $dryArgs -PassThru -Wait -WindowStyle Hidden
    Assert-True ($dry.ExitCode -eq 0) 'dryrun_exit_zero'
    Assert-True (-not (Test-Path -LiteralPath $dryLayout.GatePath)) 'dryrun_no_gate_file'
}
finally {
    Remove-Item -LiteralPath $dryLayout.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Idle → SAFE_TO_STOP
$idle = New-IsolatedGateAceLayout
try {
    $code = Invoke-IsolatedGateTest -AceRoot $idle.AceRoot -LauncherSession $idle.LauncherSession
    Assert-True ($code -eq 0) 'idle_safe_to_stop'
    Assert-True (Test-Path -LiteralPath $idle.GatePath) 'gate_file_under_temp_only'
    $gate = Get-Content -LiteralPath $idle.GatePath -Raw | ConvertFrom-Json
    Assert-True ([string]$gate.gateStatus -eq 'SAFE_TO_STOP') 'idle_gate_status_safe'
}
finally {
    Remove-Item -LiteralPath $idle.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Active NEW → timeout (no advance to snapshot phases — exit 6)
$active = New-IsolatedGateAceLayout
try {
    $sid = 'ori-20260101-120000-abcdef01'
    $sessionDir = Join-Path $active.SessionsRoot $sid
    New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $sessionDir 'session.json') -Value (@{
            status = 'IN_PROGRESS'; sessionId = $sid
        } | ConvertTo-Json) -Encoding UTF8
    $codeWait = Invoke-IsolatedGateTest -AceRoot $active.AceRoot -LauncherSession $active.LauncherSession -GateWaitSeconds 1
    Assert-True ($codeWait -eq 6) 'active_operation_gate_timeout'
    $gateW = Get-Content -LiteralPath $active.GatePath -Raw | ConvertFrom-Json
    Assert-True ([string]$gateW.gateStatus -eq 'WAITING_FOR_OPERATION') 'active_gate_waiting'
}
finally {
    Remove-Item -LiteralPath $active.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Corrupt progress → BLOCKED
$corrupt = New-IsolatedGateAceLayout
try {
    $sid = 'ori-20260101-120000-deadbeef'
    $sessionDir = Join-Path $corrupt.SessionsRoot $sid
    New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $sessionDir 'cancel-execute-progress-deadbeef.json') -Value '{bad' -Encoding UTF8
    $codeB = Invoke-IsolatedGateTest -AceRoot $corrupt.AceRoot -LauncherSession $corrupt.LauncherSession -GateWaitSeconds 3
    Assert-True ($codeB -eq 5) 'corrupt_progress_blocked'
    $gateB = Get-Content -LiteralPath $corrupt.GatePath -Raw | ConvertFrom-Json
    Assert-True ([string]$gateB.gateStatus -eq 'SHUTDOWN_BLOCKED') 'corrupt_gate_blocked'
}
finally {
    Remove-Item -LiteralPath $corrupt.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Backend unreachable → BLOCKED (direct gate API)
$unreach = New-IsolatedGateAceLayout
try {
    Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $unreach.AceRoot
        launcherSession = $unreach.LauncherSession
    } | Out-Null
    $ref = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action           = 'refresh'
        aceRoot          = $unreach.AceRoot
        launcherSession  = $unreach.LauncherSession
        sessionsRoot     = $unreach.SessionsRoot
        backendReachable = $false
    }
    Assert-True ([string]$ref.state.gateStatus -eq 'SHUTDOWN_BLOCKED') 'backend_unreachable_blocked'
}
finally {
    Remove-Item -LiteralPath $unreach.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Concurrent shutdown sessions → BLOCKED on second request
$conc = New-IsolatedGateAceLayout
try {
    Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $conc.AceRoot
        launcherSession = '20260101-120000'
    } | Out-Null
    $second = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $conc.AceRoot
        launcherSession = '20260101-999999'
    }
    Assert-True ([string]$second.state.gateStatus -eq 'SHUTDOWN_BLOCKED') 'concurrent_shutdown_blocked'
}
finally {
    Remove-Item -LiteralPath $conc.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Session mismatch on operation_start_allowed
$mis = New-IsolatedGateAceLayout
try {
    Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $mis.AceRoot
        launcherSession = $mis.LauncherSession
    } | Out-Null
    Invoke-AceLocalShutdownGatePython -RequestObject @{
        action           = 'refresh'
        aceRoot          = $mis.AceRoot
        launcherSession  = $mis.LauncherSession
        sessionsRoot     = $mis.SessionsRoot
        backendReachable = $true
    } | Out-Null
    $allow = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action                   = 'operation_start_allowed'
        aceRoot                  = $mis.AceRoot
        expectedLauncherSession  = 'wrong-session-id'
    }
    Assert-True (-not $allow.allowed) 'session_mismatch_freeze'
    Assert-True ([string]$allow.reason -eq 'ace_shutdown_gate_session_mismatch') 'session_mismatch_reason'
}
finally {
    Remove-Item -LiteralPath $mis.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Gate exception → safe failure (exit 1), mutex released for retry
$throwLayout = New-IsolatedGateAceLayout
try {
    $codeThrow = Invoke-IsolatedGateTest -AceRoot $throwLayout.AceRoot -LauncherSession $throwLayout.LauncherSession -SimulateGateThrow
    Assert-True ($codeThrow -eq 1) 'gate_simulated_throw_exit_one'
    $codeRetry = Invoke-IsolatedGateTest -AceRoot $throwLayout.AceRoot -LauncherSession $throwLayout.LauncherSession
    Assert-True ($codeRetry -eq 0) 'mutex_ok_after_gate_throw'
}
finally {
    Remove-Item -LiteralPath $throwLayout.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Isolated rejected outside TEMP
$codeLivePath = Invoke-IsolatedGateTest -AceRoot (Get-AceLocalShutdownGateBackendRoot) -LauncherSession '20260101-120000'
Assert-True ($codeLivePath -eq 7) 'isolated_rejects_non_temp_aceroot'

if ($failures.Count -gt 0) {
    Write-Host ('FAILED: ' + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All Invoke-AceLocalSafeQuit5192 tests passed.' -ForegroundColor Green
