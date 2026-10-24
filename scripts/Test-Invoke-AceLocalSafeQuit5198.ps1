#Requires -Version 5.1
<#
  Isolated tests for shutdown gate Builder1/Builder2 activity (5.19.8).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = (Resolve-Path (Join-Path $ScriptDir '..')).ProviderPath
$GateLib = Join-Path $ScriptDir 'AceLocalShutdownGateLib.ps1'
. $GateLib

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function Resolve-TestPython {
    $candidates = @(
        (Join-Path $BackendRoot '.venv312\Scripts\python.exe'),
        (Join-Path $BackendRoot 'venv\Scripts\python.exe'),
        (Join-Path $BackendRoot '.venv\Scripts\python.exe')
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) { return $path }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function New-IsolatedAceLayout5198 {
    $root = Join-Path $env:TEMP ("ace-safe-quit-5198-{0}" -f [guid]::NewGuid().ToString('N'))
    $localDev = Join-Path $root '_backups\local-dev'
    $sessions = Join-Path $root '_backups\sessions'
    New-Item -ItemType Directory -Path $localDev -Force | Out-Null
    New-Item -ItemType Directory -Path $sessions -Force | Out-Null
    $registry = @{
        launcherSession = 'test-session-5198'
        components      = @{ worker = @{ pid = $PID } }
    } | ConvertTo-Json -Depth 6
    Set-Content -LiteralPath (Join-Path $localDev 'managed-processes.json') -Value $registry -Encoding UTF8
    return @{
        AceRoot          = $root
        LauncherSession  = 'test-session-5198'
        SessionsRoot     = $sessions
    }
}

Write-Host 'Test-Invoke-AceLocalSafeQuit5198 (builder shutdown activity)...' -ForegroundColor Cyan

$py = Resolve-TestPython
Assert-True ($null -ne $py) 'python_available'
if ($py) {
    $unittest = & $py -m pytest (Join-Path $BackendRoot 'tests\test_ace_local_dev_shutdown_builder_activity.py') -q 2>&1
    $unittestText = ($unittest | Out-String).Trim()
    Write-Host $unittestText
    Assert-True ($LASTEXITCODE -eq 0) 'pytest_builder_activity'
}

$idle = New-IsolatedAceLayout5198
try {
    $refreshIdle = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'refresh'
        aceRoot         = $idle.AceRoot
        launcherSession = $idle.LauncherSession
        sessionsRoot    = $idle.SessionsRoot
    }
    Assert-True ($refreshIdle.state.gateStatus -eq 'RUNNING') 'idle_gate_running_without_request'
    Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $idle.AceRoot
        launcherSession = $idle.LauncherSession
    } | Out-Null
    $safeIdle = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'refresh'
        aceRoot         = $idle.AceRoot
        launcherSession = $idle.LauncherSession
        sessionsRoot    = $idle.SessionsRoot
    }
    Assert-True ($safeIdle.state.gateStatus -eq 'SAFE_TO_STOP') 'idle_becomes_safe_with_no_builder'
}
finally {
    Remove-Item -LiteralPath $idle.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$blocked = New-IsolatedAceLayout5198
try {
    Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $blocked.AceRoot
        launcherSession = $blocked.LauncherSession
    } | Out-Null
    $override = @{
        observations = @(
            @{ kind = 'BUILDER1'; status = 'ACTIVE'; jobId = 'b1-test'; notes = @('builder1_running') }
        )
        redisReachable = $true
        workerReachable = $true
        notes = @()
    }
    $wait = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action                   = 'refresh'
        aceRoot                  = $blocked.AceRoot
        launcherSession          = $blocked.LauncherSession
        sessionsRoot             = $blocked.SessionsRoot
        builderActivityOverride  = $override
    }
    Assert-True ($wait.state.gateStatus -eq 'WAITING_FOR_OPERATION') 'builder1_active_waits'
    $scan = $wait.state.evaluation.scan
    Assert-True ($null -ne $scan.builderActivity) 'scan_includes_builder_activity_block'
}
finally {
    Remove-Item -LiteralPath $blocked.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$redisDown = New-IsolatedAceLayout5198
try {
    Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $redisDown.AceRoot
        launcherSession = $redisDown.LauncherSession
    } | Out-Null
    $blockedRedis = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'refresh'
        aceRoot         = $redisDown.AceRoot
        launcherSession = $redisDown.LauncherSession
        sessionsRoot    = $redisDown.SessionsRoot
        redisReachable  = $false
    }
    Assert-True ($blockedRedis.state.gateStatus -eq 'SHUTDOWN_BLOCKED') 'redis_unreachable_blocks'
}
finally {
    Remove-Item -LiteralPath $redisDown.AceRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host ("Failures: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}

Write-Host 'All Invoke-AceLocalSafeQuit5198 tests passed.' -ForegroundColor Green
exit 0
