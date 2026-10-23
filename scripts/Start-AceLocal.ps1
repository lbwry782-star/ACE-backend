#Requires -Version 5.1
<#
.SYNOPSIS
  Start full ACE local dev stack (Memurai, Backend, Worker, Frontend).

.PARAMETER ValidateOnly
  Safety checks and status report only (no starts, no browser).
#>
[CmdletBinding()]
param(
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "AceLocalEnvCommon.ps1")

$ExpectedRedisUrl = "redis://127.0.0.1:6379/0"
$BackendPort = 5000
$DefaultPublicBase = "http://127.0.0.1:5000"
$FrontendPortMin = 3000
$FrontendPortMax = 3010

$BackendRoot = Resolve-AceBackendRoot -StartPath $ScriptDir
if (-not $BackendRoot) {
    Write-Err "ACE-Backend folder not found (app.py + worker_video.py)."
    exit 1
}
$FrontendRoot = Resolve-AceFrontendRoot -BackendRoot $BackendRoot
if (-not $FrontendRoot) {
    Write-Err "ACE-Frontend folder not found (package.json)."
    exit 1
}

$BackendStartCmd = Join-Path $BackendRoot "scripts\Start-AceBackendLocal.cmd"
$WorkerStartCmd = Join-Path $BackendRoot "scripts\Start-AceWorkerLocal.cmd"
$FrontendStartCmd = Join-Path $FrontendRoot "scripts\Start-AceFrontendLocal.cmd"

foreach ($path in @($BackendStartCmd, $WorkerStartCmd, $FrontendStartCmd)) {
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Err "Missing launcher: $path"
        exit 1
    }
}

function Test-LocalDevSafety([string]$RepoRoot, [string]$FrontendRoot) {
    Assert-LocalDevEnvFile -RepoRoot $RepoRoot
    Assert-LocalRedisUrl -ExpectedUrl $ExpectedRedisUrl
    Assert-LocalPublicBaseUrl -ExpectedBase $DefaultPublicBase -BackendPort $BackendPort
    $feEnv = Join-Path $FrontendRoot ".env.development.local"
    if (Test-Path -LiteralPath $feEnv) {
        Import-DotEnvFile $feEnv | Out-Null
    }
    $viteUrl = Get-EnvString "VITE_BACKEND_URL"
    if ($viteUrl -and (Test-ProductionPublicBaseUrl $viteUrl)) {
        Write-Err "VITE_BACKEND_URL points at production. Stopped."
        exit 1
    }
    if ($viteUrl) {
        try {
            $uri = [Uri]$viteUrl
            if (-not (Test-LoopbackHost $uri.Host)) {
                Write-Err "VITE_BACKEND_URL must use loopback (127.0.0.1), not production."
                exit 1
            }
        }
        catch {
            Write-Err "VITE_BACKEND_URL is invalid."
            exit 1
        }
    }
}

function Ensure-MemuraiRunning {
    $ping = Test-LocalMemuraiPing -RedisUrl $ExpectedRedisUrl
    if ($ping.Ok) {
        Write-Ok 'Memurai: PONG (127.0.0.1:6379)'
        return $true
    }
    Write-Warn ("Memurai: no PONG at {0} ({1})" -f "127.0.0.1:6379", $ping.Error)
    $svc = Get-Service -Name "Memurai" -ErrorAction SilentlyContinue
    if (-not $svc) {
        Write-Err "Windows service Memurai not found. Start Memurai Developer manually."
        return $false
    }
    Write-Info ("Memurai service status: {0}" -f $svc.Status)
    if ($ValidateOnly) {
        Write-Warn "ValidateOnly: will not start service without approval."
        return $false
    }
    $answer = Read-Host "Start Memurai Windows service now? (y/n)"
    if ($answer -notmatch "^[yY]") {
        Write-Err "Cancelled. Local Redis is required."
        return $false
    }
    try {
        if ($svc.Status -ne "Running") {
            Start-Service -Name "Memurai" -ErrorAction Stop
            Start-Sleep -Seconds 2
        }
    }
    catch {
        Write-Err ("Could not start Memurai (admin may be required): {0}" -f $_.Exception.Message)
        return $false
    }
    $ping2 = Test-LocalMemuraiPing -RedisUrl $ExpectedRedisUrl
    if (-not $ping2.Ok) {
        Write-Err "Memurai still not responding after service start."
        return $false
    }
    Write-Ok "Memurai: PONG"
    return $true
}

function Start-ComponentWindow([string]$Title, [string]$CmdPath, [string]$WorkingDirectory, [hashtable]$ExtraEnv) {
    $envLines = ""
    if ($ExtraEnv) {
        foreach ($key in $ExtraEnv.Keys) {
            $val = $ExtraEnv[$key]
            $envLines += ('set {0}={1}&& ' -f $key, $val)
        }
    }
    $inner = '{0}call "{1}"' -f $envLines, $CmdPath
    $arg = '/k title ' + $Title + ' && ' + $inner
    Start-Process -FilePath "cmd.exe" -ArgumentList $arg -WorkingDirectory $WorkingDirectory | Out-Null
}

Write-Info "=== ACE Ori - local dev stack ==="
Write-Info ("Backend: {0}" -f $BackendRoot)
Write-Info ("Frontend: {0}" -f $FrontendRoot)

Push-Location $BackendRoot
try {
    Test-LocalDevSafety -RepoRoot $BackendRoot -FrontendRoot $FrontendRoot
}
finally {
    Pop-Location
}

$memuraiOk = Ensure-MemuraiRunning
if (-not $memuraiOk -and -not $ValidateOnly) {
    Read-Host "Press Enter to close"
    exit 1
}

$backendPortState = Get-AceBackendLocalPortState -Port $BackendPort -BackendRoot $BackendRoot
$backendOk = $false
if ($backendPortState.Action -eq "AlreadyRunning") {
    $backendOk = (Test-BackendLocalDevIdentity -Port $BackendPort).Ok
}
elseif ($backendPortState.Action -ne "Start") {
    Write-Err ("Backend port/process state blocks stack start: {0} ({1})" -f $backendPortState.Action, $backendPortState.Reason)
    Write-AceBackendLocalPortStateReport -State $backendPortState
    if (-not $ValidateOnly) {
        Read-Host "Press Enter to close"
        exit 1
    }
}
$backendProcs = @(Find-LocalAceBackendProcesses -BackendRoot $BackendRoot)
$workerProcs = @(Find-LocalAceWorkerProcesses -BackendRoot $BackendRoot)
$frontendPorts = @(Find-AceFrontendVitePorts -FromPort $FrontendPortMin -ToPort $FrontendPortMax)

if ($frontendPorts.Count -gt 1) {
    Write-Warn ("Multiple ACE Vite servers detected (ports: {0}). None were stopped." -f ($frontendPorts -join ", "))
}

if ($ValidateOnly) {
    Write-Info "--- ValidateOnly: component status ---"
    Write-Host ("Memurai PONG: " + $(if ($memuraiOk) { "OK" } else { "FAILED" }))
    Write-Host ("Backend port decision: " + $backendPortState.Action + " (" + $backendPortState.Reason + ")")
    Write-Host ("Backend ACE identity: " + $(if ($backendOk) { "OK" } else { "missing / not verified" }))
    Write-Host ("Backend app.py processes in repo path: " + $backendProcs.Count)
    Write-Host ("Worker worker_video.py: " + $(if ($workerProcs.Count -gt 0) { "running ($($workerProcs.Count))" } else { "not running" }))
    Write-Host ("Frontend ACE Vite: " + $(if ($frontendPorts.Count -gt 0) { "port $($frontendPorts[0])" } else { "not running" }))
    if ($frontendPorts.Count -gt 1) {
        Write-Warn "Two or more Vite ports matched ACE frontend fingerprint."
    }
    Write-Ok "ValidateOnly: env safety OK; nothing started."
    if (-not $memuraiOk) {
        Write-Warn "ValidateOnly: Memurai not reachable (full start would stop here)."
        exit 1
    }
    exit 0
}

$started = @()

if ($backendOk) {
    Write-Ok ("Backend: already running (127.0.0.1:{0}, local dev verified)" -f $BackendPort)
}
elseif ($backendPortState.Action -ne "Start") {
    Write-Err "Backend: not started (see port/process decision above)."
    Read-Host "Press Enter to close"
    exit 1
}
else {
    Write-Info "Backend: starting new window..."
    Start-ComponentWindow -Title "ACE Backend Local" -CmdPath $BackendStartCmd -WorkingDirectory $BackendRoot -ExtraEnv @{}
    if (Wait-BackendLocalReady -Port $BackendPort -TimeoutSec 90) {
        Write-Ok "Backend: ready."
        $started += "Backend"
        $backendOk = $true
    }
    else {
        Write-Err "Backend: timed out. Check ACE Backend window."
        Read-Host "Press Enter to close"
        exit 1
    }
}

if ($workerProcs.Count -gt 0) {
    Write-Ok ("Worker: already running (PID {0})" -f $workerProcs[0].ProcId)
}
else {
    Write-Info "Worker: starting new window..."
    Start-ComponentWindow -Title "ACE Worker Local" -CmdPath $WorkerStartCmd -WorkingDirectory $BackendRoot -ExtraEnv @{}
    Start-Sleep -Seconds 3
    $workerProcs = @(Find-LocalAceWorkerProcesses -BackendRoot $BackendRoot)
    if ($workerProcs.Count -gt 0) {
        Write-Ok ("Worker: running (PID {0})" -f $workerProcs[0].ProcId)
        $started += "Worker"
    }
    else {
        Write-Warn "Worker: window opened; if it failed, check ACE Worker window."
    }
}

$frontendPort = $null
if ($frontendPorts.Count -ge 1) {
    $frontendPort = $frontendPorts[0]
    Write-Ok ("Frontend: already running (http://localhost:{0}/)" -f $frontendPort)
}
else {
    Write-Info "Frontend: starting new window..."
    Start-ComponentWindow -Title "ACE Frontend Local" -CmdPath $FrontendStartCmd -WorkingDirectory $FrontendRoot -ExtraEnv @{ ACE_FRONTEND_LOCAL_NO_PAUSE = "1" }
    $frontendPort = Wait-AceFrontendVite -FromPort $FrontendPortMin -ToPort $FrontendPortMax -TimeoutSec 120
    if ($null -ne $frontendPort) {
        Write-Ok ("Frontend: ready (http://localhost:{0}/)" -f $frontendPort)
        $started += "Frontend"
    }
    else {
        Write-Err "Frontend: timed out. Check Vite window."
        Read-Host "Press Enter to close"
        exit 1
    }
}

if (-not $backendOk) {
    Write-Err "Backend not verified. Not reporting success."
    Read-Host "Press Enter to close"
    exit 1
}

$frontendUrl = "http://localhost:$frontendPort/"
Write-Info ("Opening browser: {0}" -f $frontendUrl)
Start-Process $frontendUrl | Out-Null

Write-Host ""
Write-Ok "=== Local stack ready ==="
if ($started.Count -gt 0) {
    Write-Info ("Started now: {0}" -f ($started -join ", "))
}
else {
    Write-Info "All components were already running."
}
Write-Info ("Site: {0}" -f $frontendUrl)
Write-Info ("Backend health: http://127.0.0.1:{0}/health" -f $BackendPort)
Write-Warn "Keep Backend, Worker, and Frontend windows open."
Write-Host ""
Read-Host "Press Enter to close this launcher (servers keep running)"
