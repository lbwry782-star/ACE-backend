#Requires -Version 5.1
<#
.SYNOPSIS
  Start ACE Backend for local development (127.0.0.1 only).

.DESCRIPTION
  Loads .env.local, runs safety checks, and starts python app.py.
  Ensures a single ACE-Backend app.py instance for this repo on the loopback port.
  Does not start Redis, Worker, or paid API calls.

.PARAMETER InspectOnly
  Report port/process state and startup decision without starting app.py.
#>
[CmdletBinding()]
param(
    [switch]$InspectOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

. (Join-Path $ScriptDir "AceLocalEnvCommon.ps1")

Write-Info "ACE Backend - local startup"
Write-Info "Project folder: $RepoRoot"

Assert-LocalDevEnvFile -RepoRoot $RepoRoot

$port = 5000
$portRaw = Get-EnvString "PORT"
if ($portRaw) {
    $parsedPort = 0
    if ([int]::TryParse($portRaw, [ref]$parsedPort)) {
        $port = $parsedPort
    }
}

Assert-BackendLocalEnvSafety -RepoRoot $RepoRoot -BackendPort $port

$portState = Get-AceBackendLocalPortState -Port $port -BackendRoot $RepoRoot
Write-AceBackendLocalPortStateReport -State $portState

if ($InspectOnly) {
    $portState | Select-Object Port, BackendRoot, HealthOk, AceBackendProcessCount, ListenerPids, AceListenerPids, ForeignListenerPids, UncertainListenerPids, Action, Reason | ConvertTo-Json -Compress
    if ($portState.Action -in @("Start", "AlreadyRunning")) {
        exit 0
    }
    exit 2
}

switch ($portState.Action) {
    "AlreadyRunning" {
        Write-Ok "Backend is already running for this repo on http://127.0.0.1:$port (process + /health verified)."
        Write-Ok "No second server was started."
        exit 0
    }
    "BlockedForeign" {
        Write-Err "Port $port is in use by a non-ACE-Backend process (PID: $($portState.ForeignListenerPids -join ', '))."
        Write-Err "This script will not stop foreign programs. Close them manually or change PORT in .env.local."
        exit 1
    }
    "BlockedDuplicate" {
        Write-Err "Multiple or orphan ACE Backend (app.py) processes detected for this repo."
        foreach ($row in $portState.AceBackendProcesses) {
            Write-Err "  PID $($row.ProcId): $($row.CommandLine)"
        }
        Write-Err "Stop extra processes manually, then run this script again. Automatic restart is not performed."
        exit 1
    }
    "BlockedUnhealthy" {
        Write-Err "ACE Backend is listening on port $port (PID: $($portState.AceListenerPids -join ', ')) but /health failed."
        Write-Err "Do not start a second instance. Investigate the existing window or stop it manually before restarting."
        exit 1
    }
    "BlockedUncertain" {
        Write-Err "Cannot determine whether port $port is safe to use ($($portState.Reason))."
        Write-Err "No server was started. Resolve manually before retrying."
        exit 1
    }
    "Start" {
        break
    }
    default {
        Write-Err "Unexpected startup decision: $($portState.Action)"
        exit 1
    }
}

$python = Resolve-AceBackendPythonExe -BackendRoot $RepoRoot
if (-not $python) {
    Write-Err "Python not found. Create venv: python -m venv .venv312"
    exit 1
}

$appPy = Join-Path $RepoRoot "app.py"
if (-not (Test-Path -LiteralPath $appPy)) {
    Write-Err "app.py not found. Run this script from the ACE-Backend repository."
    exit 1
}

$env:ACE_LOCAL_DEV_ENABLED = "true"
if (-not $env:ACE_SECURITY_ENABLED) {
    $env:ACE_SECURITY_ENABLED = "false"
}
$env:PORT = "$port"

Write-Ok "Python: $python"
Write-Ok "Starting backend on http://127.0.0.1:$port (local dev, loopback only)"
Write-Info "Press Ctrl+C in this window to stop the server."
Write-Host ""

& $python $appPy
