#Requires -Version 5.1
<#
.SYNOPSIS
  Start ACE video Worker (worker_video.py) for local Builder2 development.

.PARAMETER ValidateOnly
  Run safety checks only; do not start worker_video.py.
#>
[CmdletBinding()]
param(
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

. (Join-Path $ScriptDir "AceLocalEnvCommon.ps1")

$ExpectedRedisUrl = "redis://127.0.0.1:6379/0"
$DefaultBackendPort = 5000
$DefaultPublicBase = "http://127.0.0.1:5000"

Write-Info "ACE Worker - local startup"
Write-Info "Project folder: $RepoRoot"

Assert-LocalDevEnvFile -RepoRoot $RepoRoot

$port = $DefaultBackendPort
$portRaw = Get-EnvString "PORT"
if ($portRaw) {
    $parsedPort = 0
    if ([int]::TryParse($portRaw, [ref]$parsedPort)) {
        $port = $parsedPort
    }
}

Assert-LocalRedisUrl -ExpectedUrl $ExpectedRedisUrl
Assert-LocalPublicBaseUrl -ExpectedBase $DefaultPublicBase -BackendPort $port

$ping = Test-LocalMemuraiPing -RedisUrl $ExpectedRedisUrl
if (-not $ping.Ok) {
    Write-Err "Memurai/Redis PING failed: $($ping.Error)"
    Write-Err "Ensure Memurai is running on 127.0.0.1:6379."
    exit 1
}
Write-Ok "Memurai PING OK ($ExpectedRedisUrl)"

$backend = Test-BackendLocalDevIdentity -Port $port
if (-not $backend.Ok) {
    Write-Err "Local Backend identity check failed: $($backend.Error)"
    Write-Err "Start Backend with scripts\Start-AceBackendLocal.cmd first."
    exit 1
}
Write-Ok "Local Backend verified (health + security off + Redis on server)"

$existing = @(Find-LocalAceWorkerProcesses -BackendRoot $RepoRoot)
if ($existing.Count -gt 0) {
    Write-Ok "ACE Worker already running ($($existing.Count) process(es)). Not starting another."
    foreach ($row in $existing) {
        Write-Info "  PID $($row.ProcId): $($row.CommandLine)"
    }
    exit 0
}

$python = Resolve-AceBackendPythonExe -BackendRoot $RepoRoot
if (-not $python) {
    Write-Err "Python not found. Install Python 3.8+ or create venv: python -m venv .venv312"
    exit 1
}

$workerPy = Join-Path $RepoRoot "worker_video.py"
if (-not (Test-Path -LiteralPath $workerPy)) {
    Write-Err "worker_video.py not found in repository root."
    exit 1
}

$env:ACE_LOCAL_DEV_ENABLED = "true"
if (-not (Get-EnvString "ACE_SECURITY_ENABLED")) {
    $env:ACE_SECURITY_ENABLED = "false"
}
$env:REDIS_URL = $ExpectedRedisUrl
$env:ACE_PUBLIC_BASE_URL = (Get-EnvString "ACE_PUBLIC_BASE_URL")
if (-not $env:ACE_PUBLIC_BASE_URL) {
    $env:ACE_PUBLIC_BASE_URL = $DefaultPublicBase
}

if ($ValidateOnly) {
    Write-Ok "ValidateOnly: all checks passed. Worker was NOT started."
    exit 0
}

Write-Ok "Starting worker_video.py (local Redis + Backend at $env:ACE_PUBLIC_BASE_URL)"
Write-Info "Press Ctrl+C in this window to stop the Worker."
Write-Warn "Starting the Worker will consume jobs from the local queue only when you enqueue video jobs."
Write-Host ""

& $python $workerPy
