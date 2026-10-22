#Requires -Version 5.1
<#
.SYNOPSIS
  Start ACE Backend for local development (127.0.0.1 only).

.DESCRIPTION
  Loads .env.local, runs safety checks, and starts python app.py.
  Does not start Redis, Worker, or paid API calls.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

function Write-Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}
function Write-Ok([string]$Message) {
    Write-Host $Message -ForegroundColor Green
}
function Write-Warn([string]$Message) {
    Write-Host $Message -ForegroundColor Yellow
}
function Write-Err([string]$Message) {
    Write-Host $Message -ForegroundColor Red
}

function Get-EnvString([string]$Name) {
    $item = Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
    if (-not $item) {
        return ""
    }
    return ([string]$item.Value).Trim()
}

function Import-DotEnvFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    Get-Content -LiteralPath $Path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) {
            return
        }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim()
        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path "Env:$name" -Value $value
    }
    return $true
}

function Test-LoopbackHost([string]$HostName) {
    if ([string]::IsNullOrWhiteSpace($HostName)) {
        return $false
    }
    $h = $HostName.Trim().ToLowerInvariant()
    if ($h -in @("127.0.0.1", "localhost", "::1")) {
        return $true
    }
    if ($h.StartsWith("127.")) {
        return $true
    }
    return $false
}

function Get-RedisHostFromUrl([string]$Url) {
    if ([string]::IsNullOrWhiteSpace($Url)) {
        return ""
    }
    try {
        $uri = [Uri]$Url
        return $uri.Host
    }
    catch {
        return "__invalid__"
    }
}

function Test-ProductionPublicBaseUrl([string]$Url) {
    if ([string]::IsNullOrWhiteSpace($Url)) {
        return $false
    }
    try {
        $uri = [Uri]($(if ($Url -match "://") { $Url } else { "https://$Url" }))
        $pubHost = $uri.Host.ToLowerInvariant()
    }
    catch {
        return $true
    }
    $blocked = @(
        "ace-advertising.agency",
        "www.ace-advertising.agency",
        "onrender.com",
        "render.com"
    )
    foreach ($b in $blocked) {
        if ($pubHost -eq $b -or $pubHost.EndsWith(".$b")) {
            return $true
        }
    }
    return $false
}

function Test-BackendHealth([int]$Port) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
        return ($resp.StatusCode -eq 200 -and ($resp.Content -match "ok"))
    }
    catch {
        return $false
    }
}

function Test-TcpPortOpen([int]$Port) {
    try {
        $r = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
        return [bool]$r.TcpTestSucceeded
    }
    catch {
        return $false
    }
}

Write-Info "ACE Backend - local startup"
Write-Info "Project folder: $RepoRoot"

$envFile = Join-Path $RepoRoot ".env.local"
if (-not (Import-DotEnvFile $envFile)) {
    Write-Err "Missing file: .env.local"
    Write-Err "Copy .env.local.example to .env.local and set ACE_LOCAL_DEV_ENABLED=true."
    exit 1
}

if ((Get-EnvString "ACE_LOCAL_DEV_ENABLED").ToLowerInvariant() -notin @("true", "1", "yes", "on")) {
    Write-Err "ACE_LOCAL_DEV_ENABLED must be true in .env.local"
    exit 1
}

if ((Get-EnvString "ACE_SECURITY_ENABLED").ToLowerInvariant() -notin @("false", "0", "no", "off")) {
    Write-Warn "ACE_SECURITY_ENABLED is not false; local dev expects security off."
    Write-Warn "Set ACE_SECURITY_ENABLED=false in .env.local for Ori local mode."
}

$port = 5000
$portRaw = Get-EnvString "PORT"
if ($portRaw) {
    $parsedPort = 0
    if ([int]::TryParse($portRaw, [ref]$parsedPort)) {
        $port = $parsedPort
    }
}

$redisUrl = Get-EnvString "REDIS_URL"
if ($redisUrl) {
    $redisHost = Get-RedisHostFromUrl $redisUrl
    if ($redisHost -eq "__invalid__") {
        Write-Err "REDIS_URL is not a valid URL. Use loopback only, e.g. redis://127.0.0.1:6379/0"
        exit 1
    }
    if (-not (Test-LoopbackHost $redisHost)) {
        Write-Err "REDIS_URL must point to local Redis (127.0.0.1 / localhost), not production."
        Write-Err "Remove REDIS_URL from .env.local for Builder1-only, or fix the host."
        exit 1
    }
}

if (Test-ProductionPublicBaseUrl (Get-EnvString "ACE_PUBLIC_BASE_URL")) {
    Write-Err "ACE_PUBLIC_BASE_URL looks like production (Render / public site)."
    Write-Err "Remove it or set http://127.0.0.1:$port for local use later."
    exit 1
}

if ((Get-EnvString "BUILDER1_PRODUCTION_MODE").ToLowerInvariant() -in @("true", "1", "yes", "on")) {
    Write-Err "BUILDER1_PRODUCTION_MODE must stay off for local Ori dev."
    exit 1
}

if (Test-BackendHealth $port) {
    Write-Ok "Backend is already running: http://127.0.0.1:$port/health returns OK."
    Write-Ok "No second server was started (existing process left unchanged)."
    exit 0
}

if (Test-TcpPortOpen $port) {
    Write-Err "Port $port on 127.0.0.1 is in use, but /health did not respond."
    Write-Err "Another program may be using the port. This script will not stop it automatically."
    Write-Err "Close that program manually or choose another PORT in .env.local."
    exit 1
}

$python = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Err "Python not found. Install Python 3.8+ or create venv: python -m venv .venv312"
        exit 1
    }
    $python = $pythonCmd.Source
}

$appPy = Join-Path $RepoRoot "app.py"
if (-not (Test-Path -LiteralPath $appPy)) {
    Write-Err "app.py not found. Run this script from the ACE-Backend repository."
    exit 1
}

# Ensure local dev flags for this process (app.py binds 127.0.0.1 when ACE_LOCAL_DEV_ENABLED=true)
$env:ACE_LOCAL_DEV_ENABLED = "true"
if (-not $env:ACE_SECURITY_ENABLED) {
    $env:ACE_SECURITY_ENABLED = "false"
}
$env:PORT = "$port"

Write-Ok "Starting backend on http://127.0.0.1:$port (local dev, no Worker, no Redis required for Builder1)"
Write-Info "Press Ctrl+C in this window to stop the server."
Write-Host ""

& $python $appPy
