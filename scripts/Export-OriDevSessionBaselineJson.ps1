#Requires -Version 5.1
<#
.SYNOPSIS
  Validate or create dev session baseline; emits one JSON object to stdout (local API).
#>
[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [string]$SessionId = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

$layoutParams = @{ ScriptsDirectory = $ScriptDir }
if ($env:ACE_ORI_PS_BACKEND_ROOT) { $layoutParams.BackendRoot = $env:ACE_ORI_PS_BACKEND_ROOT }
if ($env:ACE_ORI_PS_FRONTEND_ROOT) { $layoutParams.FrontendRoot = $env:ACE_ORI_PS_FRONTEND_ROOT }
if ($env:ACE_ORI_PS_REGISTRY_PATH) { $layoutParams.RegistryPath = $env:ACE_ORI_PS_REGISTRY_PATH }
if ($env:ACE_ORI_PS_SESSIONS_ROOT) { $layoutParams.SessionsRoot = $env:ACE_ORI_PS_SESSIONS_ROOT }
$layout = Resolve-OriProjectLayout @layoutParams
$BackendRoot = $layout.BackendRoot
$FrontendRoot = $layout.FrontendRoot
$RegistryPath = $layout.RegistryPath
$SessionsRoot = $layout.SessionsRoot
$Sha256Sidecar = "$RegistryPath.sha256"

if ($ValidateOnly) {
    $regCheck = Test-ProductionRegistryIntegrity -RegistryPath $RegistryPath -Sha256SidecarPath $Sha256Sidecar
    if (-not $regCheck.Ok) {
        throw ($regCheck.Errors -join ',')
    }
    $commitCheck = Test-RegistryCommitsExist -Registry $regCheck.Registry -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
    if (-not $commitCheck.Ok) {
        throw ($commitCheck.Errors -join ',')
    }
    $payload = [ordered]@{
        ok           = $true
        mode         = 'VALIDATE'
        registryPath = $RegistryPath
        sessionsRoot = $SessionsRoot
    }
    $payload | ConvertTo-Json -Depth 6 -Compress
    exit 0
}

$baselineScript = Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1'
if (-not (Test-Path -LiteralPath $baselineScript)) {
    throw 'BASELINE_SCRIPT_MISSING'
}

$baselineArgs = @()
if (-not [string]::IsNullOrWhiteSpace($SessionId)) {
    $baselineArgs += @('-SessionId', $SessionId.Trim())
}
if ($env:ACE_ORI_PS_SESSIONS_ROOT) { $baselineArgs += @('-SessionsRoot', $env:ACE_ORI_PS_SESSIONS_ROOT) }
if ($env:ACE_ORI_PS_REGISTRY_PATH) { $baselineArgs += @('-RegistryPath', $env:ACE_ORI_PS_REGISTRY_PATH) }
if ($env:ACE_ORI_PS_BACKEND_ROOT) { $baselineArgs += @('-BackendRoot', $env:ACE_ORI_PS_BACKEND_ROOT) }
if ($env:ACE_ORI_PS_FRONTEND_ROOT) { $baselineArgs += @('-FrontendRoot', $env:ACE_ORI_PS_FRONTEND_ROOT) }

try {
    $summary = & $baselineScript @baselineArgs
}
catch {
    throw
}

if (-not $summary -or -not $summary.sessionId) {
    throw 'BASELINE_SUMMARY_MISSING'
}

$payload = [ordered]@{
    ok          = $true
    mode        = 'BASELINE'
    sessionId   = [string]$summary.sessionId
    status      = [string]$summary.status
    sessionRoot = [string]$summary.sessionRoot
    summary     = $summary
}
$payload | ConvertTo-Json -Depth 8 -Compress
