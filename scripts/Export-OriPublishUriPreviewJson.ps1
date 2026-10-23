#Requires -Version 5.1
<#
.SYNOPSIS
  Read-only Publish Uri Preview; writes a single JSON document to stdout (for local API).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SessionId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')
. (Join-Path $ScriptDir 'OriDevCancelDevLib.ps1')
. (Join-Path $ScriptDir 'OriPublishUriLib.ps1')

$layoutParams = @{ ScriptsDirectory = $ScriptDir }
if ($env:ACE_ORI_PS_BACKEND_ROOT) { $layoutParams.BackendRoot = $env:ACE_ORI_PS_BACKEND_ROOT }
if ($env:ACE_ORI_PS_FRONTEND_ROOT) { $layoutParams.FrontendRoot = $env:ACE_ORI_PS_FRONTEND_ROOT }
if ($env:ACE_ORI_PS_REGISTRY_PATH) { $layoutParams.RegistryPath = $env:ACE_ORI_PS_REGISTRY_PATH }
if ($env:ACE_ORI_PS_SESSIONS_ROOT) { $layoutParams.SessionsRoot = $env:ACE_ORI_PS_SESSIONS_ROOT }
$layout = Resolve-OriProjectLayout @layoutParams
$result = Invoke-OriPublishUriPreview -Layout $layout -SessionId $SessionId -SessionsRoot $layout.SessionsRoot
$payload = [ordered]@{
    ok   = $true
    mode = [string]$result.mode
    plan = $result.plan
}
$payload | ConvertTo-Json -Depth 28 -Compress
