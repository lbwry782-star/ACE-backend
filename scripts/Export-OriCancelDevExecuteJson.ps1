#Requires -Version 5.1
<#
.SYNOPSIS
  Execute cancel-dev for a READY session; emits one JSON document to stdout (local API).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SessionId,
    [string]$ConfirmExecute = 'CANCEL',
    [switch]$AllowRealExecute
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

$invokeScript = Join-Path $ScriptDir 'Invoke-OriCancelDev.ps1'
if (-not (Test-Path -LiteralPath $invokeScript)) {
    throw 'INVOKE_CANCEL_SCRIPT_MISSING'
}

try {
    $execResult = & $invokeScript `
        -SessionId $SessionId.Trim() `
        -SessionsRoot $layout.SessionsRoot `
        -RegistryPath $layout.RegistryPath `
        -BackendRoot $layout.BackendRoot `
        -FrontendRoot $layout.FrontendRoot `
        -Execute `
        -ConfirmExecute $ConfirmExecute `
        -AllowRealExecute:$AllowRealExecute.IsPresent

    $payload = [ordered]@{
        ok     = $true
        mode   = 'EXECUTE'
        result = $execResult
    }
    $payload | ConvertTo-Json -Depth 16 -Compress
}
catch {
    $msg = $_.Exception.Message
    $recovery = $null
    try {
        $layout = Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir
        $sessionRoot = Join-Path $layout.SessionsRoot $SessionId.Trim()
        $journals = @(Get-ChildItem -LiteralPath $sessionRoot -Filter 'cancel-journal-execute-*.json' -ErrorAction SilentlyContinue | Sort-Object Name -Descending)
        if ($journals.Count -gt 0) {
            $j = Get-Content -LiteralPath $journals[0].FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($j.recovery) {
                $recovery = [ordered]@{
                    backupRoot      = [string]$j.recovery.backupRoot
                    recoveryMarkdown = [string]$j.recovery.recoveryMarkdown
                }
            }
        }
    }
    catch {
        $recovery = $null
    }

    $errPayload = [ordered]@{
        ok       = $false
        mode     = 'EXECUTE'
        error    = $msg
        recovery = $recovery
    }
    $errPayload | ConvertTo-Json -Depth 8 -Compress
    exit 1
}
