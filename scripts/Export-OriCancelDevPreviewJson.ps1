#Requires -Version 5.1
<#
.SYNOPSIS
  Read-only cancel-dev Preview; writes a single JSON document to stdout (for local API).
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

$layout = Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir
$result = Invoke-OriCancelDevPreview -Layout $layout -SessionId $SessionId -SessionsRoot $layout.SessionsRoot
$payload = [ordered]@{
    ok          = $true
    mode        = [string]$result.mode
    journalPath = [string]$result.journalPath
    plan        = $result.plan
}
$payload | ConvertTo-Json -Depth 24 -Compress
