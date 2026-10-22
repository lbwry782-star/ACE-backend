#Requires -Version 5.1
<#
.SYNOPSIS
  Update localDevInfra.backend.commitSha in production-registry.json (local file only).

.DESCRIPTION
  Verifies the new commit is a descendant of the registry value, backs up registry,
  writes atomically, refreshes SHA256 sidecar. Does not change production deploy SHAs.
#>
[CmdletBinding()]
param(
    [string]$RegistryPath = '',
    [string]$BackendRoot = '',
    [string]$NewBackendInfraCommit = 'f6d59e9246c6409858e0b326f485f5088929b192'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

$layout = Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir -BackendRoot $BackendRoot -RegistryPath $RegistryPath
$result = Update-OriProductionRegistryBackendInfraCommit `
    -RegistryPath $layout.RegistryPath `
    -NewCommitSha $NewBackendInfraCommit `
    -BackendRoot $layout.BackendRoot

Write-BaselineOk "Registry backend infra commit: $($result.OldSha) -> $($result.NewSha) (updated=$($result.Updated))"
Write-BaselineInfo "Registry SHA256: $($result.RegistryHash)"
if ($result.BackupRoot) {
    Write-BaselineInfo "Backups: $($result.BackupRoot)"
}
return $result
