#Requires -Version 5.1
<#
.SYNOPSIS
  Update localDevInfra backend + frontend commitSha in production-registry.json (local file only).

.DESCRIPTION
  Validates registry integrity, ancestry per repo, infra branch containment, and clean working trees.
  Supports -Preview (no file changes). Applies both SHAs in one atomic registry write.
  Does not change production deploy SHAs or WIP. Never logs secret values.
#>
[CmdletBinding()]
param(
    [switch]$Preview,
    [string]$RegistryPath = '',
    [string]$BackendRoot = '',
    [string]$FrontendRoot = '',
    [string]$NewBackendInfraCommit = '',
    [string]$NewFrontendInfraCommit = '',
    [string]$BackupRoot = '',
    [switch]$AllowDirtyWorkingTrees
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

$layout = Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir -BackendRoot $BackendRoot -RegistryPath $RegistryPath
if ($FrontendRoot) {
    $layout.FrontendRoot = Resolve-OriLiteralPath $FrontendRoot
}

$backendHead = ''
Push-Location -LiteralPath $layout.BackendRoot
try {
    git rev-parse HEAD 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $backendHead = (git rev-parse HEAD).Trim() }
}
finally {
    Pop-Location
}

$frontendHead = ''
Push-Location -LiteralPath $layout.FrontendRoot
try {
    git rev-parse HEAD 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $frontendHead = (git rev-parse HEAD).Trim() }
}
finally {
    Pop-Location
}

if ([string]::IsNullOrWhiteSpace($NewBackendInfraCommit)) { $NewBackendInfraCommit = $backendHead }
if ([string]::IsNullOrWhiteSpace($NewFrontendInfraCommit)) { $NewFrontendInfraCommit = $frontendHead }

if ([string]::IsNullOrWhiteSpace($NewBackendInfraCommit) -or [string]::IsNullOrWhiteSpace($NewFrontendInfraCommit)) {
    throw 'NEW_INFRA_COMMIT_SHA_REQUIRED'
}

$params = @{
    RegistryPath           = $layout.RegistryPath
    BackendRoot            = $layout.BackendRoot
    FrontendRoot           = $layout.FrontendRoot
    NewBackendInfraCommit  = $NewBackendInfraCommit
    NewFrontendInfraCommit = $NewFrontendInfraCommit
    Preview                = $Preview
}
if ($BackupRoot) { $params.BackupRoot = $BackupRoot }
if (-not $AllowDirtyWorkingTrees) { $params.RequireCleanWorkingTrees = $true }

$result = Update-OriProductionRegistryLocalDevInfraPair @params

if ($Preview) {
    Write-BaselineInfo 'Preview only — no registry files were modified.'
    $plan = $result.Plan
    Write-BaselineInfo "Registry: $($plan.RegistryPath)"
    Write-BaselineInfo "Current backend infra: $($plan.Current.backend)"
    Write-BaselineInfo "Current frontend infra: $($plan.Current.frontend)"
    Write-BaselineInfo "Target backend infra: $($plan.Target.backend)"
    Write-BaselineInfo "Target frontend infra: $($plan.Target.frontend)"
    Write-BaselineInfo "Will update: $($plan.WillUpdate)"
    Write-BaselineInfo "Can apply: $($plan.CanApply)"
    Write-BaselineInfo "Backup root (on apply): $($plan.BackupRoot)"
    Write-BaselineInfo ("Ancestry backend OK: " + $plan.Ancestry.backend.AncestryOk)
    Write-BaselineInfo ("Ancestry frontend OK: " + $plan.Ancestry.frontend.AncestryOk)
    if ($plan.Blockers.Count -gt 0) {
        Write-BaselineWarn 'Blockers:'
        foreach ($b in $plan.Blockers) { Write-BaselineWarn "  - $b" }
    }
}
else {
    if ($result.Updated) {
        Write-BaselineOk 'Registry localDevInfra pair updated (backend + frontend in one write).'
        Write-BaselineInfo "Registry SHA256: $($result.RegistryHash)"
        Write-BaselineInfo "Before backup: $($result.BeforeBackup)"
        Write-BaselineInfo "After backup: $($result.AfterBackup)"
    }
    else {
        Write-BaselineOk 'Registry already matches target infra SHAs; no write performed.'
        Write-BaselineInfo "Registry SHA256: $($result.RegistryHash)"
    }
}

return $result
