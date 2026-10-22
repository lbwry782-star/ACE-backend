#Requires -Version 5.1
<#
.SYNOPSIS
  Preview or execute local Ori "cancel development" for a READY baseline session.

.DESCRIPTION
  Default is Preview only (no repo/ENV mutations). Execute requires -Execute and -ConfirmExecute CANCEL.
  Does not push, deploy, or touch Redis. Infra target commits come from the live production registry.

.PARAMETER SessionId
  READY session folder name under _backups\sessions.

.PARAMETER Execute
  Perform cancel (blocked on real ACE layout unless -AllowRealExecute is also set).

.PARAMETER ConfirmExecute
  Must be exactly CANCEL when -Execute is set.

.PARAMETER AllowRealExecute
  Safety gate: execute mode refuses the verified local ACE tree unless this switch is set.

.PARAMETER RestartServices
  Optional service restart hint after successful execute (requires -ConfirmRestart).

.PARAMETER ConfirmRestart
  Must be RESTART when -RestartServices is set.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SessionId,
    [string]$SessionsRoot = '',
    [string]$RegistryPath = '',
    [string]$BackendRoot = '',
    [string]$FrontendRoot = '',
    [switch]$Execute,
    [string]$ConfirmExecute = '',
    [switch]$AllowRealExecute,
    [switch]$RestartServices,
    [string]$ConfirmRestart = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')
. (Join-Path $ScriptDir 'OriDevCancelDevLib.ps1')

$layout = Resolve-OriProjectLayout `
    -ScriptsDirectory $ScriptDir `
    -BackendRoot $BackendRoot `
    -FrontendRoot $FrontendRoot `
    -RegistryPath $RegistryPath `
    -SessionsRoot $SessionsRoot

if ([string]::IsNullOrWhiteSpace($SessionsRoot)) {
    $SessionsRoot = $layout.SessionsRoot
}

$envCheck = Test-OriVerifiedLocalDevEnvironment -Layout $layout
if (-not $envCheck.Ok) {
    throw ($envCheck.Errors -join ',')
}

if (-not $Execute) {
    $result = Invoke-OriCancelDevPreview -Layout $layout -SessionId $SessionId -SessionsRoot $SessionsRoot
    Write-CancelOk 'Preview complete (no changes applied).'
    Write-CancelInfo "Journal: $($result.journalPath)"
    if (@($result.plan.registryInfraDrift).Count -gt 0) {
        Write-CancelWarn ('Registry/infra drift: ' + ($result.plan.registryInfraDrift -join ', '))
    }
    if (@($result.plan.blockers).Count -gt 0) {
        Write-CancelWarn ('Blockers: ' + ($result.plan.blockers -join ', '))
    }
    if ($result.plan.cancelCycle -and [string]$result.plan.cancelCycle.status -eq 'CANCELLED') {
        Write-CancelWarn ('Cancel cycle already closed at ' + $result.plan.cancelCycle.cancelledAt)
    }
    if (@($result.plan.infraAdvancementPaths.backend).Count -gt 0) {
        Write-CancelInfo ('Infra advancement paths (backend, preserved): ' + ($result.plan.infraAdvancementPaths.backend -join ', '))
    }
    if (@($result.plan.infraAdvancementPaths.frontend).Count -gt 0) {
        Write-CancelInfo ('Infra advancement paths (frontend, preserved): ' + ($result.plan.infraAdvancementPaths.frontend -join ', '))
    }
    Write-CancelInfo 'Commit graph (backend):'
    Write-Host "  baseline product HEAD $($result.plan.commitGraph.backend.BaselineProductHead)"
    Write-Host "  current HEAD $($result.plan.commitGraph.backend.CurrentHead) ($($result.plan.commitGraph.backend.CurrentBranch))"
    Write-Host "  infra target $($result.plan.commitGraph.backend.InfraTarget)"
    Write-CancelInfo 'Commits that would leave the active branch (product session):'
    foreach ($c in $result.plan.commitsToCancel.backend) {
        Write-Host "  BACKEND $($c.shortSha) [$($c.classification)] $($c.subject) branches=$($c.branches -join ',') wipRef=$($c.preservedInWipRef) backup=$($c.preservedInBackup)"
    }
    foreach ($c in $result.plan.commitsToCancel.frontend) {
        Write-Host "  FRONTEND $($c.shortSha) [$($c.classification)] $($c.subject) branches=$($c.branches -join ',') wipRef=$($c.preservedInWipRef) backup=$($c.preservedInBackup)"
    }
    if (@($result.plan.commitGraph.backend.PreBaselineCommits).Count -gt 0) {
        Write-CancelWarn 'Pre-baseline commits (require explicit cycle scope to cancel):'
        foreach ($c in $result.plan.commitGraph.backend.PreBaselineCommits) {
            Write-Host "  BACKEND $($c.shortSha) $($c.subject) explicit=$($c.explicitlyIncluded)"
        }
    }
    Write-CancelInfo 'Changes that would be canceled (git paths):'
    foreach ($p in $result.plan.git.backend.PathsToRevert) { Write-Host "  BACKEND $p" }
    foreach ($p in $result.plan.git.frontend.PathsToRevert) { Write-Host "  FRONTEND $p" }
    Write-CancelInfo 'Infra preserved (registry targets):'
    Write-Host "  Backend  $($result.plan.infraTargets.backendBranch) @ $($result.plan.infraTargets.backendCommit)"
    Write-Host "  Frontend $($result.plan.infraTargets.frontendBranch) @ $($result.plan.infraTargets.frontendCommit)"
    return $result
}

if ($ConfirmExecute -ne 'CANCEL') {
    throw 'EXECUTE_REQUIRES_CONFIRM_EXECUTE_Cancel'
}

$realAce = $false
try {
    $realAce = ($layout.AceRoot -eq (Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir).AceRoot) `
        -and (Test-Path -LiteralPath (Join-Path $layout.SessionsRoot 'ori-20261023-010342-2d952e88'))
}
catch {
    $realAce = $false
}
if ($realAce -and -not $AllowRealExecute) {
    throw 'REAL_EXECUTE_BLOCKED:Use isolated tests or pass -AllowRealExecute after explicit user approval.'
}

try {
    $execResult = Invoke-OriCancelDevExecute -Layout $layout -SessionId $SessionId -SessionsRoot $SessionsRoot
    Write-CancelOk "Execute finished: $($execResult.status)"
    Write-CancelInfo "Backup: $($execResult.backupRoot)"
    Write-CancelInfo "Journal: $($execResult.journalPath)"

    if ($RestartServices) {
        if ($ConfirmRestart -ne 'RESTART') {
            throw 'RESTART_REQUIRES_CONFIRM_RESTART'
        }
        Invoke-OriCancelDevServiceRestart -Confirmed | Out-Null
    }
    return $execResult
}
catch {
    Write-CancelErr $_.Exception.Message
    $journals = @(Get-ChildItem -LiteralPath (Join-Path $SessionsRoot $SessionId) -Filter 'cancel-journal-execute-*.json' -ErrorAction SilentlyContinue | Sort-Object Name -Descending)
    if ($journals.Count -gt 0) {
        $j = Get-Content -LiteralPath $journals[0].FullName -Raw | ConvertFrom-Json
        if ($j.recovery) {
            Write-CancelWarn 'Partial/failed cancel — manual recovery only (no automatic reset).'
            Write-CancelInfo "Backup: $($j.recovery.backupRoot)"
            Write-CancelInfo "Guide: $($j.recovery.recoveryMarkdown)"
        }
    }
    throw
}
