#Requires -Version 5.1
<#
.SYNOPSIS
  Capture a local Ori development session baseline (Git + ENV) outside the repos.

.DESCRIPTION
  Snapshot only — does not cancel dev, deploy, or modify source repos/ENV files.
  Session data is stored under D:\אס2\ACE\_backups\sessions\<session-id>\

.PARAMETER SessionId
  Optional fixed id (tests). Default: auto-generated. Existing folder => FAILED, no overwrite.

.PARAMETER SessionsRoot
  Root directory for session folders.

.PARAMETER RegistryPath
  Path to production-registry.json

.PARAMETER BackendRoot
  ACE-Backend repository root.

.PARAMETER FrontendRoot
  ACE-Frontend repository root.

.PARAMETER FailAfterGitBackend
  Test hook: simulate failure after backend git snapshot.
#>
[CmdletBinding()]
param(
    [string]$SessionId = '',
    [string]$SessionsRoot = 'D:\אס2\ACE\_backups\sessions',
    [string]$RegistryPath = 'D:\אס2\ACE\_registry\production-registry.json',
    [string]$BackendRoot = '',
    [string]$FrontendRoot = '',
    [switch]$FailAfterGitBackend
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

if ([string]::IsNullOrWhiteSpace($SessionsRoot) -or $SessionsRoot -match '\.\.') {
    $SessionsRoot = 'D:\אס2\ACE\_backups\sessions'
}
$SessionsRoot = [System.IO.Path]::GetFullPath($SessionsRoot)
$RegistryPath = [System.IO.Path]::GetFullPath($RegistryPath)
$Sha256Sidecar = "$RegistryPath.sha256"

if ([string]::IsNullOrWhiteSpace($BackendRoot)) {
    $BackendRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
}
if ([string]::IsNullOrWhiteSpace($FrontendRoot)) {
    $candidate = Join-Path (Split-Path -Parent $BackendRoot) 'ACE-Frontend'
    if (Test-Path -LiteralPath $candidate) {
        $FrontendRoot = (Resolve-Path -LiteralPath $candidate).Path
    }
    else {
        throw 'FrontendRoot not found. Pass -FrontendRoot explicitly.'
    }
}

if ([string]::IsNullOrWhiteSpace($SessionId)) {
    $SessionId = New-OriSessionId
}
$SessionRoot = Join-Path $SessionsRoot $SessionId
$verifyErrors = @()
$sessionDirOwned = $false
$sessionState = @{
    sessionId   = $SessionId
    status      = 'IN_PROGRESS'
    createdAt   = (Get-Date).ToString('o')
    readyAt     = $null
    failedAt    = $null
    errors      = @()
    purpose     = 'Ori dev session baseline for future cancel-dev; no deploy/rollback.'
    boundaries  = @{
        productionVerifiedInRegistry = $true
        localDevInfraNotProduction   = $true
        productWipTrackedSeparately  = $true
        envInfraTemplateVsSession    = $true
        unrelatedEnvChangesRequireManualReview = $true
    }
    paths = @{
        sessionsRoot  = $SessionsRoot
        sessionRoot   = $SessionRoot
        registryPath  = $RegistryPath
        backendRoot   = $BackendRoot
        frontendRoot  = $FrontendRoot
    }
}

function Mark-Failed {
    param([string[]]$Reasons)
    $script:sessionState.status = 'FAILED'
    $script:sessionState.failedAt = (Get-Date).ToString('o')
    $script:sessionState.errors = @($script:sessionState.errors + $Reasons) | Select-Object -Unique
    if (Test-Path -LiteralPath $SessionRoot) {
        Save-SessionState -SessionRoot $SessionRoot -State $sessionState
    }
}

try {
    if (Test-Path -LiteralPath $SessionRoot) {
        throw "SESSION_DIR_EXISTS:$SessionId"
    }
    New-Item -ItemType Directory -Path $SessionRoot -Force | Out-Null
    $sessionDirOwned = $true
    Save-SessionState -SessionRoot $SessionRoot -State $sessionState

    $regCheck = Test-ProductionRegistryIntegrity -RegistryPath $RegistryPath -Sha256SidecarPath $Sha256Sidecar
    if (-not $regCheck.Ok) {
        throw ($regCheck.Errors -join ',')
    }
    $registry = $regCheck.Registry

    $regDest = Join-Path $SessionRoot 'registry'
    New-Item -ItemType Directory -Path $regDest -Force | Out-Null
    Copy-Item -LiteralPath $RegistryPath -Destination (Join-Path $regDest 'production-registry.json') -Force
    Copy-Item -LiteralPath $Sha256Sidecar -Destination (Join-Path $regDest 'production-registry.json.sha256') -Force

    $commitCheck = Test-RegistryCommitsExist -Registry $registry -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
    if (-not $commitCheck.Ok) {
        throw ($commitCheck.Errors -join ',')
    }

    $backendSnap = Export-GitWorkingTreeSnapshot -RepoRoot $BackendRoot -Label 'BACKEND' -SessionRoot $SessionRoot
    if ($FailAfterGitBackend) {
        throw 'SIMULATED_FAIL_AFTER_GIT_BACKEND'
    }
    $frontendSnap = Export-GitWorkingTreeSnapshot -RepoRoot $FrontendRoot -Label 'FRONTEND' -SessionRoot $SessionRoot

    $envDir = Join-Path $SessionRoot 'ENV'
    New-Item -ItemType Directory -Path $envDir -Force | Out-Null

    $backendEnvSrc = Join-Path $BackendRoot '.env.local'
    $frontendEnvSrc = Join-Path $FrontendRoot '.env.development.local'
    $backendTemplate = @(Get-EnvKeyTemplateFromExample -ExamplePath (Join-Path $BackendRoot '.env.local.example'))
    if ($null -eq $backendTemplate) { $backendTemplate = @() }
    $frontendTemplate = @('VITE_BACKEND_URL')

    $envSnapshots = @(
        (Export-EnvFileSnapshot -SourcePath $backendEnvSrc -DestDpapiPath (Join-Path $envDir 'backend.env.local.dpapi') -TemplateKeys $backendTemplate -LogicalName 'backend.env.local')
        (Export-EnvFileSnapshot -SourcePath $frontendEnvSrc -DestDpapiPath (Join-Path $envDir 'frontend.env.development.local.dpapi') -TemplateKeys $frontendTemplate -LogicalName 'frontend.env.development.local')
    )
    Set-DirectoryCurrentUserAcl -Path $envDir

    $envKeyRows = @()
    foreach ($snap in $envSnapshots) {
        $envKeyRows += $snap.KeyManifest
    }
    $envKeyRows | Export-Csv -LiteralPath (Join-Path $envDir 'env-keys-manifest.csv') -NoTypeInformation -Encoding UTF8

    $sessionState.production = @{
        backendWebSha    = [string]$registry.production.backendWeb.deployedCommitSha
        backendWorkerSha = [string]$registry.production.backendWorker.deployedCommitSha
        frontendSha      = [string]$registry.production.frontend.deployedCommitSha
        workflowRunId    = [string]$registry.production.frontend.workflowRunId
    }
    $sessionState.localDevInfra = @{
        backendSha  = [string]$registry.localDevInfra.backend.commitSha
        frontendSha = [string]$registry.localDevInfra.frontend.commitSha
        backendBranchExpected = [string]$registry.localDevInfra.backend.branch
        frontendBranchExpected = [string]$registry.localDevInfra.frontend.branch
    }
    $sessionState.productWip = @{
        builder1Branch = [string]$registry.wip.builder1.branch
        builder1Sha    = [string]$registry.wip.builder1.commitSha
    }
    $sessionState.git = @{
        backend  = $backendSnap
        frontend = $frontendSnap
    }
    $sessionState.envSnapshots = $envSnapshots

    $manifestRows = @()
    foreach ($block in @(
        @{ Snap = $backendSnap; Label = 'BACKEND'; Root = $BackendRoot }
        @{ Snap = $frontendSnap; Label = 'FRONTEND'; Root = $FrontendRoot }
    )) {
        foreach ($e in $block.Snap.Entries) {
            $full = Join-Path $block.Root $e.RelPath
            if ($e.ChangeType -eq 'deleted') {
                $bkp = Join-Path $SessionRoot "$($block.Label)\deleted-from-HEAD\$($e.RelPath)"
                $manifestRows += [PSCustomObject]@{
                    Area = $block.Label; RelPath = $e.RelPath; ChangeType = 'deleted'
                    SourceSha256 = ''; BackupSha256 = (Get-FileSha256Hex $bkp); SizeBytes = 0
                }
            }
            else {
                $bkp = Join-Path $SessionRoot "$($block.Label)\mirror\$($e.RelPath)"
                $manifestRows += [PSCustomObject]@{
                    Area = $block.Label; RelPath = $e.RelPath; ChangeType = $e.ChangeType
                    SourceSha256 = (Get-FileSha256Hex $full); BackupSha256 = (Get-FileSha256Hex $bkp); SizeBytes = (Get-Item -LiteralPath $full).Length
                }
            }
        }
    }
    foreach ($snap in $envSnapshots) {
        $manifestRows += [PSCustomObject]@{
            Area = 'ENV'; RelPath = $snap.LogicalName; ChangeType = $(if ($snap.Present) { 'env-dpapi' } else { 'env-missing' })
            SourceSha256 = $(if ($snap.Present) { $snap.PlainSha256 } else { '' })
            BackupSha256 = $(if ($snap.Present) { (Get-FileSha256Hex $snap.DestDpapiPath) } else { '' })
            SizeBytes = $snap.SizeBytes
        }
    }
    Write-SessionManifest -SessionRoot $SessionRoot -Rows $manifestRows

    $integrity = Test-SessionSnapshotIntegrity -SessionRoot $SessionRoot -SessionMeta $sessionState
    if (-not $integrity.Ok) {
        throw ($integrity.Errors -join ',')
    }

    $sessionState.status = 'READY'
    $sessionState.readyAt = (Get-Date).ToString('o')
    Save-SessionState -SessionRoot $SessionRoot -State $sessionState

    $summary = [ordered]@{
        sessionId = $SessionId
        status = 'READY'
        sessionRoot = $SessionRoot
        backendBranch = $backendSnap.Branch
        backendHead = $backendSnap.Head
        frontendBranch = $frontendSnap.Branch
        frontendHead = $frontendSnap.Head
        backendDirtyCount = $backendSnap.Entries.Count
        frontendDirtyCount = $frontendSnap.Entries.Count
        envBackendPresent = $envSnapshots[0].Present
        envFrontendPresent = $envSnapshots[1].Present
        envBackendKeyCount = @($envSnapshots[0].KeyManifest | Where-Object { $_.valueSha256 }).Count
        envFrontendKeyCount = @($envSnapshots[1].KeyManifest | Where-Object { $_.valueSha256 }).Count
    }
    ($summary | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath (Join-Path $SessionRoot 'VERIFY_SUMMARY.json') -Encoding UTF8

    Write-BaselineOk "Session baseline READY: $SessionId"
    Write-BaselineInfo "Location: $SessionRoot"
    return $summary
}
catch {
    $reason = $_.Exception.Message
    if ($sessionDirOwned -and $reason -notmatch '^SESSION_DIR_EXISTS:') {
        Mark-Failed -Reasons @($reason)
    }
    Write-BaselineErr "Session baseline FAILED: $SessionId"
    Write-BaselineErr $reason
    throw
}
