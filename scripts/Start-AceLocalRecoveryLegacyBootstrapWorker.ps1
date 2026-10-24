# Background worker — legacy recovery bootstrap full checkpoint (phase 5.13).
#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$AceRoot,
    [Parameter(Mandatory = $true)][string]$BackendRoot,
    [Parameter(Mandatory = $true)][string]$FrontendRoot,
    [string]$CheckpointsRootOverride = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryLegacyBootstrapLib.ps1')
$result = Invoke-AceLocalRecoveryLegacyBootstrapPipeline -AceRoot $AceRoot -BackendRoot $BackendRoot `
    -FrontendRoot $FrontendRoot -CheckpointsRootOverride $CheckpointsRootOverride
$logDir = Join-Path $AceRoot '_backups\local-dev\logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir 'recovery-legacy-bootstrap-worker.json'
$periodic = $null
if ($result.Ok) {
    $periodic = Invoke-AceLocalRecoveryPeriodicSaveAfterLegacyBootstrap -AceRoot $AceRoot `
        -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -CheckpointsRootOverride $CheckpointsRootOverride
    $result = [ordered]@{
        Ok           = $result.Ok
        Phase        = $result.Phase
        CheckpointId = $result.CheckpointId
        PeriodicSave = $periodic
    }
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $logPath -Encoding UTF8
if (-not $result.Ok) { exit 1 }
if ($periodic -and -not $periodic.Started) { exit 2 }
exit 0
