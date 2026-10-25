#Requires -Version 5.1
<#
.SYNOPSIS
  Background recovery periodic save loop (phase 4).

.DESCRIPTION
  Background loop for recovery periodic save (also spawned from Start-AceLocal when coordinator allows).
  Controlled stop: Request-AceLocalRecoveryPeriodicSaveStop or recovery-periodic-save.stop file.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BackendRoot,
    [Parameter(Mandatory = $true)][string]$FrontendRoot,
    [string]$AceRoot = '',
    [string]$LauncherSession = '',
    [int]$TickIntervalSeconds = 30,
    [string]$CheckpointsRootOverride = '',
    [switch]$BackgroundMode,
    [string]$EngineBackendRoot = '',
    [int]$BackendPort = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')

$BackendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
if (-not $EngineBackendRoot) {
    $EngineBackendRoot = $BackendRoot
}
else {
    $EngineBackendRoot = (Resolve-Path -LiteralPath $EngineBackendRoot).ProviderPath
}
if (-not (Test-Path -LiteralPath (Join-Path $EngineBackendRoot 'engine\ace_local_dev_recovery_periodic_save_guard.py'))) {
    $EngineBackendRoot = Split-Path -Parent $ScriptDir
}
$FrontendRoot = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath
if (-not $AceRoot) {
    $AceRoot = Split-Path -Parent $BackendRoot
}
else {
    $AceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
}
if (-not $LauncherSession) {
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    if ($ledger -and (Test-AceLocalRecoveryLedgerValid -Ledger $ledger).Ok) {
        $LauncherSession = [string]$ledger.launcherSession
    }
    else {
        $LauncherSession = (Get-Date).ToString('yyyyMMdd-HHmmss')
        Initialize-AceLocalRecoveryLedger -AceRoot $AceRoot -LauncherSession $LauncherSession | Out-Null
    }
}

if ($BackendPort -gt 0 -and $LauncherSession) {
    [void](Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $AceRoot -BackendPort $BackendPort `
        -LauncherSession $LauncherSession)
    $env:PORT = [string]$BackendPort
}
Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $AceRoot
$spawnToken = [string]$env:ACE_RECOVERY_PERIODIC_SAVE_SPAWN_TOKEN
Start-AceLocalRecoveryPeriodicSaveLoop -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
    -AceRoot $AceRoot -LauncherSession $LauncherSession -TickIntervalSeconds $TickIntervalSeconds `
    -CheckpointsRootOverride $CheckpointsRootOverride -BackgroundMode:$BackgroundMode `
    -SpawnToken $spawnToken -GuardBackendRoot $EngineBackendRoot
