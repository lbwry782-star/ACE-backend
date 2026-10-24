# ACE LOCAL — recovery startup coordinator (phase 5; not wired to Start-AceLocal).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-AceLocalRecoveryStartupCoordinatorScriptPath {
    $backendRoot = Split-Path -Parent $ScriptDir
    return Join-Path $backendRoot 'engine\ace_local_dev_recovery_startup_coordinator.py'
}

function Invoke-AceLocalRecoveryStartupCoordination {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$SessionsRoot = '',
        [switch]$SkipLiveProcessInspection,
        [switch]$SkipCoordinatorLock
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $py = Get-AceLocalRecoveryStartupCoordinatorScriptPath
    if (-not (Test-Path -LiteralPath $py)) {
        throw 'ACE_RECOVERY_STARTUP_COORDINATOR_SCRIPT_MISSING'
    }
    $backendRoot = Split-Path -Parent $ScriptDir
    $code = @'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_recovery_startup_coordinator import (
    evaluate_recovery_startup_coordination,
    coordinator_decision_to_dict,
)
ace = Path(sys.argv[2])
sessions = Path(sys.argv[3]) if sys.argv[3] else None
inspect_live = sys.argv[4] == '1'
acquire_lock = sys.argv[5] == '1'
decision = evaluate_recovery_startup_coordination(
    ace,
    sessions_root=sessions,
    inspect_live_processes=inspect_live,
    acquire_coordinator_lock=acquire_lock,
)
print(json.dumps(coordinator_decision_to_dict(decision), ensure_ascii=False))
'@
    $sessArg = ''
    if ($SessionsRoot) {
        $sessArg = (Resolve-Path -LiteralPath $SessionsRoot).ProviderPath
    }
    $inspectArg = if ($SkipLiveProcessInspection) { '0' } else { '1' }
    $lockArg = if ($SkipCoordinatorLock) { '0' } else { '1' }
    $raw = python -c $code $backendRoot $aceRoot $sessArg $inspectArg $lockArg 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ("ACE_RECOVERY_STARTUP_COORDINATION_FAILED:$raw")
    }
    return ($raw | ConvertFrom-Json)
}

function Get-AceLocalRecoveryPeriodicSaveActivationContract {
    param(
        [Parameter(Mandatory = $true)]$CoordinatorDecision,
        [Parameter(Mandatory = $true)][string]$LauncherSession
    )
    if ($null -eq $CoordinatorDecision.periodicSaveContract) {
        throw 'ACE_RECOVERY_COORDINATOR_DECISION_MISSING_CONTRACT'
    }
    $contract = [ordered]@{}
    foreach ($prop in $CoordinatorDecision.periodicSaveContract.PSObject.Properties) {
        $contract[$prop.Name] = $prop.Value
    }
    $contract['launcherSession'] = $LauncherSession
    $allowed = ($CoordinatorDecision.coordinatorOutcome -eq 'START_ALLOWED') -and ($LauncherSession.Trim().Length -gt 0)
    $contract['allowed'] = [bool]$allowed
    return [pscustomobject]$contract
}
