# ACE LOCAL — recovery startup gate (phase 5.5: check + report only; no restore).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryStartupCoordinatorLib.ps1')

function Get-AceLocalRecoveryStartupGateScriptPath {
    $backendRoot = Split-Path -Parent $ScriptDir
    return Join-Path $backendRoot 'engine\ace_local_dev_recovery_startup_gate.py'
}

function Invoke-AceLocalRecoveryStartupGateCheck {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [switch]$ValidateOnly,
        [switch]$SkipLiveProcessInspection,
        [switch]$SkipCoordinatorLock
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $py = Get-AceLocalRecoveryStartupGateScriptPath
    if (-not (Test-Path -LiteralPath $py)) {
        throw 'ACE_RECOVERY_STARTUP_GATE_SCRIPT_MISSING'
    }
    $backendRoot = Split-Path -Parent $ScriptDir
    $code = @'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_recovery_startup_gate import evaluate_startup_recovery_gate
ace = Path(sys.argv[2])
sessions = Path(sys.argv[3]) if sys.argv[3] else None
inspect_live = sys.argv[4] == '1'
acquire_lock = sys.argv[5] == '1'
validate_only = sys.argv[6] == '1'
payload = evaluate_startup_recovery_gate(
    ace,
    sessions_root=sessions,
    inspect_live_processes=inspect_live,
    acquire_coordinator_lock=acquire_lock,
    validate_only=validate_only,
)
print(json.dumps(payload, ensure_ascii=False))
'@
    $sessArg = Join-Path $aceRoot '_backups\sessions'
    if (-not (Test-Path -LiteralPath $sessArg)) {
        $sessArg = ''
    }
    $inspectArg = if ($SkipLiveProcessInspection) { '0' } else { '1' }
    $lockArg = if ($SkipCoordinatorLock) { '0' } else { '1' }
    $validateArg = if ($ValidateOnly) { '1' } else { '0' }
    $raw = python -c $code $backendRoot $aceRoot $sessArg $inspectArg $lockArg $validateArg 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ("ACE_RECOVERY_STARTUP_GATE_FAILED:$raw")
    }
    return ($raw | ConvertFrom-Json)
}

function Assert-AceLocalRecoveryStartupGateForLaunch {
    param(
        [Parameter(Mandatory = $true)]$GateResult,
        [switch]$ValidateOnly
    )
    if ($null -eq $GateResult) {
        throw 'ACE_RECOVERY_STARTUP_GATE_RESULT_MISSING'
    }
    if ($GateResult.checkFailed) {
        $msg = 'Recovery startup gate check failed.'
        if ($GateResult.error) {
            $msg = "{0} ({1})" -f $msg, [string]$GateResult.error
        }
        throw $msg
    }
    if (-not $GateResult.continueLocalStackStart) {
        $outcome = [string]$GateResult.coordinatorOutcome
        throw ("Recovery startup blocked (coordinatorOutcome={0}). See recovery-startup-decision-latest.json." -f $outcome)
    }
    if ($ValidateOnly) {
        return
    }
}

function Write-AceLocalRecoveryStartupGateSummary {
    param([Parameter(Mandatory = $true)]$GateResult)
    $outcome = [string]$GateResult.coordinatorOutcome
    $decision = $GateResult.report.coordinatorDecision
    Write-Host ("Recovery gate: coordinatorOutcome={0} terminationType={1}" -f $outcome, [string]$decision.terminationType)
    if ($GateResult.reportPaths.latestPath) {
        Write-Host ("Recovery gate report: {0}" -f $GateResult.reportPaths.latestPath)
    }
}
