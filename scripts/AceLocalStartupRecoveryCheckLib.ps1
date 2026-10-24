# ACE LOCAL — mandatory previous-session startup check (phase 2; not wired to Start-AceLocal).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')

function Get-AceLocalStartupRecoveryCheckScriptPath {
    $backendRoot = Split-Path -Parent $ScriptDir
    return Join-Path $backendRoot 'engine\ace_local_dev_startup_recovery_check.py'
}

function Invoke-AceLocalPreviousStartupRecoveryCheck {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$SessionsRoot = '',
        [switch]$SkipLiveProcessInspection
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $py = Get-AceLocalStartupRecoveryCheckScriptPath
    if (-not (Test-Path -LiteralPath $py)) {
        throw 'ACE_STARTUP_RECOVERY_CHECK_SCRIPT_MISSING'
    }
    $backendRoot = Split-Path -Parent $ScriptDir
    $code = @'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_startup_recovery_check import (
    evaluate_previous_local_startup,
    startup_recovery_decision_to_dict,
)
ace = Path(sys.argv[2])
sessions = Path(sys.argv[3]) if sys.argv[3] else None
inspect_live = sys.argv[4] == '1'
decision = evaluate_previous_local_startup(
    ace,
    sessions_root=sessions,
    inspect_live_processes=inspect_live,
)
print(json.dumps(startup_recovery_decision_to_dict(decision), ensure_ascii=False))
'@
    $sessArg = ''
    if ($SessionsRoot) {
        $sessArg = (Resolve-Path -LiteralPath $SessionsRoot).ProviderPath
    }
    $inspectArg = if ($SkipLiveProcessInspection) { '0' } else { '1' }
    $raw = python -c $code $backendRoot $aceRoot $sessArg $inspectArg 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ("ACE_STARTUP_RECOVERY_CHECK_FAILED:$raw")
    }
    return ($raw | ConvertFrom-Json)
}

function Start-AceLocalRecoverySessionAfterStartupCheck {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [string]$SessionsRoot = ''
    )
    $decision = Invoke-AceLocalPreviousStartupRecoveryCheck -AceRoot $AceRoot -SessionsRoot $SessionsRoot
    if (-not $decision.devOperationsAllowed) {
        throw ("ACE_STARTUP_RECOVERY_BLOCKED:$($decision.outcome)")
    }
    return Initialize-AceLocalRecoveryLedger -AceRoot $AceRoot -LauncherSession $LauncherSession
}
