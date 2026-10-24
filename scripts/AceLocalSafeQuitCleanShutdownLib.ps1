# ACE LOCAL — safe quit CLEAN_SHUTDOWN_LEDGER (5.19.7; TEMP/isolated only).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitManagedStackStopLib.ps1')
. (Join-Path $ScriptDir 'AceLocalShutdownGateLib.ps1')

function Test-AceLocalRecoveryLedgerOpenIncident {
    param([Parameter(Mandatory = $true)]$Ledger)
    if (-not $Ledger) {
        return @{ Open = $false; Reason = 'ledger_missing' }
    }
    if ($Ledger.PSObject.Properties.Match('openIncident').Count -eq 0 -or -not $Ledger.openIncident) {
        return @{ Open = $false; Reason = 'none' }
    }
    $kind = ''
    if ($Ledger.openIncident.PSObject.Properties.Match('kind').Count -gt 0) {
        $kind = [string]$Ledger.openIncident.kind
    }
    if ([string]::IsNullOrWhiteSpace($kind)) {
        return @{ Open = $true; Reason = 'open_incident_untyped'; Kind = $kind }
    }
    return @{ Open = $true; Reason = 'open_incident'; Kind = $kind }
}

function Test-AceLocalSafeQuitCleanShutdownPreconditions {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)]$GateResult,
        [Parameter(Mandatory = $true)]$QuiesceResult,
        [Parameter(Mandatory = $true)]$CaptureResult,
        [Parameter(Mandatory = $true)]$VerificationResult,
        [Parameter(Mandatory = $true)]$StackStopResult
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        return @{ Ok = $false; Reason = 'ace_root_not_isolated' }
    }
    $session = ($LauncherSession -replace '\s', '').Trim()
    if (-not $session) {
        return @{ Ok = $false; Reason = 'launcher_session_empty' }
    }
    if (-not $GateResult -or -not $GateResult.Ok -or ([string]$GateResult.GateStatus -ne 'SAFE_TO_STOP')) {
        return @{ Ok = $false; Reason = 'gate_not_safe_to_stop' }
    }
    if (-not $QuiesceResult -or -not $QuiesceResult.Ok) {
        return @{ Ok = $false; Reason = 'quiesce_incomplete' }
    }
    if (-not $CaptureResult -or -not $CaptureResult.Ok) {
        return @{ Ok = $false; Reason = 'snapshot_capture_failed' }
    }
    $snapRoot = [string]$CaptureResult.SnapshotRoot
    $metaPath = Join-Path $snapRoot 'snapshot-metadata.json'
    if (-not (Test-Path -LiteralPath $metaPath)) {
        return @{ Ok = $false; Reason = 'snapshot_metadata_missing' }
    }
    $meta = Get-Content -LiteralPath $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$meta.status -ne 'VERIFIED') {
        return @{ Ok = $false; Reason = 'snapshot_not_verified' }
    }
    if (-not $VerificationResult -or -not $VerificationResult.Ok) {
        return @{ Ok = $false; Reason = 'coordinator_verification_failed' }
    }
    if (-not $StackStopResult -or -not $StackStopResult.Ok) {
        return @{ Ok = $false; Reason = 'managed_stack_stop_incomplete' }
    }
    $ptr = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $AceRoot
    if (-not $ptr) {
        return @{ Ok = $false; Reason = 'last_verified_pointer_missing' }
    }
    if ([string]$ptr.snapshotId -ne [string]$CaptureResult.SnapshotId) {
        return @{ Ok = $false; Reason = 'pointer_snapshot_mismatch' }
    }
    if ([string]$ptr.snapshotRoot -ne [string]$CaptureResult.SnapshotRoot) {
        return @{ Ok = $false; Reason = 'pointer_root_mismatch' }
    }
    $regSession = Read-AceLocalManagedProcessesLauncherSession -AceRoot $AceRoot
    if ($regSession -and ($regSession -ne $session)) {
        return @{ Ok = $false; Reason = 'registry_session_mismatch' }
    }
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
    if (-not $valid.Ok) {
        return @{ Ok = $false; Reason = ('ledger_invalid:' + $valid.Reason) }
    }
    if ([string]$ledger.launcherSession -ne $session) {
        return @{ Ok = $false; Reason = 'ledger_session_mismatch' }
    }
    $incident = Test-AceLocalRecoveryLedgerOpenIncident -Ledger $ledger
    if ($incident.Open) {
        return @{ Ok = $false; Reason = ('recovery_incident_open:' + $incident.Reason) }
    }
    if (Test-AceLocalRecoveryCleanShutdownRecorded -AceRoot $AceRoot -LauncherSession $session) {
        return @{ Ok = $false; Reason = 'clean_shutdown_already_recorded' }
    }
    return @{ Ok = $true; Reason = 'ok' }
}

function Invoke-AceLocalSafeQuitCleanShutdownLedgerRegister {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)]$GateResult,
        [Parameter(Mandatory = $true)]$QuiesceResult,
        [Parameter(Mandatory = $true)]$CaptureResult,
        [Parameter(Mandatory = $true)]$VerificationResult,
        [Parameter(Mandatory = $true)]$StackStopResult,
        [switch]$TestSimulateLedgerRegisterThrow
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        throw 'SAFE_QUIT_LEDGER_ACE_ROOT_NOT_ISOLATED'
    }
    $pre = Test-AceLocalSafeQuitCleanShutdownPreconditions -AceRoot $AceRoot -LauncherSession $LauncherSession `
        -GateResult $GateResult -QuiesceResult $QuiesceResult -CaptureResult $CaptureResult `
        -VerificationResult $VerificationResult -StackStopResult $StackStopResult
    if (-not $pre.Ok) {
        return @{
            Ok      = $false
            Outcome = 'preconditions_failed'
            Detail  = [string]$pre.Reason
        }
    }
    try {
        if ($TestSimulateLedgerRegisterThrow) {
            throw 'SAFE_QUIT_TEST_SIMULATED_LEDGER_REGISTER_FAILURE'
        }
        $ledger = Register-AceLocalRecoveryCleanShutdownComplete -AceRoot $AceRoot -LauncherSession $LauncherSession `
            -ShutdownVerified $true
        return @{
            Ok      = $true
            Outcome = 'recorded'
            Detail  = [string]$ledger.cleanShutdown.completedAt
        }
    }
    catch {
        return @{
            Ok      = $false
            Outcome = 'register_failed'
            Detail  = $_.Exception.Message
        }
    }
}
