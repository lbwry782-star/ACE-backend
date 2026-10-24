#Requires -Version 5.1
<#
.SYNOPSIS
  Coordinator for LOCAL safe quit (5.19.7 — isolated clean-shutdown ledger; DryRun skeleton).

.PARAMETER DryRun
  Declarative phases only; no gate files or side effects.

.PARAMETER IsolatedGateTest
  Runs SHUTDOWN_GATE only under a TEMP ace root (see Test-AceLocalSafeQuitAceRootIsolated).

.PARAMETER IsolatedQuiesceTest
  TEMP only: PREFLIGHT → SHUTDOWN_GATE → PERIODIC_SAVE_QUIESCE (no snapshot/stop/ledger).

.PARAMETER IsolatedSnapshotTest
  TEMP only: through SNAPSHOT_VERIFICATION; no stack stop or clean-shutdown ledger.

.PARAMETER IsolatedStackStopTest
  TEMP only: through MANAGED_STACK_STOP (test-owned processes only); no clean-shutdown ledger.

.PARAMETER IsolatedCleanShutdownTest
  TEMP only: full pipeline through CLEAN_SHUTDOWN_LEDGER (no live LOCAL).

.PARAMETER GateWaitSeconds
  Max wait for gate to reach SAFE_TO_STOP or SHUTDOWN_BLOCKED during isolated gate test.

.PARAMETER PeriodicSaveQuiesceWaitSeconds
  Max wait for an in-flight checkpoint lock to clear during isolated quiesce test.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$AceRoot,
    [switch]$DryRun,
    [switch]$IsolatedGateTest,
    [switch]$IsolatedQuiesceTest,
    [switch]$IsolatedSnapshotTest,
    [switch]$IsolatedStackStopTest,
    [switch]$IsolatedCleanShutdownTest,
    [string]$LauncherSession = '',
    [int]$GateWaitSeconds = 30,
    [int]$PeriodicSaveQuiesceWaitSeconds = 30,
    [switch]$TestSimulateThrowAfterLock,
    [switch]$TestSimulateGateThrow,
    [switch]$TestSimulateQuiesceWaitThrow,
    [switch]$TestSimulateSnapshotFailAfterBackend,
    [switch]$TestSimulateSnapshotConcurrentDrift,
    [switch]$TestSimulateSnapshotVerificationFailure,
    [switch]$TestSimulateStackStopThrow,
    [switch]$TestSimulatePartialStackStop,
    [switch]$TestSimulateLedgerRegisterThrow,
    [int]$TestHoldLockMilliseconds = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalShutdownGateLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitPeriodicSaveLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitManagedStackStopLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitCleanShutdownLib.ps1')

$script:AceLocalSafeQuitMutexName = 'Global\ACE-LOCAL-SAFE-QUIT'

function Get-AceLocalSafeQuitPhaseDefinitions {
    return @(
        @{ Id = 'PREFLIGHT'; Description = 'Validate coordinator preconditions.' }
        @{ Id = 'SHUTDOWN_GATE'; Description = 'Request shutdown gate; observe; refresh until terminal or timeout.' }
        @{ Id = 'WAIT_FOR_ACTIVE_OPERATIONS'; Description = 'Reserved; folded into SHUTDOWN_GATE refresh loop in 5.19.2.' }
        @{ Id = 'PERIODIC_SAVE_QUIESCE'; Description = 'Stop-request; block new checkpoints; wait for live lock to clear.' }
        @{ Id = 'SHUTDOWN_SNAPSHOT'; Description = 'Capture shutdown snapshot after gate and quiesce.' }
        @{ Id = 'SNAPSHOT_VERIFICATION'; Description = 'Re-verify snapshot integrity and VERIFIED status.' }
        @{ Id = 'MANAGED_STACK_STOP'; Description = 'Stop managed stack (isolated test driver in 5.19.6).' }
        @{ Id = 'CLEAN_SHUTDOWN_LEDGER'; Description = 'Record verified clean shutdown in recovery ledger.' }
        @{ Id = 'COMPLETE'; Description = 'Future: finalize quit coordinator.' }
    )
}

function New-AceLocalSafeQuitPhaseResult {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$ExecutionStatus,
        [string]$Detail = ''
    )
    return [ordered]@{
        id              = $Id
        executionStatus = $ExecutionStatus
        detail          = $Detail
    }
}

function Enter-AceLocalSafeQuitMutex {
    $mutex = New-Object System.Threading.Mutex($false, $script:AceLocalSafeQuitMutexName)
    $owns = $false
    try {
        $owns = $mutex.WaitOne(0, $false)
    }
    catch {
        return @{
            Acquired = $false
            Owns     = $false
            Mutex    = $mutex
            Reason   = 'mutex_wait_failed:' + $_.Exception.Message
        }
    }
    if (-not $owns) {
        return @{
            Acquired = $false
            Owns     = $false
            Mutex    = $mutex
            Reason   = 'quit_already_in_progress'
        }
    }
    return @{
        Acquired = $true
        Owns     = $true
        Mutex    = $mutex
        Reason   = 'ok'
    }
}

function Exit-AceLocalSafeQuitMutex {
    param(
        [Parameter(Mandatory = $true)]$MutexHandle,
        [Parameter(Mandatory = $true)][bool]$Owns
    )
    if (-not $Owns -or -not $MutexHandle) { return }
    try { [void]$MutexHandle.ReleaseMutex() } catch { Write-Warn ('Safe quit mutex release: ' + $_.Exception.Message) }
    try { $MutexHandle.Dispose() } catch { }
}

function Invoke-AceLocalSafeQuitDryRunReport {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedAceRoot,
        [string]$Session,
        [switch]$SimulateThrowAfterLock,
        [int]$HoldLockMilliseconds = 0
    )
    $phaseResults = @()
    foreach ($def in Get-AceLocalSafeQuitPhaseDefinitions) {
        $phaseResults += New-AceLocalSafeQuitPhaseResult -Id $def.Id -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'DryRun — no side effects.'
    }
    if ($SimulateThrowAfterLock) {
        throw 'SAFE_QUIT_TEST_SIMULATED_FAILURE_AFTER_LOCK'
    }
    if ($HoldLockMilliseconds -gt 0) {
        Start-Sleep -Milliseconds $HoldLockMilliseconds
    }
    return [ordered]@{
        schemaVersion    = 1
        coordinatorPhase = '5.19.2-dryrun'
        aceRoot          = $ResolvedAceRoot
        launcherSession  = $Session
        mode             = 'DryRun'
        succeeded        = $true
        phases           = @($phaseResults)
        disclaimer       = 'DryRun is not proof that live shutdown is safe.'
    }
}

function Invoke-AceLocalSafeQuitIsolatedGateQuiesceCore {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedAceRoot,
        [Parameter(Mandatory = $true)][string]$Session,
        [int]$GateWaitSeconds,
        [int]$QuiesceWaitSeconds,
        [switch]$SimulateGateThrow,
        [switch]$SimulateQuiesceWaitThrow
    )
    $phases = @()
    $phases += New-AceLocalSafeQuitPhaseResult -Id 'PREFLIGHT' -ExecutionStatus 'EXECUTED' `
        -Detail 'Isolated TEMP ace root validated.'

    $gate = Invoke-AceLocalSafeQuitShutdownGatePhase -AceRoot $ResolvedAceRoot -LauncherSession $Session `
        -WaitSeconds $GateWaitSeconds -TestSimulateGateThrow:$SimulateGateThrow

    $gateDetail = ('Outcome={0}; GateStatus={1}' -f $gate.Outcome, $gate.GateStatus)
    if ($gate.FinalState -and $gate.FinalState.evaluation -and $gate.FinalState.evaluation.reasons) {
        $reasons = @($gate.FinalState.evaluation.reasons) -join ','
        if ($reasons) { $gateDetail += ('; reasons=' + $reasons) }
    }
    $phases += New-AceLocalSafeQuitPhaseResult -Id 'SHUTDOWN_GATE' -ExecutionStatus 'EXECUTED' -Detail $gateDetail
    $phases += New-AceLocalSafeQuitPhaseResult -Id 'WAIT_FOR_ACTIVE_OPERATIONS' -ExecutionStatus 'SKIPPED' `
        -Detail 'Gate refresh loop satisfied wait-for-operations (5.19.5).'

    $quiesce = $null
    if (-not $gate.Ok) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'PERIODIC_SAVE_QUIESCE' -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Skipped — shutdown gate did not reach SAFE_TO_STOP.'
    }
    else {
        $quiesce = Invoke-AceLocalSafeQuitPeriodicSaveQuiesce -AceRoot $ResolvedAceRoot -LauncherSession $Session `
            -WaitSeconds $QuiesceWaitSeconds -TestSimulateWaitThrow:$SimulateQuiesceWaitThrow
        $qDetail = ('Outcome={0}; {1}' -f $quiesce.Outcome, $quiesce.Detail)
        if ($quiesce.PSObject.Properties.Match('FinalLockClass').Count -gt 0 -and $quiesce.FinalLockClass) {
            $qDetail += ('; lock=' + $quiesce.FinalLockClass)
        }
        $qStatus = if ($quiesce.Ok) { 'EXECUTED' } else { 'FAILED' }
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'PERIODIC_SAVE_QUIESCE' -ExecutionStatus $qStatus -Detail $qDetail
    }
    return @{
        Phases  = $phases
        Gate    = $gate
        Quiesce = $quiesce
        Ready   = ($gate.Ok -and $quiesce -and $quiesce.Ok)
    }
}

function Invoke-AceLocalSafeQuitIsolatedSnapshotReport {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedAceRoot,
        [Parameter(Mandatory = $true)][string]$Session,
        [int]$GateWaitSeconds,
        [int]$QuiesceWaitSeconds,
        [switch]$SimulateGateThrow,
        [switch]$SimulateQuiesceWaitThrow,
        [switch]$SimulateSnapshotFailAfterBackend,
        [switch]$SimulateSnapshotConcurrentDrift,
        [switch]$SimulateSnapshotVerificationFailure,
        [switch]$IncludeManagedStackStop,
        [switch]$IncludeCleanShutdownLedger,
        [switch]$SimulateStackStopThrow,
        [switch]$SimulatePartialStackStop,
        [switch]$SimulateLedgerRegisterThrow
    )
    $core = Invoke-AceLocalSafeQuitIsolatedGateQuiesceCore -ResolvedAceRoot $ResolvedAceRoot -Session $Session `
        -GateWaitSeconds $GateWaitSeconds -QuiesceWaitSeconds $QuiesceWaitSeconds `
        -SimulateGateThrow:$SimulateGateThrow -SimulateQuiesceWaitThrow:$SimulateQuiesceWaitThrow
    $phases = @($core.Phases)
    $capture = $null
    $verify = $null

    if (-not $core.Ready) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'SHUTDOWN_SNAPSHOT' -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Skipped — gate or quiesce not safe.'
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'SNAPSHOT_VERIFICATION' -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Skipped — no snapshot capture.'
    }
    else {
        $driftHook = $null
        if ($SimulateSnapshotConcurrentDrift) {
            $roots = Resolve-AceLocalSafeQuitRepoRoots -AceRoot $ResolvedAceRoot
            if ($roots.Ok) {
                $driftPath = Join-Path $roots.BackendRoot 'safe-quit-drift-test.txt'
                if (-not (Test-Path -LiteralPath $driftPath)) {
                    Set-Content -LiteralPath $driftPath -Value 'before-backup' -Encoding UTF8
                }
                $beRoot = $roots.BackendRoot
                $driftHook = {
                    Set-Content -LiteralPath (Join-Path $beRoot 'safe-quit-drift-test.txt') -Value 'mutated-during-backup' -Encoding UTF8
                }.GetNewClosure()
            }
        }
        $capture = Invoke-AceLocalSafeQuitShutdownSnapshotCapture -AceRoot $ResolvedAceRoot -LauncherSession $Session `
            -GateResult $core.Gate -QuiesceResult $core.Quiesce -DuringBackendCopyHook $driftHook `
            -SimulateFailAfterBackend:$SimulateSnapshotFailAfterBackend
        $capDetail = ('Outcome={0}; {1}' -f $capture.Outcome, $capture.Detail)
        if ($capture.SnapshotId) { $capDetail += ('; id=' + $capture.SnapshotId) }
        $capStatus = if ($capture.Ok) { 'EXECUTED' } else { 'FAILED' }
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'SHUTDOWN_SNAPSHOT' -ExecutionStatus $capStatus -Detail $capDetail

        if (-not $capture.Ok) {
            $phases += New-AceLocalSafeQuitPhaseResult -Id 'SNAPSHOT_VERIFICATION' -ExecutionStatus 'NOT_EXECUTED' `
                -Detail 'Skipped — snapshot capture failed.'
        }
        else {
            $verify = Invoke-AceLocalSafeQuitSnapshotVerification -AceRoot $ResolvedAceRoot -CaptureResult $capture `
                -SimulateIntegrityFailure:$SimulateSnapshotVerificationFailure
            $verDetail = ('Outcome={0}; {1}' -f $verify.Outcome, $verify.Detail)
            $verStatus = if ($verify.Ok) { 'EXECUTED' } else { 'FAILED' }
            $phases += New-AceLocalSafeQuitPhaseResult -Id 'SNAPSHOT_VERIFICATION' -ExecutionStatus $verStatus -Detail $verDetail
        }
    }

    $stackStop = $null
    $snapshotReady = ($core.Ready -and $capture -and $capture.Ok -and $verify -and $verify.Ok)
    if (-not $IncludeManagedStackStop) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'MANAGED_STACK_STOP' -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Not wired in isolated snapshot mode (5.19.5).'
    }
    elseif (-not $snapshotReady) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'MANAGED_STACK_STOP' -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Skipped — snapshot pipeline not verified.'
    }
    else {
        $stackStop = Invoke-AceLocalSafeQuitManagedStackStop -AceRoot $ResolvedAceRoot -LauncherSession $Session `
            -VerificationResult $verify -CaptureResult $capture -TestSimulateStackStopThrow:$SimulateStackStopThrow `
            -TestSimulatePartialStackStop:$SimulatePartialStackStop
        $stDetail = ('Outcome={0}; {1}' -f $stackStop.Outcome, $stackStop.Detail)
        $stStatus = if ($stackStop.Ok) { 'EXECUTED' } else { 'FAILED' }
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'MANAGED_STACK_STOP' -ExecutionStatus $stStatus -Detail $stDetail
    }

    $stackReady = ($snapshotReady -and $stackStop -and $stackStop.Ok)
    $cleanLedger = $null
    if (-not $IncludeCleanShutdownLedger) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'CLEAN_SHUTDOWN_LEDGER' -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Not wired in this isolated mode.'
    }
    elseif (-not $IncludeManagedStackStop -or -not $stackReady) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'CLEAN_SHUTDOWN_LEDGER' -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Skipped — managed stack stop did not complete safely.'
    }
    else {
        $cleanLedger = Invoke-AceLocalSafeQuitCleanShutdownLedgerRegister -AceRoot $ResolvedAceRoot -LauncherSession $Session `
            -GateResult $core.Gate -QuiesceResult $core.Quiesce -CaptureResult $capture -VerificationResult $verify `
            -StackStopResult $stackStop -TestSimulateLedgerRegisterThrow:$SimulateLedgerRegisterThrow
        $clDetail = ('Outcome={0}; {1}' -f $cleanLedger.Outcome, $cleanLedger.Detail)
        $clStatus = if ($cleanLedger.Ok) { 'EXECUTED' } else { 'FAILED' }
        $phases += New-AceLocalSafeQuitPhaseResult -Id 'CLEAN_SHUTDOWN_LEDGER' -ExecutionStatus $clStatus -Detail $clDetail
    }

    $phases += New-AceLocalSafeQuitPhaseResult -Id 'COMPLETE' -ExecutionStatus 'NOT_EXECUTED' `
        -Detail 'Not wired in phase 5.19.7.'

    $succeeded = $snapshotReady
    if ($IncludeManagedStackStop) {
        $succeeded = $stackReady
    }
    if ($IncludeCleanShutdownLedger) {
        $succeeded = ($stackReady -and $cleanLedger -and $cleanLedger.Ok)
    }
    $mode = 'IsolatedSnapshotTest'
    $phaseLabel = '5.19.5-isolated-snapshot'
    if ($IncludeCleanShutdownLedger) {
        $mode = 'IsolatedCleanShutdownTest'
        $phaseLabel = '5.19.7-isolated-clean-shutdown'
    }
    elseif ($IncludeManagedStackStop) {
        $mode = 'IsolatedStackStopTest'
        $phaseLabel = '5.19.6-isolated-stack-stop'
    }
    return [ordered]@{
        schemaVersion    = 1
        coordinatorPhase = $phaseLabel
        aceRoot          = $ResolvedAceRoot
        launcherSession  = $Session
        mode             = $mode
        succeeded        = $succeeded
        gate             = $core.Gate
        quiesce          = $core.Quiesce
        snapshot         = $capture
        verification     = $verify
        stackStop        = $stackStop
        cleanShutdown    = $cleanLedger
        phases           = @($phases)
        disclaimer       = 'Isolated test only; gate does not cover all Builder work; not live shutdown.'
    }
}

function Invoke-AceLocalSafeQuitIsolatedQuiesceReport {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedAceRoot,
        [Parameter(Mandatory = $true)][string]$Session,
        [int]$GateWaitSeconds,
        [int]$QuiesceWaitSeconds,
        [switch]$SimulateGateThrow,
        [switch]$SimulateQuiesceWaitThrow
    )
    $core = Invoke-AceLocalSafeQuitIsolatedGateQuiesceCore -ResolvedAceRoot $ResolvedAceRoot -Session $Session `
        -GateWaitSeconds $GateWaitSeconds -QuiesceWaitSeconds $QuiesceWaitSeconds `
        -SimulateGateThrow:$SimulateGateThrow -SimulateQuiesceWaitThrow:$SimulateQuiesceWaitThrow
    $phases = @($core.Phases)
    $gate = $core.Gate
    $quiesce = $core.Quiesce

    foreach ($id in @('SHUTDOWN_SNAPSHOT', 'SNAPSHOT_VERIFICATION', 'MANAGED_STACK_STOP', 'CLEAN_SHUTDOWN_LEDGER', 'COMPLETE')) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id $id -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Not wired in phase 5.19.4.'
    }

    $succeeded = $core.Ready
    return [ordered]@{
        schemaVersion    = 1
        coordinatorPhase = '5.19.4-isolated-quiesce'
        aceRoot          = $ResolvedAceRoot
        launcherSession  = $Session
        mode             = 'IsolatedQuiesceTest'
        succeeded        = $succeeded
        gate             = $gate
        quiesce          = $quiesce
        phases           = @($phases)
        disclaimer       = 'Isolated quiesce test only; not live shutdown.'
    }
}

function Invoke-AceLocalSafeQuitIsolatedGateReport {
    param(
        [Parameter(Mandatory = $true)][string]$ResolvedAceRoot,
        [Parameter(Mandatory = $true)][string]$Session,
        [int]$WaitSeconds,
        [switch]$SimulateGateThrow
    )
    $phases = @()
    $phases += New-AceLocalSafeQuitPhaseResult -Id 'PREFLIGHT' -ExecutionStatus 'EXECUTED' `
        -Detail 'Isolated TEMP ace root validated.'

    $gate = Invoke-AceLocalSafeQuitShutdownGatePhase -AceRoot $ResolvedAceRoot -LauncherSession $Session `
        -WaitSeconds $WaitSeconds -TestSimulateGateThrow:$SimulateGateThrow

    $gateDetail = ('Outcome={0}; GateStatus={1}' -f $gate.Outcome, $gate.GateStatus)
    if ($gate.FinalState -and $gate.FinalState.evaluation -and $gate.FinalState.evaluation.reasons) {
        $reasons = @($gate.FinalState.evaluation.reasons) -join ','
        if ($reasons) { $gateDetail += ('; reasons=' + $reasons) }
    }
    $phases += New-AceLocalSafeQuitPhaseResult -Id 'SHUTDOWN_GATE' -ExecutionStatus 'EXECUTED' -Detail $gateDetail
    $phases += New-AceLocalSafeQuitPhaseResult -Id 'WAIT_FOR_ACTIVE_OPERATIONS' -ExecutionStatus 'SKIPPED' `
        -Detail 'Covered by SHUTDOWN_GATE refresh loop (5.19.2).'

    foreach ($id in @('PERIODIC_SAVE_QUIESCE', 'SHUTDOWN_SNAPSHOT', 'SNAPSHOT_VERIFICATION', 'MANAGED_STACK_STOP', 'CLEAN_SHUTDOWN_LEDGER', 'COMPLETE')) {
        $phases += New-AceLocalSafeQuitPhaseResult -Id $id -ExecutionStatus 'NOT_EXECUTED' `
            -Detail 'Not wired in phase 5.19.2.'
    }

    if (-not $gate.Ok) {
        return [ordered]@{
            schemaVersion    = 1
            coordinatorPhase = '5.19.2-isolated-gate'
            aceRoot          = $ResolvedAceRoot
            launcherSession  = $Session
            mode             = 'IsolatedGateTest'
            succeeded        = $false
            gate             = $gate
            phases           = @($phases)
            disclaimer       = 'Isolated gate test only; not live shutdown.'
        }
    }

    return [ordered]@{
        schemaVersion    = 1
        coordinatorPhase = '5.19.2-isolated-gate'
        aceRoot          = $ResolvedAceRoot
        launcherSession  = $Session
        mode             = 'IsolatedGateTest'
        succeeded        = $true
        gate             = $gate
        phases           = @($phases)
        disclaimer       = 'Isolated gate test only; not live shutdown.'
    }
}

function Write-AceLocalSafeQuitReport {
    param([Parameter(Mandatory = $true)]$Report)
    Write-Info ('Invoke-AceLocalSafeQuit (' + $Report.mode + ')')
    Write-Info ('AceRoot: ' + $Report.aceRoot)
    if ($Report.launcherSession) {
        Write-Info ('LauncherSession: ' + $Report.launcherSession)
    }
    foreach ($row in @($Report.phases)) {
        Write-Host ('  [{0}] {1} — {2}' -f $row.executionStatus, $row.id, $row.detail) -ForegroundColor DarkGray
    }
    if ($Report.PSObject.Properties.Match('gate').Count -gt 0 -and $null -ne $Report.gate) {
        Write-Info ('Gate outcome: ' + $Report.gate.Outcome + ' / ' + $Report.gate.GateStatus)
    }
    if ($Report.PSObject.Properties.Match('quiesce').Count -gt 0 -and $null -ne $Report.quiesce) {
        Write-Info ('Quiesce outcome: ' + $Report.quiesce.Outcome + ' — ' + $Report.quiesce.Detail)
    }
    if ($Report.PSObject.Properties.Match('snapshot').Count -gt 0 -and $null -ne $Report.snapshot) {
        Write-Info ('Snapshot outcome: ' + $Report.snapshot.Outcome + ' — ' + $Report.snapshot.Detail)
    }
    if ($Report.PSObject.Properties.Match('verification').Count -gt 0 -and $null -ne $Report.verification) {
        Write-Info ('Verification outcome: ' + $Report.verification.Outcome + ' — ' + $Report.verification.Detail)
    }
    if ($Report.PSObject.Properties.Match('stackStop').Count -gt 0 -and $null -ne $Report.stackStop) {
        Write-Info ('Stack stop outcome: ' + $Report.stackStop.Outcome + ' — ' + $Report.stackStop.Detail)
    }
    if ($Report.PSObject.Properties.Match('cleanShutdown').Count -gt 0 -and $null -ne $Report.cleanShutdown) {
        Write-Info ('Clean shutdown outcome: ' + $Report.cleanShutdown.Outcome + ' — ' + $Report.cleanShutdown.Detail)
    }
    Write-Warn $Report.disclaimer
}

function Invoke-AceLocalSafeQuitIsolatedPipelineFailureExit {
    param(
        [Parameter(Mandatory = $true)]$Report,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [switch]$AllowStackStopFailure
    )
    if ($Report.gate -and $Report.gate.Outcome -eq 'blocked') {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
        return 5
    }
    if ($Report.gate -and $Report.gate.Outcome -eq 'timeout') {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
        return 6
    }
    if ($Report.quiesce -and -not $Report.quiesce.Ok) {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
        switch ([string]$Report.quiesce.Outcome) {
            'wait_timeout' { return 9 }
            'session_mismatch' { return 10 }
            { $_ -in @('lock_unsafe', 'stop_foreign', 'still_writing') } { return 8 }
        }
        return 1
    }
    if ($Report.snapshot -and -not $Report.snapshot.Ok) {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
        return 11
    }
    if ($Report.verification -and -not $Report.verification.Ok) {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
        return 12
    }
    if ($AllowStackStopFailure -and $Report.stackStop -and -not $Report.stackStop.Ok) {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
        return 13
    }
    if ($Report.cleanShutdown -and -not $Report.cleanShutdown.Ok) {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
        return 14
    }
    Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $AceRoot -QuiesceResult $Report.quiesce
    return 1
}

function Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        $QuiesceResult
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) { return }
    try {
        $attempt = $null
        if ($QuiesceResult -and $QuiesceResult.AttemptId) {
            $attempt = [string]$QuiesceResult.AttemptId
        }
        if (-not $attempt) {
            $initPath = Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath -AceRoot $AceRoot
            if (Test-Path -LiteralPath $initPath) {
                $attempt = [string]((Get-Content -LiteralPath $initPath -Raw | ConvertFrom-Json).safeQuitAttemptId)
            }
        }
        if ($attempt) {
            [void](Reset-AceLocalSafeQuitFailedAttempt -AceRoot $AceRoot -SafeQuitAttemptId $attempt)
        }
    }
    catch {
        Write-Warn ('Safe quit attempt cleanup: ' + $_.Exception.Message)
    }
}

$modeCount = @($DryRun, $IsolatedGateTest, $IsolatedQuiesceTest, $IsolatedSnapshotTest, $IsolatedStackStopTest, $IsolatedCleanShutdownTest | Where-Object { $_ }).Count
if ($modeCount -eq 0) {
    Write-Err 'Safe quit requires a dry-run or isolated test mode switch. Live shutdown is not implemented.'
    exit 3
}
if ($modeCount -gt 1) {
    Write-Err 'Use exactly one isolated/dry-run mode switch.'
    exit 3
}

$aceResolved = $AceRoot
try {
    if (Test-Path -LiteralPath $AceRoot) {
        $aceResolved = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    }
}
catch {
    Write-Err ('AceRoot not accessible: ' + $_.Exception.Message)
    exit 4
}

if ($IsolatedGateTest -or $IsolatedQuiesceTest -or $IsolatedSnapshotTest -or $IsolatedStackStopTest -or $IsolatedCleanShutdownTest) {
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $aceResolved)) {
        Write-Err 'Isolated tests require AceRoot under the system TEMP directory only.'
        exit 7
    }
    if (-not $LauncherSession) {
        $regPath = Join-Path $aceResolved '_backups\local-dev\managed-processes.json'
        if (Test-Path -LiteralPath $regPath) {
            try {
                $reg = Get-Content -LiteralPath $regPath -Raw | ConvertFrom-Json
                $LauncherSession = [string]$reg.launcherSession
            }
            catch { }
        }
    }
    if (-not ($LauncherSession -replace '\s', '').Trim()) {
        Write-Err 'Isolated tests require -LauncherSession or managed-processes.json in temp layout.'
        exit 4
    }
}

$lock = Enter-AceLocalSafeQuitMutex
if (-not $lock.Acquired) {
    Write-Warn ('Safe quit rejected: ' + $lock.Reason)
    if ($lock.Mutex) {
        try { $lock.Mutex.Dispose() } catch { }
    }
    exit 2
}

$report = $null
try {
    if ($DryRun) {
        $report = Invoke-AceLocalSafeQuitDryRunReport -ResolvedAceRoot $aceResolved -Session $LauncherSession `
            -SimulateThrowAfterLock:$TestSimulateThrowAfterLock -HoldLockMilliseconds $TestHoldLockMilliseconds
        Write-AceLocalSafeQuitReport -Report $report
        Write-Ok 'Safe quit DryRun completed (no LOCAL state changes).'
        exit 0
    }

    if ($IsolatedSnapshotTest -or $IsolatedStackStopTest -or $IsolatedCleanShutdownTest) {
        $includeStack = ($IsolatedStackStopTest -or $IsolatedCleanShutdownTest)
        $report = Invoke-AceLocalSafeQuitIsolatedSnapshotReport -ResolvedAceRoot $aceResolved -Session $LauncherSession `
            -GateWaitSeconds $GateWaitSeconds -QuiesceWaitSeconds $PeriodicSaveQuiesceWaitSeconds `
            -SimulateGateThrow:$TestSimulateGateThrow -SimulateQuiesceWaitThrow:$TestSimulateQuiesceWaitThrow `
            -SimulateSnapshotFailAfterBackend:$TestSimulateSnapshotFailAfterBackend `
            -SimulateSnapshotConcurrentDrift:$TestSimulateSnapshotConcurrentDrift `
            -SimulateSnapshotVerificationFailure:$TestSimulateSnapshotVerificationFailure `
            -IncludeManagedStackStop:$includeStack -IncludeCleanShutdownLedger:$IsolatedCleanShutdownTest `
            -SimulateStackStopThrow:$TestSimulateStackStopThrow -SimulatePartialStackStop:$TestSimulatePartialStackStop `
            -SimulateLedgerRegisterThrow:$TestSimulateLedgerRegisterThrow
        Write-AceLocalSafeQuitReport -Report $report
        if (-not $report.succeeded) {
            Write-Warn 'Isolated pipeline did not complete safely.'
            exit (Invoke-AceLocalSafeQuitIsolatedPipelineFailureExit -Report $report -AceRoot $aceResolved `
                    -AllowStackStopFailure:$includeStack)
        }
        if ($IsolatedCleanShutdownTest) {
            Write-Ok 'Isolated clean-shutdown test completed (ledger recorded; COMPLETE not wired).'
        }
        elseif ($IsolatedStackStopTest) {
            Write-Ok 'Isolated stack stop test completed (no clean-shutdown ledger).'
        }
        else {
            Write-Ok 'Isolated snapshot test completed (verified snapshot; no stack stop/ledger).'
        }
        exit 0
    }

    if ($IsolatedQuiesceTest) {
        $report = Invoke-AceLocalSafeQuitIsolatedQuiesceReport -ResolvedAceRoot $aceResolved -Session $LauncherSession `
            -GateWaitSeconds $GateWaitSeconds -QuiesceWaitSeconds $PeriodicSaveQuiesceWaitSeconds `
            -SimulateGateThrow:$TestSimulateGateThrow -SimulateQuiesceWaitThrow:$TestSimulateQuiesceWaitThrow
        Write-AceLocalSafeQuitReport -Report $report
        if (-not $report.succeeded) {
            Write-Warn 'Isolated quiesce test did not complete safely; snapshot/stop/ledger not executed.'
            if ($report.gate -and $report.gate.Outcome -eq 'blocked') { exit 5 }
            if ($report.gate -and $report.gate.Outcome -eq 'timeout') { exit 6 }
            if ($report.quiesce) {
                switch ([string]$report.quiesce.Outcome) {
                    'wait_timeout' { exit 9 }
                    'session_mismatch' { exit 10 }
                    { $_ -in @('lock_unsafe', 'stop_foreign', 'still_writing') } { exit 8 }
                }
            }
            exit 1
        }
        Write-Ok 'Isolated quiesce test completed (no snapshot/stop/ledger).'
        exit 0
    }

    $report = Invoke-AceLocalSafeQuitIsolatedGateReport -ResolvedAceRoot $aceResolved -Session $LauncherSession `
        -WaitSeconds $GateWaitSeconds -SimulateGateThrow:$TestSimulateGateThrow
    Write-AceLocalSafeQuitReport -Report $report
    if (-not $report.succeeded) {
        Write-Warn 'Isolated gate test did not reach SAFE_TO_STOP; later phases not executed.'
        if ($report.gate.Outcome -eq 'blocked') { exit 5 }
        if ($report.gate.Outcome -eq 'timeout') { exit 6 }
        exit 1
    }
    Write-Ok 'Isolated gate test reached SAFE_TO_STOP (no snapshot/stop/ledger).'
    exit 0
}
catch {
    Write-Err ('Safe quit aborted: ' + $_.Exception.Message)
    if (($IsolatedQuiesceTest -or $IsolatedSnapshotTest -or $IsolatedStackStopTest -or $IsolatedCleanShutdownTest) -and (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $aceResolved)) {
        Invoke-AceLocalSafeQuitIsolatedCleanupAfterFailure -AceRoot $aceResolved -QuiesceResult $null
    }
    exit 1
}
finally {
    Exit-AceLocalSafeQuitMutex -MutexHandle $lock.Mutex -Owns $lock.Owns
}
