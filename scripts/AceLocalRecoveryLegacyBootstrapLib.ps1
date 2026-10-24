# ACE LOCAL — one-time legacy install recovery bootstrap (phase 5.13).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryCheckpointLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryDirtyDetectLib.ps1')
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryStartupGateLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')

$script:AceLocalRecoveryLegacyBootstrapStopRequestFile = 'recovery-legacy-bootstrap-stop.request.json'
$script:AceLocalRecoveryLegacyBootstrapDefaultBackendPort = 5000

$script:AceLocalRecoveryLegacyBootstrapStateFile = 'recovery-legacy-bootstrap-state.json'
$script:AceLocalRecoveryLegacyBootstrapLockFile = 'recovery-legacy-bootstrap.lock'

function Get-AceLocalRecoveryLegacyBootstrapStatePath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryLegacyBootstrapStateFile
}

function Get-AceLocalRecoveryLegacyBootstrapLockPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryLegacyBootstrapLockFile
}

function Read-AceLocalRecoveryLegacyBootstrapState {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalRecoveryLegacyBootstrapStatePath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try {
        return (Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    catch { return $null }
}

function Write-AceLocalRecoveryLegacyBootstrapState {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)]$Object
    )
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLegacyBootstrapStatePath -AceRoot $AceRoot) -Object $Object
}

function Test-AceLocalRecoveryLegacyBootstrapLockHeld {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalRecoveryLegacyBootstrapLockPath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) {
        return @{ Held = $false; Reason = 'no_lock' }
    }
    try {
        $lock = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        $pid = [int]$lock.pid
        if ($pid -le 0) {
            return @{ Held = $true; Reason = 'lock_invalid_pid' }
        }
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            return @{ Held = $true; Reason = 'lock_live'; Pid = $pid }
        }
        return @{ Held = $false; Reason = 'lock_stale'; Pid = $pid }
    }
    catch {
        return @{ Held = $true; Reason = 'lock_unreadable' }
    }
}

function Enter-AceLocalRecoveryLegacyBootstrapLock {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $existing = Test-AceLocalRecoveryLegacyBootstrapLockHeld -AceRoot $AceRoot
    if ($existing.Held) {
        throw ('RECOVERY_LEGACY_BOOTSTRAP_LOCK_HELD:' + [string]$existing.Reason)
    }
    $payload = [ordered]@{
        pid       = $PID
        startedAt = (Get-Date).ToUniversalTime().ToString('o')
        host      = [Environment]::MachineName
    }
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLegacyBootstrapLockPath -AceRoot $AceRoot) -Object $payload
}

function Exit-AceLocalRecoveryLegacyBootstrapLock {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalRecoveryLegacyBootstrapLockPath -AceRoot $AceRoot
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-AceLocalRecoveryLegacyBootstrapAssessment {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$SessionsRoot = ''
    )
    $backendRoot = Split-Path -Parent $ScriptDir
    $code = @'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_recovery_legacy_bootstrap import assess_legacy_install_recovery_bootstrap
ace = Path(sys.argv[2])
sess = Path(sys.argv[3]) if sys.argv[3] else None
result = assess_legacy_install_recovery_bootstrap(ace, sessions_root=sess)
print(json.dumps(result.to_dict(), ensure_ascii=False))
'@
    $ace = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    if (-not $SessionsRoot) {
        $SessionsRoot = Join-Path $ace '_backups\sessions'
    }
    if (-not (Test-Path -LiteralPath $SessionsRoot)) { $SessionsRoot = '' }
    $raw = python -c $code $backendRoot $ace $SessionsRoot 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ("RECOVERY_LEGACY_BOOTSTRAP_ASSESS_FAILED:$raw")
    }
    return ($raw | ConvertFrom-Json)
}

function Test-AceLocalRecoveryLegacyBootstrapManifestCoverage {
    param(
        [Parameter(Mandatory = $true)]$WorkspaceScan,
        [Parameter(Mandatory = $true)][string]$CheckpointRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot
    )
    $rows = @(Import-AceLocalRecoveryCheckpointManifestRows -CheckpointRoot $CheckpointRoot)
    $keys = @{}
    foreach ($r in $rows) {
        $norm = ([string]$r.RelPath -replace '\\', '/').ToLowerInvariant()
        $keys[("{0}|{1}" -f [string]$r.Area, $norm)] = $true
    }
    $missing = @()
    foreach ($side in @(
        @{ Label = 'BACKEND'; Root = $BackendRoot; Block = $WorkspaceScan.Backend }
        @{ Label = 'FRONTEND'; Root = $FrontendRoot; Block = $WorkspaceScan.Frontend }
    )) {
        foreach ($e in @($side.Block.Entries)) {
            if ([string]$e.changeType -eq 'deleted') { continue }
            $rel = ([string]$e.path) -replace '/', '\'
            $full = Join-Path $side.Root $rel
            if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
            $size = (Get-Item -LiteralPath $full).Length
            $ex = Test-AceLocalShutdownPathExcluded -RelativePath $rel -SizeBytes $size
            if ($ex.Excluded) { continue }
            $k = ("{0}|{1}" -f $side.Label, (([string]$e.path) -replace '\\', '/').ToLowerInvariant())
            if (-not $keys.ContainsKey($k)) { $missing += $k }
        }
    }
    return @{
        Ok      = ($missing.Count -eq 0)
        Missing = @($missing)
        Rows    = $rows.Count
    }
}

function Invoke-AceLocalRecoveryLegacyBootstrapPipeline {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [string]$CheckpointsRootOverride = '',
        [switch]$SkipAssessment
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $backendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
    $frontendRoot = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath

    $result = @{
        Ok       = $false
        Phase    = 'assessment'
        Errors   = @()
        Evidence = @{}
    }

    if (-not $SkipAssessment) {
        $assess = Invoke-AceLocalRecoveryLegacyBootstrapAssessment -AceRoot $aceRoot
        $result.Evidence['assessment'] = $assess
        if (-not $assess.bootstrapAllowed) {
            $result.Phase = 'assessment_blocked'
            $result.Errors = @($assess.blockReasons)
            Write-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot -Object ([ordered]@{
                status       = 'BLOCKED'
                updatedAt    = (Get-Date).ToUniversalTime().ToString('o')
                blockReasons = @($assess.blockReasons)
            })
            return $result
        }
        $managedSession = [string]$assess.managedLauncherSession
    }
    else {
        $managedSession = (Get-Content -LiteralPath (Join-Path $aceRoot '_backups\local-dev\managed-processes.json') -Raw | ConvertFrom-Json).launcherSession
    }

    try {
        Enter-AceLocalRecoveryLegacyBootstrapLock -AceRoot $aceRoot
    }
    catch {
        $result.Errors = @($_.Exception.Message)
        return $result
    }

    try {
        $result.Phase = 'ledger'
        $ledgerPath = Get-AceLocalRecoveryLedgerPath -AceRoot $aceRoot
        if (-not (Test-Path -LiteralPath $ledgerPath)) {
            $null = Initialize-AceLocalRecoveryLegacyBootstrapLedger -AceRoot $aceRoot -ManagedLauncherSession $managedSession
        }
        else {
            Set-AceLocalRecoveryLegacyBootstrapLedgerStatus -AceRoot $aceRoot -Status 'IN_PROGRESS' | Out-Null
        }

        Write-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot -Object ([ordered]@{
            status           = 'IN_PROGRESS'
            managedSession   = $managedSession
            startedAt        = (Get-Date).ToUniversalTime().ToString('o')
            phase            = 'full_checkpoint'
            checkpointId     = $null
            lastError        = $null
        })

        if (Test-AceLocalRecoveryLegacyBootstrapStopRequested -AceRoot $aceRoot) {
            $result.Phase = 'aborted_local_quit'
            $result.Errors = @('local_quit_requested')
            return $result
        }

        $result.Phase = 'full_checkpoint'
        $scanBefore = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $backendRoot -FrontendRoot $frontendRoot
        $statusBeforeBe = git -C $backendRoot status -u --porcelain=v1 2>$null | Out-String
        $headBeforeBe = (git -C $backendRoot rev-parse HEAD 2>$null).Trim()

        $cpRoot = $CheckpointsRootOverride
        if (-not $cpRoot) {
            $cpRoot = Join-Path $aceRoot '_backups\local-dev\recovery-checkpoints'
        }
        $cp = New-AceLocalRecoveryCheckpoint -BackendRoot $backendRoot -FrontendRoot $frontendRoot -AceRoot $aceRoot `
            -LauncherSession $managedSession -CheckpointType 'full' -CheckpointsRootOverride $cpRoot

        if (-not $cp.Ok -or [string]$cp.Status -ne 'VERIFIED') {
            $err = @($cp.Errors) -join ';'
            Set-AceLocalRecoveryLegacyBootstrapLedgerStatus -AceRoot $aceRoot -Status 'FAILED' -LastError $err | Out-Null
            Write-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot -Object ([ordered]@{
                status        = 'FAILED'
                updatedAt     = (Get-Date).ToUniversalTime().ToString('o')
                phase         = 'full_checkpoint'
                checkpointId  = [string]$cp.CheckpointId
                lastError     = $err
            })
            $result.Phase = 'full_checkpoint_failed'
            $result.Errors = @($err)
            return $result
        }

        $cov = Test-AceLocalRecoveryLegacyBootstrapManifestCoverage -WorkspaceScan $scanBefore `
            -CheckpointRoot $cp.CheckpointRoot -BackendRoot $backendRoot -FrontendRoot $frontendRoot
        if (-not $cov.Ok) {
            $err = 'MANIFEST_COVERAGE_GAP:' + ($cov.Missing -join ',')
            Set-AceLocalRecoveryLegacyBootstrapLedgerStatus -AceRoot $aceRoot -Status 'FAILED' -LastError $err | Out-Null
            $result.Phase = 'manifest_failed'
            $result.Errors = @($err)
            return $result
        }

        $integrity = Test-AceLocalRecoveryCheckpointIntegrity -CheckpointRoot $cp.CheckpointRoot
        if (-not $integrity.Ok) {
            $err = 'INTEGRITY:' + ($integrity.Errors -join ',')
            Set-AceLocalRecoveryLegacyBootstrapLedgerStatus -AceRoot $aceRoot -Status 'FAILED' -LastError $err | Out-Null
            $result.Phase = 'integrity_failed'
            $result.Errors = @($err)
            return $result
        }

        Set-AceLocalRecoveryLegacyBootstrapLedgerStatus -AceRoot $aceRoot -Status 'COMPLETED' -ClearNoVerifiedFlag | Out-Null
        Write-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot -Object ([ordered]@{
            status       = 'COMPLETED'
            updatedAt    = (Get-Date).ToUniversalTime().ToString('o')
            checkpointId = [string]$cp.CheckpointId
            manifestRows = [int]$cov.Rows
        })

        $statusAfterBe = git -C $backendRoot status -u --porcelain=v1 2>$null | Out-String
        $headAfterBe = (git -C $backendRoot rev-parse HEAD 2>$null).Trim()
        if ($statusBeforeBe -ne $statusAfterBe -or $headBeforeBe -ne $headAfterBe) {
            throw 'RECOVERY_LEGACY_BOOTSTRAP_SOURCE_REPO_MUTATED'
        }

        $result.Ok = $true
        $result.Phase = 'completed'
        $result.CheckpointId = [string]$cp.CheckpointId
        return $result
    }
    finally {
        Exit-AceLocalRecoveryLegacyBootstrapLock -AceRoot $aceRoot
    }
}

function Get-AceLocalRecoveryLegacyBootstrapStopRequestPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryLegacyBootstrapStopRequestFile
}

function Request-AceLocalRecoveryLegacyBootstrapStop {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLegacyBootstrapStopRequestPath -AceRoot $AceRoot) -Object ([ordered]@{
        schemaVersion = 1
        requestedAt   = (Get-Date).ToUniversalTime().ToString('o')
        reason        = 'local_quit'
    })
}

function Test-AceLocalRecoveryLegacyBootstrapStopRequested {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalRecoveryLegacyBootstrapStopRequestPath -AceRoot $AceRoot
    return (Test-Path -LiteralPath $path)
}

function Clear-AceLocalRecoveryLegacyBootstrapStopRequest {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalRecoveryLegacyBootstrapStopRequestPath -AceRoot $AceRoot
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}

function Test-AceLocalRecoveryLocalQuitInProgress {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $backendRoot = Split-Path -Parent $ScriptDir
    $runner = Join-Path $env:TEMP ("ace-legacy-quit-gate-{0}.py" -f [guid]::NewGuid().ToString('N'))
    $runnerPy = @'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_shutdown_gate import (
    GATE_SAFE_TO_STOP,
    GATE_SHUTDOWN_BLOCKED,
    GATE_SHUTDOWN_REQUESTED,
    GATE_WAITING_FOR_OPERATION,
    read_shutdown_gate_state,
)

ace = Path(sys.argv[2])
state = read_shutdown_gate_state(ace)
if not state:
    print(json.dumps({"quitInProgress": False, "gateStatus": ""}))
    raise SystemExit(0)
status = str(state.get("gateStatus") or "").upper()
quit_in_progress = status in (
    GATE_SHUTDOWN_REQUESTED,
    GATE_WAITING_FOR_OPERATION,
    GATE_SHUTDOWN_BLOCKED,
    GATE_SAFE_TO_STOP,
)
print(json.dumps({"quitInProgress": quit_in_progress, "gateStatus": status}))
'@
    Set-Content -LiteralPath $runner -Value $runnerPy -Encoding UTF8
    try {
        $raw = python $runner $backendRoot $AceRoot 2>&1
        if ($LASTEXITCODE -ne 0) {
            return @{ QuitInProgress = $false; Reason = 'gate_read_failed' }
        }
        $line = ($raw | Out-String).Trim().Split("`n") | Where-Object { $_.Trim().StartsWith('{') } | Select-Object -Last 1
        $payload = $line | ConvertFrom-Json
        return @{
            QuitInProgress = [bool]$payload.quitInProgress
            GateStatus     = [string]$payload.gateStatus
        }
    }
    finally {
        Remove-Item -LiteralPath $runner -Force -ErrorAction SilentlyContinue
    }
}

function Test-AceLocalRecoveryLegacyBootstrapPeriodicSaveTransitionReady {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [int]$BackendPort = $script:AceLocalRecoveryLegacyBootstrapDefaultBackendPort
    )
    $reasons = @()
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath

    if (Test-AceLocalRecoveryLegacyBootstrapStopRequested -AceRoot $aceRoot) {
        $reasons += 'local_quit_requested'
    }
    $quit = Test-AceLocalRecoveryLocalQuitInProgress -AceRoot $aceRoot
    if ($quit.QuitInProgress) {
        $reasons += 'local_quit_gate'
    }

    $lock = Test-AceLocalRecoveryLegacyBootstrapLockHeld -AceRoot $aceRoot
    if ($lock.Held) {
        $reasons += ('bootstrap_lock:' + [string]$lock.Reason)
    }

    $state = Read-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot
    if (-not $state -or [string]$state.status -ne 'COMPLETED') {
        $reasons += 'bootstrap_state_not_completed'
    }

    $ledger = Read-AceLocalRecoveryLedger -AceRoot $aceRoot -AllowCreateStateDir
    if (-not $ledger) {
        $reasons += 'ledger_missing'
    }
    else {
        $bootOk = $false
        if ($ledger.PSObject.Properties.Match('legacyRecoveryBootstrap').Count -gt 0) {
            $bootOk = ([string]$ledger.legacyRecoveryBootstrap.status -eq 'COMPLETED')
        }
        if (-not $bootOk) {
            $reasons += 'ledger_bootstrap_not_completed'
        }
        if (-not $ledger.lastVerifiedCheckpoint) {
            $reasons += 'ledger_missing_verified_checkpoint'
        }
        else {
            $cid = [string]$ledger.lastVerifiedCheckpoint.checkpointId
            if ($state -and [string]$state.checkpointId -and $cid -ne [string]$state.checkpointId) {
                $reasons += 'ledger_checkpoint_mismatch_state'
            }
            $cpRoot = [string]$ledger.lastVerifiedCheckpoint.checkpointPath
            if (-not $cpRoot -or -not (Test-Path -LiteralPath $cpRoot)) {
                $cpRoot = Join-Path (Join-Path $aceRoot '_backups\local-dev\recovery-checkpoints') $cid
            }
            if (-not (Test-Path -LiteralPath $cpRoot)) {
                $reasons += 'verified_checkpoint_path_missing'
            }
            else {
                $meta = Read-AceLocalRecoveryCheckpointMetadata -CheckpointRoot $cpRoot
                if (-not $meta -or [string]$meta.status -ne 'VERIFIED') {
                    $reasons += 'checkpoint_not_verified_on_disk'
                }
                else {
                    $integrity = Test-AceLocalRecoveryCheckpointIntegrity -CheckpointRoot $cpRoot
                    if (-not $integrity.Ok) {
                        $reasons += ('checkpoint_integrity:' + ($integrity.Errors -join ','))
                    }
                }
            }
        }
    }

    $managed = Read-AceLocalManagedRegistry -AceRoot $aceRoot -AllowCreateStateDir
    $managedSession = if ($managed) { [string]$managed.launcherSession } else { '' }
    $ledgerSession = if ($ledger) { [string]$ledger.launcherSession } else { '' }
    if (-not $managedSession) {
        $reasons += 'managed_launcher_session_missing'
    }
    elseif ($ledgerSession -and $managedSession -ne $ledgerSession) {
        $reasons += 'launcher_session_mismatch'
    }

    $assess = Invoke-AceLocalRecoveryLegacyBootstrapAssessment -AceRoot $aceRoot
    if ($assess.bootstrapAllowed) {
        $reasons += 'legacy_bootstrap_still_pending'
    }

    $occupied = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceRoot `
        -ExpectedLauncherSession $managedSession -ManagedRegistry $managed
    if ($occupied.Occupied -and $occupied.Reason -in @('verified_wrapper', 'verified_registry_wrapper', 'verified_service_pid', 'lock_live')) {
        $reasons += 'periodic_save_already_running'
    }

    $probe = Test-AceLocalRecoveryBackendHealthProbe -AceRoot $aceRoot -BackendPort $BackendPort `
        -EngineBackendRoot (Split-Path -Parent $ScriptDir)
    if (-not $probe.Ok) {
        $reasons += ('backend_probe_failed:' + [string]$probe.Reason)
    }

    return @{
        Ready   = ($reasons.Count -eq 0)
        Reasons = @($reasons)
    }
}

function Invoke-AceLocalRecoveryPeriodicSaveAfterLegacyBootstrap {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [int]$BackendPort = $script:AceLocalRecoveryLegacyBootstrapDefaultBackendPort,
        [string]$CheckpointsRootOverride = ''
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $result = @{
        Started = $false
        Reason  = ''
        Details = @()
    }

    if (Test-AceLocalRecoveryLegacyBootstrapStopRequested -AceRoot $aceRoot) {
        $result.Reason = 'local_quit_requested'
        return $result
    }
    $quit = Test-AceLocalRecoveryLocalQuitInProgress -AceRoot $aceRoot
    if ($quit.QuitInProgress) {
        $result.Reason = 'local_quit_gate'
        return $result
    }

    $ready = Test-AceLocalRecoveryLegacyBootstrapPeriodicSaveTransitionReady -AceRoot $aceRoot `
        -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -BackendPort $BackendPort
    if (-not $ready.Ready) {
        $result.Reason = 'transition_not_ready'
        $result.Details = @($ready.Reasons)
        return $result
    }

    try {
        $gate = Invoke-AceLocalRecoveryStartupGateCheck -AceRoot $aceRoot -SkipCoordinatorLock `
            -SkipLiveProcessInspection:$false
    }
    catch {
        $result.Reason = 'startup_gate_failed'
        $result.Details = @([string]$_.Exception.Message)
        return $result
    }

    if ($gate.checkFailed) {
        $result.Reason = 'startup_gate_check_failed'
        $result.Details = @([string]$gate.error)
        return $result
    }

    $managed = Read-AceLocalManagedRegistry -AceRoot $aceRoot -AllowCreateStateDir
    if (-not $managed) {
        $result.Reason = 'managed_registry_missing'
        return $result
    }

    $spawn = Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart -BackendRoot $BackendRoot `
        -FrontendRoot $FrontendRoot -AceRoot $aceRoot -BackendPort $BackendPort -GateResult $gate `
        -ManagedRegistry $managed -EngineBackendRoot (Split-Path -Parent $ScriptDir)

    if ($spawn.Started) {
        $state = Read-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot
        if (-not $state) { $state = [ordered]@{} }
        $state | Add-Member -NotePropertyName 'periodicSaveActivatedAt' -NotePropertyValue (Get-Date).ToUniversalTime().ToString('o') -Force
        $state | Add-Member -NotePropertyName 'periodicSaveWrapperPid' -NotePropertyValue ([int]$spawn.WrapperPid) -Force
        Write-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot -Object $state
        $result.Started = $true
        $result.Reason = 'started'
        return $result
    }

    $result.Reason = [string]$spawn.Reason
    if ($spawn.Detail) { $result.Details = @([string]$spawn.Detail) }
    return $result
}

function Stop-AceLocalRecoveryLegacyBootstrapForLocalQuit {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [int]$WaitSeconds = 30,
        [int]$CheckpointWaitSeconds = 120,
        [switch]$WhatIf
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $lock = Test-AceLocalRecoveryLegacyBootstrapLockHeld -AceRoot $aceRoot
    $state = Read-AceLocalRecoveryLegacyBootstrapState -AceRoot $aceRoot
    $workerPid = 0
    if ($state -and $state.PSObject.Properties.Match('workerPid').Count -gt 0) {
        $workerPid = [int]$state.workerPid
    }
    $active = $lock.Held -or ($workerPid -gt 0 -and (Get-Process -Id $workerPid -ErrorAction SilentlyContinue))
    if (-not $active) {
        return @{ Stopped = $true; Reason = 'not_running'; Lock = $lock }
    }
    if ($WhatIf) {
        return @{ Stopped = $true; Reason = 'whatif'; Lock = $lock; WorkerPid = $workerPid }
    }

    Request-AceLocalRecoveryLegacyBootstrapStop -AceRoot $aceRoot
    $waitBudget = if ($lock.Held) { $CheckpointWaitSeconds } else { $WaitSeconds }
    $deadline = (Get-Date).AddSeconds($waitBudget)
    do {
        $lock = Test-AceLocalRecoveryLegacyBootstrapLockHeld -AceRoot $aceRoot
        $workerAlive = ($workerPid -gt 0 -and (Get-Process -Id $workerPid -ErrorAction SilentlyContinue))
        if (-not $lock.Held -and -not $workerAlive) {
            Clear-AceLocalRecoveryLegacyBootstrapStopRequest -AceRoot $aceRoot
            return @{ Stopped = $true; Reason = 'graceful'; Lock = $lock }
        }
        Start-Sleep -Milliseconds 400
    } while ((Get-Date) -lt $deadline)

    $evidencePath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $aceRoot) 'recovery-legacy-bootstrap-stop-incomplete.json'
    Write-AceLocalRecoveryJsonAtomic -Path $evidencePath -Object @{
        schemaVersion = 1
        writtenAt     = (Get-Date).ToUniversalTime().ToString('o')
        reason        = 'stop_wait_timeout'
        waitSeconds   = $waitBudget
        lockHeld      = [bool]$lock.Held
        workerPid     = $workerPid
    }
    return @{ Stopped = $false; Reason = 'stop_timeout_evidence_saved'; Lock = $lock; WorkerPid = $workerPid }
}

function Start-AceLocalRecoveryLegacyBootstrapBackground {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [string]$CheckpointsRootOverride = ''
    )
    $lock = Test-AceLocalRecoveryLegacyBootstrapLockHeld -AceRoot $AceRoot
    if ($lock.Held) {
        return @{ Started = $false; Reason = 'bootstrap_lock_held'; Detail = $lock.Reason }
    }
    $worker = Join-Path $ScriptDir 'Start-AceLocalRecoveryLegacyBootstrapWorker.ps1'
    if (-not (Test-Path -LiteralPath $worker)) {
        throw 'RECOVERY_LEGACY_BOOTSTRAP_WORKER_MISSING'
    }
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker,
        '-AceRoot', (Resolve-Path -LiteralPath $AceRoot).ProviderPath,
        '-BackendRoot', (Resolve-Path -LiteralPath $BackendRoot).ProviderPath,
        '-FrontendRoot', (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath
    )
    if ($CheckpointsRootOverride) {
        $args += @('-CheckpointsRootOverride', $CheckpointsRootOverride)
    }
    $proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $args -WindowStyle Hidden -PassThru
    Write-AceLocalRecoveryLegacyBootstrapState -AceRoot $AceRoot -Object ([ordered]@{
        status    = 'IN_PROGRESS'
        workerPid = [int]$proc.Id
        startedAt = (Get-Date).ToUniversalTime().ToString('o')
        phase     = 'worker_spawned'
    })
    return @{
        Started    = $true
        WrapperPid = $proc.Id
        Reason     = 'background_started'
    }
}
