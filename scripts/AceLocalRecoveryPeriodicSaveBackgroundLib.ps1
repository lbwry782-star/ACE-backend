# ACE LOCAL — recovery periodic save background process (phase 5.6+; wired from Start/Stop-AceLocal in 5.9).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')

$script:AceLocalRecoveryPeriodicSaveBackgroundSpawnFileName = 'recovery-periodic-save-spawn.json'
$script:AceLocalRecoveryPeriodicSaveStopIncompleteFileName = 'recovery-periodic-save-stop-incomplete.json'
$script:AceLocalRecoveryPeriodicSaveWrapperScriptName = 'Start-AceLocalRecoveryPeriodicSave.ps1'

function Test-AceLocalRecoveryPeriodicSaveWrapperProcess {
    param(
        [Parameter(Mandatory = $true)][int]$WrapperPid,
        [string]$ExpectedWrapperStartedAt = '',
        [string]$ExpectedSpawnToken = '',
        $ServiceState = $null
    )
    if ($WrapperPid -le 0) {
        return @{ Verified = $false; Alive = $false; Reason = 'wrapper_pid_missing' }
    }
    if (-not (Test-ProcessAlive $WrapperPid)) {
        return @{ Verified = $false; Alive = $false; Reason = 'wrapper_not_alive' }
    }
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$WrapperPid" -ErrorAction SilentlyContinue
    $cmd = if ($cim) { [string]$cim.CommandLine } else { '' }
    if ($cmd -notmatch [regex]::Escape($script:AceLocalRecoveryPeriodicSaveWrapperScriptName)) {
        return @{ Verified = $false; Alive = $true; Reason = 'wrapper_command_mismatch' }
    }
    if ($ExpectedWrapperStartedAt) {
        $created = Get-Win32ProcessCreationTimeUtc -ProcId $WrapperPid
        if (-not $created) {
            return @{ Verified = $false; Alive = $true; Reason = 'wrapper_creation_unreadable' }
        }
        if (-not (Test-IsoUtcTimesMatchWithinTolerance -ExpectedIso $ExpectedWrapperStartedAt `
                    -ActualIso $created -ToleranceSeconds 180)) {
            return @{ Verified = $false; Alive = $true; Reason = 'wrapper_creation_time_mismatch' }
        }
    }
    if ($ExpectedSpawnToken -and $ServiceState -and $ServiceState.PSObject.Properties.Match('spawnToken').Count -gt 0) {
        if ([string]$ServiceState.spawnToken -and ([string]$ServiceState.spawnToken -ne $ExpectedSpawnToken)) {
            return @{ Verified = $false; Alive = $true; Reason = 'spawn_token_mismatch' }
        }
    }
    return @{ Verified = $true; Alive = $true; Reason = 'ok' }
}

function Clear-AceLocalRecoveryPeriodicSaveProvenStaleState {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$ExpectedLauncherSession = ''
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $spawn = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceRoot
    $statePath = Get-AceLocalRecoveryPeriodicSaveStatePath -AceRoot $aceRoot
    $state = $null
    if (Test-Path -LiteralPath $statePath) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            $state = $null
        }
    }

    $wrapperPid = 0
    $wrapperStartedAt = ''
    if ($spawn -and $spawn.PSObject.Properties.Match('wrapperPid').Count -gt 0) {
        $wrapperPid = [int]$spawn.wrapperPid
        $wrapperStartedAt = [string]$spawn.wrapperStartedAt
    }
    $wrapperCheck = Test-AceLocalRecoveryPeriodicSaveWrapperProcess -WrapperPid $wrapperPid `
        -ExpectedWrapperStartedAt $wrapperStartedAt -ServiceState $state
    if ($wrapperCheck.Alive) {
        return @{ Cleared = $false; Reason = 'verified_or_unverified_wrapper_alive' }
    }

    $lock = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $aceRoot `
        -ExpectedLauncherSession $ExpectedLauncherSession
    if ([string]$lock.Classification -in @('live', 'unknown', 'pid_reuse_suspected')) {
        return @{ Cleared = $false; Reason = ('lock_' + [string]$lock.Classification) }
    }
    if ([string]$lock.Classification -in @('stale_dead_pid', 'stale_old_session')) {
        Remove-Item -LiteralPath (Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $aceRoot) -Force -ErrorAction SilentlyContinue
    }

    $pidPath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $aceRoot) `
        $script:AceLocalRecoveryPeriodicSavePidFileName
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue

    if ($state -and $state.PSObject.Properties.Match('status').Count -gt 0 -and [string]$state.status -eq 'running') {
        Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $aceRoot -State @{
            status     = 'stopped'
            stoppedAt  = (Get-Date).ToUniversalTime().ToString('o')
            stopReason = 'proven_stale_reconciled'
        }
    }

    $spawnPath = Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $aceRoot
    Remove-Item -LiteralPath $spawnPath -Force -ErrorAction SilentlyContinue

    return @{ Cleared = $true; Reason = 'proven_stale_reconciled' }
}

function Test-AceLocalRecoveryPeriodicSaveInstanceOccupied {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$ExpectedLauncherSession = '',
        $ManagedRegistry = $null
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $spawn = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceRoot
    $statePath = Get-AceLocalRecoveryPeriodicSaveStatePath -AceRoot $aceRoot
    $state = $null
    if (Test-Path -LiteralPath $statePath) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            $state = $null
        }
    }

    $spawnToken = ''
    if ($spawn -and $spawn.PSObject.Properties.Match('spawnToken').Count -gt 0) {
        $spawnToken = [string]$spawn.spawnToken
    }

    if ($spawn -and $spawn.PSObject.Properties.Match('wrapperPid').Count -gt 0) {
        $wp = [int]$spawn.wrapperPid
        $ws = [string]$spawn.wrapperStartedAt
        $wrap = Test-AceLocalRecoveryPeriodicSaveWrapperProcess -WrapperPid $wp -ExpectedWrapperStartedAt $ws `
            -ExpectedSpawnToken $spawnToken -ServiceState $state
        if ($wrap.Verified) {
            return @{
                Occupied   = $true
                Reason     = 'verified_wrapper'
                Detail     = $wrap.Reason
                WrapperPid = $wp
            }
        }
        if ($wrap.Alive) {
            return @{
                Occupied   = $true
                Reason     = 'instance_unverified_process'
                Detail     = $wrap.Reason
                WrapperPid = $wp
            }
        }
    }

    if ($ManagedRegistry -and $ManagedRegistry.components `
            -and $ManagedRegistry.components.PSObject.Properties.Match('recoveryPeriodicSave').Count -gt 0) {
        $rec = $ManagedRegistry.components.recoveryPeriodicSave
        if ($rec) {
            $regPid = if ($rec.wrapperPid) { [int]$rec.wrapperPid } else { [int]$rec.pid }
            if ($regPid -gt 0) {
                $expectedCreated = ''
                if ($rec.processCreatedAt -and $rec.processCreatedAt.PSObject.Properties.Match([string]$regPid).Count -gt 0) {
                    $expectedCreated = [string]$rec.processCreatedAt.($regPid)
                }
                $wrapReg = Test-AceLocalRecoveryPeriodicSaveWrapperProcess -WrapperPid $regPid `
                    -ExpectedWrapperStartedAt $expectedCreated -ServiceState $state
                if ($wrapReg.Verified) {
                    if ($ExpectedLauncherSession -and $rec.launcherSession `
                            -and ([string]$rec.launcherSession -ne $ExpectedLauncherSession)) {
                        return @{
                            Occupied = $true
                            Reason   = 'registry_session_mismatch_live'
                            Detail   = 'wrapper_alive_other_session'
                            WrapperPid = $regPid
                        }
                    }
                    return @{
                        Occupied   = $true
                        Reason     = 'verified_registry_wrapper'
                        Detail     = $wrapReg.Reason
                        WrapperPid = $regPid
                    }
                }
                if ($wrapReg.Alive) {
                    return @{
                        Occupied   = $true
                        Reason     = 'instance_unverified_process'
                        Detail     = $wrapReg.Reason
                        WrapperPid = $regPid
                    }
                }
            }
        }
    }

    $lock = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $aceRoot `
        -ExpectedLauncherSession $ExpectedLauncherSession
    if ([string]$lock.Classification -eq 'live') {
        return @{ Occupied = $true; Reason = 'lock_live'; Detail = [string]$lock.Reason; LockPid = $lock.LockPid }
    }
    if ([string]$lock.Classification -in @('unknown', 'pid_reuse_suspected')) {
        return @{
            Occupied = $true
            Reason   = 'lock_unresolved'
            Detail   = [string]$lock.Reason
            LockPid  = $lock.LockPid
        }
    }

    $pidPath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $aceRoot) `
        $script:AceLocalRecoveryPeriodicSavePidFileName
    if (Test-Path -LiteralPath $pidPath) {
        $rawPid = (Get-Content -LiteralPath $pidPath -Raw -Encoding ASCII).Trim()
        if ($rawPid -match '^\d+$') {
            $servicePid = [int]$rawPid
            if (Test-ProcessAlive $servicePid) {
                $svcWrap = Test-AceLocalRecoveryPeriodicSaveWrapperProcess -WrapperPid $servicePid `
                    -ExpectedSpawnToken $spawnToken -ServiceState $state
                if ($svcWrap.Alive) {
                    return @{
                        Occupied   = $true
                        Reason     = if ($svcWrap.Verified) { 'verified_service_pid' } else { 'instance_unverified_process' }
                        Detail     = $svcWrap.Reason
                        WrapperPid = $servicePid
                    }
                }
            }
        }
    }

    if ($state -and $state.PSObject.Properties.Match('status').Count -gt 0 -and [string]$state.status -eq 'running') {
        return @{ Occupied = $false; Reason = 'stale_running_state'; Detail = 'needs_reconcile' }
    }

    return @{ Occupied = $false; Reason = 'no_live_instance'; Detail = '' }
}

function Get-AceLocalRecoveryPeriodicSaveGateLaunchContext {
    param(
        [Parameter(Mandatory = $true)]$GateResult
    )
    $outcome = [string]$GateResult.coordinatorOutcome
    $allow = $false
    $isolated = @()
    if ($GateResult.report -and $GateResult.report.coordinatorDecision) {
        $decision = $GateResult.report.coordinatorDecision
        $outcome = [string]$decision.coordinatorOutcome
        if ($decision.PSObject.Properties.Match('allowPeriodicSaveService').Count -gt 0) {
            $allow = [bool]$decision.allowPeriodicSaveService
        }
        if ($decision.isolatedOperations) {
            $isolated = @($decision.isolatedOperations | ForEach-Object { [string]$_ })
        }
    }
    return @{
        CoordinatorOutcome       = $outcome
        AllowPeriodicSaveService = $allow
        IsolatedOperations       = $isolated
    }
}

function Test-AceLocalRecoveryBackendHealthProbe {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][int]$BackendPort,
        [Parameter(Mandatory = $true)][string]$EngineBackendRoot
    )
    $modulePath = Join-Path $EngineBackendRoot 'engine\ace_local_dev_recovery_periodic_save_backend_probe.py'
    if (-not (Test-Path -LiteralPath $modulePath)) {
        return @{ Ok = $false; Reason = 'probe_module_missing' }
    }
    $backendRoot = (Resolve-Path -LiteralPath $EngineBackendRoot).ProviderPath
    $prevPort = $env:PORT
    $env:PORT = [string]$BackendPort
    $prevEa = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        Push-Location -LiteralPath $backendRoot
        try {
            $output = & python -m engine.ace_local_dev_recovery_periodic_save_backend_probe --ace-root $AceRoot 2>&1 `
                | Out-String
        }
        finally {
            Pop-Location
        }
        $line = ($output -split "`n" | Where-Object { $_.Trim() -match '^\{' } | Select-Object -Last 1)
        if (-not $line) {
            return @{ Ok = $false; Reason = 'probe_output_invalid' }
        }
        $payload = $line | ConvertFrom-Json
        return @{
            Ok       = [bool]$payload.ok
            Reason   = [string]$payload.reason
            Endpoint = [string]$payload.endpoint
        }
    }
    finally {
        $ErrorActionPreference = $prevEa
        if ($null -eq $prevPort) {
            Remove-Item Env:PORT -ErrorAction SilentlyContinue
        }
        else {
            $env:PORT = $prevPort
        }
    }
}

function Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][int]$BackendPort,
        [Parameter(Mandatory = $true)]$GateResult,
        [Parameter(Mandatory = $true)]$ManagedRegistry,
        [string]$EngineBackendRoot = ''
    )
    if (-not $EngineBackendRoot) {
        $EngineBackendRoot = $BackendRoot
    }
    $launcher = [string]$ManagedRegistry.launcherSession
    if ([string]::IsNullOrWhiteSpace($launcher)) {
        return @{ Started = $false; Reason = 'launcher_session_missing' }
    }
    $ctx = Get-AceLocalRecoveryPeriodicSaveGateLaunchContext -GateResult $GateResult
    $blocked = @(
        'START_BLOCKED', 'RECOVERY_UNSAFE', 'START_WITH_ISOLATED_OPERATIONS', 'GATE_CHECK_FAILED',
        'LEGACY_RECOVERY_BOOTSTRAP'
    )
    if ($blocked -contains $ctx.CoordinatorOutcome) {
        return @{ Started = $false; Reason = 'coordinator_blocked'; Outcome = $ctx.CoordinatorOutcome }
    }
    $probe = Test-AceLocalRecoveryBackendHealthProbe -AceRoot $AceRoot -BackendPort $BackendPort `
        -EngineBackendRoot $EngineBackendRoot
    if (-not $probe.Ok) {
        return @{ Started = $false; Reason = 'backend_probe_failed'; Detail = $probe.Reason }
    }
    Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $AceRoot -BackendPort $BackendPort `
        -LauncherSession $launcher
    $spawn = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
        -AceRoot $AceRoot -LauncherSession $launcher -CoordinatorOutcome $ctx.CoordinatorOutcome `
        -AllowPeriodicSaveService $ctx.AllowPeriodicSaveService -IsolatedOperations $ctx.IsolatedOperations `
        -EngineBackendRoot $EngineBackendRoot -BackendPort $BackendPort -ManagedRegistry $ManagedRegistry
    if ($spawn.Started -and $spawn.WrapperPid) {
        $logs = Get-AceLocalRecoveryPeriodicSaveBackgroundLogPaths -AceRoot $AceRoot
        $spawnRecord = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $AceRoot
        $wrapperId = [int]$spawn.WrapperPid
        $creationMap = Get-ManagedTreeProcessCreationTimes -TreePids @($wrapperId) -WrapperPid $wrapperId
        $rec = New-AceLocalManagedComponentRecord -ProcId $wrapperId -LogPath $logs.Stderr
        $rec.launcherSession = $launcher
        $rec.startedByStack = $true
        $rec.wrapperPid = $wrapperId
        if ($spawn.SpawnToken) {
            $rec | Add-Member -NotePropertyName 'spawnToken' -NotePropertyValue ([string]$spawn.SpawnToken) -Force
        }
        $createdObj = ConvertTo-ManagedProcessCreatedAtObject -Map $creationMap
        if ($createdObj) {
            $rec | Add-Member -NotePropertyName 'processCreatedAt' -NotePropertyValue $createdObj -Force
        }
        if ($ManagedRegistry.PSObject.Properties.Match('components').Count -gt 0) {
            if (-not $ManagedRegistry.components) {
                $ManagedRegistry | Add-Member -NotePropertyName 'components' -NotePropertyValue ([PSCustomObject]@{}) -Force
            }
            $ManagedRegistry.components | Add-Member -NotePropertyName 'recoveryPeriodicSave' -NotePropertyValue $rec -Force
            Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $ManagedRegistry
        }
    }
    return $spawn
}

function Stop-AceLocalRecoveryPeriodicSaveForLocalQuit {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [int]$WaitSeconds = 30,
        [int]$CheckpointWaitSeconds = 90,
        [switch]$WhatIf
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $status = Get-AceLocalRecoveryPeriodicSaveBackgroundStatus -AceRoot $aceRoot
    if (-not $status.IsRunning -and $status.StateStatus -ne 'running') {
        return @{ Stopped = $true; Reason = 'not_running'; Status = $status }
    }
    if ($WhatIf) {
        return @{ Stopped = $true; Reason = 'whatif'; Status = $status }
    }
    $lock = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $aceRoot
    $waitBudget = if ([string]$lock.Classification -eq 'live') { $CheckpointWaitSeconds } else { $WaitSeconds }
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $aceRoot
    $deadline = (Get-Date).AddSeconds($waitBudget)
    do {
        $status = Get-AceLocalRecoveryPeriodicSaveBackgroundStatus -AceRoot $aceRoot
        if (-not $status.IsRunning -and $status.StateStatus -ne 'running') {
            Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $aceRoot
            return @{ Stopped = $true; Reason = 'graceful'; Status = $status }
        }
        if ($status.ServicePid -and -not $status.ProcessAlive -and -not $status.IsRunning) {
            Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $aceRoot
            return @{ Stopped = $true; Reason = 'exited'; Status = $status }
        }
        Start-Sleep -Milliseconds 300
    } while ((Get-Date) -lt $deadline)

    $status = Get-AceLocalRecoveryPeriodicSaveBackgroundStatus -AceRoot $aceRoot
    $incompletePath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $aceRoot) `
        $script:AceLocalRecoveryPeriodicSaveStopIncompleteFileName
    Write-AceLocalRecoveryJsonAtomic -Path $incompletePath -Object @{
        schemaVersion = 1
        writtenAt     = (Get-Date).ToUniversalTime().ToString('o')
        reason        = 'stop_wait_timeout'
        waitSeconds   = $waitBudget
        lockClass     = [string]$lock.Classification
        status        = $status
    }
    return @{ Stopped = $false; Reason = 'stop_timeout_evidence_saved'; Status = $status }
}

function Get-AceLocalRecoveryPeriodicSaveSpawnPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveBackgroundSpawnFileName
}

function Get-AceLocalRecoveryPeriodicSaveBackgroundLogPaths {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $logDir = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) 'logs'
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    return @{
        Stdout = Join-Path $logDir 'recovery-periodic-save.stdout.log'
        Stderr = Join-Path $logDir 'recovery-periodic-save.stderr.log'
    }
}

function Invoke-AceLocalRecoveryPeriodicSaveActivationAssessment {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$CoordinatorOutcome,
        [Parameter(Mandatory = $true)][bool]$AllowPeriodicSaveService,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [switch]$RecoveryUnresolved,
        [string[]]$IsolatedOperations = @()
    )
    $modulePath = Join-Path $BackendRoot 'engine\ace_local_dev_recovery_periodic_save_service.py'
    if (-not (Test-Path -LiteralPath $modulePath)) {
        return @{
            allowed = $false
            reasons = @('activation_module_missing')
        }
    }
    $py = Resolve-AceLocalRecoveryGuardPython -BackendRoot $BackendRoot
    if (-not $py) {
        return @{
            allowed = $false
            reasons = @('activation_python_missing')
        }
    }
    $backendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
    $runnerPath = Join-Path $env:TEMP ("ace-ps-activation-runner-{0}.py" -f [guid]::NewGuid().ToString('N'))
    $runnerPy = @'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_recovery_periodic_save_service import evaluate_periodic_save_background_activation

raw = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
result = evaluate_periodic_save_background_activation(
    coordinator_outcome=str(raw.get("coordinatorOutcome") or ""),
    allow_periodic_save_service=bool(raw.get("allowPeriodicSaveService")),
    launcher_session=str(raw.get("launcherSession") or ""),
    recovery_unresolved=bool(raw.get("recoveryUnresolved")),
    isolated_operations=list(raw.get("isolatedOperations") or []),
)
print(json.dumps(result, ensure_ascii=False))
'@
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($runnerPath, $runnerPy, $utf8NoBom)
    $tmp = Join-Path $env:TEMP ("ace-ps-activation-{0}.json" -f [guid]::NewGuid().ToString('N'))
    $payload = @{
        coordinatorOutcome       = $CoordinatorOutcome
        allowPeriodicSaveService = $AllowPeriodicSaveService
        launcherSession          = $LauncherSession
        recoveryUnresolved       = [bool]$RecoveryUnresolved
        isolatedOperations       = @($IsolatedOperations)
    } | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($tmp, $payload, $utf8NoBom)
    try {
        $output = python $runnerPath $backendRoot $tmp 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0 -and -not ($output -match '^\s*\{')) {
            return @{ allowed = $false; reasons = @('activation_invoke_failed') }
        }
        $line = ($output -split "`n" | Where-Object { $_.Trim() -match '^\{' } | Select-Object -Last 1)
        if (-not $line) {
            return @{ allowed = $false; reasons = @('activation_output_invalid') }
        }
        return ($line | ConvertFrom-Json)
    }
    finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $runnerPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-AceLocalRecoveryPeriodicSaveBackgroundStatus {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$ExpectedLauncherSession = ''
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $statePath = Get-AceLocalRecoveryPeriodicSaveStatePath -AceRoot $aceRoot
    $pidPath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $aceRoot) $script:AceLocalRecoveryPeriodicSavePidFileName
    $state = $null
    if (Test-Path -LiteralPath $statePath) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            $state = $null
        }
    }

    $servicePid = $null
    if (Test-Path -LiteralPath $pidPath) {
        $rawPid = (Get-Content -LiteralPath $pidPath -Raw -Encoding ASCII).Trim()
        if ($rawPid -match '^\d+$') {
            $servicePid = [int]$rawPid
        }
    }

    $lockAssessment = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $aceRoot `
        -ExpectedLauncherSession $ExpectedLauncherSession

    $procAlive = $false
    if ($servicePid) {
        $procAlive = $null -ne (Get-Process -Id $servicePid -ErrorAction SilentlyContinue)
    }

    $statusText = 'stopped'
    if ($state -and $state.PSObject.Properties.Match('status').Count -gt 0) {
        $statusText = [string]$state.status
    }

    $managedRegistry = Read-AceLocalManagedRegistry -AceRoot $aceRoot -AllowCreateStateDir
    $occupancy = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceRoot `
        -ExpectedLauncherSession $ExpectedLauncherSession -ManagedRegistry $managedRegistry
    $isRunning = [bool]$occupancy.Occupied
    $crashed = ($statusText -eq 'running' -and -not $procAlive -and -not $isRunning)

    return @{
        IsRunning           = $isRunning
        ServicePid          = $servicePid
        ProcessAlive        = $procAlive
        Crashed             = $crashed
        OccupancyReason     = [string]$occupancy.Reason
        LockClassification  = [string]$lockAssessment.Classification
        StateStatus         = $statusText
        LauncherSession     = if ($state -and $state.PSObject.Properties.Match('launcherSession').Count -gt 0) {
            [string]$state.launcherSession
        } else { '' }
        SpawnRecord         = (Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceRoot)
    }
}

function Read-AceLocalRecoveryPeriodicSaveSpawnRecord {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }
    try {
        return (Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Write-AceLocalRecoveryPeriodicSaveSpawnRecord {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][hashtable]$Record
    )
    $path = Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $AceRoot
    $payload = [ordered]@{
        schemaVersion = 1
        writtenAt     = (Get-Date).ToUniversalTime().ToString('o')
    }
    foreach ($key in $Record.Keys) {
        $payload[$key] = $Record[$key]
    }
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object $payload
}

function Start-AceLocalRecoveryPeriodicSaveBackgroundProcess {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)][string]$CoordinatorOutcome,
        [Parameter(Mandatory = $true)][bool]$AllowPeriodicSaveService,
        [switch]$RecoveryUnresolved,
        [string[]]$IsolatedOperations = @(),
        [int]$TickIntervalSeconds = 30,
        [string]$CheckpointsRootOverride = '',
        [string]$EngineBackendRoot = '',
        [int]$BackendPort = 0,
        [switch]$SkipActivationCheck,
        $ManagedRegistry = $null
    )
    if (-not $EngineBackendRoot) {
        $EngineBackendRoot = Split-Path -Parent $ScriptDir
    }
    else {
        $EngineBackendRoot = (Resolve-Path -LiteralPath $EngineBackendRoot).ProviderPath
    }
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    $backendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
    $frontendRoot = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath

    if (-not $SkipActivationCheck) {
        $activation = Invoke-AceLocalRecoveryPeriodicSaveActivationAssessment -BackendRoot $EngineBackendRoot `
            -CoordinatorOutcome $CoordinatorOutcome -AllowPeriodicSaveService $AllowPeriodicSaveService `
            -LauncherSession $LauncherSession -RecoveryUnresolved:$RecoveryUnresolved `
            -IsolatedOperations $IsolatedOperations
        if (-not $activation.allowed) {
            return @{
                Started = $false
                Reason  = 'activation_denied'
                Details = @($activation.reasons)
            }
        }
    }

    if (-not $ManagedRegistry) {
        $ManagedRegistry = Read-AceLocalManagedRegistry -AceRoot $aceRoot -AllowCreateStateDir
    }

    function Resolve-PeriodicSaveSpawnDenyReason {
        param([hashtable]$Occ)
        switch ([string]$Occ.Reason) {
            'instance_unverified_process' { return 'spawn_denied_unresolved_instance' }
            'lock_unresolved' { return 'spawn_denied_unresolved_instance' }
            'verified_wrapper' { return 'already_running' }
            'verified_registry_wrapper' { return 'already_running' }
            'verified_service_pid' { return 'already_running' }
            'lock_live' { return 'already_running' }
            default { return 'already_running' }
        }
    }

    $occupied = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceRoot `
        -ExpectedLauncherSession $LauncherSession -ManagedRegistry $ManagedRegistry
    if ($occupied.Occupied) {
        return @{
            Started = $false
            Reason  = (Resolve-PeriodicSaveSpawnDenyReason -Occ $occupied)
            Detail  = [string]$occupied.Detail
            Status  = $occupied
        }
    }
    if ([string]$occupied.Reason -eq 'stale_running_state') {
        $reconcile = Clear-AceLocalRecoveryPeriodicSaveProvenStaleState -AceRoot $aceRoot `
            -ExpectedLauncherSession $LauncherSession
        if (-not $reconcile.Cleared) {
            return @{
                Started = $false
                Reason  = 'spawn_denied_stale_uncleared'
                Detail  = [string]$reconcile.Reason
            }
        }
        $occupied = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceRoot `
            -ExpectedLauncherSession $LauncherSession -ManagedRegistry $ManagedRegistry
        if ($occupied.Occupied) {
            return @{
                Started = $false
                Reason  = (Resolve-PeriodicSaveSpawnDenyReason -Occ $occupied)
                Detail  = [string]$occupied.Detail
                Status  = $occupied
            }
        }
    }

    if ($BackendPort -gt 0) {
        Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceRoot -BackendPort $BackendPort `
            -LauncherSession $LauncherSession
    }
    Write-AceLocalRecoveryPeriodicSaveActivationSnapshot -AceRoot $aceRoot -LauncherSession $LauncherSession `
        -CoordinatorOutcome $CoordinatorOutcome -AllowPeriodicSaveService $AllowPeriodicSaveService
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $aceRoot
    $logs = Get-AceLocalRecoveryPeriodicSaveBackgroundLogPaths -AceRoot $aceRoot
    $launcherScript = Join-Path $ScriptDir 'Start-AceLocalRecoveryPeriodicSave.ps1'

    $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $psExe)) {
        $psExe = 'powershell.exe'
    }

    $argList = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-WindowStyle', 'Hidden',
        '-File', $launcherScript,
        '-BackendRoot', $backendRoot,
        '-FrontendRoot', $frontendRoot,
        '-AceRoot', $aceRoot,
        '-LauncherSession', $LauncherSession,
        '-TickIntervalSeconds', [string]$TickIntervalSeconds,
        '-BackgroundMode',
        '-EngineBackendRoot', $EngineBackendRoot
    )
    if ($CheckpointsRootOverride) {
        $argList += @('-CheckpointsRootOverride', $CheckpointsRootOverride)
    }
    if ($BackendPort -gt 0) {
        $argList += @('-BackendPort', [string]$BackendPort)
    }
    $spawnToken = [guid]::NewGuid().ToString('N')
    $prevAceRoot = $env:ACE_ORI_ACE_ROOT
    $prevPort = $env:PORT
    $prevSpawn = $env:ACE_RECOVERY_PERIODIC_SAVE_SPAWN_TOKEN
    $env:ACE_ORI_ACE_ROOT = $aceRoot
    $env:ACE_RECOVERY_PERIODIC_SAVE_SPAWN_TOKEN = $spawnToken
    if ($BackendPort -gt 0) {
        $env:PORT = [string]$BackendPort
    }
    try {
        $wrapper = Start-Process -FilePath $psExe -ArgumentList $argList -WorkingDirectory $ScriptDir `
            -WindowStyle Hidden -PassThru -ErrorAction Stop
    }
    finally {
        if ($null -eq $prevSpawn) { Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_SPAWN_TOKEN -ErrorAction SilentlyContinue }
        else { $env:ACE_RECOVERY_PERIODIC_SAVE_SPAWN_TOKEN = $prevSpawn }
        if ($BackendPort -gt 0) {
            if ($null -eq $prevPort) { Remove-Item Env:PORT -ErrorAction SilentlyContinue }
            else { $env:PORT = $prevPort }
        }
        if ($null -eq $prevAceRoot) { Remove-Item Env:ACE_ORI_ACE_ROOT -ErrorAction SilentlyContinue }
        else { $env:ACE_ORI_ACE_ROOT = $prevAceRoot }
    }
    if (-not $wrapper) {
        return @{ Started = $false; Reason = 'spawn_failed' }
    }

    $wrapperStartedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceRoot -Record @{
        wrapperPid          = $wrapper.Id
        wrapperStartedAt    = $wrapperStartedAt
        spawnToken          = $spawnToken
        launcherSession     = $LauncherSession
        coordinatorOutcome  = $CoordinatorOutcome
        tickIntervalSeconds = $TickIntervalSeconds
        backendPort         = $(if ($BackendPort -gt 0) { $BackendPort } else { $null })
        scriptPath          = $launcherScript
        logStdout           = $logs.Stdout
        logStderr           = $logs.Stderr
    }

    return @{
        Started     = $true
        Reason      = 'ok'
        WrapperPid  = $wrapper.Id
        SpawnToken  = $spawnToken
        LogStdout   = $logs.Stdout
        LogStderr   = $logs.Stderr
    }
}

function Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [int]$WaitSeconds = 15
    )
    $aceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $aceRoot
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    do {
        $status = Get-AceLocalRecoveryPeriodicSaveBackgroundStatus -AceRoot $aceRoot
        if (-not $status.IsRunning -and $status.StateStatus -ne 'running') {
            break
        }
        if ($status.ServicePid -and -not $status.ProcessAlive) {
            break
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)

    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $aceRoot
    return Get-AceLocalRecoveryPeriodicSaveBackgroundStatus -AceRoot $aceRoot
}
