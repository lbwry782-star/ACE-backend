# ACE LOCAL — safe quit MANAGED_STACK_STOP (5.19.6; TEMP/isolated test driver only).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')
. (Join-Path $ScriptDir 'AceLocalMemuraiWatchLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryLegacyBootstrapLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalShutdownGateLib.ps1')

$script:AceLocalSafeQuitTestManagedLogPrefix = 'safe-quit-test-managed://'
$script:AceLocalSafeQuitTestProcessMarker = 'AceLocalSafeQuitTestProc=1'

function Test-AceLocalSafeQuitProcessHasTestMarker {
    param([Parameter(Mandatory = $true)][int]$ProcId)
    if ($ProcId -le 0) { return $false }
    if (-not (Test-ProcessAlive $ProcId)) { return $false }
    try {
        $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction Stop
        $cmd = [string]$cim.CommandLine
        return ($cmd -match [regex]::Escape($script:AceLocalSafeQuitTestProcessMarker))
    }
    catch {
        return $false
    }
}

function Test-AceLocalSafeQuitIsolatedManagedRecord {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)][string]$ExpectedLauncherSession
    )
    if (-not $Record -or -not $Record.startedByStack) {
        return @{ Ok = $false; Reason = 'not_started_by_stack' }
    }
    $logPath = [string]$Record.logPath
    if ($logPath -notlike ($script:AceLocalSafeQuitTestManagedLogPrefix + '*')) {
        return @{ Ok = $false; Reason = 'not_isolated_test_managed_record' }
    }
    if ($ExpectedLauncherSession -and $Record.launcherSession -and ([string]$Record.launcherSession -ne $ExpectedLauncherSession)) {
        return @{ Ok = $false; Reason = 'launcher_session_mismatch' }
    }
    $rootPid = if ($Record.rootPid) { [int]$Record.rootPid } else { [int]$Record.pid }
    if ($rootPid -le 0) {
        return @{ Ok = $false; Reason = 'missing_root_pid' }
    }
    if (-not (Test-ProcessAlive $rootPid)) {
        return @{ Ok = $false; Reason = 'tree_not_running' }
    }
    if (-not (Test-AceLocalSafeQuitProcessHasTestMarker -ProcId $rootPid)) {
        return @{ Ok = $false; Reason = 'root_not_test_marker' }
    }
    $treePids = @(Get-ManagedComponentTreePids -Record $Record)
    foreach ($pidVal in $treePids) {
        if (-not (Test-ProcessAlive $pidVal)) { continue }
        if (-not (Test-AceLocalSafeQuitProcessHasTestMarker -ProcId $pidVal)) {
            return @{ Ok = $false; Reason = ("tree_pid_{0}_not_test_marker" -f $pidVal) }
        }
    }
    return @{
        Ok       = $true
        Reason   = 'ok'
        RootPid  = $rootPid
        TreePids = $treePids
        WrapperPid = if ($Record.wrapperPid) { [int]$Record.wrapperPid } else { 0 }
    }
}

function Get-AceLocalSafeQuitIsolatedManagedStopActions {
    param(
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$ExpectedLauncherSession
    )
    $actions = @()
    foreach ($name in @('frontend', 'worker', 'backend')) {
        $rec = $Registry.components.$name
        if (-not $rec) { continue }
        $check = Test-AceLocalSafeQuitIsolatedManagedRecord -Record $rec -ExpectedLauncherSession $ExpectedLauncherSession
        if (-not $check.Ok) {
            $actions += [PSCustomObject]@{
                Component = $name
                Action    = 'skip'
                Reason    = $check.Reason
            }
            continue
        }
        $actions += [PSCustomObject]@{
            Component  = $name
            Action     = 'stop'
            Reason     = 'verified_test_managed_tree'
            RootPid    = [int]$check.RootPid
            TreePids   = @($check.TreePids)
            WrapperPid = [int]$check.WrapperPid
        }
    }
    return @($actions)
}

function Stop-AceLocalSafeQuitVerifiedTestProcessTree {
    param([Parameter(Mandatory = $true)][int]$RootPid)
    if (-not (Test-AceLocalSafeQuitProcessHasTestMarker -ProcId $RootPid)) {
        return @{ Stopped = $false; Reason = 'ownership_not_verified' }
    }
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ParentProcessId -eq $RootPid } | ForEach-Object {
        if (Test-AceLocalSafeQuitProcessHasTestMarker -ProcId $_.ProcessId) {
            [void](Stop-AceLocalSafeQuitVerifiedTestProcessTree -RootPid $_.ProcessId)
        }
    }
    if (Test-ProcessAlive $RootPid) {
        Stop-Process -Id $RootPid -ErrorAction Stop
    }
    return @{ Stopped = $true; Reason = 'ok' }
}

function Stop-AceLocalSafeQuitIsolatedTestPeriodicSave {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $pidPath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) 'recovery-periodic-save.pid'
    if (-not (Test-Path -LiteralPath $pidPath)) {
        return @{ Stopped = $true; Reason = 'not_running'; Step = 'periodic_save' }
    }
    $raw = (Get-Content -LiteralPath $pidPath -Raw).Trim()
    if ($raw -notmatch '^\d+$') {
        return @{ Stopped = $false; Reason = 'invalid_pid_file'; Step = 'periodic_save' }
    }
    $pidVal = [int]$raw
    if (-not (Test-AceLocalSafeQuitProcessHasTestMarker -ProcId $pidVal)) {
        return @{ Stopped = $false; Reason = 'periodic_pid_not_test_owned'; Step = 'periodic_save' }
    }
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $AceRoot
    Stop-Process -Id $pidVal -ErrorAction Stop
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    return @{ Stopped = $true; Reason = 'test_graceful'; Step = 'periodic_save' }
}

function Stop-AceLocalSafeQuitIsolatedTestLegacyBootstrap {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $statePath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) 'recovery-legacy-bootstrap-worker.json'
    if (-not (Test-Path -LiteralPath $statePath)) {
        return @{ Stopped = $true; Reason = 'not_running'; Step = 'legacy_bootstrap' }
    }
    try {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    }
    catch {
        return @{ Stopped = $false; Reason = 'state_unreadable'; Step = 'legacy_bootstrap' }
    }
    $workerPid = 0
    if ($state.PSObject.Properties.Match('workerPid').Count -gt 0) {
        $workerPid = [int]$state.workerPid
    }
    if ($workerPid -le 0 -or -not (Test-ProcessAlive $workerPid)) {
        return @{ Stopped = $true; Reason = 'not_running'; Step = 'legacy_bootstrap' }
    }
    if (-not (Test-AceLocalSafeQuitProcessHasTestMarker -ProcId $workerPid)) {
        return @{ Stopped = $false; Reason = 'legacy_worker_not_test_owned'; Step = 'legacy_bootstrap' }
    }
    Stop-Process -Id $workerPid -ErrorAction Stop
    Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    return @{ Stopped = $true; Reason = 'test_graceful'; Step = 'legacy_bootstrap' }
}

function Stop-AceLocalSafeQuitIsolatedTestMemuraiWatch {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $pidPath = Get-AceLocalMemuraiWatchPidPath -AceRoot $AceRoot
    $pidVal = Read-AceLocalMemuraiWatchPid -PidPath $pidPath
    if (-not $pidVal) {
        return @{ Stopped = $true; Reason = 'not_running'; Step = 'memurai_watch' }
    }
    if (-not (Test-AceLocalSafeQuitProcessHasTestMarker -ProcId $pidVal)) {
        return @{ Stopped = $false; Reason = 'memurai_watch_not_test_owned'; Step = 'memurai_watch' }
    }
    Stop-Process -Id $pidVal -ErrorAction Stop
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    return @{ Stopped = $true; Reason = 'test_graceful'; Step = 'memurai_watch' }
}

function Test-AceLocalSafeQuitReadyForManagedStackStop {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)]$VerificationResult,
        [Parameter(Mandatory = $true)]$CaptureResult
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        return @{ Ok = $false; Reason = 'ace_root_not_isolated' }
    }
    if (-not $VerificationResult -or -not $VerificationResult.Ok) {
        return @{ Ok = $false; Reason = 'snapshot_verification_failed' }
    }
    $ptr = Get-AceLocalShutdownLastVerifiedSnapshot -AceRoot $AceRoot
    if (-not $ptr) {
        return @{ Ok = $false; Reason = 'last_verified_pointer_missing' }
    }
    if ($CaptureResult -and $CaptureResult.SnapshotId -and ([string]$ptr.snapshotId -ne [string]$CaptureResult.SnapshotId)) {
        return @{ Ok = $false; Reason = 'pointer_snapshot_mismatch' }
    }
    $session = ($LauncherSession -replace '\s', '').Trim()
    $regSession = Read-AceLocalManagedProcessesLauncherSession -AceRoot $AceRoot
    if ($regSession -and ($regSession -ne $session)) {
        return @{ Ok = $false; Reason = 'registry_session_mismatch' }
    }
    return @{ Ok = $true; Reason = 'ok' }
}

function Invoke-AceLocalSafeQuitManagedStackStop {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)]$VerificationResult,
        [Parameter(Mandatory = $true)]$CaptureResult,
        [switch]$TestSimulateStackStopThrow,
        [switch]$TestSimulatePartialStackStop
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        throw 'SAFE_QUIT_STACK_STOP_ACE_ROOT_NOT_ISOLATED'
    }
    $pre = Test-AceLocalSafeQuitReadyForManagedStackStop -AceRoot $AceRoot -LauncherSession $LauncherSession `
        -VerificationResult $VerificationResult -CaptureResult $CaptureResult
    if (-not $pre.Ok) {
        return @{
            Ok      = $false
            Outcome = 'preconditions_failed'
            Detail  = [string]$pre.Reason
            Steps   = @()
        }
    }
    $roots = Resolve-AceLocalSafeQuitRepoRoots -AceRoot $AceRoot
    if (-not $roots.Ok) {
        return @{
            Ok      = $false
            Outcome = 'repo_roots_invalid'
            Detail  = [string]$roots.Reason
            Steps   = @()
        }
    }
    $registry = Read-AceLocalSafeQuitManagedRegistry -AceRoot $AceRoot
    if (-not $registry) {
        return @{
            Ok      = $false
            Outcome = 'registry_missing'
            Detail  = 'managed-processes.json'
            Steps   = @()
        }
    }
    $session = ($LauncherSession -replace '\s', '').Trim()
    $steps = @()

    try {
        if ($TestSimulateStackStopThrow) {
            throw 'SAFE_QUIT_TEST_SIMULATED_STACK_STOP_FAILURE'
        }
        $ps = Stop-AceLocalSafeQuitIsolatedTestPeriodicSave -AceRoot $AceRoot
        $steps += $ps
        if (-not $ps.Stopped) {
            return @{ Ok = $false; Outcome = 'periodic_save_failed'; Detail = $ps.Reason; Steps = $steps }
        }

        $bs = Stop-AceLocalSafeQuitIsolatedTestLegacyBootstrap -AceRoot $AceRoot
        $steps += $bs
        if (-not $bs.Stopped) {
            return @{ Ok = $false; Outcome = 'legacy_bootstrap_failed'; Detail = $bs.Reason; Steps = $steps }
        }

        $actions = @(Get-AceLocalSafeQuitIsolatedManagedStopActions -Registry $registry -ExpectedLauncherSession $session)
        foreach ($act in $actions) {
            if ($TestSimulatePartialStackStop -and $act.Component -eq 'backend') {
                $steps += @{ Stopped = $false; Reason = 'simulated_partial'; Step = $act.Component }
                return @{ Ok = $false; Outcome = 'partial_stop'; Detail = 'backend_simulated_failure'; Steps = $steps }
            }
            if ($act.Action -eq 'skip') {
                $steps += @{ Stopped = $true; Reason = ('skip:' + $act.Reason); Step = $act.Component; Skipped = $true }
                $registry.components.($act.Component) = $null
                continue
            }
            $stopResult = Stop-AceLocalSafeQuitVerifiedTestProcessTree -RootPid ([int]$act.RootPid)
            $steps += @{
                Stopped = [bool]$stopResult.Stopped
                Reason  = [string]$stopResult.Reason
                Step    = $act.Component
                RootPid = [int]$act.RootPid
            }
            if (-not $stopResult.Stopped) {
                return @{
                    Ok      = $false
                    Outcome = 'managed_component_failed'
                    Detail  = ($act.Component + ':' + $stopResult.Reason)
                    Steps   = $steps
                }
            }
            $registry.components.($act.Component) = $null
        }

        $mw = Stop-AceLocalSafeQuitIsolatedTestMemuraiWatch -AceRoot $AceRoot
        $steps += $mw
        if (-not $mw.Stopped) {
            return @{ Ok = $false; Outcome = 'memurai_watch_failed'; Detail = $mw.Reason; Steps = $steps }
        }

        $registry.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
        if ($registry.components.PSObject.Properties.Match('recoveryPeriodicSave').Count -gt 0) {
            $registry.components.recoveryPeriodicSave = $null
        }
        Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $registry

        $stepLabels = @($steps | ForEach-Object { if ($_.Step) { [string]$_.Step } } | Where-Object { $_ } | Select-Object -Unique)
        return @{
            Ok      = $true
            Outcome = 'stopped'
            Detail  = ('steps=' + ($stepLabels -join ','))
            Steps   = $steps
        }
    }
    catch {
        return @{
            Ok      = $false
            Outcome = 'exception'
            Detail  = $_.Exception.Message
            Steps   = $steps
        }
    }
}

function Start-AceLocalSafeQuitTestSleepProcess {
    param([string]$Label = 'generic')
    $marker = $script:AceLocalSafeQuitTestProcessMarker
    $cmd = "Start-Sleep -Seconds 600 # $marker $Label"
    return Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') `
        -ArgumentList @('-NoProfile', '-Command', $cmd) -PassThru -WindowStyle Hidden
}

function Register-AceLocalSafeQuitTestManagedStack {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [switch]$IncludeLegacyBootstrap,
        [switch]$IncludeMemuraiWatch,
        [switch]$IncludePeriodicSave
    )
    $registry = Read-AceLocalSafeQuitManagedRegistry -AceRoot $AceRoot
    if (-not $registry) {
        $roots = Resolve-AceLocalSafeQuitRepoRoots -AceRoot $AceRoot
        $registry = New-AceLocalManagedRegistryShell -BackendRoot $roots.BackendRoot -FrontendRoot $roots.FrontendRoot `
            -LauncherSessionId $LauncherSession
    }
    $prefix = $script:AceLocalSafeQuitTestManagedLogPrefix
    foreach ($pair in @(
            @{ Name = 'frontend'; Label = 'frontend' }
            @{ Name = 'worker'; Label = 'worker' }
            @{ Name = 'backend'; Label = 'backend' }
        )) {
        $proc = Start-AceLocalSafeQuitTestSleepProcess -Label $pair.Label
        Start-Sleep -Milliseconds 200
        $registry.components.($pair.Name) = New-AceLocalManagedComponentTreeRecord `
            -RootPid $proc.Id -TreePids @($proc.Id) -LogPath ($prefix + $pair.Name) `
            -LauncherSession $LauncherSession
    }
    if ($IncludePeriodicSave) {
        $psProc = Start-AceLocalSafeQuitTestSleepProcess -Label 'periodic-save'
        Set-Content -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) 'recovery-periodic-save.pid') `
            -Value ([string]$psProc.Id) -Encoding ASCII
    }
    if ($IncludeLegacyBootstrap) {
        $bsProc = Start-AceLocalSafeQuitTestSleepProcess -Label 'legacy-bootstrap'
        Write-AceLocalRecoveryJsonAtomic -Path (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) 'recovery-legacy-bootstrap-worker.json') `
            -Object @{ schemaVersion = 1; workerPid = $bsProc.Id; launcherSession = $LauncherSession }
    }
    if ($IncludeMemuraiWatch) {
        $mwProc = Start-AceLocalSafeQuitTestSleepProcess -Label 'memurai-watch'
        Set-Content -LiteralPath (Get-AceLocalMemuraiWatchPidPath -AceRoot $AceRoot) -Value ([string]$mwProc.Id) -Encoding ASCII
    }
    Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $registry
    return $registry
}
