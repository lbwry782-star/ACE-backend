# ACE LOCAL — safe quit PERIODIC_SAVE_QUIESCE (5.19.4; TEMP/isolated only).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')
. (Join-Path $ScriptDir 'AceLocalShutdownGateLib.ps1')

$script:AceLocalSafeQuitPeriodicSaveQuiesceInitFileName = 'recovery-periodic-save.stop.safe-quit-init.json'

function Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalSafeQuitPeriodicSaveQuiesceInitFileName
}

function Read-AceLocalManagedProcessesLauncherSession {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $regPath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) 'managed-processes.json'
    if (-not (Test-Path -LiteralPath $regPath)) {
        return $null
    }
    try {
        $reg = Get-Content -LiteralPath $regPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return [string]$reg.launcherSession
    }
    catch {
        return $null
    }
}

function Test-AceLocalSafeQuitPeriodicSaveLockSessionBinding {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$ExpectedLauncherSession
    )
    $path = Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) {
        return @{ Ok = $true; Reason = 'no_lock' }
    }
    try {
        $existing = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        return @{ Ok = $false; Reason = 'lock_unreadable' }
    }
    $lockSession = ''
    if ($existing.PSObject.Properties.Match('launcherSession').Count -gt 0) {
        $lockSession = [string]$existing.launcherSession
    }
    if ($lockSession -and ($lockSession -ne $ExpectedLauncherSession)) {
        $livePid = $null
        if ($existing.PSObject.Properties.Match('pid').Count -gt 0) {
            $livePid = [int]$existing.pid
        }
        $procAlive = $false
        if ($livePid) {
            $procAlive = $null -ne (Get-Process -Id $livePid -ErrorAction SilentlyContinue)
        }
        return @{
            Ok         = $false
            Reason     = if ($procAlive) { 'lock_live_old_session' } else { 'lock_stale_old_session' }
            LockSession = $lockSession
            ProcAlive  = $procAlive
        }
    }
    return @{ Ok = $true; Reason = 'session_ok' }
}

function Test-AceLocalSafeQuitPeriodicSaveLockSafeForQuiesce {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$ExpectedLauncherSession
    )
    $binding = Test-AceLocalSafeQuitPeriodicSaveLockSessionBinding -AceRoot $AceRoot `
        -ExpectedLauncherSession $ExpectedLauncherSession
    if (-not $binding.Ok) {
        return @{
            Safe       = $false
            Reason     = [string]$binding.Reason
            Assessment = @{ Classification = 'stale_old_session'; Reason = [string]$binding.Reason }
        }
    }
    $assessment = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $AceRoot `
        -ExpectedLauncherSession $ExpectedLauncherSession
    $class = [string]$assessment.Classification
    switch ($class) {
        'none' { return @{ Safe = $true; Assessment = $assessment } }
        'live' { return @{ Safe = $true; Assessment = $assessment; ActiveWrite = $true } }
        'stale_dead_pid' { return @{ Safe = $true; Assessment = $assessment; ActiveWrite = $false } }
        'stale_old_session' {
            return @{
                Safe       = $false
                Reason     = 'lock_live_old_session'
                Assessment = $assessment
            }
        }
        'pid_reuse_suspected' {
            return @{
                Safe       = $false
                Reason     = 'lock_pid_reuse_suspected'
                Assessment = $assessment
            }
        }
        default {
            return @{
                Safe       = $false
                Reason     = ('lock_unsafe:' + [string]$assessment.Reason)
                Assessment = $assessment
            }
        }
    }
}

function Request-AceLocalSafeQuitPeriodicSaveStop {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)][string]$SafeQuitAttemptId
    )
    Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $AceRoot
    $initPath = Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath -AceRoot $AceRoot
    Write-AceLocalRecoveryJsonAtomic -Path $initPath -Object @{
        schemaVersion     = 1
        launcherSession   = $LauncherSession
        safeQuitAttemptId = $SafeQuitAttemptId
        writtenAt         = (Get-Date).ToUniversalTime().ToString('o')
    }
}

function Reset-AceLocalSafeQuitPeriodicSaveQuiesce {
    <#
      Clears stop-request only when the safe-quit init marker matches (TEMP/isolated retry).
      Never removes checkpoint lock files.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$SafeQuitAttemptId = '',
        [switch]$ForceIfInitMatchesAttemptOnly
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        throw 'SAFE_QUIT_QUIESCE_RESET_REQUIRES_TEMP_ACE_ROOT'
    }
    $initPath = Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $initPath)) {
        return @{ Cleared = $false; Reason = 'no_init_marker' }
    }
    try {
        $init = Get-Content -LiteralPath $initPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        return @{ Cleared = $false; Reason = 'init_unreadable' }
    }
    $attempt = [string]$init.safeQuitAttemptId
    if ($ForceIfInitMatchesAttemptOnly -and $SafeQuitAttemptId -and ($attempt -ne $SafeQuitAttemptId)) {
        return @{ Cleared = $false; Reason = 'init_attempt_mismatch' }
    }
    Clear-AceLocalRecoveryPeriodicSaveStopRequest -AceRoot $AceRoot
    Remove-Item -LiteralPath $initPath -Force -ErrorAction SilentlyContinue
    return @{ Cleared = $true; Reason = 'ok' }
}

function Invoke-AceLocalSafeQuitPeriodicSaveQuiesce {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$WaitSeconds = 30,
        [int]$PollMilliseconds = 250,
        [switch]$TestSimulateWaitThrow
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        throw 'SAFE_QUIT_QUIESCE_ACE_ROOT_NOT_ISOLATED'
    }
    $session = ($LauncherSession -replace '\s', '').Trim()
    if (-not $session) {
        return @{
            Ok      = $false
            Outcome = 'session_invalid'
            Detail  = 'launcher_session_empty'
        }
    }

    $registrySession = Read-AceLocalManagedProcessesLauncherSession -AceRoot $AceRoot
    if ($registrySession -and ($registrySession -ne $session)) {
        return @{
            Ok      = $false
            Outcome = 'session_mismatch'
            Detail  = ('managed_processes=' + $registrySession)
        }
    }

    if (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot) {
        $initPath = Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath -AceRoot $AceRoot
        if (-not (Test-Path -LiteralPath $initPath)) {
            return @{
                Ok      = $false
                Outcome = 'stop_foreign'
                Detail  = 'stop_requested_without_safe_quit_init'
            }
        }
        try {
            $init = Get-Content -LiteralPath $initPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ([string]$init.launcherSession -ne $session) {
                return @{
                    Ok      = $false
                    Outcome = 'stop_foreign'
                    Detail  = 'stop_init_session_mismatch'
                }
            }
        }
        catch {
            return @{
                Ok      = $false
                Outcome = 'stop_foreign'
                Detail  = 'stop_init_unreadable'
            }
        }
    }

    $preLock = Test-AceLocalSafeQuitPeriodicSaveLockSafeForQuiesce -AceRoot $AceRoot -ExpectedLauncherSession $session
    if (-not $preLock.Safe) {
        return @{
            Ok      = $false
            Outcome = 'lock_unsafe'
            Detail  = [string]$preLock.Reason
            Lock    = $preLock.Assessment
        }
    }

    $attemptId = [guid]::NewGuid().ToString('N')
    $stopWasAlreadyOurs = Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot
    if (-not $stopWasAlreadyOurs) {
        Request-AceLocalSafeQuitPeriodicSaveStop -AceRoot $AceRoot -LauncherSession $session -SafeQuitAttemptId $attemptId
    }
    else {
        $attemptId = (Get-Content -LiteralPath (Get-AceLocalSafeQuitPeriodicSaveQuiesceInitPath -AceRoot $AceRoot) -Raw | ConvertFrom-Json).safeQuitAttemptId
    }

    if (-not (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot)) {
        return @{
            Ok      = $false
            Outcome = 'stop_not_recorded'
            Detail  = 'stop_file_missing_after_request'
        }
    }

    $preWrite = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $AceRoot -LauncherSession $session `
        -GuardBackendRoot (Split-Path -Parent $ScriptDir) -SkipLiveAssessment
    if ($preWrite.Allowed) {
        return @{
            Ok      = $false
            Outcome = 'pre_write_not_blocked'
            Detail  = ('phase=' + [string]$preWrite.Phase)
        }
    }
    $reasons = @($preWrite.Reasons | ForEach-Object { [string]$_ })
    if ($reasons -notcontains 'stop_requested') {
        return @{
            Ok      = $false
            Outcome = 'pre_write_not_blocked'
            Detail  = ('reasons=' + ($reasons -join ','))
        }
    }

    $hadLiveLock = ([string]$preLock.Assessment.Classification -eq 'live')
    if ($hadLiveLock) {
        $deadline = (Get-Date).AddSeconds($WaitSeconds)
        do {
            if ($TestSimulateWaitThrow) {
                throw 'SAFE_QUIT_TEST_SIMULATED_QUIESCE_WAIT_FAILURE'
            }
            $assessment = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $AceRoot `
                -ExpectedLauncherSession $session
            $class = [string]$assessment.Classification
            if ($class -ne 'live') {
                break
            }
            Start-Sleep -Milliseconds $PollMilliseconds
        } while ((Get-Date) -lt $deadline)

        $finalLock = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $AceRoot `
            -ExpectedLauncherSession $session
        if ([string]$finalLock.Classification -eq 'live') {
            return @{
                Ok          = $false
                Outcome     = 'wait_timeout'
                Detail      = ('lock_pid=' + [string]$finalLock.LockPid)
                AttemptId   = $attemptId
                Lock        = $finalLock
            }
        }
    }

    $postLock = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $AceRoot -ExpectedLauncherSession $session
    if ([string]$postLock.Classification -eq 'live') {
        return @{
            Ok        = $false
            Outcome   = 'still_writing'
            Detail    = 'live_lock_after_wait'
            AttemptId = $attemptId
            Lock      = $postLock
        }
    }

    return @{
        Ok              = $true
        Outcome         = 'quiesced'
        Detail          = if ($hadLiveLock) { 'waited_for_checkpoint' } else { 'idle_no_live_lock' }
        AttemptId       = $attemptId
        StopRequested   = $true
        FinalLockClass  = [string]$postLock.Classification
    }
}
