# ACE LOCAL — recovery periodic save service (phase 4; not wired to Start-AceLocal).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryCheckpointLib.ps1')

$script:AceLocalRecoveryPeriodicSaveLockFileName = 'recovery-periodic-save.lock'
$script:AceLocalRecoveryPeriodicSaveStopFileName = 'recovery-periodic-save.stop'
$script:AceLocalRecoveryPeriodicSavePidFileName = 'recovery-periodic-save.pid'
$script:AceLocalRecoveryPeriodicSaveGuardFileName = 'recovery-periodic-save-guard.json'
$script:AceLocalRecoveryPeriodicSaveServiceStateFileName = 'recovery-periodic-save-service.json'
$script:AceLocalRecoveryPeriodicSaveDeferFileName = 'recovery-periodic-save-defer-until.json'
$script:AceLocalRecoveryPeriodicSaveActivationSnapshotFileName = 'recovery-periodic-save-activation-snapshot.json'
$script:AceLocalRecoveryPeriodicSaveRuntimeFileName = 'recovery-periodic-save-runtime.json'
$script:AceLocalRecoveryPeriodicSaveTickInProcess = $false
$script:AceLocalRecoveryPeriodicSavePreWriteReassessSeconds = 5

function Get-AceLocalRecoveryPeriodicSaveStatePath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveServiceStateFileName
}

function Get-AceLocalRecoveryPeriodicSaveLockPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveLockFileName
}

function Get-AceLocalRecoveryPeriodicSaveStopPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveStopFileName
}

function Get-AceLocalRecoveryPeriodicSaveGuardPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveGuardFileName
}

function Test-AceLocalRecoveryPeriodicSaveStopRequested {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return (Test-Path -LiteralPath (Get-AceLocalRecoveryPeriodicSaveStopPath -AceRoot $AceRoot))
}

function Request-AceLocalRecoveryPeriodicSaveStop {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalRecoveryPeriodicSaveStopPath -AceRoot $AceRoot
    $dir = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Set-Content -LiteralPath $path -Value ((Get-Date).ToUniversalTime().ToString('o')) -Encoding UTF8
}

function Clear-AceLocalRecoveryPeriodicSaveStopRequest {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    Remove-Item -LiteralPath (Get-AceLocalRecoveryPeriodicSaveStopPath -AceRoot $AceRoot) -Force -ErrorAction SilentlyContinue
}

function Wait-AceLocalRecoveryPeriodicSaveInterTickDelay {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][int]$TotalSeconds
    )
    if ($TotalSeconds -le 0) {
        return
    }
    $deadline = (Get-Date).AddSeconds($TotalSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
}

function Read-AceLocalRecoveryPeriodicSaveGuard {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        $GuardOverride = $null
    )
    if ($null -ne $GuardOverride) {
        return $GuardOverride
    }
    $path = Get-AceLocalRecoveryPeriodicSaveGuardPath -AceRoot $AceRoot
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

function Resolve-AceLocalRecoveryGuardPython {
    param([Parameter(Mandatory = $true)][string]$BackendRoot)
    $candidates = @(
        (Join-Path $BackendRoot '.venv312\Scripts\python.exe'),
        (Join-Path $BackendRoot 'venv\Scripts\python.exe'),
        (Join-Path $BackendRoot '.venv\Scripts\python.exe')
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return $path
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return $null
}

function Invoke-AceLocalRecoveryPeriodicSaveGuardAssessment {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        $GuardOverride = $null,
        [switch]$SkipLiveAssessment
    )
    if ($null -ne $GuardOverride) {
        $guard = $GuardOverride
        $block = $false
        $allow = $false
        $reason = 'guard_override'
        if ($guard -is [System.Collections.IDictionary]) {
            if ($guard.Contains('blockCheckpoint')) { $block = [bool]$guard['blockCheckpoint'] }
            if ($guard.Contains('allowCheckpoint')) { $allow = [bool]$guard['allowCheckpoint'] }
            if ($guard.Contains('reason')) { $reason = [string]$guard['reason'] }
        }
        else {
            if ($guard.PSObject.Properties.Match('blockCheckpoint').Count -gt 0) {
                $block = [bool]$guard.blockCheckpoint
            }
            if ($guard.PSObject.Properties.Match('allowCheckpoint').Count -gt 0) {
                $allow = [bool]$guard.allowCheckpoint
            }
            if ($guard.PSObject.Properties.Match('reason').Count -gt 0) {
                $reason = [string]$guard.reason
            }
        }
        if ($block) {
            return @{ Allowed = $false; Blocked = $true; Reason = $reason; Source = 'test_guard'; Confidence = 'high' }
        }
        if ($allow) {
            return @{ Allowed = $true; Blocked = $false; Reason = ''; Source = 'test_guard'; Confidence = 'high' }
        }
    }

    if ($SkipLiveAssessment) {
        return @{ Allowed = $true; Blocked = $false; Reason = ''; Source = 'skipped_live'; Confidence = 'low' }
    }

    $py = Resolve-AceLocalRecoveryGuardPython -BackendRoot $BackendRoot
    if (-not $py) {
        return @{
            Allowed    = $false
            Blocked    = $true
            Reason     = 'assessment_python_missing'
            Source     = 'unknown'
            Confidence = 'unknown'
        }
    }
    $modulePath = Join-Path $BackendRoot 'engine\ace_local_dev_recovery_periodic_save_guard.py'
    if (-not (Test-Path -LiteralPath $modulePath)) {
        return @{
            Allowed    = $false
            Blocked    = $true
            Reason     = 'assessment_module_missing'
            Source     = 'unknown'
            Confidence = 'unknown'
        }
    }
    $prevAceRoot = $env:ACE_ORI_ACE_ROOT
    $env:ACE_ORI_ACE_ROOT = $AceRoot
    try {
        $output = & $py $modulePath --ace-root $AceRoot 2>&1 | Out-String
        $line = ($output -split "`n" | Where-Object { $_.Trim() -match '^\{' } | Select-Object -Last 1)
        if (-not $line) {
            return @{
                Allowed    = $false
                Blocked    = $true
                Reason     = 'assessment_output_invalid'
                Source     = 'unknown'
                Confidence = 'unknown'
            }
        }
        $payload = $line | ConvertFrom-Json
        $allowed = [bool]$payload.allowed
        return @{
            Allowed    = $allowed
            Blocked    = (-not $allowed)
            Reason     = [string]$payload.reason
            Source     = [string]$payload.source
            Confidence = $(if ($allowed) { 'high' } else { 'high' })
        }
    }
    catch {
        return @{
            Allowed    = $false
            Blocked    = $true
            Reason     = 'assessment_invoke_failed'
            Source     = 'unknown'
            Confidence = 'unknown'
        }
    }
    finally {
        if ($null -eq $prevAceRoot) {
            Remove-Item Env:ACE_ORI_ACE_ROOT -ErrorAction SilentlyContinue
        }
        else {
            $env:ACE_ORI_ACE_ROOT = $prevAceRoot
        }
    }
}

function Test-AceLocalRecoveryPeriodicSaveCriticalBlocked {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        $GuardOverride = $null,
        [switch]$SkipLiveAssessment
    )
    $assessment = Invoke-AceLocalRecoveryPeriodicSaveGuardAssessment -AceRoot $AceRoot -BackendRoot $BackendRoot `
        -GuardOverride $GuardOverride -SkipLiveAssessment:$SkipLiveAssessment
    if ($assessment.Blocked) {
        return @{
            Blocked    = $true
            Reason     = [string]$assessment.Reason
            Source     = [string]$assessment.Source
            Confidence = [string]$assessment.Confidence
        }
    }
    return @{ Blocked = $false; Reason = ''; Source = [string]$assessment.Source; Confidence = [string]$assessment.Confidence }
}

function Get-AceLocalRecoveryPeriodicSaveLockAssessment {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$ExpectedLauncherSession = ''
    )
    $path = Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) {
        return @{ Classification = 'none'; Reason = 'no_lock'; LockPid = $null }
    }
    try {
        $existing = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        return @{ Classification = 'unknown'; Reason = 'lock_unreadable'; LockPid = $null }
    }

    $lockPid = $null
    if ($existing.PSObject.Properties.Match('pid').Count -gt 0) {
        $lockPid = [int]$existing.pid
    }
    $lockSession = ''
    if ($existing.PSObject.Properties.Match('launcherSession').Count -gt 0) {
        $lockSession = [string]$existing.launcherSession
    }
    $startedAt = $null
    if ($existing.PSObject.Properties.Match('startedAt').Count -gt 0) {
        $startedAt = [DateTime]::Parse([string]$existing.startedAt, $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
    }

    if (-not $lockPid) {
        return @{ Classification = 'unknown'; Reason = 'lock_missing_pid'; LockPid = $null }
    }

    $proc = Get-Process -Id $lockPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        return @{ Classification = 'stale_dead_pid'; Reason = 'pid_not_running'; LockPid = $lockPid }
    }

    if ($startedAt) {
        try {
            $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$lockPid" -ErrorAction Stop
            $procStart = [Management.ManagementDateTimeConverter]::ToDateTime($cim.CreationDate).ToUniversalTime()
            if ($procStart -gt $startedAt.AddSeconds(2)) {
                return @{ Classification = 'pid_reuse_suspected'; Reason = 'process_newer_than_lock'; LockPid = $lockPid }
            }
        }
        catch {
            return @{ Classification = 'live'; Reason = 'process_running_start_unverified'; LockPid = $lockPid }
        }
    }

    if ($ExpectedLauncherSession -and $lockSession -and ($lockSession -ne $ExpectedLauncherSession)) {
        return @{ Classification = 'stale_old_session'; Reason = 'launcher_session_mismatch'; LockPid = $lockPid }
    }

    return @{ Classification = 'live'; Reason = 'save_process_active'; LockPid = $lockPid }
}

function Enter-AceLocalRecoveryPeriodicSaveLock {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$LauncherSession = ''
    )
    $path = Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $AceRoot
    $dir = Split-Path -Parent $path
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    if (Test-Path -LiteralPath $path) {
        $assessment = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $AceRoot `
            -ExpectedLauncherSession $LauncherSession
        switch ([string]$assessment.Classification) {
            'none' { }
            'live' {
                return @{ Acquired = $false; Reason = 'lock_live'; Classification = 'live' }
            }
            'stale_dead_pid' {
                Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
            }
            'stale_old_session' {
                $liveCheck = Get-Process -Id ([int]$assessment.LockPid) -ErrorAction SilentlyContinue
                if ($liveCheck) {
                    return @{
                        Acquired       = $false
                        Reason         = 'lock_live_old_session'
                        Classification = 'stale_old_session'
                    }
                }
                Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
            }
            default {
                return @{
                    Acquired       = $false
                    Reason         = ('lock_' + [string]$assessment.Reason)
                    Classification = [string]$assessment.Classification
                }
            }
        }
    }

    $payload = [ordered]@{
        pid              = $PID
        startedAt        = (Get-Date).ToUniversalTime().ToString('o')
        launcherSession  = $LauncherSession
        aceRoot          = $AceRoot
    }
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object $payload
    return @{ Acquired = $true; Reason = 'ok'; Classification = 'acquired' }
}

function Exit-AceLocalRecoveryPeriodicSaveLock {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    Remove-Item -LiteralPath (Get-AceLocalRecoveryPeriodicSaveLockPath -AceRoot $AceRoot) -Force -ErrorAction SilentlyContinue
}

function Get-AceLocalRecoveryPeriodicSaveRuntimePath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveRuntimeFileName
}

function Write-AceLocalRecoveryPeriodicSaveRuntimeConfig {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][int]$BackendPort,
        [Parameter(Mandatory = $true)][string]$LauncherSession
    )
    $path = Get-AceLocalRecoveryPeriodicSaveRuntimePath -AceRoot $AceRoot
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object @{
        schemaVersion   = 1
        backendPort     = $BackendPort
        launcherSession = $LauncherSession
        writtenAt       = (Get-Date).ToUniversalTime().ToString('o')
    }
}

function Get-AceLocalRecoveryPeriodicSaveActivationSnapshotPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveActivationSnapshotFileName
}

function Write-AceLocalRecoveryPeriodicSaveActivationSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)][string]$CoordinatorOutcome,
        [Parameter(Mandatory = $true)][bool]$AllowPeriodicSaveService
    )
    $path = Get-AceLocalRecoveryPeriodicSaveActivationSnapshotPath -AceRoot $AceRoot
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object @{
        schemaVersion            = 1
        writtenAt                = (Get-Date).ToUniversalTime().ToString('o')
        launcherSession          = $LauncherSession
        coordinatorOutcome       = $CoordinatorOutcome
        allowPeriodicSaveService = $AllowPeriodicSaveService
    }
}

function Invoke-AceLocalRecoveryPeriodicSavePreWriteGate {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)][string]$GuardBackendRoot,
        [switch]$SkipLiveAssessment,
        [switch]$LiveAssessmentOnly,
        [string]$PriorAssessmentAt = '',
        $GuardOverride = $null
    )
    if (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot) {
        return @{
            Allowed = $false
            Reasons = @('stop_requested')
            Phase   = 'pre_write'
        }
    }
    if ($null -ne $GuardOverride) {
        if ($GuardOverride -is [System.Collections.IDictionary]) {
            if ($GuardOverride.Contains('blockCheckpoint') -and [bool]$GuardOverride['blockCheckpoint']) {
                return @{
                    Allowed = $false
                    Reasons = @([string]$GuardOverride['reason'])
                    Phase   = 'pre_write_guard_override'
                }
            }
        }
        elseif ($GuardOverride.PSObject.Properties.Match('blockCheckpoint').Count -gt 0 -and [bool]$GuardOverride.blockCheckpoint) {
            return @{
                Allowed = $false
                Reasons = @([string]$GuardOverride.reason)
                Phase   = 'pre_write_guard_override'
            }
        }
    }
    if ($SkipLiveAssessment) {
        $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
        if (-not $ledger -or ([string]$ledger.launcherSession -ne $LauncherSession)) {
            return @{ Allowed = $false; Reasons = @('ledger_launcher_session_mismatch'); Phase = 'pre_write_test_skip' }
        }
        $phase = if ($LiveAssessmentOnly) { 'pre_write_test_skip_reassess' } else { 'pre_write_test_skip' }
        return @{ Allowed = $true; Reasons = @(); Phase = $phase; AssessedAt = (Get-Date).ToUniversalTime().ToString('o') }
    }
    $backendRoot = (Resolve-Path -LiteralPath $GuardBackendRoot).ProviderPath
    $modulePath = Join-Path $backendRoot 'engine\ace_local_dev_recovery_periodic_save_pre_write.py'
    if (-not (Test-Path -LiteralPath $modulePath)) {
        return @{ Allowed = $false; Reasons = @('pre_write_module_missing'); Phase = 'pre_write' }
    }
    $runnerPy = @'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_recovery_periodic_save_pre_write import evaluate_pre_checkpoint_write_gate

ace = Path(sys.argv[2])
launcher = sys.argv[3]
live_only = sys.argv[4] == '1'
prior = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
payload = evaluate_pre_checkpoint_write_gate(
    ace,
    launcher,
    live_assessment_only=live_only,
    prior_assessment_at_iso=prior,
)
print(json.dumps(payload, ensure_ascii=False))
'@
    $runnerPath = Join-Path $env:TEMP ("ace-pre-write-gate-{0}.py" -f [guid]::NewGuid().ToString('N'))
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($runnerPath, $runnerPy, $utf8NoBom)
    try {
        $liveFlag = if ($LiveAssessmentOnly) { '1' } else { '0' }
        $priorArg = if ($PriorAssessmentAt) { $PriorAssessmentAt } else { '' }
        $output = python $runnerPath $backendRoot $AceRoot $LauncherSession $liveFlag $priorArg 2>&1 | Out-String
        $line = ($output -split "`n" | Where-Object { $_.Trim() -match '^\{' } | Select-Object -Last 1)
        if (-not $line) {
            return @{ Allowed = $false; Reasons = @('pre_write_output_invalid'); Phase = 'pre_write' }
        }
        $payload = $line | ConvertFrom-Json
        $reasons = @()
        if ($payload.reasons) {
            $reasons = @($payload.reasons | ForEach-Object { [string]$_ })
        }
        return @{
            Allowed    = [bool]$payload.allowed
            Reasons    = $reasons
            Phase      = if ($LiveAssessmentOnly) { 'pre_write_reassess' } else { 'pre_write' }
            AssessedAt = [string]$payload.assessedAt
            Payload    = $payload
        }
    }
    finally {
        Remove-Item -LiteralPath $runnerPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-AceLocalRecoveryPeriodicSaveDeferPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSaveDeferFileName
}

function Test-AceLocalRecoveryPeriodicSaveCheckpointDeferred {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [DateTime]$NowUtc = [DateTime]::MinValue
    )
    if ($NowUtc -eq [DateTime]::MinValue) {
        $NowUtc = (Get-Date).ToUniversalTime()
    }
    $path = Get-AceLocalRecoveryPeriodicSaveDeferPath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) {
        return @{ Deferred = $false; Reason = '' }
    }
    try {
        $payload = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        $untilIso = [string]$payload.deferUntil
        if ([string]::IsNullOrWhiteSpace($untilIso)) {
            return @{ Deferred = $false; Reason = '' }
        }
        $until = [DateTime]::Parse($untilIso, $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
        if ($NowUtc -lt $until) {
            return @{ Deferred = $true; Reason = [string]$payload.reason }
        }
    }
    catch {
        return @{ Deferred = $false; Reason = '' }
    }
    Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    return @{ Deferred = $false; Reason = '' }
}

function Set-AceLocalRecoveryPeriodicSaveCheckpointDefer {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][DateTime]$DeferUntilUtc,
        [string]$Reason = 'io_load'
    )
    $path = Get-AceLocalRecoveryPeriodicSaveDeferPath -AceRoot $AceRoot
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object @{
        deferUntil = $DeferUntilUtc.ToUniversalTime().ToString('o')
        reason     = $Reason
    }
}

function Write-AceLocalRecoveryPeriodicSaveServiceState {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][hashtable]$State
    )
    $path = Get-AceLocalRecoveryPeriodicSaveStatePath -AceRoot $AceRoot
    $payload = [ordered]@{
        schemaVersion = 1
        updatedAt     = (Get-Date).ToUniversalTime().ToString('o')
    }
    foreach ($key in $State.Keys) {
        $payload[$key] = $State[$key]
    }
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object $payload
}

function Invoke-AceLocalRecoveryPeriodicSaveTick {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [DateTime]$NowUtc = [DateTime]::MinValue,
        [string]$CheckpointsRootOverride = '',
        $CriticalGuardOverride = $null,
        [scriptblock]$CheckpointInvoker = $null,
        [switch]$SkipLock,
        [switch]$SkipLiveAssessment,
        [string]$GuardBackendRoot = '',
        [switch]$BackgroundMode
    )
    $guardBackend = if ($GuardBackendRoot) { $GuardBackendRoot } else { $BackendRoot }
    if ($NowUtc -eq [DateTime]::MinValue) {
        $NowUtc = (Get-Date).ToUniversalTime()
    }

    $result = @{
        Ok               = $true
        Outcome          = 'noop'
        DecisionAction   = 'none'
        CheckpointStatus = ''
        Reasons          = @()
    }

    if ($script:AceLocalRecoveryPeriodicSaveTickInProcess) {
        $result.Outcome = 'skipped_reentrant'
        $result.Reasons = @('tick_already_in_progress')
        return $result
    }
    $script:AceLocalRecoveryPeriodicSaveTickInProcess = $true
    try {

    if (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot) {
        $result.Outcome = 'stop_requested'
        return $result
    }

    if ($BackgroundMode -and $SkipLiveAssessment) {
        $result.Outcome = 'blocked_background_assessment_bypass'
        $result.Reasons = @('skip_live_assessment_forbidden_in_background_mode')
        return $result
    }

    if ($null -ne $CriticalGuardOverride) {
        $critical = Test-AceLocalRecoveryPeriodicSaveCriticalBlocked -AceRoot $AceRoot -BackendRoot $guardBackend `
            -GuardOverride $CriticalGuardOverride -SkipLiveAssessment
        if ($critical.Blocked) {
            if ([string]$critical.Confidence -eq 'unknown' -or [string]$critical.Reason -like 'assessment_*') {
                $result.Outcome = 'blocked_assessment_unknown'
            }
            else {
                $result.Outcome = 'blocked_critical'
            }
            $result.Reasons = @([string]$critical.Reason)
            $result.AssessmentSource = [string]$critical.Source
            return $result
        }
    }

    if (-not $SkipLock) {
        $lock = Enter-AceLocalRecoveryPeriodicSaveLock -AceRoot $AceRoot -LauncherSession $LauncherSession
        if (-not $lock.Acquired) {
            if ([string]$lock.Classification -eq 'live') {
                $result.Outcome = 'skipped_lock_live'
            }
            elseif ([string]$lock.Reason -like 'lock_*') {
                $result.Outcome = 'skipped_lock_unknown'
            }
            else {
                $result.Outcome = 'skipped_concurrent'
            }
            $result.Reasons = @([string]$lock.Reason)
            $result.LockClassification = [string]$lock.Classification
            return $result
        }
    }

    try {
        $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
        $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
        if (-not $valid.Ok) {
            $result.Outcome = 'ledger_invalid'
            $result.Reasons = @([string]$valid.Reason)
            return $result
        }

        if ($BackgroundMode -and ([string]$ledger.launcherSession -ne $LauncherSession)) {
            $result.Outcome = 'blocked_launcher_session'
            $result.Reasons = @('ledger_launcher_session_mismatch')
            return $result
        }

        $scan = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
        $hasChanges = Test-AceLocalRecoveryWorkspaceHasWorkingTreeChanges -WorkspaceScan $scan
        $prevFp = [string]$ledger.lastDirtyFingerprint
        $fp = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scan
        $contentChanged = Test-AceLocalRecoveryWorkspaceDirtyChanged -PreviousFingerprint $prevFp -CurrentFingerprint $fp

        $decision = Get-AceLocalRecoveryPeriodicSaveTimingDecision -Ledger $ledger -NowUtc $NowUtc `
            -WorkspaceDirtyChanged $contentChanged -HasWorkingTreeChanges $hasChanges
        $result.DecisionAction = [string]$decision.Action
        $result.Reasons = @($decision.Reasons)

        if ($decision.Action -eq $script:AceLocalRecoveryCheckpointActionDirtyScan) {
            Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $AceRoot -DirtyFingerprint $fp `
                -DirtyScanAt $NowUtc.ToString('o') | Out-Null
            $result.Outcome = 'dirty_scan'
            return $result
        }

        if ($decision.Action -in @(
                $script:AceLocalRecoveryCheckpointActionLightCheckpoint,
                $script:AceLocalRecoveryCheckpointActionFullCheckpoint)) {

            $preWrite = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $AceRoot `
                -LauncherSession $LauncherSession -GuardBackendRoot $guardBackend `
                -SkipLiveAssessment:$SkipLiveAssessment -GuardOverride $CriticalGuardOverride
            if (-not $preWrite.Allowed) {
                if ($preWrite.Reasons -contains 'stop_requested') {
                    $result.Outcome = 'stop_requested'
                }
                elseif ($preWrite.Phase -eq 'pre_write_guard_override' -or ($preWrite.Reasons | Where-Object { $_ -like 'critical_*' -or $_ -like '*operation*' })) {
                    $result.Outcome = 'blocked_critical'
                }
                elseif ($preWrite.Reasons -contains 'coordinator_evidence_missing' `
                    -or ($preWrite.Reasons | Where-Object { $_ -like 'coordinator_*' -or $_ -like 'periodic_save_*' })) {
                    $result.Outcome = 'blocked_coordinator_gate'
                }
                elseif ($preWrite.Reasons -contains 'ledger_launcher_session_mismatch') {
                    $result.Outcome = 'blocked_launcher_session'
                }
                elseif ($preWrite.Reasons | Where-Object { $_ -like 'backend_probe*' -or $_ -like 'probe_url_*' }) {
                    $result.Outcome = 'blocked_backend_unreachable'
                }
                else {
                    $result.Outcome = 'blocked_pre_write'
                }
                $result.Reasons += @($preWrite.Reasons)
                return $result
            }

            $defer = Test-AceLocalRecoveryPeriodicSaveCheckpointDeferred -AceRoot $AceRoot -NowUtc $NowUtc
            if ($defer.Deferred) {
                $result.Outcome = 'checkpoint_deferred'
                $result.Reasons += [string]$defer.Reason
                return $result
            }

            if (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot) {
                $result.Outcome = 'stop_requested'
                return $result
            }

            $reassess = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $AceRoot `
                -LauncherSession $LauncherSession -GuardBackendRoot $guardBackend `
                -SkipLiveAssessment:$SkipLiveAssessment -LiveAssessmentOnly `
                -PriorAssessmentAt ([string]$preWrite.AssessedAt) -GuardOverride $CriticalGuardOverride
            if (-not $reassess.Allowed) {
                if ($reassess.Reasons -contains 'stop_requested') {
                    $result.Outcome = 'stop_requested'
                }
                elseif ($reassess.Reasons | Where-Object { $_ -like 'backend_probe*' -or $_ -like 'probe_url_*' }) {
                    $result.Outcome = 'blocked_backend_unreachable'
                }
                else {
                    $result.Outcome = 'blocked_pre_write_reassess'
                }
                $result.Reasons += @($reassess.Reasons)
                return $result
            }

            if ($CheckpointInvoker) {
                $cp = & $CheckpointInvoker $decision.Action
            }
            else {
                $cp = Invoke-AceLocalRecoveryCheckpointFromDecision -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
                    -AceRoot $AceRoot -LauncherSession $LauncherSession -DecisionAction $decision.Action `
                    -CheckpointsRootOverride $CheckpointsRootOverride
            }

            $result.CheckpointStatus = [string]$cp.Status
            if ($cp.Skipped) {
                $result.Outcome = 'checkpoint_skipped'
                $result.Reasons += [string]$cp.Reason
                return $result
            }
            if ($cp.Ok -and [string]$cp.Status -eq 'VERIFIED') {
                Update-AceLocalRecoveryLedgerTimingAfterDecision -AceRoot $AceRoot -DecisionAction $decision.Action `
                    -NowIso $NowUtc.ToString('o') | Out-Null
                Update-AceLocalRecoveryLedgerDirtyScan -AceRoot $AceRoot -DirtyFingerprint $fp `
                    -DirtyScanAt $NowUtc.ToString('o') | Out-Null
                $result.Outcome = 'checkpoint_verified'
                $result.CheckpointId = [string]$cp.CheckpointId
                $retention = Get-AceLocalRecoveryCheckpointRetentionPlan -AceRoot $AceRoot
                $result.VerifiedCount = [int]$retention.VerifiedCount
                return $result
            }

            $result.Outcome = 'checkpoint_incomplete'
            $result.Reasons += @($cp.Errors)
            return $result
        }

        $result.Outcome = 'noop'
        return $result
    }
    finally {
        if (-not $SkipLock) {
            Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $AceRoot
        }
    }
    }
    finally {
        $script:AceLocalRecoveryPeriodicSaveTickInProcess = $false
    }
}

function Start-AceLocalRecoveryPeriodicSaveLoop {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$TickIntervalSeconds = 30,
        [string]$CheckpointsRootOverride = '',
        [switch]$BackgroundMode,
        [string]$SpawnToken = '',
        [string]$GuardBackendRoot = ''
    )
    $BackendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
    $FrontendRoot = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath
    $AceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath

    $pidPath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryPeriodicSavePidFileName
    Set-Content -LiteralPath $pidPath -Value $PID -Encoding ASCII

    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $AceRoot -State @{
        status          = 'running'
        launcherSession = $LauncherSession
        pid             = $PID
        startedAt       = (Get-Date).ToUniversalTime().ToString('o')
        backgroundMode  = [bool]$BackgroundMode
        spawnToken      = $SpawnToken
        parentPid       = $PID
    }

    try {
        while ($true) {
            if (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot) {
                break
            }
            $tick = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
                -AceRoot $AceRoot -LauncherSession $LauncherSession -CheckpointsRootOverride $CheckpointsRootOverride `
                -GuardBackendRoot $GuardBackendRoot -BackgroundMode:$BackgroundMode
            $stateUpdate = @{
                status       = 'running'
                lastOutcome  = $tick.Outcome
                lastDecision = $tick.DecisionAction
                lastTickAt   = (Get-Date).ToUniversalTime().ToString('o')
            }
            if ($tick.Reasons -and $tick.Outcome -like 'blocked*') {
                $stateUpdate['lastSkipReason'] = ($tick.Reasons | Select-Object -First 1)
            }
            Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $AceRoot -State $stateUpdate
            if ($tick.Outcome -eq 'stop_requested') {
                break
            }
            Wait-AceLocalRecoveryPeriodicSaveInterTickDelay -AceRoot $AceRoot -TotalSeconds $TickIntervalSeconds
        }
    }
    finally {
        Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $AceRoot -State @{
            status   = 'stopped'
            stoppedAt = (Get-Date).ToUniversalTime().ToString('o')
        }
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        Exit-AceLocalRecoveryPeriodicSaveLock -AceRoot $AceRoot
    }
}
