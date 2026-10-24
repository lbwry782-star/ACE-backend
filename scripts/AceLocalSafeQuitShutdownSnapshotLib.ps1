# ACE LOCAL — safe quit SHUTDOWN_SNAPSHOT + verification (5.19.5; TEMP/isolated only).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalSafeQuitPeriodicSaveLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')

function Read-AceLocalSafeQuitManagedRegistry {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $regPath = Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) 'managed-processes.json'
    if (-not (Test-Path -LiteralPath $regPath)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $regPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Resolve-AceLocalSafeQuitRepoRoots {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $reg = Read-AceLocalSafeQuitManagedRegistry -AceRoot $AceRoot
    if (-not $reg) {
        return @{ Ok = $false; Reason = 'managed_registry_missing' }
    }
    $be = ''
    $fe = ''
    if ($reg.PSObject.Properties.Match('backendRoot').Count -gt 0) {
        $be = [string]$reg.backendRoot
    }
    if ($reg.PSObject.Properties.Match('frontendRoot').Count -gt 0) {
        $fe = [string]$reg.frontendRoot
    }
    if (-not $be -or -not (Test-Path -LiteralPath $be)) {
        return @{ Ok = $false; Reason = 'backend_root_missing_or_inaccessible' }
    }
    if (-not $fe -or -not (Test-Path -LiteralPath $fe)) {
        return @{ Ok = $false; Reason = 'frontend_root_missing_or_inaccessible' }
    }
    return @{
        Ok           = $true
        BackendRoot  = (Resolve-Path -LiteralPath $be).ProviderPath
        FrontendRoot = (Resolve-Path -LiteralPath $fe).ProviderPath
        Registry     = $reg
    }
}

function Test-AceLocalSafeQuitShutdownSnapshotPreconditions {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)]$GateResult,
        [Parameter(Mandatory = $true)]$QuiesceResult
    )
    $session = ($LauncherSession -replace '\s', '').Trim()
    if (-not $session) {
        return @{ Ok = $false; Reason = 'launcher_session_empty' }
    }
    if (-not $GateResult -or -not $GateResult.Ok) {
        return @{ Ok = $false; Reason = 'gate_not_safe_to_stop' }
    }
    if ([string]$GateResult.GateStatus -ne 'SAFE_TO_STOP') {
        return @{ Ok = $false; Reason = ('gate_status:' + [string]$GateResult.GateStatus) }
    }
    if (-not $QuiesceResult -or -not $QuiesceResult.Ok) {
        return @{ Ok = $false; Reason = 'quiesce_not_complete' }
    }
    $registrySession = Read-AceLocalManagedProcessesLauncherSession -AceRoot $AceRoot
    if ($registrySession -and ($registrySession -ne $session)) {
        return @{ Ok = $false; Reason = 'managed_registry_session_mismatch' }
    }
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    if ($ledger -and ([string]$ledger.launcherSession -ne $session)) {
        return @{ Ok = $false; Reason = 'ledger_session_mismatch' }
    }
    if (-not (Test-AceLocalRecoveryPeriodicSaveStopRequested -AceRoot $AceRoot)) {
        return @{ Ok = $false; Reason = 'periodic_stop_not_requested' }
    }
    $lock = Get-AceLocalRecoveryPeriodicSaveLockAssessment -AceRoot $AceRoot -ExpectedLauncherSession $session
    if ([string]$lock.Classification -eq 'live') {
        return @{ Ok = $false; Reason = 'checkpoint_lock_still_live' }
    }
    $binding = Test-AceLocalSafeQuitPeriodicSaveLockSessionBinding -AceRoot $AceRoot -ExpectedLauncherSession $session
    if (-not $binding.Ok) {
        return @{ Ok = $false; Reason = [string]$binding.Reason }
    }
    $roots = Resolve-AceLocalSafeQuitRepoRoots -AceRoot $AceRoot
    if (-not $roots.Ok) {
        return @{ Ok = $false; Reason = [string]$roots.Reason }
    }
    $engineBackend = (Split-Path -Parent $ScriptDir)
    $preWrite = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $AceRoot -LauncherSession $session `
        -GuardBackendRoot $engineBackend -SkipLiveAssessment
    if ($preWrite.Allowed) {
        return @{ Ok = $false; Reason = 'pre_write_still_allowed' }
    }
    return @{ Ok = $true; Reason = 'ok'; RepoRoots = $roots }
}

function Invoke-AceLocalSafeQuitShutdownSnapshotCapture {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)]$GateResult,
        [Parameter(Mandatory = $true)]$QuiesceResult,
        [string]$SnapshotsRootOverride = '',
        [scriptblock]$DuringBackendCopyHook = $null,
        [switch]$SimulateFailAfterBackend,
        [switch]$SimulateFailBeforeVerify
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        throw 'SAFE_QUIT_SNAPSHOT_ACE_ROOT_NOT_ISOLATED'
    }
    $pre = Test-AceLocalSafeQuitShutdownSnapshotPreconditions -AceRoot $AceRoot -LauncherSession $LauncherSession `
        -GateResult $GateResult -QuiesceResult $QuiesceResult
    if (-not $pre.Ok) {
        return @{
            Ok      = $false
            Outcome = 'preconditions_failed'
            Detail  = [string]$pre.Reason
        }
    }
    $roots = $pre.RepoRoots
    $sessionsRoot = Join-Path $AceRoot '_backups\sessions'
    $snapParams = @{
        BackendRoot  = $roots.BackendRoot
        FrontendRoot = $roots.FrontendRoot
        AceRoot      = $AceRoot
        SessionsRoot = $sessionsRoot
    }
    if ($SnapshotsRootOverride) {
        $snapParams['SnapshotsRootOverride'] = $SnapshotsRootOverride
    }
    if ($null -ne $DuringBackendCopyHook) {
        $snapParams['DuringBackendCopyHook'] = $DuringBackendCopyHook
    }
    if ($SimulateFailAfterBackend) {
        $snapParams['SimulateFailAfterBackend'] = $true
    }
    if ($SimulateFailBeforeVerify) {
        $snapParams['SimulateFailBeforeVerify'] = $true
    }
    $snapParams['DeferLastVerifiedPointerUpdate'] = $true
    $snap = New-AceLocalShutdownSnapshot @snapParams
    if (-not $snap.Ok) {
        return @{
            Ok           = $false
            Outcome      = 'capture_failed'
            Detail       = ($snap.Errors -join ';')
            SnapshotId   = $snap.SnapshotId
            SnapshotRoot = $snap.SnapshotRoot
            Status       = $snap.Status
        }
    }
    return @{
        Ok           = $true
        Outcome      = 'captured'
        Detail       = $snap.SnapshotId
        SnapshotId   = $snap.SnapshotId
        SnapshotRoot = $snap.SnapshotRoot
        Status       = $snap.Status
    }
}

function Commit-AceLocalSafeQuitVerifiedSnapshotPointer {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)]$CaptureResult
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        throw 'SAFE_QUIT_SNAPSHOT_POINTER_COMMIT_REQUIRES_TEMP'
    }
    if (-not $CaptureResult -or -not $CaptureResult.Ok) {
        return @{ Committed = $false; Reason = 'capture_not_ok' }
    }
    $root = [string]$CaptureResult.SnapshotRoot
    $metaPath = Join-Path $root 'snapshot-metadata.json'
    if (-not (Test-Path -LiteralPath $metaPath)) {
        return @{ Committed = $false; Reason = 'metadata_missing' }
    }
    $meta = Get-Content -LiteralPath $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$meta.status -ne 'VERIFIED') {
        return @{ Committed = $false; Reason = 'metadata_not_verified' }
    }
    $pointer = [ordered]@{
        schemaVersion = 1
        snapshotId    = [string]$CaptureResult.SnapshotId
        snapshotRoot  = $root
        verifiedAt    = [string]$meta.verifiedAt
        committedBy   = 'safe_quit_coordinator'
    }
    Write-AceLocalShutdownJsonAtomic -Path (Get-AceLocalShutdownLastVerifiedPointerPath -AceRoot $AceRoot) -Object $pointer
    return @{ Committed = $true; Reason = 'ok'; Pointer = $pointer }
}

function Invoke-AceLocalSafeQuitSnapshotVerification {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)]$CaptureResult,
        [switch]$SimulateIntegrityFailure
    )
    if (-not $CaptureResult -or -not $CaptureResult.Ok) {
        return @{
            Ok      = $false
            Outcome = 'skipped'
            Detail  = 'no_successful_capture'
        }
    }
    $root = [string]$CaptureResult.SnapshotRoot
    if (-not $root -or -not (Test-Path -LiteralPath $root)) {
        return @{
            Ok      = $false
            Outcome = 'missing_snapshot_root'
            Detail  = 'snapshot_root_not_found'
        }
    }
    $metaPath = Join-Path $root 'snapshot-metadata.json'
    if (-not (Test-Path -LiteralPath $metaPath)) {
        return @{
            Ok      = $false
            Outcome = 'metadata_missing'
            Detail  = 'snapshot-metadata.json'
        }
    }
    $meta = Get-Content -LiteralPath $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ([string]$meta.status -ne 'VERIFIED') {
        return @{
            Ok      = $false
            Outcome = 'not_verified_status'
            Detail  = [string]$meta.status
        }
    }
    $integrity = Test-AceLocalShutdownSnapshotIntegrity -SnapshotRoot $root `
        -SimulateManifestHashMismatch:$SimulateIntegrityFailure
    if (-not $integrity.Ok) {
        return @{
            Ok       = $false
            Outcome  = 'integrity_failed'
            Detail   = ($integrity.Errors -join ',')
            Partial  = [bool]$integrity.Partial
            Errors   = @($integrity.Errors)
        }
    }
    $commit = Commit-AceLocalSafeQuitVerifiedSnapshotPointer -AceRoot $AceRoot -CaptureResult $CaptureResult
    if (-not $commit.Committed) {
        return @{
            Ok      = $false
            Outcome = 'pointer_commit_failed'
            Detail  = [string]$commit.Reason
        }
    }
    return @{
        Ok      = $true
        Outcome = 'verified'
        Detail  = 'integrity_ok'
        Errors  = @()
        Pointer = $commit.Pointer
    }
}

function Reset-AceLocalSafeQuitFailedAttempt {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$SafeQuitAttemptId = ''
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        return @{ Cleared = $false; Reason = 'not_temp_ace_root' }
    }
    $result = Reset-AceLocalSafeQuitPeriodicSaveQuiesce -AceRoot $AceRoot -SafeQuitAttemptId $SafeQuitAttemptId `
        -ForceIfInitMatchesAttemptOnly:([bool]$SafeQuitAttemptId)
    return $result
}
