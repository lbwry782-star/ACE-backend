# ACE LOCAL — recovery session ledger (phase 1; not wired to Start/AceUndo/restore).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

$script:AceLocalRecoveryLedgerSchemaVersion = 1
$script:AceLocalRecoveryLedgerFileName = 'recovery-session-ledger.json'

function Get-AceLocalRecoveryDevStateDir {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $dir = Join-Path $AceRoot '_backups\local-dev'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return $dir
}

function Get-AceLocalRecoveryLedgerPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $AceRoot) $script:AceLocalRecoveryLedgerFileName
}

function Write-AceLocalRecoveryJsonAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Object
    )
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $json = $Object | ConvertTo-Json -Depth 12
    $tmp = Join-Path $dir ("{0}.tmp.{1}" -f (Split-Path -Leaf $Path), [guid]::NewGuid().ToString('N'))
    Set-Content -LiteralPath $tmp -Value $json -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function New-AceLocalRecoveryLedgerShell {
    param(
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [string]$SessionStartedAt = ''
    )
    if (-not $SessionStartedAt) {
        $SessionStartedAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    return [ordered]@{
        schemaVersion           = $script:AceLocalRecoveryLedgerSchemaVersion
        launcherSession         = $LauncherSession
        sessionStartedAt        = $SessionStartedAt
        lastDirtyScanAt         = $null
        lastDirtyFingerprint    = $null
        lastVerifiedCheckpoint      = $null
        lastFullVerifiedCheckpoint  = $null
        cleanShutdown           = [ordered]@{
            completed       = $false
            completedAt     = $null
            launcherSession = $null
        }
        openIncident            = $null
        timingState             = [ordered]@{
            lastLightCheckpointDueAt  = $null
            lastFullCheckpointDueAt   = $null
        }
        updatedAt               = (Get-Date).ToUniversalTime().ToString('o')
    }
}

function Read-AceLocalRecoveryLedger {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [switch]$AllowCreateStateDir
    )
    if ($AllowCreateStateDir) {
        $path = Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot
    }
    else {
        $path = Join-Path $AceRoot '_backups\local-dev\recovery-session-ledger.json'
    }
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }
    try {
        $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $null
        }
        return ($raw | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Test-AceLocalRecoveryLedgerValid {
    param([Parameter(Mandatory = $true)]$Ledger)
    if (-not $Ledger) {
        return @{ Ok = $false; Reason = 'ledger_missing' }
    }
    if ([int]$Ledger.schemaVersion -ne $script:AceLocalRecoveryLedgerSchemaVersion) {
        return @{ Ok = $false; Reason = 'schema_unsupported' }
    }
    if ([string]::IsNullOrWhiteSpace([string]$Ledger.launcherSession)) {
        return @{ Ok = $false; Reason = 'launcher_session_missing' }
    }
    return @{ Ok = $true; Reason = 'ok' }
}

function Initialize-AceLocalRecoveryLegacyBootstrapLedger {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$ManagedLauncherSession
    )
    $session = ($ManagedLauncherSession -replace '\s', '').Trim()
    if (-not $session) {
        throw 'RECOVERY_LEGACY_BOOTSTRAP_SESSION_REQUIRED'
    }
    $path = Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot
    if (Test-Path -LiteralPath $path) {
        $existing = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
        if ($existing -and $existing.lastVerifiedCheckpoint) {
            throw 'RECOVERY_LEGACY_BOOTSTRAP_LEDGER_ALREADY_INITIALIZED'
        }
        if ($existing -and $existing.legacyRecoveryBootstrap) {
            $st = [string]$existing.legacyRecoveryBootstrap.status
            if ($st -eq 'COMPLETED') {
                throw 'RECOVERY_LEGACY_BOOTSTRAP_ALREADY_COMPLETED'
            }
        }
        if ($existing -and -not $existing.legacyRecoveryBootstrap) {
            throw 'RECOVERY_LEGACY_BOOTSTRAP_LEDGER_EXISTS_WITHOUT_BOOTSTRAP'
        }
    }
    $ledger = New-AceLocalRecoveryLedgerShell -LauncherSession $session
    $ledger.legacyRecoveryBootstrap = [ordered]@{
        status                         = 'PENDING_FULL_CHECKPOINT'
        startedAt                      = (Get-Date).ToUniversalTime().ToString('o')
        managedLauncherSessionAtBootstrap = $session
        noVerifiedCheckpointYet        = $true
        completedAt                    = $null
        lastError                      = $null
    }
    $ledger.cleanShutdown = [ordered]@{
        completed       = $false
        completedAt     = $null
        launcherSession = $null
    }
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object $ledger
    return $ledger
}

function Set-AceLocalRecoveryLegacyBootstrapLedgerStatus {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][ValidateSet('PENDING_FULL_CHECKPOINT', 'IN_PROGRESS', 'COMPLETED', 'FAILED')][string]$Status,
        [string]$LastError = '',
        [switch]$ClearNoVerifiedFlag
    )
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
    if (-not $valid.Ok) {
        throw ('RECOVERY_LEDGER_INVALID:' + $valid.Reason)
    }
    if (-not $ledger.legacyRecoveryBootstrap) {
        throw 'RECOVERY_LEGACY_BOOTSTRAP_METADATA_MISSING'
    }
    $ledger.legacyRecoveryBootstrap.status = $Status
    if ($Status -eq 'COMPLETED') {
        $ledger.legacyRecoveryBootstrap.completedAt = (Get-Date).ToUniversalTime().ToString('o')
        if ($ClearNoVerifiedFlag) {
            $ledger.legacyRecoveryBootstrap.noVerifiedCheckpointYet = $false
        }
    }
    if ($LastError) {
        $ledger.legacyRecoveryBootstrap.lastError = $LastError.Substring(0, [Math]::Min(480, $LastError.Length))
    }
    $ledger.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot) -Object $ledger
    return $ledger
}

function Initialize-AceLocalRecoveryLedger {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession
    )
    $session = ($LauncherSession -replace '\s', '').Trim()
    if (-not $session) {
        throw 'RECOVERY_LEDGER_LAUNCHER_SESSION_REQUIRED'
    }
    $path = Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot
    $existing = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $preserveCheckpoint = $null
    $preserveFullCheckpoint = $null
    if ($existing -and (Test-AceLocalRecoveryLedgerValid -Ledger $existing).Ok) {
        if ($existing.lastVerifiedCheckpoint) {
            $preserveCheckpoint = $existing.lastVerifiedCheckpoint
        }
        if ($existing.lastFullVerifiedCheckpoint) {
            $preserveFullCheckpoint = $existing.lastFullVerifiedCheckpoint
        }
    }
    $ledger = New-AceLocalRecoveryLedgerShell -LauncherSession $session
    if ($preserveCheckpoint) {
        $ledger.lastVerifiedCheckpoint = $preserveCheckpoint
    }
    if ($preserveFullCheckpoint) {
        $ledger.lastFullVerifiedCheckpoint = $preserveFullCheckpoint
    }
    Write-AceLocalRecoveryJsonAtomic -Path $path -Object $ledger
    return $ledger
}

function Update-AceLocalRecoveryLedgerDirtyScan {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$DirtyFingerprint,
        [string]$DirtyScanAt = ''
    )
    if (-not $DirtyScanAt) {
        $DirtyScanAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
    if (-not $valid.Ok) {
        throw ('RECOVERY_LEDGER_INVALID:' + $valid.Reason)
    }
    $ledger.lastDirtyScanAt = $DirtyScanAt
    $ledger.lastDirtyFingerprint = $DirtyFingerprint
    $ledger.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot) -Object $ledger
    return $ledger
}

function Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$CheckpointId,
        [Parameter(Mandatory = $true)][string]$CheckpointPath,
        [string]$VerifiedAt = '',
        [ValidateSet('', 'full', 'light')][string]$CheckpointType = ''
    )
    if (-not $VerifiedAt) {
        $VerifiedAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
    if (-not $valid.Ok) {
        throw ('RECOVERY_LEDGER_INVALID:' + $valid.Reason)
    }
    $entry = [ordered]@{
        checkpointId   = $CheckpointId
        checkpointPath = $CheckpointPath
        verifiedAt     = $VerifiedAt
    }
    $ledger.lastVerifiedCheckpoint = $entry
    if ($CheckpointType -eq 'full') {
        $ledger.lastFullVerifiedCheckpoint = $entry
    }
    $ledger.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot) -Object $ledger
    return $ledger
}

function Open-AceLocalRecoveryLedgerIncident {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$Kind,
        [string]$Detail = ''
    )
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
    if (-not $valid.Ok) {
        throw ('RECOVERY_LEDGER_INVALID:' + $valid.Reason)
    }
    $detailText = ($Detail -replace '\s+', ' ').Trim()
    if ($detailText.Length -gt 240) {
        $detailText = $detailText.Substring(0, 237) + '...'
    }
    $incidentPayload = [ordered]@{
        kind     = $Kind
        openedAt = (Get-Date).ToUniversalTime().ToString('o')
        detail   = $detailText
    }
    if ($ledger.PSObject.Properties.Match('openIncident').Count -gt 0) {
        $ledger.openIncident = $incidentPayload
    }
    else {
        $ledger | Add-Member -NotePropertyName 'openIncident' -NotePropertyValue $incidentPayload -Force
    }
    if ($ledger.cleanShutdown) {
        $ledger.cleanShutdown.completed = $false
        $ledger.cleanShutdown.completedAt = $null
        $ledger.cleanShutdown.launcherSession = $null
    }
    $ledger.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot) -Object $ledger
    return $ledger
}

function Register-AceLocalRecoveryCleanShutdownComplete {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)][bool]$ShutdownVerified,
        [string]$CompletedAt = ''
    )
    if (-not $ShutdownVerified) {
        throw 'RECOVERY_CLEAN_SHUTDOWN_NOT_VERIFIED'
    }
    $session = ($LauncherSession -replace '\s', '').Trim()
    if (-not $session) {
        throw 'RECOVERY_CLEAN_SHUTDOWN_SESSION_REQUIRED'
    }
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
    if (-not $valid.Ok) {
        throw ('RECOVERY_LEDGER_INVALID:' + $valid.Reason)
    }
    $bound = [string]$ledger.launcherSession
    if ($bound -ne $session) {
        throw 'RECOVERY_CLEAN_SHUTDOWN_SESSION_MISMATCH'
    }
    if (-not $CompletedAt) {
        $CompletedAt = (Get-Date).ToUniversalTime().ToString('o')
    }
    $ledger.cleanShutdown = [ordered]@{
        completed       = $true
        completedAt     = $CompletedAt
        launcherSession = $session
    }
    if ($ledger.PSObject.Properties.Match('openIncident').Count -gt 0) {
        $ledger.openIncident = $null
    }
    else {
        $ledger | Add-Member -NotePropertyName 'openIncident' -NotePropertyValue $null -Force
    }
    $ledger.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot) -Object $ledger
    return $ledger
}

function Test-AceLocalRecoveryCleanShutdownRecorded {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$LauncherSession = ''
    )
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot
    if (-not $ledger) {
        return $false
    }
    $cs = $ledger.cleanShutdown
    if (-not $cs -or -not $cs.completed) {
        return $false
    }
    if ($LauncherSession -and ([string]$cs.launcherSession -ne $LauncherSession)) {
        return $false
    }
    return $true
}
