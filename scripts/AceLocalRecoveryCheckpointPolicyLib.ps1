# ACE LOCAL — recovery checkpoint policy constants and timing decisions (no scheduler/checkpoints in phase 1).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')

$script:AceLocalRecoveryDirtyScanIntervalSeconds = 120
$script:AceLocalRecoveryLightCheckpointIntervalSeconds = 300
$script:AceLocalRecoveryFullCheckpointIntervalSeconds = 1800
$script:AceLocalRecoveryMinimumVerifiedCheckpointRetention = 5

$script:AceLocalRecoveryCheckpointActionNone = 'none'
$script:AceLocalRecoveryCheckpointActionDirtyScan = 'dirty_scan'
$script:AceLocalRecoveryCheckpointActionLightCheckpoint = 'light_checkpoint'
$script:AceLocalRecoveryCheckpointActionFullCheckpoint = 'full_checkpoint'

function Get-AceLocalRecoveryCheckpointPolicy {
    return [ordered]@{
        dirtyScanIntervalSeconds            = $script:AceLocalRecoveryDirtyScanIntervalSeconds
        lightCheckpointIntervalSeconds      = $script:AceLocalRecoveryLightCheckpointIntervalSeconds
        fullCheckpointIntervalSeconds       = $script:AceLocalRecoveryFullCheckpointIntervalSeconds
        minimumVerifiedCheckpointRetention  = $script:AceLocalRecoveryMinimumVerifiedCheckpointRetention
    }
}

function ConvertTo-RecoveryLedgerDateTime {
    param([string]$IsoText)
    if ([string]::IsNullOrWhiteSpace($IsoText)) {
        return $null
    }
    try {
        $dt = [DateTime]::Parse($IsoText, $null, [System.Globalization.DateTimeStyles]::RoundtripKind)
        if ($dt.Kind -eq [DateTimeKind]::Unspecified) {
            return [DateTime]::SpecifyKind($dt, [DateTimeKind]::Utc)
        }
        return $dt.ToUniversalTime()
    }
    catch {
        return $null
    }
}

function Get-AceLocalRecoveryCheckpointTimingDecision {
    param(
        [Parameter(Mandatory = $true)]$Ledger,
        [Parameter(Mandatory = $true)][DateTime]$NowUtc,
        [Parameter(Mandatory = $true)][bool]$WorkspaceDirtyChanged,
        [Parameter(Mandatory = $true)][bool]$HasWorkingTreeChanges
    )
    $reasons = @()
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $Ledger
    if (-not $valid.Ok) {
        return @{
            Action  = $script:AceLocalRecoveryCheckpointActionNone
            Reasons = @('ledger_invalid')
        }
    }

    $lastScan = ConvertTo-RecoveryLedgerDateTime -IsoText ([string]$Ledger.lastDirtyScanAt)
    $secondsSinceScan = if ($lastScan) { ($NowUtc - $lastScan).TotalSeconds } else { [double]::MaxValue }

    $timing = $Ledger.timingState
    $lastLight = $null
    $lastFull = $null
    if ($timing) {
        $lightIso = ''
        $fullIso = ''
        if ($timing.PSObject.Properties.Match('lastLightCheckpointDueAt').Count -gt 0) {
            $lightIso = [string]$timing.lastLightCheckpointDueAt
        }
        if ($timing.PSObject.Properties.Match('lastFullCheckpointDueAt').Count -gt 0) {
            $fullIso = [string]$timing.lastFullCheckpointDueAt
        }
        $lastLight = ConvertTo-RecoveryLedgerDateTime -IsoText $lightIso
        $lastFull = ConvertTo-RecoveryLedgerDateTime -IsoText $fullIso
    }
    # No prior light/full anchor => not overdue (only dirty scan may run on first interval).
    $secondsSinceLight = if ($lastLight) { ($NowUtc - $lastLight).TotalSeconds } else { 0 }
    $secondsSinceFull = if ($lastFull) { ($NowUtc - $lastFull).TotalSeconds } else { 0 }

    $action = $script:AceLocalRecoveryCheckpointActionNone

    if ($secondsSinceScan -ge $script:AceLocalRecoveryDirtyScanIntervalSeconds) {
        $action = $script:AceLocalRecoveryCheckpointActionDirtyScan
        $reasons += 'dirty_scan_interval_elapsed'
    }

    if ($HasWorkingTreeChanges -and $WorkspaceDirtyChanged `
            -and $secondsSinceLight -ge $script:AceLocalRecoveryLightCheckpointIntervalSeconds) {
        $action = $script:AceLocalRecoveryCheckpointActionLightCheckpoint
        $reasons += 'light_checkpoint_due_dirty'
    }

    if ($secondsSinceFull -ge $script:AceLocalRecoveryFullCheckpointIntervalSeconds) {
        if ($action -eq $script:AceLocalRecoveryCheckpointActionLightCheckpoint) {
            $reasons += 'full_checkpoint_overrides_light'
        }
        $action = $script:AceLocalRecoveryCheckpointActionFullCheckpoint
        $reasons += 'full_checkpoint_interval_elapsed'
    }

    if ($action -eq $script:AceLocalRecoveryCheckpointActionNone) {
        $reasons += 'no_checkpoint_due'
    }

    return @{
        Action           = $action
        Reasons          = @($reasons)
        SecondsSinceScan = $secondsSinceScan
        Policy           = Get-AceLocalRecoveryCheckpointPolicy
    }
}

function Update-AceLocalRecoveryLedgerTimingAfterDecision {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$DecisionAction,
        [string]$NowIso = ''
    )
    if (-not $NowIso) {
        $NowIso = (Get-Date).ToUniversalTime().ToString('o')
    }
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $ledger
    if (-not $valid.Ok) {
        throw ('RECOVERY_LEDGER_INVALID:' + $valid.Reason)
    }
    if (-not $ledger.timingState) {
        $ledger | Add-Member -NotePropertyName 'timingState' -NotePropertyValue ([ordered]@{}) -Force
    }
    switch ($DecisionAction) {
        $script:AceLocalRecoveryCheckpointActionLightCheckpoint {
            $ledger.timingState.lastLightCheckpointDueAt = $NowIso
        }
        $script:AceLocalRecoveryCheckpointActionFullCheckpoint {
            $ledger.timingState.lastFullCheckpointDueAt = $NowIso
        }
    }
    $ledger.updatedAt = $NowIso
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot) -Object $ledger
    return $ledger
}

function Get-AceLocalRecoveryPeriodicSaveTimingDecision {
    <#
      Phase 4 — intervals measured from last VERIFIED checkpoint (ledger), not timing anchors alone.
      Full/light require working-tree changes AND content fingerprint change since last scan baseline.
      Does not advance timing anchors (caller updates only after VERIFIED checkpoint).
    #>
    param(
        [Parameter(Mandatory = $true)]$Ledger,
        [Parameter(Mandatory = $true)][DateTime]$NowUtc,
        [Parameter(Mandatory = $true)][bool]$WorkspaceDirtyChanged,
        [Parameter(Mandatory = $true)][bool]$HasWorkingTreeChanges
    )
    $reasons = @()
    $valid = Test-AceLocalRecoveryLedgerValid -Ledger $Ledger
    if (-not $valid.Ok) {
        return @{
            Action  = $script:AceLocalRecoveryCheckpointActionNone
            Reasons = @('ledger_invalid')
        }
    }

    $lastScan = ConvertTo-RecoveryLedgerDateTime -IsoText ([string]$Ledger.lastDirtyScanAt)
    $secondsSinceScan = if ($lastScan) { ($NowUtc - $lastScan).TotalSeconds } else { [double]::MaxValue }

    $lastAnyVerifiedAt = $null
    if ($Ledger.lastVerifiedCheckpoint) {
        $lastAnyVerifiedAt = ConvertTo-RecoveryLedgerDateTime -IsoText ([string]$Ledger.lastVerifiedCheckpoint.verifiedAt)
    }
    $secondsSinceAnyVerified = if ($lastAnyVerifiedAt) { ($NowUtc - $lastAnyVerifiedAt).TotalSeconds } else { [double]::MaxValue }

    $lastFullVerifiedAt = $null
    if ($Ledger.lastFullVerifiedCheckpoint) {
        $lastFullVerifiedAt = ConvertTo-RecoveryLedgerDateTime -IsoText ([string]$Ledger.lastFullVerifiedCheckpoint.verifiedAt)
    }
    $hasPriorFullVerified = ($null -ne $lastFullVerifiedAt)
    $secondsSinceFullVerified = if ($lastFullVerifiedAt) { ($NowUtc - $lastFullVerifiedAt).TotalSeconds } else { [double]::MaxValue }

    $action = $script:AceLocalRecoveryCheckpointActionNone

    if ($secondsSinceScan -ge $script:AceLocalRecoveryDirtyScanIntervalSeconds) {
        $action = $script:AceLocalRecoveryCheckpointActionDirtyScan
        $reasons += 'dirty_scan_interval_elapsed'
    }

    if ($HasWorkingTreeChanges -and $WorkspaceDirtyChanged `
            -and $secondsSinceAnyVerified -ge $script:AceLocalRecoveryLightCheckpointIntervalSeconds) {
        if ($hasPriorFullVerified) {
            $action = $script:AceLocalRecoveryCheckpointActionLightCheckpoint
            $reasons += 'light_checkpoint_due_since_last_verified'
        }
    }

    if ($HasWorkingTreeChanges -and $WorkspaceDirtyChanged) {
        if (-not $hasPriorFullVerified) {
            $action = $script:AceLocalRecoveryCheckpointActionFullCheckpoint
            $reasons += 'first_full_required_no_prior_full_verified'
        }
        elseif ($secondsSinceFullVerified -ge $script:AceLocalRecoveryFullCheckpointIntervalSeconds) {
            if ($action -eq $script:AceLocalRecoveryCheckpointActionLightCheckpoint) {
                $reasons += 'full_checkpoint_overrides_light'
            }
            $action = $script:AceLocalRecoveryCheckpointActionFullCheckpoint
            $reasons += 'full_checkpoint_due_since_last_full_verified'
        }
    }

    if ($action -eq $script:AceLocalRecoveryCheckpointActionNone) {
        $reasons += 'no_checkpoint_due'
    }

    return @{
        Action                 = $action
        Reasons                = @($reasons)
        SecondsSinceScan       = $secondsSinceScan
        SecondsSinceAnyVerified  = $secondsSinceAnyVerified
        SecondsSinceFullVerified = $secondsSinceFullVerified
        HasPriorFullVerified     = $hasPriorFullVerified
        Policy                   = Get-AceLocalRecoveryCheckpointPolicy
    }
}
