#Requires -Version 5.1
<#
.SYNOPSIS
  Stop ACE LOCAL stack components that Start-AceLocal started and registered (not foreign processes).

.DESCRIPTION
  Stops managed Backend, Worker, and Frontend process trees only after fingerprint + session verification.
  Stops the Memurai watch helper. Does not stop the Memurai Windows service.
  -WhatIf performs read-only inspection (no registry, PID, or progress writes).
#>
[CmdletBinding()]
param(
    [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')
. (Join-Path $ScriptDir 'AceLocalMemuraiWatchLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryLegacyBootstrapLib.ps1')

$BackendRoot = Resolve-AceBackendRoot -StartPath $ScriptDir
if (-not $BackendRoot) {
    Write-Err 'ACE-Backend folder not found.'
    exit 1
}
$FrontendRoot = Resolve-AceFrontendRoot -BackendRoot $BackendRoot
if (-not $FrontendRoot) {
    Write-Err 'ACE-Frontend folder not found.'
    exit 1
}
$AceRoot = Split-Path -Parent $BackendRoot
$BackendPort = 5000
$readOnlyFs = [bool]$WhatIf

$registry = Read-AceLocalManagedRegistry -AceRoot $AceRoot -AllowCreateStateDir:(-not $readOnlyFs)

$bsStop = Stop-AceLocalRecoveryLegacyBootstrapForLocalQuit -AceRoot $AceRoot -WhatIf:$WhatIf
if ($WhatIf) {
    if (-not $bsStop.Stopped -and [string]$bsStop.Reason -ne 'not_running') {
        Write-Info 'WhatIf: would wait for legacy recovery bootstrap worker/checkpoint.'
    }
}
elseif (-not $bsStop.Stopped -and [string]$bsStop.Reason -eq 'stop_timeout_evidence_saved') {
    Write-Warn 'Legacy recovery bootstrap: stop wait timed out; evidence saved (worker left to finish checkpoint if active).'
}

$psStop = Stop-AceLocalRecoveryPeriodicSaveForLocalQuit -AceRoot $AceRoot -WhatIf:$WhatIf
if ($WhatIf) {
    if ($psStop.Status -and $psStop.Status.IsRunning) {
        Write-Info 'WhatIf: would request graceful stop of recovery periodic save background service.'
    }
}
elseif (-not $psStop.Stopped -and [string]$psStop.Reason -eq 'stop_timeout_evidence_saved') {
    Write-Warn 'Recovery periodic save: stop wait timed out; evidence saved (process left to finish checkpoint if active).'
}
if (-not $WhatIf -and $psStop.Stopped -and $registry -and $registry.components `
        -and $registry.components.PSObject.Properties.Match('recoveryPeriodicSave').Count -gt 0) {
    $registry.components.recoveryPeriodicSave = $null
}

if (-not $registry) {
    Write-Info 'No managed-process registry found; nothing to stop.'
}
elseif ($WhatIf) {
    $report = Get-AceLocalManagedStopWhatIfReport -Registry $registry `
        -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -BackendPort $BackendPort
    foreach ($stale in @($report.StaleComponents)) {
        Write-Info ("WhatIf: would clear stale managed record for {0} ({1})." -f $stale.Component, $stale.Reason)
    }
    if (@($report.StaleComponents).Count -eq 0) {
        Write-Info 'WhatIf: no stale managed records to clear.'
    }
    foreach ($act in @($report.StopActions)) {
        if ($act.Action -eq 'skip') {
            Write-Warn ("WhatIf: managed {0} would be skipped ({1})." -f $act.Component, $act.Reason)
            continue
        }
        $treeText = ($act.TreePids -join ',')
        $wrapText = if ($act.WrapperPid -gt 0) { [string]$act.WrapperPid } else { '-' }
        Write-Info ("WhatIf: would stop managed {0} rootPid={1} treePids=[{2}] wrapperPid={3} session={4}" -f `
                $act.Component, $act.RootPid, $treeText, $wrapText, $report.LauncherSession)
    }
    if (@($report.StopActions).Count -eq 0) {
        Write-Info 'WhatIf: no managed component trees to stop.'
    }
}
else {
    [void](Remove-StaleManagedComponents -Registry $registry -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -BackendPort $BackendPort)
    $actions = @(Get-ManagedComponentStopActions -Registry $registry -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -BackendPort $BackendPort)
    foreach ($act in $actions) {
        if ($act.Action -eq 'skip') {
            Write-Warn ("Managed {0}: skipped ({1})." -f $act.Component, $act.Reason)
            $registry.components.($act.Component) = $null
            continue
        }
        $treeText = ($act.TreePids -join ',')
        $wrapText = if ($act.WrapperPid -gt 0) { [string]$act.WrapperPid } else { '-' }
        $rec = $registry.components.($act.Component)
        $preStop = Confirm-ManagedComponentStopAllowed -Record $rec -Component $act.Component `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
            -ExpectedLauncherSession ([string]$registry.launcherSession) -BackendPort $BackendPort
        if (-not $preStop.Ok) {
            Write-Warn ("Managed {0}: stop aborted before kill ({1})." -f $act.Component, $preStop.Reason)
            $registry.components.($act.Component) = $null
            continue
        }
        Write-Info ("Stopping managed {0} tree (root {1}, PIDs: {2})..." -f $act.Component, $act.RootPid, $treeText)
        try {
            Stop-ProcessTreeSafe -ProcId ([int]$preStop.RootPid)
        }
        catch {
            Write-Warn ("Managed {0}: stop root failed ({1})." -f $act.Component, $_.Exception.Message)
        }
        $wrapperToStop = [int]$preStop.WrapperPid
        if ($wrapperToStop -gt 0 -and (Test-ProcessAlive $wrapperToStop)) {
            $wrapperCreatedOk = Test-ManagedProcessCreationTimesValid -Record $rec -TreePids @($preStop.TreePids) -WrapperPid $wrapperToStop
            if (-not $wrapperCreatedOk.Ok) {
                Write-Warn ("Managed {0}: wrapper PID {1} failed creation-time check; left running." -f $act.Component, $wrapperToStop)
            }
            elseif (Test-ManagedWrapperPid -ProcId $wrapperToStop -Component $act.Component) {
                try {
                    Stop-Process -Id $wrapperToStop -Force -ErrorAction Stop
                }
                catch {
                    Write-Warn ("Managed {0}: wrapper stop failed ({1})." -f $act.Component, $_.Exception.Message)
                }
            }
            else {
                Write-Warn ("Managed {0}: wrapper PID {1} failed verification; left running." -f $act.Component, $act.WrapperPid)
            }
        }
        $registry.components.($act.Component) = $null
    }
    $registry.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $registry
}

$watchPidPath = Get-AceLocalMemuraiWatchPidPath -AceRoot $AceRoot -AllowCreateStateDir:(-not $readOnlyFs)
$watchPid = Read-AceLocalMemuraiWatchPid -PidPath $watchPidPath
if ($WhatIf) {
    if ($watchPid) {
        Write-Info ("WhatIf: would stop Memurai watch helper PID {0}." -f $watchPid)
    }
    else {
        Write-Info 'WhatIf: Memurai watch helper not running.'
    }
}
else {
    if (Stop-AceLocalMemuraiWatchProcess -PidPath $watchPidPath) {
        Write-Ok 'Memurai watch: stopped.'
    }
}

Write-Ok 'Stop-AceLocal finished (Memurai Windows service was not stopped).'
exit 0
