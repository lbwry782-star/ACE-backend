#Requires -Version 5.1
<#
  Isolated tests: retain proven managed ownership on already-running reattach.
  TEMP only. Does not start/stop live LOCAL, AceUndo, or live recovery data.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalManagedOwnershipRetain (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Sync-AceLocalAlreadyRunningManagedComponent') 'start_calls_sync_retain'
Assert-True ($startText -notmatch '\$managedRegistry\.components\.backend = \$null') 'start_no_null_backend'
Assert-True ($startText -notmatch '\$managedRegistry\.components\.worker = \$null') 'start_no_null_worker'
Assert-True ($startText -notmatch '\$managedRegistry\.components\.frontend = \$null') 'start_no_null_frontend'
Assert-True ($startText -match 'ownership not verified') 'start_reports_unverified_ownership'

$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($stopText -match 'Confirm-ManagedComponentStopAllowed') 'stop_still_revalidates'
Assert-True ($stopText -match 'ExpectedLauncherSession') 'stop_still_checks_launcher_session'

$libText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1') -Raw
Assert-True ($libText -match 'function Test-ManagedComponentRecordRetainable') 'retain_helper_exists'
Assert-True ($libText -match 'StopEligible\s*=\s*\$false') 'retain_does_not_mark_stop_eligible'
Assert-True ($libText -match 'function Confirm-ManagedComponentStopAllowed') 'stop_confirm_unchanged'

$bgText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1') -Raw
Assert-True ($bgText -match 'PreserveExistingIdentity') 'periodic_already_running_preserves_runtime_identity'
Assert-True ($bgText -notmatch 'Write-AceLocalRecoveryPeriodicSaveRuntimeConfig[\s\S]{0,80}LauncherSession \$launcher\r?\n\s+\$spawn = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess') 'periodic_does_not_write_runtime_before_spawn'

$tmpRoot = Join-Path $env:TEMP ("ace-own-retain-{0}" -f [guid]::NewGuid().ToString('N'))
$tmpAce = Join-Path $tmpRoot 'ACE'
$tmpBackend = Join-Path $tmpAce 'ACE-Backend'
$tmpFrontend = Join-Path $tmpAce 'ACE-Frontend'
New-Item -ItemType Directory -Path $tmpBackend, $tmpFrontend -Force | Out-Null
$dummyPids = @()
$dummyProc = $null
$childProc = $null

try {
    $py = Join-Path ((Resolve-Path (Join-Path $ScriptDir '..')).Path) '.venv312\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $py)) {
        $cmdPy = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($cmdPy) { $py = [string]$cmdPy.Source }
    }
    Assert-True (Test-Path -LiteralPath $py) 'dummy_python_available'

    $appPy = Join-Path $tmpBackend 'app.py'
    @'
import os
import subprocess
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == "child":
    while True:
        time.sleep(0.25)
else:
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "child"])
    while True:
        time.sleep(0.25)
'@ | Set-Content -LiteralPath $appPy -Encoding UTF8

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $py
    $psi.Arguments = ('"{0}"' -f $appPy)
    $psi.WorkingDirectory = $tmpBackend
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $dummyProc = [System.Diagnostics.Process]::Start($psi)
    Assert-True ($null -ne $dummyProc) 'dummy_parent_started'
    $dummyPids += [int]$dummyProc.Id
    Start-Sleep -Milliseconds 400
    $found = @(Find-LocalAceBackendProcesses -BackendRoot $tmpBackend)
    Assert-True ($found.Count -ge 1) 'dummy_backend_fingerprint_visible'
    $parentPid = [int]$dummyProc.Id
    $childPid = 0
    foreach ($row in $found) {
        if ([int]$row.ProcId -ne $parentPid) {
            $childPid = [int]$row.ProcId
            break
        }
    }
    if ($childPid -le 0) {
        Start-Sleep -Milliseconds 600
        $found = @(Find-LocalAceBackendProcesses -BackendRoot $tmpBackend)
        foreach ($row in $found) {
            if ([int]$row.ProcId -ne $parentPid) {
                $childPid = [int]$row.ProcId
                break
            }
        }
    }
    Assert-True ($childPid -gt 0) 'dummy_child_detected'
    $dummyPids += $childPid
    $childProc = Get-Process -Id $childPid -ErrorAction SilentlyContinue

    $parentCreated = Get-Win32ProcessCreationTimeUtc -ProcId $parentPid
    $childCreated = Get-Win32ProcessCreationTimeUtc -ProcId $childPid
    Assert-True ($null -ne $parentCreated -and $null -ne $childCreated) 'dummy_creation_times'

    $originSession = '20261024-040249'
    $attachSession = '20261025-044706'
    $reg = New-AceLocalManagedRegistryShell -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -LauncherSessionId $attachSession
    $createdMap = @{
        ([string]$parentPid) = $parentCreated
        ([string]$childPid)  = $childCreated
    }
    $reg.components.backend = New-AceLocalManagedComponentTreeRecord -RootPid $parentPid `
        -TreePids @($parentPid, $childPid) -LogPath (Join-Path $tmpRoot 'backend.log') `
        -LauncherSession $originSession -ProcessCreatedAt $createdMap
    Write-AceLocalManagedRegistry -AceRoot $tmpAce -RegistryObject $reg

    $validStop = Test-ManagedComponentRecordValid -Record $reg.components.backend -Component 'backend' `
        -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -ExpectedLauncherSession $attachSession `
        -RequireProcessCreationTimes
    Assert-True (-not $validStop.Ok -and $validStop.Reason -eq 'launcher_session_mismatch') 'stop_check_rejects_origin_session_mismatch'

    $retainCheck = Test-ManagedComponentRecordRetainable -Record $reg.components.backend -Component 'backend' `
        -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend
    Assert-True $retainCheck.Ok ('retain_check_ignores_current_launcher_session:' + [string]$retainCheck.Reason)

    $sync = Sync-AceLocalAlreadyRunningManagedComponent -Registry $reg -Name 'backend' -AceRoot $tmpAce `
        -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -CurrentLauncherSession $attachSession
    Assert-True ($sync.Outcome -eq 'retained') 'valid_record_retained'
    Assert-True ($sync.OriginalSession -eq $originSession) 'origin_session_preserved'
    Assert-True (-not $sync.StopEligible) 'retain_not_stop_authorization'
    $readBack = Read-AceLocalManagedRegistry -AceRoot $tmpAce
    Assert-True ([string]$readBack.components.backend.launcherSession -eq $originSession) 'registry_origin_session_unchanged'
    $attachedProp = $readBack.components.backend.PSObject.Properties.Match('lastAttachedLauncherSession')
    Assert-True ($attachedProp.Count -gt 0 -and [string]$readBack.components.backend.lastAttachedLauncherSession -eq $attachSession) 'attached_session_recorded_separately'
    Assert-True ([int]$readBack.components.backend.rootPid -eq $parentPid) 'tree_root_preserved'
    Assert-True (@($readBack.components.backend.treePids).Count -eq 2) 'parent_child_kept_as_one_component'

    $readBack.launcherSession = $attachSession
    $stopActs = @(Get-ManagedComponentStopActions -Registry $readBack -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend)
    $backendAct = $stopActs | Where-Object { $_.Component -eq 'backend' } | Select-Object -First 1
    Assert-True ($null -ne $backendAct -and $backendAct.Action -eq 'skip') 'stop_actions_skip_session_mismatch'
    Assert-True ($backendAct.Reason -eq 'launcher_session_mismatch') 'stop_skip_reason_is_session'

    $reuseMap = @{
        ([string]$parentPid) = '2010-01-01T00:00:00.0000000Z'
        ([string]$childPid)  = $childCreated
    }
    $reg.components.backend = New-AceLocalManagedComponentTreeRecord -RootPid $parentPid `
        -TreePids @($parentPid, $childPid) -LogPath (Join-Path $tmpRoot 'backend.log') `
        -LauncherSession $originSession -ProcessCreatedAt $reuseMap
    $reuseSync = Sync-AceLocalAlreadyRunningManagedComponent -Registry $reg -Name 'backend' -AceRoot $tmpAce `
        -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -CurrentLauncherSession $attachSession
    Assert-True ($reuseSync.Outcome -eq 'unverified') 'pid_reuse_rejected'
    Assert-True ($reuseSync.Reason -like 'process_creation_time_mismatch_pid_*') 'pid_reuse_reason'
    Assert-True ([string]$reg.components.backend.launcherSession -eq $originSession) 'rejected_reuse_does_not_rewrite_session'
    Assert-True ([bool]$reg.components.backend.startedByStack) 'rejected_reuse_does_not_invent_new_ownership_flag'

    $reg.components.worker = $null
    $missing = Sync-AceLocalAlreadyRunningManagedComponent -Registry $reg -Name 'worker' -AceRoot $tmpAce `
        -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -CurrentLauncherSession $attachSession
    Assert-True ($missing.Outcome -eq 'unverified' -and $missing.Reason -eq 'missing_record') 'missing_record_unverified'
    Assert-True ($null -eq $reg.components.worker) 'missing_record_not_created'
    Assert-True (-not $missing.StopEligible) 'missing_record_not_stop_eligible'

    $broken = New-AceLocalManagedComponentTreeRecord -RootPid $parentPid -TreePids @($parentPid) `
        -LogPath (Join-Path $tmpRoot 'broken.log') -LauncherSession $originSession -ProcessCreatedAt @{ ([string]$parentPid) = $parentCreated }
    $broken.startedByStack = $false
    $reg.components.frontend = $broken
    $brokenSync = Sync-AceLocalAlreadyRunningManagedComponent -Registry $reg -Name 'frontend' -AceRoot $tmpAce `
        -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -CurrentLauncherSession $attachSession
    Assert-True ($brokenSync.Outcome -eq 'unverified' -and $brokenSync.Reason -eq 'not_started_by_stack') 'corrupt_not_promoted'
    Assert-True (-not [bool]$reg.components.frontend.startedByStack) 'wrapper_without_record_not_startedByStack'

    $gone = New-AceLocalManagedComponentTreeRecord -RootPid 999991 -TreePids @(999991) `
        -LogPath (Join-Path $tmpRoot 'gone.log') -LauncherSession $originSession `
        -ProcessCreatedAt @{ '999991' = '2026-01-01T00:00:00.0000000Z' }
    $reg.components.worker = $gone
    $goneSync = Sync-AceLocalAlreadyRunningManagedComponent -Registry $reg -Name 'worker' -AceRoot $tmpAce `
        -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -CurrentLauncherSession $attachSession
    Assert-True ($goneSync.Outcome -eq 'unverified') 'vanished_process_unverified'
    Assert-True ($goneSync.Reason -eq 'tree_not_running') 'vanished_reason_tree_not_running'
    Assert-True (-not $goneSync.StopEligible) 'vanished_not_stop_eligible'

    $foreignReg = New-AceLocalManagedRegistryShell -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend -LauncherSessionId $attachSession
    $foreignActs = @(Get-ManagedComponentStopActions -Registry $foreignReg -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend)
    Assert-True (@($foreignActs | Where-Object { $_.Action -eq 'stop' }).Count -eq 0) 'foreign_or_unregistered_not_stop_candidate'

    $stalePreview = @(Get-StaleManagedComponentsPreview -Registry $readBack -BackendRoot $tmpBackend -FrontendRoot $tmpFrontend)
    $staleBackend = @($stalePreview | Where-Object { $_.Component -eq 'backend' })
    Assert-True ($staleBackend.Count -eq 0) 'session_change_does_not_mark_valid_tree_stale'

    $runtimeDir = Join-Path $tmpAce '_backups\local-dev'
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $tmpAce -BackendPort 5000 -LauncherSession $originSession | Out-Null
    $preserve = Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $tmpAce -BackendPort 5101 `
        -LauncherSession $attachSession -AttachedByLauncherSession $attachSession -PreserveExistingIdentity `
        -VerifiedLauncherSession $originSession
    Assert-True $preserve.Ok 'periodic_preserve_ok'
    $runtime = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $tmpAce
    Assert-True ([string]$runtime.launcherSession -eq $originSession) 'periodic_runtime_keeps_origin_session'
    Assert-True ([string]$runtime.attachedByLauncherSession -eq $attachSession) 'periodic_attached_by_separate_field'
    Assert-True ([int]$runtime.backendPort -eq 5000) 'periodic_port_not_replaced_by_attach'

    $emptyAce = Join-Path $tmpRoot 'ACE-empty'
    New-Item -ItemType Directory -Path $emptyAce -Force | Out-Null
    $unknown = Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $emptyAce -BackendPort 5000 `
        -LauncherSession $attachSession -AttachedByLauncherSession $attachSession -PreserveExistingIdentity
    Assert-True (-not $unknown.Ok -and $unknown.Reason -eq 'runtime_identity_unknown') 'periodic_missing_runtime_not_invented'
}
finally {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object {
        ([string]$_.CommandLine) -like ('*{0}*' -f $tmpBackend)
    } | ForEach-Object {
        try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch { }
    }
    foreach ($pidVal in $dummyPids) {
        try { Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue } catch { }
    }
    if ($dummyProc -and -not $dummyProc.HasExited) {
        try { $dummyProc.Kill() } catch { }
    }
    if ($childProc -and -not $childProc.HasExited) {
        try { Stop-Process -Id $childProc.Id -Force -ErrorAction SilentlyContinue } catch { }
    }
    Start-Sleep -Milliseconds 300
    Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalManagedOwnershipRetain tests passed.' -ForegroundColor Green
