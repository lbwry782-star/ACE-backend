#Requires -Version 5.1
<#
  Isolated Stop/Clear contract for recovery periodic save.
  TEMP AceRoot and test-owned processes only (PID + creation time + path).
  Does not start/stop live LOCAL.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function Write-Diag([string]$title, $obj) {
    Write-Host ("  DIAG {0}" -f $title) -ForegroundColor DarkCyan
    if ($null -eq $obj) { Write-Host '    (null)'; return }
    if ($obj -is [hashtable]) {
        foreach ($k in @($obj.Keys | Sort-Object)) {
            $v = $obj[$k]
            if ($k -in @('SpawnRecord', 'Occupancy', 'Status')) { continue }
            Write-Host ("    {0}={1}" -f $k, $v)
        }
        return
    }
    $obj.PSObject.Properties | ForEach-Object {
        if ($_.Name -in @('SpawnRecord', 'Occupancy')) { return }
        Write-Host ("    {0}={1}" -f $_.Name, $_.Value)
    }
}

function New-TempAce {
    $root = Join-Path $env:TEMP ("ace-ps-stop-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $be = Join-Path $ace 'ACE-Backend'
    $fe = Join-Path $ace 'ACE-Frontend'
    New-Item -ItemType Directory -Path $be, $fe -Force | Out-Null
    'v1' | Set-Content -LiteralPath (Join-Path $be 'base.txt') -Encoding UTF8
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260201-150000' | Out-Null
    return @{
        TempRoot = $root
        AceRoot  = $ace
        Backend  = $be
        Frontend = $fe
        Registry = (New-AceLocalManagedRegistryShell -BackendRoot $be -FrontendRoot $fe -LauncherSessionId '20260201-150000')
        CpRoot   = (Join-Path $ace '_backups\recovery-checkpoints-test')
    }
}

function New-OwnedDummyWrapper {
    param([Parameter(Mandatory = $true)][string]$Dir)
    $scriptPath = Join-Path $Dir 'Start-AceLocalRecoveryPeriodicSave.ps1'
    @'
Start-Sleep -Seconds 600
'@ | Set-Content -LiteralPath $scriptPath -Encoding ASCII
    $psExe = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $proc = Start-Process -FilePath $psExe -ArgumentList @('-NoProfile', '-WindowStyle', 'Hidden', '-File', $scriptPath) `
        -PassThru -WindowStyle Hidden -ErrorAction Stop
    Start-Sleep -Milliseconds 400
    $created = Get-Win32ProcessCreationTimeUtc -ProcId $proc.Id
    return @{
        Proc      = $proc
        Pid       = [int]$proc.Id
        CreatedAt = [string]$created
        Script    = $scriptPath
    }
}

function Test-OwnedProcessStillSame {
    param(
        [Parameter(Mandatory = $true)][int]$ProcId,
        [Parameter(Mandatory = $true)][string]$CreatedAt,
        [Parameter(Mandatory = $true)][string]$ScriptPath
    )
    if (-not (Test-ProcessAlive $ProcId)) { return $false }
    $nowCreated = Get-Win32ProcessCreationTimeUtc -ProcId $ProcId
    if (-not (Test-IsoUtcTimesMatchWithinTolerance -ExpectedIso $CreatedAt -ActualIso $nowCreated -ToleranceSeconds 2)) {
        return $false
    }
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction SilentlyContinue
    $cmd = if ($cim) { [string]$cim.CommandLine } else { '' }
    $leaf = [IO.Path]::GetFileName($ScriptPath)
    return ($cmd.IndexOf($leaf, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
}

function Stop-OwnedDummyIfSame {
    param($Dummy)
    if (-not $Dummy) { return }
    if (Test-OwnedProcessStillSame -ProcId $Dummy.Pid -CreatedAt $Dummy.CreatedAt -ScriptPath $Dummy.Script) {
        Stop-Process -Id $Dummy.Pid -Force -ErrorAction SilentlyContinue
    }
}

Write-Host 'Test-AceLocalPeriodicSaveStopContract (isolated)...' -ForegroundColor Cyan

$libText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1') -Raw
Assert-True ($libText -match 'stop_timeout_instance_alive') 'stop_timeout_reason_exists'
Assert-True ($libText -match 'stop_blocked_unresolved_instance') 'stop_unresolved_reason_exists'
Assert-True ($libText -match 'Resolve-AceLocalRecoveryPeriodicSaveOccupancyLiveness') 'liveness_not_blind_occupied'
Assert-True ($libText -notmatch 'if \(\$status\.ServicePid -and -not \$status\.ProcessAlive\)') 'stop_does_not_exit_on_dead_service_pid_alone'

$launcher = '20260201-150000'
$env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'
$layouts = @()
$dummies = @()

try {
    # A — real TEMP wrapper honors stop; verified end.
    $a = New-TempAce
    $layouts += $a
    New-Item -ItemType Directory -Path $a.CpRoot -Force | Out-Null
    $firstA = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $a.Backend -FrontendRoot $a.Frontend `
        -AceRoot $a.AceRoot -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' `
        -AllowPeriodicSaveService $true -TickIntervalSeconds 2 -CheckpointsRootOverride $a.CpRoot `
        -EngineBackendRoot $BackendRoot -ManagedRegistry $a.Registry
    Assert-True ([bool]$firstA.Started) 'a_spawn_ok'
    Start-Sleep -Seconds 1
    $spawnA = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $a.AceRoot
    $occBeforeA = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $a.AceRoot -ExpectedLauncherSession $launcher `
        -ManagedRegistry $a.Registry
    $createdA = Get-Win32ProcessCreationTimeUtc -ProcId ([int]$spawnA.wrapperPid)
    Write-Diag 'A occupancy before Stop' $occBeforeA
    Write-Host ("    wrapperPid={0} createdAt={1} script={2}" -f $spawnA.wrapperPid, $createdA, $spawnA.scriptPath)
    $stopA = Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $a.AceRoot -WaitSeconds 40
    Write-Diag 'A Stop result' $stopA
    $occAfterA = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $a.AceRoot -ExpectedLauncherSession $launcher `
        -ManagedRegistry $a.Registry
    Write-Diag 'A occupancy after Stop' $occAfterA
    $aliveA = Test-ProcessAlive ([int]$spawnA.wrapperPid)
    Assert-True ([bool]$stopA.Stopped) 'a_stop_reports_stopped'
    Assert-True ([string]$stopA.Reason -eq 'stopped') 'a_stop_reason_stopped'
    Assert-True (-not [bool]$occAfterA.Occupied) 'a_occupancy_not_occupied'
    Assert-True ((Resolve-AceLocalRecoveryPeriodicSaveOccupancyLiveness -Occupancy $occAfterA) -eq 'ended') 'a_liveness_ended'
    Assert-True (-not $aliveA) 'a_owned_wrapper_exited'

    # C — stale files only, no live process (after A ended, sibling root).
    $c = New-TempAce
    $layouts += $c
    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $c.AceRoot -State @{
        status = 'running'; launcherSession = $launcher; pid = 999999; spawnToken = 'dead-token'
    }
    Set-Content -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $c.AceRoot) 'recovery-periodic-save.pid') `
        -Value '999999' -Encoding ASCII
    $occC = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $c.AceRoot -ExpectedLauncherSession $launcher
    Assert-True ([string]$occC.Reason -eq 'stale_running_state') 'c_occupancy_stale_running_state'
    $clearC = Clear-AceLocalRecoveryPeriodicSaveProvenStaleState -AceRoot $c.AceRoot -ExpectedLauncherSession $launcher
    Write-Diag 'C Clear' $clearC
    Assert-True ([bool]$clearC.Cleared) 'c_clear_ok'
    Assert-True ([string]$clearC.Reason -eq 'proven_stale_reconciled') 'c_clear_reason'

    # E — start after proven stale cleanup; no second live instance beforehand.
    New-Item -ItemType Directory -Path $c.CpRoot -Force | Out-Null
    $startE = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $c.Backend -FrontendRoot $c.Frontend `
        -AceRoot $c.AceRoot -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' `
        -AllowPeriodicSaveService $true -TickIntervalSeconds 2 -CheckpointsRootOverride $c.CpRoot `
        -EngineBackendRoot $BackendRoot -ManagedRegistry $c.Registry
    Write-Diag 'E Start after Clear' $startE
    Assert-True ([bool]$startE.Started) 'e_start_after_proven_stale'
    $spawnE = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $c.AceRoot
    $dupE = Start-AceLocalRecoveryPeriodicSaveBackgroundProcess -BackendRoot $c.Backend -FrontendRoot $c.Frontend `
        -AceRoot $c.AceRoot -LauncherSession $launcher -CoordinatorOutcome 'START_ALLOWED' `
        -AllowPeriodicSaveService $true -TickIntervalSeconds 2 -CheckpointsRootOverride $c.CpRoot `
        -EngineBackendRoot $BackendRoot -ManagedRegistry $c.Registry
    Assert-True (-not [bool]$dupE.Started) 'e_no_double_spawn'
    $stopE = Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $c.AceRoot -WaitSeconds 40
    Write-Diag 'E Stop of started instance' $stopE
    Assert-True ([bool]$stopE.Stopped) 'e_started_instance_stopped'

    # B — owned dummy ignores stop file; Stop must not report success; spawn kept.
    $b = New-TempAce
    $layouts += $b
    $dummyB = New-OwnedDummyWrapper -Dir $b.TempRoot
    $dummies += $dummyB
    $tokenB = [guid]::NewGuid().ToString('N')
    Write-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $b.AceRoot -Record @{
        wrapperPid       = $dummyB.Pid
        wrapperStartedAt = $dummyB.CreatedAt
        spawnToken       = $tokenB
        launcherSession  = $launcher
        scriptPath       = $dummyB.Script
    }
    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $b.AceRoot -State @{
        status = 'running'; launcherSession = $launcher; spawnToken = $tokenB
    }
    $occBeforeB = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $b.AceRoot -ExpectedLauncherSession $launcher
    Write-Diag 'B occupancy before Stop' $occBeforeB
    Write-Host ("    dummyPid={0} createdAt={1} path={2}" -f $dummyB.Pid, $dummyB.CreatedAt, $dummyB.Script)
    Assert-True ([bool]$occBeforeB.Occupied -and [string]$occBeforeB.Reason -eq 'verified_wrapper') 'b_dummy_verified_live'
    $spawnBeforeB = Get-Content -LiteralPath (Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $b.AceRoot) -Raw -Encoding UTF8
    $stopB = Stop-AceLocalRecoveryPeriodicSaveBackgroundProcess -AceRoot $b.AceRoot -WaitSeconds 2
    Write-Diag 'B Stop result' $stopB
    $occAfterB = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $b.AceRoot -ExpectedLauncherSession $launcher
    Write-Diag 'B occupancy after Stop' $occAfterB
    $spawnAfterB = Get-Content -LiteralPath (Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $b.AceRoot) -Raw -Encoding UTF8
    Assert-True (-not [bool]$stopB.Stopped) 'b_stop_not_success'
    Assert-True ([string]$stopB.Reason -eq 'stop_timeout_instance_alive') 'b_timeout_reason'
    Assert-True ([bool]$occAfterB.Occupied) 'b_still_occupied'
    Assert-True ($spawnBeforeB -eq $spawnAfterB) 'b_spawn_evidence_kept'
    Assert-True (Test-Path -LiteralPath (Get-AceLocalRecoveryPeriodicSaveStopPath -AceRoot $b.AceRoot)) 'b_stop_request_kept'
    Assert-True (Test-OwnedProcessStillSame -ProcId $dummyB.Pid -CreatedAt $dummyB.CreatedAt -ScriptPath $dummyB.Script) `
        'b_owned_dummy_still_same_process'

    # D — live/unverified: Clear refuses. Do not delete spawn of live dummy.
    $clearD = Clear-AceLocalRecoveryPeriodicSaveProvenStaleState -AceRoot $b.AceRoot -ExpectedLauncherSession $launcher
    Write-Diag 'D Clear while dummy live' $clearD
    Assert-True (-not [bool]$clearD.Cleared) 'd_clear_refuses_live'
    Assert-True ([string]$clearD.Reason -eq 'verified_or_unverified_wrapper_alive') 'd_clear_reason_wrapper_alive'
    Assert-True (Test-Path -LiteralPath (Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $b.AceRoot)) 'd_spawn_not_removed'
}
finally {
    Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE -ErrorAction SilentlyContinue
    foreach ($lay in $layouts) {
        if ($lay.AceRoot) {
            Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $lay.AceRoot -ErrorAction SilentlyContinue
        }
    }
    foreach ($d in $dummies) { Stop-OwnedDummyIfSame -Dummy $d }
    Start-Sleep -Milliseconds 400
    foreach ($lay in $layouts) {
        if ($lay.TempRoot -and (Test-Path -LiteralPath $lay.TempRoot)) {
            Remove-Item -LiteralPath $lay.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

if ($failures.Count -gt 0) {
    Write-Host ("FAILED {0}: {1}" -f $failures.Count, ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'ALL PASS' -ForegroundColor Green
exit 0
