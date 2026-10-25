#Requires -Version 5.1
<#
  Isolated tests: runtime JSON identity vs verified periodic-save wrapper.
  TEMP dummy processes only. Does not start/stop live LOCAL or write live recovery data.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalPeriodicSaveRuntimeIdentity (isolated)...' -ForegroundColor Cyan

$libText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveLib.ps1') -Raw
$bgText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1') -Raw

Assert-True ($bgText -match 'function Test-AceLocalRecoveryPeriodicSaveWrapperProcess') 'occupancy_verifies_wrapper_process'
Assert-True ($bgText -match 'wrapper_creation_time_mismatch') 'occupancy_binds_creation_time'
Assert-True ($bgText -match 'spawn_token_mismatch') 'occupancy_binds_spawn_token'
Assert-True ($bgText -match 'wrapperPid') 'spawn_record_has_pid'
Assert-True ($libText -match 'PreserveExistingIdentity') 'preserve_identity_exists'
Assert-True ($libText -match 'VerifiedLauncherSession') 'preserve_requires_verified_origin_session'
Assert-True ($libText -match 'runtime_identity_mismatch') 'preserve_mismatch_reason_exists'
Assert-True ($bgText -match 'Get-AceLocalRecoveryPeriodicSaveVerifiedInstanceIdentity') 'occupancy_identity_passed_to_preserve'
Assert-True ($bgText -match 'already_running') 'already_running_path_exists'
Assert-True ($bgText -notmatch 'Initialize-AceLocalRecoveryLedger') 'already_running_does_not_reinit_ledger'

function New-DummyPeriodicSaveWrapper {
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

function Write-OriginSpawn {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][int]$WrapperPid,
        [Parameter(Mandatory = $true)][string]$WrapperStartedAt,
        [Parameter(Mandatory = $true)][string]$SpawnToken,
        [Parameter(Mandatory = $true)][string]$LauncherSession
    )
    Write-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $AceRoot -Record @{
        wrapperPid       = $WrapperPid
        wrapperStartedAt = $WrapperStartedAt
        spawnToken       = $SpawnToken
        launcherSession  = $LauncherSession
    }
    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $AceRoot -State @{
        status           = 'running'
        launcherSession  = $LauncherSession
        spawnToken       = $SpawnToken
    }
}

function Invoke-PreserveWithOccupancyIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$AttachSession,
        [int]$BackendPort = 5000
    )
    $occ = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $AceRoot -ExpectedLauncherSession $AttachSession
    $id = Get-AceLocalRecoveryPeriodicSaveVerifiedInstanceIdentity -AceRoot $AceRoot -Occupancy $occ
    $beforeRuntime = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $AceRoot
    $beforeSpawn = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $AceRoot
    $beforeLedger = $null
    $ledgerPath = Get-AceLocalRecoveryLedgerPath -AceRoot $AceRoot
    if (Test-Path -LiteralPath $ledgerPath) {
        $beforeLedger = Get-Content -LiteralPath $ledgerPath -Raw -Encoding UTF8
    }
    $merge = Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $AceRoot -BackendPort $BackendPort `
        -LauncherSession $AttachSession -AttachedByLauncherSession $AttachSession -PreserveExistingIdentity `
        -VerifiedLauncherSession $id.LauncherSession -VerifiedWrapperPid $id.WrapperPid `
        -VerifiedWrapperStartedAt $id.WrapperStartedAt -VerifiedSpawnToken $id.SpawnToken
    return @{
        Occupancy    = $occ
        Identity     = $id
        Merge        = $merge
        BeforeRuntime = $beforeRuntime
        BeforeSpawn  = $beforeSpawn
        BeforeLedger = $beforeLedger
    }
}

$tmpRoot = Join-Path $env:TEMP ("ace-ps-runtime-id-{0}" -f [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null
$dummy = $null

try {
    $dummy = New-DummyPeriodicSaveWrapper -Dir $tmpRoot
    Assert-True ($dummy.Pid -gt 0) 'dummy_wrapper_started'
    Assert-True (-not [string]::IsNullOrWhiteSpace($dummy.CreatedAt)) 'dummy_creation_time_readable'

    $originSession = '20261024-040249'
    $attachSession = '20261025-044706'
    $foreignSession = '20261020-999999'
    $token = [guid]::NewGuid().ToString('N')

    $aceA = Join-Path $tmpRoot 'ace-a'
    New-Item -ItemType Directory -Path $aceA -Force | Out-Null
    Write-OriginSpawn -AceRoot $aceA -WrapperPid $dummy.Pid -WrapperStartedAt $dummy.CreatedAt `
        -SpawnToken $token -LauncherSession $originSession
    [void](Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceA -BackendPort 5000 `
        -LauncherSession $originSession)

    $occA = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceA -ExpectedLauncherSession $attachSession
    Assert-True ([bool]$occA.Occupied) 'a_live_service_proven_occupied'
    Assert-True ([string]$occA.Reason -eq 'verified_wrapper') 'a_occupancy_verified_wrapper_not_runtime'
    $wrapA = Test-AceLocalRecoveryPeriodicSaveWrapperProcess -WrapperPid $dummy.Pid `
        -ExpectedWrapperStartedAt $dummy.CreatedAt -ExpectedSpawnToken $token `
        -ServiceState (Get-Content -LiteralPath (Get-AceLocalRecoveryPeriodicSaveStatePath -AceRoot $aceA) -Raw -Encoding UTF8 | ConvertFrom-Json)
    Assert-True ([bool]$wrapA.Verified) 'a_pid_creation_token_bind_wrapper'

    $preserveA = Invoke-PreserveWithOccupancyIdentity -AceRoot $aceA -AttachSession $attachSession
    Assert-True ([bool]$preserveA.Identity.Verified) 'a_verified_identity_from_occupancy'
    Assert-True ([string]$preserveA.Identity.LauncherSession -eq $originSession) 'a_verified_session_is_origin_not_attach'
    Assert-True ([bool]$preserveA.Merge.Ok) 'a_preserve_ok'
    $rtA = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceA
    Assert-True ([string]$rtA.launcherSession -eq $originSession) 'a_origin_identity_kept'
    Assert-True ([string]$rtA.attachedByLauncherSession -eq $attachSession) 'a_attached_by_separate'
    Assert-True ($null -eq $rtA.PSObject.Properties['wrapperPid'] -or -not $rtA.wrapperPid) 'a_runtime_still_has_no_pid'
    $spawnA = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceA
    Assert-True ([string]$spawnA.launcherSession -eq [string]$rtA.launcherSession) 'a_runtime_session_matches_verified_spawn'

    # B — verified process (origin spawn) + foreign runtime file.
    $aceB = Join-Path $tmpRoot 'ace-b'
    New-Item -ItemType Directory -Path $aceB -Force | Out-Null
    Write-OriginSpawn -AceRoot $aceB -WrapperPid $dummy.Pid -WrapperStartedAt $dummy.CreatedAt `
        -SpawnToken $token -LauncherSession $originSession
    [void](Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceB -BackendPort 5000 `
        -LauncherSession $foreignSession)
    $occB = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceB -ExpectedLauncherSession $attachSession
    Assert-True ([bool]$occB.Occupied -and [string]$occB.Reason -eq 'verified_wrapper') 'b_process_verified_independently_of_runtime'
    $rtBBefore = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceB
    Assert-True ([string]$rtBBefore.launcherSession -eq $foreignSession) 'b_runtime_file_is_foreign'
    $preserveB = Invoke-PreserveWithOccupancyIdentity -AceRoot $aceB -AttachSession $attachSession
    $rtB = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceB
    $spawnB = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceB
    Assert-True ([bool]$preserveB.Identity.Verified) 'b_process_identity_still_verified'
    Assert-True (-not [bool]$preserveB.Merge.Ok) 'b_foreign_runtime_not_accepted'
    Assert-True ([string]$preserveB.Merge.Reason -eq 'runtime_identity_mismatch') 'b_foreign_reason_mismatch'
    Assert-True (-not [bool]$preserveB.Merge.Matched) 'b_not_reported_as_matched'
    Assert-True ([string]$rtB.launcherSession -eq $foreignSession) 'b_foreign_file_not_overwritten'
    Assert-True ([string]$rtB.launcherSession -ne [string]$spawnB.launcherSession) `
        'b_foreign_runtime_must_not_become_verified_process_identity'
    Assert-True ([string]$spawnB.launcherSession -eq $originSession) 'b_spawn_unchanged'
    Assert-True (-not (Test-Path -LiteralPath (Get-AceLocalRecoveryLedgerPath -AceRoot $aceB))) 'b_no_ledger_write'

    # C — same PID, wrong creation time / token: not a verified match.
    $staleCreated = ([DateTime]::UtcNow.AddHours(-6)).ToString('o')
    $wrapCTime = Test-AceLocalRecoveryPeriodicSaveWrapperProcess -WrapperPid $dummy.Pid `
        -ExpectedWrapperStartedAt $staleCreated
    Assert-True (-not [bool]$wrapCTime.Verified) 'c_creation_mismatch_not_verified'
    Assert-True ([string]$wrapCTime.Reason -eq 'wrapper_creation_time_mismatch') 'c_creation_mismatch_reason'
    Assert-True ([bool]$wrapCTime.Alive) 'c_pid_alive_is_not_enough'

    $wrongTokenState = [pscustomobject]@{ spawnToken = 'ffffffffffffffffffffffffffffffff' }
    $wrapCTok = Test-AceLocalRecoveryPeriodicSaveWrapperProcess -WrapperPid $dummy.Pid `
        -ExpectedWrapperStartedAt $dummy.CreatedAt -ExpectedSpawnToken $token -ServiceState $wrongTokenState
    Assert-True (-not [bool]$wrapCTok.Verified) 'c_token_mismatch_not_verified'
    Assert-True ([string]$wrapCTok.Reason -eq 'spawn_token_mismatch') 'c_token_mismatch_reason'

    $aceC = Join-Path $tmpRoot 'ace-c'
    New-Item -ItemType Directory -Path $aceC -Force | Out-Null
    Write-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceC -Record @{
        wrapperPid       = $dummy.Pid
        wrapperStartedAt = $staleCreated
        spawnToken       = $token
        launcherSession  = $originSession
    }
    $occC = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceC
    Assert-True ([string]$occC.Reason -ne 'verified_wrapper') 'c_occupancy_does_not_treat_pid_reuse_as_verified_match'

    $aceC2 = Join-Path $tmpRoot 'ace-c2'
    New-Item -ItemType Directory -Path $aceC2 -Force | Out-Null
    Write-OriginSpawn -AceRoot $aceC2 -WrapperPid $dummy.Pid -WrapperStartedAt $dummy.CreatedAt `
        -SpawnToken $token -LauncherSession $originSession
    [void](Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceC2 -BackendPort 5000 `
        -LauncherSession $originSession)
    $rtPathC2 = Get-AceLocalRecoveryPeriodicSaveRuntimePath -AceRoot $aceC2
    $rtC2obj = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceC2
    $rtC2obj | Add-Member -NotePropertyName 'wrapperPid' -NotePropertyValue $dummy.Pid -Force
    $rtC2obj | Add-Member -NotePropertyName 'wrapperStartedAt' -NotePropertyValue $staleCreated -Force
    Write-AceLocalRecoveryJsonAtomic -Path $rtPathC2 -Object $rtC2obj
    $preserveC2 = Invoke-PreserveWithOccupancyIdentity -AceRoot $aceC2 -AttachSession $attachSession
    Assert-True ([bool]$preserveC2.Identity.Verified) 'c2_occupancy_verified'
    Assert-True (-not [bool]$preserveC2.Merge.Ok) 'c2_runtime_creation_conflict_rejected'
    Assert-True ([string]$preserveC2.Merge.Reason -eq 'runtime_identity_mismatch') 'c2_creation_conflict_mismatch'
    $rtC2after = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceC2
    Assert-True ([string]$rtC2after.wrapperStartedAt -eq $staleCreated) 'c2_runtime_not_rewritten'

    $aceC3 = Join-Path $tmpRoot 'ace-c3'
    New-Item -ItemType Directory -Path $aceC3 -Force | Out-Null
    Write-OriginSpawn -AceRoot $aceC3 -WrapperPid $dummy.Pid -WrapperStartedAt $dummy.CreatedAt `
        -SpawnToken $token -LauncherSession $originSession
    [void](Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceC3 -BackendPort 5000 `
        -LauncherSession $originSession)
    $rtC3 = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceC3
    $rtC3 | Add-Member -NotePropertyName 'spawnToken' -NotePropertyValue 'ffffffffffffffffffffffffffffffff' -Force
    Write-AceLocalRecoveryJsonAtomic -Path (Get-AceLocalRecoveryPeriodicSaveRuntimePath -AceRoot $aceC3) -Object $rtC3
    $preserveC3 = Invoke-PreserveWithOccupancyIdentity -AceRoot $aceC3 -AttachSession $attachSession
    Assert-True (-not [bool]$preserveC3.Merge.Ok -and [string]$preserveC3.Merge.Reason -eq 'runtime_identity_mismatch') `
        'c3_runtime_token_conflict_rejected'

    # D — missing runtime / missing bind fields: identity unknown, no invented session, no spawn.
    $aceD = Join-Path $tmpRoot 'ace-d'
    New-Item -ItemType Directory -Path $aceD -Force | Out-Null
    $rtD0 = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceD
    Assert-True ($null -eq $rtD0) 'd_runtime_absent'
    $preserveD = Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceD -BackendPort 5000 `
        -LauncherSession $attachSession -AttachedByLauncherSession $attachSession -PreserveExistingIdentity `
        -VerifiedLauncherSession $originSession -VerifiedWrapperPid $dummy.Pid `
        -VerifiedWrapperStartedAt $dummy.CreatedAt -VerifiedSpawnToken $token
    Assert-True (-not [bool]$preserveD.Ok) 'd_preserve_refuses_unknown'
    Assert-True ([string]$preserveD.Reason -eq 'runtime_identity_unknown') 'd_reason_unknown'
    $rtD1 = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceD
    Assert-True ($null -eq $rtD1) 'd_does_not_invent_runtime_session'
    $spawnD = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceD
    Assert-True ($null -eq $spawnD) 'd_does_not_double_spawn'
    $occD = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceD -ExpectedLauncherSession $attachSession
    Assert-True (-not [bool]$occD.Occupied) 'd_missing_bind_not_occupied_as_verified'
    Assert-True ([string]$occD.Reason -ne 'verified_wrapper') 'd_not_verified_wrapper'

    $aceD2 = Join-Path $tmpRoot 'ace-d2'
    New-Item -ItemType Directory -Path $aceD2 -Force | Out-Null
    Write-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceD2 -Record @{
        launcherSession = $originSession
    }
    $occD2 = Test-AceLocalRecoveryPeriodicSaveInstanceOccupied -AceRoot $aceD2
    Assert-True (-not [bool]$occD2.Occupied -or [string]$occD2.Reason -ne 'verified_wrapper') 'd_spawn_without_pid_not_verified'
    $idD2 = Get-AceLocalRecoveryPeriodicSaveVerifiedInstanceIdentity -AceRoot $aceD2 -Occupancy $occD2
    Assert-True (-not [bool]$idD2.Verified) 'd_incomplete_occupancy_not_verified_identity'

    $aceD3 = Join-Path $tmpRoot 'ace-d3'
    New-Item -ItemType Directory -Path $aceD3 -Force | Out-Null
    Write-OriginSpawn -AceRoot $aceD3 -WrapperPid $dummy.Pid -WrapperStartedAt $dummy.CreatedAt `
        -SpawnToken $token -LauncherSession $originSession
    [void](Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceD3 -BackendPort 5000 `
        -LauncherSession $originSession)
    $preserveD3 = Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceD3 -BackendPort 5000 `
        -LauncherSession $attachSession -AttachedByLauncherSession $attachSession -PreserveExistingIdentity
    Assert-True (-not [bool]$preserveD3.Ok -and [string]$preserveD3.Reason -eq 'runtime_identity_unknown') `
        'd_old_runtime_without_verified_identity_is_unknown'
    $rtD3 = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceD3
    Assert-True ([string]$rtD3.launcherSession -eq $originSession) 'd_unknown_does_not_rewrite_runtime'
    Assert-True ($null -eq $rtD3.PSObject.Properties['attachedByLauncherSession'] -or -not $rtD3.attachedByLauncherSession) `
        'd_unknown_does_not_attach'

    # E — new launcher attach: ledger session unchanged.
    $aceE = Join-Path $tmpRoot 'ace-e'
    New-Item -ItemType Directory -Path $aceE -Force | Out-Null
    [void](Initialize-AceLocalRecoveryLedger -AceRoot $aceE -LauncherSession $originSession)
    Write-OriginSpawn -AceRoot $aceE -WrapperPid $dummy.Pid -WrapperStartedAt $dummy.CreatedAt `
        -SpawnToken $token -LauncherSession $originSession
    [void](Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceE -BackendPort 5000 `
        -LauncherSession $originSession)
    $ledgerBeforeE = Get-Content -LiteralPath (Get-AceLocalRecoveryLedgerPath -AceRoot $aceE) -Raw -Encoding UTF8
    $preserveE = Invoke-PreserveWithOccupancyIdentity -AceRoot $aceE -AttachSession $attachSession
    Assert-True ([bool]$preserveE.Merge.Ok) 'e_preserve_ok'
    $ledgerAfterE = Get-Content -LiteralPath (Get-AceLocalRecoveryLedgerPath -AceRoot $aceE) -Raw -Encoding UTF8
    Assert-True ($ledgerBeforeE -eq $ledgerAfterE) 'e_ledger_bytes_unchanged'
    $ledgerE = Read-AceLocalRecoveryLedger -AceRoot $aceE -AllowCreateStateDir
    Assert-True ([string]$ledgerE.launcherSession -eq $originSession) 'e_ledger_session_unchanged'
    $rtE = Read-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $aceE
    Assert-True ([string]$rtE.launcherSession -eq $originSession) 'e_runtime_session_unchanged'
    Assert-True ([string]$rtE.attachedByLauncherSession -eq $attachSession) 'e_attach_recorded_separately'
    $spawnE = Read-AceLocalRecoveryPeriodicSaveSpawnRecord -AceRoot $aceE
    Assert-True ([string]$spawnE.launcherSession -eq $originSession) 'e_spawn_session_unchanged'
}
finally {
    if ($dummy -and $dummy.Proc -and -not $dummy.Proc.HasExited) {
        Stop-Process -Id $dummy.Pid -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ''
if ($failures.Count -gt 0) {
    Write-Host ("FAILED {0}: {1}" -f $failures.Count, ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'ALL PASS' -ForegroundColor Green
exit 0
