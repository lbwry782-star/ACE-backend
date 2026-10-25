#Requires -Version 5.1
<#
  Isolated tests for LOCAL managed PID registry and helpers (no stack start/stop on live services).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')

$BackendRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$FrontendRoot = Resolve-AceFrontendRoot -BackendRoot $BackendRoot
if (-not $FrontendRoot) {
    Write-Host ' FAIL frontend_root_missing' -ForegroundColor Red
    exit 1
}

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalManagedProcess (isolated)...' -ForegroundColor Cyan

$startLocalText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startLocalText -match 'ShowComponentWindows') 'start_ace_local_has_diagnostic_switch'
Assert-True ($startLocalText -match 'Start-ComponentSilent') 'start_ace_local_has_silent_start'
Assert-True ($startLocalText -match 'Discover-AceLocalManagedBackendTree') 'start_local_discovers_backend_tree'
Assert-True ($startLocalText -match 'Register-ManagedComponentTree') 'start_local_registers_tree'
Assert-True ($startLocalText -match 'Sync-AceLocalAlreadyRunningManagedComponent') 'start_local_syncs_already_running_ownership'
Assert-True ($startLocalText -notmatch '\$managedRegistry\.components\.backend = \$null') 'start_local_does_not_null_already_running_backend'
Assert-True ($startLocalText -match 'Start-Transcript') 'start_ace_local_has_launcher_log'
Assert-True ($startLocalText -match 'if \(\$SilentMode\)[\s\S]*Stop managed stack') 'silent_mode_shows_log_and_stop_hint'
Assert-True ($startLocalText -match 'ShowComponentWindows[\s\S]*Read-Host "Press Enter to close this launcher') 'diagnostic_mode_keeps_launcher_readhost'
Assert-True (Test-Path -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1')) 'stop_ace_local_exists'

$cmdText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.cmd') -Raw
Assert-True ($cmdText -match '-WindowStyle Hidden') 'start_ace_local_cmd_hidden'

$tmpAce = Join-Path $env:TEMP ("ace-local-test-{0}" -f [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmpAce -Force | Out-Null
try {
    $reg = New-AceLocalManagedRegistryShell -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -LauncherSessionId 'test-session'
    Write-AceLocalManagedRegistry -AceRoot $tmpAce -RegistryObject $reg
    $read = Read-AceLocalManagedRegistry -AceRoot $tmpAce
    Assert-True ($null -ne $read) 'registry_roundtrip'
    Assert-True ($read.launcherSession -eq 'test-session') 'registry_session_id'

    $reg.components.backend = New-AceLocalManagedComponentRecord -ProcId 999999 -LogPath 'C:\fake\backend.log'
    Assert-True (-not (Test-ProcessAlive 999999)) 'stale_pid_not_alive'
    $changed = Remove-StaleManagedComponents -Registry $reg -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
    Assert-True $changed 'stale_pid_removed'
    Assert-True ($null -eq $reg.components.backend) 'stale_backend_cleared'

    $before = @(100, 200)
    $after = @(100, 200, 301)
    Assert-True ((Resolve-SingleNewProcessId -Before $before -After $after) -eq 301) 'resolve_single_new_pid'
    $afterDup = @(100, 301, 302)
    Assert-True ($null -eq (Resolve-SingleNewProcessId -Before $before -After $afterDup)) 'resolve_ambiguous_new_pid'

    Assert-True ((Resolve-SingleNewProcessId -Before @() -After @(4242)) -eq 4242) 'empty_before_single_after_binding'
    Assert-True ($null -eq (Resolve-SingleNewProcessId -Before @() -After @())) 'empty_before_empty_after'
    Assert-True ($null -eq (Resolve-SingleNewProcessId -Before @() -After @(11, 22))) 'empty_before_multiple_after_no_guess'
    Assert-True ($null -eq (Resolve-SingleNewProcessId -Before @(1) -After @())) 'empty_after'

    $parentMap = @{
        30980 = 8208
        8208  = 33008
        20952 = 29604
        29604 = 9848
    }
    $backendTree = @(
        [PSCustomObject]@{ ProcId = 8208 }
        [PSCustomObject]@{ ProcId = 30980 }
    )
    $roots = @(Get-AceIndependentTreeRootIds -Processes $backendTree -ParentProcessIdByPid $parentMap)
    Assert-True ($roots.Count -eq 1 -and $roots[0] -eq 8208) 'parent_child_single_root'
    $members = @(Get-AceProcessTreeMemberIds -RootPid 8208 -Processes $backendTree -ParentProcessIdByPid $parentMap)
    Assert-True ($members.Count -eq 2 -and $members -contains 30980) 'tree_members_parent_child'

    $beforeProcs = @([PSCustomObject]@{ ProcId = 100 })
    $afterProcs = @(
        [PSCustomObject]@{ ProcId = 100 }
        [PSCustomObject]@{ ProcId = 8208 }
        [PSCustomObject]@{ ProcId = 30980 }
    )
    $afterMap = @{ 30980 = 8208; 8208 = 33008; 100 = 50 }
    $newRoot = Find-NewAceIndependentTreeRoot -BeforeProcesses $beforeProcs -AfterProcesses $afterProcs -ParentProcessIdByPid $afterMap
    Assert-True ($newRoot -eq 8208) 'find_new_tree_root_one'

    $afterTwoTrees = @(
        [PSCustomObject]@{ ProcId = 100 }
        [PSCustomObject]@{ ProcId = 200 }
        [PSCustomObject]@{ ProcId = 201 }
        [PSCustomObject]@{ ProcId = 8208 }
        [PSCustomObject]@{ ProcId = 30980 }
    )
    $twoMap = @{
        201   = 200
        200   = 9999
        30980 = 8208
        8208  = 9998
        100   = 50
    }
    $ambiguous = Find-NewAceIndependentTreeRoot -BeforeProcesses $beforeProcs -AfterProcesses $afterTwoTrees -ParentProcessIdByPid $twoMap
    Assert-True ($null -eq $ambiguous) 'find_new_tree_rejects_parallel_foreign'

    $treeRec = New-AceLocalManagedComponentTreeRecord -RootPid 8208 -TreePids @(8208, 30980) -LogPath 'C:\fake\log' -LauncherSession 'sess-1' -WrapperPid 33008 -ListenerPid 30980
    Assert-True ($treeRec.rootPid -eq 8208 -and $treeRec.treePids.Count -eq 2) 'tree_record_shape'
    Assert-True ($treeRec.launcherSession -eq 'sess-1') 'tree_record_session'

    $reg.components.backend = $treeRec
    $actions = @(Get-ManagedComponentStopActions -Registry $reg -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot)
    $backendAction = $actions | Where-Object { $_.Component -eq 'backend' } | Select-Object -First 1
    Assert-True ($null -ne $backendAction) 'stop_actions_has_backend'
    Assert-True ($backendAction.Action -eq 'skip') 'stop_skips_unverified_live_tree'

    $livePid = [int]$PID
    $liveCreated = Get-Win32ProcessCreationTimeUtc -ProcId $livePid
    if ($liveCreated) {
        $fakeMap = @{ ([string]$livePid) = $liveCreated }
        $fakeRec = New-AceLocalManagedComponentTreeRecord -RootPid $livePid -TreePids @($livePid) `
            -LogPath 'C:\fake\reuse.log' -LauncherSession 'reuse-test' -ProcessCreatedAt $fakeMap
        $creationOk = Test-ManagedProcessCreationTimesValid -Record $fakeRec -TreePids @($livePid)
        Assert-True $creationOk.Ok 'creation_time_matches_live_pid'
        $wrongMap = @{ ([string]$livePid) = '2010-01-01T00:00:00.0000000Z' }
        $wrongRec = New-AceLocalManagedComponentTreeRecord -RootPid $livePid -TreePids @($livePid) `
            -LogPath 'C:\fake\reuse.log' -LauncherSession 'reuse-test' -ProcessCreatedAt $wrongMap
        $creationBad = Test-ManagedProcessCreationTimesValid -Record $wrongRec -TreePids @($livePid)
        Assert-True (-not $creationBad.Ok) 'creation_time_mismatch_rejected'
        Assert-True ($creationBad.Reason -like 'process_creation_time_mismatch_pid_*') 'creation_time_mismatch_reason'
    }
    else {
        Write-Host '  skip live PID creation-time checks (creation time unavailable)' -ForegroundColor DarkYellow
    }

    $oldStyleRec = New-AceLocalManagedComponentTreeRecord -RootPid 8208 -TreePids @(8208, 30980) `
        -LogPath 'C:\fake\old.log' -LauncherSession 'test-session'
    $missingCreation = Test-ManagedProcessCreationTimesValid -Record $oldStyleRec -TreePids @(8208, 30980)
    Assert-True (-not $missingCreation.Ok) 'legacy_record_missing_creation_times'
    $reg.components.backend = $oldStyleRec
    $oldActions = @(Get-ManagedComponentStopActions -Registry $reg -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot)
    $oldAct = $oldActions | Where-Object { $_.Component -eq 'backend' } | Select-Object -First 1
    Assert-True ($oldAct.Action -eq 'skip') 'legacy_record_stop_skipped'
    Assert-True ($oldAct.Reason -in @(
            'missing_process_creation_times',
            'tree_not_running',
            'root_fingerprint_mismatch'
        )) 'legacy_skip_conservative_reason'

    $reg.components.backend = New-AceLocalManagedComponentRecord -ProcId 999998 -LogPath 'C:\fake\reboot.log'
    $rebootChanged = Remove-StaleManagedComponents -Registry $reg -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
    Assert-True $rebootChanged 'reboot_stale_dead_tree_removed'
    Assert-True ($null -eq $reg.components.backend) 'reboot_stale_cleared'

    Assert-True ($script:ManagedComponentInactiveReasons -notcontains 'missing_process_creation_times') `
        'missing_creation_times_not_treated_as_inactive'
    Assert-True ($script:ManagedComponentInactiveReasons -notcontains 'process_creation_time_mismatch_pid_1') `
        'creation_mismatch_not_treated_as_inactive'

    $corruptPath = Get-AceLocalManagedRegistryPath -AceRoot $tmpAce
    Set-Content -LiteralPath $corruptPath -Value '{ not valid json' -Encoding UTF8
    $corruptRead = Read-AceLocalManagedRegistry -AceRoot $tmpAce
    Assert-True ($null -eq $corruptRead) 'corrupt_managed_json_returns_null'
    $tmpPartial = Join-Path (Split-Path -Parent $corruptPath) 'managed-processes.json.tmp.partial-test'
    Set-Content -LiteralPath $tmpPartial -Value '{"schemaVersion":2' -Encoding UTF8
    Assert-True (Test-Path -LiteralPath $corruptPath) 'atomic_main_unchanged_when_only_tmp_exists'

    $logDir = Get-AceLocalLogDir -AceRoot $tmpAce
    Assert-True (Test-Path -LiteralPath $logDir) 'log_dir_created'

    $libPath = Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1'
    $cpPath = Join-Path $ScriptDir 'AceLocalRecoveryCheckpointLib.ps1'
    $parseErr = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile($libPath, [ref]$null, [ref]$parseErr)
    Assert-True (-not $parseErr -or $parseErr.Count -eq 0) 'managed_process_lib_ast_parse'
    $parseErrCp = $null
    $null = [System.Management.Automation.Language.Parser]::ParseFile($cpPath, [ref]$null, [ref]$parseErrCp)
    Assert-True (-not $parseErrCp -or $parseErrCp.Count -eq 0) 'recovery_checkpoint_lib_ast_parse'

    $hiddenLogRoot = Join-Path $env:TEMP ("ace-hidden-log-{0}" -f [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $hiddenLogRoot -Force | Out-Null
    try {
        $fastScript = Join-Path $hiddenLogRoot 'fast.ps1'
        @'
Set-StrictMode -Version Latest
Write-Output 'ACE_HIDDEN_STDOUT_FAST'
[Console]::Error.WriteLine('ACE_HIDDEN_STDERR_FAST')
exit 0
'@ | Set-Content -LiteralPath $fastScript -Encoding UTF8
        $fastOut = Join-Path $hiddenLogRoot 'fast.stdout.log'
        $fastErr = Join-Path $hiddenLogRoot 'fast.stderr.log'
        $fastLaunch = Start-AceLocalHiddenPowerShell -ScriptPath $fastScript -WorkingDirectory $hiddenLogRoot `
            -StdoutLog $fastOut -StderrLog $fastErr
        $null = $fastLaunch.Process.WaitForExit(10000)
        Close-AceLocalHiddenPowerShellStreams -LaunchHandle $fastLaunch -JoinMs 10000
        $fastOutText = if (Test-Path $fastOut) { (Get-Content -LiteralPath $fastOut -Raw) } else { '' }
        $fastErrText = if (Test-Path $fastErr) { (Get-Content -LiteralPath $fastErr -Raw) } else { '' }
        Assert-True ($fastOutText -match 'ACE_HIDDEN_STDOUT_FAST') 'hidden_fast_stdout_captured'
        Assert-True ($fastErrText -match 'ACE_HIDDEN_STDERR_FAST') 'hidden_fast_stderr_captured'

        $longScript = Join-Path $hiddenLogRoot 'long.ps1'
        @'
Set-StrictMode -Version Latest
Write-Output 'ACE_HIDDEN_STDOUT_LONG_START'
Start-Sleep -Seconds 8
Write-Output 'ACE_HIDDEN_STDOUT_LONG_END'
exit 0
'@ | Set-Content -LiteralPath $longScript -Encoding UTF8
        $longOut = Join-Path $hiddenLogRoot 'long.stdout.log'
        $longErr = Join-Path $hiddenLogRoot 'long.stderr.log'
        $longLaunch = Start-AceLocalHiddenPowerShell -ScriptPath $longScript -WorkingDirectory $hiddenLogRoot `
            -StdoutLog $longOut -StderrLog $longErr
        $longPid = [int]$longLaunch.WrapperProcessId
        Assert-True ($longPid -gt 0) 'hidden_long_wrapper_pid'
        Start-Sleep -Milliseconds 400
        Assert-True (-not $longLaunch.Process.HasExited) 'hidden_long_survives_after_launch_return'
        Assert-True (Test-ProcessAlive $longPid) 'hidden_long_pid_alive'
        $null = $longLaunch.Process.WaitForExit(20000)
        Close-AceLocalHiddenPowerShellStreams -LaunchHandle $longLaunch -JoinMs 10000
        $longOutText = (Get-Content -LiteralPath $longOut -Raw)
        Assert-True ($longOutText -match 'ACE_HIDDEN_STDOUT_LONG_START') 'hidden_long_start_line'
        Assert-True ($longOutText -match 'ACE_HIDDEN_STDOUT_LONG_END') 'hidden_long_end_line_no_loss'
    }
    finally {
        Remove-Item -LiteralPath $hiddenLogRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
finally {
    Remove-Item -LiteralPath $tmpAce -Recurse -Force -ErrorAction SilentlyContinue
}

$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($stopText -match 'Get-ManagedComponentStopActions') 'stop_uses_managed_actions'
Assert-True ($stopText -match 'Confirm-ManagedComponentStopAllowed') 'stop_revalidates_before_kill'
Assert-True ($stopText -match 'treePids') 'stop_whatif_lists_tree'
Assert-True ($stopText -notmatch 'Stop-Service') 'stop_does_not_stop_memurai_service'
Assert-True ($script:AceLocalManagedRegistrySchemaVersion -eq 2) 'registry_schema_v2'

$startMemuraiText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startMemuraiText -match 'MemuraiStartupFailureDetail') 'memurai_failure_detail_logged'
Assert-True ($startMemuraiText -match 'Fail-AceLocalProgressState') 'memurai_failure_fails_progress'
Assert-True ($startMemuraiText -match 'if \(-not \$AllowInteractive\)') 'silent_mode_memurai_branch'
Assert-True ($startMemuraiText -match 'Silent mode does not start services automatically') 'silent_mode_no_auto_start_message'

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalManagedProcess tests passed.' -ForegroundColor Green
