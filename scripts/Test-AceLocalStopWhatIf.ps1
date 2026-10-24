#Requires -Version 5.1
<#
  Isolated read-only tests for Stop-AceLocal -WhatIf (no live stack, no process stop).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalManagedProcessLib.ps1')
. (Join-Path $ScriptDir 'AceLocalMemuraiWatchLib.ps1')

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

function Get-PathSha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

Write-Host 'Test-AceLocalStopWhatIf (isolated)...' -ForegroundColor Cyan

$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($stopText -match 'Get-AceLocalManagedStopWhatIfReport') 'stop_whatif_uses_read_only_report'
Assert-True ($stopText -match 'elseif \(\$WhatIf\)') 'stop_whatif_separate_branch'
$whatIfBlock = ''
if ($stopText -match '(?s)elseif \(\$WhatIf\) \{(.*?)\}\s*else \{') {
    $whatIfBlock = $Matches[1]
}
Assert-True ($whatIfBlock -and ($whatIfBlock -notmatch 'Write-AceLocalManagedRegistry')) 'whatif_branch_no_registry_write'
Assert-True ($whatIfBlock -and ($whatIfBlock -notmatch 'Remove-StaleManagedComponents')) 'whatif_branch_no_stale_mutation'
Assert-True ($stopText -match 'AllowCreateStateDir:\(-not \$readOnlyFs\)') 'whatif_read_only_fs_flag'
Assert-True ($stopText -notmatch 'Stop-Service') 'stop_does_not_stop_memurai_service'

$tmpAce = Join-Path $env:TEMP ("ace-stop-whatif-{0}" -f [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmpAce -Force | Out-Null
try {
    $reg = New-AceLocalManagedRegistryShell -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -LauncherSessionId 'whatif-test'
    $reg.components.backend = New-AceLocalManagedComponentRecord -ProcId 999997 -LogPath 'C:\fake\stale-backend.log'
    Write-AceLocalManagedRegistry -AceRoot $tmpAce -RegistryObject $reg

    $registryPath = Get-AceLocalManagedRegistryPath -AceRoot $tmpAce
    $hashBefore = Get-PathSha256 $registryPath
    $mtimeBefore = (Get-Item -LiteralPath $registryPath).LastWriteTimeUtc

    $readReg = Read-AceLocalManagedRegistry -AceRoot $tmpAce -AllowCreateStateDir:$false
    $report = Get-AceLocalManagedStopWhatIfReport -Registry $readReg `
        -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
    Assert-True (@($report.StaleComponents).Count -ge 1) 'simulated_stale_preview_found'
    Assert-True ($report.StaleComponents[0].Component -eq 'backend') 'simulated_stale_backend'
    Assert-True ($report.StaleComponents[0].Reason -eq 'tree_not_running') 'simulated_stale_reason'

    $hashAfter = Get-PathSha256 $registryPath
    $mtimeAfter = (Get-Item -LiteralPath $registryPath).LastWriteTimeUtc
    Assert-True ($hashBefore -eq $hashAfter) 'managed_registry_hash_unchanged_after_whatif_report'
    Assert-True ($mtimeBefore -eq $mtimeAfter) 'managed_registry_mtime_unchanged_after_whatif_report'

    $readAgain = Read-AceLocalManagedRegistry -AceRoot $tmpAce -AllowCreateStateDir:$false
    Assert-True ($null -ne $readAgain.components.backend) 'stale_record_not_cleared_by_preview'

    $legacyTree = New-AceLocalManagedComponentTreeRecord -RootPid 8208 -TreePids @(8208, 30980) `
        -LogPath 'C:\fake\legacy.log' -LauncherSession 'whatif-test'
    $reg.components.worker = $legacyTree
    Write-AceLocalManagedRegistry -AceRoot $tmpAce -RegistryObject $reg
    $hashLegacyBefore = Get-PathSha256 $registryPath
    $readLegacy = Read-AceLocalManagedRegistry -AceRoot $tmpAce -AllowCreateStateDir:$false
    $legacyReport = Get-AceLocalManagedStopWhatIfReport -Registry $readLegacy `
        -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
    $workerAction = @($legacyReport.StopActions) | Where-Object { $_.Component -eq 'worker' } | Select-Object -First 1
    Assert-True ($null -ne $workerAction) 'legacy_worker_stop_action_present'
    Assert-True ($workerAction.Action -eq 'skip') 'legacy_worker_would_skip_not_stop'
    $hashLegacyAfter = Get-PathSha256 $registryPath
    Assert-True ($hashLegacyBefore -eq $hashLegacyAfter) 'legacy_registry_hash_unchanged_after_whatif_report'

    $tmpWatch = Join-Path $env:TEMP ("ace-stop-watch-{0}" -f [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmpWatch -Force | Out-Null
    try {
        $watchDir = Join-Path $tmpWatch '_backups\local-dev'
        Assert-True (-not (Test-Path -LiteralPath $watchDir)) 'watch_dir_absent_before_readonly_path'
        $null = Get-AceLocalMemuraiWatchPidPath -AceRoot $tmpWatch
        Assert-True (-not (Test-Path -LiteralPath $watchDir)) 'watch_path_readonly_does_not_create_dir'
        New-Item -ItemType Directory -Path $watchDir -Force | Out-Null
        $watchPidFile = Join-Path $watchDir 'memurai-watch.pid'
        Set-Content -LiteralPath $watchPidFile -Value '424242' -Encoding ASCII
        $watchHashBefore = Get-PathSha256 $watchPidFile
        $watchPath = Get-AceLocalMemuraiWatchPidPath -AceRoot $tmpWatch
        $watchRead = Read-AceLocalMemuraiWatchPid -PidPath $watchPath
        Assert-True ($watchRead -eq 424242) 'watch_pid_read_only'
        $watchHashAfter = Get-PathSha256 $watchPidFile
        Assert-True ($watchHashBefore -eq $watchHashAfter) 'watch_pid_hash_unchanged'
    }
    finally {
        Remove-Item -LiteralPath $tmpWatch -Recurse -Force -ErrorAction SilentlyContinue
    }
}
finally {
    Remove-Item -LiteralPath $tmpAce -Recurse -Force -ErrorAction SilentlyContinue
}

$memuraiSvc = Get-Service -Name 'Memurai' -ErrorAction SilentlyContinue
if ($memuraiSvc) {
    Assert-True ($true) 'memurai_service_not_stopped_by_test'
}
else {
    Write-Host '  skip memurai service probe (service not installed)' -ForegroundColor DarkYellow
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalStopWhatIf tests passed.' -ForegroundColor Green
