#Requires -Version 5.1
<#
.SYNOPSIS
  TEMP / simulated tests for Start-AceBackendLocal port and process logic (no starts, no kills).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "AceLocalEnvCommon.ps1")

$BackendRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$failures = 0

function Assert-Test([string]$Name, [bool]$Condition) {
    if ($Condition) {
        Write-Host "  ok $Name" -ForegroundColor Green
    }
    else {
        Write-Host "  FAIL $Name" -ForegroundColor Red
        $script:failures++
    }
}

Write-Host "Test-StartAceBackendLocal (simulated)" -ForegroundColor Cyan

$fullApp = Join-Path $BackendRoot "app.py"
$cmdFull = "`"$BackendRoot\.venv312\Scripts\python.exe`" $fullApp"
Assert-Test "commandline_full_path" (Test-Win32CommandLineIsAceBackendApp -CommandLine $cmdFull -BackendRoot $BackendRoot)

$cmdRel = '"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" app.py'
Assert-Test "commandline_relative_not_repo" (-not (Test-Win32CommandLineIsAceBackendApp -CommandLine $cmdRel -BackendRoot $BackendRoot))

$cmdForeign = '"C:\Python\python.exe" C:\Other\app.py'
Assert-Test "commandline_foreign" (-not (Test-Win32CommandLineIsAceBackendApp -CommandLine $cmdForeign -BackendRoot $BackendRoot))

$venvPy = Resolve-AceBackendPythonExe -BackendRoot $BackendRoot
if (Test-Path -LiteralPath (Join-Path $BackendRoot ".venv312\Scripts\python.exe")) {
    Assert-Test "python_prefers_venv312" ($venvPy -match "\\\.venv312\\Scripts\\python\.exe$")
}
else {
    Assert-Test "python_fallback_exists" (-not [string]::IsNullOrWhiteSpace($venvPy))
}

$simAlready = Get-AceBackendLocalPortState -Port 5000 -BackendRoot $BackendRoot -Simulate @{
    HealthOk              = $true
    AceBackendProcessCount = 1
    ListenerPids          = @(1234)
    AceListenerPids       = @(1234)
    ForeignListenerPids   = @()
    UncertainListenerPids = @()
    Action                = "AlreadyRunning"
    Reason                = "ace_backend_listening_health_ok"
}
Assert-Test "simulate_already_running" ($simAlready.Action -eq "AlreadyRunning")

$simForeign = Get-AceBackendLocalPortState -Port 5000 -BackendRoot $BackendRoot -Simulate @{
    HealthOk              = $true
    AceBackendProcessCount = 0
    ListenerPids          = @(9999)
    AceListenerPids       = @()
    ForeignListenerPids   = @(9999)
    UncertainListenerPids = @()
    Action                = "BlockedForeign"
    Reason                = "foreign_process_on_port"
}
Assert-Test "simulate_foreign" ($simForeign.Action -eq "BlockedForeign")

$simDup = Get-AceBackendLocalPortState -Port 5000 -BackendRoot $BackendRoot -Simulate @{
    HealthOk               = $false
    AceBackendProcessCount = 2
    AceBackendTreeCount    = 2
    ListenerPids           = @()
    AceListenerPids        = @()
    ForeignListenerPids    = @()
    UncertainListenerPids  = @()
    Action                 = "BlockedDuplicate"
    Reason                 = "multiple_independent_backend_trees"
}
Assert-Test "simulate_duplicate" ($simDup.Action -eq "BlockedDuplicate")

function New-BackendProcRow([int]$ProcId) {
    return [PSCustomObject]@{ ProcId = $ProcId; CommandLine = "app.py" }
}

$parentChildParents = @{ 27040 = 16748; 28888 = 27040 }
$parentChildProcs = @((New-BackendProcRow 27040), (New-BackendProcRow 28888))
Assert-Test "tree_parent_child_count" (
    (Get-AceBackendIndependentTreeCount -BackendProcesses $parentChildProcs -ParentProcessIdByPid $parentChildParents) -eq 1
)

$twoTreeParents = @{ 101 = 50; 102 = 60 }
$twoTreeProcs = @((New-BackendProcRow 101), (New-BackendProcRow 102))
Assert-Test "tree_two_independent" (
    (Get-AceBackendIndependentTreeCount -BackendProcesses $twoTreeProcs -ParentProcessIdByPid $twoTreeParents) -eq 2
)

$simParentChildLive = Get-AceBackendLocalPortState -Port 5000 -BackendRoot $BackendRoot -Simulate @{
    HealthOk               = $true
    AceBackendProcessCount = 2
    AceBackendTreeCount    = 0
    ListenerPids           = @(28888)
    AceListenerPids        = @(28888)
    ForeignListenerPids    = @()
    UncertainListenerPids  = @()
    Action                 = "AlreadyRunning"
    Reason                 = "ace_backend_listening_health_ok"
}
Assert-Test "simulate_parent_child_already_running" ($simParentChildLive.Action -eq "AlreadyRunning")

$liveProcs = @(Find-LocalAceBackendProcesses -BackendRoot $BackendRoot -ListenerPort 5000)
if ($liveProcs.Count -ge 2) {
    $liveTrees = Get-AceBackendIndependentTreeCount -BackendProcesses $liveProcs
    $liveState = Get-AceBackendLocalPortState -Port 5000 -BackendRoot $BackendRoot
    Assert-Test "live_parent_child_not_blocked_as_duplicate" (
        ($liveTrees -eq 1) -and ($liveState.Action -eq "AlreadyRunning")
    )
}
elseif ($liveProcs.Count -eq 0) {
    Write-Host "  skip live_parent_child (no backend running)" -ForegroundColor DarkGray
}

$simStart = Get-AceBackendLocalPortState -Port 5000 -BackendRoot $BackendRoot -Simulate @{
    HealthOk              = $false
    AceBackendProcessCount = 0
    ListenerPids          = @()
    AceListenerPids       = @()
    ForeignListenerPids   = @()
    UncertainListenerPids = @()
    Action                = "Start"
    Reason                = "port_available"
}
Assert-Test "simulate_start" ($simStart.Action -eq "Start")

Write-Host ""
Write-Host "InspectOnly (read-only live state, no start):" -ForegroundColor Cyan
$inspectJson = & (Join-Path $ScriptDir "Start-AceBackendLocal.ps1") -InspectOnly 2>&1 | Select-Object -Last 1
try {
    $live = $inspectJson | ConvertFrom-Json
    Assert-Test "inspect_json" ($null -ne $live.Action)
    Write-Host ("  live Action={0} Reason={1} aceProcs={2}" -f $live.Action, $live.Reason, $live.AceBackendProcessCount) -ForegroundColor DarkGray
}
catch {
    Assert-Test "inspect_json" $false
}

Write-Host ""
if ($failures -eq 0) {
    Write-Host "Done. Failure count: 0" -ForegroundColor Green
    exit 0
}
Write-Host "Done. Failure count: $failures" -ForegroundColor Red
exit 1
