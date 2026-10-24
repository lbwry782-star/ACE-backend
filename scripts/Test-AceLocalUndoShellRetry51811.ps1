#Requires -Version 5.1
<#
  Isolated tests for Wait-AceFrontendViteOnPort + LOCAL shell launch guards (no GUI, no stack start).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalUndoShellRetry51811...' -ForegroundColor Cyan

$startLocalText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startLocalText -match 'Wait-AceFrontendViteOnPort') 'start_local_uses_vite_wait_on_port'
Assert-True ($startLocalText -match "Get-Process -Name 'AceUndo'") 'start_local_skips_duplicate_aceundo'

$script:mockViteResults = [System.Collections.Generic.Queue[bool]]::new()
function Test-AceFrontendViteOnPort {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$RequestTimeoutSec = 1
    )
    if ($script:mockViteResults.Count -gt 0) {
        return [bool]$script:mockViteResults.Dequeue()
    }
    return $false
}

$script:mockViteResults = [System.Collections.Generic.Queue[bool]]::new()
[void]$script:mockViteResults.Enqueue($true)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$okImmediate = Wait-AceFrontendViteOnPort -Port 3000 -TotalTimeoutSec 10 -RequestTimeoutSec 2
$sw.Stop()
Assert-True $okImmediate 'wait_immediate_success'
Assert-True ($sw.Elapsed.TotalSeconds -lt 2) 'wait_immediate_fast'

$script:mockViteResults = [System.Collections.Generic.Queue[bool]]::new()
[void]$script:mockViteResults.Enqueue($false)
[void]$script:mockViteResults.Enqueue($false)
[void]$script:mockViteResults.Enqueue($true)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$okDelayed = Wait-AceFrontendViteOnPort -Port 3000 -TotalTimeoutSec 10 -RequestTimeoutSec 2
$sw.Stop()
Assert-True $okDelayed 'wait_delayed_success'
Assert-True ($sw.Elapsed.TotalSeconds -ge 0.2) 'wait_delayed_not_instant'

$script:mockViteResults = [System.Collections.Generic.Queue[bool]]::new()
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$okFail = Wait-AceFrontendViteOnPort -Port 3000 -TotalTimeoutSec 1 -RequestTimeoutSec 1
$sw.Stop()
Assert-True (-not $okFail) 'wait_permanent_failure'
Assert-True ($sw.Elapsed.TotalSeconds -ge 0.9 -and $sw.Elapsed.TotalSeconds -le 3) 'wait_failure_bounded'

if (@(Get-Process -Name 'AceUndo' -ErrorAction SilentlyContinue).Count -gt 0) {
    Write-Host '  ok skip_live_gui_test_aceundo_already_running' -ForegroundColor DarkYellow
}
else {
    Write-Host '  ok no_aceundo_for_duplicate_guard_test' -ForegroundColor DarkGray
}

if ($failures.Count -gt 0) {
    Write-Host ("FAILED: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalUndoShellRetry51811 tests passed.' -ForegroundColor Green
