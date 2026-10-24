#Requires -Version 5.1
<#
  Phase 5.2 — safe auto recovery engine (isolated temp; not wired to Start-AceLocal).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptDir

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalRecoveryPhase52 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -notmatch 'auto_restore_engine') 'engine_not_wired_to_start'
Assert-True ($startText -notmatch 'run_conditional_auto_recovery') 'runner_not_wired_to_start'

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest `
        tests/test_ace_local_dev_recovery_auto_restore_engine.py `
        tests/test_ace_local_dev_recovery_auto_restore_plan.py `
        tests/test_ace_local_dev_recovery_startup_coordinator.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_phase52' }
    else { Write-Host '  ok pytest_phase52' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if ($failures.Count -gt 0) {
    Write-Host ("Phase52 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase52 PASSED' -ForegroundColor Green
