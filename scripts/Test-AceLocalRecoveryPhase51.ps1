#Requires -Version 5.1
<#
  Phase 5.1 — automatic recovery coordinator (isolated; not wired to Start-AceLocal).
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

Write-Host 'Test-AceLocalRecoveryPhase51 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -match 'AceLocalRecoveryStartupGate') 'gate_wired_to_start'
Assert-True ($startText -notmatch 'run_conditional_auto_recovery') 'no_restore_engine_in_start'

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_startup_coordinator.py tests/test_ace_local_dev_recovery_auto_restore_plan.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_phase51' }
    else { Write-Host '  ok pytest_phase51' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if ($failures.Count -gt 0) {
    Write-Host ("Phase51 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase51 PASSED' -ForegroundColor Green
