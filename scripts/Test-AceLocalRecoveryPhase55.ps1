#Requires -Version 5.1
<#
  Phase 5.5 — recovery startup gate wired to Start-AceLocal (check only; no restore).
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

Write-Host 'Test-AceLocalRecoveryPhase55 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Invoke-AceLocalRecoveryStartupGateCheck') 'gate_wired_to_start'
Assert-True ($startText -notmatch 'run_conditional_auto_recovery') 'restore_engine_not_in_start'
Assert-True ($startText -notmatch 'Initialize-AceLocalRecoveryLedger') 'ledger_not_init_in_start'

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_startup_gate.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_gate' }
    else { Write-Host '  ok pytest_gate' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if ($failures.Count -gt 0) {
    Write-Host ("Phase55 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase55 PASSED' -ForegroundColor Green
