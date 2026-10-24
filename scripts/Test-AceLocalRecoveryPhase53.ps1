#Requires -Version 5.1
<#
  Phase 5.3 — restore completeness verification (temp only; not wired to Start-AceLocal).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptDir

Write-Host 'Test-AceLocalRecoveryPhase53 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
if ($startText -match 'restore_completeness') { throw 'PHASE53_WIRED_TO_START' }

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_phase53_completeness.py -q
    if ($LASTEXITCODE -ne 0) { throw 'PYTEST_PHASE53_FAILED' }
}
finally { Pop-Location }

Write-Host 'Phase53 PASSED' -ForegroundColor Green
