#Requires -Version 5.1
<#
  Phase 5.4 — ENV DPAPI restore (temp only; not wired to Start-AceLocal).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptDir

Write-Host 'Test-AceLocalRecoveryPhase54 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
if ($startText -match 'AceLocalRecoveryEnvRestore') { throw 'PHASE54_WIRED_TO_START' }

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_phase54_env.py tests/test_ace_local_dev_recovery_phase53_completeness.py -q
    if ($LASTEXITCODE -ne 0) { throw 'PYTEST_PHASE54_FAILED' }
}
finally { Pop-Location }

Write-Host 'Phase54 PASSED' -ForegroundColor Green
