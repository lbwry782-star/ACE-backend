#Requires -Version 5.1
<#
  Tests for LOCAL shell auto-launch helpers (no full stack start).
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
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

Write-Host 'Test-Start-AceLocalUndoLaunch...' -ForegroundColor Cyan

$startLocalPath = Join-Path $ScriptDir 'Start-AceLocal.ps1'
$startLocalText = Get-Content -LiteralPath $startLocalPath -Raw
Assert-True ($startLocalText -match 'SkipLocalUndoShell') 'start_ace_local_has_skip_switch'
Assert-True ($startLocalText -match 'Start-AceLocalUndoShellAsync') 'start_ace_local_has_async_launch'
Assert-True ($startLocalText -match 'Start-AceUndoLocal\.ps1') 'start_ace_local_invokes_undo_script'
Assert-True ($startLocalText -match '-VitePort') 'start_ace_local_passes_vite_port'

$undoScript = Join-Path $FrontendRoot 'scripts\Start-AceUndoLocal.ps1'
Assert-True (Test-Path -LiteralPath $undoScript) 'undo_launcher_exists'

Push-Location $FrontendRoot
try {
    $bad = & powershell -NoProfile -ExecutionPolicy Bypass -File $undoScript -VitePort 9999 2>&1
    Assert-True ($LASTEXITCODE -ne 0) 'undo_invalid_port_exits_nonzero'
}
finally {
    Pop-Location
}

$validateOut = & powershell -NoProfile -ExecutionPolicy Bypass -File $startLocalPath -ValidateOnly 2>&1
$validateCode = $LASTEXITCODE
Assert-True ($validateCode -eq 0 -or $validateCode -eq 1) 'validate_only_runs'
Assert-True ($validateOut -notmatch 'LOCAL shell: launch requested') 'validate_only_does_not_launch_local'

$ports = @(Find-AceFrontendVitePorts -FromPort 3000 -ToPort 3010)
if ($ports.Count -ge 1) {
    $p = $ports[0]
    Push-Location $FrontendRoot
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $undoScript -VitePort $p | Out-Null
        Assert-True ($LASTEXITCODE -eq 0) 'undo_with_live_vite_port_succeeds'
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host '  skip undo_with_live_vite_port (no running Vite)' -ForegroundColor DarkYellow
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All Start-AceLocal LOCAL launch tests passed.' -ForegroundColor Green
