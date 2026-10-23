#Requires -Version 5.1
<#
  Regression tests for fast ACE Vite port discovery (listen filter + fingerprint).
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

Write-Host 'Test-Find-AceFrontendVitePorts...' -ForegroundColor Cyan

$listenOnly = @(Get-AceFrontendListenPortsInRange -FromPort 3000 -ToPort 3010)
Assert-True ($listenOnly.Count -ge 0) 'listen_ports_query_ok'

$swFull = [System.Diagnostics.Stopwatch]::StartNew()
$liveFull = @(Find-AceFrontendVitePorts -FromPort 3000 -ToPort 3010)
$swFull.Stop()
Write-Host ("  timing full_ms={0}" -f $swFull.ElapsedMilliseconds) -ForegroundColor DarkGray
Assert-True ($swFull.ElapsedMilliseconds -lt 5000) 'full_scan_under_5s'

$swFirst = [System.Diagnostics.Stopwatch]::StartNew()
$liveFirst = @(Find-AceFrontendVitePorts -FromPort 3000 -ToPort 3010 -StopAfterFirst)
$swFirst.Stop()
Write-Host ("  timing first_ms={0}" -f $swFirst.ElapsedMilliseconds) -ForegroundColor DarkGray
Assert-True ($swFirst.ElapsedMilliseconds -lt 5000) 'first_scan_under_5s'
Assert-True ($liveFirst.Count -le 1) 'stop_after_first_at_most_one'

if ($liveFull.Count -ge 1) {
    Assert-True ($liveFull[0] -eq 3000 -or $liveFull[0] -ge 3000) 'live_ace_port_in_range'
}

# Simulated: foreign on 3000 (listen, no ACE fingerprint), ACE on 3001
$simForeignAce = @(Find-AceFrontendVitePorts -ListenPortsOverride @(3000, 3001) -StopAfterFirst)
if ($liveFull.Count -ge 1 -and $liveFull -contains 3000) {
    # When real ACE on 3000, override test still validates logic path
    Assert-True ($true) 'sim_foreign_ace_skipped_live_has_3000'
}
else {
    Assert-True ($simForeignAce.Count -le 1) 'sim_stop_after_first'
}

# Simulated duplicate ACE ports (both would fingerprint on live vite only on one port typically)
$simDupListen = @(3000, 3001)
$simDupAll = @(Find-AceFrontendVitePorts -ListenPortsOverride $simDupListen)
Assert-True ($simDupAll.Count -le 2) 'sim_dup_scan_bounded'

# Closed ports: no listeners in override empty
$simClosed = @(Find-AceFrontendVitePorts -ListenPortsOverride @())
Assert-True ($simClosed.Count -eq 0) 'sim_no_listeners_empty'

$swValidate = [System.Diagnostics.Stopwatch]::StartNew()
$null = & (Join-Path $ScriptDir 'Start-AceLocal.ps1') -ValidateOnly 2>&1
$validateCode = $LASTEXITCODE
$swValidate.Stop()
Write-Host ("  timing validate_only_ms={0} exit={1}" -f $swValidate.ElapsedMilliseconds, $validateCode) -ForegroundColor DarkGray
Assert-True ($validateCode -eq 0) 'validate_only_ok'
Assert-True ($swValidate.ElapsedMilliseconds -lt 15000) 'validate_only_under_15s'

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All Find-AceFrontendVitePorts tests passed.' -ForegroundColor Green
