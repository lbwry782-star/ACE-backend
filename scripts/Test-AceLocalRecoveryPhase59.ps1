#Requires -Version 5.1
<#
  Phase 5.9 — periodic save wired to Start/Stop (temp/mocks only).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function Initialize-TempGitRepo {
    param([Parameter(Mandatory = $true)][string]$Root)
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Push-Location -LiteralPath $Root
    try {
        git init -q 2>$null | Out-Null
        git config user.email 'recovery-cp59@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP59' 2>$null | Out-Null
        'v1' | Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp59-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    New-Item -ItemType Directory -Path $backend, $frontend -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260201-120000' | Out-Null
    return @{ TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend }
}

function New-MockGate {
    param([string]$Outcome = 'START_ALLOWED', [bool]$AllowPeriodic = $true)
    return [PSCustomObject]@{
        coordinatorOutcome = $Outcome
        report             = [PSCustomObject]@{
            coordinatorDecision = [PSCustomObject]@{
                coordinatorOutcome       = $Outcome
                allowPeriodicSaveService = $AllowPeriodic
                isolatedOperations       = @()
            }
        }
    }
}

Write-Host 'Test-AceLocalRecoveryPhase59 (isolated)...' -ForegroundColor Cyan
$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
$stopText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Stop-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart') 'start_wired'
Assert-True ($stopText -match 'Stop-AceLocalRecoveryPeriodicSaveForLocalQuit') 'stop_wired'
Assert-True ($startText -notmatch 'Complete-AceLocalProgressState[\s\S]{0,400}Recovery periodic') 'progress_unchanged'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $registry = New-AceLocalManagedRegistryShell -BackendRoot $be -FrontendRoot $fe -LauncherSessionId '20260201-120000'
    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'

    $gate = New-MockGate
    $startOk = Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart -BackendRoot $be -FrontendRoot $fe `
        -AceRoot $ace -BackendPort 5000 -GateResult $gate -ManagedRegistry $registry -EngineBackendRoot $BackendRoot
    Assert-True $startOk.Started 'start_ok'
    Start-Sleep -Seconds 1
    $second = Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart -BackendRoot $be -FrontendRoot $fe `
        -AceRoot $ace -BackendPort 5000 -GateResult $gate -ManagedRegistry $registry -EngineBackendRoot $BackendRoot
    Assert-True (-not $second.Started -and $second.Reason -eq 'already_running') 'double_start_blocked'

    Write-AceLocalRecoveryPeriodicSaveRuntimeConfig -AceRoot $ace -BackendPort 5101 -LauncherSession '20260201-120000'
    $runtimePath = Get-AceLocalRecoveryPeriodicSaveRuntimePath -AceRoot $ace
    $runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
    Assert-True ([int]$runtime.backendPort -eq 5101) 'custom_port_runtime'

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'unreachable'
    $gateBlocked = New-MockGate -Outcome 'RECOVERY_UNSAFE' -AllowPeriodic $false
    $deny = Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart -BackendRoot $be -FrontendRoot $fe `
        -AceRoot $ace -BackendPort 5000 -GateResult $gateBlocked -ManagedRegistry $registry -EngineBackendRoot $BackendRoot
    Assert-True (-not $deny.Started) 'recovery_unsafe_denied'

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'
    $stop = Stop-AceLocalRecoveryPeriodicSaveForLocalQuit -AceRoot $ace -WaitSeconds 60
    Assert-True $stop.Stopped 'stop_idle'

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'
    Remove-Item -LiteralPath (Get-AceLocalRecoveryPeriodicSaveSpawnPath -AceRoot $ace) -Force -ErrorAction SilentlyContinue
    Write-AceLocalRecoveryPeriodicSaveServiceState -AceRoot $ace -State @{ status = 'running'; pid = 999999; launcherSession = '20260201-120000' }
    Set-Content -LiteralPath (Join-Path (Get-AceLocalRecoveryDevStateDir -AceRoot $ace) 'recovery-periodic-save.pid') -Value '999999' -Encoding ASCII
    $crash = Get-AceLocalRecoveryPeriodicSaveBackgroundStatus -AceRoot $ace
    Assert-True $crash.Crashed 'crash_detected'
    [void](Stop-AceLocalRecoveryPeriodicSaveForLocalQuit -AceRoot $ace -WaitSeconds 5)

    $bgText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalRecoveryPeriodicSaveBackgroundLib.ps1') -Raw
    Assert-True ($bgText -match 'WindowStyle Hidden') 'no_visible_window'
}
finally {
    Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE -ErrorAction SilentlyContinue
    if ($layout.TempRoot) {
        Request-AceLocalRecoveryPeriodicSaveStop -AceRoot $layout.AceRoot -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_periodic_save_service.py tests/test_ace_local_dev_recovery_periodic_save_backend_probe.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest' }
    else { Write-Host '  ok pytest' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if ($failures.Count -gt 0) {
    Write-Host ("Phase59 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase59 PASSED' -ForegroundColor Green
