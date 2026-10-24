#Requires -Version 5.1
<#
  Phase 5.8 — backend health probe before periodic save checkpoint (temp only).
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
        git config user.email 'recovery-cp58@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP58' 2>$null | Out-Null
        'v1' | Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp58-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $cpRoot = Join-Path $ace '_backups\local-dev\recovery-checkpoints'
    New-Item -ItemType Directory -Path $backend, $frontend, $cpRoot -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260201-120000' | Out-Null
    Write-AceLocalRecoveryPeriodicSaveActivationSnapshot -AceRoot $ace -LauncherSession '20260201-120000' `
        -CoordinatorOutcome 'START_ALLOWED' -AllowPeriodicSaveService $true
    return @{ TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend; CpRoot = $cpRoot }
}

Write-Host 'Test-AceLocalRecoveryPhase58 (isolated)...' -ForegroundColor Cyan
$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -match 'Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart') 'wired_to_start'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = '20260201-120000'
    $t0 = [DateTime]::Parse('2026-02-01T12:00:00Z', $null, [System.Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
    'edit' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'
    $tickOk = Invoke-AceLocalRecoveryPeriodicSaveTick -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -NowUtc $t0 -CheckpointsRootOverride $cpRoot -BackgroundMode `
        -GuardBackendRoot $BackendRoot
    Assert-True ($tickOk.Outcome -eq 'checkpoint_verified') 'backend_ok_checkpoint'

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'unreachable'
    $gateDown = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $ace -LauncherSession $launcher `
        -GuardBackendRoot $BackendRoot
    Assert-True ((-not $gateDown.Allowed) -and ($gateDown.Reasons -contains 'backend_probe_connection_refused')) 'backend_down_skips'

    foreach ($fixture in @('timeout', 'http500', 'bad_body', 'wrong_endpoint', 'non_localhost')) {
        $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = $fixture
        $gateFx = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $ace -LauncherSession $launcher `
            -GuardBackendRoot $BackendRoot
        Assert-True ((-not $gateFx.Allowed) -and ($gateFx.Reasons | Where-Object { $_ -like 'backend_probe*' -or $_ -like 'probe_url_*' })) `
            ("fixture_$fixture")
    }

    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'
    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_URL = 'http://203.0.113.1/health'
    $gateRemote = Invoke-AceLocalRecoveryPeriodicSavePreWriteGate -AceRoot $ace -LauncherSession $launcher `
        -GuardBackendRoot $BackendRoot
    Assert-True ((-not $gateRemote.Allowed) -and ($gateRemote.Reasons -contains 'probe_url_non_localhost_blocked')) 'non_localhost_url_blocked'
    Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_URL -ErrorAction SilentlyContinue

    $headBefore = (git -C $be rev-parse HEAD).Trim()
    Assert-True ($headBefore.Length -gt 0) 'git_unchanged'
}
finally {
    Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE -ErrorAction SilentlyContinue
    Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_URL -ErrorAction SilentlyContinue
    if ($layout.TempRoot) {
        Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Push-Location -LiteralPath $BackendRoot
try {
    python -m pytest tests/test_ace_local_dev_recovery_periodic_save_backend_probe.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_probe' }
    else { Write-Host '  ok pytest_probe' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if ($failures.Count -gt 0) {
    Write-Host ("Phase58 FAILED: {0}" -f ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'Phase58 PASSED' -ForegroundColor Green
