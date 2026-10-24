#Requires -Version 5.1
<#
  Phase 5.14 — legacy bootstrap → periodic save transition (temp only).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryLegacyBootstrapLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')

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
        git config user.email 'recovery-514@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery 514' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-LegacyLayout {
    param([string]$Launcher = '20261024-514000')
    $root = Join-Path $env:TEMP ("ace-recovery-514-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $state = Join-Path $ace '_backups\local-dev'
    $sessions = Join-Path $ace '_backups\sessions'
    $registryDir = Join-Path $ace '_registry'
    $cpRoot = Join-Path $state 'recovery-checkpoints'
    New-Item -ItemType Directory -Path $backend, $frontend, $state, $sessions, $registryDir, $cpRoot -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    $managed = @{
        schemaVersion   = 2
        launcherSession = $Launcher
        components      = $null
        updatedAt       = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Depth 5
    Set-Content -LiteralPath (Join-Path $state 'managed-processes.json') -Value $managed -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $sessions 'active-dev-cycle.json') -Value (@{ cycleId = 'open-514'; phase = 'OPEN' } | ConvertTo-Json) -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $registryDir 'production-registry.json') -Value (@{ activePublishCandidateId = 'a1' } | ConvertTo-Json) -Encoding UTF8
    return @{
        TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend
        CpRoot = $cpRoot; State = $state; Sessions = $sessions; Launcher = $Launcher
    }
}

Write-Host 'Test-AceLocalRecoveryPhase514 (isolated)...' -ForegroundColor Cyan
$env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'ok'

$layout = New-LegacyLayout
$extraTemps = @()
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = $layout.Launcher

    'dirty' | Set-Content -LiteralPath (Join-Path $be 'work.txt') -Encoding UTF8
    'fe' | Set-Content -LiteralPath (Join-Path $fe 'fe.txt') -Encoding UTF8

    $openHash = (Get-FileHash -LiteralPath (Join-Path $layout.Sessions 'active-dev-cycle.json') -Algorithm SHA256).Hash
    $regHash = (Get-FileHash -LiteralPath (Join-Path $ace '_registry\production-registry.json') -Algorithm SHA256).Hash

    $pipe = Invoke-AceLocalRecoveryLegacyBootstrapPipeline -AceRoot $ace -BackendRoot $be -FrontendRoot $fe `
        -CheckpointsRootOverride $cpRoot
    Assert-True $pipe.Ok 'bootstrap_succeeded'

    $ready = Test-AceLocalRecoveryLegacyBootstrapPeriodicSaveTransitionReady -AceRoot $ace -BackendRoot $be -FrontendRoot $fe
    Assert-True $ready.Ready ('transition_ready:' + ($ready.Reasons -join ';'))

    $ps1 = Invoke-AceLocalRecoveryPeriodicSaveAfterLegacyBootstrap -AceRoot $ace -BackendRoot $be -FrontendRoot $fe
    Assert-True $ps1.Started 'periodic_save_started_once'

    $ps2 = Invoke-AceLocalRecoveryPeriodicSaveAfterLegacyBootstrap -AceRoot $ace -BackendRoot $be -FrontendRoot $fe
    Assert-True (-not $ps2.Started) 'no_duplicate_periodic_spawn'
    Assert-True ([string]$ps2.Reason -match 'already_running|transition_not_ready') 'duplicate_blocked'

    Assert-True ($openHash -eq (Get-FileHash -LiteralPath (Join-Path $layout.Sessions 'active-dev-cycle.json') -Algorithm SHA256).Hash) 'open_unchanged'
    Assert-True ($regHash -eq (Get-FileHash -LiteralPath (Join-Path $ace '_registry\production-registry.json') -Algorithm SHA256).Hash) 'registry_unchanged'

    $layoutFail = New-LegacyLayout -Launcher '20261024-514fail'
    $extraTemps += $layoutFail.TempRoot
    'x' | Set-Content -LiteralPath (Join-Path $layoutFail.Backend 'x.txt') -Encoding UTF8
    Initialize-AceLocalRecoveryLegacyBootstrapLedger -AceRoot $layoutFail.AceRoot -ManagedLauncherSession $layoutFail.Launcher | Out-Null
    Set-AceLocalRecoveryLegacyBootstrapLedgerStatus -AceRoot $layoutFail.AceRoot -Status 'FAILED' -LastError 'sim' | Out-Null
    Write-AceLocalRecoveryLegacyBootstrapState -AceRoot $layoutFail.AceRoot -Object ([ordered]@{
        status = 'FAILED'; updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    })
    $readyFail = Test-AceLocalRecoveryLegacyBootstrapPeriodicSaveTransitionReady -AceRoot $layoutFail.AceRoot `
        -BackendRoot $layoutFail.Backend -FrontendRoot $layoutFail.Frontend
    Assert-True (-not $readyFail.Ready) 'failed_bootstrap_blocks_transition'

    $layoutQuit = New-LegacyLayout -Launcher '20261024-514quit'
    $extraTemps += $layoutQuit.TempRoot
    Request-AceLocalRecoveryLegacyBootstrapStop -AceRoot $layoutQuit.AceRoot
    $psQuit = Invoke-AceLocalRecoveryPeriodicSaveAfterLegacyBootstrap -AceRoot $layoutQuit.AceRoot `
        -BackendRoot $layoutQuit.Backend -FrontendRoot $layoutQuit.Frontend
    Assert-True (-not $psQuit.Started -and $psQuit.Reason -eq 'local_quit_requested') 'quit_blocks_periodic_spawn'

    $layoutProbe = New-LegacyLayout -Launcher '20261024-514probe'
    $extraTemps += $layoutProbe.TempRoot
    'p' | Set-Content -LiteralPath (Join-Path $layoutProbe.Backend 'p.txt') -Encoding UTF8
    'f' | Set-Content -LiteralPath (Join-Path $layoutProbe.Frontend 'f.txt') -Encoding UTF8
    $pipeP = Invoke-AceLocalRecoveryLegacyBootstrapPipeline -AceRoot $layoutProbe.AceRoot -BackendRoot $layoutProbe.Backend `
        -FrontendRoot $layoutProbe.Frontend -CheckpointsRootOverride $layoutProbe.CpRoot
    Assert-True $pipeP.Ok 'bootstrap_for_probe_test'
    $prevFixture = $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE
    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = 'unreachable'
    $readyProbe = Test-AceLocalRecoveryLegacyBootstrapPeriodicSaveTransitionReady -AceRoot $layoutProbe.AceRoot `
        -BackendRoot $layoutProbe.Backend -FrontendRoot $layoutProbe.Frontend
    Assert-True (-not $readyProbe.Ready) 'backend_down_blocks_transition'
    $env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE = $prevFixture

    Enter-AceLocalRecoveryLegacyBootstrapLock -AceRoot $layoutQuit.AceRoot
    try {
        $stopWait = Stop-AceLocalRecoveryLegacyBootstrapForLocalQuit -AceRoot $layoutQuit.AceRoot -WaitSeconds 1 -CheckpointWaitSeconds 1
        Assert-True (-not $stopWait.Stopped) 'stop_waits_on_live_lock'
    }
    finally {
        Exit-AceLocalRecoveryLegacyBootstrapLock -AceRoot $layoutQuit.AceRoot
    }
}
finally {
    Remove-Item Env:ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    foreach ($tr in $extraTemps) {
        if ($tr) { Remove-Item -LiteralPath $tr -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

Push-Location (Split-Path -Parent $ScriptDir)
try {
    python -m pytest tests/test_ace_local_dev_recovery_startup_coordinator.py -q --tb=no -k 'not lock' 2>$null
    if ($LASTEXITCODE -ne 0) {
        python -m pytest tests/test_ace_local_dev_recovery_startup_coordinator.py -q --tb=line
        if ($LASTEXITCODE -ne 0) { $failures += 'pytest_coordinator_regression' }
    }
    else { Write-Host '  ok pytest_coordinator_regression' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if (@($failures).Count -gt 0) {
    Write-Host ("`nFailures: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalRecovery phase 5.14 tests passed.' -ForegroundColor Green
exit 0
