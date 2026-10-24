#Requires -Version 5.1
<#
  Phase 5.13 — legacy recovery bootstrap (temp only).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRootRepo = Split-Path -Parent $ScriptDir
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
        git config user.email 'recovery-513@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery 513' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-LegacyLayout {
    param([string]$Launcher = '20261024-040249')
    $root = Join-Path $env:TEMP ("ace-recovery-513-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $state = Join-Path $ace '_backups\local-dev'
    $sessions = Join-Path $ace '_backups\sessions'
    $registry = Join-Path $ace '_registry'
    $cpRoot = Join-Path $state 'recovery-checkpoints'
    New-Item -ItemType Directory -Path $backend, $frontend, $state, $sessions, $registry, $cpRoot -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    $managed = @{
        schemaVersion   = 2
        launcherSession = $Launcher
        components      = $null
        updatedAt       = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Depth 5
    Set-Content -LiteralPath (Join-Path $state 'managed-processes.json') -Value $managed -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $sessions 'active-dev-cycle.json') -Value (
        (@{ cycleId = 'ori-open-test'; phase = 'OPEN' } | ConvertTo-Json)
    ) -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $registry 'production-registry.json') -Value (
        (@{ activePublishCandidateId = 'active-1' } | ConvertTo-Json)
    ) -Encoding UTF8
    return @{
        TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend; CpRoot = $cpRoot
        Sessions = $sessions; State = $state
    }
}

Write-Host 'Test-AceLocalRecoveryPhase513 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -notmatch 'run_conditional_auto_recovery') 'no_auto_restore_during_bootstrap'

Push-Location -LiteralPath $BackendRootRepo
try {
    python -m pytest tests/test_ace_local_dev_recovery_legacy_bootstrap.py -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_legacy_bootstrap' }
    else { Write-Host '  ok pytest_legacy_bootstrap' -ForegroundColor DarkGray }
}
finally { Pop-Location }

$layout = New-LegacyLayout
$extraTemps = @()
$bgLayout = $null
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $openHash = (Get-FileHash -LiteralPath (Join-Path $layout.Sessions 'active-dev-cycle.json') -Algorithm SHA256).Hash
    $regHash = (Get-FileHash -LiteralPath (Join-Path $ace '_registry\production-registry.json') -Algorithm SHA256).Hash

    'dirty' | Set-Content -LiteralPath (Join-Path $be 'legacy-dirty.txt') -Encoding UTF8
    'fe-dirty' | Set-Content -LiteralPath (Join-Path $fe 'legacy-fe.txt') -Encoding UTF8
    '{"k":1}' | Set-Content -LiteralPath (Join-Path $be '.env.local') -Encoding UTF8

    $assess = Invoke-AceLocalRecoveryLegacyBootstrapAssessment -AceRoot $ace
    Assert-True ($assess.bootstrapAllowed -and $assess.classification -eq 'LEGACY_INSTALL_RECOVERY_BOOTSTRAP') 'legacy_assessment_allowed'

    $pipe = Invoke-AceLocalRecoveryLegacyBootstrapPipeline -AceRoot $ace -BackendRoot $be -FrontendRoot $fe `
        -CheckpointsRootOverride $cpRoot
    if (-not $pipe.Ok) {
        $script:failures += ('bootstrap_pipeline_completed:' + (($pipe.Errors | ForEach-Object { $_ }) -join ';'))
        Write-Host (" FAIL bootstrap_pipeline_completed ($($pipe.Phase))") -ForegroundColor Red
    }
    else { Write-Host '  ok bootstrap_pipeline_completed' -ForegroundColor DarkGray }
    if (-not $pipe.Ok) { throw 'STOP_AFTER_PIPELINE_FAIL' }
    Assert-True (Test-Path -LiteralPath (Join-Path $layout.State 'recovery-session-ledger.json')) 'ledger_created'
    $ledger = Read-AceLocalRecoveryLedger -AceRoot $ace -AllowCreateStateDir
    $bootSt = $null
    if ($ledger.PSObject.Properties.Match('legacyRecoveryBootstrap').Count -gt 0) {
        $bootSt = [string]$ledger.legacyRecoveryBootstrap.status
    }
    Assert-True ($bootSt -eq 'COMPLETED') 'ledger_bootstrap_completed'
    Assert-True ($ledger.lastVerifiedCheckpoint) 'verified_checkpoint_recorded'

    Assert-True ($openHash -eq (Get-FileHash -LiteralPath (Join-Path $layout.Sessions 'active-dev-cycle.json') -Algorithm SHA256).Hash) 'open_unchanged'
    Assert-True ($regHash -eq (Get-FileHash -LiteralPath (Join-Path $ace '_registry\production-registry.json') -Algorithm SHA256).Hash) 'registry_unchanged'

    $cpDir = Join-Path $cpRoot $pipe.CheckpointId
    $guidePath = Join-Path $cpDir 'RECOVERY-GUIDE.txt'
    if (Test-Path -LiteralPath $guidePath) {
        $guideText = Get-Content -LiteralPath $guidePath -Raw
        Assert-True ($guideText -notmatch 'ACE_TEST=secret') 'no_plaintext_env_in_guide'
    }

    $assess2 = Invoke-AceLocalRecoveryLegacyBootstrapAssessment -AceRoot $ace
    Assert-True (-not $assess2.bootstrapAllowed) 'second_bootstrap_blocked'

    $layout2 = New-LegacyLayout -Launcher '20261024-999999'
    'x' | Set-Content -LiteralPath (Join-Path $layout2.Backend 'x.txt') -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $layout2.State 'recovery-auto-restore-journal.json') -Value '{"status":"IN_PROGRESS"}' -Encoding UTF8
    $aJournal = Invoke-AceLocalRecoveryLegacyBootstrapAssessment -AceRoot $layout2.AceRoot
    Assert-True (-not $aJournal.bootstrapAllowed) 'restore_journal_blocks'

    $layout3 = New-LegacyLayout
    Initialize-AceLocalRecoveryLedger -AceRoot $layout3.AceRoot -LauncherSession '20261024-111111' | Out-Null
    $aLedger = Invoke-AceLocalRecoveryLegacyBootstrapAssessment -AceRoot $layout3.AceRoot
    Assert-True (-not $aLedger.bootstrapAllowed) 'ledger_without_bootstrap_blocks'

    $layout4 = New-LegacyLayout
    New-Item -ItemType Directory -Path (Join-Path $layout4.CpRoot 'cp-leftover') -Force | Out-Null
    $aCp = Invoke-AceLocalRecoveryLegacyBootstrapAssessment -AceRoot $layout4.AceRoot
    Assert-True (-not $aCp.bootstrapAllowed) 'checkpoint_dir_without_ledger_blocks'

    $layout5 = New-LegacyLayout
    'y' | Set-Content -LiteralPath (Join-Path $layout5.Backend 'fail-me.txt') -Encoding UTF8
    Enter-AceLocalRecoveryLegacyBootstrapLock -AceRoot $layout5.AceRoot
    try {
        $dup = Invoke-AceLocalRecoveryLegacyBootstrapPipeline -AceRoot $layout5.AceRoot -BackendRoot $layout5.Backend `
            -FrontendRoot $layout5.Frontend -CheckpointsRootOverride $layout5.CpRoot
        Assert-True (-not $dup.Ok) 'duplicate_bootstrap_blocked_by_lock'
    }
    finally {
        Exit-AceLocalRecoveryLegacyBootstrapLock -AceRoot $layout5.AceRoot
    }

    $layout6 = New-LegacyLayout
    'z' | Set-Content -LiteralPath (Join-Path $layout6.Backend 'z.txt') -Encoding UTF8
    Initialize-AceLocalRecoveryLegacyBootstrapLedger -AceRoot $layout6.AceRoot -ManagedLauncherSession '20261024-040249' | Out-Null
    $null = New-AceLocalRecoveryCheckpoint -BackendRoot $layout6.Backend -FrontendRoot $layout6.Frontend `
        -AceRoot $layout6.AceRoot -LauncherSession '20261024-040249' -CheckpointType 'full' `
        -CheckpointsRootOverride $layout6.CpRoot -SimulateFailBeforeVerify
    $retry = Invoke-AceLocalRecoveryLegacyBootstrapPipeline -AceRoot $layout6.AceRoot -BackendRoot $layout6.Backend `
        -FrontendRoot $layout6.Frontend -CheckpointsRootOverride $layout6.CpRoot -SkipAssessment
    Assert-True $retry.Ok 'bootstrap_retry_after_failed_checkpoint'

    $extraTemps += @($layout2.TempRoot, $layout3.TempRoot, $layout4.TempRoot, $layout5.TempRoot, $layout6.TempRoot)

    $bgLayout = New-LegacyLayout -Launcher '20261024-bg001'
    'bg' | Set-Content -LiteralPath (Join-Path $bgLayout.Backend 'bg.txt') -Encoding UTF8
    $bg = Start-AceLocalRecoveryLegacyBootstrapBackground -AceRoot $bgLayout.AceRoot -BackendRoot $bgLayout.Backend `
        -FrontendRoot $bgLayout.Frontend -CheckpointsRootOverride $bgLayout.CpRoot
    Assert-True $bg.Started 'background_worker_started'
    $deadline = (Get-Date).AddSeconds(120)
    $workerOk = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        $st = Read-AceLocalRecoveryLegacyBootstrapState -AceRoot $bgLayout.AceRoot
        if ($st -and [string]$st.status -eq 'COMPLETED') { $workerOk = $true; break }
        if ($st -and [string]$st.status -eq 'FAILED') { break }
    }
    Assert-True $workerOk 'background_bootstrap_completed'
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    foreach ($tr in @($extraTemps)) {
        if ($tr) { Remove-Item -LiteralPath $tr -Recurse -Force -ErrorAction SilentlyContinue }
    }
    if ($bgLayout -and $bgLayout.TempRoot) {
        Remove-Item -LiteralPath $bgLayout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (@($failures).Count -gt 0) {
    Write-Host ("`nFailures: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalRecovery phase 5.13 tests passed.' -ForegroundColor Green
exit 0
