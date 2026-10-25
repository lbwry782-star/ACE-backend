#Requires -Version 5.1
<#
  Phase 5.12.2 — full checkpoint coverage vs MANIFEST, drift, FIRST_START / LIVE-like bootstrap (temp only).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRootRepo = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir 'AceLocalRecoveryCheckpointLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryStartupGateLib.ps1')

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
        git config user.email 'recovery-5122@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery 5122' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-5122-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $cpRoot = Join-Path $ace '_backups\local-dev\recovery-checkpoints'
    $sessions = Join-Path $ace '_backups\sessions'
    $registryDir = Join-Path $ace '_registry'
    New-Item -ItemType Directory -Path $backend, $frontend, $cpRoot, $sessions, $registryDir -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    return @{
        TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend
        CpRoot = $cpRoot; Sessions = $sessions; RegistryDir = $registryDir
    }
}

function Get-ManifestKey {
    param([string]$Area, [string]$RelPath)
    $norm = ($RelPath -replace '\\', '/').Trim().ToLowerInvariant()
    return ("{0}|{1}" -f $Area.ToUpperInvariant(), $norm)
}

function Compare-FullCheckpointManifestCoverage {
    param(
        [Parameter(Mandatory = $true)]$WorkspaceScan,
        [Parameter(Mandatory = $true)][string]$CheckpointRoot,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot
    )
    $rows = @(Import-AceLocalRecoveryCheckpointManifestRows -CheckpointRoot $CheckpointRoot)
    $manifestKeys = @{}
    foreach ($r in $rows) {
        $manifestKeys[(Get-ManifestKey -Area ([string]$r.Area) -RelPath ([string]$r.RelPath))] = $r
    }

    $missing = @()
    $unexpected = @()
    $excludedOk = @()

    foreach ($side in @(
        @{ Label = 'BACKEND'; Root = $BackendRoot; Block = $WorkspaceScan.Backend }
        @{ Label = 'FRONTEND'; Root = $FrontendRoot; Block = $WorkspaceScan.Frontend }
    )) {
        foreach ($e in @($side.Block.Entries)) {
            $rel = ([string]$e.path) -replace '/', '\'
            $ct = [string]$e.changeType
            $key = Get-ManifestKey -Area $side.Label -RelPath $rel
            if ($ct -eq 'deleted') {
                if (-not $manifestKeys.ContainsKey($key)) { $missing += "deleted:$key" }
                continue
            }
            $full = Join-Path $side.Root $rel
            if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
            $ex = Test-AceLocalShutdownPathExcluded -RelativePath $rel -SizeBytes (Get-Item -LiteralPath $full).Length
            if ($ex.Excluded) {
                if ($manifestKeys.ContainsKey($key)) { $unexpected += "denylist_in_manifest:$key" }
                $excludedOk += [PSCustomObject]@{ Path = $rel; Reason = $ex.Reason }
                continue
            }
            if (-not $manifestKeys.ContainsKey($key)) {
                $missing += "mirror:$key"
            }
        }
        foreach ($ig in @($side.Block.IgnoredExcluded)) {
            $rel = ([string]$ig.RelPath) -replace '/', '\'
            $key = Get-ManifestKey -Area $side.Label -RelPath $rel
            if ($manifestKeys.ContainsKey($key)) { $unexpected += "ignored_excluded_in_manifest:$key" }
            $excludedOk += [PSCustomObject]@{ Path = $rel; Reason = [string]$ig.Reason }
        }
    }

    return @{
        Missing    = @($missing)
        Unexpected = @($unexpected)
        Excluded   = @($excludedOk)
        RowCount   = $rows.Count
        ManifestKeys = $manifestKeys
    }
}

function Invoke-StartupCheckPython {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [switch]$SkipLiveProcessInspection
    )
    $code = @'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from engine.ace_local_dev_startup_recovery_check import evaluate_previous_local_startup
from engine.ace_local_dev_recovery_startup_coordinator import evaluate_recovery_startup_coordination
ace = Path(sys.argv[2])
sess = Path(sys.argv[3]) if sys.argv[3] else None
inspect_live = sys.argv[4] == '1'
startup = evaluate_previous_local_startup(ace, sessions_root=sess, inspect_live_processes=inspect_live)
coord = evaluate_recovery_startup_coordination(
    ace, sessions_root=sess, inspect_live_processes=inspect_live, acquire_coordinator_lock=False,
)
print(json.dumps({
    'startupOutcome': startup.outcome,
    'startupReasons': list(startup.reasons),
    'devOpsAllowed': startup.dev_operations_allowed,
    'coordinatorOutcome': coord.coordinator_outcome,
    'allowLedgerInit': coord.allow_ledger_initialization,
    'allowPeriodicSave': coord.allow_periodic_save_service,
    'continueStack': coord.allow_local_processes,
}, ensure_ascii=False))
'@
    $sessArg = Join-Path $AceRoot '_backups\sessions'
    if (-not (Test-Path -LiteralPath $sessArg)) { $sessArg = '' }
    $inspectArg = if ($SkipLiveProcessInspection) { '0' } else { '1' }
    $raw = python -c $code $BackendRootRepo $AceRoot $sessArg $inspectArg 2>&1
    if ($LASTEXITCODE -ne 0) { throw "STARTUP_CHECK_FAILED:$raw" }
    return ($raw | ConvertFrom-Json)
}

Write-Host 'Test-AceLocalRecoveryPhase5122 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -notmatch 'run_conditional_auto_recovery') 'start_does_not_auto_restore'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = '20261024-512200'

    $hebrewBase = ([char]0x05E2) + ([char]0x05D1) + ([char]0x05E8) + ([char]0x05D9) + ([char]0x05EA)
    New-Item -ItemType Directory -Path (Join-Path $be 'engine') -Force | Out-Null
    'marker' | Set-Content -LiteralPath (Join-Path $be "engine\ace_local_dev_recovery_5122_marker.py") -Encoding UTF8
    'modified' | Set-Content -LiteralPath (Join-Path $be 'tracked-edit.txt') -Encoding UTF8
    git -C $be add tracked-edit.txt 2>$null | Out-Null
    git -C $be commit -q -m 'tracked seed' 2>$null | Out-Null
    'new-untracked' | Set-Content -LiteralPath (Join-Path $be 'brand-new.txt') -Encoding UTF8
    $hebrewRel = "$hebrewBase.txt"
    'hebrew' | Set-Content -LiteralPath (Join-Path $be $hebrewRel) -Encoding UTF8
    git -C $be mv tracked-edit.txt 'renamed with spaces.txt' 2>$null | Out-Null
    Remove-Item -LiteralPath (Join-Path $be 'base.txt') -Force

    @('prefs.local.json', 'skip-me.txt', 'node_modules/') | Set-Content -LiteralPath (Join-Path $be '.gitignore') -Encoding UTF8
    '{"local":true}' | Set-Content -LiteralPath (Join-Path $be 'prefs.local.json') -Encoding UTF8
    'deny' | Set-Content -LiteralPath (Join-Path $be 'skip-me.txt') -Encoding UTF8
    New-Item -ItemType Directory -Path (Join-Path $be 'node_modules\pkg') -Force | Out-Null
    'nm' | Set-Content -LiteralPath (Join-Path $be 'node_modules\pkg\index.js') -Encoding UTF8
    'tiny' | Set-Content -LiteralPath (Join-Path $be 'clip.mp4') -Encoding UTF8
    "ACE_TEST=secret`n" | Set-Content -LiteralPath (Join-Path $be '.env.local') -Encoding UTF8
    'VITE_BACKEND_URL=http://127.0.0.1:5000' | Set-Content -LiteralPath (Join-Path $fe '.env.development.local') -Encoding UTF8
    'fe-edit' | Set-Content -LiteralPath (Join-Path $fe 'frontend-change.txt') -Encoding UTF8

    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession $launcher | Out-Null
    $scan = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    $cp = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType full -CheckpointsRootOverride $cpRoot
    if (-not ($cp.Ok -and $cp.Status -eq 'VERIFIED')) {
        $detail = @($cp.Errors) -join ';'
        if (-not $detail) { $detail = [string]$cp.Reason + ';status=' + [string]$cp.Status }
        $script:failures += ('full_checkpoint_verified:' + $detail)
        Write-Host (" FAIL full_checkpoint_verified ($detail)") -ForegroundColor Red
    }
    else { Write-Host '  ok full_checkpoint_verified' -ForegroundColor DarkGray }

    $cov = Compare-FullCheckpointManifestCoverage -WorkspaceScan $scan -CheckpointRoot $cp.CheckpointRoot `
        -BackendRoot $be -FrontendRoot $fe
    Assert-True ($cov.Missing.Count -eq 0) ('manifest_missing:' + ($cov.Missing -join ';'))
    Assert-True ($cov.Unexpected.Count -eq 0) ('manifest_unexpected:' + ($cov.Unexpected -join ';'))
    Assert-True (@($cov.Excluded | Where-Object { $_.Path -match 'skip-me' }).Count -ge 1) 'ignored_not_in_allowlist_excluded'
    Assert-True (@($cov.Excluded | Where-Object { $_.Reason -match 'denylist_segment:node_modules' }).Count -ge 1) 'node_modules_excluded'
    Assert-True (@($cov.Excluded | Where-Object { $_.Reason -match 'large_media' }).Count -ge 1) 'large_media_excluded'
    Assert-True ($cov.ManifestKeys.ContainsKey((Get-ManifestKey -Area 'ENV' -RelPath 'backend.env.local'))) 'env_backend_in_manifest'
    Assert-True ($cov.ManifestKeys.ContainsKey((Get-ManifestKey -Area 'ENV' -RelPath 'frontend.env.development.local'))) 'env_frontend_in_manifest'

    $manifest = Import-AceLocalRecoveryCheckpointManifestRows -CheckpointRoot $cp.CheckpointRoot
    Assert-True (@($manifest | Where-Object { [string]$_.RelPath -match 'ace_local_dev_recovery_5122' }).Count -ge 1) 'recovery_file_in_manifest'
    Assert-True (@($manifest | Where-Object { [string]$_.Area -eq 'ENV' }).Count -ge 2) 'env_rows_in_manifest'

    $envDpapi = Join-Path $cp.CheckpointRoot 'ENV\backend.env.local.dpapi'
    Assert-True (Test-Path -LiteralPath $envDpapi) 'env_dpapi_on_disk'
    $plain = Read-AceLocalShutdownEnvPlain -DpapiPath $envDpapi
    Assert-True ($plain -match 'ACE_TEST') 'env_dpapi_roundtrip'

    $verifiedBefore = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    $driftCp = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType full -CheckpointsRootOverride $cpRoot `
        -DuringBackendCopyHook {
            'mutated-during-copy' | Set-Content -LiteralPath (Join-Path $be 'brand-new.txt') -Encoding UTF8 -Force
        }
    Assert-True (-not $driftCp.Ok -and $driftCp.Status -eq 'INCOMPLETE') 'concurrent_drift_not_verified'
    $verifiedAfter = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    Assert-True ($verifiedAfter.Count -eq $verifiedBefore.Count) 'verified_count_unchanged_after_drift'
    if ($verifiedBefore.Count -gt 0) {
        Assert-True ([string]$verifiedAfter[0].CheckpointId -eq [string]$verifiedBefore[0].CheckpointId) 'last_verified_unchanged_after_drift'
    }

    # --- FIRST_START (no prior managed registry) ---
    $firstRoot = Join-Path $layout.TempRoot 'first-start-ace'
    $firstBe = Join-Path $firstRoot 'ACE-Backend'
    $firstFe = Join-Path $firstRoot 'ACE-Frontend'
    New-Item -ItemType Directory -Path $firstBe, $firstFe, (Join-Path $firstRoot '_backups\sessions') -Force | Out-Null
    Initialize-TempGitRepo -Root $firstBe
    Initialize-TempGitRepo -Root $firstFe
    $firstCheck = Invoke-StartupCheckPython -AceRoot $firstRoot -SkipLiveProcessInspection
    Assert-True ([string]$firstCheck.startupOutcome -eq 'FIRST_START') 'pure_first_start_outcome'
    Assert-True ([string]$firstCheck.coordinatorOutcome -eq 'START_ALLOWED') 'pure_first_start_coordinator'

    # --- LIVE-like: managed session, no ledger, OPEN/ACTIVE stubs ---
    $liveAce = Join-Path $layout.TempRoot 'live-like-ace'
    $liveBe = Join-Path $liveAce 'ACE-Backend'
    $liveFe = Join-Path $liveAce 'ACE-Frontend'
    $liveState = Join-Path $liveAce '_backups\local-dev'
    $liveSessions = Join-Path $liveAce '_backups\sessions'
    $liveRegistry = Join-Path $liveAce '_registry'
    New-Item -ItemType Directory -Path $liveBe, $liveFe, $liveState, $liveSessions, $liveRegistry -Force | Out-Null
    Initialize-TempGitRepo -Root $liveBe
    Initialize-TempGitRepo -Root $liveFe
    'dirty' | Set-Content -LiteralPath (Join-Path $liveBe 'dirty.txt') -Encoding UTF8

    $managed = @{
        schemaVersion     = 2
        launcherSession   = '20261024-040249'
        components        = $null
        updatedAt         = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Depth 6
    Set-Content -LiteralPath (Join-Path $liveState 'managed-processes.json') -Value $managed -Encoding UTF8

    $openCycle = @{
        cycleId = 'ori-20261023-094907-078ee0d3'
        phase   = 'OPEN'
    } | ConvertTo-Json
    Set-Content -LiteralPath (Join-Path $liveSessions 'active-dev-cycle.json') -Value $openCycle -Encoding UTF8

    $prodReg = @{
        activePublishCandidateId = 'test-active'
    } | ConvertTo-Json
    Set-Content -LiteralPath (Join-Path $liveRegistry 'production-registry.json') -Value $prodReg -Encoding UTF8

    $openHash = (Get-FileHash -LiteralPath (Join-Path $liveSessions 'active-dev-cycle.json') -Algorithm SHA256).Hash
    $regHash = (Get-FileHash -LiteralPath (Join-Path $liveRegistry 'production-registry.json') -Algorithm SHA256).Hash

    $liveCheck = Invoke-StartupCheckPython -AceRoot $liveAce -SkipLiveProcessInspection
    Assert-True ([string]$liveCheck.startupOutcome -ne 'FIRST_START') 'live_like_not_first_start'
    Assert-True ([string]$liveCheck.startupOutcome -eq 'LEGACY_INSTALL_RECOVERY_BOOTSTRAP') 'live_like_legacy_bootstrap'
    Assert-True ([string]$liveCheck.coordinatorOutcome -eq 'LEGACY_RECOVERY_BOOTSTRAP') 'live_like_coordinator_legacy_bootstrap'

    Initialize-AceLocalRecoveryLedger -AceRoot $liveAce -LauncherSession '20261024-512299'
    $openHashAfter = (Get-FileHash -LiteralPath (Join-Path $liveSessions 'active-dev-cycle.json') -Algorithm SHA256).Hash
    $regHashAfter = (Get-FileHash -LiteralPath (Join-Path $liveRegistry 'production-registry.json') -Algorithm SHA256).Hash
    Assert-True ($openHash -eq $openHashAfter) 'ledger_init_does_not_touch_open'
    Assert-True ($regHash -eq $regHashAfter) 'ledger_init_does_not_touch_registry'
    Assert-True (Test-Path -LiteralPath (Join-Path $liveState 'recovery-session-ledger.json')) 'ledger_created'

    # Start-AceLocal: already-running path retains a valid record instead of clearing it.
    Assert-True ($startText -match 'already running') 'start_handles_already_running'
    Assert-True ($startText -match 'Test-BackendLocalDevIdentity') 'start_verifies_backend_identity'
    Assert-True ($startText -match 'Sync-AceLocalAlreadyRunningManagedComponent') 'already_running_syncs_ownership'
    Assert-True ($startText -notmatch '\$managedRegistry\.components\.backend = \$null') 'already_running_does_not_clear_backend_slot'
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Push-Location -LiteralPath $BackendRootRepo
try {
    python -m pytest tests/test_ace_local_dev_startup_recovery_check.py::TestAceLocalStartupRecoveryCheck::test_first_start -q
    if ($LASTEXITCODE -ne 0) { $failures += 'pytest_first_start' }
    else { Write-Host '  ok pytest_first_start' -ForegroundColor DarkGray }
}
finally { Pop-Location }

if (@($failures).Count -gt 0) {
    Write-Host ("`nFailures: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalRecovery phase 5.12.2 tests passed.' -ForegroundColor Green
exit 0
