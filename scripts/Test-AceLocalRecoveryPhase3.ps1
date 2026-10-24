#Requires -Version 5.1
<#
  Phase 3 recovery checkpoint tests — temp git repos only.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalRecoveryCheckpointLib.ps1')

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
        git config user.email 'recovery-cp3@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery CP3' 2>$null | Out-Null
        Set-Content -LiteralPath (Join-Path $Root 'base.txt') -Value 'v1' -Encoding UTF8
        git add base.txt 2>$null | Out-Null
        git commit -q -m 'init' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-cp3-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $cpRoot = Join-Path $ace '_backups\local-dev\recovery-checkpoints'
    New-Item -ItemType Directory -Path $backend, $frontend, $cpRoot -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20260101-120000' | Out-Null
    return @{
        TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend; CpRoot = $cpRoot
    }
}

function Get-RepoStatusText {
    param([string]$RepoRoot)
    return (git -C $RepoRoot status -u --porcelain=v1 | Out-String).Trim()
}

Write-Host 'Test-AceLocalRecoveryPhase3 (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -notmatch 'AceLocalRecoveryCheckpoint') 'not_wired_to_start'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot
    $launcher = '20260101-120000'

    'tracked' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    git -C $be add edit-me.txt 2>$null | Out-Null
    'new' | Set-Content -LiteralPath (Join-Path $be 'new.txt') -Encoding UTF8
    Remove-Item -LiteralPath (Join-Path $be 'base.txt') -Force
    @('prefs.local.json', 'skip-me.txt') | Set-Content -LiteralPath (Join-Path $be '.gitignore') -Encoding UTF8
    '{"x":1}' | Set-Content -LiteralPath (Join-Path $be 'prefs.local.json') -Encoding UTF8
    'skip' | Set-Content -LiteralPath (Join-Path $be 'skip-me.txt') -Encoding UTF8
    "ACE_TEST=secret`n" | Set-Content -LiteralPath (Join-Path $be '.env.local') -Encoding UTF8
    'VITE_BACKEND_URL=http://127.0.0.1:1' | Set-Content -LiteralPath (Join-Path $fe '.env.development.local') -Encoding UTF8
    'fe-change' | Set-Content -LiteralPath (Join-Path $fe 'fe.txt') -Encoding UTF8

    $statusBefore = Get-RepoStatusText -RepoRoot $be
    $headBefore = (git -C $be rev-parse HEAD).Trim()

    $full1 = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType 'full' -CheckpointsRootOverride $cpRoot
    Assert-True $full1.Ok 'full_checkpoint_verified'
    Assert-True ($full1.Status -eq 'VERIFIED') 'full_status'
    $ledger1 = Read-AceLocalRecoveryLedger -AceRoot $ace -AllowCreateStateDir
    Assert-True ($ledger1.lastVerifiedCheckpoint.checkpointId -eq $full1.CheckpointId) 'ledger_updated_after_full'

    $intFull = Test-AceLocalRecoveryCheckpointIntegrity -CheckpointRoot $full1.CheckpointRoot
    Assert-True $intFull.Ok 'full_integrity'

    Assert-True ((Get-RepoStatusText -RepoRoot $be) -eq $statusBefore) 'source_repo_unchanged_after_full'
    Assert-True ((git -C $be rev-parse HEAD).Trim() -eq $headBefore) 'source_head_unchanged_after_full'

    $scanA = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    $fpA = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scanA
    'changed-again' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    $scanB = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    $statusAfterEdit = Get-RepoStatusText -RepoRoot $be
    $statusLineCountSame = (@(($statusBefore -split "`n") | Where-Object { $_.Trim() }).Count -eq @(($statusAfterEdit -split "`n") | Where-Object { $_.Trim() }).Count)
    Assert-True $statusLineCountSame 'content_change_status_line_count_unchanged'
    Assert-True (Test-AceLocalRecoveryWorkspaceContentChanged -PreviousScan $scanA -CurrentScan $scanB) 'content_change_detected'
    Assert-True ((Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scanB) -ne $fpA) 'content_fingerprint_changed'

    'round2' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    'round3' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8
    $scanC = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True (Test-AceLocalRecoveryWorkspaceContentChanged -PreviousScan $scanB -CurrentScan $scanC) 'repeated_file_edits_detected'

    $light1 = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType 'light' -BaseCheckpointId $full1.CheckpointId `
        -CheckpointsRootOverride $cpRoot
    Assert-True $light1.Ok 'light_checkpoint_verified'
    $metaLight = Read-AceLocalRecoveryCheckpointMetadata -CheckpointRoot $light1.CheckpointRoot
    Assert-True ([string]$metaLight.baseCheckpointId -eq $full1.CheckpointId) 'light_has_base'

    $chain = @(Get-AceLocalRecoveryCheckpointChain -AceRoot $ace -CheckpointId $light1.CheckpointId)
    Assert-True ($chain.Count -eq 2) 'incremental_chain_length'
    Assert-True ([string]$chain[0].CheckpointType -eq 'full') 'chain_root_full'

    $cleanBe = Join-Path $layout.TempRoot 'clean-be'
    $cleanFe = Join-Path $layout.TempRoot 'clean-fe'
    Initialize-TempGitRepo -Root $cleanBe
    Initialize-TempGitRepo -Root $cleanFe
    $skip = New-AceLocalRecoveryCheckpoint -BackendRoot $cleanBe -FrontendRoot $cleanFe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType 'full' -CheckpointsRootOverride $cpRoot
    Assert-True ($skip.Skipped -and $skip.Reason -eq 'no_working_tree_changes') 'checkpoint_skipped_no_changes'

    $failCp = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType 'full' -CheckpointsRootOverride $cpRoot `
        -SimulateFailAfterBackend
    Assert-True (-not $failCp.Ok) 'simulated_power_fail_incomplete'
    Assert-True ($failCp.Status -eq 'INCOMPLETE') 'fail_status_incomplete'
    $ledgerAfterFail = Read-AceLocalRecoveryLedger -AceRoot $ace -AllowCreateStateDir
    Assert-True ($ledgerAfterFail.lastVerifiedCheckpoint.checkpointId -eq $light1.CheckpointId) 'previous_verified_preserved_on_fail'

    $driftCp = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession $launcher -CheckpointType 'full' -CheckpointsRootOverride $cpRoot `
        -DuringBackendCopyHook { 'mutated' | Set-Content -LiteralPath (Join-Path $be 'edit-me.txt') -Encoding UTF8 -Force }
    Assert-True (-not $driftCp.Ok) 'concurrent_drift_fails'
    Assert-True ((-not $driftCp.Ok) -and ($driftCp.Status -eq 'INCOMPLETE')) 'drift_error'

    $restoreBe = Join-Path $layout.TempRoot 'restore-be'
    $restoreFe = Join-Path $layout.TempRoot 'restore-fe'
    git clone -q $be $restoreBe 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'git_clone_backend_restore_failed' }
    git clone -q $fe $restoreFe 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'git_clone_frontend_restore_failed' }
    $restore = Restore-AceLocalRecoveryCheckpointToTempRepos -AceRoot $ace -CheckpointId $light1.CheckpointId `
        -TargetBackendRoot $restoreBe -TargetFrontendRoot $restoreFe
    Assert-True ((Get-Content -LiteralPath (Join-Path $restoreBe 'edit-me.txt') -Raw).Trim() -eq 'round3') 'restore_modified'
    Assert-True (Test-Path -LiteralPath (Join-Path $restoreBe 'new.txt')) 'restore_untracked'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $restoreBe 'base.txt'))) 'restore_deleted'
    Assert-True (Test-Path -LiteralPath (Join-Path $restoreBe 'prefs.local.json')) 'restore_ignored_allowlist'
    Assert-True (Test-Path -LiteralPath (Join-Path $restoreFe 'fe.txt')) 'restore_frontend'

    $envDpapi = Join-Path $light1.CheckpointRoot 'ENV\backend.env.local.dpapi'
    Assert-True (Test-Path -LiteralPath $envDpapi) 'env_dpapi_present'
    $plain = Read-AceLocalShutdownEnvPlain -DpapiPath $envDpapi
    Assert-True ($plain -match 'ACE_TEST') 'env_dpapi_roundtrip'

    $badChain = $false
    try {
        $null = Get-AceLocalRecoveryCheckpointChain -AceRoot $ace -CheckpointId 'cp-does-not-exist'
        if (@(Get-AceLocalRecoveryCheckpointChain -AceRoot $ace -CheckpointId 'cp-does-not-exist').Count -eq 0) {
            $badChain = $true
        }
    }
    catch { $badChain = $true }
    Assert-True $badChain 'missing_checkpoint_chain_empty'

    for ($i = 0; $i -lt 6; $i++) {
        "v$i" | Set-Content -LiteralPath (Join-Path $be "extra-$i.txt") -Encoding UTF8
        $null = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
            -LauncherSession $launcher -CheckpointType 'light' -BaseCheckpointId $light1.CheckpointId `
            -CheckpointsRootOverride $cpRoot
    }
    $plan = Get-AceLocalRecoveryCheckpointRetentionPlan -AceRoot $ace
    Assert-True ($plan.VerifiedCount -ge 5) 'retention_at_least_five_verified'
    Assert-True ($plan.ProtectedIds.Count -ge 5) 'retention_protects_minimum'
    Assert-True ($plan.Note -match 'no_files_deleted') 'retention_decision_only'

    $corruptMeta = Join-Path $full1.CheckpointRoot 'MANIFEST.csv'
    $manifestBackup = Get-Content -LiteralPath $corruptMeta -Raw
    Set-Content -LiteralPath $corruptMeta -Value 'Area,RelPath,broken' -Encoding UTF8
    $badInt = Test-AceLocalRecoveryCheckpointIntegrity -CheckpointRoot $full1.CheckpointRoot -SimulateManifestHashMismatch
    Assert-True (-not $badInt.Ok) 'corrupt_manifest_detected'
    Set-Content -LiteralPath $corruptMeta -Value $manifestBackup -Encoding UTF8

    Open-AceLocalRecoveryLedgerIncident -AceRoot $ace -Kind 'test' -Detail 'retention incident' | Out-Null
    $plan2 = Get-AceLocalRecoveryCheckpointRetentionPlan -AceRoot $ace
    $incidentProtected = $false
    foreach ($reasonList in @($plan2.ProtectedReasons.Values)) {
        if (@($reasonList) -contains 'open_recovery_incident') { $incidentProtected = $true }
    }
    Assert-True $incidentProtected 'incident_checkpoint_protected'
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (@($failures).Count -gt 0) {
    Write-Host ("`nFailures: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalRecovery phase 3 tests passed.' -ForegroundColor Green
exit 0
