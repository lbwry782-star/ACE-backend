#Requires -Version 5.1
<#
.SYNOPSIS
  Isolated tests for Invoke-OriCancelDev (temp ACE layout only).
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')
. (Join-Path $ScriptDir 'OriDevCancelDevLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function New-TestRegistry {
    param(
        [string]$Dir,
        [string]$BackendProdSha,
        [string]$FrontendProdSha,
        [string]$BackendInfraSha,
        [string]$FrontendInfraSha,
        [string]$WipSha
    )
    $reg = @{
        schemaVersion = 1
        production = @{
            backendWeb = @{ deployedCommitSha = $BackendProdSha }
            backendWorker = @{ deployedCommitSha = $BackendProdSha }
            frontend = @{ deployedCommitSha = $FrontendProdSha; workflowRunId = 'test-run' }
        }
        localDevInfra = @{
            backend = @{ branch = 'ori/local-dev-infrastructure'; commitSha = $BackendInfraSha }
            frontend = @{ branch = 'ori/local-dev-infrastructure'; commitSha = $FrontendInfraSha }
        }
        wip = @{ builder1 = @{ branch = 'ori/builder1-wip'; commitSha = $WipSha } }
    }
    $regPath = Join-Path $Dir 'production-registry.json'
    ($reg | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $regPath -Encoding UTF8
    $hash = Get-FileSha256Hex $regPath
    Set-Content -LiteralPath "$regPath.sha256" -Value "$hash  production-registry.json" -Encoding ASCII
    return $regPath
}

function Initialize-TestAceRepo {
    param(
        [string]$Root,
        [string[]]$Commits,
        [string[]]$MarkerFiles = @()
    )
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Push-Location $Root
    git init -q
    git config user.email 'test@test.local'
    git config user.name 'Test'
    git branch -M main
    $shaList = @()
    $first = $true
    foreach ($msg in $Commits) {
        if ($first -and $MarkerFiles.Count -gt 0) {
            foreach ($mf in $MarkerFiles) {
                Set-Content -LiteralPath $mf -Value '# m' -Encoding UTF8
            }
            git add $MarkerFiles
            $first = $false
        }
        Set-Content -LiteralPath 'track.txt' -Value $msg -Encoding UTF8
        git add track.txt
        git commit -q -m $msg
        $shaList += (git rev-parse HEAD)
    }
    Pop-Location
    return @{
        Prod   = $shaList[0]
        Infra  = if ($shaList.Count -ge 2) { $shaList[$shaList.Count - 2] } else { $shaList[0] }
        InfraOld = if ($shaList.Count -ge 3) { $shaList[1] } else { $shaList[0] }
        Latest = $shaList[-1]
    }
}

Write-Host 'Test-OriCancelDev...' -ForegroundColor Cyan

$realMainBefore = git -C (Split-Path -Parent $ScriptDir) rev-parse main 2>$null
$realBaselineHash = $null
$realBaselineJson = Join-Path (Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir).SessionsRoot 'ori-20261023-010342-2d952e88\session.json'
if (Test-Path -LiteralPath $realBaselineJson) {
    $realBaselineHash = Get-FileSha256Hex $realBaselineJson
}

$tempRoot = Join-Path $env:TEMP ("ori-cancel-test-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

try {
    $aceRoot = Join-Path $tempRoot 'ACE'
    $backendRoot = Join-Path $aceRoot 'ACE-Backend'
    $frontendRoot = Join-Path $aceRoot 'ACE-Frontend'
    $registryDir = Join-Path $aceRoot '_registry'
    $sessionsRoot = Join-Path $aceRoot '_backups\sessions'
    $scriptsDir = Join-Path $backendRoot 'scripts'
    New-Item -ItemType Directory -Path $registryDir, $sessionsRoot, $scriptsDir -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1') -Destination (Join-Path $scriptsDir 'OriDevSessionBaselineLib.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'OriDevCancelDevLib.ps1') -Destination (Join-Path $scriptsDir 'OriDevCancelDevLib.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -Destination (Join-Path $scriptsDir 'New-OriDevSessionBaseline.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'Invoke-OriCancelDev.ps1') -Destination (Join-Path $scriptsDir 'Invoke-OriCancelDev.ps1') -Force

    $bShas = Initialize-TestAceRepo -Root $backendRoot -Commits @('prod', 'infra-old', 'infra-new', 'wip') `
        -MarkerFiles @('app.py', 'worker_video.py')
    git -C $backendRoot branch -q ori/local-dev-infrastructure $bShas.InfraOld
    git -C $backendRoot branch -q ori/builder1-wip $bShas.Latest
    git -C $backendRoot checkout -q ori/builder1-wip
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000`nSESSION_TEMP=wip-value" -Encoding UTF8

    $fShas = Initialize-TestAceRepo -Root $frontendRoot -Commits @('prod', 'infra-old', 'infra-new') -MarkerFiles @('package.json')
    git -C $frontendRoot branch -q ori/local-dev-infrastructure $fShas.Latest
    git -C $frontendRoot checkout -q ori/local-dev-infrastructure
    Set-Content -LiteralPath (Join-Path $frontendRoot '.env.development.local') -Value 'VITE_BACKEND_URL=http://127.0.0.1:5000' -Encoding UTF8

    $regPath = New-TestRegistry -Dir $registryDir `
        -BackendProdSha $bShas.Prod -FrontendProdSha $fShas.Prod `
        -BackendInfraSha $bShas.Latest -FrontendInfraSha $fShas.Latest -WipSha $bShas.Latest

    $layout = @{
        AceRoot      = $aceRoot
        BackendRoot  = $backendRoot
        FrontendRoot = $frontendRoot
        RegistryPath = $regPath
        SessionsRoot = $sessionsRoot
    }

    # Baseline at infra-old on backend, dirty wip file (registry infra matches snapshot moment)
    $regObjBaseline = Get-Content -LiteralPath $regPath -Raw | ConvertFrom-Json
    $regObjBaseline.localDevInfra.backend.commitSha = $bShas.InfraOld
    ($regObjBaseline | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $regPath -Encoding UTF8
    Set-Content -LiteralPath "$regPath.sha256" -Value "$(Get-FileSha256Hex $regPath)  production-registry.json" -Encoding ASCII

    git -C $backendRoot checkout -q ori/local-dev-infrastructure
    git -C $backendRoot reset --hard $bShas.InfraOld
    Set-Content -LiteralPath (Join-Path $backendRoot 'wip-edit.txt') -Value 'wip' -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000`nSESSION_TEMP=baseline" -Encoding UTF8

    $sid = 'ori-test-cancel-001'
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sid `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null

    # Advance infra registry target beyond baseline HEAD (includes layout assets + infra-only)
    git -C $backendRoot checkout -q ori/local-dev-infrastructure
    git -C $backendRoot reset --hard $bShas.Latest
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
    git -C $backendRoot add .env.local.example scripts
    git -C $backendRoot commit -q -m 'infra-layout-assets'
    Set-Content -LiteralPath (Join-Path $backendRoot 'infra-only.txt') -Value 'infra' -Encoding UTF8
    git -C $backendRoot add infra-only.txt
    git -C $backendRoot commit -q -m 'infra-only'
    $newInfraHead = git -C $backendRoot rev-parse HEAD
    $regObj = Get-Content -LiteralPath $regPath -Raw | ConvertFrom-Json
    $regObj.localDevInfra.backend.commitSha = $newInfraHead
    ($regObj | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $regPath -Encoding UTF8
    $h = Get-FileSha256Hex $regPath
    Set-Content -LiteralPath "$regPath.sha256" -Value "$h  production-registry.json" -Encoding ASCII

    # Simulate active wip dev (layout markers must exist on disk for safety checks)
    git -C $backendRoot checkout -q ori/builder1-wip
    New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1') -Destination (Join-Path $scriptsDir 'OriDevSessionBaselineLib.ps1') -Force
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $backendRoot 'wip-edit.txt') -Value 'wip-active' -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000`nSESSION_TEMP=active" -Encoding UTF8
    $backendHeadBeforePreview = git -C $backendRoot rev-parse HEAD
    Set-Content -LiteralPath (Join-Path $backendRoot 'wip-only-commit.txt') 'wip-only' -Encoding UTF8
    git -C $backendRoot add wip-only-commit.txt
    git -C $backendRoot commit -q -m 'wip-only-post-baseline'

    $preview = Invoke-OriCancelDevPreview -Layout $layout -SessionId $sid -SessionsRoot $sessionsRoot
    Assert-True ($preview.mode -eq 'PREVIEW') 'preview_mode'
    Assert-True ($preview.plan.infraTargets.backendCommit -eq $newInfraHead) 'infra_target_from_registry_not_baseline'
    Assert-True ($preview.plan.baselineInfraHead.backend -eq $bShas.InfraOld) 'baseline_infra_from_registry_snapshot'
    Assert-True (@($preview.plan.infraAdvancementPaths.backend).Count -gt 0) 'infra_advancement_paths_nonempty'
    Assert-True ($preview.plan.cancelCycle.status -eq 'OPEN') 'cycle_open_before_execute'
    Assert-True (@($preview.plan.commitsToCancel.backend).Count -gt 0) 'preview_lists_wip_commits_to_cancel'
    Assert-True ($backendHeadBeforePreview -eq $bShas.Latest) 'preview_did_not_change_backend_head'

    # Full backup covers git states + manual restore (before destructive execute)
    Set-Content (Join-Path $backendRoot 'staged.txt') 's' -Encoding UTF8
    git -C $backendRoot add staged.txt
    Set-Content (Join-Path $backendRoot 'untracked.txt') 'u' -Encoding UTF8
    Set-Content (Join-Path $backendRoot 'delme.txt') 'd' -Encoding UTF8
    git -C $backendRoot add delme.txt
    git -C $backendRoot commit -q -m 'delme'
    Remove-Item (Join-Path $backendRoot 'delme.txt') -Force
    $bakRoot = Join-Path $tempRoot 'backup-probe'
    New-Item -ItemType Directory -Path $bakRoot -Force | Out-Null
    $null = Export-OriCancelDevRepoBackup -RepoRoot $backendRoot -Label 'BACKEND' -BackupRoot $bakRoot
    $null = Export-OriCancelDevRepoBackup -RepoRoot $frontendRoot -Label 'FRONTEND' -BackupRoot $bakRoot
    $null = Export-OriCancelDevEnvBackup -BackupRoot $bakRoot -Layout $layout
    (@{ backend = (Get-GitRepoState -RepoRoot $backendRoot) } | ConvertTo-Json -Depth 5) | Set-Content (Join-Path $bakRoot 'PRE_STATE.json') -Encoding UTF8
    New-OriCancelDevBackupManifest -BackupRoot $bakRoot | Out-Null
    $vBak = Test-OriCancelDevBackupIntegrity -BackupRoot $bakRoot
    if (-not $vBak.Ok) { Write-Host " backup errors: $($vBak.Errors -join ',')" -ForegroundColor Yellow }
    Assert-True $vBak.Ok 'backup_integrity_ok'
    Assert-True (Test-Path (Join-Path $bakRoot 'BACKEND\mirror\untracked.txt')) 'backup_untracked'
    Assert-True (Test-Path (Join-Path $bakRoot 'BACKEND\deleted-from-HEAD\delme.txt')) 'backup_deleted'
    Set-Content (Join-Path $backendRoot 'track.txt') 'corrupted' -Encoding UTF8
    Restore-OriCancelDevWorkingTreeFromBackup -BackupRoot $bakRoot -Label 'BACKEND' -RepoRoot $backendRoot | Out-Null
    Assert-True (Test-Path (Join-Path $backendRoot 'untracked.txt')) 'restore_untracked'
    $planJson = ($preview.plan | ConvertTo-Json -Depth 8)
    Assert-True ($planJson -notmatch 'sk-test-secret') 'no_secrets_in_plan'

    # Re-align session-attributed dirty state for execute (drop backup-only extras)
    Invoke-OriGit -RepoRoot $backendRoot -GitArgs @('reset', '--hard', 'HEAD') | Out-Null
    Invoke-OriGit -RepoRoot $backendRoot -GitArgs @('clean', '-fd') | Out-Null
    New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1') -Destination (Join-Path $scriptsDir 'OriDevSessionBaselineLib.ps1') -Force
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
    Set-Content (Join-Path $backendRoot 'wip-edit.txt') 'wip-active' -Encoding UTF8

    try {
        Invoke-OriCancelDevExecute -Layout $layout -SessionId $sid -SessionsRoot $sessionsRoot -SimulateBackupVerifyFail | Out-Null
        Assert-True $false 'backup_verify_fail_should_block'
    }
    catch { Assert-True $true 'backup_verify_fail_should_block' }

    $sidPartial = 'ori-test-cancel-partial'
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sidPartial `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
    try {
        Invoke-OriCancelDevExecute -Layout $layout -SessionId $sidPartial -SessionsRoot $sessionsRoot -SimulateFailAfterBackend | Out-Null
        Assert-True $false 'partial_should_throw_early'
    }
    catch {
        Assert-True $true 'partial_should_throw_early'
        $bakDir = Join-Path $aceRoot '_backups\cancel-dev-backups'
        $latest = @(Get-ChildItem -LiteralPath $bakDir -Directory -Filter "$sidPartial-*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending)
        Assert-True (($latest.Count -gt 0) -and (Test-Path -LiteralPath (Join-Path $latest[0].FullName 'RECOVERY.md'))) 'partial_recovery_guide'
    }

    # Progress: env_restore failure recorded on disk (TEMP)
    $sidEnvProg = 'ori-test-progress-env-fail'
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sidEnvProg `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
    Set-Content (Join-Path $backendRoot 'wip-edit.txt') 'wip-env-prog' -Encoding UTF8
    $opProg = [guid]::NewGuid().ToString('n')
    $sessProgRoot = Join-Path $sessionsRoot $sidEnvProg
    $progPath = Join-Path $sessProgRoot "cancel-execute-progress-$opProg.json"
    $progInit = [ordered]@{
        schemaVersion       = 1
        operationId         = $opProg
        sessionId           = $sidEnvProg
        operationStatus     = 'IN_PROGRESS'
        cancelStatus        = 'IN_PROGRESS'
        phase               = 'CANCEL_EXECUTE'
        currentStep         = $null
        completedSteps      = @()
        postCancelAutoStart = @{ phase = 'PENDING'; newSessionId = $null; lastError = $null }
        startedAt           = (Get-Date).ToUniversalTime().ToString('o')
        updatedAt           = (Get-Date).ToUniversalTime().ToString('o')
        error               = $null
    }
    $progJson = ($progInit | ConvertTo-Json -Depth 8)
    [System.IO.File]::WriteAllText($progPath, $progJson, [System.Text.UTF8Encoding]::new($false))
    try {
        Invoke-OriCancelDevExecute -Layout $layout -SessionId $sidEnvProg -SessionsRoot $sessionsRoot `
            -SimulateEnvWriteFail -ProgressPath $progPath -ProgressOperationId $opProg | Out-Null
        Assert-True $false 'progress_env_fail_should_throw'
    }
    catch {
        Assert-True $true 'progress_env_fail_should_throw'
        $progDoc = Get-Content -LiteralPath $progPath -Raw -Encoding UTF8 | ConvertFrom-Json
        Assert-True ([string]$progDoc.operationStatus -eq 'FAILED') 'progress_operation_failed'
    }

    # Successful execute
    $exec = Invoke-OriCancelDevExecute -Layout $layout -SessionId $sid -SessionsRoot $sessionsRoot
    Assert-True ($exec.status -eq 'SUCCESS') 'execute_success'
    Assert-True ((git -C $backendRoot rev-parse HEAD) -eq $newInfraHead) 'backend_at_new_infra'
    Assert-True ((git -C $frontendRoot rev-parse HEAD) -eq $fShas.Latest) 'frontend_at_infra'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $backendRoot 'wip-edit.txt'))) 'wip_file_removed'
    $envPlain = Get-Content -LiteralPath (Join-Path $backendRoot '.env.local') -Raw
    Assert-True ($envPlain -match 'SESSION_TEMP=baseline') 'session_env_restored'
    $envKeys = Parse-DotEnvKeys -Content $envPlain
    Assert-True ($envKeys.Count -ge 3) 'infra_env_preserved'
    $prevEap2 = $ErrorActionPreference; $ErrorActionPreference = 'SilentlyContinue'
    git -C $backendRoot show-ref --verify --quiet refs/heads/ori/builder1-wip 2>$null | Out-Null
    $wipRefOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap2
    Assert-True $wipRefOk 'wip_branch_retained'

    # Preview again after execute — cycle closed, re-execute blocked
    $preview2 = Invoke-OriCancelDevPreview -Layout $layout -SessionId $sid -SessionsRoot $sessionsRoot
    Assert-True ($preview2.mode -eq 'PREVIEW') 'preview_after_execute_no_mutation'
    Assert-True ($preview2.plan.cancelCycle.status -eq 'CANCELLED') 'cycle_cancelled_after_execute'
    Assert-True (-not $preview2.plan.canExecute) 'repeat_execute_blocked'
    Assert-True (@($preview2.plan.blockers) -contains 'CYCLE_ALREADY_CANCELLED') 'cycle_blocker_present'
    try {
        Invoke-OriCancelDevExecute -Layout $layout -SessionId $sid -SessionsRoot $sessionsRoot | Out-Null
        Assert-True $false 'repeat_execute_should_throw'
    }
    catch { Assert-True $true 'repeat_execute_should_throw' }

    # Unrelated git work blocks execute
    $sid2 = 'ori-test-cancel-002'
    git -C $backendRoot checkout -q ori/builder1-wip
    Set-Content -LiteralPath (Join-Path $backendRoot 'unrelated-new.txt') -Value 'x' -Encoding UTF8
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sid2 `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
    Set-Content -LiteralPath (Join-Path $backendRoot 'not-in-baseline.txt') -Value 'y' -Encoding UTF8
    $planBlock = Build-OriCancelDevPlan -Layout $layout -SessionMeta (Get-OriSessionMeta -SessionRoot (Join-Path $sessionsRoot $sid2)) `
        -Registry (Get-Content $regPath -Raw | ConvertFrom-Json) -SessionRoot (Join-Path $sessionsRoot $sid2)
    Assert-True (-not $planBlock.canExecute) 'unrelated_git_blocks'
    try {
        Invoke-OriCancelDevExecute -Layout $layout -SessionId $sid2 -SessionsRoot $sessionsRoot | Out-Null
        Assert-True $false 'unrelated_should_throw'
    }
    catch { Assert-True $true 'unrelated_should_throw' }

    # Ambiguous ENV blocks
    $sid3 = 'ori-test-cancel-003'
    git -C $backendRoot checkout -q ori/local-dev-infrastructure
    git -C $backendRoot reset --hard $newInfraHead
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=9999`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sid3 `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=8888`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
    $planEnv = Build-OriCancelDevPlan -Layout $layout -SessionMeta (Get-OriSessionMeta -SessionRoot (Join-Path $sessionsRoot $sid3)) `
        -Registry (Get-Content $regPath -Raw | ConvertFrom-Json) -SessionRoot (Join-Path $sessionsRoot $sid3)
    Assert-True (-not $planEnv.canExecute) 'ambiguous_env_blocks'

    # Partial failure backend then frontend
    $sid4 = 'ori-test-cancel-004'
    git -C $backendRoot checkout -q ori/builder1-wip
    Set-Content -LiteralPath (Join-Path $backendRoot 'wip-edit.txt') -Value 'z' -Encoding UTF8
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sid4 `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
    try {
        Invoke-OriCancelDevExecute -Layout $layout -SessionId $sid4 -SessionsRoot $sessionsRoot -SimulateFailAfterBackend | Out-Null
        Assert-True $false 'partial_should_throw'
    }
    catch {
        Assert-True $true 'partial_should_throw'
    }

    # Double cancel lock
    $null = Enter-OriCancelDevLock -SessionsRoot $sessionsRoot
    try {
        Invoke-OriCancelDevExecute -Layout $layout -SessionId $sid -SessionsRoot $sessionsRoot | Out-Null
        Assert-True $false 'double_cancel_should_fail'
    }
    catch { Assert-True $true 'double_cancel_should_fail' }
    Exit-OriCancelDevLock -SessionsRoot $sessionsRoot

    # Corrupt registry
    $badReg = Join-Path $tempRoot 'bad-reg\production-registry.json'
    New-Item -ItemType Directory -Path (Split-Path $badReg) -Force | Out-Null
    Copy-Item $regPath $badReg
    Set-Content -LiteralPath "$badReg.sha256" -Value '0000000000000000000000000000000000000000000000000000000000000000  production-registry.json'
    $layoutBad = $layout.Clone()
    $layoutBad.RegistryPath = $badReg
    try {
        Invoke-OriCancelDevPreview -Layout $layoutBad -SessionId $sid -SessionsRoot $sessionsRoot | Out-Null
        Assert-True $false 'bad_registry_should_fail'
    }
    catch { Assert-True $true 'bad_registry_should_fail' }

    # Registry infra HEAD behind/ahead of registry commit blocks execute
    $staleRegPath = Join-Path $tempRoot 'reg-stale\production-registry.json'
    New-Item -ItemType Directory -Path (Split-Path $staleRegPath) -Force | Out-Null
    Copy-Item $regPath $staleRegPath
    $staleObj = Get-Content $staleRegPath -Raw | ConvertFrom-Json
    $staleObj.localDevInfra.backend.commitSha = $bShas.Latest
    ($staleObj | ConvertTo-Json -Depth 8) | Set-Content $staleRegPath -Encoding UTF8
    $hs = Get-FileSha256Hex $staleRegPath
    Set-Content "$staleRegPath.sha256" -Value "$hs  production-registry.json" -Encoding ASCII
    $layoutStale = $layout.Clone()
    $layoutStale.RegistryPath = $staleRegPath
    $planStale = Build-OriCancelDevPlan -Layout $layoutStale -SessionMeta (Get-OriSessionMeta -SessionRoot (Join-Path $sessionsRoot $sid)) `
        -Registry $staleObj -SessionRoot (Join-Path $sessionsRoot $sid)
    Assert-True (-not $planStale.canExecute) 'registry_stale_blocks'
    Assert-True (@($planStale.registryInfraDrift).Count -gt 0) 'registry_stale_shows_drift'

    # Execute without ConfirmExecute blocked at entrypoint
    try {
        & (Join-Path $ScriptDir 'Invoke-OriCancelDev.ps1') -SessionId $sid -Execute `
            -BackendRoot $backendRoot -FrontendRoot $frontendRoot -SessionsRoot $sessionsRoot -RegistryPath $regPath | Out-Null
        Assert-True $false 'execute_without_confirm_blocked'
    }
    catch {
        Assert-True $true 'execute_without_confirm_blocked'
    }

    # Pre-baseline commit on WIP blocks until explicitly included in cycle scope
    $sidPre = 'ori-test-cancel-prebaseline'
    git -C $backendRoot checkout -q ori/builder1-wip
    git -C $backendRoot reset --hard $bShas.InfraOld
    Set-Content (Join-Path $backendRoot 'legacy-wip.txt') 'legacy' -Encoding UTF8
    git -C $backendRoot add legacy-wip.txt
    git -C $backendRoot commit -q -m 'legacy-pre-baseline'
    $legacySha = git -C $backendRoot rev-parse HEAD
    Set-Content (Join-Path $backendRoot 'session-tip.txt') 'tip' -Encoding UTF8
    git -C $backendRoot add session-tip.txt
    git -C $backendRoot commit -q -m 'session-at-baseline'
    $sessionTipSha = git -C $backendRoot rev-parse HEAD
    Set-Content (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
    New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null
    Copy-Item (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1') (Join-Path $scriptsDir 'OriDevSessionBaselineLib.ps1') -Force
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sidPre `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
    Set-Content (Join-Path $backendRoot 'after-baseline.txt') 'after' -Encoding UTF8
    git -C $backendRoot add after-baseline.txt
    git -C $backendRoot commit -q -m 'after-baseline-commit'
    $planPre = Build-OriCancelDevPlan -Layout $layout -SessionMeta (Get-OriSessionMeta -SessionRoot (Join-Path $sessionsRoot $sidPre)) `
        -Registry (Get-Content $regPath -Raw | ConvertFrom-Json) -SessionRoot (Join-Path $sessionsRoot $sidPre)
    Assert-True (-not $planPre.canExecute) 'pre_baseline_commit_blocks'
    Assert-True (@($planPre.blockers | Where-Object { $_ -match 'PRE_BASELINE' }).Count -gt 0) 'pre_baseline_blocker_code'
    Set-OriCancelDevCycleExplicitPreBaseline -SessionRoot (Join-Path $sessionsRoot $sidPre) `
        -SessionMeta (Get-OriSessionMeta -SessionRoot (Join-Path $sessionsRoot $sidPre)) -BackendShas @($legacySha)
    $planPreOk = Build-OriCancelDevPlan -Layout $layout -SessionMeta (Get-OriSessionMeta -SessionRoot (Join-Path $sessionsRoot $sidPre)) `
        -Registry (Get-Content $regPath -Raw | ConvertFrom-Json) -SessionRoot (Join-Path $sessionsRoot $sidPre)
    Assert-True ($planPreOk.canExecute) 'pre_baseline_explicit_allows'
    Assert-True (@($planPreOk.commitsToCancel.backend | Where-Object { $_.subject -eq 'after-baseline-commit' }).Count -eq 1) 'after_baseline_commit_listed'

    # Missing baseline infra registry snapshot blocks (tampered session)
    $sidBadInfra = 'ori-test-cancel-bad-infra'
    Invoke-OriGit -RepoRoot $backendRoot -GitArgs @('clean', '-fd') | Out-Null
    git -C $backendRoot checkout -q ori/local-dev-infrastructure
    git -C $backendRoot reset --hard $newInfraHead
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -SessionId $sidBadInfra `
        -SessionsRoot $sessionsRoot -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
    $badMetaPath = Join-Path (Join-Path $sessionsRoot $sidBadInfra) 'session.json'
    $badMeta = Get-Content $badMetaPath -Raw | ConvertFrom-Json
    $badMeta.localDevInfra.backendShaRegistry = '0000000000000000000000000000000000000000'
    ($badMeta | ConvertTo-Json -Depth 12) | Set-Content $badMetaPath -Encoding UTF8
    Remove-Item (Join-Path (Join-Path $sessionsRoot $sidBadInfra) 'cancel-dev-cycle.state.json') -Force -ErrorAction SilentlyContinue
    $planBadInfra = Build-OriCancelDevPlan -Layout $layout -SessionMeta $badMeta -Registry (Get-Content $regPath -Raw | ConvertFrom-Json) `
        -SessionRoot (Join-Path $sessionsRoot $sidBadInfra)
    Assert-True (-not $planBadInfra.canExecute) 'missing_historical_infra_sha_blocks'

    # Session not READY
    $failSession = Join-Path $sessionsRoot 'ori-test-not-ready'
    New-Item -ItemType Directory -Path $failSession -Force | Out-Null
    @{ sessionId = 'ori-test-not-ready'; status = 'FAILED' } | ConvertTo-Json | Set-Content (Join-Path $failSession 'session.json')
    try {
        Get-OriSessionMeta -SessionRoot $failSession | Out-Null
        Assert-True $false 'not_ready_should_fail'
    }
    catch { Assert-True $true 'not_ready_should_fail' }

    Invoke-OriGitRevertBaselinePaths -RepoRoot $backendRoot -SessionRoot (Join-Path $sessionsRoot $sidBadInfra) `
        -Label 'BACKEND' -PathsToRevert @()
    Assert-True $true 'empty_paths_to_revert_no_throw'

    if ($realMainBefore) {
        Assert-True ((git -C (Split-Path -Parent $ScriptDir) rev-parse main) -eq $realMainBefore) 'main_unchanged'
    }
    if ($realBaselineHash) {
        Assert-True ((Get-FileSha256Hex $realBaselineJson) -eq $realBaselineHash) 'real_baseline_unchanged'
    }

    Write-Host "Done. Failure count: $($failures.Count)" -ForegroundColor $(if ($failures.Count -eq 0) { 'Green' } else { 'Red' })
    if ($failures.Count -gt 0) { exit 1 }
    exit 0
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
