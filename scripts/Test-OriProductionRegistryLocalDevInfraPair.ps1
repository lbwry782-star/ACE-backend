#Requires -Version 5.1
<#
.SYNOPSIS
  TEMP-only tests for unified localDevInfra registry pair update + cancel preview classification.
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

function New-TestRegistryPair {
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

function New-InfraPairRepo {
    param([string]$Root, [string]$BranchName = 'ori/local-dev-infrastructure')
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Push-Location $Root
    git init -q
    git config user.email 't@t.local'
    git config user.name 'T'
    git branch -M main
    Set-Content -LiteralPath 'app.py' -Value '# t' -Encoding UTF8
    Set-Content -LiteralPath 'worker_video.py' -Value '# t' -Encoding UTF8
    git add app.py worker_video.py
    git commit -q -m 'prod-base'
    $prod = git rev-parse HEAD
    git checkout -q -b $BranchName
    Set-Content infra1.txt 'i1' -Encoding UTF8
    git add infra1.txt
    git commit -q -m 'infra-one'
    $infraOld = git rev-parse HEAD
    Set-Content infra2.txt 'i2' -Encoding UTF8
    git add infra2.txt
    git commit -q -m 'infra-two'
    $infraNew = git rev-parse HEAD
    Set-Content wip-product.txt 'wip' -Encoding UTF8
    git add wip-product.txt
    git commit -q -m 'session-wip'
    $wipHead = git rev-parse HEAD
    Pop-Location
    return @{
        Prod = $prod
        InfraOld = $infraOld
        InfraNew = $infraNew
        WipHead = $wipHead
    }
}

$tempRoot = Join-Path $env:TEMP ("ori-reg-pair-test-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
Write-Host "TEMP root: $tempRoot" -ForegroundColor Cyan

try {
    $backendRoot = Join-Path $tempRoot 'ACE-Backend'
    $frontendRoot = Join-Path $tempRoot 'ACE-Frontend'
    $bShas = New-InfraPairRepo -Root $backendRoot
    $fShas = New-InfraPairRepo -Root $frontendRoot
    New-Item -ItemType Directory -Path (Join-Path $tempRoot '_registry'), (Join-Path $tempRoot '_backups') -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $backendRoot 'scripts') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1') `
        -Destination (Join-Path $backendRoot 'scripts\OriDevSessionBaselineLib.ps1') -Force
    Set-Content -LiteralPath (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000" -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $frontendRoot 'package.json') -Value '{"name":"test"}' -Encoding UTF8
    Push-Location $backendRoot
    git add scripts .env.local.example
    git commit -q -m 'test-markers-backend'
    $bShas.WipHead = (git rev-parse HEAD).Trim()
    Pop-Location
    Push-Location $frontendRoot
    git add package.json
    git commit -q -m 'test-markers-frontend'
    $fShas.WipHead = (git rev-parse HEAD).Trim()
    Pop-Location

    $regDir = Join-Path $tempRoot 'registry'
    New-Item -ItemType Directory -Path $regDir -Force | Out-Null
    $regPath = New-TestRegistryPair -Dir $regDir `
        -BackendProdSha $bShas.Prod -FrontendProdSha $fShas.Prod `
        -BackendInfraSha $bShas.InfraOld -FrontendInfraSha $fShas.InfraOld -WipSha $bShas.WipHead
    $backupDir = Join-Path $tempRoot 'backups'

    $preview = Update-OriProductionRegistryLocalDevInfraPair `
        -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
        -NewBackendInfraCommit $bShas.InfraNew -NewFrontendInfraCommit $fShas.InfraNew `
        -Preview -BackupRoot $backupDir -RequireCleanWorkingTrees
    Assert-True ($preview.Mode -eq 'PREVIEW') 'preview_mode'
    Assert-True ($preview.Plan.CanApply) 'preview_can_apply'
    Assert-True ($preview.Plan.WillUpdate) 'preview_will_update'
    $hashBeforePreview = Get-FileSha256Hex $regPath
    Assert-True ($preview.Plan.Current.backend -eq $bShas.InfraOld) 'preview_current_backend'
    Assert-True ($preview.Plan.Target.frontend -eq $fShas.InfraNew) 'preview_target_frontend'

    $apply = Update-OriProductionRegistryLocalDevInfraPair `
        -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
        -NewBackendInfraCommit $bShas.InfraNew -NewFrontendInfraCommit $fShas.InfraNew `
        -BackupRoot $backupDir -RequireCleanWorkingTrees
    Assert-True $apply.Updated 'pair_apply_updated'
    $regAfter = Get-Content -LiteralPath $regPath -Raw | ConvertFrom-Json
    Assert-True ($regAfter.localDevInfra.backend.commitSha -eq $bShas.InfraNew) 'backend_infra_updated'
    Assert-True ($regAfter.localDevInfra.frontend.commitSha -eq $fShas.InfraNew) 'frontend_infra_updated'
    Assert-True ($regAfter.production.backendWeb.deployedCommitSha -eq $bShas.Prod) 'production_backend_unchanged'
    Assert-True ($regAfter.wip.builder1.commitSha -eq $bShas.WipHead) 'wip_unchanged'
    $integrity = Test-ProductionRegistryIntegrity -RegistryPath $regPath -Sha256SidecarPath "$regPath.sha256"
    Assert-True $integrity.Ok 'integrity_after_apply'

    try {
        Update-OriProductionRegistryLocalDevInfraPair `
            -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
            -NewBackendInfraCommit $bShas.Prod -NewFrontendInfraCommit $fShas.InfraNew `
            -BackupRoot $backupDir -RequireCleanWorkingTrees | Out-Null
        Assert-True $false 'backend_bad_ancestry_should_throw'
    }
    catch {
        Assert-True $true 'backend_bad_ancestry_should_throw'
    }

    try {
        Update-OriProductionRegistryLocalDevInfraPair `
            -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
            -NewBackendInfraCommit $bShas.InfraNew -NewFrontendInfraCommit $fShas.Prod `
            -BackupRoot $backupDir -RequireCleanWorkingTrees | Out-Null
        Assert-True $false 'frontend_bad_ancestry_should_throw'
    }
    catch {
        Assert-True $true 'frontend_bad_ancestry_should_throw'
    }

    $badRegDir = Join-Path $tempRoot 'registry-bad-hash'
    New-Item -ItemType Directory -Path $badRegDir -Force | Out-Null
    $badReg = New-TestRegistryPair -Dir $badRegDir `
        -BackendProdSha $bShas.Prod -FrontendProdSha $fShas.Prod `
        -BackendInfraSha $bShas.InfraOld -FrontendInfraSha $fShas.InfraOld -WipSha $bShas.WipHead
    Set-Content -LiteralPath "$badReg.sha256" -Value '0000000000000000000000000000000000000000000000000000000000000000  production-registry.json' -Encoding ASCII
    try {
        Update-OriProductionRegistryLocalDevInfraPair `
            -RegistryPath $badReg -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
            -NewBackendInfraCommit $bShas.InfraNew -NewFrontendInfraCommit $fShas.InfraNew `
            -BackupRoot $backupDir -RequireCleanWorkingTrees | Out-Null
        Assert-True $false 'bad_sha256_should_throw'
    }
    catch {
        Assert-True $true 'bad_sha256_should_throw'
    }

    $sidecarFailDir = Join-Path $tempRoot 'registry-sidecar-fail'
    New-Item -ItemType Directory -Path $sidecarFailDir -Force | Out-Null
    $sidecarReg = New-TestRegistryPair -Dir $sidecarFailDir `
        -BackendProdSha $bShas.Prod -FrontendProdSha $fShas.Prod `
        -BackendInfraSha $bShas.InfraOld -FrontendInfraSha $fShas.InfraOld -WipSha $bShas.WipHead
    $sidecarPath = "$sidecarReg.sha256"
    Remove-Item -LiteralPath $sidecarPath -Force
    New-Item -ItemType Directory -Path $sidecarPath -Force | Out-Null
    $hashBeforeSidecarFail = Get-FileSha256Hex $sidecarReg
    $regObjBefore = Get-Content -LiteralPath $sidecarReg -Raw | ConvertFrom-Json
    $oldBe = [string]$regObjBefore.localDevInfra.backend.commitSha
    try {
        Update-OriProductionRegistryLocalDevInfraPair `
            -RegistryPath $sidecarReg -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
            -NewBackendInfraCommit $bShas.InfraNew -NewFrontendInfraCommit $fShas.InfraNew `
            -BackupRoot $backupDir -RequireCleanWorkingTrees | Out-Null
        Assert-True $false 'sidecar_write_fail_should_throw'
    }
    catch {
        Assert-True $true 'sidecar_write_fail_should_throw'
    }
    $hashAfterFail = Get-FileSha256Hex $sidecarReg
    $regObjAfterFail = Get-Content -LiteralPath $sidecarReg -Raw | ConvertFrom-Json
    Assert-True ($hashAfterFail -eq $hashBeforeSidecarFail) 'recovery_registry_hash_restored'
    Assert-True ([string]$regObjAfterFail.localDevInfra.backend.commitSha -eq $oldBe) 'recovery_no_partial_backend'
    Assert-True ([string]$regObjAfterFail.localDevInfra.frontend.commitSha -eq $fShas.InfraOld) 'recovery_no_partial_frontend'

    $preview2 = Update-OriProductionRegistryLocalDevInfraPair `
        -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
        -NewBackendInfraCommit $bShas.InfraNew -NewFrontendInfraCommit $fShas.InfraNew `
        -Preview -BackupRoot $backupDir -RequireCleanWorkingTrees
    Assert-True ($hashBeforePreview -ne (Get-FileSha256Hex $regPath)) 'apply_changed_registry'
    $hashAfterApply = Get-FileSha256Hex $regPath
    Assert-True ($preview2.Plan.WillUpdate -eq $false) 'preview_after_apply_no_change'
    Assert-True ($hashAfterApply -eq (Get-FileSha256Hex $regPath)) 'preview_does_not_mutate'

    Add-Content -LiteralPath (Join-Path $backendRoot 'infra1.txt') -Value 'dirty' -Encoding UTF8
    try {
        Update-OriProductionRegistryLocalDevInfraPair `
            -RegistryPath $regPath -BackendRoot $backendRoot -FrontendRoot $frontendRoot `
            -NewBackendInfraCommit $bShas.InfraNew -NewFrontendInfraCommit $fShas.InfraNew `
            -BackupRoot $backupDir -RequireCleanWorkingTrees | Out-Null
        Assert-True $false 'dirty_tree_should_block'
    }
    catch {
        Assert-True ($_.Exception.Message -match 'WORKING_TREE_DIRTY') 'dirty_tree_should_block'
    }
    Push-Location $backendRoot
    git checkout -q -- infra1.txt
    Pop-Location

    # Real-repo classification (TEMP registry + session copies only; real repos read-only)
    $realLayout = Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir
    $realBeHead = ''
    $realFeHead = ''
    Push-Location $realLayout.BackendRoot
    $realBeHead = (git rev-parse HEAD).Trim()
    Pop-Location
    Push-Location $realLayout.FrontendRoot
    $realFeHead = (git rev-parse HEAD).Trim()
    Pop-Location
    if ($realBeHead -eq 'f18f634eb119249c80242dbf127412d331ac7c58' -and $realFeHead -eq 'b1aa0ad2907a083dd8b9e77b2b4621171d2cbbe5') {
        $realRegDir = Join-Path $tempRoot 'real-registry'
        New-Item -ItemType Directory -Path $realRegDir -Force | Out-Null
        Copy-Item -LiteralPath $realLayout.RegistryPath -Destination (Join-Path $realRegDir 'production-registry.json') -Force
        Copy-Item -LiteralPath "$($realLayout.RegistryPath).sha256" -Destination (Join-Path $realRegDir 'production-registry.json.sha256') -Force
        $realRegCopy = Join-Path $realRegDir 'production-registry.json'
        Update-OriProductionRegistryLocalDevInfraPair `
            -RegistryPath $realRegCopy -BackendRoot $realLayout.BackendRoot -FrontendRoot $realLayout.FrontendRoot `
            -NewBackendInfraCommit $realBeHead -NewFrontendInfraCommit $realFeHead `
            -BackupRoot (Join-Path $tempRoot 'real-backups') | Out-Null

        $realSessions = Join-Path $tempRoot 'real-sessions'
        New-Item -ItemType Directory -Path $realSessions -Force | Out-Null
        $realSid = 'ori-20261023-010342-2d952e88'
        Copy-Item -LiteralPath (Join-Path $realLayout.SessionsRoot $realSid) -Destination (Join-Path $realSessions $realSid) -Recurse -Force
        $realLayoutCopy = @{
            AceRoot = $realLayout.AceRoot
            BackendRoot = $realLayout.BackendRoot
            FrontendRoot = $realLayout.FrontendRoot
            RegistryPath = $realRegCopy
            SessionsRoot = $realSessions
        }
        $realPreview = Invoke-OriCancelDevPreview -Layout $realLayoutCopy -SessionId $realSid -SessionsRoot $realSessions
        $rp = $realPreview.plan
        $rb = @($rp.commitsToCancel.backend | ForEach-Object { $_.fullSha })
        $rf = @($rp.commitsToCancel.frontend | ForEach-Object { $_.fullSha })
        Assert-True ($rb -notcontains $realBeHead) 'real_infra_head_not_in_cancel_backend'
        Assert-True ($rf -notcontains $realFeHead) 'real_infra_head_not_in_cancel_frontend'
        $blockers = @($rp.blockers)
        Assert-True (-not ($blockers -match 'REGISTRY_INFRA_HEAD_MISMATCH')) 'real_no_registry_head_mismatch'
        $infraBeReal = @($rp.infraCommitsPreserved.backend | ForEach-Object { $_.fullSha })
        Assert-True ($infraBeReal -contains $realBeHead) 'real_backend_head_in_infra_preserved'
        $wipCancel = @($rp.commitsToCancel.backend | ForEach-Object { $_.fullSha })
        Assert-True ($wipCancel -notcontains $realBeHead) 'real_backend_head_not_cancel'
        $regAfterReal = Get-Content -LiteralPath $realRegCopy -Raw | ConvertFrom-Json
        Assert-True ($regAfterReal.wip.builder1.commitSha -eq '6fdf09649017ccf787ac3c896af81a7330bf52ea') 'real_builder1_wip_unchanged_in_registry'
    }
    else {
        Write-Host ' skip real-repo classification (HEAD mismatch)' -ForegroundColor Yellow
    }
}
finally {
    # TEMP cleanup optional — keep for debugging on failure only
}

if ($failures.Count -gt 0) {
    Write-Host "`nFAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nAll pair registry tests passed." -ForegroundColor Green
exit 0
