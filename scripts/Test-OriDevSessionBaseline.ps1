#Requires -Version 5.1
<#
.SYNOPSIS
  Isolated tests for Ori dev session baseline (temp dirs only).
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

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

function New-ThreeCommitRepo {
    param([string]$Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    Push-Location $Path
    git init -q
    git config user.email 'test@test.local'
    git config user.name 'Test'
    git branch -M main
    foreach ($msg in @('prod', 'infra', 'wip')) {
        Set-Content -LiteralPath 'file.txt' -Value $msg -Encoding UTF8
        git add file.txt
        git commit -q -m $msg
    }
    $wip = git rev-parse HEAD
    $infra = git rev-parse HEAD~1
    $prod = git rev-parse HEAD~2
    Pop-Location
    return @{ Prod = $prod; Infra = $infra; Wip = $wip }
}

function New-TwoCommitRepo {
    param([string]$Path)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    Push-Location $Path
    git init -q
    git config user.email 'test@test.local'
    git config user.name 'Test'
    git branch -M main
    foreach ($msg in @('prod', 'infra')) {
        Set-Content -LiteralPath 'file.txt' -Value $msg -Encoding UTF8
        git add file.txt
        git commit -q -m $msg
    }
    $infra = git rev-parse HEAD
    $prod = git rev-parse HEAD~1
    Pop-Location
    return @{ Prod = $prod; Infra = $infra }
}

Write-Host 'Test-OriDevSessionBaseline...' -ForegroundColor Cyan

$realBaselineSessionJson = Join-Path (Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir).SessionsRoot `
    'ori-20261023-010342-2d952e88\session.json'
$baselineHashBefore = $null
if (Test-Path -LiteralPath $realBaselineSessionJson) {
    $baselineHashBefore = Get-FileSha256Hex $realBaselineSessionJson
}

$tempRoot = Join-Path $env:TEMP ("ori-baseline-test-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

try {
    $backendRoot = Join-Path $tempRoot 'ACE-Backend'
    $frontendRoot = Join-Path $tempRoot 'ACE-Frontend'
    $sessionsRoot = Join-Path $tempRoot 'sessions'
    $registryDir = Join-Path $tempRoot 'registry'
    New-Item -ItemType Directory -Path $registryDir, $sessionsRoot -Force | Out-Null

    $bShas = New-ThreeCommitRepo -Path $backendRoot
    Push-Location $backendRoot
    Set-Content -LiteralPath 'app.py' -Value '# test marker' -Encoding UTF8
    Set-Content -LiteralPath 'worker_video.py' -Value '# test marker' -Encoding UTF8
    git add app.py worker_video.py
    git commit -q -m 'ace-layout-markers'
    $bShas.Wip = git rev-parse HEAD
    Set-Content -LiteralPath '.env.local.example' -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000" -Encoding UTF8
    Set-Content -LiteralPath '.env.local' -Value "ACE_LOCAL_DEV_ENABLED=true`nOPENAI_API_KEY=sk-test-secret-value`nSESSION_FLAG=temp" -Encoding UTF8
    Set-Content -LiteralPath 'dirty.txt' -Value 'original' -Encoding UTF8
    git add dirty.txt
    git commit -q -m 'add-dirty'
    Set-Content -LiteralPath 'dirty.txt' -Value 'modified-working-tree' -Encoding UTF8
    Set-Content -LiteralPath 'new-untracked.txt' -Value 'new' -Encoding UTF8
    Pop-Location

    $fShas = New-TwoCommitRepo -Path $frontendRoot
    Push-Location $frontendRoot
    Set-Content -LiteralPath 'package.json' -Value '{"name":"ace-frontend-test"}' -Encoding UTF8
    git add package.json
    git commit -q -m 'ace-layout-marker'
    $fShas.Infra = git rev-parse HEAD
    Set-Content -LiteralPath '.env.development.local' -Value 'VITE_BACKEND_URL=http://127.0.0.1:5000' -Encoding UTF8
    Pop-Location
    Push-Location $backendRoot
    $bShas.Wip = git rev-parse HEAD
    Pop-Location

    $regPath = New-TestRegistry -Dir $registryDir `
        -BackendProdSha $bShas.Prod -FrontendProdSha $fShas.Prod `
        -BackendInfraSha $bShas.Infra -FrontendInfraSha $fShas.Infra -WipSha $bShas.Wip

    $sid1 = 'ori-test-valid-001'
    $summary = & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') `
        -SessionId $sid1 -SessionsRoot $sessionsRoot -RegistryPath $regPath `
        -BackendRoot $backendRoot -FrontendRoot $frontendRoot
    $session1 = Join-Path $sessionsRoot $sid1
    $meta1 = Get-Content (Join-Path $session1 'session.json') -Raw | ConvertFrom-Json
    Assert-True ($meta1.status -eq 'READY') 'valid_session_ready'
    Assert-True (Test-Path (Join-Path $session1 'ENV\backend.env.local.dpapi')) 'env_dpapi_backend'
    Assert-True (Test-Path (Join-Path $session1 'BACKEND\mirror\dirty.txt')) 'git_modified_mirrored'
    Assert-True (Test-Path (Join-Path $session1 'BACKEND\mirror\new-untracked.txt')) 'git_untracked_mirrored'

    $badReg = Join-Path $registryDir 'bad-registry.json'
    Copy-Item $regPath $badReg
    Set-Content -LiteralPath "$badReg.sha256" -Value '0000000000000000000000000000000000000000000000000000000000000000  production-registry.json'
    $sid2 = 'ori-test-bad-reg-002'
    try {
        & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') `
            -SessionId $sid2 -SessionsRoot $sessionsRoot -RegistryPath $badReg `
            -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
        Assert-True $false 'corrupt_registry_should_fail'
    }
    catch {
        Assert-True $true 'corrupt_registry_should_fail'
    }
    $meta2 = Get-Content (Join-Path (Join-Path $sessionsRoot $sid2) 'session.json') -Raw | ConvertFrom-Json
    Assert-True ($meta2.status -eq 'FAILED') 'corrupt_registry_marked_failed'

    $sidDup = 'ori-test-dup-003'
    try {
        & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') `
            -SessionId $sid1 -SessionsRoot $sessionsRoot -RegistryPath $regPath `
            -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
        Assert-True $false 'duplicate_session_should_fail'
    }
    catch {
        Assert-True $true 'duplicate_session_should_fail'
    }
    $meta1AfterDup = Get-Content (Join-Path $session1 'session.json') -Raw | ConvertFrom-Json
    Assert-True ($meta1AfterDup.status -eq 'READY') 'duplicate_does_not_corrupt_existing_session'

    $backendNoEnv = Join-Path $tempRoot 'ACE-Backend-NoEnv'
    Copy-Item -LiteralPath $backendRoot -Destination $backendNoEnv -Recurse -Force
    Remove-Item -LiteralPath (Join-Path $backendNoEnv '.env.local') -Force -ErrorAction SilentlyContinue
    $sid4 = 'ori-test-no-env-004'
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') `
        -SessionId $sid4 -SessionsRoot $sessionsRoot -RegistryPath $regPath `
        -BackendRoot $backendNoEnv -FrontendRoot $frontendRoot | Out-Null
    $meta4 = Get-Content (Join-Path (Join-Path $sessionsRoot $sid4) 'session.json') -Raw | ConvertFrom-Json
    Assert-True ($meta4.status -eq 'READY') 'missing_env_still_ready'
    $envBackend = $meta4.envSnapshots | Where-Object { $_.LogicalName -eq 'backend.env.local' } | Select-Object -First 1
    Assert-True (-not $envBackend.Present) 'missing_env_flagged_absent'

    $rawSession = Get-Content (Join-Path $session1 'session.json') -Raw
    Assert-True ($rawSession -notmatch 'sk-test-secret') 'no_secret_in_session_json'
    $manifestCsv = Get-Content (Join-Path $session1 'ENV\env-keys-manifest.csv') -Raw
    Assert-True ($manifestCsv -notmatch 'sk-test-secret') 'no_secret_in_env_manifest_csv'

    $sid7 = 'ori-test-midfail-007'
    try {
        & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') `
            -SessionId $sid7 -SessionsRoot $sessionsRoot -RegistryPath $regPath `
            -BackendRoot $backendRoot -FrontendRoot $frontendRoot -FailAfterGitBackend | Out-Null
        Assert-True $false 'mid_fail_should_throw'
    }
    catch {
        Assert-True $true 'mid_fail_should_throw'
    }
    $meta7 = Get-Content (Join-Path (Join-Path $sessionsRoot $sid7) 'session.json') -Raw | ConvertFrom-Json
    Assert-True ($meta7.status -eq 'FAILED') 'mid_fail_not_ready'
    Assert-True (-not (Test-Path (Join-Path (Join-Path $sessionsRoot $sid7) 'VERIFY_SUMMARY.json'))) 'mid_fail_no_verify_summary'

    $dpapi = Join-Path $session1 'ENV\backend.env.local.dpapi'
    $bytes = [System.IO.File]::ReadAllBytes($dpapi)
    $plain = [System.Text.Encoding]::UTF8.GetString((Unprotect-ContentForCurrentUser -ProtectedBytes $bytes))
    $expectedSha = (Get-Content (Join-Path $session1 'MANIFEST.csv') | ConvertFrom-Csv | Where-Object { $_.RelPath -eq 'backend.env.local' }).SourceSha256
    Assert-True ((Get-TextSha256Hex $plain) -eq $expectedSha) 'env_restore_roundtrip_hash'

    $okReg = Test-ProductionRegistryIntegrity -RegistryPath $regPath -Sha256SidecarPath "$regPath.sha256"
    Assert-True $okReg.Ok 'registry_integrity_helper'

    # Layout resolution without explicit paths (from this repo scripts dir)
    $layoutNoParams = Resolve-OriProjectLayout -ScriptsDirectory $ScriptDir
    Assert-True (Test-Path -LiteralPath $layoutNoParams.RegistryPath) 'real_layout_registry_resolves'
    $validateReal = & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -ValidateOnly
    Assert-True ($validateReal.status -eq 'VALIDATE_OK') 'real_validate_only_no_path_params'

    # Registry infra commit bump (temp copy only)
    $regBumpDir = Join-Path $tempRoot 'reg-bump'
    New-Item -ItemType Directory -Path $regBumpDir -Force | Out-Null
    $regBumpPath = Join-Path $regBumpDir 'production-registry.json'
    Copy-Item -LiteralPath $regPath -Destination $regBumpPath -Force
    Copy-Item -LiteralPath "$regPath.sha256" -Destination "$regBumpPath.sha256" -Force
    $prodBefore = (Get-Content -LiteralPath $regBumpPath -Raw | ConvertFrom-Json).production.backendWeb.deployedCommitSha
    $regBackupDir = Join-Path $tempRoot 'reg-bump-backups'
    $bump = Update-OriProductionRegistryBackendInfraCommit `
        -RegistryPath $regBumpPath -NewCommitSha $bShas.Wip -BackendRoot $backendRoot -BackupRoot $regBackupDir
    Assert-True $bump.Updated 'registry_infra_bump_updated'
    $regAfter = Get-Content -LiteralPath $regBumpPath -Raw | ConvertFrom-Json
    Assert-True ($regAfter.localDevInfra.backend.commitSha -eq $bShas.Wip) 'registry_infra_sha_bumped'
    Assert-True ($regAfter.production.backendWeb.deployedCommitSha -eq $prodBefore) 'registry_production_sha_unchanged'
    $bumpAgain = Update-OriProductionRegistryBackendInfraCommit `
        -RegistryPath $regBumpPath -NewCommitSha $bShas.Wip -BackendRoot $backendRoot -BackupRoot $regBackupDir
    Assert-True (-not $bumpAgain.Updated) 'registry_infra_idempotent'
    try {
        Update-OriProductionRegistryBackendInfraCommit `
            -RegistryPath $regBumpPath -NewCommitSha $bShas.Prod -BackendRoot $backendRoot -BackupRoot $regBackupDir | Out-Null
        Assert-True $false 'registry_non_descendant_should_fail'
    }
    catch {
        Assert-True $true 'registry_non_descendant_should_fail'
    }

    # Deleted file test (separate mini repo)
    $delRoot = Join-Path $tempRoot 'ACE-Backend-Del'
    New-Item -ItemType Directory -Path $delRoot -Force | Out-Null
    Push-Location $delRoot
    git init -q
    git config user.email 't@t.local'
    git config user.name 'T'
    git branch -M main
    Set-Content -LiteralPath 'app.py' -Value '# t' -Encoding UTF8
    Set-Content -LiteralPath 'worker_video.py' -Value '# t' -Encoding UTF8
    Set-Content removed.txt 'x' -Encoding UTF8
    git add app.py worker_video.py removed.txt
    git commit -q -m 'one'
    $onlySha = git rev-parse HEAD
    Remove-Item -LiteralPath 'removed.txt' -Force
    Pop-Location
    $regDelDir = Join-Path $tempRoot 'reg-del'
    New-Item -ItemType Directory -Path $regDelDir -Force | Out-Null
    $regDel = New-TestRegistry -Dir $regDelDir `
        -BackendProdSha $onlySha -FrontendProdSha $fShas.Prod `
        -BackendInfraSha $onlySha -FrontendInfraSha $fShas.Infra -WipSha $onlySha
    $sidDel = 'ori-test-deleted-008'
    & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') `
        -SessionId $sidDel -SessionsRoot $sessionsRoot -RegistryPath $regDel `
        -BackendRoot $delRoot -FrontendRoot $frontendRoot | Out-Null
    Assert-True (Test-Path (Join-Path $sessionsRoot "$sidDel\BACKEND\deleted-from-HEAD\removed.txt")) 'deleted_file_snapshot'

    # Hebrew ACE root layout + invocation without explicit registry/sessions paths
    $heRoot = Join-Path $tempRoot 'אורי-ACE'
    $heBackend = Join-Path $heRoot 'ACE-Backend'
    $heFrontend = Join-Path $heRoot 'ACE-Frontend'
    Copy-Item -LiteralPath $backendRoot -Destination $heBackend -Recurse -Force
    Copy-Item -LiteralPath $frontendRoot -Destination $heFrontend -Recurse -Force
    $heRegistryDir = Join-Path $heRoot '_registry'
    $heSessions = Join-Path $heRoot '_backups\sessions'
    New-Item -ItemType Directory -Path $heRegistryDir, $heSessions -Force | Out-Null
    Copy-Item -LiteralPath $regPath -Destination (Join-Path $heRegistryDir 'production-registry.json') -Force
    Copy-Item -LiteralPath "$regPath.sha256" -Destination (Join-Path $heRegistryDir 'production-registry.json.sha256') -Force
    $heScripts = Join-Path $heBackend 'scripts'
    New-Item -ItemType Directory -Path $heScripts -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1') -Destination (Join-Path $heScripts 'OriDevSessionBaselineLib.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') -Destination (Join-Path $heScripts 'New-OriDevSessionBaseline.ps1') -Force
    $heLayout = Resolve-OriProjectLayout -ScriptsDirectory $heScripts
    Assert-True (Test-Path -LiteralPath $heLayout.RegistryPath) 'hebrew_layout_registry'
    $heValidate = & (Join-Path $heBackend 'scripts\New-OriDevSessionBaseline.ps1') -ValidateOnly
    Assert-True ($heValidate.status -eq 'VALIDATE_OK') 'hebrew_validate_only_no_params'
    $sidHe = 'ori-test-hebrew-no-params-010'
    & (Join-Path $heBackend 'scripts\New-OriDevSessionBaseline.ps1') -SessionId $sidHe | Out-Null
    $metaHe = Get-Content -LiteralPath (Join-Path (Join-Path $heSessions $sidHe) 'session.json') -Raw | ConvertFrom-Json
    Assert-True ($metaHe.status -eq 'READY') 'hebrew_no_explicit_paths_session'

    # Registry with older infra SHA (e41f418) should still validate if commits exist in backend
    $oldInfraRegDir = Join-Path $tempRoot 'reg-old-infra'
    New-Item -ItemType Directory -Path $oldInfraRegDir -Force | Out-Null
    $oldInfraReg = New-TestRegistry -Dir $oldInfraRegDir `
        -BackendProdSha $bShas.Prod -FrontendProdSha $fShas.Prod `
        -BackendInfraSha 'e41f41834392858f7b5ab20c8acfc5ee162b7957' -FrontendInfraSha $fShas.Infra -WipSha $bShas.Wip
    # only works if e41f418 exists in backendRoot - skip if not
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    git -C $backendRoot rev-parse --verify 'e41f41834392858f7b5ab20c8acfc5ee162b7957^{commit}' 2>$null | Out-Null
    $hasRealOldInfra = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap
    if ($hasRealOldInfra) {
        $sidOld = 'ori-test-old-infra-reg-009'
        & (Join-Path $ScriptDir 'New-OriDevSessionBaseline.ps1') `
            -SessionId $sidOld -SessionsRoot $sessionsRoot -RegistryPath $oldInfraReg `
            -BackendRoot $backendRoot -FrontendRoot $frontendRoot | Out-Null
        $metaOld = Get-Content -LiteralPath (Join-Path (Join-Path $sessionsRoot $sidOld) 'session.json') -Raw | ConvertFrom-Json
        Assert-True ($metaOld.status -eq 'READY') 'registry_old_infra_sha_still_valid'
    }

    if ($baselineHashBefore) {
        Assert-True ((Get-FileSha256Hex $realBaselineSessionJson) -eq $baselineHashBefore) 'existing_real_baseline_unchanged'
    }

    Write-Host "Done. Failure count: $($failures.Count)" -ForegroundColor $(if ($failures.Count -eq 0) { 'Green' } else { 'Red' })
    if ($failures.Count -gt 0) { exit 1 }
    exit 0
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
