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
    Set-Content -LiteralPath '.env.development.local' -Value 'VITE_BACKEND_URL=http://127.0.0.1:5000' -Encoding UTF8
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

    # Deleted file test (separate mini repo)
    $delRoot = Join-Path $tempRoot 'ACE-Backend-Del'
    New-Item -ItemType Directory -Path $delRoot -Force | Out-Null
    Push-Location $delRoot
    git init -q
    git config user.email 't@t.local'
    git config user.name 'T'
    git branch -M main
    Set-Content removed.txt 'x' -Encoding UTF8
    git add removed.txt
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

    Write-Host "Done. Failure count: $($failures.Count)" -ForegroundColor $(if ($failures.Count -eq 0) { 'Green' } else { 'Red' })
    if ($failures.Count -gt 0) { exit 1 }
    exit 0
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
