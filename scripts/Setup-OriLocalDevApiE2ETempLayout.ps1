#Requires -Version 5.1
<#
.SYNOPSIS
  Creates isolated ACE layout for local-dev API E2E (TEMP only). Writes layout JSON path to stdout.
#>
[CmdletBinding()]
param(
    [string]$OutputLayoutJsonPath = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')
. (Join-Path $ScriptDir 'OriDevCancelDevLib.ps1')

function Initialize-TestAceRepo {
    param(
        [string]$Root,
        [string[]]$Commits,
        [string[]]$MarkerFiles = @()
    )
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Push-Location $Root
    git init -q
    git config user.email 'e2e@test.local'
    git config user.name 'E2E'
    git branch -M main
    $shaList = @()
    $first = $true
    foreach ($msg in $Commits) {
        if ($first -and $MarkerFiles.Count -gt 0) {
            foreach ($mf in $MarkerFiles) {
                Set-Content -LiteralPath $mf -Value '# marker' -Encoding UTF8
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
        Prod     = $shaList[0]
        InfraOld = $shaList[1]
        InfraMid = $shaList[2]
        Latest   = $shaList[-1]
    }
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
        production    = @{
            backendWeb    = @{ deployedCommitSha = $BackendProdSha }
            backendWorker = @{ deployedCommitSha = $BackendProdSha }
            frontend      = @{ deployedCommitSha = $FrontendProdSha; workflowRunId = 'e2e-run' }
        }
        localDevInfra = @{
            backend  = @{ branch = 'ori/local-dev-infrastructure'; commitSha = $BackendInfraSha }
            frontend = @{ branch = 'ori/local-dev-infrastructure'; commitSha = $FrontendInfraSha }
        }
        wip           = @{
            builder1 = @{ branch = 'ori/builder1-wip'; commitSha = $WipSha }
        }
    }
    $regPath = Join-Path $Dir 'production-registry.json'
    ($reg | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $regPath -Encoding UTF8
    $hash = Get-FileSha256Hex $regPath
    Set-Content -LiteralPath "$regPath.sha256" -Value "$hash  production-registry.json" -Encoding ASCII
    return $regPath
}

function Copy-OriDevScriptsToBackend {
    param([string]$BackendScriptsDir)
    New-Item -ItemType Directory -Path $BackendScriptsDir -Force | Out-Null
    foreach ($name in @(
            'OriDevSessionBaselineLib.ps1',
            'OriDevCancelDevLib.ps1',
            'New-OriDevSessionBaseline.ps1',
            'Invoke-OriCancelDev.ps1'
        )) {
        Copy-Item -LiteralPath (Join-Path $ScriptDir $name) -Destination (Join-Path $BackendScriptsDir $name) -Force
    }
}

$tempRoot = Join-Path $env:TEMP ("ori-api-e2e-" + [Guid]::NewGuid().ToString('N'))
$aceRoot = Join-Path $tempRoot 'ACE'
$backendRoot = Join-Path $aceRoot 'ACE-Backend'
$frontendRoot = Join-Path $aceRoot 'ACE-Frontend'
$registryDir = Join-Path $aceRoot '_registry'
$sessionsRoot = Join-Path $aceRoot '_backups\sessions'
$scriptsDir = Join-Path $backendRoot 'scripts'
New-Item -ItemType Directory -Path $registryDir, $sessionsRoot -Force | Out-Null

Copy-OriDevScriptsToBackend -BackendScriptsDir $scriptsDir

$bShas = Initialize-TestAceRepo -Root $backendRoot -Commits @('prod', 'infra-old', 'infra-mid', 'wip-base') `
    -MarkerFiles @('app.py', 'worker_video.py')
git -C $backendRoot branch -q ori/local-dev-infrastructure $bShas.InfraOld
git -C $backendRoot branch -q ori/builder1-wip $bShas.Latest

$fShas = Initialize-TestAceRepo -Root $frontendRoot -Commits @('f-prod', 'f-infra-old', 'f-infra-new') `
    -MarkerFiles @('package.json')
git -C $frontendRoot branch -q ori/local-dev-infrastructure $fShas.InfraMid

# Infra branch tip includes layout assets (matches Test-OriCancelDev)
git -C $backendRoot checkout -q ori/local-dev-infrastructure
git -C $backendRoot reset --hard $bShas.InfraMid
Set-Content -LiteralPath (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
git -C $backendRoot add .env.local.example scripts
git -C $backendRoot commit -q -m 'infra-layout-assets'
Set-Content -LiteralPath (Join-Path $backendRoot 'infra-only.txt') -Value 'infra' -Encoding UTF8
git -C $backendRoot add infra-only.txt
git -C $backendRoot commit -q -m 'infra-only'
$backendInfraHead = (git -C $backendRoot rev-parse HEAD).Trim()
$frontendInfraHead = (git -C $frontendRoot rev-parse HEAD).Trim()

$regPath = New-TestRegistry -Dir $registryDir `
    -BackendProdSha $bShas.Prod -FrontendProdSha $fShas.Prod `
    -BackendInfraSha $backendInfraHead -FrontendInfraSha $frontendInfraHead `
    -WipSha $bShas.Latest

# Active WIP dev state before Start Cycle (API will snapshot this into new baseline)
git -C $backendRoot checkout -q ori/builder1-wip
Copy-OriDevScriptsToBackend -BackendScriptsDir $scriptsDir
Set-Content -LiteralPath (Join-Path $backendRoot '.env.local.example') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000" -Encoding UTF8
Set-Content -LiteralPath (Join-Path $backendRoot '.env.local') -Value "ACE_LOCAL_DEV_ENABLED=true`nPORT=5000`nACE_PUBLIC_BASE_URL=http://127.0.0.1:5000`nSESSION_TEMP=pre-start" -Encoding UTF8
Set-Content -LiteralPath (Join-Path $frontendRoot '.env.development.local') -Value 'VITE_BACKEND_URL=http://127.0.0.1:5000' -Encoding UTF8
Set-Content -LiteralPath (Join-Path $backendRoot 'wip-cycle.txt') -Value 'pre-start-wip' -Encoding UTF8

$layout = [ordered]@{
    tempRoot     = $tempRoot
    aceRoot      = $aceRoot
    backendRoot  = $backendRoot
    frontendRoot = $frontendRoot
    registryPath = $regPath
    sessionsRoot = $sessionsRoot
    backendInfraHead = $backendInfraHead
    frontendInfraHead = $frontendInfraHead
    wipBaseSha   = $bShas.Latest
}

if ([string]::IsNullOrWhiteSpace($OutputLayoutJsonPath)) {
    $OutputLayoutJsonPath = Join-Path $tempRoot 'e2e-layout.json'
}
($layout | ConvertTo-Json -Compress) | Set-Content -LiteralPath $OutputLayoutJsonPath -Encoding UTF8
Write-Output $OutputLayoutJsonPath
