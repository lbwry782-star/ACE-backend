#Requires -Version 5.1
<#
  Isolated tests for Publish Uri path classification and plan (temp git repos).
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')
. (Join-Path $ScriptDir 'OriDevCancelDevLib.ps1')
. (Join-Path $ScriptDir 'OriPublishUriLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-OriPublishUriPreview...' -ForegroundColor Cyan

Assert-True ((Get-OriPublishPathClassification -RelativePath 'engine/ace_local_dev_routes.py') -eq 'forbiddenLocal') 'path_forbidden_local'
Assert-True ((Get-OriPublishPathClassification -RelativePath 'engine/builder1_idea_memory.py') -eq 'product') 'path_product_builder1'
Assert-True ((Get-OriPublishPathClassification -RelativePath '.env.local') -eq 'forbiddenSecret') 'path_secret_env'
Assert-True ((Get-OriPublishPathClassification -RelativePath 'random/unknown.txt') -eq 'unclear') 'path_unclear'

$tempRoot = Join-Path $env:TEMP ("ori-publish-uri-test-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
try {
    $backendRoot = Join-Path $tempRoot 'ACE-Backend'
    New-Item -ItemType Directory -Path $backendRoot -Force | Out-Null
    Push-Location $backendRoot
    git init -q
    git config user.email 'test@test.local'
    git config user.name 'Test'
    git branch -M main
    New-Item -ItemType Directory -Path 'engine' -Force | Out-Null
    Set-Content -LiteralPath 'engine/builder1_track.txt' -Value 'prod0' -Encoding UTF8
    git add engine/builder1_track.txt
    git commit -q -m 'prod baseline'
    $prodSha = (git rev-parse HEAD)

    git checkout -q -b ori/builder1-wip
    Set-Content -LiteralPath 'engine/builder1_track.txt' -Value 'prod1' -Encoding UTF8
    git add engine/builder1_track.txt
    git commit -q -m 'product only'
    $productSha = (git rev-parse HEAD)

    Set-Content -LiteralPath 'engine/ace_local_dev_marker.txt' -Value 'x' -Encoding UTF8
    git add engine/ace_local_dev_marker.txt
    git commit -q -m 'infra marker on wip'
    $mixedSha = (git rev-parse HEAD)

    git branch -q ori/local-dev-infrastructure
    Pop-Location

    $rowMixed = Analyze-OriPublishUriCommit -RepoRoot $backendRoot -Sha $mixedSha -Label 'BACKEND' `
        -WipBranch 'ori/builder1-wip' -InfraBranch 'ori/local-dev-infrastructure'
    Assert-True ($rowMixed.classification -eq 'forbiddenLocalOnly') 'forbidden_local_on_wip'
    Assert-True (@($rowMixed.blockers).Count -gt 0) 'forbidden_commit_has_blockers'

    $graph = Get-OriPublishUriBackendCommits -RepoRoot $backendRoot -ProductionSha $prodSha `
        -WipSha $mixedSha -WipBranch 'ori/builder1-wip' -InfraBranch 'ori/local-dev-infrastructure'
    Assert-True (@($graph.Product).Count -ge 1) 'product_commit_in_range'
    Assert-True (@($graph.Blockers).Count -gt 0) 'wip_head_blocked_due_to_infra_file'

    $rowProd = Analyze-OriPublishUriCommit -RepoRoot $backendRoot -Sha $productSha -Label 'BACKEND' `
        -WipBranch 'ori/builder1-wip' -InfraBranch 'ori/local-dev-infrastructure'
    Assert-True ($rowProd.classification -eq 'productCandidate') 'pure_product_commit'
    Assert-True ($rowProd.Ok) 'pure_product_ok'

    $badCand = Test-OriPublishUriCandidateBackendSha -RepoRoot $backendRoot -ProductionSha $prodSha -CandidateSha 'notsha'
    Assert-True (-not $badCand.Ok) 'candidate_sha_invalid_rejected'
    $eqProd = Test-OriPublishUriCandidateBackendSha -RepoRoot $backendRoot -ProductionSha $prodSha -CandidateSha $prodSha
    Assert-True (-not $eqProd.Ok) 'candidate_equals_production_rejected'
    $goodCand = Test-OriPublishUriCandidateBackendSha -RepoRoot $backendRoot -ProductionSha $prodSha -CandidateSha $productSha
    Assert-True ($goodCand.Ok) 'candidate_descendant_ok'
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All Publish Uri preview tests passed.' -ForegroundColor Green
