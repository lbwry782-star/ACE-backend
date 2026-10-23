#Requires -Version 5.1
<#
  Read-only Publish Uri plan — product vs local-dev-infra separation for live deploy preview.
#>
Set-StrictMode -Version Latest
# Requires OriDevCancelDevLib.ps1 to be dot-sourced by the caller.

function Get-OriPublishUriForbiddenPathPatterns {
    return @(
        '(?i)^engine/ace_local_dev_'
        '(?i)^engine/ace_local_dev/'
        '(?i)^scripts/Ori'
        '(?i)^scripts/Export-Ori'
        '(?i)^scripts/Update-OriProduction'
        '(?i)^scripts/New-OriDev'
        '(?i)^scripts/Start-Ace'
        '(?i)^scripts/Stop-Ace'
        '(?i)DevLocalControl'
        '(?i)OriDevCancelDevLib'
        '(?i)OriDevSessionBaselineLib'
        '(?i)OriPublishUriLib'
        '(?i)Export-OriPublishUri'
        '(?i)ace_local_dev_'
        '(?i)cancel-dev'
        '(?i)local-dev-infrastructure'
    )
}

function Get-OriPublishUriSecretPathPatterns {
    return @(
        '(?i)^\.env(\.|$)'
        '(?i)credentials'
        '(?i)dpapi'
        '(?i)secrets\.json'
        '(?i)\.pem$'
        '(?i)\.key$'
        '(?i)id_rsa'
    )
}

function Get-OriPublishUriProductPathPatterns {
    return @(
        '(?i)^engine/builder1_'
        '(?i)^engine/builder2_'
        '(?i)^app\.py$'
        '(?i)^tests/test_builder'
        '(?i)^src/'
        '(?i)^public/'
        '(?i)^index\.html$'
        '(?i)^vite\.config'
    )
}

function Get-OriPublishPathClassification {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath
    )
    $norm = ($RelativePath -replace '\\', '/').Trim()
    if ([string]::IsNullOrWhiteSpace($norm)) { return 'unclear' }
    foreach ($pat in (Get-OriPublishUriSecretPathPatterns)) {
        if ($norm -match $pat) { return 'forbiddenSecret' }
    }
    foreach ($pat in (Get-OriPublishUriForbiddenPathPatterns)) {
        if ($norm -match $pat) { return 'forbiddenLocal' }
    }
    foreach ($pat in (Get-OriPublishUriProductPathPatterns)) {
        if ($norm -match $pat) { return 'product' }
    }
    return 'unclear'
}

function Get-GitCommitChangedPaths {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Sha
    )
    $r = Invoke-OriGitLines -RepoRoot $RepoRoot -GitArgs @(
        'diff-tree', '--no-commit-id', '--name-only', '-r', $Sha
    )
    if ($r.Code -ne 0) { return @() }
    return @($r.Lines | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Test-OriPublishUriCommitSecurityPatch {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Sha
    )
    $r = Invoke-OriGitLines -RepoRoot $RepoRoot -GitArgs @('show', $Sha, '--format=', '-p')
    if ($r.Code -ne 0) { return @() }
    $text = ($r.Lines -join "`n")
    $flags = @()
    if ($text -match '(?i)ACE_SECURITY_ENABLED\s*=\s*[''"]?(false|0)') { $flags += 'ACE_SECURITY_WEAKENED' }
    if ($text -match '(?i)ACE_LOCAL_DEV_ENABLED\s*=\s*[''"]?(true|1)') { $flags += 'ACE_LOCAL_DEV_ENABLED_IN_DIFF' }
    if ($text -match '(?i)BUILDER1_PRODUCTION_MODE\s*=\s*[''"]?(false|0)') { $flags += 'BUILDER1_PRODUCTION_MODE_DISABLED' }
    if ($text -match '(?i)ace_payment.*(?:skip|bypass|disable)') { $flags += 'PAYMENT_BYPASS_HINT' }
    return @($flags | Select-Object -Unique)
}

function Test-OriPublishCommitBranchContext {
    param(
        [Parameter(Mandatory = $true)][string[]]$Branches,
        [Parameter(Mandatory = $true)][string]$WipBranch,
        [Parameter(Mandatory = $true)][string]$InfraBranch
    )
    $onWip = $false
    $onInfra = $false
    foreach ($b in $Branches) {
        $bl = $b -replace '^remotes/origin/', ''
        if ($bl -eq $WipBranch -or $bl -like "*$WipBranch") { $onWip = $true }
        if ($bl -eq $InfraBranch -or $bl -like "*$InfraBranch") { $onInfra = $true }
    }
    return @{ OnWip = $onWip; OnInfra = $onInfra; InfraOnly = ($onInfra -and -not $onWip) }
}

function Analyze-OriPublishUriCommit {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Sha,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$WipBranch,
        [Parameter(Mandatory = $true)][string]$InfraBranch
    )
    $sum = Get-GitCommitSummary -RepoRoot $RepoRoot -Sha $Sha
    if (-not $sum) {
        return @{
            Ok = $false
            Blocker = "${Label}_COMMIT_SUMMARY_MISSING:$Sha"
        }
    }
    $branches = @(Get-GitBranchesContaining -RepoRoot $RepoRoot -Sha $Sha)
    $branchCtx = Test-OriPublishCommitBranchContext -Branches $branches -WipBranch $WipBranch -InfraBranch $InfraBranch
    $paths = @(Get-GitCommitChangedPaths -RepoRoot $RepoRoot -Sha $Sha)
    $pathRows = @()
    $hasProduct = $false
    $hasForbidden = $false
    $hasUnclear = $false
    foreach ($p in $paths) {
        $kind = Get-OriPublishPathClassification -RelativePath $p
        $pathRows += @{ path = $p; classification = $kind }
        switch ($kind) {
            'product' { $hasProduct = $true }
            'forbiddenLocal' { $hasForbidden = $true }
            'forbiddenSecret' { $hasForbidden = $true }
            default { $hasUnclear = $true }
        }
    }
    $securityFlags = @(Test-OriPublishUriCommitSecurityPatch -RepoRoot $RepoRoot -Sha $Sha)
    $classification = 'productCandidate'
    $blockers = @()
    if ($branchCtx.InfraOnly) {
        $classification = 'excludedLocalInfraBranch'
    }
    elseif ($hasForbidden -and ($hasProduct -or $hasUnclear)) {
        $classification = 'mixedProductAndLocal'
        $blockers += "${Label}_PUBLISH_MIXED_COMMIT:$($sum.ShortSha)"
    }
    elseif ($hasForbidden) {
        $classification = 'forbiddenLocalOnly'
        if ($branchCtx.OnWip) {
            $blockers += "${Label}_PUBLISH_FORBIDDEN_ON_WIP:$($sum.ShortSha)"
        }
    }
    elseif ($hasUnclear) {
        $classification = 'unclearPaths'
        $blockers += "${Label}_PUBLISH_UNCLEAR_PATHS:$($sum.ShortSha)"
    }
    elseif (-not $hasProduct -and $paths.Count -eq 0) {
        $classification = 'emptyCommit'
    }
    foreach ($f in $securityFlags) {
        $blockers += "${Label}_PUBLISH_SECURITY:$($sum.ShortSha):$f"
    }
    return @{
        Ok = ($blockers.Count -eq 0)
        fullSha = $sum.FullSha
        shortSha = $sum.ShortSha
        subject = $sum.Subject
        branches = $branches
        classification = $classification
        paths = $pathRows
        securityFlags = $securityFlags
        blockers = $blockers
        branchContext = $branchCtx
    }
}

function Get-OriWipRegistryDrift {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$WipBranch,
        [Parameter(Mandatory = $true)][string]$RegistryWipSha
    )
    Push-Location -LiteralPath $BackendRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        git rev-parse --verify "$RegistryWipSha^{commit}" 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            return @("REGISTRY_WIP_COMMIT_MISSING:backend:$RegistryWipSha")
        }
        $branchHead = (git rev-parse $WipBranch 2>$null).Trim()
        if ([string]::IsNullOrWhiteSpace($branchHead)) {
            return @("REGISTRY_WIP_BRANCH_MISSING:backend:$WipBranch")
        }
        if ($branchHead -ne $RegistryWipSha) {
            return @("REGISTRY_WIP_HEAD_MISMATCH:backend|branchHead=$branchHead|registry=$RegistryWipSha")
        }
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
    return @()
}

function Get-OriPublishUriBackendCommits {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$ProductionSha,
        [Parameter(Mandatory = $true)][string]$WipSha,
        [Parameter(Mandatory = $true)][string]$WipBranch,
        [Parameter(Mandatory = $true)][string]$InfraBranch
    )
    $errors = @()
    if (-not (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $ProductionSha)) {
        $errors += "BACKEND_PRODUCTION_SHA_MISSING:$ProductionSha"
    }
    if (-not (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $WipSha)) {
        $errors += "BACKEND_WIP_SHA_MISSING:$WipSha"
    }
    if ($errors.Count -gt 0) {
        return @{ Ok = $false; Errors = $errors; Product = @(); Excluded = @(); Blockers = @('BACKEND_PUBLISH_GRAPH_INCOMPLETE') }
    }
    $shas = @()
    if ($ProductionSha -ne $WipSha) {
        $shas = @(Get-GitRevList -RepoRoot $RepoRoot -Revisions @("$ProductionSha..$WipSha") -Reverse)
    }
    $product = @()
    $excluded = @()
    $blockers = @()
    $mixed = @()
    foreach ($sha in $shas) {
        $row = Analyze-OriPublishUriCommit -RepoRoot $RepoRoot -Sha $sha -Label 'BACKEND' `
            -WipBranch $WipBranch -InfraBranch $InfraBranch
        if (-not $row.Ok) { $blockers += @($row.blockers) }
        $entry = [ordered]@{
            fullSha        = $row.fullSha
            shortSha       = $row.shortSha
            subject        = $row.subject
            branches       = $row.branches
            classification = $row.classification
            paths          = $row.paths
            securityFlags  = $row.securityFlags
        }
        switch ($row.classification) {
            { $_ -in @('excludedLocalInfraBranch', 'forbiddenLocalOnly', 'emptyCommit') } {
                $excluded += $entry
            }
            'mixedProductAndLocal' {
                $mixed += $entry
                $excluded += $entry
            }
            default {
                if ($row.classification -eq 'productCandidate' -and $row.Ok) {
                    $product += $entry
                }
                else {
                    $excluded += $entry
                }
            }
        }
    }
    return @{
        Ok = ($errors.Count -eq 0)
        Errors = $errors
        Blockers = @($blockers | Select-Object -Unique)
        Product = $product
        Excluded = $excluded
        Mixed = $mixed
        Range = @{ from = $ProductionSha; to = $WipSha; count = $shas.Count }
    }
}

function Get-OriPublishUriFrontendCommits {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$ProductionSha,
        [Parameter(Mandatory = $true)][string]$MainBranch,
        [Parameter(Mandatory = $true)][string]$InfraBranch,
        [Parameter(Mandatory = $true)][string]$InfraSha
    )
    $errors = @()
    Push-Location -LiteralPath $RepoRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        git rev-parse --verify "$ProductionSha^{commit}" 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { $errors += "FRONTEND_PRODUCTION_SHA_MISSING:$ProductionSha" }
        $mainHead = (git rev-parse $MainBranch 2>$null).Trim()
        if ([string]::IsNullOrWhiteSpace($mainHead)) { $errors += "FRONTEND_MAIN_HEAD_MISSING:$MainBranch" }
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
    if ($errors.Count -gt 0) {
        return @{ Ok = $false; Errors = $errors; Product = @(); Excluded = @(); Blockers = @('FRONTEND_PUBLISH_GRAPH_INCOMPLETE') }
    }
    $mainShas = @()
    if ($ProductionSha -ne $mainHead) {
        $mainShas = Get-GitRevList -RepoRoot $RepoRoot -Revisions @("$ProductionSha..$mainHead") -Reverse
    }
    $infraOnlyShas = @()
    if ($InfraSha -and (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $InfraSha)) {
        $infraOnlyShas = Get-GitRevList -RepoRoot $RepoRoot -Revisions @($InfraSha, '--not', $mainHead) -Reverse
    }
    $product = @()
    $excluded = @()
    $blockers = @()
    $mixed = @()
    foreach ($sha in $mainShas) {
        $row = Analyze-OriPublishUriCommit -RepoRoot $RepoRoot -Sha $sha -Label 'FRONTEND' `
            -WipBranch 'ori/builder1-wip' -InfraBranch $InfraBranch
        if (-not $row.Ok) { $blockers += @($row.blockers) }
        $entry = [ordered]@{
            fullSha        = $row.fullSha
            shortSha       = $row.shortSha
            subject        = $row.subject
            branches       = $row.branches
            classification = $row.classification
            paths          = $row.paths
            securityFlags  = $row.securityFlags
        }
        if ($row.classification -eq 'productCandidate' -and $row.Ok) {
            $product += $entry
        }
        elseif ($row.classification -eq 'mixedProductAndLocal') {
            $mixed += $entry
            $excluded += $entry
        }
        else {
            $excluded += $entry
        }
    }
    foreach ($sha in $infraOnlyShas) {
        $sum = Get-GitCommitSummary -RepoRoot $RepoRoot -Sha $sha
        if (-not $sum) { continue }
        $excluded += [ordered]@{
            fullSha        = $sum.FullSha
            shortSha       = $sum.ShortSha
            subject        = $sum.Subject
            branches       = @(Get-GitBranchesContaining -RepoRoot $RepoRoot -Sha $sha)
            classification = 'excludedLocalInfraBranch'
            paths          = @()
            securityFlags  = @()
            note           = 'On local-dev-infra branch; not published via GitHub Pages main.'
        }
    }
    return @{
        Ok = ($errors.Count -eq 0)
        Errors = $errors
        Blockers = @($blockers | Select-Object -Unique)
        Product = $product
        Excluded = $excluded
        Mixed = $mixed
        Range = @{ from = $ProductionSha; to = $mainHead; mainBranch = $MainBranch; count = $mainShas.Count }
    }
}

function Build-OriPublishUriPlan {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout,
        [Parameter(Mandatory = $true)]$SessionMeta,
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$SessionRoot
    )

    $backendRoot = [string]$SessionMeta.paths.backendRoot
    $frontendRoot = [string]$SessionMeta.paths.frontendRoot
    if ($Layout.BackendRoot -ne $backendRoot -or $Layout.FrontendRoot -ne $frontendRoot) {
        throw 'SESSION_PATHS_MISMATCH_LAYOUT'
    }

    $prodBackend = [string]$Registry.production.backendWeb.deployedCommitSha
    $prodWorker = [string]$Registry.production.backendWorker.deployedCommitSha
    $prodFrontend = [string]$Registry.production.frontend.deployedCommitSha
    $infraBackendBranch = [string]$Registry.localDevInfra.backend.branch
    $infraFrontendBranch = [string]$Registry.localDevInfra.frontend.branch
    $wipBranch = [string]$SessionMeta.productWip.builder1Branch
    if ([string]::IsNullOrWhiteSpace($wipBranch)) {
        $wipBranch = [string]$Registry.wip.builder1.branch
    }
    $wipSha = [string]$Registry.wip.builder1.commitSha
    $frontendMainBranch = [string]$Registry.production.frontend.branch
    if ([string]::IsNullOrWhiteSpace($frontendMainBranch)) { $frontendMainBranch = 'main' }
    $backendMainBranch = [string]$Registry.production.backendWeb.branch
    if ([string]::IsNullOrWhiteSpace($backendMainBranch)) { $backendMainBranch = 'main' }

    $cycleState = Get-OriCancelDevCycleState -SessionRoot $SessionRoot -SessionMeta $SessionMeta
    $registryInfraDrift = @(Get-OriInfraRegistryDrift -Layout $Layout -Registry $Registry)
    $registryWipDrift = @(Get-OriWipRegistryDrift -BackendRoot $backendRoot -WipBranch $wipBranch -RegistryWipSha $wipSha)

    $backendGraph = Get-OriPublishUriBackendCommits -RepoRoot $backendRoot -ProductionSha $prodBackend `
        -WipSha $wipSha -WipBranch $wipBranch -InfraBranch $infraBackendBranch
    $frontendGraph = Get-OriPublishUriFrontendCommits -RepoRoot $frontendRoot -ProductionSha $prodFrontend `
        -MainBranch $frontendMainBranch -InfraBranch $infraFrontendBranch `
        -InfraSha ([string]$Registry.localDevInfra.frontend.commitSha)

    $blockers = @()
    $blockers += @($registryInfraDrift)
    $blockers += @($registryWipDrift)
    $blockers += @($backendGraph.Blockers)
    $blockers += @($frontendGraph.Blockers)
    if ($prodBackend -ne $prodWorker) {
        $blockers += 'PRODUCTION_BACKEND_WORKER_SHA_MISMATCH'
    }
    if ([string]$cycleState.status -eq 'CANCELLED') {
        $blockers += 'CYCLE_ALREADY_CANCELLED'
    }
    if ([string]$cycleState.status -ne 'OPEN') {
        $blockers += "PUBLISH_CYCLE_STATUS_BLOCKED:$($cycleState.status)"
    }

    $mixedCommits = @()
    foreach ($m in @($backendGraph.Mixed) + @($frontendGraph.Mixed)) {
        if ($m) { $mixedCommits += $m }
    }

    $proposedBackendHead = $wipSha
    $proposedFrontendHead = $frontendGraph.Range.to
    $hasProductBackend = @($backendGraph.Product).Count -gt 0
    $hasProductFrontend = @($frontendGraph.Product).Count -gt 0
    $hasProduct = $hasProductBackend -or $hasProductFrontend
    if (-not $hasProduct -and $blockers.Count -eq 0) {
        # informational only — not a blocker
    }

    $blockers = @($blockers | Select-Object -Unique)
    $verifiedSafe = ($blockers.Count -eq 0) -and $hasProduct -and (@($mixedCommits).Count -eq 0)
    $canPrepare = (
        [string]$cycleState.status -eq 'OPEN'
    ) -and (
        -not ($blockers -contains 'CYCLE_ALREADY_CANCELLED')
    ) -and (
        @($registryInfraDrift).Count -eq 0
    ) -and (
        @($registryWipDrift).Count -eq 0
    ) -and (
        $backendGraph.Ok -and $frontendGraph.Ok
    )

    $allFilesBackend = @()
    foreach ($c in $backendGraph.Product) {
        foreach ($p in $c.paths) { $allFilesBackend += $p.path }
    }
    $allFilesFrontend = @()
    foreach ($c in $frontendGraph.Product) {
        foreach ($p in $c.paths) { $allFilesFrontend += $p.path }
    }

    return [ordered]@{
        sessionId           = [string]$SessionMeta.sessionId
        sessionRoot         = $SessionRoot
        mode                = 'PUBLISH_URI_PREVIEW'
        production          = [ordered]@{
            backendWeb    = $prodBackend
            backendWorker = $prodWorker
            backendBranch = $backendMainBranch
            frontend      = $prodFrontend
            frontendBranch = $frontendMainBranch
        }
        proposedChanges     = [ordered]@{
            backend = [ordered]@{
                deployTargetBranch = $backendMainBranch
                sourceRef          = $wipBranch
                proposedHeadSha    = $proposedBackendHead
                commits            = @($backendGraph.Product)
                files              = @($allFilesBackend | Select-Object -Unique)
            }
            frontend = [ordered]@{
                deployTargetBranch = $frontendMainBranch
                sourceRef          = $frontendMainBranch
                proposedHeadSha    = $proposedFrontendHead
                commits            = @($frontendGraph.Product)
                files              = @($allFilesFrontend | Select-Object -Unique)
            }
        }
        workerImpact        = [ordered]@{
            platform              = 'render'
            deployWithBackendWeb  = $true
            proposedCommitSha     = if ($hasProductBackend) { $proposedBackendHead } else { $prodBackend }
            note                  = 'Worker deploys from backend main; publish backend product commits updates worker when merged to main.'
        }
        commits             = [ordered]@{
            productCandidate = [ordered]@{
                backend  = @($backendGraph.Product)
                frontend = @($frontendGraph.Product)
            }
            excludedLocal    = [ordered]@{
                backend  = @($backendGraph.Excluded)
                frontend = @($frontendGraph.Excluded)
            }
            mixedOrBlocked   = @($mixedCommits)
        }
        alreadyInProduction = [ordered]@{
            backend  = @($prodBackend)
            frontend = @($prodFrontend)
            note     = 'Registry deployed SHAs; commits at or behind these are live.'
        }
        registryInfraDrift  = @($registryInfraDrift)
        registryWipDrift    = @($registryWipDrift)
        cancelCycle         = [ordered]@{
            status = [string]$cycleState.status
        }
        blockers            = @($blockers)
        mixedCommits        = @($mixedCommits)
        readiness           = [ordered]@{
            hasProductChanges               = $hasProduct
            canPrepareLocalPublish          = [bool]$canPrepare
            verifiedSafeForLiveProduction   = [bool]$verifiedSafe
            executePublishAvailable         = $false
            notes                           = @(
                'Preview only; no push, deploy, or registry production update.'
                'canPrepareLocalPublish requires OPEN cancel cycle and registry HEAD alignment.'
                'verifiedSafeForLiveProduction additionally requires zero blockers, product changes, and no mixed commits.'
            )
        }
        canExecutePublish   = $false
    }
}

function Test-OriPublishUriCandidateBackendSha {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$ProductionSha,
        [Parameter(Mandatory = $true)][string]$CandidateSha
    )
    $errors = @()
    $normalized = ($CandidateSha -replace '\s', '').ToLowerInvariant()
    if ($normalized -notmatch '^[0-9a-f]{40}$') {
        return @{ Ok = $false; Errors = @('CANDIDATE_BACKEND_SHA_INVALID'); NormalizedSha = '' }
    }
    if (-not (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $normalized)) {
        return @{ Ok = $false; Errors = @("CANDIDATE_BACKEND_SHA_MISSING:$normalized"); NormalizedSha = $normalized }
    }
    if (-not (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $ProductionSha)) {
        return @{ Ok = $false; Errors = @("BACKEND_PRODUCTION_SHA_MISSING:$ProductionSha"); NormalizedSha = $normalized }
    }
    Push-Location -LiteralPath $RepoRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        git merge-base --is-ancestor $ProductionSha $normalized 2>$null
        if ($LASTEXITCODE -ne 0) {
            $errors += "CANDIDATE_NOT_DESCENDANT_OF_PRODUCTION:$normalized"
        }
        if ($ProductionSha -eq $normalized) {
            $errors += 'CANDIDATE_EQUALS_PRODUCTION_BASELINE'
        }
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
    return @{
        Ok = ($errors.Count -eq 0)
        Errors = $errors
        NormalizedSha = $normalized
    }
}

function Build-OriPublishUriCandidatePlan {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout,
        [Parameter(Mandatory = $true)]$SessionMeta,
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)][string]$CandidateBackendSha
    )

    $backendRoot = [string]$SessionMeta.paths.backendRoot
    $frontendRoot = [string]$SessionMeta.paths.frontendRoot
    if ($Layout.BackendRoot -ne $backendRoot -or $Layout.FrontendRoot -ne $frontendRoot) {
        throw 'SESSION_PATHS_MISMATCH_LAYOUT'
    }

    $prodBackend = [string]$Registry.production.backendWeb.deployedCommitSha
    $prodWorker = [string]$Registry.production.backendWorker.deployedCommitSha
    $prodFrontend = [string]$Registry.production.frontend.deployedCommitSha
    $infraBackendBranch = [string]$Registry.localDevInfra.backend.branch
    $infraFrontendBranch = [string]$Registry.localDevInfra.frontend.branch
    $wipBranch = [string]$SessionMeta.productWip.builder1Branch
    if ([string]::IsNullOrWhiteSpace($wipBranch)) {
        $wipBranch = [string]$Registry.wip.builder1.branch
    }
    $frontendMainBranch = [string]$Registry.production.frontend.branch
    if ([string]::IsNullOrWhiteSpace($frontendMainBranch)) { $frontendMainBranch = 'main' }
    $backendMainBranch = [string]$Registry.production.backendWeb.branch
    if ([string]::IsNullOrWhiteSpace($backendMainBranch)) { $backendMainBranch = 'main' }

    $candidateCheck = Test-OriPublishUriCandidateBackendSha -RepoRoot $backendRoot `
        -ProductionSha $prodBackend -CandidateSha $CandidateBackendSha
    $candidateSha = [string]$candidateCheck.NormalizedSha

    $cycleState = Get-OriCancelDevCycleState -SessionRoot $SessionRoot -SessionMeta $SessionMeta
    $registryInfraDrift = @(Get-OriInfraRegistryDrift -Layout $Layout -Registry $Registry)
    $registryWipDrift = @(Get-OriWipRegistryDrift -BackendRoot $backendRoot -WipBranch $wipBranch `
        -RegistryWipSha ([string]$Registry.wip.builder1.commitSha))

    $candidateBlockers = @()
    $warnings = @()
    $warnings += @($registryInfraDrift)
    $warnings += @($registryWipDrift)
    if (-not $candidateCheck.Ok) {
        $candidateBlockers += @($candidateCheck.Errors)
    }

    $backendGraph = @{
        Ok = $false
        Product = @()
        Excluded = @()
        Mixed = @()
        Blockers = @()
        Range = @{ from = $prodBackend; to = $candidateSha; count = 0 }
    }
    if ($candidateCheck.Ok) {
        $backendGraph = Get-OriPublishUriBackendCommits -RepoRoot $backendRoot -ProductionSha $prodBackend `
            -WipSha $candidateSha -WipBranch $wipBranch -InfraBranch $infraBackendBranch
        $candidateBlockers += @($backendGraph.Blockers)
    }

    $frontendGraph = Get-OriPublishUriFrontendCommits -RepoRoot $frontendRoot -ProductionSha $prodFrontend `
        -MainBranch $frontendMainBranch -InfraBranch $infraFrontendBranch `
        -InfraSha ([string]$Registry.localDevInfra.frontend.commitSha)

    if ($prodBackend -ne $prodWorker) {
        $candidateBlockers += 'PRODUCTION_BACKEND_WORKER_SHA_MISMATCH'
    }

    if ([string]$cycleState.status -ne 'OPEN') {
        $warnings += "PUBLISH_CYCLE_STATUS:$($cycleState.status)"
    }
    if ([string]$cycleState.status -eq 'CANCELLED') {
        $warnings += 'CYCLE_ALREADY_CANCELLED'
    }

    $mixedCommits = @()
    foreach ($m in @($backendGraph.Mixed)) {
        if ($m) { $mixedCommits += $m }
    }

    $hasProductBackend = @($backendGraph.Product).Count -gt 0
    $hasProductFrontend = @($frontendGraph.Product).Count -gt 0
    $hasProduct = $hasProductBackend -or $hasProductFrontend

    $candidateBlockers = @($candidateBlockers | Select-Object -Unique)
    $warnings = @($warnings | Select-Object -Unique)

    $candidateVerifiedSafe = (
        $candidateCheck.Ok
    ) -and (
        $candidateBlockers.Count -eq 0
    ) -and (
        $hasProductBackend
    ) -and (
        @($mixedCommits).Count -eq 0
    )

    $canPrepare = (
        [string]$cycleState.status -eq 'OPEN'
    ) -and (
        -not ($warnings -contains 'CYCLE_ALREADY_CANCELLED')
    ) -and (
        @($registryInfraDrift).Count -eq 0
    ) -and (
        @($registryWipDrift).Count -eq 0
    ) -and (
        $candidateVerifiedSafe
    ) -and (
        $backendGraph.Ok
    )

    $allFilesBackend = @()
    foreach ($c in $backendGraph.Product) {
        foreach ($p in $c.paths) { $allFilesBackend += $p.path }
    }

    $candidateCommitShas = @()
    foreach ($c in @($backendGraph.Product) + @($backendGraph.Excluded) + @($backendGraph.Mixed)) {
        if ($c.fullSha) { $candidateCommitShas += [string]$c.fullSha }
    }
    $candidateCommitShas = @($candidateCommitShas | Select-Object -Unique)

    return [ordered]@{
        sessionId              = [string]$SessionMeta.sessionId
        sessionRoot            = $SessionRoot
        mode                   = 'PUBLISH_URI_CANDIDATE_PREVIEW'
        candidateBackendSha    = $candidateSha
        production             = [ordered]@{
            backendWeb     = $prodBackend
            backendWorker  = $prodWorker
            backendBranch  = $backendMainBranch
            frontend       = $prodFrontend
            frontendBranch = $frontendMainBranch
        }
        proposedChanges        = [ordered]@{
            backend = [ordered]@{
                deployTargetBranch = $backendMainBranch
                sourceRef          = 'explicit-candidate-sha'
                proposedHeadSha    = $candidateSha
                commits            = @($backendGraph.Product)
                files              = @($allFilesBackend | Select-Object -Unique)
            }
            frontend = [ordered]@{
                deployTargetBranch = $frontendMainBranch
                sourceRef          = $frontendMainBranch
                proposedHeadSha    = $frontendGraph.Range.to
                commits            = @($frontendGraph.Product)
                files              = @()
            }
        }
        workerImpact           = [ordered]@{
            platform             = 'render'
            deployWithBackendWeb = $true
            proposedCommitSha    = if ($hasProductBackend) { $candidateSha } else { $prodBackend }
            note                 = 'Worker deploys from backend main; candidate backend SHA only.'
        }
        commits                = [ordered]@{
            productCandidate = [ordered]@{
                backend  = @($backendGraph.Product)
                frontend = @($frontendGraph.Product)
            }
            excludedLocal    = [ordered]@{
                backend  = @($backendGraph.Excluded)
                frontend = @($frontendGraph.Excluded)
            }
            mixedOrBlocked   = @($mixedCommits)
        }
        candidateCommitShas    = @($candidateCommitShas)
        registryInfraDrift     = @($registryInfraDrift)
        registryWipDrift       = @($registryWipDrift)
        cancelCycle            = [ordered]@{
            status = [string]$cycleState.status
        }
        candidateBlockers      = @($candidateBlockers)
        warnings               = @($warnings)
        blockers               = @($candidateBlockers)
        mixedCommits           = @($mixedCommits)
        readiness              = [ordered]@{
            hasProductChanges                 = [bool]$hasProductBackend
            hasBackendProductChanges          = [bool]$hasProductBackend
            hasFrontendProductChanges         = [bool]$hasProductFrontend
            candidateVerifiedSafeForPublish   = [bool]$candidateVerifiedSafe
            verifiedSafeForLiveProduction     = [bool]$candidateVerifiedSafe
            canPrepareLocalPublish            = [bool]$canPrepare
            candidateDryRunAvailable          = $true
            executePublishAvailable           = $false
            notes                             = @(
                'Candidate preview; production baseline to explicit backend SHA only.'
                'candidateBlockers apply to the publish candidate; warnings include local infra drift.'
                'canPrepareLocalPublish requires OPEN cycle, zero infra/wip drift, and candidateVerifiedSafeForPublish.'
            )
        }
        canExecutePublish      = $false
    }
}

function Invoke-OriPublishUriCandidatePreview {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout,
        [Parameter(Mandatory = $true)][string]$SessionId,
        [Parameter(Mandatory = $true)][string]$CandidateBackendSha,
        [string]$SessionsRoot = ''
    )
    if ([string]::IsNullOrWhiteSpace($SessionsRoot)) {
        $SessionsRoot = $Layout.SessionsRoot
    }
    $sessionRoot = Join-Path $SessionsRoot $SessionId
    $sessionMeta = Get-OriSessionMeta -SessionRoot $sessionRoot
    $regPath = $Layout.RegistryPath
    $regCheck = Test-ProductionRegistryIntegrity -RegistryPath $regPath -Sha256SidecarPath "$regPath.sha256"
    if (-not $regCheck.Ok) {
        throw ($regCheck.Errors -join ',')
    }
    $commitCheck = Test-RegistryCommitsExist -Registry $regCheck.Registry -BackendRoot $Layout.BackendRoot -FrontendRoot $Layout.FrontendRoot
    if (-not $commitCheck.Ok) {
        throw ($commitCheck.Errors -join ',')
    }
    if (-not (Test-GitCommitExistsInRepo -RepoRoot $Layout.BackendRoot -Sha $CandidateBackendSha)) {
        throw "CANDIDATE_BACKEND_SHA_MISSING:$CandidateBackendSha"
    }
    $envCheck = Test-OriVerifiedLocalDevEnvironment -Layout $Layout
    if (-not $envCheck.Ok) {
        throw ($envCheck.Errors -join ',')
    }
    $plan = Build-OriPublishUriCandidatePlan -Layout $Layout -SessionMeta $sessionMeta -Registry $regCheck.Registry `
        -SessionRoot $sessionRoot -CandidateBackendSha $CandidateBackendSha
    return @{
        mode = 'CANDIDATE_PREVIEW'
        plan = $plan
    }
}

function Invoke-OriPublishUriPreview {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout,
        [Parameter(Mandatory = $true)][string]$SessionId,
        [string]$SessionsRoot = ''
    )
    if ([string]::IsNullOrWhiteSpace($SessionsRoot)) {
        $SessionsRoot = $Layout.SessionsRoot
    }
    $sessionRoot = Join-Path $SessionsRoot $SessionId
    $sessionMeta = Get-OriSessionMeta -SessionRoot $sessionRoot
    $regPath = $Layout.RegistryPath
    $regCheck = Test-ProductionRegistryIntegrity -RegistryPath $regPath -Sha256SidecarPath "$regPath.sha256"
    if (-not $regCheck.Ok) {
        throw ($regCheck.Errors -join ',')
    }
    $commitCheck = Test-RegistryCommitsExist -Registry $regCheck.Registry -BackendRoot $Layout.BackendRoot -FrontendRoot $Layout.FrontendRoot
    if (-not $commitCheck.Ok) {
        throw ($commitCheck.Errors -join ',')
    }
    $envCheck = Test-OriVerifiedLocalDevEnvironment -Layout $Layout
    if (-not $envCheck.Ok) {
        throw ($envCheck.Errors -join ',')
    }
    $plan = Build-OriPublishUriPlan -Layout $Layout -SessionMeta $sessionMeta -Registry $regCheck.Registry -SessionRoot $sessionRoot
    return @{
        mode = 'PREVIEW'
        plan = $plan
    }
}
