# Ori cancel-dev engine — shared helpers (dot-source; no automatic side effects).
Set-StrictMode -Version Latest

function Write-CancelInfo([string]$Message) { Write-Host $Message -ForegroundColor Cyan }
function Write-CancelOk([string]$Message) { Write-Host $Message -ForegroundColor Green }
function Write-CancelWarn([string]$Message) { Write-Host $Message -ForegroundColor Yellow }
function Write-CancelErr([string]$Message) { Write-Host $Message -ForegroundColor Red }

function New-OriCancelJournal {
    param(
        [Parameter(Mandatory = $true)][string]$SessionId,
        [Parameter(Mandatory = $true)][string]$Mode
    )
    return @{
        sessionId   = $SessionId
        mode        = $Mode
        startedAt   = (Get-Date).ToString('o')
        finishedAt  = $null
        status      = 'IN_PROGRESS'
        steps       = @()
    }
}

function Add-OriCancelJournalStep {
    param(
        [Parameter(Mandatory = $true)]$Journal,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$Detail = ''
    )
    $Journal.steps += [ordered]@{
        at     = (Get-Date).ToString('o')
        name   = $Name
        status = $Status
        detail = $Detail
    }
}

function Save-OriCancelJournal {
    param(
        [Parameter(Mandatory = $true)]$Journal,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $Journal.finishedAt = (Get-Date).ToString('o')
    ($Journal | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Set-OriCancelExecuteProgressBinding {
    param(
        [string]$ProgressPath = '',
        [string]$ProgressOperationId = ''
    )
    if ([string]::IsNullOrWhiteSpace($ProgressPath) -or [string]::IsNullOrWhiteSpace($ProgressOperationId)) {
        $script:OriCancelExecuteProgressBinding = $null
        return
    }
    $script:OriCancelExecuteProgressBinding = @{
        Path        = $ProgressPath.Trim()
        OperationId = $ProgressOperationId.Trim().ToLowerInvariant()
    }
}

function Get-OriCancelExecuteProgressContext {
    $path = [string]$env:ACE_CANCEL_EXECUTE_PROGRESS_PATH
    $op = [string]$env:ACE_CANCEL_EXECUTE_OPERATION_ID
    if (($path.Trim().Length -eq 0 -or $op.Trim().Length -eq 0) -and $script:OriCancelExecuteProgressBinding) {
        $path = [string]$script:OriCancelExecuteProgressBinding.Path
        $op = [string]$script:OriCancelExecuteProgressBinding.OperationId
    }
    if ([string]::IsNullOrWhiteSpace($path) -or [string]::IsNullOrWhiteSpace($op)) {
        return $null
    }
    return @{
        Path        = $path.Trim()
        OperationId = $op.Trim().ToLowerInvariant()
    }
}

function Protect-OriCancelProgressErrorMessage {
    param([string]$Message)
    $text = [string]$Message
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    $text = $text -replace '[A-Za-z]:\\[^\s,;]+', '[path]'
    $text = $text -replace '\\\\[^\s,;]+', '[path]'
    $text = $text -replace '/[^\s,;]+', '[path]'
    $text = $text -replace '\s+', ' '
    if ($text.Length -gt 240) { $text = $text.Substring(0, 237) + '...' }
    return $text.Trim()
}

function Read-OriCancelExecuteProgressDocument {
    param(
        [Parameter(Mandatory = $true)][string]$ProgressPath,
        [Parameter(Mandatory = $true)][string]$OperationId
    )
    if (-not (Test-Path -LiteralPath $ProgressPath)) {
        throw 'PROGRESS_FILE_MISSING'
    }
    $raw = Get-Content -LiteralPath $ProgressPath -Raw -Encoding UTF8
    $doc = $raw | ConvertFrom-Json
    $docOp = [string]$doc.operationId
    if ($docOp.ToLowerInvariant() -ne $OperationId.ToLowerInvariant()) {
        throw 'PROGRESS_OPERATION_MISMATCH'
    }
    return $doc
}

function Save-OriCancelExecuteProgressDocument {
    param(
        [Parameter(Mandatory = $true)]$Document,
        [Parameter(Mandatory = $true)][string]$ProgressPath
    )
    $dir = Split-Path -Parent $ProgressPath
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $Document.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    $tmp = "$ProgressPath.tmp.$([guid]::NewGuid().ToString('n'))"
    $json = ($Document | ConvertTo-Json -Depth 12)
    [System.IO.File]::WriteAllText($tmp, $json, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $ProgressPath -Force
}

function Start-OriCancelExecuteProgressStep {
    param(
        [Parameter(Mandatory = $true)][string]$StepId
    )
    $ctx = Get-OriCancelExecuteProgressContext
    if (-not $ctx) { return }
    $doc = Read-OriCancelExecuteProgressDocument -ProgressPath $ctx.Path -OperationId $ctx.OperationId
    $now = (Get-Date).ToUniversalTime().ToString('o')
    $doc.currentStep = [ordered]@{
        id        = $StepId
        status    = 'IN_PROGRESS'
        startedAt = $now
    }
    Save-OriCancelExecuteProgressDocument -Document $doc -ProgressPath $ctx.Path
}

function Complete-OriCancelExecuteProgressStep {
    param(
        [Parameter(Mandatory = $true)][string]$StepId,
        [string]$Status = 'SUCCESS'
    )
    $ctx = Get-OriCancelExecuteProgressContext
    if (-not $ctx) { return }
    $doc = Read-OriCancelExecuteProgressDocument -ProgressPath $ctx.Path -OperationId $ctx.OperationId
    $now = (Get-Date).ToUniversalTime().ToString('o')
    $started = $now
    if ($doc.currentStep -and [string]$doc.currentStep.id -eq $StepId) {
        $started = [string]$doc.currentStep.startedAt
    }
    $steps = @()
    if ($doc.completedSteps) { $steps = @($doc.completedSteps) }
    $steps += [ordered]@{
        id         = $StepId
        status     = $Status
        startedAt  = $started
        finishedAt = $now
    }
    $doc.completedSteps = $steps
    $doc.currentStep = $null
    Save-OriCancelExecuteProgressDocument -Document $doc -ProgressPath $ctx.Path
}

function Fail-OriCancelExecuteProgressStep {
    param(
        [Parameter(Mandatory = $true)][string]$StepId,
        [string]$ErrorMessage = ''
    )
    Complete-OriCancelExecuteProgressStep -StepId $StepId -Status 'FAILED'
    $ctx = Get-OriCancelExecuteProgressContext
    if (-not $ctx) { return }
    $doc = Read-OriCancelExecuteProgressDocument -ProgressPath $ctx.Path -OperationId $ctx.OperationId
    $doc.operationStatus = 'FAILED'
    if (-not $doc.cancelStatus) { $doc | Add-Member -NotePropertyName 'cancelStatus' -NotePropertyValue 'FAILED' -Force }
    else { $doc.cancelStatus = 'FAILED' }
    $doc.error = Protect-OriCancelProgressErrorMessage -Message $ErrorMessage
    Save-OriCancelExecuteProgressDocument -Document $doc -ProgressPath $ctx.Path
}

function Set-OriCancelExecuteProgressOperationStatus {
    param(
        [Parameter(Mandatory = $true)][string]$OperationStatus,
        [string]$ErrorMessage = ''
    )
    $ctx = Get-OriCancelExecuteProgressContext
    if (-not $ctx) { return }
    $doc = Read-OriCancelExecuteProgressDocument -ProgressPath $ctx.Path -OperationId $ctx.OperationId
    $doc.operationStatus = $OperationStatus
    $doc.currentStep = $null
    if (-not [string]::IsNullOrWhiteSpace($ErrorMessage)) {
        $doc.error = Protect-OriCancelProgressErrorMessage -Message $ErrorMessage
    }
    Save-OriCancelExecuteProgressDocument -Document $doc -ProgressPath $ctx.Path
}

function Set-OriCancelExecuteProgressCancelStatus {
    param(
        [Parameter(Mandatory = $true)][string]$CancelStatus,
        [string]$ErrorMessage = ''
    )
    $ctx = Get-OriCancelExecuteProgressContext
    if (-not $ctx) { return }
    $doc = Read-OriCancelExecuteProgressDocument -ProgressPath $ctx.Path -OperationId $ctx.OperationId
    if (-not $doc.cancelStatus) { $doc | Add-Member -NotePropertyName 'cancelStatus' -NotePropertyValue $CancelStatus -Force }
    else { $doc.cancelStatus = $CancelStatus }
    if (-not [string]::IsNullOrWhiteSpace($ErrorMessage)) {
        $doc.error = Protect-OriCancelProgressErrorMessage -Message $ErrorMessage
    }
    Save-OriCancelExecuteProgressDocument -Document $doc -ProgressPath $ctx.Path
}

function Get-OriCancelLockPath {
    param([Parameter(Mandatory = $true)][string]$SessionsRoot)
    return Join-Path $SessionsRoot '.ori-cancel-dev.lock'
}

function Enter-OriCancelDevLock {
    param([Parameter(Mandatory = $true)][string]$SessionsRoot)
    $lockPath = Get-OriCancelLockPath -SessionsRoot $SessionsRoot
    if (Test-Path -LiteralPath $lockPath) {
        throw 'CANCEL_DEV_LOCK_HELD'
    }
    $payload = @{
        pid       = $PID
        startedAt = (Get-Date).ToString('o')
        user      = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    } | ConvertTo-Json -Compress
    Set-Content -LiteralPath $lockPath -Value $payload -Encoding UTF8
    return $lockPath
}

function Exit-OriCancelDevLock {
    param([Parameter(Mandatory = $true)][string]$SessionsRoot)
    $lockPath = Get-OriCancelLockPath -SessionsRoot $SessionsRoot
    if (Test-Path -LiteralPath $lockPath) {
        Remove-Item -LiteralPath $lockPath -Force
    }
}

function Test-OriVerifiedLocalDevEnvironment {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout
    )
    $errors = @()
    $backend = $Layout.BackendRoot
    $frontend = $Layout.FrontendRoot
    $aceRoot = $Layout.AceRoot

    foreach ($marker in @('app.py', 'worker_video.py', 'scripts\OriDevSessionBaselineLib.ps1')) {
        if (-not (Test-Path -LiteralPath (Join-Path $backend $marker))) {
            $errors += "BACKEND_MARKER_MISSING:$marker"
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $frontend 'package.json'))) {
        $errors += 'FRONTEND_MARKER_MISSING'
    }
    $registryDir = Join-Path $aceRoot '_registry'
    $backupsDir = Join-Path $aceRoot '_backups'
    if (-not (Test-Path -LiteralPath $registryDir)) { $errors += 'REGISTRY_DIR_MISSING' }
    if (-not (Test-Path -LiteralPath $backupsDir)) { $errors += 'BACKUPS_DIR_MISSING' }

    $examplePath = Join-Path $backend '.env.local.example'
    if (-not (Test-Path -LiteralPath $examplePath)) {
        $errors += 'ENV_EXAMPLE_MISSING'
    }
    else {
        $ex = Get-Content -LiteralPath $examplePath -Raw -Encoding UTF8
        if ($ex -notmatch 'ACE_LOCAL_DEV_ENABLED') {
            $errors += 'ACE_LOCAL_DEV_MARKER_MISSING'
        }
    }

    if ($Layout.RegistryPath -match 'render\.com|github\.com') {
        $errors += 'REMOTE_REGISTRY_PATH_FORBIDDEN'
    }
    return @{ Ok = ($errors.Count -eq 0); Errors = $errors }
}

function Get-OriSessionMeta {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot
    )
    $sessionFile = Join-Path $SessionRoot 'session.json'
    if (-not (Test-Path -LiteralPath $sessionFile)) {
        throw 'SESSION_JSON_MISSING'
    }
    $meta = Get-Content -LiteralPath $sessionFile -Raw | ConvertFrom-Json
    if ([string]$meta.status -ne 'READY') {
        throw "SESSION_NOT_READY:$($meta.status)"
    }
    return $meta
}

function Get-OriBaselineInfraHead {
    param($SessionMeta)
    $infra = $SessionMeta.localDevInfra
    if ($infra.PSObject.Properties['backendShaRegistry']) {
        return [string]$infra.backendShaRegistry
    }
    if ($infra.PSObject.Properties['backendSha']) {
        return [string]$infra.backendSha
    }
    return ''
}

function Get-OriBaselineFrontendHead {
    param($SessionMeta)
    $infra = $SessionMeta.localDevInfra
    if ($infra.PSObject.Properties['frontendShaRegistry']) {
        return [string]$infra.frontendShaRegistry
    }
    if ($infra.PSObject.Properties['frontendSha']) {
        return [string]$infra.frontendSha
    }
    return ''
}

function ConvertTo-StringArray {
    param($Value)
    if ($null -eq $Value) { return @() }
    if ($Value -is [System.Array]) {
        return @($Value | ForEach-Object { [string]$_ })
    }
    return @([string]$Value)
}

function Get-OriCancelDevCycleStatePath {
    param([Parameter(Mandatory = $true)][string]$SessionRoot)
    return Join-Path $SessionRoot 'cancel-dev-cycle.state.json'
}

function Get-OriCancelDevCycleState {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)]$SessionMeta
    )
    $path = Get-OriCancelDevCycleStatePath -SessionRoot $SessionRoot
    if (Test-Path -LiteralPath $path) {
        return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    }
    $state = [ordered]@{
        schemaVersion = 1
        status        = 'OPEN'
        openedAt      = (Get-Date).ToString('o')
        cancelledAt   = $null
        lastSuccessfulBackupRoot = $null
        anchors       = [ordered]@{
            backendProductHeadAtBaseline  = [string]$SessionMeta.git.backend.Head
            frontendProductHeadAtBaseline = [string]$SessionMeta.git.frontend.Head
            backendInfraRegistryAtBaseline  = Get-OriBaselineInfraHead -SessionMeta $SessionMeta
            frontendInfraRegistryAtBaseline = Get-OriBaselineFrontendHead -SessionMeta $SessionMeta
        }
        explicitPreBaselineCommitShas = [ordered]@{
            backend  = @()
            frontend = @()
        }
    }
    ($state | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $path -Encoding UTF8
    return ($state | ConvertTo-Json -Depth 6 | ConvertFrom-Json)
}

function Set-OriCancelDevCycleExplicitPreBaseline {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)]$SessionMeta,
        [string[]]$BackendShas = @(),
        [string[]]$FrontendShas = @()
    )
    $state = Get-OriCancelDevCycleState -SessionRoot $SessionRoot -SessionMeta $SessionMeta
    $state.explicitPreBaselineCommitShas.backend = @($BackendShas)
    $state.explicitPreBaselineCommitShas.frontend = @($FrontendShas)
    ($state | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Get-OriCancelDevCycleStatePath -SessionRoot $SessionRoot) -Encoding UTF8
}

function Set-OriCancelDevCycleCancelled {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)]$SessionMeta,
        [Parameter(Mandatory = $true)][string]$BackupRoot
    )
    $state = Get-OriCancelDevCycleState -SessionRoot $SessionRoot -SessionMeta $SessionMeta
    $state.status = 'CANCELLED'
    $state.cancelledAt = (Get-Date).ToString('o')
    $state.lastSuccessfulBackupRoot = $BackupRoot
    ($state | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Get-OriCancelDevCycleStatePath -SessionRoot $SessionRoot) -Encoding UTF8
}

function Invoke-OriGitLines {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string[]]$GitArgs
    )
    Push-Location -LiteralPath $RepoRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $out = & git @GitArgs 2>$null
        return @{ Code = $LASTEXITCODE; Lines = @($out | ForEach-Object { "$_" }) }
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
}

function Test-GitCommitExistsInRepo {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Sha
    )
    if ([string]::IsNullOrWhiteSpace($Sha)) { return $false }
    return (Invoke-OriGit -RepoRoot $RepoRoot -GitArgs @('rev-parse', '--verify', "${Sha}^{commit}")) -eq 0
}

function Test-GitIsAncestor {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Ancestor,
        [Parameter(Mandatory = $true)][string]$Descendant
    )
    return (Invoke-OriGit -RepoRoot $RepoRoot -GitArgs @('merge-base', '--is-ancestor', $Ancestor, $Descendant)) -eq 0
}

function Get-GitRevList {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string[]]$Revisions,
        [switch]$Reverse
    )
    $args = @('rev-list')
    if ($Reverse) { $args += '--reverse' }
    $args += $Revisions
    $r = Invoke-OriGitLines -RepoRoot $RepoRoot -GitArgs $args
    if ($r.Code -ne 0) { return @() }
    return @($r.Lines | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Get-GitCommitSummary {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Sha
    )
    $r = Invoke-OriGitLines -RepoRoot $RepoRoot -GitArgs @('show', '-s', '--format=%H|%h|%s', $Sha)
    if ($r.Code -ne 0 -or $r.Lines.Count -eq 0) { return $null }
    $parts = $r.Lines[0].Split('|', 3)
    if ($parts.Count -lt 3) { return $null }
    return @{ FullSha = $parts[0]; ShortSha = $parts[1]; Subject = $parts[2] }
}

function Get-GitBranchesContaining {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Sha
    )
    $r = Invoke-OriGitLines -RepoRoot $RepoRoot -GitArgs @('branch', '-a', '--contains', $Sha)
    if ($r.Code -ne 0) { return @() }
    return @($r.Lines | ForEach-Object { $_.Trim().TrimStart('*').Trim() } | Where-Object { $_ })
}

function Get-OriCancelDevCommitGraphForRepo {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$BaselineProductHead,
        [Parameter(Mandatory = $true)][string]$BaselineInfraRegistrySha,
        [Parameter(Mandatory = $true)][string]$InfraTargetSha,
        [Parameter(Mandatory = $true)][string]$InfraBranch,
        [Parameter(Mandatory = $true)][string]$WipBranch,
        [string[]]$ExplicitPreBaselineShas = @()
    )
    $errors = @()
    $blockers = @()

    if ([string]::IsNullOrWhiteSpace($BaselineInfraRegistrySha)) {
        $errors += "${Label}_BASELINE_INFRA_REGISTRY_SHA_MISSING"
    }
    elseif (-not (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $BaselineInfraRegistrySha)) {
        $errors += "${Label}_BASELINE_INFRA_REGISTRY_SHA_MISSING:$BaselineInfraRegistrySha"
    }
    if (-not (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $InfraTargetSha)) {
        $errors += "${Label}_INFRA_TARGET_SHA_MISSING:$InfraTargetSha"
    }
    if (-not (Test-GitCommitExistsInRepo -RepoRoot $RepoRoot -Sha $BaselineProductHead)) {
        $errors += "${Label}_BASELINE_PRODUCT_HEAD_MISSING:$BaselineProductHead"
    }

    $state = Get-GitRepoState -RepoRoot $RepoRoot
    $currentHead = [string]$state.Head
    $currentBranch = [string]$state.Branch

    if ($errors.Count -gt 0) {
        return @{
            Ok = $false
            Errors = $errors
            Blockers = @("${Label}_COMMIT_GRAPH_INCOMPLETE")
            BaselineProductHead = $BaselineProductHead
            CurrentHead = $currentHead
            CurrentBranch = $currentBranch
            InfraTarget = $InfraTargetSha
            SessionWipCommits = @()
            InfraCommits = @()
            PreBaselineCommits = @()
            CommitsToCancel = @()
        }
    }

    if (-not (Test-GitIsAncestor -RepoRoot $RepoRoot -Ancestor $BaselineInfraRegistrySha -Descendant $InfraTargetSha)) {
        $blockers += "${Label}_INFRA_ADVANCEMENT_NOT_ANCESTRAL"
    }

    $infraCommitShas = @()
    if ($BaselineInfraRegistrySha -ne $InfraTargetSha) {
        $infraCommitShas = Get-GitRevList -RepoRoot $RepoRoot -Revisions @("$BaselineInfraRegistrySha..$InfraTargetSha") -Reverse
    }

    $lostFromActive = Get-GitRevList -RepoRoot $RepoRoot -Revisions @($currentHead, '--not', $InfraTargetSha) -Reverse

    $sessionAfterBaseline = @()
    if ($BaselineProductHead -ne $currentHead) {
        $sessionAfterBaseline = Get-GitRevList -RepoRoot $RepoRoot -Revisions @("$BaselineProductHead..$currentHead") -Reverse
    }

    $preBaseline = Get-GitRevList -RepoRoot $RepoRoot -Revisions @($BaselineProductHead, '--not', $InfraTargetSha) -Reverse
    $preBaseline = @($preBaseline | Where-Object { $_ -ne $BaselineProductHead })

    $explicitSet = @{}
    foreach ($s in $ExplicitPreBaselineShas) { if ($s) { $explicitSet[$s] = $true } }

    $sessionSet = @{}
    foreach ($s in $sessionAfterBaseline) { $sessionSet[$s] = $true }

    foreach ($sha in $lostFromActive) {
        if ($infraCommitShas -contains $sha) { continue }
        if ($sessionSet.ContainsKey($sha)) { continue }
        if ($sha -eq $BaselineProductHead) { continue }
        if ($BaselineProductHead -eq $currentHead) {
            if (Test-GitIsAncestor -RepoRoot $RepoRoot -Ancestor $sha -Descendant $BaselineProductHead) { continue }
        }
        if ($preBaseline -contains $sha) { continue }
        $blockers += "${Label}_COMMIT_UNATTRIBUTABLE:$sha"
    }

    $preBaselineOut = @()
    foreach ($sha in $preBaseline) {
        if ($lostFromActive -notcontains $sha) { continue }
        if ($BaselineProductHead -eq $currentHead) { continue }
        $sum = Get-GitCommitSummary -RepoRoot $RepoRoot -Sha $sha
        if (-not $sum) {
            $blockers += "${Label}_COMMIT_UNATTRIBUTABLE:$sha"
            continue
        }
        $preBaselineOut += [ordered]@{
            fullSha            = $sum.FullSha
            shortSha           = $sum.ShortSha
            subject            = $sum.Subject
            branches           = @(Get-GitBranchesContaining -RepoRoot $RepoRoot -Sha $sha)
            classification     = 'preBaseline'
            preservedInWipRef  = $true
            preservedInBackup  = $true
            explicitlyIncluded = [bool]$explicitSet.ContainsKey($sha)
        }
        if (-not $explicitSet.ContainsKey($sha)) {
            $blockers += "${Label}_PRE_BASELINE_COMMIT_NOT_IN_SCOPE:$($sum.ShortSha)"
        }
    }

    $commitsToCancel = @()
    foreach ($sha in $sessionAfterBaseline) {
        if ($lostFromActive -notcontains $sha) { continue }
        $sum = Get-GitCommitSummary -RepoRoot $RepoRoot -Sha $sha
        if (-not $sum) {
            $blockers += "${Label}_COMMIT_UNATTRIBUTABLE:$sha"
            continue
        }
        $commitsToCancel += [ordered]@{
            fullSha           = $sum.FullSha
            shortSha          = $sum.ShortSha
            subject           = $sum.Subject
            branches          = @(Get-GitBranchesContaining -RepoRoot $RepoRoot -Sha $sha)
            classification    = 'sessionWip'
            preservedInWipRef = $true
            preservedInBackup = $true
        }
    }
    $infraOut = @()
    foreach ($sha in $infraCommitShas) {
        $sum = Get-GitCommitSummary -RepoRoot $RepoRoot -Sha $sha
        if (-not $sum) { continue }
        $infraOut += [ordered]@{
            fullSha           = $sum.FullSha
            shortSha          = $sum.ShortSha
            subject           = $sum.Subject
            branches          = @(Get-GitBranchesContaining -RepoRoot $RepoRoot -Sha $sha)
            classification    = 'infra'
            preservedInWipRef = $true
            preservedInBackup = $true
        }
    }

    return @{
        Ok                    = ($blockers.Count -eq 0 -and $errors.Count -eq 0)
        Errors                = $errors
        Blockers              = $blockers
        BaselineProductHead   = $BaselineProductHead
        CurrentHead           = $currentHead
        CurrentBranch         = $currentBranch
        InfraTarget           = $InfraTargetSha
        InfraBranch           = $InfraBranch
        WipBranch             = $WipBranch
        SessionWipCommits     = $commitsToCancel
        InfraCommits          = $infraOut
        PreBaselineCommits    = $preBaselineOut
        CommitsToCancel       = $commitsToCancel
    }
}

function Invoke-OriGit {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string[]]$GitArgs
    )
    Push-Location -LiteralPath $RepoRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        & git @GitArgs 2>$null | Out-Null
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
}

function Get-GitRepoState {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)
    Push-Location -LiteralPath $RepoRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        $branch = (git rev-parse --abbrev-ref HEAD 2>$null).Trim()
        $head = (git rev-parse HEAD 2>$null).Trim()
        $entries = @()
        foreach ($line in (git status -u --porcelain=v1 2>$null)) {
            $p = Parse-GitStatusLine -Line $line
            if ($p) { $entries += $p }
        }
        return @{
            Branch  = $branch
            Head    = $head
            Entries = $entries
        }
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
}

function Test-SessionBaselineArchiveIntegrity {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)]$SessionMeta
    )
    $errors = @()
    if (-not (Test-Path -LiteralPath (Join-Path $SessionRoot 'session.json'))) {
        $errors += 'SESSION_JSON_MISSING'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SessionRoot 'MANIFEST.csv'))) {
        $errors += 'MANIFEST_MISSING'
    }
    foreach ($block in @('BACKEND', 'FRONTEND')) {
        if (-not (Test-Path -LiteralPath (Join-Path $SessionRoot "$block\git\HEAD.txt"))) {
            $errors += "MISSING_${block}_HEAD"
        }
    }
    foreach ($envItem in $SessionMeta.envSnapshots) {
        if (-not $envItem.Present) { continue }
        if (-not (Test-Path -LiteralPath $envItem.DestDpapiPath)) {
            $errors += "ENV_DPAPI_MISSING:$($envItem.LogicalName)"
            continue
        }
        try {
            $bytes = [System.IO.File]::ReadAllBytes($envItem.DestDpapiPath)
            $plain = [System.Text.Encoding]::UTF8.GetString((Unprotect-ContentForCurrentUser -ProtectedBytes $bytes))
            if ((Get-TextSha256Hex $plain) -ne $envItem.PlainSha256) {
                $errors += "ENV_DPAPI_ROUNDTRIP_FAIL:$($envItem.LogicalName)"
            }
        }
        catch {
            $errors += "ENV_DPAPI_UNPROTECT_FAIL:$($envItem.LogicalName)"
        }
    }
    return @{ Ok = ($errors.Count -eq 0); Errors = $errors }
}

function Get-OriInfraRegistryDrift {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout,
        [Parameter(Mandatory = $true)]$Registry
    )
    $drifts = @()
    foreach ($spec in @(
        @{ Repo = 'backend'; Root = $Layout.BackendRoot; Branch = [string]$Registry.localDevInfra.backend.branch; RegSha = [string]$Registry.localDevInfra.backend.commitSha }
        @{ Repo = 'frontend'; Root = $Layout.FrontendRoot; Branch = [string]$Registry.localDevInfra.frontend.branch; RegSha = [string]$Registry.localDevInfra.frontend.commitSha }
    )) {
        Push-Location -LiteralPath $spec.Root
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        try {
            git rev-parse --verify "$($spec.RegSha)^{commit}" 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $drifts += "REGISTRY_INFRA_COMMIT_MISSING:$($spec.Repo):$($spec.RegSha)"
                continue
            }
            $branchHead = (git rev-parse $spec.Branch 2>$null).Trim()
            if ([string]::IsNullOrWhiteSpace($branchHead)) {
                $drifts += "REGISTRY_INFRA_BRANCH_MISSING:$($spec.Repo):$($spec.Branch)"
                continue
            }
            if ($branchHead -ne $spec.RegSha) {
                $drifts += "REGISTRY_INFRA_HEAD_MISMATCH:$($spec.Repo)|branchHead=$branchHead|registry=$($spec.RegSha)"
            }
        }
        finally {
            $ErrorActionPreference = $prevEap
            Pop-Location
        }
    }
    return $drifts
}

function Get-GitInfraAdvancementPaths {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [string]$FromSha = '',
        [Parameter(Mandatory = $true)][string]$ToSha
    )
    if ([string]::IsNullOrWhiteSpace($FromSha) -or $FromSha -eq $ToSha) {
        return @()
    }
    Push-Location -LiteralPath $RepoRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        git merge-base --is-ancestor $FromSha $ToSha 2>$null
        if ($LASTEXITCODE -ne 0) {
            return @()
        }
        $lines = @(git diff --name-only "$FromSha..$ToSha" 2>$null)
        return @($lines | ForEach-Object { $_ -replace '/', '\' } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
    }
}

function Compare-GitWorkToSessionBaseline {
    param(
        [Parameter(Mandatory = $true)]$BaselineGitSnap,
        [Parameter(Mandatory = $true)]$CurrentState,
        [AllowEmptyCollection()][string[]]$InfraAdvancementPaths = @()
    )
    $baselinePaths = @{}
    foreach ($e in $BaselineGitSnap.Entries) {
        $baselinePaths[$e.RelPath] = $e
    }
    $currentPaths = @{}
    foreach ($e in $CurrentState.Entries) {
        $currentPaths[$e.RelPath] = $e
    }
    $infraSet = @{}
    foreach ($p in $InfraAdvancementPaths) { $infraSet[$p] = $true }

    $toRevert = @($BaselineGitSnap.Entries | ForEach-Object { $_.RelPath })
    $unrelated = @()
    $preservedInfraPaths = @()
    foreach ($rel in ($currentPaths.Keys | Sort-Object)) {
        if ($baselinePaths.ContainsKey($rel)) { continue }
        if ($infraSet.ContainsKey($rel)) {
            $preservedInfraPaths += $rel
            continue
        }
        $unrelated += $rel
    }
    return @{
        BaselineDirtyCount   = @($BaselineGitSnap.Entries).Count
        CurrentDirtyCount    = @($CurrentState.Entries).Count
        PathsToRevert        = $toRevert
        UnrelatedPaths       = $unrelated
        PreservedInfraPaths  = $preservedInfraPaths
        BaselineBranch       = [string]$BaselineGitSnap.Branch
        BaselineHead         = [string]$BaselineGitSnap.Head
        CurrentBranch        = [string]$CurrentState.Branch
        CurrentHead          = [string]$CurrentState.Head
    }
}

function Export-OriCancelDevRepoBackup {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$BackupRoot
    )
    $dest = Join-Path $BackupRoot $Label
    $gitDir = Join-Path $dest 'git'
    $mirrorDir = Join-Path $dest 'mirror'
    $deletedDir = Join-Path $dest 'deleted-from-HEAD'
    New-Item -ItemType Directory -Path $gitDir, $mirrorDir, $deletedDir -Force | Out-Null

    Push-Location -LiteralPath $RepoRoot
    try {
        $branch = (git rev-parse --abbrev-ref HEAD).Trim()
        $head = (git rev-parse HEAD).Trim()
        Set-Content -LiteralPath (Join-Path $gitDir 'branch.txt') -Value $branch -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'HEAD.txt') -Value $head -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'status-porcelain.txt') -Value (git status -u --porcelain=v1) -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'diff-HEAD.patch') -Value (git diff HEAD | Out-String) -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'diff-staged.patch') -Value (git diff --cached | Out-String) -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'refs-heads.txt') -Value (git for-each-ref --format='%(refname) %(objectname)' refs/heads) -Encoding UTF8

        $entries = @()
        foreach ($line in (git status -u --porcelain=v1)) {
            $p = Parse-GitStatusLine -Line $line
            if ($p) { $entries += $p }
        }
        foreach ($e in $entries) {
            $full = Join-Path $RepoRoot $e.RelPath
            if ($e.ChangeType -eq 'deleted') {
                $destDeleted = Join-Path $deletedDir $e.RelPath
                New-Item -ItemType Directory -Path (Split-Path -Parent $destDeleted) -Force | Out-Null
                $blob = git show "HEAD:$($e.RelPath -replace '\\','/')"
                Set-Content -LiteralPath $destDeleted -Value $blob -Encoding UTF8
            }
            elseif (Test-Path -LiteralPath $full -PathType Leaf) {
                $destMirror = Join-Path $mirrorDir $e.RelPath
                New-Item -ItemType Directory -Path (Split-Path -Parent $destMirror) -Force | Out-Null
                Copy-Item -LiteralPath $full -Destination $destMirror -Force
            }
        }

        $bundlePath = Join-Path $dest 'all-refs.bundle'
        $bundleRc = 1
        if (Get-Command git -ErrorAction SilentlyContinue) {
            git bundle create $bundlePath --all 2>$null | Out-Null
            $bundleRc = $LASTEXITCODE
        }
        if ($bundleRc -ne 0 -or -not (Test-Path -LiteralPath $bundlePath)) {
            throw "BACKUP_BUNDLE_FAILED:$Label"
        }
        return @{
            Label      = $Label
            Branch     = $branch
            Head       = $head
            Entries    = $entries
            BundlePath = $bundlePath
            DestRoot   = $dest
        }
    }
    finally {
        Pop-Location
    }
}

function Export-OriCancelDevEnvBackup {
    param(
        [Parameter(Mandatory = $true)][string]$BackupRoot,
        [Parameter(Mandatory = $true)][hashtable]$Layout
    )
    $envDir = Join-Path $BackupRoot 'ENV'
    New-Item -ItemType Directory -Path $envDir -Force | Out-Null
    $items = @(
        @{ Name = 'backend.env.local'; Path = Join-Path $Layout.BackendRoot '.env.local' }
        @{ Name = 'frontend.env.development.local'; Path = Join-Path $Layout.FrontendRoot '.env.development.local' }
    )
    $meta = @()
    foreach ($item in $items) {
        if (-not (Test-Path -LiteralPath $item.Path)) {
            $meta += [PSCustomObject]@{ logicalName = $item.Name; present = $false; plainSha256 = ''; dpapiSha256 = '' }
            continue
        }
        $raw = [System.IO.File]::ReadAllText($item.Path, [System.Text.Encoding]::UTF8)
        $plainSha = Get-TextSha256Hex $raw
        $dpapiPath = Join-Path $envDir ($item.Name + '.dpapi')
        $protected = Protect-ContentForCurrentUser -PlainBytes ([System.Text.Encoding]::UTF8.GetBytes($raw))
        [System.IO.File]::WriteAllBytes($dpapiPath, $protected)
        $meta += [PSCustomObject]@{
            logicalName = $item.Name
            present     = $true
            plainSha256 = $plainSha
            dpapiSha256 = (Get-FileSha256Hex $dpapiPath)
        }
    }
    $meta | Export-Csv -LiteralPath (Join-Path $envDir 'env-backup-index.csv') -NoTypeInformation -Encoding UTF8
    return $meta
}

function New-OriCancelDevBackupManifest {
    param([Parameter(Mandatory = $true)][string]$BackupRoot)
    $rows = @()
    Get-ChildItem -LiteralPath $BackupRoot -Recurse -File | ForEach-Object {
        $rel = $_.FullName.Substring($BackupRoot.Length).TrimStart('\')
        $rows += [PSCustomObject]@{
            RelPath = $rel
            Sha256  = (Get-FileSha256Hex $_.FullName)
            SizeBytes = $_.Length
        }
    }
    $rows | Export-Csv -LiteralPath (Join-Path $BackupRoot 'MANIFEST.csv') -NoTypeInformation -Encoding UTF8
    return $rows
}

function Test-OriCancelDevBackupIntegrity {
    param(
        [Parameter(Mandatory = $true)][string]$BackupRoot,
        [switch]$SimulateCorrupt
    )
    $errors = @()
    if (-not (Test-Path -LiteralPath (Join-Path $BackupRoot 'MANIFEST.csv'))) {
        $errors += 'BACKUP_MANIFEST_MISSING'
        return @{ Ok = $false; Errors = $errors }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $BackupRoot 'PRE_STATE.json'))) {
        $errors += 'BACKUP_PRE_STATE_MISSING'
    }
    foreach ($label in @('BACKEND', 'FRONTEND')) {
        $bundle = Join-Path $BackupRoot "$label\all-refs.bundle"
        if (-not (Test-Path -LiteralPath $bundle)) {
            $errors += "BACKUP_BUNDLE_MISSING:$label"
            continue
        }
        $labelRoot = Join-Path $BackupRoot $label
        Push-Location -LiteralPath $labelRoot
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        git bundle list-heads 'all-refs.bundle' 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            git bundle verify 'all-refs.bundle' 2>$null | Out-Null
        }
        if ($LASTEXITCODE -ne 0) { $errors += "BACKUP_BUNDLE_INVALID:$label" }
        $ErrorActionPreference = $prevEap
        Pop-Location
        foreach ($req in @('git\branch.txt', 'git\HEAD.txt', 'git\status-porcelain.txt', 'git\diff-HEAD.patch', 'git\diff-staged.patch')) {
            if (-not (Test-Path -LiteralPath (Join-Path $BackupRoot "$label\$req"))) {
                $errors += "BACKUP_GIT_META_MISSING:${label}:$req"
            }
        }
    }
    $manifest = Get-Content -LiteralPath (Join-Path $BackupRoot 'MANIFEST.csv') | ConvertFrom-Csv
    foreach ($row in $manifest) {
        $full = Join-Path $BackupRoot $row.RelPath
        if (-not (Test-Path -LiteralPath $full)) {
            $errors += "BACKUP_FILE_MISSING:$($row.RelPath)"
            continue
        }
        $hash = Get-FileSha256Hex $full
        if ($SimulateCorrupt) { $hash = '0' * 64 }
        if ($hash -ne $row.Sha256) {
            $errors += "BACKUP_HASH_MISMATCH:$($row.RelPath)"
        }
    }
    $envIndex = Join-Path $BackupRoot 'ENV\env-backup-index.csv'
    if (Test-Path -LiteralPath $envIndex) {
        foreach ($envRow in (Get-Content -LiteralPath $envIndex | ConvertFrom-Csv)) {
            if ($envRow.present -ne 'True') { continue }
            $dpapi = Join-Path $BackupRoot "ENV\$($envRow.logicalName).dpapi"
            if (-not (Test-Path -LiteralPath $dpapi)) {
                $errors += "BACKUP_ENV_DPAPI_MISSING:$($envRow.logicalName)"
                continue
            }
            try {
                $bytes = [System.IO.File]::ReadAllBytes($dpapi)
                $plain = [System.Text.Encoding]::UTF8.GetString((Unprotect-ContentForCurrentUser -ProtectedBytes $bytes))
                if ((Get-TextSha256Hex $plain) -ne $envRow.plainSha256) {
                    $errors += "BACKUP_ENV_ROUNDTRIP_FAIL:$($envRow.logicalName)"
                }
            }
            catch {
                $errors += "BACKUP_ENV_UNPROTECT_FAIL:$($envRow.logicalName)"
            }
        }
    }
    return @{ Ok = ($errors.Count -eq 0); Errors = $errors }
}

function Write-OriCancelDevRecoveryGuide {
    param(
        [Parameter(Mandatory = $true)][string]$BackupRoot,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$FailureDetail = ''
    )
    $lines = @(
        '# Ori cancel-dev recovery (manual only)'
        ''
        "Status: $Status"
        "BackupRoot: $BackupRoot"
        ''
        'Do NOT run automated reset from this folder unless you verified the backup manifest.'
        ''
        '## Restore git (per repo BACKEND / FRONTEND)'
        '1. Record current branch/HEAD (already in PRE_STATE.json and git/HEAD.txt).'
        '2. To restore working tree files: copy from mirror/ and deleted-from-HEAD/ as needed.'
        '3. Apply git/diff-HEAD.patch and git/diff-staged.patch manually if appropriate.'
        '4. Optional: git clone from all-refs.bundle into a scratch dir to inspect refs.'
        ''
        '## Restore ENV'
        '1. Decrypt ENV/*.dpapi on this machine (same Windows user).'
        '2. Copy to .env.local / .env.development.local in the repos.'
        ''
        '## Partial cancel'
        'If BACKEND was reset but FRONTEND was not, restore FRONTEND from this backup before retrying.'
        ''
        "Failure detail: $FailureDetail"
    )
    Set-Content -LiteralPath (Join-Path $BackupRoot 'RECOVERY.md') -Value ($lines -join "`n") -Encoding UTF8
}

function Restore-OriCancelDevWorkingTreeFromBackup {
    param(
        [Parameter(Mandatory = $true)][string]$BackupRoot,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )
    $src = Join-Path $BackupRoot $Label
    $gitDir = Join-Path $src 'git'
    $branch = (Get-Content -LiteralPath (Join-Path $gitDir 'branch.txt') -Raw).Trim()
    $head = (Get-Content -LiteralPath (Join-Path $gitDir 'HEAD.txt') -Raw).Trim()
    if ((Invoke-OriGit -RepoRoot $RepoRoot -GitArgs @('checkout', '-f', $branch)) -ne 0) {
        throw "RESTORE_CHECKOUT_FAILED:$Label"
    }
    if ((Invoke-OriGit -RepoRoot $RepoRoot -GitArgs @('reset', '--hard', $head)) -ne 0) {
        throw "RESTORE_RESET_FAILED:$Label"
    }
    $mirror = Join-Path $src 'mirror'
    if (Test-Path -LiteralPath $mirror) {
        Get-ChildItem -LiteralPath $mirror -Recurse -File | ForEach-Object {
            $rel = $_.FullName.Substring($mirror.Length).TrimStart('\')
            $dest = Join-Path $RepoRoot $rel
            New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
        }
    }
    return @{ Branch = $branch; Head = $head }
}

function Read-EnvDpapiPlain {
    param([Parameter(Mandatory = $true)][string]$DpapiPath)
    if (-not (Test-Path -LiteralPath $DpapiPath)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($DpapiPath)
    return [System.Text.Encoding]::UTF8.GetString((Unprotect-ContentForCurrentUser -ProtectedBytes $bytes))
}

function Get-EnvKeyManifestMap {
    param($EnvSnapshotItem)
    $map = @{}
    if (-not $EnvSnapshotItem -or -not $EnvSnapshotItem.Present) { return $map }
    $rows = $EnvSnapshotItem.KeyManifest
    if ($null -eq $rows) { return $map }
    foreach ($row in @($rows)) {
        if ($row -and $row.key) { $map[[string]$row.key] = [string]$row.classification }
    }
    return $map
}

function Compare-EnvToSessionBaseline {
    param(
        [Parameter(Mandatory = $true)]$EnvSnapshotItem,
        [Parameter(Mandatory = $true)][string]$CurrentEnvPath,
        [Parameter(Mandatory = $true)][string]$ExamplePath
    )
    $result = @{
        LogicalName       = [string]$EnvSnapshotItem.LogicalName
        PresentAtBaseline = [bool]$EnvSnapshotItem.Present
        AddedKeys         = @()
        RemovedKeys       = @()
        ChangedKeys       = @()
        PreservedKeys     = @()
        AmbiguousKeys     = @()
        UnattributedKeys  = @()
        CanApply          = $true
    }
    if (-not $EnvSnapshotItem.Present) {
        if (Test-Path -LiteralPath $CurrentEnvPath) {
            $result.UnattributedKeys += '(entire file present but missing at baseline)'
            $result.CanApply = $false
        }
        return $result
    }

    $baselinePlain = Read-EnvDpapiPlain -DpapiPath ([string]$EnvSnapshotItem.DestDpapiPath)
    $baselineKeys = Parse-DotEnvKeys -Content $baselinePlain
    $classMap = Get-EnvKeyManifestMap -EnvSnapshotItem $EnvSnapshotItem
    $exampleKeys = @{}
    if (Test-Path -LiteralPath $ExamplePath) {
        $exampleKeys = Parse-DotEnvKeys -Content (Get-Content -LiteralPath $ExamplePath -Raw -Encoding UTF8)
    }
    $currentKeys = @{}
    if (Test-Path -LiteralPath $CurrentEnvPath) {
        $currentKeys = Parse-DotEnvKeys -Content (Get-Content -LiteralPath $CurrentEnvPath -Raw -Encoding UTF8)
    }

    foreach ($name in ($baselineKeys.Keys | Sort-Object)) {
        $class = if ($classMap.Contains($name)) { $classMap[$name] } else { 'sessionOrCustom' }
        $bHash = Get-TextSha256Hex ([string]$baselineKeys[$name])
        if ($currentKeys.Keys -notcontains $name) {
            $result.RemovedKeys += $name
            continue
        }
        $cHash = Get-TextSha256Hex ([string]$currentKeys[$name])
        if ($cHash -eq $bHash) {
            if ($class -eq 'infraTemplate') { $result.PreservedKeys += $name }
            continue
        }
        if ($class -eq 'infraTemplate') {
            $exHash = if ($exampleKeys.Keys -contains $name) { Get-TextSha256Hex ([string]$exampleKeys[$name]) } else { '' }
            if ($exHash -and $cHash -eq $exHash) {
                $result.PreservedKeys += $name
            }
            else {
                $result.AmbiguousKeys += $name
                $result.CanApply = $false
            }
        }
        else {
            $result.ChangedKeys += $name
        }
    }
    foreach ($name in ($currentKeys.Keys | Sort-Object)) {
        if ($baselineKeys.Keys -notcontains $name) {
            $result.AddedKeys += $name
        }
    }
    return $result
}

function Format-DotEnvFromKeys {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Keys,
        [Parameter(Mandatory = $true)][string]$OrderSourcePlain
    )
    $order = @()
    foreach ($line in ($OrderSourcePlain -split "`n")) {
        $t = $line.Trim()
        if ($t -match '^([^#=]+)=') {
            $k = $Matches[1].Trim()
            if (($Keys.Keys -contains $k) -and ($order -notcontains $k)) { $order += $k }
        }
    }
    foreach ($k in ($Keys.Keys | Sort-Object)) {
        if ($order -notcontains $k) { $order += $k }
    }
    $lines = @()
    foreach ($k in $order) {
        $v = [string]$Keys[$k]
        if ($v -match '[#\s]') { $lines += "$k=`"$v`"" } else { $lines += "$k=$v" }
    }
    return ($lines -join "`n") + "`n"
}

function Build-EnvRestoreKeys {
    param(
        [Parameter(Mandatory = $true)]$EnvSnapshotItem,
        [Parameter(Mandatory = $true)][string]$CurrentEnvPath,
        [Parameter(Mandatory = $true)][string]$ExamplePath,
        [Parameter(Mandatory = $true)]$Analysis
    )
    $baselinePlain = Read-EnvDpapiPlain -DpapiPath ([string]$EnvSnapshotItem.DestDpapiPath)
    $baselineKeys = Parse-DotEnvKeys -Content $baselinePlain
    $currentKeys = @{}
    if (Test-Path -LiteralPath $CurrentEnvPath) {
        $currentKeys = Parse-DotEnvKeys -Content (Get-Content -LiteralPath $CurrentEnvPath -Raw -Encoding UTF8)
    }
    $classMap = Get-EnvKeyManifestMap -EnvSnapshotItem $EnvSnapshotItem
    $out = [ordered]@{}
    foreach ($name in ($baselineKeys.Keys | Sort-Object)) {
        $class = if ($classMap.Contains($name)) { $classMap[$name] } else { 'sessionOrCustom' }
        if ($class -eq 'infraTemplate' -and ($Analysis.PreservedKeys -contains $name)) {
            $out[$name] = [string]$currentKeys[$name]
        }
        else {
            $out[$name] = [string]$baselineKeys[$name]
        }
    }
    foreach ($name in ($Analysis.PreservedKeys | Sort-Object)) {
        if ($out.Keys -notcontains $name -and ($currentKeys.Keys -contains $name)) {
            $out[$name] = [string]$currentKeys[$name]
        }
    }
    return @{
        EnvKeys    = $out
        OrderPlain = $baselinePlain
    }
}

function Build-OriCancelDevPlan {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout,
        [Parameter(Mandatory = $true)]$SessionMeta,
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$SessionRoot
    )

    $integrity = Test-SessionBaselineArchiveIntegrity -SessionRoot $SessionRoot -SessionMeta $SessionMeta
    if (-not $integrity.Ok) {
        throw ($integrity.Errors -join ',')
    }

    $backendRoot = [string]$SessionMeta.paths.backendRoot
    $frontendRoot = [string]$SessionMeta.paths.frontendRoot
    if ($Layout.BackendRoot -ne $backendRoot -or $Layout.FrontendRoot -ne $frontendRoot) {
        throw 'SESSION_PATHS_MISMATCH_LAYOUT'
    }

    $infraBackendSha = [string]$Registry.localDevInfra.backend.commitSha
    $infraFrontendSha = [string]$Registry.localDevInfra.frontend.commitSha
    $infraBackendBranch = [string]$Registry.localDevInfra.backend.branch
    $infraFrontendBranch = [string]$Registry.localDevInfra.frontend.branch
    $wipBranch = [string]$SessionMeta.productWip.builder1Branch
    $wipSha = [string]$SessionMeta.productWip.builder1Sha

    $baselineBackendInfraSha = Get-OriBaselineInfraHead -SessionMeta $SessionMeta
    $baselineFrontendInfraSha = Get-OriBaselineFrontendHead -SessionMeta $SessionMeta
    $baselineBackendProductHead = [string]$SessionMeta.git.backend.Head
    $baselineFrontendProductHead = [string]$SessionMeta.git.frontend.Head
    $backendInfraPaths = Get-GitInfraAdvancementPaths -RepoRoot $backendRoot -FromSha $baselineBackendInfraSha -ToSha $infraBackendSha
    $frontendInfraPaths = Get-GitInfraAdvancementPaths -RepoRoot $frontendRoot -FromSha $baselineFrontendInfraSha -ToSha $infraFrontendSha
    $cycleState = Get-OriCancelDevCycleState -SessionRoot $SessionRoot -SessionMeta $SessionMeta
    $explicitPreBackend = @()
    $explicitPreFrontend = @()
    if ($cycleState.explicitPreBaselineCommitShas) {
        $explicitPreBackend = ConvertTo-StringArray -Value $cycleState.explicitPreBaselineCommitShas.backend
        $explicitPreFrontend = ConvertTo-StringArray -Value $cycleState.explicitPreBaselineCommitShas.frontend
    }
    $commitGraphBackend = Get-OriCancelDevCommitGraphForRepo -RepoRoot $backendRoot -Label 'BACKEND' `
        -BaselineProductHead $baselineBackendProductHead -BaselineInfraRegistrySha $baselineBackendInfraSha `
        -InfraTargetSha $infraBackendSha -InfraBranch $infraBackendBranch -WipBranch $wipBranch `
        -ExplicitPreBaselineShas $explicitPreBackend
    $commitGraphFrontend = Get-OriCancelDevCommitGraphForRepo -RepoRoot $frontendRoot -Label 'FRONTEND' `
        -BaselineProductHead $baselineFrontendProductHead -BaselineInfraRegistrySha $baselineFrontendInfraSha `
        -InfraTargetSha $infraFrontendSha -InfraBranch $infraFrontendBranch -WipBranch $wipBranch `
        -ExplicitPreBaselineShas $explicitPreFrontend
    $registryDrift = Get-OriInfraRegistryDrift -Layout $Layout -Registry $Registry

    $backendState = Get-GitRepoState -RepoRoot $backendRoot
    $frontendState = Get-GitRepoState -RepoRoot $frontendRoot
    $backendGitCmp = Compare-GitWorkToSessionBaseline -BaselineGitSnap $SessionMeta.git.backend -CurrentState $backendState -InfraAdvancementPaths $backendInfraPaths
    $frontendGitCmp = Compare-GitWorkToSessionBaseline -BaselineGitSnap $SessionMeta.git.frontend -CurrentState $frontendState -InfraAdvancementPaths $frontendInfraPaths

    $backendEnvPath = Join-Path $backendRoot '.env.local'
    $frontendEnvPath = Join-Path $frontendRoot '.env.development.local'
    $backendEnvSnap = $SessionMeta.envSnapshots | Where-Object { $_.LogicalName -eq 'backend.env.local' } | Select-Object -First 1
    $frontendEnvSnap = $SessionMeta.envSnapshots | Where-Object { $_.LogicalName -eq 'frontend.env.development.local' } | Select-Object -First 1
    $backendEnvAnalysis = Compare-EnvToSessionBaseline -EnvSnapshotItem $backendEnvSnap `
        -CurrentEnvPath $backendEnvPath -ExamplePath (Join-Path $backendRoot '.env.local.example')
    $frontendEnvAnalysis = Compare-EnvToSessionBaseline -EnvSnapshotItem $frontendEnvSnap `
        -CurrentEnvPath $frontendEnvPath -ExamplePath (Join-Path $frontendRoot '.env.example')

    $blockers = @()
    if (@($backendGitCmp.UnrelatedPaths).Count -gt 0) {
        $blockers += @($backendGitCmp.UnrelatedPaths | ForEach-Object { "BACKEND_UNRELATED:$_" })
    }
    if (@($frontendGitCmp.UnrelatedPaths).Count -gt 0) {
        $blockers += @($frontendGitCmp.UnrelatedPaths | ForEach-Object { "FRONTEND_UNRELATED:$_" })
    }
    if (-not $backendEnvAnalysis.CanApply) { $blockers += 'BACKEND_ENV_AMBIGUOUS' }
    if (-not $frontendEnvAnalysis.CanApply) { $blockers += 'FRONTEND_ENV_AMBIGUOUS' }
    if (@($registryDrift).Count -gt 0) {
        $blockers += @($registryDrift)
    }
    if ([string]$cycleState.status -eq 'CANCELLED') {
        $blockers += 'CYCLE_ALREADY_CANCELLED'
    }
    if (-not $commitGraphBackend.Ok) {
        $blockers += @($commitGraphBackend.Blockers)
        $blockers += @($commitGraphBackend.Errors)
    }
    else {
        $blockers += @($commitGraphBackend.Blockers)
    }
    if (-not $commitGraphFrontend.Ok) {
        $blockers += @($commitGraphFrontend.Blockers)
        $blockers += @($commitGraphFrontend.Errors)
    }
    else {
        $blockers += @($commitGraphFrontend.Blockers)
    }
    if ([string]::IsNullOrWhiteSpace($baselineBackendInfraSha) -or [string]::IsNullOrWhiteSpace($baselineFrontendInfraSha)) {
        $blockers += 'BASELINE_INFRA_REGISTRY_SNAPSHOT_INCOMPLETE'
    }

    return [ordered]@{
        sessionId              = [string]$SessionMeta.sessionId
        sessionRoot            = $SessionRoot
        production             = @{
            backendWeb    = [string]$Registry.production.backendWeb.deployedCommitSha
            backendWorker = [string]$Registry.production.backendWorker.deployedCommitSha
            frontend      = [string]$Registry.production.frontend.deployedCommitSha
        }
        infraTargets           = @{
            backendBranch  = $infraBackendBranch
            backendCommit  = $infraBackendSha
            frontendBranch = $infraFrontendBranch
            frontendCommit = $infraFrontendSha
            note           = 'Targets from live registry; not baseline HEAD.'
        }
        baselineInfraHead      = @{
            backend  = $baselineBackendInfraSha
            frontend = $baselineFrontendInfraSha
            note     = 'From baseline registry snapshot (backendShaRegistry), not WIP HEAD at snapshot.'
        }
        baselineProductHead    = @{
            backend  = $baselineBackendProductHead
            frontend = $baselineFrontendProductHead
        }
        cancelCycle            = @{
            status                 = [string]$cycleState.status
            cancelledAt            = [string]$cycleState.cancelledAt
            lastSuccessfulBackupRoot = [string]$cycleState.lastSuccessfulBackupRoot
            anchors                = $cycleState.anchors
        }
        commitGraph            = @{
            backend  = $commitGraphBackend
            frontend = $commitGraphFrontend
        }
        commitsToCancel        = @{
            backend  = @($commitGraphBackend.CommitsToCancel)
            frontend = @($commitGraphFrontend.CommitsToCancel)
        }
        infraCommitsPreserved  = @{
            backend  = @($commitGraphBackend.InfraCommits)
            frontend = @($commitGraphFrontend.InfraCommits)
        }
        registryInfraDrift     = $registryDrift
        infraAdvancementPaths  = @{
            backend  = $backendInfraPaths
            frontend = $frontendInfraPaths
        }
        productWip             = @{
            branch = $wipBranch
            sha    = $wipSha
            action = 'WIP refs retained; full pre-cancel backup includes all-refs.bundle per repo.'
        }
        git                    = @{
            backend  = $backendGitCmp
            frontend = $frontendGitCmp
        }
        env                    = @{
            backend  = $backendEnvAnalysis
            frontend = $frontendEnvAnalysis
        }
        preservedInfra         = @{
            items = @(
                'Local dev scripts and branches (ori/local-dev-infrastructure)'
                'Files introduced on infra branch between baseline HEAD and registry infra commit'
                'Production SHAs in registry (read-only verification)'
                'WIP branch refs (not deleted)'
                'infraTemplate ENV keys matching example or unchanged'
                'Redis data (not touched)'
                'main / origin/main (not checked out or modified)'
            )
            paths = @{
                backend  = $backendGitCmp.PreservedInfraPaths
                frontend = $frontendGitCmp.PreservedInfraPaths
            }
        }
        willCancel             = @{
            git = @(
                'Verified full backup of both repos + ENV (manifest + bundle) before any reset'
                "Checkout $infraBackendBranch @ $infraBackendSha (backend) only if registry matches branch HEAD"
                "Checkout $infraFrontendBranch @ $infraFrontendSha (frontend) only if registry matches branch HEAD"
                'Revert working-tree paths captured in baseline snapshot (session-attributed only)'
                'Active branch reset drops session WIP commits listed in commitsToCancel (WIP refs and bundle retained)'
            )
            env = @(
                'Restore sessionOrCustom keys from baseline DPAPI'
                'Remove session-added keys'
                'Re-add baseline keys removed during session'
            )
        }
        blockers               = $blockers
        canExecute             = ($blockers.Count -eq 0)
    }
}

function Invoke-OriCancelDevPreview {
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

    $journal = New-OriCancelJournal -SessionId $SessionId -Mode 'PREVIEW'
    Add-OriCancelJournalStep -Journal $journal -Name 'load_session' -Status 'SUCCESS'
    Add-OriCancelJournalStep -Journal $journal -Name 'verify_registry' -Status 'SUCCESS'
    Add-OriCancelJournalStep -Journal $journal -Name 'verify_local_dev_layout' -Status 'SUCCESS'

    $plan = Build-OriCancelDevPlan -Layout $Layout -SessionMeta $sessionMeta -Registry $regCheck.Registry -SessionRoot $sessionRoot
    Add-OriCancelJournalStep -Journal $journal -Name 'build_plan' -Status $(if ($plan.canExecute) { 'SUCCESS' } else { 'BLOCKED' }) `
        -Detail $(if ($plan.blockers.Count -gt 0) { ($plan.blockers -join ';') } else { '' })

    $journal.status = if ($plan.canExecute) { 'SUCCESS' } else { 'BLOCKED' }
    $journalPath = Join-Path $sessionRoot "cancel-journal-preview-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    Save-OriCancelJournal -Journal $journal -Path $journalPath

    return @{
        mode        = 'PREVIEW'
        plan        = $plan
        journalPath = $journalPath
        journal     = $journal
    }
}

function Invoke-OriGitRevertBaselinePaths {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)][string]$Label,
        [AllowEmptyCollection()]
        [array]$PathsToRevert = @()
    )
    if ($null -eq $PathsToRevert -or $PathsToRevert.Count -eq 0) {
        return
    }
    foreach ($rel in $PathsToRevert) {
        $full = Join-Path $RepoRoot $rel
        $mirror = Join-Path $SessionRoot "$Label\mirror\$rel"
        $deletedBkp = Join-Path $SessionRoot "$Label\deleted-from-HEAD\$rel"
        if (Test-Path -LiteralPath $deletedBkp) {
            Invoke-OriGit -RepoRoot $RepoRoot -GitArgs @('checkout', 'HEAD', '--', $rel) | Out-Null
        }
        elseif (Test-Path -LiteralPath $mirror) {
            Invoke-OriGit -RepoRoot $RepoRoot -GitArgs @('checkout', 'HEAD', '--', $rel) | Out-Null
            if (Test-Path -LiteralPath $full -PathType Leaf) {
                Remove-Item -LiteralPath $full -Force
            }
        }
        elseif (Test-Path -LiteralPath $full -PathType Leaf) {
            Remove-Item -LiteralPath $full -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-OriCancelDevExecute {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Layout,
        [Parameter(Mandatory = $true)][string]$SessionId,
        [switch]$SimulateFailAfterBackend,
        [switch]$SimulateEnvWriteFail,
        [switch]$SimulateBackupVerifyFail,
        [string]$SessionsRoot = '',
        [string]$ProgressPath = '',
        [string]$ProgressOperationId = ''
    )
    if ([string]::IsNullOrWhiteSpace($SessionsRoot)) {
        $SessionsRoot = $Layout.SessionsRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($ProgressPath)) {
        $env:ACE_CANCEL_EXECUTE_PROGRESS_PATH = $ProgressPath.Trim()
    }
    if (-not [string]::IsNullOrWhiteSpace($ProgressOperationId)) {
        $env:ACE_CANCEL_EXECUTE_OPERATION_ID = $ProgressOperationId.Trim().ToLowerInvariant()
    }
    Set-OriCancelExecuteProgressBinding -ProgressPath $ProgressPath -ProgressOperationId $ProgressOperationId
    $sessionRoot = Join-Path $SessionsRoot $SessionId
    $lockPath = $null
    $journal = New-OriCancelJournal -SessionId $SessionId -Mode 'EXECUTE'
    $backupRoot = $null
    $overallStatus = 'FAILED'
    $progressStep = $null

    try {
        $lockPath = Enter-OriCancelDevLock -SessionsRoot $SessionsRoot
        Add-OriCancelJournalStep -Journal $journal -Name 'acquire_lock' -Status 'SUCCESS'

        $preview = Invoke-OriCancelDevPreview -Layout $Layout -SessionId $SessionId -SessionsRoot $SessionsRoot
        $plan = $preview.plan
        if (-not $plan.canExecute) {
            throw ('PLAN_BLOCKED:' + ($plan.blockers -join ','))
        }
        Add-OriCancelJournalStep -Journal $journal -Name 'preview_plan' -Status 'SUCCESS'

        if ($plan.registryInfraDrift -and $plan.registryInfraDrift.Count -gt 0) {
            throw ('REGISTRY_INFRA_DRIFT:' + ($plan.registryInfraDrift -join ','))
        }

        $progressStep = 'backup_creation'
        Start-OriCancelExecuteProgressStep -StepId $progressStep

        $ts = Get-Date -Format 'yyyy-MM-dd_HHmmss'
        $backupRoot = Join-Path $Layout.AceRoot "_backups\cancel-dev-backups\$SessionId-$ts"
        New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

        $preBackend = Get-GitRepoState -RepoRoot $Layout.BackendRoot
        $preFrontend = Get-GitRepoState -RepoRoot $Layout.FrontendRoot
        $preState = [ordered]@{
            capturedAt = (Get-Date).ToString('o')
            backend    = $preBackend
            frontend   = $preFrontend
            infraTargets = $plan.infraTargets
        }
        ($preState | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $backupRoot 'PRE_STATE.json') -Encoding UTF8
        Add-OriCancelJournalStep -Journal $journal -Name 'pre_state_snapshot' -Status 'SUCCESS' -Detail (Join-Path $backupRoot 'PRE_STATE.json')

        $bakBackend = Export-OriCancelDevRepoBackup -RepoRoot $Layout.BackendRoot -Label 'BACKEND' -BackupRoot $backupRoot
        $bakFrontend = Export-OriCancelDevRepoBackup -RepoRoot $Layout.FrontendRoot -Label 'FRONTEND' -BackupRoot $backupRoot
        $null = Export-OriCancelDevEnvBackup -BackupRoot $backupRoot -Layout $Layout
        New-OriCancelDevBackupManifest -BackupRoot $backupRoot | Out-Null
        $backupVerify = Test-OriCancelDevBackupIntegrity -BackupRoot $backupRoot -SimulateCorrupt:$SimulateBackupVerifyFail
        if (-not $backupVerify.Ok) {
            Write-OriCancelDevRecoveryGuide -BackupRoot $backupRoot -Status 'BACKUP_VERIFY_FAILED' -FailureDetail ($backupVerify.Errors -join ',')
            throw ('BACKUP_VERIFY_FAILED:' + ($backupVerify.Errors -join ','))
        }
        Write-OriCancelDevRecoveryGuide -BackupRoot $backupRoot -Status 'BACKUP_VERIFIED' -FailureDetail ''
        Add-OriCancelJournalStep -Journal $journal -Name 'pre_cancel_backup_verified' -Status 'SUCCESS' -Detail $backupRoot
        Complete-OriCancelExecuteProgressStep -StepId 'backup_creation'
        $progressStep = $null

        $progressStep = 'backend_restore'
        Start-OriCancelExecuteProgressStep -StepId $progressStep

        $sessionMeta = Get-OriSessionMeta -SessionRoot $sessionRoot
        $backendRoot = $Layout.BackendRoot
        $frontendRoot = $Layout.FrontendRoot

        Invoke-OriGitRevertBaselinePaths -RepoRoot $backendRoot -SessionRoot $sessionRoot -Label 'BACKEND' -PathsToRevert $plan.git.backend.PathsToRevert
        if ((Invoke-OriGit -RepoRoot $backendRoot -GitArgs @('checkout', '-f', $plan.infraTargets.backendBranch)) -ne 0) {
            throw 'BACKEND_CHECKOUT_INFRA_FAILED'
        }
        if ((Invoke-OriGit -RepoRoot $backendRoot -GitArgs @('reset', '--hard', $plan.infraTargets.backendCommit)) -ne 0) {
            throw 'BACKEND_RESET_INFRA_FAILED'
        }
        Add-OriCancelJournalStep -Journal $journal -Name 'backend_git_restore' -Status 'SUCCESS'
        Complete-OriCancelExecuteProgressStep -StepId 'backend_restore'
        $progressStep = $null

        if ($SimulateFailAfterBackend) {
            throw 'SIMULATED_FAIL_AFTER_BACKEND'
        }

        $progressStep = 'frontend_restore'
        Start-OriCancelExecuteProgressStep -StepId $progressStep

        Invoke-OriGitRevertBaselinePaths -RepoRoot $frontendRoot -SessionRoot $sessionRoot -Label 'FRONTEND' -PathsToRevert $plan.git.frontend.PathsToRevert
        if ((Invoke-OriGit -RepoRoot $frontendRoot -GitArgs @('checkout', '-f', $plan.infraTargets.frontendBranch)) -ne 0) {
            throw 'FRONTEND_CHECKOUT_INFRA_FAILED'
        }
        if ((Invoke-OriGit -RepoRoot $frontendRoot -GitArgs @('reset', '--hard', $plan.infraTargets.frontendCommit)) -ne 0) {
            throw 'FRONTEND_RESET_INFRA_FAILED'
        }
        Add-OriCancelJournalStep -Journal $journal -Name 'frontend_git_restore' -Status 'SUCCESS'
        Complete-OriCancelExecuteProgressStep -StepId 'frontend_restore'
        $progressStep = $null

        $backendEnvSnap = $sessionMeta.envSnapshots | Where-Object { $_.LogicalName -eq 'backend.env.local' } | Select-Object -First 1
        $frontendEnvSnap = $sessionMeta.envSnapshots | Where-Object { $_.LogicalName -eq 'frontend.env.development.local' } | Select-Object -First 1
        $backendEnvPath = Join-Path $backendRoot '.env.local'
        $frontendEnvPath = Join-Path $frontendRoot '.env.development.local'

        $progressStep = 'env_restore'
        Start-OriCancelExecuteProgressStep -StepId $progressStep

        if ($SimulateEnvWriteFail) {
            throw 'SIMULATED_ENV_WRITE_FAIL'
        }

        foreach ($block in @(
            @{
                Snap = $backendEnvSnap; Path = $backendEnvPath; Example = (Join-Path $backendRoot '.env.local.example'); Analysis = $plan.env.backend
            }
            @{
                Snap = $frontendEnvSnap; Path = $frontendEnvPath; Example = (Join-Path $frontendRoot '.env.example'); Analysis = $plan.env.frontend
            }
        )) {
            if (-not $block.Snap.Present) { continue }
            $envTargetPath = [string]$block.Path
            $built = Build-EnvRestoreKeys -EnvSnapshotItem $block.Snap -CurrentEnvPath $envTargetPath -ExamplePath $block.Example -Analysis $block.Analysis
            $text = Format-DotEnvFromKeys -Keys $built.EnvKeys -OrderSourcePlain $built.OrderPlain
            $tmp = "$envTargetPath.tmp"
            [System.IO.File]::WriteAllText($tmp, $text, [System.Text.UTF8Encoding]::new($false))
            Move-Item -LiteralPath $tmp -Destination $envTargetPath -Force
        }
        Add-OriCancelJournalStep -Journal $journal -Name 'env_restore' -Status 'SUCCESS'
        Complete-OriCancelExecuteProgressStep -StepId 'env_restore'
        $progressStep = $null

        $progressStep = 'final_verify'
        Start-OriCancelExecuteProgressStep -StepId $progressStep

        $verifyBackend = Get-GitRepoState -RepoRoot $backendRoot
        $verifyFrontend = Get-GitRepoState -RepoRoot $frontendRoot
        $ignoredUntracked = @(
            '.env.local'
            '.env.development.local'
        )
        $backendDirty = @($verifyBackend.Entries | Where-Object {
            -not ($_.ChangeType -eq 'untracked' -and ($ignoredUntracked -contains ($_.RelPath -replace '/', '\')))
        })
        $frontendDirty = @($verifyFrontend.Entries | Where-Object {
            -not ($_.ChangeType -eq 'untracked' -and ($ignoredUntracked -contains ($_.RelPath -replace '/', '\')))
        })
        $vErrors = @()
        if ($verifyBackend.Head -ne $plan.infraTargets.backendCommit) { $vErrors += 'BACKEND_HEAD_MISMATCH' }
        if ($verifyFrontend.Head -ne $plan.infraTargets.frontendCommit) { $vErrors += 'FRONTEND_HEAD_MISMATCH' }
        if ($backendDirty.Count -gt 0) { $vErrors += 'BACKEND_DIRTY_AFTER' }
        if ($frontendDirty.Count -gt 0) { $vErrors += 'FRONTEND_DIRTY_AFTER' }
        if ($vErrors.Count -gt 0) {
            throw ($vErrors -join ',')
        }
        Add-OriCancelJournalStep -Journal $journal -Name 'final_verify' -Status 'SUCCESS'
        Complete-OriCancelExecuteProgressStep -StepId 'final_verify'
        $progressStep = $null

        $progressStep = 'close_cancel_cycle'
        Start-OriCancelExecuteProgressStep -StepId $progressStep

        Set-OriCancelDevCycleCancelled -SessionRoot $sessionRoot -SessionMeta $sessionMeta -BackupRoot $backupRoot
        Add-OriCancelJournalStep -Journal $journal -Name 'close_cancel_cycle' -Status 'SUCCESS'
        Complete-OriCancelExecuteProgressStep -StepId 'close_cancel_cycle'
        $progressStep = $null

        $overallStatus = 'SUCCESS'
        $journal.status = 'SUCCESS'
        Set-OriCancelExecuteProgressCancelStatus -CancelStatus 'COMPLETED'
    }
    catch {
        $msg = $_.Exception.Message
        if ($progressStep) {
            Fail-OriCancelExecuteProgressStep -StepId $progressStep -ErrorMessage $msg
        }
        else {
            Set-OriCancelExecuteProgressOperationStatus -OperationStatus 'FAILED' -ErrorMessage $msg
        }
        Add-OriCancelJournalStep -Journal $journal -Name 'failure' -Status 'FAILED' -Detail $msg
        if ($journal.steps | Where-Object { $_.status -eq 'SUCCESS' -and $_.name -match 'backend|frontend|env' }) {
            $journal.status = 'PARTIAL'
            $overallStatus = 'PARTIAL'
        }
        else {
            $journal.status = 'FAILED'
            $overallStatus = 'FAILED'
        }
        if ($overallStatus -eq 'PARTIAL') {
            Set-OriCancelExecuteProgressCancelStatus -CancelStatus 'PARTIAL' -ErrorMessage $msg
            Set-OriCancelExecuteProgressOperationStatus -OperationStatus 'FAILED' -ErrorMessage $msg
        }
        if ($backupRoot -and (Test-Path -LiteralPath $backupRoot)) {
            Write-OriCancelDevRecoveryGuide -BackupRoot $backupRoot -Status $overallStatus -FailureDetail $msg
            $journal | Add-Member -NotePropertyName 'recovery' -NotePropertyValue ([ordered]@{
                backupRoot       = $backupRoot
                recoveryMarkdown = (Join-Path $backupRoot 'RECOVERY.md')
                manualOnly       = $true
                autoRecovery     = 'disabled'
                hint             = 'Restore FRONTEND/BACKEND from backup mirror and PRE_STATE; do not run unverified reset.'
            }) -Force
        }
        throw
    }
    finally {
        if ($lockPath) {
            Exit-OriCancelDevLock -SessionsRoot $SessionsRoot
            Add-OriCancelJournalStep -Journal $journal -Name 'release_lock' -Status 'SUCCESS'
        }
        $journalPath = Join-Path $sessionRoot "cancel-journal-execute-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
        Save-OriCancelJournal -Journal $journal -Path $journalPath
    }

    return @{
        mode        = 'EXECUTE'
        status      = $overallStatus
        journalPath = $journalPath
        backupRoot  = $backupRoot
        plan        = $plan
        recoveryMarkdown = $(if ($backupRoot) { Join-Path $backupRoot 'RECOVERY.md' } else { $null })
    }
}

function Invoke-OriCancelDevServiceRestart {
    param(
        [Parameter(Mandatory = $true)][switch]$Confirmed
    )
    if (-not $Confirmed) {
        throw 'SERVICE_RESTART_REQUIRES_CONFIRM'
    }
    Write-CancelWarn 'Service restart is not automated in this build; restart Backend, Worker, and Frontend manually when ready.'
    Write-CancelInfo 'Memurai/Redis is intentionally not stopped.'
    return @{ status = 'DEFERRED_MANUAL' }
}
