# ACE LOCAL — recovery checkpoints (phase 3; not wired to Start-AceLocal or scheduler).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryDirtyDetectLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryCheckpointPolicyLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryLedgerLib.ps1')

$script:AceLocalRecoveryCheckpointSchemaVersion = 1
$script:AceLocalRecoveryCheckpointTypeFull = 'full'
$script:AceLocalRecoveryCheckpointTypeLight = 'light'

function Get-AceLocalRecoveryCheckpointsRoot {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $dir = Join-Path $AceRoot '_backups\local-dev\recovery-checkpoints'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return $dir
}

function Get-AceLocalRecoveryCheckpointMetadataPath {
    param([Parameter(Mandatory = $true)][string]$CheckpointRoot)
    return Join-Path $CheckpointRoot 'checkpoint-metadata.json'
}

function Read-AceLocalRecoveryCheckpointMetadata {
    param([Parameter(Mandatory = $true)][string]$CheckpointRoot)
    $path = Get-AceLocalRecoveryCheckpointMetadataPath -CheckpointRoot $CheckpointRoot
    if (-not (Test-Path -LiteralPath $path)) {
        return $null
    }
    try {
        return (Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Write-AceLocalRecoveryCheckpointRecoveryGuide {
    param(
        [Parameter(Mandatory = $true)][string]$CheckpointRoot,
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$CheckpointType = 'full',
        [string]$BaseCheckpointId = ''
    )
    $lines = @(
        '# ACE LOCAL recovery checkpoint (manual restore to TEMP repos only)'
        ''
        "Status: $Status"
        "CheckpointType: $CheckpointType"
        "CheckpointRoot: $CheckpointRoot"
        $(if ($BaseCheckpointId) { "BaseCheckpointId: $BaseCheckpointId" } else { '' })
        ''
        '1. Verify checkpoint-metadata.json status is VERIFIED.'
        '2. For light checkpoints, resolve the verified chain back to the root full checkpoint.'
        '3. Restore BACKEND/FRONTEND on a TEMP git clone via Restore-AceLocalRecoveryCheckpointToTempRepos.'
        '4. ENV: decrypt *.dpapi under ENV/ with Read-AceLocalShutdownEnvPlain (same Windows user).'
        '5. Do not auto-restore onto live ACE repos or mutate production registry.'
    ) | Where-Object { $_ -ne '' }
    Set-Content -LiteralPath (Join-Path $CheckpointRoot 'RECOVERY.md') -Value ($lines -join "`n") -Encoding UTF8
}

function Import-AceLocalRecoveryCheckpointManifestRows {
    param([Parameter(Mandatory = $true)][string]$CheckpointRoot)
    $manifestPath = Join-Path $CheckpointRoot 'MANIFEST.csv'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        return @()
    }
    return @(Get-Content -LiteralPath $manifestPath | ConvertFrom-Csv)
}

function ConvertTo-AceLocalRecoveryManifestSnap {
    param([Parameter(Mandatory = $true)]$Snap)
    $entries = @()
    foreach ($e in @($Snap.Entries)) {
        if ($null -eq $e) { continue }
        $changeType = ''
        $relPath = ''
        if ($e.PSObject.Properties.Match('ChangeType').Count -gt 0) {
            $changeType = [string]$e.ChangeType
        }
        elseif ($e.PSObject.Properties.Match('changeType').Count -gt 0) {
            $changeType = [string]$e.changeType
        }
        if ($e.PSObject.Properties.Match('RelPath').Count -gt 0) {
            $relPath = [string]$e.RelPath
        }
        elseif ($e.PSObject.Properties.Match('path').Count -gt 0) {
            $relPath = ([string]$e.path) -replace '/', '\'
        }
        if (-not $relPath) { continue }
        $entries += [PSCustomObject]@{
            XY         = ''
            RelPath    = $relPath
            ChangeType = $(if ($changeType) { $changeType } else { 'modified' })
        }
    }
    return [PSCustomObject]@{
        Label           = [string]$Snap.Label
        Branch          = [string]$Snap.Branch
        Head            = [string]$Snap.Head
        Entries         = @($entries)
        ConcurrentDrift = @($Snap.ConcurrentDrift)
        IgnoredReview   = @()
        DestRoot        = [string]$Snap.DestRoot
    }
}

function Get-AceLocalRecoveryVerifiedCheckpoints {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$CheckpointsRootOverride = ''
    )
    $root = if ($CheckpointsRootOverride) { $CheckpointsRootOverride } else { Get-AceLocalRecoveryCheckpointsRoot -AceRoot $AceRoot }
    $list = @()
    foreach ($dir in @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue)) {
        $meta = Read-AceLocalRecoveryCheckpointMetadata -CheckpointRoot $dir.FullName
        if (-not $meta) { continue }
        if ([string]$meta.status -ne 'VERIFIED') { continue }
        $list += [PSCustomObject]@{
            CheckpointId   = [string]$meta.checkpointId
            CheckpointRoot = $dir.FullName
            VerifiedAt     = [string]$meta.verifiedAt
            CheckpointType = [string]$meta.checkpointType
            BaseCheckpointId = [string]$meta.baseCheckpointId
            ChainRootId    = [string]$meta.chainRootId
        }
    }
    return @($list | Sort-Object { $_.VerifiedAt })
}

function Get-AceLocalRecoveryLightDeltaEntries {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][ValidateSet('BACKEND', 'FRONTEND')][string]$Label,
        [Parameter(Mandatory = $true)][array]$BaseManifestRows,
        [AllowEmptyCollection()][string[]]$IgnoredIncludePatterns = $script:AceLocalShutdownIgnoredIncludePatterns
    )
    $repoRoot = (Resolve-Path -LiteralPath $RepoRoot).ProviderPath
    $scan = Get-AceLocalRecoveryRepoDirtyScan -RepoRoot $repoRoot -Label $Label -IgnoredIncludePatterns $IgnoredIncludePatterns
    $entries = @($scan['Entries'])

    $baseByPath = @{}
    foreach ($row in @($BaseManifestRows)) {
        if ([string]$row.Area -ne $Label) { continue }
        $key = ([string]$row.RelPath -replace '\\', '/').ToLowerInvariant()
        $baseByPath[$key] = $row
    }

    $delta = @()
    foreach ($e in @($entries)) {
        $rel = ([string]$e.path) -replace '/', '\'
        if ([string]$e.changeType -eq 'deleted') {
            $delta += [PSCustomObject]@{
                XY = ' D'
                RelPath = $rel
                ChangeType = 'deleted'
            }
            continue
        }
        $full = Join-Path $repoRoot $rel
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
        $key = ([string]$e.path).ToLowerInvariant()
        $curHash = Get-FileSha256Hex $full
        if (-not $baseByPath.ContainsKey($key)) {
            $delta += [PSCustomObject]@{
                XY = '??'
                RelPath = $rel
                ChangeType = [string]$e.changeType
            }
            continue
        }
        $baseSha = [string]$baseByPath[$key].BackupSha256
        if ($baseSha -and ($baseSha -ne $curHash)) {
            $delta += [PSCustomObject]@{
                XY = ' M'
                RelPath = $rel
                ChangeType = [string]$e.changeType
            }
        }
    }
    return @($delta)
}

function Get-AceLocalRecoveryEntryChangeType {
    param($Entry)
    if ($null -eq $Entry) { return '' }
    if ($Entry.PSObject.Properties.Match('ChangeType').Count -gt 0) {
        return [string]$Entry.ChangeType
    }
    if ($Entry.PSObject.Properties.Match('changeType').Count -gt 0) {
        return [string]$Entry.changeType
    }
    return ''
}

function Get-AceLocalRecoveryEntryRelPath {
    param($Entry)
    if ($null -eq $Entry) { return '' }
    if ($Entry.PSObject.Properties.Match('RelPath').Count -gt 0) {
        return [string]$Entry.RelPath
    }
    if ($Entry.PSObject.Properties.Match('path').Count -gt 0) {
        return ([string]$Entry.path) -replace '/', '\'
    }
    return ''
}

function Export-AceLocalRecoveryRepoSnapshotEntries {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][ValidateSet('BACKEND', 'FRONTEND')][string]$Label,
        [Parameter(Mandatory = $true)][string]$CheckpointRoot,
        [AllowEmptyCollection()][array]$Entries = @(),
        [scriptblock]$AfterInitialCaptureHook = $null,
        [switch]$SimulateFailBeforeComplete
    )
    $out = Join-Path $CheckpointRoot $Label
    $gitDir = Join-Path $out 'git'
    $mirrorDir = Join-Path $out 'mirror'
    $deletedDir = Join-Path $out 'deleted-from-HEAD'
    New-Item -ItemType Directory -Path $gitDir, $mirrorDir, $deletedDir -Force | Out-Null

    Push-Location -LiteralPath $RepoRoot
    try {
        $branch = (git rev-parse --abbrev-ref HEAD).Trim()
        $head = (git rev-parse HEAD).Trim()
        Set-Content -LiteralPath (Join-Path $gitDir 'branch.txt') -Value $branch -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'HEAD.txt') -Value $head -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'status-porcelain.txt') -Value (git status -u --porcelain=v1) -Encoding UTF8

        $sourceHashAtStart = @{}
        foreach ($e in @($Entries)) {
            $changeType = Get-AceLocalRecoveryEntryChangeType -Entry $e
            $relPath = Get-AceLocalRecoveryEntryRelPath -Entry $e
            if ($changeType -eq 'deleted') { continue }
            if (-not $relPath) { continue }
            $full = Join-Path $RepoRoot $relPath
            if (Test-Path -LiteralPath $full -PathType Leaf) {
                $key = ($relPath -replace '\\', '/').ToLowerInvariant()
                $sourceHashAtStart[$key] = Get-FileSha256Hex $full
            }
        }

        if ($AfterInitialCaptureHook) {
            & $AfterInitialCaptureHook
        }

        $concurrentDrift = @()
        foreach ($e in @($Entries)) {
            $changeType = Get-AceLocalRecoveryEntryChangeType -Entry $e
            $relPath = Get-AceLocalRecoveryEntryRelPath -Entry $e
            if (-not $relPath) { continue }
            if ($changeType -eq 'deleted') {
                $destDeleted = Join-Path $deletedDir $relPath
                New-Item -ItemType Directory -Path (Split-Path -Parent $destDeleted) -Force | Out-Null
                $blob = git show "HEAD:$($relPath -replace '\\','/')"
                Set-Content -LiteralPath $destDeleted -Value $blob -Encoding UTF8
                continue
            }
            $copy = Copy-AceLocalShutdownRepoFileToMirror -RepoRoot $RepoRoot -MirrorRoot $mirrorDir `
                -RelPath $relPath -SourceHashAtStart $sourceHashAtStart
            if ($null -eq $copy -or $copy.Skipped) { continue }
            if ($copy.ConcurrentDrift) {
                $concurrentDrift += [PSCustomObject]@{
                    RelPath = $relPath
                    StartSha256 = $copy.StartSourceSha256
                    EndSha256 = $copy.SourceSha256
                }
            }
        }

        if ($SimulateFailBeforeComplete) {
            throw 'RECOVERY_CHECKPOINT_SIMULATED_FAIL'
        }

        return [PSCustomObject]@{
            Label           = $Label
            Branch          = $branch
            Head            = $head
            Entries         = @($Entries)
            ConcurrentDrift = @($concurrentDrift)
            DestRoot        = $out
        }
    }
    finally {
        Pop-Location
    }
}

function Test-AceLocalRecoveryCheckpointIntegrity {
    param(
        [Parameter(Mandatory = $true)][string]$CheckpointRoot,
        [switch]$SimulateManifestHashMismatch
    )
    $errors = @()
    $metaPath = Get-AceLocalRecoveryCheckpointMetadataPath -CheckpointRoot $CheckpointRoot
    if (-not (Test-Path -LiteralPath $metaPath)) {
        $errors += 'CHECKPOINT_METADATA_MISSING'
    }
    $manifestPath = Join-Path $CheckpointRoot 'MANIFEST.csv'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        $errors += 'MANIFEST_MISSING'
    }
    else {
        $rows = @(Get-Content -LiteralPath $manifestPath | ConvertFrom-Csv)
        if ($rows.Count -eq 0) {
            $errors += 'MANIFEST_EMPTY'
        }
        foreach ($row in $rows) {
            $area = [string]$row.Area
            $rel = [string]$row.RelPath
            $backupSha = [string]$row.BackupSha256
            $full = $null
            switch ($area) {
                'BACKEND' {
                    if ([string]$row.ChangeType -eq 'deleted') {
                        $full = Join-Path $CheckpointRoot "BACKEND\deleted-from-HEAD\$rel"
                    }
                    else {
                        $full = Join-Path $CheckpointRoot "BACKEND\mirror\$rel"
                    }
                }
                'FRONTEND' {
                    if ([string]$row.ChangeType -eq 'deleted') {
                        $full = Join-Path $CheckpointRoot "FRONTEND\deleted-from-HEAD\$rel"
                    }
                    else {
                        $full = Join-Path $CheckpointRoot "FRONTEND\mirror\$rel"
                    }
                }
                'ENV' {
                    if ([string]$row.ChangeType -eq 'env-dpapi') {
                        $full = Join-Path $CheckpointRoot "ENV\$rel.dpapi"
                    }
                }
            }
            if (-not $full -or -not (Test-Path -LiteralPath $full)) {
                if ([string]$row.ChangeType -ne 'env-missing') {
                    $errors += "MANIFEST_FILE_MISSING:${area}:${rel}"
                }
                continue
            }
            $actual = Get-FileSha256Hex $full
            if ($SimulateManifestHashMismatch) {
                $actual = '0' * 64
            }
            if ($backupSha -and ($actual -ne $backupSha)) {
                $errors += "MANIFEST_HASH_MISMATCH:${area}:${rel}"
            }
        }
    }
    foreach ($label in @('BACKEND', 'FRONTEND')) {
        $headFile = Join-Path $CheckpointRoot "$label\git\HEAD.txt"
        $branchFile = Join-Path $CheckpointRoot "$label\git\branch.txt"
        if (-not (Test-Path -LiteralPath $headFile)) { $errors += "MISSING_${label}_HEAD" }
        if (-not (Test-Path -LiteralPath $branchFile)) { $errors += "MISSING_${label}_BRANCH" }
    }
    return @{
        Ok     = ($errors.Count -eq 0)
        Errors = @($errors)
    }
}

function New-AceLocalRecoveryCheckpoint {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)][ValidateSet('full', 'light')][string]$CheckpointType,
        [string]$BaseCheckpointId = '',
        [string]$CheckpointsRootOverride = '',
        [scriptblock]$DuringBackendCopyHook = $null,
        [switch]$SimulateFailAfterBackend,
        [switch]$SimulateFailBeforeVerify,
        [switch]$ForceWithoutWorkingTreeChanges
    )
    $BackendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
    $FrontendRoot = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath
    $AceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath

    try {
        $scan = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
    }
    catch {
        return @{
            Ok      = $false
            Skipped = $false
            Reason  = 'dirty_scan_failed'
            Status  = 'FAILED'
            Errors  = @([string]$_.Exception.Message)
        }
    }
    $hasChanges = Test-AceLocalRecoveryWorkspaceHasWorkingTreeChanges -WorkspaceScan $scan
    if (-not $ForceWithoutWorkingTreeChanges -and -not $hasChanges) {
        return @{
            Ok = $false
            Skipped = $true
            Reason = 'no_working_tree_changes'
            Status = 'SKIPPED'
        }
    }

    $checkpointsRoot = if ($CheckpointsRootOverride) {
        $CheckpointsRootOverride
    }
    else {
        Get-AceLocalRecoveryCheckpointsRoot -AceRoot $AceRoot
    }
    New-Item -ItemType Directory -Path $checkpointsRoot -Force | Out-Null

    $baseRoot = $null
    $baseManifest = @()
    $chainRootId = ''
    if ($CheckpointType -eq $script:AceLocalRecoveryCheckpointTypeLight) {
        if (-not $BaseCheckpointId) {
            throw 'RECOVERY_CHECKPOINT_BASE_REQUIRED'
        }
        $verified = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $AceRoot -CheckpointsRootOverride $checkpointsRoot)
        $base = @($verified | Where-Object { $_.CheckpointId -eq $BaseCheckpointId } | Select-Object -First 1)
        if (-not $base) {
            throw 'RECOVERY_CHECKPOINT_BASE_NOT_VERIFIED'
        }
        $baseRoot = [string]$base.CheckpointRoot
        $baseManifest = @(Import-AceLocalRecoveryCheckpointManifestRows -CheckpointRoot $baseRoot)
        $chainRootId = if ([string]$base.ChainRootId) { [string]$base.ChainRootId } else { [string]$base.CheckpointId }
    }

    $checkpointId = 'cp-{0}-{1}' -f (Get-Date -Format 'yyyy-MM-dd_HHmmss'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
    $checkpointRoot = Join-Path $checkpointsRoot $checkpointId
    New-Item -ItemType Directory -Path $checkpointRoot -Force | Out-Null

    $metadataPath = Get-AceLocalRecoveryCheckpointMetadataPath -CheckpointRoot $checkpointRoot
    $metadata = [ordered]@{
        schemaVersion     = $script:AceLocalRecoveryCheckpointSchemaVersion
        checkpointId      = $checkpointId
        checkpointType    = $CheckpointType
        baseCheckpointId  = $(if ($BaseCheckpointId) { $BaseCheckpointId } else { $null })
        chainRootId       = $(if ($chainRootId) { $chainRootId } else { $checkpointId })
        launcherSession   = $LauncherSession
        status            = 'INCOMPLETE'
        startedAt         = (Get-Date).ToUniversalTime().ToString('o')
        finishedAt        = $null
        verifiedAt        = $null
        backendRoot       = $BackendRoot
        frontendRoot      = $FrontendRoot
        aceRoot           = $AceRoot
        errors            = @()
        concurrentDrift   = @()
        integrity         = $null
        contentFingerprint = (Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $scan)
    }
    Write-AceLocalRecoveryJsonAtomic -Path $metadataPath -Object $metadata

    $ledgerBefore = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
    $prevVerifiedId = $null
    if ($ledgerBefore -and $ledgerBefore.lastVerifiedCheckpoint) {
        $prevVerifiedId = [string]$ledgerBefore.lastVerifiedCheckpoint.checkpointId
    }

    $result = @{
        Ok             = $false
        Skipped        = $false
        CheckpointId   = $checkpointId
        CheckpointRoot = $checkpointRoot
        Status         = 'INCOMPLETE'
        Errors         = @()
    }

    try {
        if ($CheckpointType -eq $script:AceLocalRecoveryCheckpointTypeFull) {
            $backendSnap = [PSCustomObject](Export-AceLocalShutdownRepoSnapshot -RepoRoot $BackendRoot -Label 'BACKEND' `
                -SnapshotRoot $checkpointRoot -AfterInitialCaptureHook $DuringBackendCopyHook)
        }
        else {
            $beDelta = Get-AceLocalRecoveryLightDeltaEntries -RepoRoot $BackendRoot -Label 'BACKEND' -BaseManifestRows $baseManifest
            $backendSnap = Export-AceLocalRecoveryRepoSnapshotEntries -RepoRoot $BackendRoot -Label 'BACKEND' `
                -CheckpointRoot $checkpointRoot -Entries $beDelta -AfterInitialCaptureHook $DuringBackendCopyHook
        }

        if ($SimulateFailAfterBackend) {
            throw 'RECOVERY_CHECKPOINT_SIMULATED_FAIL_AFTER_BACKEND'
        }

        if ($CheckpointType -eq $script:AceLocalRecoveryCheckpointTypeFull) {
            $frontendSnap = [PSCustomObject](Export-AceLocalShutdownRepoSnapshot -RepoRoot $FrontendRoot -Label 'FRONTEND' `
                -SnapshotRoot $checkpointRoot)
        }
        else {
            $feDelta = Get-AceLocalRecoveryLightDeltaEntries -RepoRoot $FrontendRoot -Label 'FRONTEND' -BaseManifestRows $baseManifest
            $frontendSnap = Export-AceLocalRecoveryRepoSnapshotEntries -RepoRoot $FrontendRoot -Label 'FRONTEND' `
                -CheckpointRoot $checkpointRoot -Entries $feDelta
        }

        $envSnaps = @(Export-AceLocalShutdownEnvSnapshots -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -SnapshotRoot $checkpointRoot)

        $backendForManifest = ConvertTo-AceLocalRecoveryManifestSnap -Snap $backendSnap
        $frontendForManifest = ConvertTo-AceLocalRecoveryManifestSnap -Snap $frontendSnap
        $manifestRows = New-AceLocalShutdownSnapshotManifestRows -BackendSnap $backendForManifest -FrontendSnap $frontendForManifest `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -SnapshotRoot $checkpointRoot `
            -EnvSnapshots $envSnaps -LocalStateCopied @('__recovery_checkpoint_no_local_state__')
        $result.PreviousVerifiedCheckpointId = $prevVerifiedId
        Write-AceLocalShutdownSnapshotManifest -SnapshotRoot $checkpointRoot -Rows $manifestRows | Out-Null

        $drift = @($backendSnap.ConcurrentDrift) + @($frontendSnap.ConcurrentDrift)
        if ($drift.Count -gt 0) {
            $metadata.concurrentDrift = @($drift)
            throw ('RECOVERY_CHECKPOINT_CONCURRENT_DRIFT:' + ($drift | ForEach-Object { $_.RelPath } | Select-Object -First 5 -Join ','))
        }

        if ($SimulateFailBeforeVerify) {
            throw 'RECOVERY_CHECKPOINT_SIMULATED_FAIL_BEFORE_VERIFY'
        }

        $integrity = Test-AceLocalRecoveryCheckpointIntegrity -CheckpointRoot $checkpointRoot
        $metadata.integrity = $integrity
        if (-not $integrity.Ok) {
            throw ('RECOVERY_CHECKPOINT_INTEGRITY_FAILED:' + ($integrity.Errors -join ','))
        }

        $metadata.status = 'VERIFIED'
        $metadata.verifiedAt = (Get-Date).ToUniversalTime().ToString('o')
        $metadata.finishedAt = $metadata.verifiedAt
        Write-AceLocalRecoveryCheckpointRecoveryGuide -CheckpointRoot $checkpointRoot -Status 'VERIFIED' `
            -CheckpointType $CheckpointType -BaseCheckpointId $(if ($BaseCheckpointId) { $BaseCheckpointId } else { '' })
        Write-AceLocalRecoveryJsonAtomic -Path $metadataPath -Object $metadata

        Set-AceLocalRecoveryLedgerLastVerifiedCheckpoint -AceRoot $AceRoot -CheckpointId $checkpointId `
            -CheckpointPath $checkpointRoot -VerifiedAt $metadata.verifiedAt -CheckpointType $CheckpointType | Out-Null

        $result.Ok = $true
        $result.Status = 'VERIFIED'
        return $result
    }
    catch {
        $msg = $_.Exception.Message
        $result.Errors = @($msg)
        $metadata.status = 'INCOMPLETE'
        $metadata.finishedAt = (Get-Date).ToUniversalTime().ToString('o')
        $metadata.errors = @($msg)
        if (-not $metadata.integrity) {
            $metadata.integrity = Test-AceLocalRecoveryCheckpointIntegrity -CheckpointRoot $checkpointRoot
        }
        Write-AceLocalRecoveryCheckpointRecoveryGuide -CheckpointRoot $checkpointRoot -Status 'INCOMPLETE' `
            -CheckpointType $CheckpointType -BaseCheckpointId $(if ($BaseCheckpointId) { $BaseCheckpointId } else { '' })
        Write-AceLocalRecoveryJsonAtomic -Path $metadataPath -Object $metadata
        return $result
    }
}

function Get-AceLocalRecoveryCheckpointChain {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$CheckpointId
    )
    $verified = @{}
    foreach ($row in @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $AceRoot)) {
        $verified[[string]$row.CheckpointId] = $row
    }
    if (-not $verified.ContainsKey($CheckpointId)) {
        return @()
    }
    $chain = @()
    $cur = $CheckpointId
    $guard = 0
    while ($cur -and $guard -lt 64) {
        $guard++
        if (-not $verified.ContainsKey($cur)) {
            return @()
        }
        $row = $verified[$cur]
        $chain = ,$row + $chain
        $cur = [string]$row.BaseCheckpointId
    }
    return @($chain)
}

function Apply-AceLocalRecoveryCheckpointRepoDelta {
    param(
        [Parameter(Mandatory = $true)][string]$CheckpointRoot,
        [Parameter(Mandatory = $true)][ValidateSet('BACKEND', 'FRONTEND')][string]$Label,
        [Parameter(Mandatory = $true)][string]$TargetRepoRoot,
        [switch]$IsFullCheckpoint
    )
    $src = Join-Path $CheckpointRoot $Label
    if ($IsFullCheckpoint) {
        return Restore-AceLocalShutdownRepoToTempRoot -SnapshotRoot $CheckpointRoot -Label $Label -TargetRepoRoot $TargetRepoRoot
    }
    $gitDir = Join-Path $src 'git'
    if (Test-Path -LiteralPath $gitDir) {
        $branch = (Get-Content -LiteralPath (Join-Path $gitDir 'branch.txt') -Raw).Trim()
        $head = (Get-Content -LiteralPath (Join-Path $gitDir 'HEAD.txt') -Raw).Trim()
        Push-Location -LiteralPath $TargetRepoRoot
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        try {
            git checkout -f $branch 2>$null | Out-Null
            git reset --hard $head 2>$null | Out-Null
        }
        finally {
            $ErrorActionPreference = $prevEap
            Pop-Location
        }
    }
    $mirror = Join-Path $src 'mirror'
    if (Test-Path -LiteralPath $mirror) {
        Get-ChildItem -LiteralPath $mirror -Recurse -File | ForEach-Object {
            $rel = $_.FullName.Substring($mirror.Length).TrimStart('\')
            $dest = Join-Path $TargetRepoRoot $rel
            New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
        }
    }
    $deletedDir = Join-Path $src 'deleted-from-HEAD'
    if (Test-Path -LiteralPath $deletedDir) {
        Get-ChildItem -LiteralPath $deletedDir -Recurse -File | ForEach-Object {
            $rel = $_.FullName.Substring($deletedDir.Length).TrimStart('\')
            $dest = Join-Path $TargetRepoRoot $rel
            if (Test-Path -LiteralPath $dest) {
                Remove-Item -LiteralPath $dest -Force
            }
        }
    }
    return @{ Branch = $branch; Head = $head }
}

function Restore-AceLocalRecoveryCheckpointToTempRepos {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$CheckpointId,
        [Parameter(Mandatory = $true)][string]$TargetBackendRoot,
        [Parameter(Mandatory = $true)][string]$TargetFrontendRoot
    )
    $chain = @(Get-AceLocalRecoveryCheckpointChain -AceRoot $AceRoot -CheckpointId $CheckpointId)
    if ($chain.Count -eq 0) {
        throw 'RECOVERY_CHECKPOINT_CHAIN_INVALID'
    }
    $root = $chain[0]
    if ([string]$root.CheckpointType -ne $script:AceLocalRecoveryCheckpointTypeFull) {
        throw 'RECOVERY_CHECKPOINT_CHAIN_ROOT_NOT_FULL'
    }
    $beResult = Apply-AceLocalRecoveryCheckpointRepoDelta -CheckpointRoot $root.CheckpointRoot -Label 'BACKEND' `
        -TargetRepoRoot $TargetBackendRoot -IsFullCheckpoint
    $feResult = Apply-AceLocalRecoveryCheckpointRepoDelta -CheckpointRoot $root.CheckpointRoot -Label 'FRONTEND' `
        -TargetRepoRoot $TargetFrontendRoot -IsFullCheckpoint
    foreach ($cp in @($chain | Select-Object -Skip 1)) {
        Apply-AceLocalRecoveryCheckpointRepoDelta -CheckpointRoot $cp.CheckpointRoot -Label 'BACKEND' `
            -TargetRepoRoot $TargetBackendRoot | Out-Null
        Apply-AceLocalRecoveryCheckpointRepoDelta -CheckpointRoot $cp.CheckpointRoot -Label 'FRONTEND' `
            -TargetRepoRoot $TargetFrontendRoot | Out-Null
    }
    return @{
        Chain = @($chain | ForEach-Object { $_.CheckpointId })
        Backend = $beResult
        Frontend = $feResult
    }
}

function Get-AceLocalRecoveryCheckpointRetentionPlan {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $verified = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $AceRoot)
    $protected = @{}
    $reasons = @{}

    function Protect([string]$Id, [string]$Reason) {
        if (-not $Id) { return }
        $protected[$Id] = $true
        if (-not $reasons.ContainsKey($Id)) { $reasons[$Id] = @() }
        $reasons[$Id] += $Reason
    }

    if ($verified.Count -gt 0) {
        $last = $verified[-1]
        Protect ([string]$last.CheckpointId) 'last_verified_checkpoint'
    }

    foreach ($row in @($verified)) {
        $id = [string]$row.CheckpointId
        foreach ($other in @($verified)) {
            if ([string]$other.BaseCheckpointId -eq $id) {
                Protect $id 'incremental_chain_base'
            }
        }
    }

    $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot
    if ($ledger -and $ledger.openIncident -and $ledger.lastVerifiedCheckpoint) {
        Protect ([string]$ledger.lastVerifiedCheckpoint.checkpointId) 'open_recovery_incident'
    }

    $minKeep = $script:AceLocalRecoveryMinimumVerifiedCheckpointRetention
    $sorted = @($verified | Sort-Object { $_.VerifiedAt })
    $newestFirst = @($sorted | Sort-Object { $_.VerifiedAt } -Descending)
    for ($i = 0; $i -lt [Math]::Min($minKeep, $newestFirst.Count); $i++) {
        Protect ([string]$newestFirst[$i].CheckpointId) 'minimum_retention_count'
    }

    $deletable = @()
    foreach ($row in @($verified)) {
        $id = [string]$row.CheckpointId
        if ($protected.ContainsKey($id)) { continue }
        $deletable += $row
    }

    return @{
        VerifiedCount   = $verified.Count
        ProtectedIds    = @($protected.Keys | Sort-Object)
        ProtectedReasons = $reasons
        DeletableIds    = @($deletable | ForEach-Object { $_.CheckpointId })
        WouldDeleteCount = $deletable.Count
        Note            = 'decision_only_no_files_deleted'
        ExternalProtectedPaths = @(
            '_backups/local-dev/shutdown-snapshots'
            '_backups/**/cancel-dev-backups'
        )
    }
}

function Invoke-AceLocalRecoveryCheckpointFromDecision {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [Parameter(Mandatory = $true)][string]$DecisionAction,
        [string]$CheckpointsRootOverride = ''
    )
    switch ($DecisionAction) {
        $script:AceLocalRecoveryCheckpointActionFullCheckpoint {
            return New-AceLocalRecoveryCheckpoint -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
                -AceRoot $AceRoot -LauncherSession $LauncherSession -CheckpointType 'full' `
                -CheckpointsRootOverride $CheckpointsRootOverride
        }
        $script:AceLocalRecoveryCheckpointActionLightCheckpoint {
            $ledger = Read-AceLocalRecoveryLedger -AceRoot $AceRoot -AllowCreateStateDir
            $baseId = ''
            if ($ledger -and $ledger.lastVerifiedCheckpoint) {
                $baseId = [string]$ledger.lastVerifiedCheckpoint.checkpointId
            }
            if (-not $baseId) {
                return @{ Ok = $false; Skipped = $true; Reason = 'no_verified_base_for_light' }
            }
            return New-AceLocalRecoveryCheckpoint -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
                -AceRoot $AceRoot -LauncherSession $LauncherSession -CheckpointType 'light' `
                -BaseCheckpointId $baseId -CheckpointsRootOverride $CheckpointsRootOverride
        }
        default {
            return @{ Ok = $false; Skipped = $true; Reason = 'not_a_checkpoint_action' }
        }
    }
}
