# ACE LOCAL — shutdown development-work snapshot (read-only on source repos; not wired to X/Stop).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

$script:AceLocalShutdownSnapshotSchemaVersion = 1
$script:AceLocalShutdownSnapshotMaxFileBytes = 64MB

# Explicit allowlist: ignored paths that must still be captured (not excluded solely due to .gitignore).
$script:AceLocalShutdownIgnoredIncludePatterns = @(
    '*.local.json'
    '*.local.yaml'
    '*.local.yml'
    'local-overrides/**'
    'local-dev-overrides/**'
    '.cursor/rules/**'
)

# Hard denylist segments (always skip copy; may appear in userReview report if ignored).
$script:AceLocalShutdownPathExcludeSegmentPatterns = @(
    'node_modules'
    '.git'
    '__pycache__'
    '.pytest_cache'
    '.mypy_cache'
    '.ruff_cache'
    '.venv'
    'venv'
    '.tox'
    'dist'
    'build'
    '.next'
    '.nuxt'
    'coverage'
    '.coverage'
    'htmlcov'
)

$script:AceLocalShutdownLargeMediaExtensions = @(
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.m4v',
    '.wav', '.flac', '.zip', '.7z', '.tar', '.gz', '.iso'
)

function Get-AceLocalShutdownSnapshotsRoot {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path $AceRoot '_backups\local-dev\shutdown-snapshots'
}

function Get-AceLocalShutdownLastVerifiedPointerPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    return Join-Path (Get-AceLocalShutdownSnapshotsRoot -AceRoot $AceRoot) 'last-verified-shutdown-snapshot.json'
}

function Write-AceLocalShutdownJsonAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Object
    )
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $json = $Object | ConvertTo-Json -Depth 12
    $tmp = Join-Path $dir ("{0}.tmp.{1}" -f (Split-Path -Leaf $Path), [guid]::NewGuid().ToString('N'))
    Set-Content -LiteralPath $tmp -Value $json -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Test-AceLocalShutdownSimpleGlobMatch {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Pattern
    )
    $pathNorm = ($RelativePath -replace '\\', '/').TrimStart('/')
    $patNorm = ($Pattern -replace '\\', '/').TrimStart('/')
    if ($patNorm -match '\*\*/') {
        $suffix = ($patNorm -replace '^\*\*/', '')
        if ($suffix -match '\*') {
            $rx = '^' + ($suffix -replace '\.', '\.' -replace '\*', '.*') + '$'
            return [bool]($pathNorm -match $rx)
        }
        return $pathNorm.EndsWith($suffix) -or ($pathNorm -like "*/$suffix")
    }
    if ($patNorm -match '\*') {
        $rx = '^' + ($patNorm -replace '\.', '\.' -replace '\*', '.*') + '$'
        return [bool]($pathNorm -match $rx)
    }
    return ($pathNorm -eq $patNorm)
}

function Test-AceLocalShutdownIgnoredIncludeMatch {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [AllowEmptyCollection()][string[]]$IncludePatterns = $script:AceLocalShutdownIgnoredIncludePatterns
    )
    foreach ($pat in @($IncludePatterns)) {
        if (Test-AceLocalShutdownSimpleGlobMatch -RelativePath $RelativePath -Pattern $pat) {
            return $true
        }
    }
    return $false
}

function Test-AceLocalShutdownPathExcluded {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [int]$SizeBytes = -1
    )
    $norm = ($RelativePath -replace '\\', '/').TrimStart('/').ToLowerInvariant()
    foreach ($seg in $script:AceLocalShutdownPathExcludeSegmentPatterns) {
        $segL = $seg.ToLowerInvariant()
        if ($norm -eq $segL -or $norm -like "*/$segL/*" -or $norm -like "$segL/*" -or $norm.EndsWith("/$segL")) {
            return @{ Excluded = $true; Reason = "denylist_segment:$seg" }
        }
    }
    $ext = [System.IO.Path]::GetExtension($norm)
    if ($ext -and ($script:AceLocalShutdownLargeMediaExtensions -contains $ext)) {
        return @{ Excluded = $true; Reason = "large_media_extension:$ext" }
    }
    if ($SizeBytes -ge 0 -and $SizeBytes -gt $script:AceLocalShutdownSnapshotMaxFileBytes) {
        return @{ Excluded = $true; Reason = 'file_size_over_limit' }
    }
    return @{ Excluded = $false; Reason = '' }
}

function Test-AceLocalShutdownRepoRelativePathResolvable {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$RelPath,
        [Parameter(Mandatory = $true)][string]$ChangeType
    )
    if ([string]::IsNullOrWhiteSpace($RelPath)) {
        throw 'RECOVERY_DIRTY_SCAN_EMPTY_PATH'
    }
    if ($RelPath.IndexOfAny([System.IO.Path]::GetInvalidPathChars()) -ge 0) {
        throw ("RECOVERY_DIRTY_SCAN_INVALID_PATH_CHARS:{0}" -f $RelPath)
    }
    if ($RelPath.IndexOf('"') -ge 0) {
        throw ("RECOVERY_DIRTY_SCAN_PATH_QUOTE_CHAR:{0}" -f $RelPath)
    }
    if ([string]$ChangeType -eq 'deleted') {
        return
    }
    $full = Join-Path $RepoRoot $RelPath
    try {
        $null = Test-Path -LiteralPath $full
    }
    catch {
        throw ("RECOVERY_DIRTY_SCAN_PATH_TEST_FAILED:{0}:{1}" -f $RelPath, $_.Exception.Message)
    }
}

function Get-AceLocalShutdownGitIgnoredUntrackedPaths {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)
    $repoRoot = (Resolve-Path -LiteralPath $RepoRoot).ProviderPath
    $bytes = Invoke-GitCommandStdoutBytes -WorkingDirectory $repoRoot -Arguments @(
        'ls-files', '-z', '-o', '-i', '--exclude-standard'
    )
    $parsed = ConvertFrom-GitPathListZBytes -Bytes $bytes
    if (-not $parsed.Ok) {
        throw ('RECOVERY_GIT_IGNORED_PATHS_PARSE_FAILED:' + ($parsed.Errors -join ','))
    }
    return @($parsed.Paths)
}

function Get-AceLocalShutdownGitPorcelainEntries {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)
    $repoRoot = (Resolve-Path -LiteralPath $RepoRoot).ProviderPath
    $bytes = Invoke-GitCommandStdoutBytes -WorkingDirectory $repoRoot -Arguments @(
        'status', '--porcelain=v1', '-z', '-u'
    )
    $parsed = ConvertFrom-GitPorcelainV1ZBytes -Bytes $bytes
    if (-not $parsed.Ok) {
        throw ('RECOVERY_GIT_PORCELAIN_PARSE_FAILED:' + ($parsed.Errors -join ','))
    }
    return @($parsed.Entries)
}

function Copy-AceLocalShutdownRepoFileToMirror {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$MirrorRoot,
        [Parameter(Mandatory = $true)][string]$RelPath,
        [hashtable]$SourceHashAtStart
    )
    $full = Join-Path $RepoRoot $RelPath
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
        return $null
    }
    $size = (Get-Item -LiteralPath $full).Length
    $ex = Test-AceLocalShutdownPathExcluded -RelativePath $RelPath -SizeBytes $size
    if ($ex.Excluded) {
        return @{ Skipped = $true; Reason = $ex.Reason }
    }
    $destMirror = Join-Path $MirrorRoot $RelPath
    New-Item -ItemType Directory -Path (Split-Path -Parent $destMirror) -Force | Out-Null
    Copy-Item -LiteralPath $full -Destination $destMirror -Force
    $srcHash = Get-FileSha256Hex $full
    $bkpHash = Get-FileSha256Hex $destMirror
    $key = ($RelPath -replace '\\', '/').ToLowerInvariant()
    $startHash = [string]$SourceHashAtStart[$key]
    $drift = ($startHash -and $srcHash -ne $startHash)
    if ($srcHash -ne $bkpHash) {
        throw "SHUTDOWN_SNAPSHOT_COPY_HASH_MISMATCH:$RelPath"
    }
    return @{
        Skipped          = $false
        SourceSha256     = $srcHash
        BackupSha256     = $bkpHash
        SizeBytes        = $size
        ConcurrentDrift  = [bool]$drift
        StartSourceSha256 = $startHash
    }
}

function Export-AceLocalShutdownRepoSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$SnapshotRoot,
        [AllowEmptyCollection()][string[]]$IgnoredIncludePatterns = $script:AceLocalShutdownIgnoredIncludePatterns,
        [scriptblock]$AfterInitialCaptureHook = $null,
        [switch]$SimulateFailBeforeComplete
    )
    $out = Join-Path $SnapshotRoot $Label
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
        Set-Content -LiteralPath (Join-Path $gitDir 'diff-HEAD.patch') -Value (git diff HEAD | Out-String) -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $gitDir 'diff-staged.patch') -Value (git diff --cached | Out-String) -Encoding UTF8

        $entries = @(Get-AceLocalShutdownGitPorcelainEntries -RepoRoot $RepoRoot)
        $ignoredReview = @()
        $sourceHashAtStart = @{}

        foreach ($e in $entries) {
            Test-AceLocalShutdownRepoRelativePathResolvable -RepoRoot $RepoRoot -RelPath ([string]$e.RelPath) `
                -ChangeType ([string]$e.ChangeType)
            if ($e.ChangeType -eq 'deleted') { continue }
            $full = Join-Path $RepoRoot $e.RelPath
            if (Test-Path -LiteralPath $full -PathType Leaf) {
                $key = ($e.RelPath -replace '\\', '/').ToLowerInvariant()
                $sourceHashAtStart[$key] = Get-FileSha256Hex $full
            }
        }

        $ignoredPaths = @(Get-AceLocalShutdownGitIgnoredUntrackedPaths -RepoRoot $RepoRoot)
        foreach ($relUnix in $ignoredPaths) {
            $rel = $relUnix -replace '/', '\'
            Test-AceLocalShutdownRepoRelativePathResolvable -RepoRoot $RepoRoot -RelPath $rel -ChangeType 'ignored_included'
            $full = Join-Path $RepoRoot $rel
            if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
            $size = (Get-Item -LiteralPath $full).Length
            $ex = Test-AceLocalShutdownPathExcluded -RelativePath $rel -SizeBytes $size
            if ($ex.Excluded) {
                $ignoredReview += [PSCustomObject]@{
                    RelPath = $relUnix
                    Disposition = 'excluded_denylist'
                    Reason = $ex.Reason
                }
                continue
            }
            if (-not (Test-AceLocalShutdownIgnoredIncludeMatch -RelativePath $relUnix -IncludePatterns $IgnoredIncludePatterns)) {
                $ignoredReview += [PSCustomObject]@{
                    RelPath = $relUnix
                    Disposition = 'ignored_not_in_allowlist'
                    Reason = 'requires_user_policy_decision'
                }
                continue
            }
            $key = ($rel -replace '\\', '/').ToLowerInvariant()
            $sourceHashAtStart[$key] = Get-FileSha256Hex $full
            $entries += [PSCustomObject]@{
                XY = '!!'
                RelPath = $rel
                ChangeType = 'ignored_included'
            }
        }

        if ($AfterInitialCaptureHook) {
            & $AfterInitialCaptureHook
        }

        $concurrentDrift = @()
        foreach ($e in @($entries)) {
            if ($e.ChangeType -eq 'deleted') {
                $destDeleted = Join-Path $deletedDir $e.RelPath
                New-Item -ItemType Directory -Path (Split-Path -Parent $destDeleted) -Force | Out-Null
                $blob = git show "HEAD:$($e.RelPath -replace '\\','/')"
                Set-Content -LiteralPath $destDeleted -Value $blob -Encoding UTF8
                continue
            }
            $copy = Copy-AceLocalShutdownRepoFileToMirror -RepoRoot $RepoRoot -MirrorRoot $mirrorDir `
                -RelPath $e.RelPath -SourceHashAtStart $sourceHashAtStart
            if ($null -eq $copy) { continue }
            if ($copy.Skipped) {
                if ($e.ChangeType -eq 'ignored_included') {
                    $ignoredReview += [PSCustomObject]@{
                        RelPath = ($e.RelPath -replace '\\', '/')
                        Disposition = 'allowlist_but_skipped'
                        Reason = $copy.Reason
                    }
                }
                continue
            }
            if ($copy.ConcurrentDrift) {
                $concurrentDrift += [PSCustomObject]@{
                    RelPath = $e.RelPath
                    StartSha256 = $copy.StartSourceSha256
                    EndSha256 = $copy.SourceSha256
                }
            }
        }

        if ($SimulateFailBeforeComplete) {
            throw 'SHUTDOWN_SNAPSHOT_SIMULATED_FAIL'
        }

        return @{
            Label            = $Label
            Branch           = $branch
            Head             = $head
            Entries          = @($entries)
            IgnoredReview    = @($ignoredReview)
            ConcurrentDrift  = @($concurrentDrift)
            DestRoot         = $out
        }
    }
    finally {
        Pop-Location
    }
}

function Copy-AceLocalShutdownLocalStateReadOnly {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$SnapshotRoot,
        [Parameter(Mandatory = $true)][string]$SessionsRoot
    )
    $dest = Join-Path $SnapshotRoot 'local-state'
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    $copied = @()

    $candidates = @(
        @{ Name = 'managed-processes.json'; Path = Join-Path $AceRoot '_backups\local-dev\managed-processes.json' }
        @{ Name = 'active-dev-cycle.json'; Path = Join-Path $SessionsRoot 'active-dev-cycle.json' }
        @{ Name = 'production-registry.json'; Path = Join-Path $AceRoot '_registry\production-registry.json' }
        @{ Name = 'production-registry.json.sha256'; Path = Join-Path $AceRoot '_registry\production-registry.json.sha256' }
        @{ Name = 'publish-candidate-registry.json'; Path = Join-Path $AceRoot '_registry\publish-candidate-registry.json' }
        @{ Name = 'publish-candidate-registry.json.sha256'; Path = Join-Path $AceRoot '_registry\publish-candidate-registry.json.sha256' }
    )

    foreach ($item in $candidates) {
        if (-not (Test-Path -LiteralPath $item.Path)) { continue }
        $target = Join-Path $dest $item.Name
        Copy-Item -LiteralPath $item.Path -Destination $target -Force
        $copied += $item.Name
    }

    $openCyclePath = $null
    $activePath = Join-Path $SessionsRoot 'active-dev-cycle.json'
    if (Test-Path -LiteralPath $activePath) {
        try {
            $active = Get-Content -LiteralPath $activePath -Raw -Encoding UTF8 | ConvertFrom-Json
            $sid = [string]$active.sessionId
            if ($sid) {
                $cyclePath = Join-Path $SessionsRoot (Join-Path $sid 'cancel-dev-cycle.state.json')
                if (Test-Path -LiteralPath $cyclePath) {
                    $openCyclePath = Join-Path $dest 'cancel-dev-cycle.state.json'
                    Copy-Item -LiteralPath $cyclePath -Destination $openCyclePath -Force
                    $copied += 'cancel-dev-cycle.state.json'
                }
            }
        }
        catch {
            # read-only copy only; invalid JSON is recorded in metadata later
        }
    }

    return @{
        DestRoot = $dest
        Copied   = @($copied)
        ActiveCyclePath = $activePath
    }
}

function Export-AceLocalShutdownEnvSnapshots {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$SnapshotRoot
    )
    $envDir = Join-Path $SnapshotRoot 'ENV'
    New-Item -ItemType Directory -Path $envDir -Force | Out-Null
    $backendTemplate = @(Get-EnvKeyTemplateFromExample -ExamplePath (Join-Path $BackendRoot '.env.local.example'))
    if ($null -eq $backendTemplate) { $backendTemplate = @() }
    $frontendTemplate = @('VITE_BACKEND_URL')
    $snapshots = @(
        (Export-EnvFileSnapshot -SourcePath (Join-Path $BackendRoot '.env.local') `
            -DestDpapiPath (Join-Path $envDir 'backend.env.local.dpapi') -TemplateKeys $backendTemplate -LogicalName 'backend.env.local')
        (Export-EnvFileSnapshot -SourcePath (Join-Path $FrontendRoot '.env.development.local') `
            -DestDpapiPath (Join-Path $envDir 'frontend.env.development.local.dpapi') -TemplateKeys $frontendTemplate -LogicalName 'frontend.env.development.local')
    )
    $snapshots | Export-Csv -LiteralPath (Join-Path $envDir 'env-snapshot-index.csv') -NoTypeInformation -Encoding UTF8
    return @($snapshots)
}

function New-AceLocalShutdownSnapshotManifestRows {
    param(
        [Parameter(Mandatory = $true)]$BackendSnap,
        [Parameter(Mandatory = $true)]$FrontendSnap,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$SnapshotRoot,
        [Parameter(Mandatory = $true)][array]$EnvSnapshots,
        [Parameter(Mandatory = $true)][array]$LocalStateCopied
    )
    $rows = @()
    foreach ($block in @(
        @{ Snap = $BackendSnap; Label = 'BACKEND'; Root = $BackendRoot }
        @{ Snap = $FrontendSnap; Label = 'FRONTEND'; Root = $FrontendRoot }
    )) {
        foreach ($e in @($block.Snap.Entries)) {
            if ($e.ChangeType -eq 'deleted') {
                $bkp = Join-Path $SnapshotRoot "$($block.Label)\deleted-from-HEAD\$($e.RelPath)"
                $rows += [PSCustomObject]@{
                    Area = $block.Label
                    RelPath = $e.RelPath
                    ChangeType = 'deleted'
                    SourceSha256 = ''
                    BackupSha256 = (Get-FileSha256Hex $bkp)
                    SizeBytes = $(if (Test-Path -LiteralPath $bkp) { (Get-Item -LiteralPath $bkp).Length } else { 0 })
                    IgnoredIncluded = $false
                }
            }
            else {
                $full = Join-Path $block.Root $e.RelPath
                $bkp = Join-Path $SnapshotRoot "$($block.Label)\mirror\$($e.RelPath)"
                if (-not (Test-Path -LiteralPath $bkp)) { continue }
                $rows += [PSCustomObject]@{
                    Area = $block.Label
                    RelPath = $e.RelPath
                    ChangeType = $e.ChangeType
                    SourceSha256 = $(if (Test-Path -LiteralPath $full -PathType Leaf) { Get-FileSha256Hex $full } else { '' })
                    BackupSha256 = (Get-FileSha256Hex $bkp)
                    SizeBytes = (Get-Item -LiteralPath $bkp).Length
                    IgnoredIncluded = ($e.ChangeType -eq 'ignored_included')
                }
            }
        }
    }
    foreach ($snap in @($EnvSnapshots)) {
        $rows += [PSCustomObject]@{
            Area = 'ENV'
            RelPath = [string]$snap.LogicalName
            ChangeType = $(if ($snap.Present) { 'env-dpapi' } else { 'env-missing' })
            SourceSha256 = $(if ($snap.Present) { $snap.PlainSha256 } else { '' })
            BackupSha256 = $(if ($snap.Present) { (Get-FileSha256Hex $snap.DestDpapiPath) } else { '' })
            SizeBytes = [int]$snap.SizeBytes
            IgnoredIncluded = $false
        }
    }
    $localDir = Join-Path $SnapshotRoot 'local-state'
    foreach ($name in @($LocalStateCopied)) {
        $path = Join-Path $localDir $name
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $rows += [PSCustomObject]@{
            Area = 'LOCAL_STATE'
            RelPath = $name
            ChangeType = 'readonly_copy'
            SourceSha256 = (Get-FileSha256Hex $path)
            BackupSha256 = (Get-FileSha256Hex $path)
            SizeBytes = (Get-Item -LiteralPath $path).Length
            IgnoredIncluded = $false
        }
    }
    return @($rows)
}

function Write-AceLocalShutdownSnapshotManifest {
    param(
        [Parameter(Mandatory = $true)][string]$SnapshotRoot,
        [Parameter(Mandatory = $true)][array]$Rows
    )
    $path = Join-Path $SnapshotRoot 'MANIFEST.csv'
    $Rows | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
    return $path
}

function Test-AceLocalShutdownSnapshotIntegrity {
    param(
        [Parameter(Mandatory = $true)][string]$SnapshotRoot,
        [switch]$SimulateManifestHashMismatch
    )
    $errors = @()
    $manifestPath = Join-Path $SnapshotRoot 'MANIFEST.csv'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        return @{ Ok = $false; Errors = @('MANIFEST_MISSING'); Partial = $true }
    }
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
                    $full = Join-Path $SnapshotRoot "BACKEND\deleted-from-HEAD\$rel"
                }
                else {
                    $full = Join-Path $SnapshotRoot "BACKEND\mirror\$rel"
                }
            }
            'FRONTEND' {
                if ([string]$row.ChangeType -eq 'deleted') {
                    $full = Join-Path $SnapshotRoot "FRONTEND\deleted-from-HEAD\$rel"
                }
                else {
                    $full = Join-Path $SnapshotRoot "FRONTEND\mirror\$rel"
                }
            }
            'ENV' {
                if ([string]$row.ChangeType -eq 'env-dpapi') {
                    $full = Join-Path $SnapshotRoot "ENV\$rel.dpapi"
                }
            }
            'LOCAL_STATE' {
                $full = Join-Path $SnapshotRoot "local-state\$rel"
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
    foreach ($label in @('BACKEND', 'FRONTEND')) {
        $headFile = Join-Path $SnapshotRoot "$label\git\HEAD.txt"
        if (-not (Test-Path -LiteralPath $headFile)) {
            $errors += "MISSING_${label}_HEAD"
        }
    }
    $metaPath = Join-Path $SnapshotRoot 'snapshot-metadata.json'
    if (-not (Test-Path -LiteralPath $metaPath)) {
        $errors += 'SNAPSHOT_METADATA_MISSING'
    }
    return @{
        Ok      = ($errors.Count -eq 0)
        Errors  = @($errors)
        Partial = ($errors.Count -gt 0)
    }
}

function Write-AceLocalShutdownRecoveryGuide {
    param(
        [Parameter(Mandatory = $true)][string]$SnapshotRoot,
        [Parameter(Mandatory = $true)][string]$Status
    )
    $lines = @(
        '# ACE LOCAL shutdown snapshot recovery (manual only)'
        ''
        "Status: $Status"
        "SnapshotRoot: $SnapshotRoot"
        ''
        'This snapshot is read-only. Do not auto-restore onto live ACE repos.'
        ''
        '1. Verify snapshot-metadata.json status is VERIFIED.'
        '2. Review MANIFEST.csv hashes via Test-AceLocalShutdownSnapshotIntegrity.'
        '3. Restore git state on a TEMP clone: checkout branch/HEAD from */git/*.txt, copy */mirror/*, restore deleted-from-HEAD as needed.'
        '4. ENV: decrypt *.dpapi with Read-AceLocalShutdownEnvPlain (same user) into a temp folder only.'
        '5. LOCAL_STATE copies are reference only — do not overwrite production registry or active cycle on live ACE.'
    )
    Set-Content -LiteralPath (Join-Path $SnapshotRoot 'RECOVERY.md') -Value ($lines -join "`n") -Encoding UTF8
}

function Read-AceLocalShutdownEnvPlain {
    param([Parameter(Mandatory = $true)][string]$DpapiPath)
    if (-not (Test-Path -LiteralPath $DpapiPath)) { return $null }
    $bytes = [System.IO.File]::ReadAllBytes($DpapiPath)
    return [System.Text.Encoding]::UTF8.GetString((Unprotect-ContentForCurrentUser -ProtectedBytes $bytes))
}

function Restore-AceLocalShutdownRepoToTempRoot {
    param(
        [Parameter(Mandatory = $true)][string]$SnapshotRoot,
        [Parameter(Mandatory = $true)][ValidateSet('BACKEND', 'FRONTEND')][string]$Label,
        [Parameter(Mandatory = $true)][string]$TargetRepoRoot
    )
    $src = Join-Path $SnapshotRoot $Label
    $gitDir = Join-Path $src 'git'
    if (-not (Test-Path -LiteralPath $gitDir)) {
        throw "SHUTDOWN_RESTORE_GIT_META_MISSING:$Label"
    }
    $branch = (Get-Content -LiteralPath (Join-Path $gitDir 'branch.txt') -Raw).Trim()
    $head = (Get-Content -LiteralPath (Join-Path $gitDir 'HEAD.txt') -Raw).Trim()
    Push-Location -LiteralPath $TargetRepoRoot
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try {
        git checkout -f $branch 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "SHUTDOWN_RESTORE_CHECKOUT_FAILED:$Label"
        }
        git reset --hard $head 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "SHUTDOWN_RESTORE_RESET_FAILED:$Label"
        }
    }
    finally {
        $ErrorActionPreference = $prevEap
        Pop-Location
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
    return @{ Branch = $branch; Head = $head }
}

function New-AceLocalShutdownSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [string]$SessionsRoot = '',
        [string]$SnapshotsRootOverride = '',
        [AllowEmptyCollection()][string[]]$IgnoredIncludePatterns = $script:AceLocalShutdownIgnoredIncludePatterns,
        [scriptblock]$DuringBackendCopyHook = $null,
        [switch]$SimulateFailAfterBackend,
        [switch]$SimulateFailBeforeVerify,
        [switch]$DeferLastVerifiedPointerUpdate
    )
    $BackendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
    $FrontendRoot = (Resolve-Path -LiteralPath $FrontendRoot).ProviderPath
    $AceRoot = (Resolve-Path -LiteralPath $AceRoot).ProviderPath
    if ([string]::IsNullOrWhiteSpace($SessionsRoot)) {
        $SessionsRoot = Join-Path $AceRoot '_backups\sessions'
    }
    else {
        $SessionsRoot = (Resolve-Path -LiteralPath $SessionsRoot).ProviderPath
    }

    $snapshotsRoot = if ($SnapshotsRootOverride) {
        $SnapshotsRootOverride
    }
    else {
        Get-AceLocalShutdownSnapshotsRoot -AceRoot $AceRoot
    }
    New-Item -ItemType Directory -Path $snapshotsRoot -Force | Out-Null

    $snapshotId = '{0}-{1}' -f (Get-Date -Format 'yyyy-MM-dd_HHmmss'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
    $snapshotRoot = Join-Path $snapshotsRoot $snapshotId
    New-Item -ItemType Directory -Path $snapshotRoot -Force | Out-Null

    $metadataPath = Join-Path $snapshotRoot 'snapshot-metadata.json'
    $metadata = [ordered]@{
        schemaVersion = $script:AceLocalShutdownSnapshotSchemaVersion
        snapshotId    = $snapshotId
        status        = 'INCOMPLETE'
        startedAt     = (Get-Date).ToUniversalTime().ToString('o')
        finishedAt    = $null
        verifiedAt    = $null
        backendRoot   = $BackendRoot
        frontendRoot  = $FrontendRoot
        aceRoot       = $AceRoot
        sessionsRoot  = $SessionsRoot
        errors        = @()
        concurrentDrift = @()
        ignoredUserReview = @()
        integrity     = $null
    }
    Write-AceLocalShutdownJsonAtomic -Path $metadataPath -Object $metadata

    $result = @{
        Ok           = $false
        SnapshotId   = $snapshotId
        SnapshotRoot = $snapshotRoot
        Status       = 'INCOMPLETE'
        Errors       = @()
    }

    try {
        $localState = Copy-AceLocalShutdownLocalStateReadOnly -AceRoot $AceRoot -SnapshotRoot $snapshotRoot -SessionsRoot $SessionsRoot

        $backendSnap = Export-AceLocalShutdownRepoSnapshot -RepoRoot $BackendRoot -Label 'BACKEND' `
            -SnapshotRoot $snapshotRoot -IgnoredIncludePatterns $IgnoredIncludePatterns `
            -AfterInitialCaptureHook $DuringBackendCopyHook
        if ($SimulateFailAfterBackend) {
            throw 'SHUTDOWN_SNAPSHOT_SIMULATED_FAIL_AFTER_BACKEND'
        }

        $frontendSnap = Export-AceLocalShutdownRepoSnapshot -RepoRoot $FrontendRoot -Label 'FRONTEND' `
            -SnapshotRoot $snapshotRoot -IgnoredIncludePatterns $IgnoredIncludePatterns

        $envSnaps = @(Export-AceLocalShutdownEnvSnapshots -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -SnapshotRoot $snapshotRoot)

        $manifestRows = New-AceLocalShutdownSnapshotManifestRows -BackendSnap $backendSnap -FrontendSnap $frontendSnap `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -SnapshotRoot $snapshotRoot `
            -EnvSnapshots $envSnaps -LocalStateCopied $localState.Copied
        Write-AceLocalShutdownSnapshotManifest -SnapshotRoot $snapshotRoot -Rows $manifestRows | Out-Null

        $drift = @($backendSnap.ConcurrentDrift) + @($frontendSnap.ConcurrentDrift)
        $review = @($backendSnap.IgnoredReview) + @($frontendSnap.IgnoredReview)
        if ($drift.Count -gt 0) {
            $metadata.concurrentDrift = @($drift)
            throw ('SHUTDOWN_SNAPSHOT_CONCURRENT_DRIFT:' + ($drift | ForEach-Object { $_.RelPath } | Select-Object -First 5 -Join ','))
        }

        if ($SimulateFailBeforeVerify) {
            throw 'SHUTDOWN_SNAPSHOT_SIMULATED_FAIL_BEFORE_VERIFY'
        }

        $integrity = Test-AceLocalShutdownSnapshotIntegrity -SnapshotRoot $snapshotRoot
        $metadata.integrity = $integrity
        if (-not $integrity.Ok) {
            throw ('SHUTDOWN_SNAPSHOT_INTEGRITY_FAILED:' + ($integrity.Errors -join ','))
        }

        $metadata.status = 'VERIFIED'
        $metadata.verifiedAt = (Get-Date).ToUniversalTime().ToString('o')
        $metadata.finishedAt = $metadata.verifiedAt
        $metadata.ignoredUserReview = @($review | ForEach-Object {
            @{ relPath = $_.RelPath; disposition = $_.Disposition; reason = $_.Reason }
        })
        Write-AceLocalShutdownRecoveryGuide -SnapshotRoot $snapshotRoot -Status 'VERIFIED'
        Write-AceLocalShutdownJsonAtomic -Path $metadataPath -Object $metadata

        if (-not $DeferLastVerifiedPointerUpdate) {
            $pointer = [ordered]@{
                schemaVersion = 1
                snapshotId    = $snapshotId
                snapshotRoot  = $snapshotRoot
                verifiedAt    = $metadata.verifiedAt
            }
            Write-AceLocalShutdownJsonAtomic -Path (Get-AceLocalShutdownLastVerifiedPointerPath -AceRoot $AceRoot) -Object $pointer
        }

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
            $metadata.integrity = Test-AceLocalShutdownSnapshotIntegrity -SnapshotRoot $snapshotRoot
        }
        Write-AceLocalShutdownRecoveryGuide -SnapshotRoot $snapshotRoot -Status 'INCOMPLETE'
        Write-AceLocalShutdownJsonAtomic -Path $metadataPath -Object $metadata
        return $result
    }
}

function Get-AceLocalShutdownLastVerifiedSnapshot {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $path = Get-AceLocalShutdownLastVerifiedPointerPath -AceRoot $AceRoot
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try {
        return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        return $null
    }
}
