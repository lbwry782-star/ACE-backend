# ACE LOCAL — read-only workspace dirty detection for recovery (no copy, no git mutation).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')

function Get-AceLocalRecoveryRepoDirtyScan {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][ValidateSet('BACKEND', 'FRONTEND')][string]$Label,
        [AllowEmptyCollection()][string[]]$IgnoredIncludePatterns = $script:AceLocalShutdownIgnoredIncludePatterns
    )
    $repoRoot = (Resolve-Path -LiteralPath $RepoRoot).ProviderPath
    Push-Location -LiteralPath $repoRoot
    try {
        $branch = (git rev-parse --abbrev-ref HEAD 2>$null).Trim()
        $head = (git rev-parse HEAD 2>$null).Trim()
    }
    finally {
        Pop-Location
    }

    $entries = @(Get-AceLocalShutdownGitPorcelainEntries -RepoRoot $repoRoot)
    $ignoredIncluded = @()
    $ignoredExcluded = @()

    foreach ($e in @($entries)) {
        Test-AceLocalShutdownRepoRelativePathResolvable -RepoRoot $repoRoot -RelPath ([string]$e.RelPath) `
            -ChangeType ([string]$e.ChangeType)
    }

    foreach ($relUnix in @(Get-AceLocalShutdownGitIgnoredUntrackedPaths -RepoRoot $repoRoot)) {
        $rel = $relUnix -replace '/', '\'
        Test-AceLocalShutdownRepoRelativePathResolvable -RepoRoot $repoRoot -RelPath $rel -ChangeType 'ignored_included'
        $full = Join-Path $repoRoot $rel
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            continue
        }
        $size = (Get-Item -LiteralPath $full).Length
        $ex = Test-AceLocalShutdownPathExcluded -RelativePath $rel -SizeBytes $size
        if ($ex.Excluded) {
            $ignoredExcluded += [PSCustomObject]@{
                RelPath = $relUnix
                Reason  = $ex.Reason
            }
            continue
        }
        if (Test-AceLocalShutdownIgnoredIncludeMatch -RelativePath $relUnix -IncludePatterns $IgnoredIncludePatterns) {
            $ignoredIncluded += [PSCustomObject]@{
                RelPath    = $relUnix
                ChangeType = 'ignored_included'
            }
            $entries += [PSCustomObject]@{
                XY         = '!!'
                RelPath    = $rel
                ChangeType = 'ignored_included'
            }
        }
        else {
            $ignoredExcluded += [PSCustomObject]@{
                RelPath = $relUnix
                Reason  = 'ignored_not_in_allowlist'
            }
        }
    }

    $contentSha256 = @{}
    foreach ($e in @($entries)) {
        if ([string]$e.ChangeType -eq 'deleted') {
            continue
        }
        $rel = [string]$e.RelPath
        $full = Join-Path $repoRoot $rel
        try {
            if (Test-Path -LiteralPath $full -PathType Leaf) {
                $key = ($rel -replace '\\', '/').ToLowerInvariant()
                $contentSha256[$key] = Get-FileSha256Hex $full
            }
        }
        catch {
            throw ("RECOVERY_DIRTY_SCAN_HASH_PATH_FAILED:{0}:{1}" -f $rel, $_.Exception.Message)
        }
    }

    $normalized = @($entries | ForEach-Object {
        $relUnix = ($_.RelPath -replace '\\', '/')
        $key = $relUnix.ToLowerInvariant()
        $row = [ordered]@{
            path       = $relUnix
            changeType = [string]$_.ChangeType
        }
        if ($contentSha256.ContainsKey($key)) {
            $row.contentSha256 = [string]$contentSha256[$key]
        }
        $row
    } | Sort-Object { $_.path })

    return [ordered]@{
        Label            = $Label
        Branch           = $branch
        Head             = $head
        ScanOk           = $true
        ScanErrors       = @()
        Entries          = @($normalized)
        ContentSha256    = $contentSha256
        IgnoredIncluded  = @($ignoredIncluded)
        IgnoredExcluded  = @($ignoredExcluded)
    }
}

function Get-AceLocalRecoveryWorkspaceDirtyScan {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [AllowEmptyCollection()][string[]]$IgnoredIncludePatterns = $script:AceLocalShutdownIgnoredIncludePatterns
    )
    try {
        $backend = Get-AceLocalRecoveryRepoDirtyScan -RepoRoot $BackendRoot -Label 'BACKEND' -IgnoredIncludePatterns $IgnoredIncludePatterns
        $frontend = Get-AceLocalRecoveryRepoDirtyScan -RepoRoot $FrontendRoot -Label 'FRONTEND' -IgnoredIncludePatterns $IgnoredIncludePatterns
    }
    catch {
        throw ("RECOVERY_DIRTY_SCAN_FAILED:{0}" -f $_.Exception.Message)
    }
    return @{
        ScanOk   = $true
        Backend  = $backend
        Frontend = $frontend
    }
}

function Get-AceLocalRecoveryDirtyFingerprint {
    param([Parameter(Mandatory = $true)]$WorkspaceScan)
    $payload = @{
        backend  = @{
            head          = [string]$WorkspaceScan.Backend.Head
            branch        = [string]$WorkspaceScan.Backend.Branch
            entries       = @($WorkspaceScan.Backend.Entries)
            contentSha256 = @{}
        }
        frontend = @{
            head          = [string]$WorkspaceScan.Frontend.Head
            branch        = [string]$WorkspaceScan.Frontend.Branch
            entries       = @($WorkspaceScan.Frontend.Entries)
            contentSha256 = @{}
        }
    }
    if ($WorkspaceScan.Backend.ContentSha256) {
        $payload.backend.contentSha256 = @{}
        foreach ($k in @($WorkspaceScan.Backend.ContentSha256.Keys | Sort-Object)) {
            $payload.backend.contentSha256[$k] = [string]$WorkspaceScan.Backend.ContentSha256[$k]
        }
    }
    if ($WorkspaceScan.Frontend.ContentSha256) {
        $payload.frontend.contentSha256 = @{}
        foreach ($k in @($WorkspaceScan.Frontend.ContentSha256.Keys | Sort-Object)) {
            $payload.frontend.contentSha256[$k] = [string]$WorkspaceScan.Frontend.ContentSha256[$k]
        }
    }
    $json = ($payload | ConvertTo-Json -Depth 10 -Compress)
    return (Get-TextSha256Hex $json)
}

function Test-AceLocalRecoveryWorkspaceContentChanged {
    param(
        [Parameter(Mandatory = $true)]$PreviousScan,
        [Parameter(Mandatory = $true)]$CurrentScan
    )
    $prevFp = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $PreviousScan
    $curFp = Get-AceLocalRecoveryDirtyFingerprint -WorkspaceScan $CurrentScan
    return (Test-AceLocalRecoveryWorkspaceDirtyChanged -PreviousFingerprint $prevFp -CurrentFingerprint $curFp)
}

function Test-AceLocalRecoveryWorkspaceDirtyChanged {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$PreviousFingerprint,
        [Parameter(Mandatory = $true)][string]$CurrentFingerprint
    )
    $prev = ($PreviousFingerprint -replace '\s', '').Trim().ToLowerInvariant()
    $cur = ($CurrentFingerprint -replace '\s', '').Trim().ToLowerInvariant()
    if (-not $prev) {
        return $true
    }
    return ($prev -ne $cur)
}

function Test-AceLocalRecoveryWorkspaceHasWorkingTreeChanges {
    param([Parameter(Mandatory = $true)]$WorkspaceScan)
    $be = @($WorkspaceScan.Backend.Entries)
    $fe = @($WorkspaceScan.Frontend.Entries)
    return (($be.Count + $fe.Count) -gt 0)
}
