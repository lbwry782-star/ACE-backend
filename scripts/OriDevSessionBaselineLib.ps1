# Ori local dev session baseline — shared helpers (no side effects on dot-source).
Set-StrictMode -Version Latest

function Write-BaselineInfo([string]$Message) { Write-Host $Message -ForegroundColor Cyan }
function Write-BaselineOk([string]$Message) { Write-Host $Message -ForegroundColor Green }
function Write-BaselineWarn([string]$Message) { Write-Host $Message -ForegroundColor Yellow }
function Write-BaselineErr([string]$Message) { Write-Host $Message -ForegroundColor Red }

function Get-FileSha256Hex([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256Hex([string]$Text) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant())
    }
    finally {
        $sha.Dispose()
    }
}

function New-OriSessionId {
    $ts = Get-Date -Format 'yyyyMMdd-HHmmss'
    $suffix = ([Guid]::NewGuid().ToString('N')).Substring(0, 8)
    return "ori-$ts-$suffix"
}

function Test-ProductionRegistryIntegrity {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryPath,
        [Parameter(Mandatory = $true)][string]$Sha256SidecarPath
    )
    $errors = @()
    if (-not (Test-Path -LiteralPath $RegistryPath)) {
        return @{ Ok = $false; Errors = @('REGISTRY_MISSING'); Registry = $null }
    }
    if (-not (Test-Path -LiteralPath $Sha256SidecarPath)) {
        return @{ Ok = $false; Errors = @('REGISTRY_SHA256_SIDECAR_MISSING'); Registry = $null }
    }
    $expectedLine = (Get-Content -LiteralPath $Sha256SidecarPath -Raw).Trim()
    $expectedHash = ($expectedLine -split '\s+')[0].ToLowerInvariant()
    $actualHash = Get-FileSha256Hex $RegistryPath
    if ($expectedHash -ne $actualHash) {
        $errors += 'REGISTRY_SHA256_MISMATCH'
    }
    try {
        $registry = Get-Content -LiteralPath $RegistryPath -Raw | ConvertFrom-Json
    }
    catch {
        return @{ Ok = $false; Errors = @('REGISTRY_JSON_INVALID'); Registry = $null }
    }
    if ($registry.schemaVersion -ne 1) {
        $errors += 'REGISTRY_SCHEMA_UNSUPPORTED'
    }
    if ($errors.Count -gt 0) {
        return @{ Ok = $false; Errors = $errors; Registry = $registry }
    }
    return @{ Ok = $true; Errors = @(); Registry = $registry }
}

function Get-RegistryCommitShas {
    param([Parameter(Mandatory = $true)]$Registry)
    $list = @()
    $prod = $Registry.production
    if ($prod.backendWeb.deployedCommitSha) { $list += @{ Repo = 'backend'; Role = 'productionWeb'; Sha = [string]$prod.backendWeb.deployedCommitSha } }
    if ($prod.backendWorker.deployedCommitSha) {
        $sha = [string]$prod.backendWorker.deployedCommitSha
        if (-not ($list | Where-Object { $_.Repo -eq 'backend' -and $_.Sha -eq $sha })) {
            $list += @{ Repo = 'backend'; Role = 'productionWorker'; Sha = $sha }
        }
    }
    if ($prod.frontend.deployedCommitSha) { $list += @{ Repo = 'frontend'; Role = 'production'; Sha = [string]$prod.frontend.deployedCommitSha } }
    $infra = $Registry.localDevInfra
    if ($infra.backend.commitSha) { $list += @{ Repo = 'backend'; Role = 'localDevInfra'; Sha = [string]$infra.backend.commitSha } }
    if ($infra.frontend.commitSha) { $list += @{ Repo = 'frontend'; Role = 'localDevInfra'; Sha = [string]$infra.frontend.commitSha } }
    if ($Registry.wip.builder1.commitSha) { $list += @{ Repo = 'backend'; Role = 'builder1Wip'; Sha = [string]$Registry.wip.builder1.commitSha } }
    return $list
}

function Test-RegistryCommitsExist {
    param(
        [Parameter(Mandatory = $true)]$Registry,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot
    )
    $errors = @()
    foreach ($entry in (Get-RegistryCommitShas -Registry $Registry)) {
        $root = if ($entry.Repo -eq 'backend') { $BackendRoot } else { $FrontendRoot }
        Push-Location -LiteralPath $root
        try {
            git rev-parse --verify "$($entry.Sha)^{commit}" 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $errors += "COMMIT_MISSING:$($entry.Role):$($entry.Sha)"
            }
        }
        finally {
            Pop-Location
        }
    }
    return @{ Ok = ($errors.Count -eq 0); Errors = $errors }
}

function Protect-ContentForCurrentUser {
    param([Parameter(Mandatory = $true)][byte[]]$PlainBytes)
    Add-Type -AssemblyName System.Security
    return [System.Security.Cryptography.ProtectedData]::Protect(
        $PlainBytes,
        $null,
        [System.Security.Cryptography.DataProtectionScope]::CurrentUser
    )
}

function Unprotect-ContentForCurrentUser {
    param([Parameter(Mandatory = $true)][byte[]]$ProtectedBytes)
    Add-Type -AssemblyName System.Security
    return [System.Security.Cryptography.ProtectedData]::Unprotect(
        $ProtectedBytes,
        $null,
        [System.Security.Cryptography.DataProtectionScope]::CurrentUser
    )
}

function Set-DirectoryCurrentUserAcl {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    icacls $Path /inheritance:r /grant:r "${user}:(OI)(CI)F" /Q 2>&1 | Out-Null
    Get-ChildItem -LiteralPath $Path -File -ErrorAction SilentlyContinue | ForEach-Object {
        icacls $_.FullName /inheritance:r /grant:r "${user}:F" /Q 2>&1 | Out-Null
    }
}

function Parse-DotEnvKeys {
    param([Parameter(Mandatory = $true)][string]$Content)
    $keys = [ordered]@{}
    foreach ($line in ($Content -split "`n")) {
        $t = $line.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        $eq = $t.IndexOf('=')
        if ($eq -lt 1) { continue }
        $name = $t.Substring(0, $eq).Trim()
        $value = $t.Substring($eq + 1).Trim()
        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $keys[$name] = $value
    }
    return $keys
}

function Get-EnvKeyTemplateFromExample {
    param([Parameter(Mandatory = $true)][string]$ExamplePath)
    if (-not (Test-Path -LiteralPath $ExamplePath)) {
        return @()
    }
    $content = Get-Content -LiteralPath $ExamplePath -Raw -Encoding UTF8
    $parsed = Parse-DotEnvKeys -Content $content
    return @($parsed.Keys)
}

function Build-EnvKeyManifest {
    param(
        [Parameter(Mandatory = $true)][hashtable]$KeyValues,
        [AllowEmptyCollection()][string[]]$TemplateKeys = @(),
        [Parameter(Mandatory = $true)][string]$LogicalName
    )
    $rows = @()
    foreach ($name in ($KeyValues.Keys | Sort-Object)) {
        $value = [string]$KeyValues[$name]
        $classification = if ($TemplateKeys -contains $name) { 'infraTemplate' } else { 'sessionOrCustom' }
        $rows += [PSCustomObject]@{
            logicalName = $LogicalName
            key         = $name
            valueSha256 = (Get-TextSha256Hex $value)
            classification = $classification
        }
    }
    foreach ($name in ($TemplateKeys | Sort-Object)) {
        if (-not $KeyValues.ContainsKey($name)) {
            $rows += [PSCustomObject]@{
                logicalName = $LogicalName
                key         = $name
                valueSha256 = ''
                classification = 'infraTemplateMissingAtBaseline'
            }
        }
    }
    return $rows
}

function Parse-GitStatusLine {
    param([string]$Line)
    if ([string]::IsNullOrWhiteSpace($Line)) { return $null }
    $xy = $Line.Substring(0, 2)
    $pathPart = $Line.Substring(3).Trim()
    if ($pathPart -match '^"(.+)"$') { $pathPart = $Matches[1] }
    elseif ($pathPart -match ' -> ') { $pathPart = ($pathPart -split ' -> ', 2)[-1].Trim('"') }
    $rel = $pathPart -replace '/', '\'
    $changeType = switch -Regex ($xy) {
        '^\?\?' { 'untracked' }
        '^ D|^D ' { 'deleted' }
        default { 'modified' }
    }
    [PSCustomObject]@{ XY = $xy; RelPath = $rel; ChangeType = $changeType }
}

function Export-GitWorkingTreeSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$SessionRoot
    )
    $out = Join-Path $SessionRoot $Label
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
        return @{
            Label   = $Label
            Branch  = $branch
            Head    = $head
            Entries = $entries
        }
    }
    finally {
        Pop-Location
    }
}

function Export-EnvFileSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestDpapiPath,
        [AllowEmptyCollection()][string[]]$TemplateKeys = @(),
        [Parameter(Mandatory = $true)][string]$LogicalName
    )
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        return @{
            Present = $false
            LogicalName = $LogicalName
            SourcePath = $SourcePath
            DestDpapiPath = $DestDpapiPath
            KeyManifest = @()
            SizeBytes = 0
        }
    }
    $raw = [System.IO.File]::ReadAllText($SourcePath, [System.Text.Encoding]::UTF8)
    $protected = Protect-ContentForCurrentUser -PlainBytes ([System.Text.Encoding]::UTF8.GetBytes($raw))
    [System.IO.File]::WriteAllBytes($DestDpapiPath, $protected)
    $keys = Parse-DotEnvKeys -Content $raw
    $manifest = Build-EnvKeyManifest -KeyValues $keys -TemplateKeys $TemplateKeys -LogicalName $LogicalName
    return @{
        Present = $true
        LogicalName = $LogicalName
        SourcePath = $SourcePath
        DestDpapiPath = $DestDpapiPath
        PlainSha256 = (Get-TextSha256Hex $raw)
        KeyManifest = $manifest
        SizeBytes = (Get-Item -LiteralPath $SourcePath).Length
    }
}

function Test-SessionSnapshotIntegrity {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)]$SessionMeta
    )
    $errors = @()
    $sessionFile = Join-Path $SessionRoot 'session.json'
    if (-not (Test-Path -LiteralPath $sessionFile)) {
        $errors += 'SESSION_JSON_MISSING'
        return @{ Ok = $false; Errors = $errors }
    }
    foreach ($block in @('BACKEND', 'FRONTEND')) {
        $headFile = Join-Path $SessionRoot "$block\git\HEAD.txt"
        if (-not (Test-Path -LiteralPath $headFile)) { $errors += "MISSING_${block}_HEAD" }
    }
    $manifestPath = Join-Path $SessionRoot 'MANIFEST.csv'
    if (-not (Test-Path -LiteralPath $manifestPath)) { $errors += 'MANIFEST_MISSING' }

    foreach ($envItem in $SessionMeta.envSnapshots) {
        if (-not $envItem.Present) { continue }
        if (-not (Test-Path -LiteralPath $envItem.DestDpapiPath)) {
            $errors += "ENV_DPAPI_MISSING:$($envItem.LogicalName)"
            continue
        }
        try {
            $bytes = [System.IO.File]::ReadAllBytes($envItem.DestDpapiPath)
            $plain = [System.Text.Encoding]::UTF8.GetString((Unprotect-ContentForCurrentUser -ProtectedBytes $bytes))
            $round = Get-TextSha256Hex $plain
            if ($round -ne $envItem.PlainSha256) {
                $errors += "ENV_DPAPI_ROUNDTRIP_FAIL:$($envItem.LogicalName)"
            }
        }
        catch {
            $errors += "ENV_DPAPI_UNPROTECT_FAIL:$($envItem.LogicalName)"
        }
    }

    foreach ($block in @(
        @{ Label = 'BACKEND'; RootKey = 'backend' }
        @{ Label = 'FRONTEND'; RootKey = 'frontend' }
    )) {
        $snap = $SessionMeta.git.$($block.RootKey)
        if (-not $snap) { continue }
        foreach ($e in $snap.Entries) {
            $repoRoot = if ($block.Label -eq 'BACKEND') { $SessionMeta.paths.backendRoot } else { $SessionMeta.paths.frontendRoot }
            $full = Join-Path $repoRoot $e.RelPath
            if ($e.ChangeType -eq 'deleted') {
                $bkp = Join-Path $SessionRoot "$($block.Label)\deleted-from-HEAD\$($e.RelPath)"
                if (-not (Test-Path -LiteralPath $bkp)) { $errors += "MISSING_DELETED:$($block.Label):$($e.RelPath)" }
            }
            else {
                if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { continue }
                $bkp = Join-Path $SessionRoot "$($block.Label)\mirror\$($e.RelPath)"
                if (-not (Test-Path -LiteralPath $bkp)) { $errors += "MISSING_MIRROR:$($block.Label):$($e.RelPath)" }
                elseif ((Get-FileSha256Hex $full) -ne (Get-FileSha256Hex $bkp)) {
                    $errors += "HASH_MISMATCH:$($block.Label):$($e.RelPath)"
                }
            }
        }
    }
    return @{ Ok = ($errors.Count -eq 0); Errors = $errors }
}

function Write-SessionManifest {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)][array]$Rows
    )
    $path = Join-Path $SessionRoot 'MANIFEST.csv'
    $Rows | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
}

function Save-SessionState {
    param(
        [Parameter(Mandatory = $true)][string]$SessionRoot,
        [Parameter(Mandatory = $true)][hashtable]$State
    )
    ($State | ConvertTo-Json -Depth 12) | Set-Content -LiteralPath (Join-Path $SessionRoot 'session.json') -Encoding UTF8
}

function Resolve-OriLiteralPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    if (Test-Path -LiteralPath $Path) {
        return (Resolve-Path -LiteralPath $Path).ProviderPath
    }
    return $Path
}

function Resolve-OriProjectLayout {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptsDirectory,
        [string]$BackendRoot = '',
        [string]$FrontendRoot = '',
        [string]$RegistryPath = '',
        [string]$SessionsRoot = ''
    )
    $backendCandidate = if ([string]::IsNullOrWhiteSpace($BackendRoot)) {
        Split-Path -Parent $ScriptsDirectory
    }
    else {
        $BackendRoot
    }
    $backendCandidate = Resolve-OriLiteralPath $backendCandidate
    if (-not (Test-Path -LiteralPath (Join-Path $backendCandidate 'app.py'))) {
        throw 'ACE_BACKEND_LAYOUT_NOT_FOUND'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $backendCandidate 'worker_video.py'))) {
        throw 'ACE_BACKEND_LAYOUT_NOT_FOUND'
    }

    $aceRoot = Split-Path -Parent $backendCandidate
    $registryCandidate = if ([string]::IsNullOrWhiteSpace($RegistryPath)) {
        Join-Path $aceRoot '_registry\production-registry.json'
    }
    else {
        $RegistryPath
    }
    $sessionsCandidate = if ([string]::IsNullOrWhiteSpace($SessionsRoot)) {
        Join-Path $aceRoot '_backups\sessions'
    }
    else {
        $SessionsRoot
    }
    $frontendCandidate = if ([string]::IsNullOrWhiteSpace($FrontendRoot)) {
        Join-Path $aceRoot 'ACE-Frontend'
    }
    else {
        $FrontendRoot
    }
    if (-not (Test-Path -LiteralPath (Join-Path $frontendCandidate 'package.json'))) {
        throw 'ACE_FRONTEND_LAYOUT_NOT_FOUND'
    }
    $frontendCandidate = Resolve-OriLiteralPath $frontendCandidate

    return @{
        AceRoot       = $aceRoot
        BackendRoot   = $backendCandidate
        FrontendRoot  = $frontendCandidate
        RegistryPath  = $registryCandidate
        SessionsRoot  = $sessionsCandidate
    }
}

function Write-ProductionRegistrySha256Sidecar {
    param([Parameter(Mandatory = $true)][string]$RegistryPath)
    $hash = Get-FileSha256Hex $RegistryPath
    $sidecar = "$RegistryPath.sha256"
    [System.IO.File]::WriteAllText($sidecar, "$hash  production-registry.json", [System.Text.UTF8Encoding]::new($false))
    return $hash
}

function Update-OriProductionRegistryBackendInfraCommit {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryPath,
        [Parameter(Mandatory = $true)][string]$NewCommitSha,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [string]$BackupRoot = ''
    )
    $regCheck = Test-ProductionRegistryIntegrity -RegistryPath $RegistryPath -Sha256SidecarPath "$RegistryPath.sha256"
    if (-not $regCheck.Ok) {
        throw ($regCheck.Errors -join ',')
    }
    $registry = $regCheck.Registry
    $oldSha = [string]$registry.localDevInfra.backend.commitSha
    if ($oldSha -eq $NewCommitSha) {
        return @{ Updated = $false; OldSha = $oldSha; NewSha = $NewCommitSha; RegistryHash = (Get-FileSha256Hex $RegistryPath) }
    }

    Push-Location -LiteralPath $BackendRoot
    try {
        git rev-parse --verify "$NewCommitSha^{commit}" 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw 'NEW_INFRA_COMMIT_NOT_FOUND'
        }
        git merge-base --is-ancestor $oldSha $NewCommitSha 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw 'NEW_INFRA_COMMIT_NOT_DESCENDANT_OF_REGISTRY'
        }
    }
    finally {
        Pop-Location
    }

    if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
        $aceRoot = Split-Path -Parent (Split-Path -Parent $RegistryPath)
        $BackupRoot = Join-Path $aceRoot '_backups\registry-updates'
    }
    New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
    $ts = Get-Date -Format 'yyyy-MM-dd_HHmmss'
    Copy-Item -LiteralPath $RegistryPath -Destination (Join-Path $BackupRoot "production-registry.before-$ts.json") -Force
    Copy-Item -LiteralPath "$RegistryPath.sha256" -Destination (Join-Path $BackupRoot "production-registry.before-$ts.json.sha256") -Force

    $registry.localDevInfra.backend.commitSha = $NewCommitSha
    $nowIso = (Get-Date).ToString('o')
    $registry | Add-Member -NotePropertyName 'localDevInfraBackendUpdatedAt' -NotePropertyValue $nowIso -Force
    $registry | Add-Member -NotePropertyName 'lastVerifiedAt' -NotePropertyValue $nowIso -Force

    $tmp = "$RegistryPath.tmp"
    $json = $registry | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText($tmp, $json, [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $tmp -Destination $RegistryPath -Force

    $newHash = Write-ProductionRegistrySha256Sidecar -RegistryPath $RegistryPath
    Copy-Item -LiteralPath $RegistryPath -Destination (Join-Path $BackupRoot "production-registry.after-$ts.json") -Force
    Copy-Item -LiteralPath "$RegistryPath.sha256" -Destination (Join-Path $BackupRoot "production-registry.after-$ts.json.sha256") -Force

    $verify = Test-ProductionRegistryIntegrity -RegistryPath $RegistryPath -Sha256SidecarPath "$RegistryPath.sha256"
    if (-not $verify.Ok) {
        throw 'REGISTRY_POST_UPDATE_VERIFY_FAILED'
    }
    return @{ Updated = $true; OldSha = $oldSha; NewSha = $NewCommitSha; RegistryHash = $newHash; BackupRoot = $BackupRoot }
}

function Test-OriGitWorkingTreeClean {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)
    if (-not (Test-Path -LiteralPath $RepoRoot)) {
        return @{ Ok = $false; Errors = @('REPO_ROOT_MISSING') }
    }
    Push-Location -LiteralPath $RepoRoot
    try {
        git status --porcelain 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            return @{ Ok = $false; Errors = @('GIT_STATUS_FAILED') }
        }
        # Ignore untracked files (e.g. local .env.local) — only block staged/modified tracked paths.
        $lines = @(git status --porcelain -uno 2>$null | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($lines.Count -gt 0) {
            return @{ Ok = $false; Errors = @('WORKING_TREE_DIRTY') }
        }
        return @{ Ok = $true; Errors = @() }
    }
    finally {
        Pop-Location
    }
}

function Test-OriLocalDevInfraCommitCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$RegistrySha,
        [Parameter(Mandatory = $true)][string]$NewCommitSha,
        [Parameter(Mandatory = $true)][string]$ExpectedBranch
    )
    $errors = @()
    $ancestryOk = $false
    $branchOk = $false
    $commitFound = $false

    Push-Location -LiteralPath $RepoRoot
    try {
        git rev-parse --verify "$NewCommitSha^{commit}" 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $errors += "${Label}_NEW_INFRA_COMMIT_NOT_FOUND"
        }
        else {
            $commitFound = $true
            $resolved = (git rev-parse "$NewCommitSha^{commit}" 2>$null).Trim().ToLowerInvariant()
            git merge-base --is-ancestor $RegistrySha $resolved 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $errors += "${Label}_NEW_INFRA_COMMIT_NOT_DESCENDANT_OF_REGISTRY"
            }
            else {
                $ancestryOk = $true
            }
            $branches = @(git branch -a --contains $resolved 2>$null | ForEach-Object { $_.Trim() -replace '^\*?\s*', '' })
            $branchOk = @($branches | Where-Object {
                $_ -eq $ExpectedBranch -or $_ -eq "remotes/origin/$ExpectedBranch"
            }).Count -gt 0
            if (-not $branchOk) {
                $errors += "${Label}_COMMIT_NOT_ON_INFRA_BRANCH:$ExpectedBranch"
            }
        }
    }
    finally {
        Pop-Location
    }

    return @{
        Ok          = ($errors.Count -eq 0)
        Errors      = $errors
        AncestryOk  = $ancestryOk
        BranchOk    = $branchOk
        CommitFound = $commitFound
        RegistrySha = $RegistrySha
        NewCommitSha = $NewCommitSha
        ExpectedBranch = $ExpectedBranch
    }
}

function Get-OriProductionRegistryLocalDevInfraPairFingerprint {
    param([Parameter(Mandatory = $true)]$Registry)
    return [ordered]@{
        productionBackendWeb = [string]$Registry.production.backendWeb.deployedCommitSha
        productionBackendWorker = [string]$Registry.production.backendWorker.deployedCommitSha
        productionFrontend = [string]$Registry.production.frontend.deployedCommitSha
        wipBuilder1 = [string]$Registry.wip.builder1.commitSha
    }
}

function Get-OriProductionRegistryLocalDevInfraPairUpdatePlan {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryPath,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$NewBackendInfraCommit,
        [Parameter(Mandatory = $true)][string]$NewFrontendInfraCommit,
        [switch]$RequireCleanWorkingTrees
    )
    $blockers = @()
    $regCheck = Test-ProductionRegistryIntegrity -RegistryPath $RegistryPath -Sha256SidecarPath "$RegistryPath.sha256"
    if (-not $regCheck.Ok) {
        return @{
            Ok = $false
            Blockers = @($regCheck.Errors)
            Mode = 'PLAN'
            CanApply = $false
            WillUpdate = $false
        }
    }
    $registry = $regCheck.Registry
    $oldBackend = [string]$registry.localDevInfra.backend.commitSha
    $oldFrontend = [string]$registry.localDevInfra.frontend.commitSha
    $backendBranch = [string]$registry.localDevInfra.backend.branch
    $frontendBranch = [string]$registry.localDevInfra.frontend.branch
    if ([string]::IsNullOrWhiteSpace($backendBranch)) { $backendBranch = 'ori/local-dev-infrastructure' }
    if ([string]::IsNullOrWhiteSpace($frontendBranch)) { $frontendBranch = 'ori/local-dev-infrastructure' }

    $aceRoot = Split-Path -Parent (Split-Path -Parent $RegistryPath)
    $defaultBackupRoot = Join-Path $aceRoot '_backups\registry-updates'

    $backendCheck = Test-OriLocalDevInfraCommitCandidate `
        -RepoRoot $BackendRoot -Label 'BACKEND' -RegistrySha $oldBackend `
        -NewCommitSha $NewBackendInfraCommit -ExpectedBranch $backendBranch
    $frontendCheck = Test-OriLocalDevInfraCommitCandidate `
        -RepoRoot $FrontendRoot -Label 'FRONTEND' -RegistrySha $oldFrontend `
        -NewCommitSha $NewFrontendInfraCommit -ExpectedBranch $frontendBranch
    $blockers += @($backendCheck.Errors) + @($frontendCheck.Errors)

    $backendClean = Test-OriGitWorkingTreeClean -RepoRoot $BackendRoot
    $frontendClean = Test-OriGitWorkingTreeClean -RepoRoot $FrontendRoot
    if ($RequireCleanWorkingTrees) {
        if (-not $backendClean.Ok) { $blockers += @('BACKEND_' + ($backendClean.Errors -join ',')) }
        if (-not $frontendClean.Ok) { $blockers += @('FRONTEND_' + ($frontendClean.Errors -join ',')) }
    }

    $willUpdate = ($oldBackend -ne $NewBackendInfraCommit) -or ($oldFrontend -ne $NewFrontendInfraCommit)
    $canApply = ($blockers.Count -eq 0)

    return [ordered]@{
        Ok = $canApply
        Mode = 'PLAN'
        CanApply = $canApply
        WillUpdate = $willUpdate
        Blockers = $blockers
        RegistryPath = $RegistryPath
        BackupRoot = $defaultBackupRoot
        Current = [ordered]@{
            backend = $oldBackend
            frontend = $oldFrontend
        }
        Target = [ordered]@{
            backend = $NewBackendInfraCommit
            frontend = $NewFrontendInfraCommit
        }
        FieldsToUpdate = @(
            'localDevInfra.backend.commitSha'
            'localDevInfra.frontend.commitSha'
        )
        Ancestry = [ordered]@{
            backend = $backendCheck
            frontend = $frontendCheck
        }
        WorkingTrees = [ordered]@{
            backend = $backendClean
            frontend = $frontendClean
        }
        ProtectedFingerprint = (Get-OriProductionRegistryLocalDevInfraPairFingerprint -Registry $registry)
    }
}

function Restore-OriProductionRegistryFromBackupFiles {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryPath,
        [Parameter(Mandatory = $true)][string]$RegistryBackupPath,
        [Parameter(Mandatory = $true)][string]$SidecarBackupPath
    )
    Copy-Item -LiteralPath $RegistryBackupPath -Destination $RegistryPath -Force
    Copy-Item -LiteralPath $SidecarBackupPath -Destination "$RegistryPath.sha256" -Force
}

function Update-OriProductionRegistryLocalDevInfraPair {
    param(
        [Parameter(Mandatory = $true)][string]$RegistryPath,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$FrontendRoot,
        [Parameter(Mandatory = $true)][string]$NewBackendInfraCommit,
        [Parameter(Mandatory = $true)][string]$NewFrontendInfraCommit,
        [switch]$Preview,
        [string]$BackupRoot = '',
        [switch]$RequireCleanWorkingTrees
    )

    $plan = Get-OriProductionRegistryLocalDevInfraPairUpdatePlan `
        -RegistryPath $RegistryPath -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot `
        -NewBackendInfraCommit $NewBackendInfraCommit -NewFrontendInfraCommit $NewFrontendInfraCommit `
        -RequireCleanWorkingTrees:$RequireCleanWorkingTrees

    if ($Preview) {
        return [ordered]@{
            Mode = 'PREVIEW'
            PreviewOnly = $true
            Plan = $plan
        }
    }

    if (-not $plan.CanApply) {
        throw ($plan.Blockers -join ',')
    }
    if (-not $plan.WillUpdate) {
        return [ordered]@{
            Mode = 'APPLY'
            Updated = $false
            Plan = $plan
            RegistryHash = (Get-FileSha256Hex $RegistryPath)
        }
    }

    $regCheck = Test-ProductionRegistryIntegrity -RegistryPath $RegistryPath -Sha256SidecarPath "$RegistryPath.sha256"
    $registry = $regCheck.Registry
    $protectedBefore = Get-OriProductionRegistryLocalDevInfraPairFingerprint -Registry $registry

    if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
        $BackupRoot = [string]$plan.BackupRoot
    }
    New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
    $ts = Get-Date -Format 'yyyy-MM-dd_HHmmss'
    $beforeReg = Join-Path $BackupRoot "production-registry.before-$ts.json"
    $beforeSide = Join-Path $BackupRoot "production-registry.before-$ts.json.sha256"
    Copy-Item -LiteralPath $RegistryPath -Destination $beforeReg -Force
    Copy-Item -LiteralPath "$RegistryPath.sha256" -Destination $beforeSide -Force

    $registry.localDevInfra.backend.commitSha = $NewBackendInfraCommit
    $registry.localDevInfra.frontend.commitSha = $NewFrontendInfraCommit
    $nowIso = (Get-Date).ToString('o')
    $registry | Add-Member -NotePropertyName 'localDevInfraBackendUpdatedAt' -NotePropertyValue $nowIso -Force
    $registry | Add-Member -NotePropertyName 'localDevInfraFrontendUpdatedAt' -NotePropertyValue $nowIso -Force
    $registry | Add-Member -NotePropertyName 'lastVerifiedAt' -NotePropertyValue $nowIso -Force

    $tmp = "$RegistryPath.tmp.$([Guid]::NewGuid().ToString('N'))"
    $json = $registry | ConvertTo-Json -Depth 12
    try {
        [System.IO.File]::WriteAllText($tmp, $json, [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $tmp -Destination $RegistryPath -Force
        try {
            $newHash = Write-ProductionRegistrySha256Sidecar -RegistryPath $RegistryPath
        }
        catch {
            Restore-OriProductionRegistryFromBackupFiles -RegistryPath $RegistryPath `
                -RegistryBackupPath $beforeReg -SidecarBackupPath $beforeSide
            throw 'REGISTRY_SIDECAR_WRITE_FAILED_RESTORED'
        }
    }
    catch {
        if (Test-Path -LiteralPath $tmp) {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path -LiteralPath $RegistryPath) {
            Restore-OriProductionRegistryFromBackupFiles -RegistryPath $RegistryPath `
                -RegistryBackupPath $beforeReg -SidecarBackupPath $beforeSide
        }
        throw
    }

    $verify = Test-ProductionRegistryIntegrity -RegistryPath $RegistryPath -Sha256SidecarPath "$RegistryPath.sha256"
    if (-not $verify.Ok) {
        Restore-OriProductionRegistryFromBackupFiles -RegistryPath $RegistryPath `
            -RegistryBackupPath $beforeReg -SidecarBackupPath $beforeSide
        throw 'REGISTRY_POST_UPDATE_VERIFY_FAILED_RESTORED'
    }

    $protectedAfter = Get-OriProductionRegistryLocalDevInfraPairFingerprint -Registry $verify.Registry
    foreach ($key in $protectedBefore.Keys) {
        if ([string]$protectedBefore[$key] -ne [string]$protectedAfter[$key]) {
            Restore-OriProductionRegistryFromBackupFiles -RegistryPath $RegistryPath `
                -RegistryBackupPath $beforeReg -SidecarBackupPath $beforeSide
            throw 'REGISTRY_PROTECTED_FIELDS_CHANGED_RESTORED'
        }
    }

    Copy-Item -LiteralPath $RegistryPath -Destination (Join-Path $BackupRoot "production-registry.after-$ts.json") -Force
    Copy-Item -LiteralPath "$RegistryPath.sha256" -Destination (Join-Path $BackupRoot "production-registry.after-$ts.json.sha256") -Force

    return [ordered]@{
        Mode = 'APPLY'
        Updated = $true
        Plan = $plan
        RegistryHash = $newHash
        BackupRoot = $BackupRoot
        BeforeBackup = $beforeReg
        AfterBackup = (Join-Path $BackupRoot "production-registry.after-$ts.json")
    }
}
