#Requires -Version 5.1
<#
  Phase 5.12.1 — git porcelain -z path parsing; temp repos only for mutation tests.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryDirtyDetectLib.ps1')
. (Join-Path $ScriptDir 'AceLocalRecoveryCheckpointLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

function Initialize-TempGitRepo {
    param([Parameter(Mandatory = $true)][string]$Root)
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    Push-Location -LiteralPath $Root
    try {
        git init -q 2>$null | Out-Null
        git config user.email 'recovery-5121@ace.local' 2>$null | Out-Null
        git config user.name 'Recovery 5121' 2>$null | Out-Null
    }
    finally { Pop-Location }
}

function New-TempLayout {
    $root = Join-Path $env:TEMP ("ace-recovery-5121-{0}" -f [guid]::NewGuid().ToString('N'))
    $ace = Join-Path $root 'ACE'
    $backend = Join-Path $ace 'ACE-Backend'
    $frontend = Join-Path $ace 'ACE-Frontend'
    $cpRoot = Join-Path $ace '_backups\local-dev\recovery-checkpoints'
    New-Item -ItemType Directory -Path $backend, $frontend, $cpRoot -Force | Out-Null
    Initialize-TempGitRepo -Root $backend
    Initialize-TempGitRepo -Root $frontend
    return @{
        TempRoot = $root; AceRoot = $ace; Backend = $backend; Frontend = $frontend; CpRoot = $cpRoot
    }
}

Write-Host 'Test-AceLocalRecoveryPhase5121 (isolated)...' -ForegroundColor Cyan

$badParse = ConvertFrom-GitPorcelainV1ZBytes -Bytes ([System.Text.Encoding]::UTF8.GetBytes("?? `"bad`""))
Assert-True (-not $badParse.Ok) 'parse_rejects_quoted_path_blob'

$badHeader = ConvertFrom-GitPorcelainV1ZBytes -Bytes ([System.Text.Encoding]::UTF8.GetBytes("badpath`0"))
Assert-True (-not $badHeader.Ok) 'parse_rejects_invalid_header'

$layout = New-TempLayout
try {
    $ace = $layout.AceRoot
    $be = $layout.Backend
    $fe = $layout.Frontend
    $cpRoot = $layout.CpRoot

    'v1' | Set-Content -LiteralPath (Join-Path $be 'base.txt') -Encoding UTF8
    git -C $be add base.txt 2>$null | Out-Null
    git -C $be commit -q -m 'init' 2>$null | Out-Null
    'fe' | Set-Content -LiteralPath (Join-Path $fe 'base.txt') -Encoding UTF8
    git -C $fe add base.txt 2>$null | Out-Null
    git -C $fe commit -q -m 'init' 2>$null | Out-Null

    New-Item -ItemType Directory -Path (Join-Path $be 'dir with spaces') -Force | Out-Null
    'spaces' | Set-Content -LiteralPath (Join-Path $be 'dir with spaces\inside.txt') -Encoding UTF8
    $hebrewBase = ([char]0x05E2) + ([char]0x05D1) + ([char]0x05E8) + ([char]0x05D9) + ([char]0x05EA)
    $hebrewRel = "$hebrewBase.txt"
    'hebrew' | Set-Content -LiteralPath (Join-Path $be $hebrewRel) -Encoding UTF8
    git -C $be add 'dir with spaces/inside.txt' 2>$null | Out-Null
    git -C $be commit -q -m 'add inside' 2>$null | Out-Null
    git -C $be mv base.txt 'renamed base.txt' 2>$null | Out-Null
    'untracked' | Set-Content -LiteralPath (Join-Path $be 'brand new.txt') -Encoding UTF8
    Remove-Item -LiteralPath (Join-Path $be 'dir with spaces\inside.txt') -Force

    $scan = Get-AceLocalRecoveryWorkspaceDirtyScan -BackendRoot $be -FrontendRoot $fe
    Assert-True ($scan.ScanOk -and $scan.Backend.ScanOk) 'scan_ok_flag'
    $paths = @($scan.Backend.Entries | ForEach-Object { $_.path })
    Assert-True (@($paths | Where-Object { $_ -match 'renamed base' }).Count -ge 1) 'rename_detected'
    Assert-True (@($paths | Where-Object { $_ -match 'brand new' }).Count -ge 1) 'untracked_spaces'
    Assert-True (@($paths | Where-Object { $_ -like "*$hebrewBase*" }).Count -ge 1) 'hebrew_path'
    Assert-True (@($scan.Backend.Entries | Where-Object { $_.changeType -eq 'deleted' }).Count -ge 1) 'deleted_detected'
    Assert-True (-not (@($paths | Where-Object { $_ -match '"' })).Count) 'no_quote_in_paths'

    'edit' | Set-Content -LiteralPath (Join-Path $be 'renamed base.txt') -Encoding UTF8
    Initialize-AceLocalRecoveryLedger -AceRoot $ace -LauncherSession '20261024-512100' | Out-Null
    $cp = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession '20261024-512100' -CheckpointType full -CheckpointsRootOverride $cpRoot
    Assert-True ($cp.Ok -and $cp.Status -eq 'VERIFIED') 'full_checkpoint_verified'

    $manifest = Import-AceLocalRecoveryCheckpointManifestRows -CheckpointRoot $cp.CheckpointRoot
    $beRows = @($manifest | Where-Object { [string]$_.Area -eq 'BACKEND' })
    Assert-True ($beRows.Count -ge 3) 'manifest_includes_backend_files'
    $manifestPaths = @($beRows | ForEach-Object { ([string]$_.RelPath -replace '\\', '/').ToLowerInvariant() })
    Assert-True (@($manifestPaths | Where-Object { $_ -match 'renamed base' }).Count -ge 1) 'manifest_has_renamed'
    Assert-True (@($manifestPaths | Where-Object { $_ -like "*$($hebrewBase.ToLowerInvariant())*" }).Count -ge 1) 'manifest_has_hebrew'

    $corruptFailed = $false
    try {
        $null = ConvertFrom-GitPorcelainV1ZBytes -Bytes ([byte[]](0x3F, 0x3F, 0x20, 0x22, 0x62, 0x61, 0x64, 0x22, 0x00))
        if (-not $corruptFailed) {
            $p2 = ConvertFrom-GitPorcelainV1ZBytes -Bytes ([byte[]](0x3F, 0x3F, 0x20, 0x22, 0x62, 0x61, 0x64, 0x22, 0x00))
            if (-not $p2.Ok) { $corruptFailed = $true }
        }
    }
    catch { $corruptFailed = $true }
    Assert-True $corruptFailed 'corrupt_porcelain_fails_safe'

    $cpFail = New-AceLocalRecoveryCheckpoint -BackendRoot $be -FrontendRoot $fe -AceRoot $ace `
        -LauncherSession '20261024-512100' -CheckpointType full -CheckpointsRootOverride $cpRoot `
        -SimulateFailBeforeVerify
    Assert-True ($cpFail.Status -eq 'INCOMPLETE') 'incomplete_not_verified_on_sim_fail'
    $verified = @(Get-AceLocalRecoveryVerifiedCheckpoints -AceRoot $ace -CheckpointsRootOverride $cpRoot)
    Assert-True (@($verified | Where-Object { $_.CheckpointId -eq $cp.CheckpointId }).Count -eq 1) 'prior_verified_retained'
}
finally {
    Remove-Item -LiteralPath $layout.TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (@($failures).Count -gt 0) {
    Write-Host ("`nFailures: " + ($failures -join ', ')) -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalRecovery phase 5.12.1 tests passed.' -ForegroundColor Green
exit 0
