# ACE LOCAL — ENV restore via existing DPAPI (phase 5.4; temp/isolated use only).
param(
    [string]$DpapiPath = '',
    [string]$TargetPath = '',
    [string]$CreateTestBlobPath = '',
    [string]$CreateTestPlainText = ''
)

Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalShutdownSnapshotLib.ps1')
. (Join-Path $ScriptDir 'OriDevSessionBaselineLib.ps1')

function Restore-AceLocalRecoveryEnvFromDpapi {
    param(
        [Parameter(Mandatory = $true)][string]$DpapiPath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    if (-not (Test-Path -LiteralPath $DpapiPath)) {
        Write-Error 'ACE_RECOVERY_ENV_DPAPI_MISSING'
    }
    if (Test-Path -LiteralPath $TargetPath) {
        Write-Error 'ACE_RECOVERY_ENV_TARGET_EXISTS'
    }
    $plain = Read-AceLocalShutdownEnvPlain -DpapiPath $DpapiPath
    if ($null -eq $plain) {
        Write-Error 'ACE_RECOVERY_ENV_DPAPI_DECRYPT_FAILED'
    }
    $targetDir = Split-Path -Parent $TargetPath
    if (-not (Test-Path -LiteralPath $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }
    $tmp = Join-Path $targetDir (('.env-restore-{0}.tmp' -f [guid]::NewGuid().ToString('N')))
    try {
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($tmp, $plain, $utf8)
        Move-Item -LiteralPath $tmp -Destination $TargetPath -Force
        Set-DirectoryCurrentUserAcl -Path $targetDir
        if (Test-Path -LiteralPath $TargetPath) {
            icacls $TargetPath /inheritance:r /grant:r "$([System.Security.Principal.WindowsIdentity]::GetCurrent().Name):(R,W)" /Q 2>&1 | Out-Null
        }
    }
    finally {
        if (Test-Path -LiteralPath $tmp) {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        }
    }
    if (-not (Test-Path -LiteralPath $TargetPath)) {
        Write-Error 'ACE_RECOVERY_ENV_WRITE_VERIFY_FAILED'
    }
}

function Invoke-AceLocalRecoveryEnvRestoreCli {
    param(
        [Parameter(Mandatory = $true)][string]$DpapiPath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    try {
        Restore-AceLocalRecoveryEnvFromDpapi -DpapiPath $DpapiPath -TargetPath $TargetPath | Out-Null
        exit 0
    }
    catch {
        Write-Error ($_.Exception.Message)
        exit 1
    }
}

function New-AceLocalRecoveryEnvDpapiTestBlob {
    param(
        [Parameter(Mandatory = $true)][string]$DestDpapiPath,
        [Parameter(Mandatory = $true)][string]$PlainText
    )
    $dir = Split-Path -Parent $DestDpapiPath
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $protected = Protect-ContentForCurrentUser -PlainBytes ([System.Text.Encoding]::UTF8.GetBytes($PlainText))
    [System.IO.File]::WriteAllBytes($DestDpapiPath, $protected)
    $hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($protected)
    Write-Output ([BitConverter]::ToString($hash).Replace('-', '').ToLowerInvariant())
}

if ($CreateTestBlobPath -and $CreateTestPlainText) {
    New-AceLocalRecoveryEnvDpapiTestBlob -DestDpapiPath $CreateTestBlobPath -PlainText $CreateTestPlainText
    exit 0
}
if ($DpapiPath -and $TargetPath) {
    Invoke-AceLocalRecoveryEnvRestoreCli -DpapiPath $DpapiPath -TargetPath $TargetPath
}
