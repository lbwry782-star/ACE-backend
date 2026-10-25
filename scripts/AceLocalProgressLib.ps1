# ACE LOCAL — minimal PROGRESS UI state + host launcher (WinForms, no extra deps).
Set-StrictMode -Version Latest

$script:AceLocalProgressMutexName = 'Global\AceLocalProgressUi'

function Get-AceLocalProgressStatePath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $dir = Join-Path $AceRoot '_backups\local-dev'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return Join-Path $dir 'local-start-progress.json'
}

function Get-AceLocalProgressStepPercent {
    param([Parameter(Mandatory = $true)][ValidateSet('Init', 'SafetyMemurai', 'Backend', 'Worker', 'Frontend', 'LocalUi', 'Complete')][string]$Step)
    switch ($Step) {
        'Init' { return 5 }
        'SafetyMemurai' { return 20 }
        'Backend' { return 45 }
        'Worker' { return 65 }
        'Frontend' { return 85 }
        'LocalUi' { return 95 }
        'Complete' { return 100 }
    }
    return 0
}

function Test-AceLocalProgressReplaceTransient {
    param([Parameter(Mandatory = $true)]$ErrorRecord)
    $ex = $ErrorRecord.Exception
    while ($ex) {
        if ($ex -is [System.IO.IOException] -or $ex -is [System.UnauthorizedAccessException]) {
            $hr = [int]$ex.HResult
            # ERROR_ACCESS_DENIED (5), ERROR_SHARING_VIOLATION (32), ERROR_LOCK_VIOLATION (33), ERROR_ALREADY_EXISTS (183)
            if ($hr -eq -2147024891 -or $hr -eq -2147024864 -or $hr -eq -2147024863 -or $hr -eq -2147024713) {
                return $true
            }
            $msg = [string]$ex.Message
            if ($msg -match 'already exists|being used by another process|Access is denied|cannot access the file') {
                return $true
            }
        }
        $ex = $ex.InnerException
    }
    return $false
}

function Write-AceLocalProgressState {
    param(
        [Parameter(Mandatory = $true)][string]$StatePath,
        [Parameter(Mandatory = $true)][int]$Percent,
        [Parameter(Mandatory = $true)][ValidateSet('running', 'complete', 'error')][string]$Status
    )
    $pct = [Math]::Max(0, [Math]::Min(100, $Percent))
    $payload = [PSCustomObject]@{
        schemaVersion = 1
        percent       = $pct
        status        = $Status
        updatedAt     = (Get-Date).ToUniversalTime().ToString('o')
    }
    $json = $payload | ConvertTo-Json -Compress
    $dir = Split-Path -Parent $StatePath
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $writeId = [guid]::NewGuid().ToString('N')
    $tmp = Join-Path $dir ("local-start-progress.json.tmp.{0}" -f $writeId)
    $bak = Join-Path $dir ("local-start-progress.json.bak.{0}" -f $writeId)
    Set-Content -LiteralPath $tmp -Value $json -Encoding UTF8

    $maxAttempts = 8
    $retryDelayMs = 50
    $deadlineUtc = [DateTime]::UtcNow.AddSeconds(2)
    $committed = $false
    try {
        for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
            try {
                if (Test-Path -LiteralPath $StatePath) {
                    [System.IO.File]::Replace($tmp, $StatePath, $bak)
                }
                else {
                    [System.IO.File]::Move($tmp, $StatePath)
                }
                $committed = $true
                break
            }
            catch {
                $canRetry = (Test-AceLocalProgressReplaceTransient -ErrorRecord $_) -and
                    ($attempt -lt $maxAttempts) -and
                    ([DateTime]::UtcNow -lt $deadlineUtc)
                if (-not $canRetry) {
                    throw
                }
                Start-Sleep -Milliseconds $retryDelayMs
            }
        }
    }
    finally {
        if (-not $committed -and (Test-Path -LiteralPath $tmp)) {
            Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path -LiteralPath $bak) {
            Remove-Item -LiteralPath $bak -Force -ErrorAction SilentlyContinue
        }
    }
}

function Initialize-AceLocalProgressState {
    param([Parameter(Mandatory = $true)][string]$StatePath)
    Write-AceLocalProgressState -StatePath $StatePath -Percent 0 -Status 'running'
}

function Set-AceLocalProgressStep {
    param(
        [Parameter(Mandatory = $true)][string]$StatePath,
        [Parameter(Mandatory = $true)][ValidateSet('Init', 'SafetyMemurai', 'Backend', 'Worker', 'Frontend', 'LocalUi', 'Complete')][string]$Step
    )
    $pct = Get-AceLocalProgressStepPercent -Step $Step
    Write-AceLocalProgressState -StatePath $StatePath -Percent $pct -Status 'running'
}

function Complete-AceLocalProgressState {
    param([Parameter(Mandatory = $true)][string]$StatePath)
    Write-AceLocalProgressState -StatePath $StatePath -Percent 100 -Status 'complete'
}

function Fail-AceLocalProgressState {
    param(
        [Parameter(Mandatory = $true)][string]$StatePath,
        [int]$Percent = -1
    )
    $current = Read-AceLocalProgressState -StatePath $StatePath
    $pct = if ($Percent -ge 0) { $Percent } elseif ($current) { [int]$current.percent } else { 0 }
    Write-AceLocalProgressState -StatePath $StatePath -Percent $pct -Status 'error'
}

function Read-AceLocalProgressState {
    param([Parameter(Mandatory = $true)][string]$StatePath)
    if (-not (Test-Path -LiteralPath $StatePath)) {
        return $null
    }
    try {
        $raw = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $null
        }
        return ($raw | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Start-AceLocalProgressUiProcess {
    param(
        [Parameter(Mandatory = $true)][string]$StatePath,
        [Parameter(Mandatory = $true)][string]$WindowScriptPath
    )
    if (-not (Test-Path -LiteralPath $WindowScriptPath)) {
        return $false
    }
    $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $psExe)) {
        $psExe = 'powershell.exe'
    }
    $args = @(
        '-NoLogo',
        '-NoProfile',
        '-Sta',
        '-ExecutionPolicy', 'Bypass',
        '-File', $WindowScriptPath,
        '-StatePath', $StatePath
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $psExe
    $psi.Arguments = ($args | ForEach-Object { if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ } }) -join ' '
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    try {
        [void][System.Diagnostics.Process]::Start($psi)
        return $true
    }
    catch {
        return $false
    }
}
