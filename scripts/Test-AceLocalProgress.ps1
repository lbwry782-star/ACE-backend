#Requires -Version 5.1
<#
  Isolated tests for LOCAL PROGRESS UI state (no window launch, no stack start).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalProgressLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalProgress (isolated)...' -ForegroundColor Cyan

$startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
Assert-True ($startText -match 'AceLocalProgressLib') 'start_local_dots_progress_lib'
Assert-True ($startText -match 'Start-AceLocalProgressUiProcess') 'start_local_launches_progress_ui'
Assert-True ($startText -match 'Complete-AceLocalProgressState') 'start_local_completes_progress'
Assert-True ($startText -match 'Fail-AceLocalProgressState') 'start_local_fails_progress_on_error'
Assert-True ($startText -match 'if \(\$ProgressUiEnabled -and \$memuraiOk\)') 'progress_memurai_step_only_when_ok'
Assert-True ($startText -notmatch 'Set-AceLocalProgressStep[\s\S]*Complete[\s\S]*Memurai') 'no_complete_on_memurai_path'
Assert-True (Test-Path -LiteralPath (Join-Path $ScriptDir 'AceLocalProgressWindow.ps1')) 'progress_window_script_exists'
$windowText = Get-Content -LiteralPath (Join-Path $ScriptDir 'AceLocalProgressWindow.ps1') -Raw
Assert-True ($windowText -match "Text = 'PROGRESS'") 'window_title_progress'
Assert-True ($windowText -match 'Global\\AceLocalProgressUi') 'window_mutex_duplicate_guard'
Assert-True ($windowText -notmatch 'System\.Windows\.Forms\.ProgressBar') 'no_native_progressbar'
Assert-True ($windowText -match '0x1E,\s*0x5C,\s*0x3A') 'fill_color_1e5c3a'
Assert-True ($windowText -match '0xDE,\s*0xDE,\s*0xD5') 'track_color_deded5'
Assert-True ($windowText -match '0xF7,\s*0xF6,\s*0xF2') 'window_bg_f7f6f2'
Assert-True ($windowText -match 'TrackWidth = 360') 'track_width_360'
Assert-True ($windowText -match 'TrackHeight = 10') 'track_height_10'
Assert-True ($windowText -match 'CornerRadius = 6') 'corner_radius_6'
Assert-True ($windowText -match 'AnimDurationMs = 240') 'anim_duration_240ms'
Assert-True ($windowText -match 'SmoothingMode.*AntiAlias') 'anti_alias'
Assert-True ($windowText -notmatch '\$completeWait') 'no_unscoped_complete_wait'
Assert-True ($windowText -match 'script:CompleteCloseTimer') 'complete_close_timer_script_scope'

Write-Host '  strict callback smoke...' -ForegroundColor DarkGray
Set-StrictMode -Version Latest
$script:CompleteCloseTimer = $null
Add-Type -AssemblyName System.Windows.Forms
$script:CompleteCloseTimer = New-Object System.Windows.Forms.Timer
$script:CompleteCloseTimer.Add_Tick({
        if ($null -ne $script:CompleteCloseTimer) {
            $script:CompleteCloseTimer.Stop()
        }
    })
Assert-True ($null -ne $script:CompleteCloseTimer) 'strictmode_nested_timer_callback'
try { $script:CompleteCloseTimer.Dispose() } catch { }
$script:CompleteCloseTimer = $null

Assert-True ((Get-AceLocalProgressStepPercent -Step 'Backend') -eq 45) 'step_backend_percent'
Assert-True ((Get-AceLocalProgressStepPercent -Step 'Complete') -eq 100) 'step_complete_percent'

$tmp = Join-Path $env:TEMP ("ace-progress-test-{0}" -f [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
$statePath = Join-Path $tmp 'local-start-progress.json'
$libPath = Join-Path $ScriptDir 'AceLocalProgressLib.ps1'
$ps51 = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
try {
    Initialize-AceLocalProgressState -StatePath $statePath
    $s0 = Read-AceLocalProgressState -StatePath $statePath
    Assert-True ($s0.status -eq 'running' -and [int]$s0.percent -eq 0) 'init_running_zero'

    Set-AceLocalProgressStep -StatePath $statePath -Step 'Worker'
    $s1 = Read-AceLocalProgressState -StatePath $statePath
    Assert-True ([int]$s1.percent -eq 65) 'step_worker_written'

    Complete-AceLocalProgressState -StatePath $statePath
    $s2 = Read-AceLocalProgressState -StatePath $statePath
    Assert-True ($s2.status -eq 'complete' -and [int]$s2.percent -eq 100) 'complete_100'

    Set-AceLocalProgressStep -StatePath $statePath -Step 'Backend'
    Fail-AceLocalProgressState -StatePath $statePath
    $s3 = Read-AceLocalProgressState -StatePath $statePath
    Assert-True ($s3.status -eq 'error' -and [int]$s3.percent -ne 100) 'error_not_forced_to_100'

    $replacePath = Join-Path $tmp 'replace-existing.json'
    Assert-True (-not (Test-Path -LiteralPath $replacePath)) 'replace_start_without_dest'
    Write-AceLocalProgressState -StatePath $replacePath -Percent 0 -Status 'running'
    $r0 = Read-AceLocalProgressState -StatePath $replacePath
    Assert-True ($r0.status -eq 'running' -and [int]$r0.percent -eq 0 -and [int]$r0.schemaVersion -eq 1) 'first_write_creates_dest'
    Write-AceLocalProgressState -StatePath $replacePath -Percent 45 -Status 'running'
    Write-AceLocalProgressState -StatePath $replacePath -Percent 65 -Status 'running'
    $r1 = Read-AceLocalProgressState -StatePath $replacePath
    Assert-True ([int]$r1.percent -eq 65 -and $r1.status -eq 'running') 'repeat_write_overwrites_existing'
    $rawRepeat = Get-Content -LiteralPath $replacePath -Raw -Encoding UTF8
    Assert-True ($null -ne ($rawRepeat | ConvertFrom-Json)) 'repeat_write_valid_json'

    $readDuringPath = Join-Path $tmp 'read-during.json'
    Write-AceLocalProgressState -StatePath $readDuringPath -Percent 0 -Status 'running'
    $readerLog = Join-Path $tmp 'reader-errors.txt'
    $readerScript = Join-Path $tmp 'reader.ps1'
    $readerBody = @"
`$ErrorActionPreference = 'Continue'
`$end = [DateTime]::UtcNow.AddSeconds(4)
`$errors = New-Object System.Collections.Generic.List[string]
while ([DateTime]::UtcNow -lt `$end) {
    try {
        if (Test-Path -LiteralPath '$readDuringPath') {
            `$raw = Get-Content -LiteralPath '$readDuringPath' -Raw -Encoding UTF8
            if (-not [string]::IsNullOrWhiteSpace(`$raw)) {
                `$obj = `$raw | ConvertFrom-Json
                if (`$null -eq `$obj) { `$errors.Add('null_json') }
            }
        }
    }
    catch {
        `$errors.Add(`$_.Exception.Message)
    }
    Start-Sleep -Milliseconds 5
}
Set-Content -LiteralPath '$readerLog' -Value (`$errors -join "`n") -Encoding UTF8
"@
    Set-Content -LiteralPath $readerScript -Value $readerBody -Encoding UTF8
    $readerProc = Start-Process -FilePath $ps51 -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $readerScript) -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 50
    for ($i = 1; $i -le 25; $i++) {
        Write-AceLocalProgressState -StatePath $readDuringPath -Percent $i -Status 'running'
    }
    try { Wait-Process -Id $readerProc.Id -Timeout 8 } catch { }
    $readerErrText = ''
    if (Test-Path -LiteralPath $readerLog) {
        $readerErrText = [string](Get-Content -LiteralPath $readerLog -Raw -ErrorAction SilentlyContinue)
    }
    $partialHits = @()
    if ($readerErrText) {
        $partialHits = @(
            $readerErrText -split "`r?`n" |
                Where-Object { $_ -and ($_ -match 'Invalid JSON|Unexpected end|partial|incomplete') }
        )
    }
    Assert-True ($partialHits.Count -eq 0) 'reader_no_partial_json'
    $afterRead = Read-AceLocalProgressState -StatePath $readDuringPath
    Assert-True ([int]$afterRead.percent -eq 25) 'reader_final_json_intact'

    $lockPath = Join-Path $tmp 'locked.json'
    Write-AceLocalProgressState -StatePath $lockPath -Percent 20 -Status 'running'
    $lockFs = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    $lockWriter = Join-Path $tmp 'lock-writer.ps1'
    $lockWriterLog = Join-Path $tmp 'lock-writer.log'
    @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
. '$libPath'
try {
    Write-AceLocalProgressState -StatePath '$lockPath' -Percent 45 -Status 'running'
    Set-Content -LiteralPath '$lockWriterLog' -Value 'OK' -Encoding UTF8
    exit 0
}
catch {
    Set-Content -LiteralPath '$lockWriterLog' -Value `$_.Exception.Message -Encoding UTF8
    exit 1
}
"@ | Set-Content -LiteralPath $lockWriter -Encoding UTF8
    $lockProc = Start-Process -FilePath $ps51 -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $lockWriter) -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 300
    $lockFs.Dispose()
    $lockFs = $null
    try { Wait-Process -Id $lockProc.Id -Timeout 8 } catch { }
    $lockLog = ''
    if (Test-Path -LiteralPath $lockWriterLog) {
        $lockLog = [string](Get-Content -LiteralPath $lockWriterLog -Raw)
    }
    Assert-True ($lockProc.ExitCode -eq 0) ('temp_lock_write_succeeds:' + $lockLog.Trim())
    $afterLock = Read-AceLocalProgressState -StatePath $lockPath
    Assert-True ([int]$afterLock.percent -eq 45) 'temp_lock_content_updated'

    $persistPath = Join-Path $tmp 'persist-lock.json'
    Write-AceLocalProgressState -StatePath $persistPath -Percent 65 -Status 'running'
    $beforePersist = Get-Content -LiteralPath $persistPath -Raw -Encoding UTF8
    $persistFs = [System.IO.File]::Open($persistPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    $persistThrew = $false
    $persistMsg = ''
    try {
        Write-AceLocalProgressState -StatePath $persistPath -Percent 85 -Status 'complete'
    }
    catch {
        $persistThrew = $true
        $persistMsg = [string]$_.Exception.Message
    }
    finally {
        $persistFs.Dispose()
    }
    $afterPersist = Get-Content -LiteralPath $persistPath -Raw -Encoding UTF8
    Assert-True $persistThrew 'persistent_lock_throws'
    Assert-True ($persistMsg -and $persistMsg -notmatch '(?i)complete') 'persistent_lock_error_is_clear'
    Assert-True ($afterPersist -eq $beforePersist) 'persistent_lock_keeps_previous_dest'
    $persistState = Read-AceLocalProgressState -StatePath $persistPath
    Assert-True ($persistState.status -eq 'running' -and [int]$persistState.percent -eq 65) 'persistent_lock_not_marked_complete'

    $parallelDir = Join-Path $tmp 'parallel'
    New-Item -ItemType Directory -Path $parallelDir -Force | Out-Null
    $parallelPath = Join-Path $parallelDir 'local-start-progress.json'
    Write-AceLocalProgressState -StatePath $parallelPath -Percent 0 -Status 'running'
    $workerScript = Join-Path $parallelDir 'writer.ps1'
    @'
param([string]$LibPath,[string]$StatePath,[int]$Seed)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. $LibPath
for ($i = 0; $i -lt 12; $i++) {
    Write-AceLocalProgressState -StatePath $StatePath -Percent (($Seed + $i) % 100) -Status 'running'
}
'@ | Set-Content -LiteralPath $workerScript -Encoding UTF8
    $workers = @()
    foreach ($seed in @(1, 20, 40)) {
        $workers += Start-Process -FilePath $ps51 -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $workerScript,
            '-LibPath', $libPath, '-StatePath', $parallelPath, '-Seed', "$seed"
        ) -PassThru -WindowStyle Hidden
    }
    foreach ($w in $workers) {
        try { Wait-Process -Id $w.Id -Timeout 15 } catch { }
    }
    $leftoverTmp = @(Get-ChildItem -LiteralPath $parallelDir -Filter 'local-start-progress.json.tmp.*' -ErrorAction SilentlyContinue)
    $leftoverBak = @(Get-ChildItem -LiteralPath $parallelDir -Filter 'local-start-progress.json.bak.*' -ErrorAction SilentlyContinue)
    Assert-True ($leftoverTmp.Count -eq 0) 'parallel_no_leftover_tmp'
    Assert-True ($leftoverBak.Count -eq 0) 'parallel_no_leftover_bak'
    $parallelRaw = Get-Content -LiteralPath $parallelPath -Raw -Encoding UTF8
    $parallelObj = $null
    try { $parallelObj = $parallelRaw | ConvertFrom-Json } catch { }
    Assert-True ($null -ne $parallelObj) 'parallel_final_json_valid'
    Assert-True ($parallelObj.status -eq 'running') 'parallel_final_status_running'
}
finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalProgress tests passed.' -ForegroundColor Green
