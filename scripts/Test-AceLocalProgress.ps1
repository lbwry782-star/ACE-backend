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
}
finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All AceLocalProgress tests passed.' -ForegroundColor Green
