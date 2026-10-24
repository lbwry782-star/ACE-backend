#Requires -Version 5.1
<#
  Unit tests for Memurai instruction dialog logic (no live Memurai control).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalMemuraiInstructionLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalMemuraiInstruction...' -ForegroundColor Cyan

$pingOk = { param($Url) @{ Ok = $true; Error = '' } }
$pingFail = { param($Url) @{ Ok = $false; Error = 'ping_no_pong' } }
$pingAmbiguous = { param($Url) @{ Ok = $false; Error = 'memurai_cli_not_found' } }
$svcRunning = { [PSCustomObject]@{ Status = 'Running' } }
$svcStopped = { [PSCustomObject]@{ Status = 'Stopped' } }
$svcMissing = { $null }

$dOk = Get-AceLocalMemuraiDiagnosis -RedisUrl 'redis://127.0.0.1:6379/0' -PingFn $pingOk -ServiceFn $svcStopped
Assert-True ($dOk.Scenario -eq 'PongOk') 'diag_pong_ok'
Assert-True (-not $dOk.ShowInstruction) 'diag_pong_no_instruction'

$dStop = Get-AceLocalMemuraiDiagnosis -RedisUrl 'redis://127.0.0.1:6379/0' -PingFn $pingFail -ServiceFn $svcStopped
Assert-True ($dStop.Scenario -eq 'ServiceStopped') 'diag_service_stopped'
Assert-True $dStop.ShowInstruction 'diag_stopped_show_instruction'

$dRunNoPong = Get-AceLocalMemuraiDiagnosis -RedisUrl 'redis://127.0.0.1:6379/0' -PingFn $pingFail -ServiceFn $svcRunning
Assert-True ($dRunNoPong.Scenario -eq 'ServiceRunningNoPong') 'diag_running_no_pong'
Assert-True $dRunNoPong.ShowInstruction 'diag_running_show_instruction'

$dAmb = Get-AceLocalMemuraiDiagnosis -RedisUrl 'redis://127.0.0.1:6379/0' -PingFn $pingAmbiguous -ServiceFn $svcStopped
Assert-True ($dAmb.Scenario -eq 'AmbiguousCheckError') 'diag_ambiguous'
Assert-True (-not $dAmb.ShowInstruction) 'diag_ambiguous_no_instruction'

$dMissing = Get-AceLocalMemuraiDiagnosis -RedisUrl 'redis://127.0.0.1:6379/0' -PingFn $pingFail -ServiceFn $svcMissing
Assert-True ($dMissing.Scenario -eq 'ServiceNotInstalled') 'diag_not_installed'

$body = Get-AceLocalMemuraiInstructionDialogBody
Assert-True ($body -match 'services\.msc') 'dialog_body_services_msc'
Assert-True ($body -match 'Win \+ R') 'dialog_body_win_r'

$tmpAce = Join-Path $env:TEMP ("ace-memurai-instr-test-{0}" -f [guid]::NewGuid().ToString('n'))
New-Item -ItemType Directory -Path $tmpAce -Force | Out-Null
try {
    $script:dialogCalls = 0
    $mockDialog = {
        param($Body)
        $script:dialogCalls++
    }

    $r1 = Invoke-AceLocalMemuraiUserInstruction -AceRoot $tmpAce -RedisUrl 'redis://127.0.0.1:6379/0' `
        -IsNotifyDownTransition -PingFn $pingFail -ServiceFn $svcStopped -ShowDialog $mockDialog
    Assert-True $r1.Shown 'first_down_shows'
    Assert-True ($script:dialogCalls -eq 1) 'first_down_one_dialog'

    $r2 = Invoke-AceLocalMemuraiUserInstruction -AceRoot $tmpAce -RedisUrl 'redis://127.0.0.1:6379/0' `
        -PingFn $pingFail -ServiceFn $svcStopped -ShowDialog $mockDialog
    Assert-True (-not $r2.Shown) 'repeat_while_down_silent'
    Assert-True ($script:dialogCalls -eq 1) 'repeat_still_one_dialog'

    Clear-AceLocalMemuraiOutageInstructionState -AceRoot $tmpAce
    $availPath = Get-AceLocalMemuraiAvailabilityPath -AceRoot $tmpAce
    Assert-True (Test-Path -LiteralPath $availPath) 'availability_file_written'
    $availRaw = Get-Content -LiteralPath $availPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-True ($availRaw.blocksNewRedisWork -eq $true) 'availability_blocks_redis'

    $script:dialogCalls = 0
    $r3 = Invoke-AceLocalMemuraiUserInstruction -AceRoot $tmpAce -RedisUrl 'redis://127.0.0.1:6379/0' `
        -IsNotifyDownTransition -PingFn $pingFail -ServiceFn $svcStopped -ShowDialog $mockDialog
    Assert-True $r3.Shown 'second_outage_after_recovery_shows'
    Assert-True ($script:dialogCalls -eq 1) 'second_outage_one_dialog'

    $state = Read-AceLocalMemuraiInstructionState -AceRoot $tmpAce
    Assert-True ($state.instructionAcknowledged -eq $true) 'ok_sets_acknowledged'

    $shouldStartup = Test-ShouldShowAceLocalMemuraiInstruction -Diagnosis $dStop `
        -InstructionState $state -IsNotifyDownTransition:$false -StartupAttempt
    Assert-True $shouldStartup 'startup_always_shows_when_instructable'

    $startText = Get-Content -LiteralPath (Join-Path $ScriptDir 'Start-AceLocal.ps1') -Raw
    Assert-True ($startText -notmatch 'Start-Service\s+-Name\s+"Memurai"') 'no_auto_start_service'
}
finally {
    Remove-Item -LiteralPath $tmpAce -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All Memurai instruction tests passed.' -ForegroundColor Green
