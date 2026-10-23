#Requires -Version 5.1
<#
  Unit tests for Memurai watch state machine (no live Redis).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalMemuraiWatchLib.ps1')

$failures = @()
function Assert-True($cond, [string]$name) {
    if (-not $cond) { $script:failures += $name; Write-Host " FAIL $name" -ForegroundColor Red }
    else { Write-Host "  ok $name" -ForegroundColor DarkGray }
}

Write-Host 'Test-AceLocalMemuraiWatch...' -ForegroundColor Cyan

$t1 = Get-AceMemuraiWatchTickResult -CurrentNotifyState 'Grace' -PingOk:$false -InGracePeriod:$true
Assert-True (-not $t1.NotifyDown) 'grace_fail_no_down_alert'
Assert-True (-not $t1.NotifyUp) 'grace_fail_no_up_alert'

$t2 = Get-AceMemuraiWatchTickResult -CurrentNotifyState 'Up' -PingOk:$false -InGracePeriod:$false
Assert-True $t2.NotifyDown 'up_to_down_alerts_once'
Assert-True ($t2.NextNotifyState -eq 'Down') 'up_to_down_state'

$t3 = Get-AceMemuraiWatchTickResult -CurrentNotifyState 'Down' -PingOk:$false -InGracePeriod:$false
Assert-True (-not $t3.NotifyDown) 'down_stays_silent'
Assert-True (-not $t3.NotifyUp) 'down_stays_no_up'

$t4 = Get-AceMemuraiWatchTickResult -CurrentNotifyState 'Down' -PingOk:$true -InGracePeriod:$false
Assert-True $t4.NotifyUp 'down_to_up_recovery_alert'
Assert-True ($t4.NextNotifyState -eq 'Up') 'down_to_up_state'

$t5 = Get-AceMemuraiWatchTickResult -CurrentNotifyState 'Up' -PingOk:$true -InGracePeriod:$false
Assert-True (-not $t5.NotifyDown) 'up_stays_silent'
Assert-True (-not $t5.NotifyUp) 'up_stays_no_repeat_up'

$t6 = Get-AceMemuraiWatchTickResult -CurrentNotifyState 'Grace' -PingOk:$false -InGracePeriod:$false
Assert-True $t6.NotifyDown 'grace_end_fail_first_down'
$t7 = Get-AceMemuraiWatchTickResult -CurrentNotifyState 'Grace' -PingOk:$true -InGracePeriod:$false
Assert-True (-not $t7.NotifyDown) 'grace_end_ok_no_alert'
Assert-True ($t7.NextNotifyState -eq 'Up') 'grace_end_ok_up'

$downMsg = Get-AceLocalMemuraiAlertText -Kind Down
Assert-True ($downMsg -match 'PING') 'down_alert_text_utf8'

if ($env:ACE_TEST_MEMURAI_TOAST -eq '1') {
    Write-Host 'Manual toast probe (ACE_TEST_MEMURAI_TOAST=1)...' -ForegroundColor Yellow
    $r = Show-AceLocalMemuraiWindowsNotification -Title 'ACE LOCAL test' -Message 'Toast/balloon probe' -Level Info
    Assert-True $r.Ok "toast_or_balloon_shown:$($r.Method)"
}

if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All Memurai watch tests passed.' -ForegroundColor Green
