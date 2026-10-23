#Requires -Version 5.1
<#
.SYNOPSIS
  Background Memurai PING monitor for ACE local dev stack (hidden process).

.DESCRIPTION
  Polls every 30s; Windows toast/balloon on down/up transitions only.
  Exits when Backend+Worker for this repo are no longer running.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BackendRoot,
    [string]$RedisUrl = 'redis://127.0.0.1:6379/0',
    [int]$IntervalSec = 30,
    [int]$GraceSec = 90,
    [int]$StackAbsentExitTicks = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'SilentlyContinue'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir 'AceLocalEnvCommon.ps1')
. (Join-Path $ScriptDir 'AceLocalMemuraiWatchLib.ps1')

$BackendRoot = (Resolve-Path -LiteralPath $BackendRoot).ProviderPath
$AceRoot = Split-Path -Parent $BackendRoot
$pidPath = Get-AceLocalMemuraiWatchPidPath -AceRoot $AceRoot
Set-Content -LiteralPath $pidPath -Value $PID -Encoding ASCII

$notifyState = 'Grace'
$graceUntil = (Get-Date).AddSeconds($GraceSec)
$stackAbsentCount = 0

try {
    while ($true) {
        if (-not (Test-AceLocalDevStackRunning -BackendRoot $BackendRoot)) {
            $stackAbsentCount++
            if ($stackAbsentCount -ge $StackAbsentExitTicks) {
                break
            }
        }
        else {
            $stackAbsentCount = 0
        }

        $ping = Test-LocalMemuraiPing -RedisUrl $RedisUrl
        $inGrace = (Get-Date) -lt $graceUntil
        $tick = Get-AceMemuraiWatchTickResult -CurrentNotifyState $notifyState -PingOk:$ping.Ok -InGracePeriod:$inGrace

        if ($tick.NotifyDown) {
            Invoke-AceLocalMemuraiDownAlert
        }
        if ($tick.NotifyUp) {
            Invoke-AceLocalMemuraiUpAlert
        }

        if (-not $inGrace) {
            $notifyState = [string]$tick.NextNotifyState
        }

        Start-Sleep -Seconds $IntervalSec
    }
}
finally {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
