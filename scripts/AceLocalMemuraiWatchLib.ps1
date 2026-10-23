# ACE local Memurai watch — shared helpers (dot-sourced).
Set-StrictMode -Version Latest

function Get-AceLocalMemuraiWatchPidPath {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    $dir = Join-Path $AceRoot '_backups\local-dev'
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    return Join-Path $dir 'memurai-watch.pid'
}

function Read-AceLocalMemuraiWatchPid {
    param([Parameter(Mandatory = $true)][string]$PidPath)
    if (-not (Test-Path -LiteralPath $PidPath)) {
        return $null
    }
    $raw = (Get-Content -LiteralPath $PidPath -Raw -ErrorAction SilentlyContinue).Trim()
    if ($raw -match '^\d+$') {
        return [int]$raw
    }
    return $null
}

function Stop-AceLocalMemuraiWatchProcess {
    param([Parameter(Mandatory = $true)][string]$PidPath)
    $pidVal = Read-AceLocalMemuraiWatchPid -PidPath $PidPath
    if (-not $pidVal) {
        return $false
    }
    $proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $PidPath -Force -ErrorAction SilentlyContinue
    return $true
}

function Test-AceLocalDevStackRunning {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot
    )
    $backend = @(Find-LocalAceBackendProcesses -BackendRoot $BackendRoot)
    if ($backend.Count -gt 0) {
        return $true
    }
    $worker = @(Find-LocalAceWorkerProcesses -BackendRoot $BackendRoot)
    return ($worker.Count -gt 0)
}

function Get-AceMemuraiWatchTickResult {
    <#
      Pure state transition for one 30s tick (unit-testable).
      CurrentNotifyState: Up | Down | Grace (no user notifications during Grace)
    #>
    param(
        [Parameter(Mandatory = $true)][string]$CurrentNotifyState,
        [Parameter(Mandatory = $true)][bool]$PingOk,
        [Parameter(Mandatory = $true)][bool]$InGracePeriod
    )
    $notifyDown = $false
    $notifyUp = $false
    $next = $CurrentNotifyState

    if ($InGracePeriod) {
        return @{
            NextNotifyState = 'Grace'
            NotifyDown        = $false
            NotifyUp          = $false
        }
    }

    if ($CurrentNotifyState -eq 'Grace') {
        if ($PingOk) {
            return @{
                NextNotifyState = 'Up'
                NotifyDown        = $false
                NotifyUp          = $false
            }
        }
        return @{
            NextNotifyState = 'Down'
            NotifyDown        = $true
            NotifyUp          = $false
        }
    }

    if ($PingOk) {
        if ($CurrentNotifyState -eq 'Down') {
            $notifyUp = $true
        }
        $next = 'Up'
    }
    else {
        if ($CurrentNotifyState -ne 'Down') {
            $notifyDown = $true
        }
        $next = 'Down'
    }

    return @{
        NextNotifyState = $next
        NotifyDown        = $notifyDown
        NotifyUp          = $notifyUp
    }
}

function Show-AceLocalMemuraiWindowsNotification {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet('Info', 'Warning')]
        [string]$Level = 'Warning'
    )
    $shown = $false
    try {
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        $escapedTitle = [System.Security.SecurityElement]::Escape($Title)
        $escapedMessage = [System.Security.SecurityElement]::Escape($Message)
        $xmlContent = @"
<toast duration="long">
  <visual>
    <binding template="ToastText02">
      <text id="1">$escapedTitle</text>
      <text id="2">$escapedMessage</text>
    </binding>
  </visual>
</toast>
"@
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml($xmlContent)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
        $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
        $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId)
        $notifier.Show($toast)
        $shown = $true
    }
    catch {
        $shown = $false
    }

    if ($shown) {
        return @{ Ok = $true; Method = 'toast' }
    }

    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        Add-Type -AssemblyName System.Drawing -ErrorAction Stop
        $icon = [System.Drawing.SystemIcons]::Information
        if ($Level -eq 'Warning') {
            $icon = [System.Drawing.SystemIcons]::Warning
        }
        $balloonIcon = [System.Windows.Forms.ToolTipIcon]::Info
        if ($Level -eq 'Warning') {
            $balloonIcon = [System.Windows.Forms.ToolTipIcon]::Warning
        }
        $ni = New-Object System.Windows.Forms.NotifyIcon
        $ni.Icon = $icon
        $ni.Visible = $true
        $ni.ShowBalloonTip(15000, $Title, $Message, $balloonIcon)
        Start-Sleep -Milliseconds 800
        $ni.Visible = $false
        $ni.Dispose()
        return @{ Ok = $true; Method = 'balloon' }
    }
    catch {
        return @{ Ok = $false; Method = 'none'; Error = $_.Exception.Message }
    }
}

function Get-AceLocalMemuraiAlertText {
    param([Parameter(Mandatory = $true)][ValidateSet('Down', 'Up')][string]$Kind)
    $utf8 = [System.Text.Encoding]::UTF8
    if ($Kind -eq 'Down') {
        return $utf8.GetString([Convert]::FromBase64String(
                'TWVtdXJhaSAoUmVkaXMpINeQ15nXoNeVINee15LXmdeRINecLVBJTkcg15EtMTI3LjAuMC4xOjYzNzkuIEJ1aWxkZXIyL1dvcmtlciDXotec15XXnNeZ150g15zXlNeZ15vXqdecLg=='))
    }
    return $utf8.GetString([Convert]::FromBase64String(
            'TWVtdXJhaSAoUmVkaXMpINeX15bXqCDXnNeU15LXmdeRIC0gUE9ORyDXqten15nXny4='))
}

function Invoke-AceLocalMemuraiDownAlert {
    Show-AceLocalMemuraiWindowsNotification -Title 'ACE LOCAL - Memurai' `
        -Message (Get-AceLocalMemuraiAlertText -Kind Down) `
        -Level Warning | Out-Null
}

function Invoke-AceLocalMemuraiUpAlert {
    Show-AceLocalMemuraiWindowsNotification -Title 'ACE LOCAL - Memurai' `
        -Message (Get-AceLocalMemuraiAlertText -Kind Up) `
        -Level Info | Out-Null
}
