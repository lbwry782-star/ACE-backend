#Requires -Version 5.1
# Minimal centered PROGRESS window (custom bar matching CANCEL/PUBLISH). Mutex prevents duplicates.
param(
    [Parameter(Mandatory = $true)][string]$StatePath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$mutex = New-Object System.Threading.Mutex($false, 'Global\AceLocalProgressUi')
$ownsMutex = $false
try {
    $ownsMutex = $mutex.WaitOne(0, $false)
}
catch {
    exit 0
}
if (-not $ownsMutex) {
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$TrackWidth = 360
$TrackHeight = 10
$CornerRadius = 6
$AnimDurationMs = 240
$ColorWindowBg = [System.Drawing.Color]::FromArgb(0xF7, 0xF6, 0xF2)
$ColorTrack = [System.Drawing.Color]::FromArgb(0xDE, 0xDE, 0xD5)
$ColorFill = [System.Drawing.Color]::FromArgb(0x1E, 0x5C, 0x3A)

function Get-EaseInOut([double]$t) {
    if ($t -le 0) { return 0.0 }
    if ($t -ge 1) { return 1.0 }
    if ($t -lt 0.5) {
        return 4.0 * $t * $t * $t
    }
    return 1.0 - [Math]::Pow(-2.0 * $t + 2.0, 3.0) / 2.0
}

function New-RoundedRectPath {
    param(
        [System.Drawing.RectangleF]$Rect,
        [float]$Radius
    )
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = [Math]::Min($Radius * 2.0, [Math]::Min($Rect.Width, $Rect.Height))
    if ($d -le 0) {
        $path.AddRectangle($Rect)
        return $path
    }
    $r = $d / 2.0
    $x = $Rect.X
    $y = $Rect.Y
    $w = $Rect.Width
    $h = $Rect.Height
    $path.StartFigure()
    $path.AddArc($x, $y, $d, $d, 180, 90)
    $path.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $path.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $path.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    return $path
}

$form = New-Object System.Windows.Forms.Form
$form.Text = 'PROGRESS'
$form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.ControlBox = $true
$form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$form.BackColor = $ColorWindowBg
$form.ClientSize = New-Object System.Drawing.Size($TrackWidth, 34)
$form.ShowInTaskbar = $true
$form.TopMost = $false

$trackPanel = New-Object System.Windows.Forms.Panel
$trackPanel.Location = New-Object System.Drawing.Point(0, 12)
$trackPanel.Size = New-Object System.Drawing.Size($TrackWidth, $TrackHeight)
$trackPanel.BackColor = $ColorWindowBg
$trackPanel.TabStop = $false

$script:DisplayRatio = 0.0
$script:TargetRatio = 0.0
$script:AnimFromRatio = 0.0
$script:AnimStartUtc = [datetime]::MinValue
$script:LastStatus = 'running'
$script:CompleteCloseTimer = $null

$trackPanel.Add_Paint({
        param($sender, $e)
        $g = $e.Graphics
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $trackRect = New-Object System.Drawing.RectangleF(0.0, 0.0, [float]$TrackWidth, [float]$TrackHeight)
        $trackPath = New-RoundedRectPath -Rect $trackRect -Radius ([float]$CornerRadius)
        $trackBrush = New-Object System.Drawing.SolidBrush($ColorTrack)
        try {
            $g.FillPath($trackBrush, $trackPath)
        }
        finally {
            $trackBrush.Dispose()
        }

        $ratio = [Math]::Max(0.0, [Math]::Min(1.0, $script:DisplayRatio))
        if ($ratio -gt 0.001) {
            $fillW = [float]$TrackWidth * $ratio
            if ($fillW -lt 1.0) { $fillW = 1.0 }
            $fillRect = New-Object System.Drawing.RectangleF(0.0, 0.0, $fillW, [float]$TrackHeight)
            $fillPath = New-RoundedRectPath -Rect $fillRect -Radius ([float]$CornerRadius)
            $fillBrush = New-Object System.Drawing.SolidBrush($ColorFill)
            try {
                $clipRegion = New-Object System.Drawing.Region($trackPath)
                $prevClip = $g.Clip
                $g.SetClip($clipRegion, [System.Drawing.Drawing2D.CombineMode]::Intersect)
                try {
                    $g.FillPath($fillBrush, $fillPath)
                }
                finally {
                    $g.Clip = $prevClip
                    $clipRegion.Dispose()
                }
            }
            finally {
                $fillBrush.Dispose()
                $fillPath.Dispose()
            }
        }
        $trackPath.Dispose()
    })

[void]$form.Controls.Add($trackPanel)

function Set-TargetRatio([double]$ratio) {
    $clamped = [Math]::Max(0.0, [Math]::Min(1.0, $ratio))
    if ([Math]::Abs($clamped - $script:TargetRatio) -lt 0.0001) {
        return
    }
    $script:AnimFromRatio = $script:DisplayRatio
    $script:TargetRatio = $clamped
    $script:AnimStartUtc = [DateTime]::UtcNow
}

$pollTimer = New-Object System.Windows.Forms.Timer
$pollTimer.Interval = 150

$animTimer = New-Object System.Windows.Forms.Timer
$animTimer.Interval = 15

$animTimer.Add_Tick({
        $elapsed = ([DateTime]::UtcNow - $script:AnimStartUtc).TotalMilliseconds
        if ($script:AnimStartUtc -eq [datetime]::MinValue) {
            return
        }
        $t = $elapsed / [double]$AnimDurationMs
        if ($t -ge 1.0) {
            $script:DisplayRatio = $script:TargetRatio
            $script:AnimStartUtc = [datetime]::MinValue
            $trackPanel.Invalidate()
            return
        }
        $eased = Get-EaseInOut -t $t
        $script:DisplayRatio = $script:AnimFromRatio + (($script:TargetRatio - $script:AnimFromRatio) * $eased)
        $trackPanel.Invalidate()
    })

$pollTimer.Add_Tick({
        try {
            if (-not (Test-Path -LiteralPath $StatePath)) {
                return
            }
            $raw = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8
            if ([string]::IsNullOrWhiteSpace($raw)) {
                return
            }
            $state = $raw | ConvertFrom-Json
            $pct = [int]$state.percent
            if ($pct -lt 0) { $pct = 0 }
            if ($pct -gt 100) { $pct = 100 }
            Set-TargetRatio -ratio ($pct / 100.0)

            $st = [string]$state.status
            if ($st -eq 'complete' -and $script:LastStatus -ne 'complete') {
                $script:LastStatus = 'complete'
                Set-TargetRatio -ratio 1.0
                $pollTimer.Stop()
                if ($null -ne $script:CompleteCloseTimer) {
                    try {
                        $script:CompleteCloseTimer.Stop()
                        $script:CompleteCloseTimer.Dispose()
                    }
                    catch { }
                    $script:CompleteCloseTimer = $null
                }
                $script:CompleteCloseTimer = New-Object System.Windows.Forms.Timer
                $script:CompleteCloseTimer.Interval = $AnimDurationMs + 80
                $script:CompleteCloseTimer.Add_Tick({
                        try {
                            if ($null -ne $script:CompleteCloseTimer) {
                                $script:CompleteCloseTimer.Stop()
                            }
                            $animTimer.Stop()
                            if (-not $form.IsDisposed) {
                                $form.Close()
                            }
                        }
                        catch {
                            # Avoid surfacing WinForms timer errors to the user.
                        }
                    })
                $script:CompleteCloseTimer.Start()
            }
            elseif ($st -eq 'error' -and $script:LastStatus -ne 'error') {
                $script:LastStatus = 'error'
                $pollTimer.Stop()
                $animTimer.Stop()
                $form.Close()
            }
        }
        catch {
            # Keep polling on transient read errors.
        }
    })

$form.Add_FormClosed({
        if ($null -ne $script:CompleteCloseTimer) {
            try {
                $script:CompleteCloseTimer.Stop()
                $script:CompleteCloseTimer.Dispose()
            }
            catch { }
            $script:CompleteCloseTimer = $null
        }
        try {
            $pollTimer.Stop()
            $pollTimer.Dispose()
        }
        catch { }
        try {
            $animTimer.Stop()
            $animTimer.Dispose()
        }
        catch { }
        if ($ownsMutex) {
            try { [void]$mutex.ReleaseMutex() } catch { }
        }
        try { $mutex.Dispose() } catch { }
    })

[void]$form.Add_Shown({
        $pollTimer.Start()
        $animTimer.Start()
    })

[void][System.Windows.Forms.Application]::Run($form)
