#Requires -Version 5.1
# Creates desktop shortcut for unified local dev launcher (UTF-8 name, silent headless start).
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LauncherPs1 = Join-Path $ScriptDir "Start-AceLocal.ps1"
if (-not (Test-Path -LiteralPath $LauncherPs1)) {
    Write-Error "Start-AceLocal.ps1 not found."
    exit 1
}

$psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
if (-not (Test-Path -LiteralPath $psExe)) {
    $psExe = 'powershell.exe'
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutName = [string]::Concat(
    [char]0x05D0, [char]0x05D5, [char]0x05E8, [char]0x05D9, " - ",
    [char]0x05E4, [char]0x05D9, [char]0x05EA, [char]0x05D5, [char]0x05D7, " ",
    [char]0x05DE, [char]0x05E7, [char]0x05D5, [char]0x05DE, [char]0x05D9
)
$ShortcutPath = Join-Path $Desktop ($ShortcutName + ".lnk")

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $psExe
$Shortcut.Arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $LauncherPs1 + '"'
$Shortcut.WorkingDirectory = $ScriptDir
$Shortcut.WindowStyle = 7
$Shortcut.Description = "ACE Ori local development (silent start)"
$Shortcut.Save()

Write-Host ("Shortcut created: " + $ShortcutPath) -ForegroundColor Green
Write-Host "Diagnostic mode: run Start-AceLocal.ps1 with -ShowComponentWindows" -ForegroundColor DarkGray
