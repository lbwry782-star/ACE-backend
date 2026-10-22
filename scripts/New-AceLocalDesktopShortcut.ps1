#Requires -Version 5.1
# Creates desktop shortcut for unified local dev launcher (UTF-8 name).
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $ScriptDir "Start-AceLocal.cmd"
if (-not (Test-Path -LiteralPath $Launcher)) {
    Write-Error "Start-AceLocal.cmd not found."
    exit 1
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
$Shortcut.TargetPath = $Launcher
$Shortcut.WorkingDirectory = $ScriptDir
$Shortcut.WindowStyle = 1
$Shortcut.Description = "ACE Ori local development"
$Shortcut.Save()

Write-Host ("Shortcut created: " + $ShortcutPath) -ForegroundColor Green
