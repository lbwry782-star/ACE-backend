#Requires -Version 5.1
<#
.SYNOPSIS
  Start full ACE local dev stack (Memurai, Backend, Worker, Frontend).

.PARAMETER ValidateOnly
  Safety checks and status report only (no starts, no browser).

.PARAMETER SkipLocalUndoShell
  Do not launch the standalone LOCAL undo window after Vite is ready.

.PARAMETER ShowComponentWindows
  Open visible CMD windows for Backend, Worker, and Frontend (diagnostics only).
  Default is silent headless start with log files under _backups\local-dev\logs.
#>
[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [switch]$SkipLocalUndoShell,
    [switch]$ShowComponentWindows
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "AceLocalEnvCommon.ps1")
. (Join-Path $ScriptDir "AceLocalManagedProcessLib.ps1")
. (Join-Path $ScriptDir "AceLocalProgressLib.ps1")
. (Join-Path $ScriptDir "AceLocalMemuraiInstructionLib.ps1")
. (Join-Path $ScriptDir "AceLocalRecoveryStartupGateLib.ps1")
. (Join-Path $ScriptDir "AceLocalRecoveryPeriodicSaveBackgroundLib.ps1")
. (Join-Path $ScriptDir "AceLocalRecoveryLegacyBootstrapLib.ps1")

$ExpectedRedisUrl = "redis://127.0.0.1:6379/0"
$BackendPort = 5000
$DefaultPublicBase = "http://127.0.0.1:5000"
$FrontendPortMin = 3000
$FrontendPortMax = 3010

$SilentMode = -not $ShowComponentWindows
$LauncherLogPath = ""
$TranscriptStarted = $false
$ProgressStatePath = ""
$ProgressUiEnabled = $false
$MemuraiStartupFailureDetail = ""
$MemuraiInstructionShown = $false

function Stop-AceLocalLauncherTranscript {
    if ($script:TranscriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
        $script:TranscriptStarted = $false
    }
}

function Invoke-AceLocalFailure {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [int]$ExitCode = 1,
        [switch]$SkipUserAlert
    )
    Write-Err $Message
    if ($script:ProgressUiEnabled -and $script:ProgressStatePath) {
        Fail-AceLocalProgressState -StatePath $script:ProgressStatePath
    }
    Stop-AceLocalLauncherTranscript
    if ($SilentMode -and -not $ValidateOnly -and -not $SkipUserAlert) {
        Show-AceLocalUserAlert -Message $Message -LogPath $script:LauncherLogPath
    }
    elseif (-not $ValidateOnly -and $ShowComponentWindows) {
        Read-Host "Press Enter to close"
    }
    exit $ExitCode
}

$BackendRoot = Resolve-AceBackendRoot -StartPath $ScriptDir
if (-not $BackendRoot) {
    Write-Err "ACE-Backend folder not found (app.py + worker_video.py)."
    exit 1
}
$FrontendRoot = Resolve-AceFrontendRoot -BackendRoot $BackendRoot
if (-not $FrontendRoot) {
    Write-Err "ACE-Frontend folder not found (package.json)."
    exit 1
}
$AceRoot = Split-Path -Parent $BackendRoot
$LogDir = Get-AceLocalLogDir -AceRoot $AceRoot
$ProgressUiEnabled = $SilentMode -and -not $ValidateOnly
if ($ProgressUiEnabled) {
    $script:ProgressUiEnabled = $true
    $script:ProgressStatePath = Get-AceLocalProgressStatePath -AceRoot $AceRoot
    Initialize-AceLocalProgressState -StatePath $script:ProgressStatePath
    $progressScript = Join-Path $ScriptDir 'AceLocalProgressWindow.ps1'
    [void](Start-AceLocalProgressUiProcess -StatePath $script:ProgressStatePath -WindowScriptPath $progressScript)
    Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'Init'
}

$BackendStartCmd = Join-Path $BackendRoot "scripts\Start-AceBackendLocal.cmd"
$WorkerStartCmd = Join-Path $BackendRoot "scripts\Start-AceWorkerLocal.cmd"
$FrontendStartCmd = Join-Path $FrontendRoot "scripts\Start-AceFrontendLocal.cmd"
$BackendStartPs1 = Join-Path $BackendRoot "scripts\Start-AceBackendLocal.ps1"
$WorkerStartPs1 = Join-Path $BackendRoot "scripts\Start-AceWorkerLocal.ps1"
$FrontendStartPs1 = Join-Path $FrontendRoot "scripts\Start-AceFrontendLocal.ps1"

foreach ($path in @($BackendStartCmd, $WorkerStartCmd, $FrontendStartCmd, $BackendStartPs1, $WorkerStartPs1, $FrontendStartPs1)) {
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Err "Missing launcher: $path"
        exit 1
    }
}

function Test-LocalDevSafety([string]$RepoRoot, [string]$FrontendRoot) {
    Assert-LocalDevEnvFile -RepoRoot $RepoRoot
    Assert-LocalRedisUrl -ExpectedUrl $ExpectedRedisUrl
    Assert-LocalPublicBaseUrl -ExpectedBase $DefaultPublicBase -BackendPort $BackendPort
    $feEnv = Join-Path $FrontendRoot ".env.development.local"
    if (Test-Path -LiteralPath $feEnv) {
        Import-DotEnvFile $feEnv | Out-Null
    }
    $viteUrl = Get-EnvString "VITE_BACKEND_URL"
    if ($viteUrl -and (Test-ProductionPublicBaseUrl $viteUrl)) {
        Write-Err "VITE_BACKEND_URL points at production. Stopped."
        exit 1
    }
    if ($viteUrl) {
        try {
            $uri = [Uri]$viteUrl
            if (-not (Test-LoopbackHost $uri.Host)) {
                Write-Err "VITE_BACKEND_URL must use loopback (127.0.0.1), not production."
                exit 1
            }
        }
        catch {
            Write-Err "VITE_BACKEND_URL is invalid."
            exit 1
        }
    }
}

function Ensure-MemuraiRunning {
    param([switch]$AllowInteractive)
    $script:MemuraiStartupFailureDetail = ''
    $diag = Get-AceLocalMemuraiDiagnosis -RedisUrl $ExpectedRedisUrl
    Write-AceLocalMemuraiAvailabilityState -AceRoot $AceRoot -Diagnosis $diag | Out-Null
    if ($diag.PingOk) {
        Write-Ok 'Memurai: PONG (127.0.0.1:6379)'
        return $true
    }

    $pingErr = if ($diag.PingError) { [string]$diag.PingError } else { 'unknown' }
    $svcStatus = [string]$diag.WindowsServiceStatus
    Write-Warn ("Memurai: no PONG at {0} ({1}) scenario={2}" -f '127.0.0.1:6379', $pingErr, $diag.Scenario)
    if ($svcStatus) {
        Write-Info ("Memurai service status: {0}" -f $svcStatus)
    }

    switch ([string]$diag.Scenario) {
        'ServiceNotInstalled' {
            $script:MemuraiStartupFailureDetail = @(
                'Redis PING failed (127.0.0.1:6379).'
                ("PING error: {0}" -f $pingErr)
                'Windows service "Memurai" was not found.'
                'Install or start Memurai Developer manually, then run LOCAL again.'
            ) -join "`n"
            Write-Err 'Windows service Memurai not found. Start Memurai Developer manually.'
        }
        'ServiceRunningNoPong' {
            $script:MemuraiStartupFailureDetail = @(
                'Memurai Windows service is Running but Redis PING failed.'
                ("PING error: {0}" -f $pingErr)
                'Check Memurai configuration, port 6379, and firewall.'
            ) -join "`n"
            Write-Err 'Memurai service is running but Redis PING failed. Check Memurai configuration.'
        }
        'ServiceStopped' {
            $script:MemuraiStartupFailureDetail = @(
                'Memurai/Redis is not responding on 127.0.0.1:6379.'
                ("PING error: {0}" -f $pingErr)
                ("Windows service status: {0}" -f $svcStatus)
                'Start the Memurai Windows service manually (MEMURAI instructions), then run LOCAL again.'
            ) -join "`n"
            Write-Err 'Memurai is not responding. Start the Memurai Windows service manually, then run LOCAL again.'
        }
        default {
            $script:MemuraiStartupFailureDetail = @(
                'Memurai check failed without proving the Windows service stopped.'
                ("PING error: {0}" -f $pingErr)
                ("Scenario: {0}" -f $diag.Scenario)
            ) -join "`n"
            Write-Err 'Memurai check failed. Verify Memurai installation and Redis on 127.0.0.1:6379.'
        }
    }

    if ($ValidateOnly) {
        Write-Warn 'ValidateOnly: Memurai/Redis is not reachable.'
        return $false
    }

    if ($diag.ShowInstruction) {
        $null = Invoke-AceLocalMemuraiUserInstruction -AceRoot $AceRoot -RedisUrl $ExpectedRedisUrl -StartupAttempt
        $script:MemuraiInstructionShown = $true
    }
    return $false
}

function Start-AceLocalMemuraiWatchAsync {
    param(
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [Parameter(Mandatory = $true)][string]$RedisUrl
    )
    $watchScript = Join-Path $ScriptDir 'Start-AceLocalMemuraiWatch.ps1'
    if (-not (Test-Path -LiteralPath $watchScript)) {
        Write-Warn 'Memurai watch: script missing; monitoring not started.'
        return
    }
    . (Join-Path $ScriptDir 'AceLocalMemuraiWatchLib.ps1')
    $aceRoot = Split-Path -Parent $BackendRoot
    $pidPath = Get-AceLocalMemuraiWatchPidPath -AceRoot $aceRoot -AllowCreateStateDir
    Stop-AceLocalMemuraiWatchProcess -PidPath $pidPath | Out-Null
    try {
        $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
        if (-not (Test-Path -LiteralPath $psExe)) {
            $psExe = 'powershell.exe'
        }
        $psArgs = @(
            '-NoProfile',
            '-ExecutionPolicy', 'Bypass',
            '-WindowStyle', 'Hidden',
            '-File', $watchScript,
            '-BackendRoot', $BackendRoot,
            '-RedisUrl', $RedisUrl
        )
        Start-Process -FilePath $psExe -ArgumentList $psArgs -WorkingDirectory $ScriptDir -WindowStyle Hidden -ErrorAction Stop | Out-Null
        Write-Ok 'Memurai watch: background PING monitor started (30s, Windows alerts on state change).'
    }
    catch {
        Write-Warn ("Memurai watch: could not start ({0})." -f $_.Exception.Message)
    }
}

function Start-AceLocalUndoShellAsync {
    param(
        [Parameter(Mandatory = $true)][int]$VitePort,
        [Parameter(Mandatory = $true)][string]$FrontendRoot
    )
    $undoScript = Join-Path $FrontendRoot 'scripts\Start-AceUndoLocal.ps1'
    if (-not (Test-Path -LiteralPath $undoScript)) {
        Write-Warn ('LOCAL shell: script not found ({0}). Dev stack continues.' -f $undoScript)
        return
    }
    if (@(Get-Process -Name 'AceUndo' -ErrorAction SilentlyContinue).Count -gt 0) {
        Write-Ok 'LOCAL shell: AceUndo already running; skipping duplicate launch.'
        return
    }
    if (-not (Wait-AceFrontendViteOnPort -Port $VitePort -TotalTimeoutSec 10 -RequestTimeoutSec 2)) {
        Write-Warn ("LOCAL shell: ACE Vite not verified on port {0}. LOCAL was not started." -f $VitePort)
        return
    }
    try {
        $psExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
        if (-not (Test-Path -LiteralPath $psExe)) {
            $psExe = 'powershell.exe'
        }
        $psArgs = @(
            '-NoProfile',
            '-ExecutionPolicy', 'Bypass',
            '-WindowStyle', 'Hidden',
            '-File', $undoScript,
            '-VitePort', ([string]$VitePort)
        )
        Start-Process -FilePath $psExe -ArgumentList $psArgs -WorkingDirectory $FrontendRoot -WindowStyle Hidden -ErrorAction Stop | Out-Null
        Write-Ok ("LOCAL shell: launch requested (Vite port {0})." -f $VitePort)
    }
    catch {
        Write-Warn ('LOCAL shell: launch failed ({0}). Dev stack continues.' -f $_.Exception.Message)
    }
}

function Start-ComponentWindow([string]$Title, [string]$CmdPath, [string]$WorkingDirectory, [hashtable]$ExtraEnv) {
    $envLines = ""
    if ($ExtraEnv) {
        foreach ($key in $ExtraEnv.Keys) {
            $val = $ExtraEnv[$key]
            $envLines += ('set {0}={1}&& ' -f $key, $val)
        }
    }
    $inner = '{0}call "{1}"' -f $envLines, $CmdPath
    $arg = '/k title ' + $Title + ' && ' + $inner
    Start-Process -FilePath "cmd.exe" -ArgumentList $arg -WorkingDirectory $WorkingDirectory | Out-Null
}

function Start-ComponentSilent {
    param(
        [Parameter(Mandatory = $true)][string]$ComponentName,
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [hashtable]$ExtraEnv = $null
    )
    $stdoutLog = Join-Path $LogDir ("{0}.stdout.log" -f $ComponentName.ToLowerInvariant())
    $stderrLog = Join-Path $LogDir ("{0}.stderr.log" -f $ComponentName.ToLowerInvariant())
    return Start-AceLocalHiddenPowerShell -ScriptPath $ScriptPath -WorkingDirectory $WorkingDirectory `
        -StdoutLog $stdoutLog -StderrLog $stderrLog -ExtraEnv $ExtraEnv
}

if (-not $ValidateOnly) {
    $script:LauncherLogPath = Join-Path $LogDir 'launcher.log'
    try {
        Start-Transcript -LiteralPath $script:LauncherLogPath -Force | Out-Null
        $script:TranscriptStarted = $true
    }
    catch {
        Write-Warn ("Could not start launcher transcript: {0}" -f $_.Exception.Message)
    }
}

Write-Info "=== ACE Ori - local dev stack ==="
Write-Info ("Backend: {0}" -f $BackendRoot)
Write-Info ("Frontend: {0}" -f $FrontendRoot)
if ($SilentMode) {
    Write-Info ("Silent mode: logs in {0}" -f $LogDir)
}
else {
    Write-Warn "Diagnostic mode: component CMD windows will open."
}

Push-Location $BackendRoot
try {
    Test-LocalDevSafety -RepoRoot $BackendRoot -FrontendRoot $FrontendRoot
}
finally {
    Pop-Location
}

$script:RecoveryStartupGateResult = $null
try {
    $script:RecoveryStartupGateResult = Invoke-AceLocalRecoveryStartupGateCheck -AceRoot $AceRoot -ValidateOnly:$ValidateOnly
    Write-AceLocalRecoveryStartupGateSummary -GateResult $script:RecoveryStartupGateResult
    Assert-AceLocalRecoveryStartupGateForLaunch -GateResult $script:RecoveryStartupGateResult -ValidateOnly:$ValidateOnly
}
catch {
    if ($ValidateOnly) {
        Write-Err $_.Exception.Message
        Stop-AceLocalLauncherTranscript
        exit 1
    }
    Invoke-AceLocalFailure -Message $_.Exception.Message
}

$assignNewLauncherSession = $false
if ($script:RecoveryStartupGateResult -and $script:RecoveryStartupGateResult.assignNewLauncherSession) {
    $assignNewLauncherSession = $true
}

$managedRegistry = Read-AceLocalManagedRegistry -AceRoot $AceRoot
if (-not $managedRegistry) {
    $managedRegistry = New-AceLocalManagedRegistryShell -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
}
else {
    [void](Remove-StaleManagedComponents -Registry $managedRegistry -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot)
    if ($assignNewLauncherSession) {
        $managedRegistry.launcherSession = (Get-Date).ToString('yyyyMMdd-HHmmss')
    }
    $managedRegistry.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
}
Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $managedRegistry

$memuraiOk = Ensure-MemuraiRunning -AllowInteractive:($ShowComponentWindows)
if (-not $memuraiOk -and -not $ValidateOnly) {
    $memuraiUserMsg = "Memurai/Redis is not available. LOCAL cannot start."
    if ($script:MemuraiStartupFailureDetail) {
        $memuraiUserMsg = "{0}`n`n{1}" -f $memuraiUserMsg, $script:MemuraiStartupFailureDetail
    }
    Invoke-AceLocalFailure -Message $memuraiUserMsg -SkipUserAlert:($script:MemuraiInstructionShown)
}
if ($ProgressUiEnabled -and $memuraiOk) {
    Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'SafetyMemurai'
}

$backendPortState = Get-AceBackendLocalPortState -Port $BackendPort -BackendRoot $BackendRoot
$backendOk = $false
if ($backendPortState.Action -eq "AlreadyRunning") {
    $backendOk = (Test-BackendLocalDevIdentity -Port $BackendPort).Ok
}
elseif ($backendPortState.Action -ne "Start") {
    Write-Err ("Backend port/process state blocks stack start: {0} ({1})" -f $backendPortState.Action, $backendPortState.Reason)
    Write-AceBackendLocalPortStateReport -State $backendPortState
    if (-not $ValidateOnly) {
        Invoke-AceLocalFailure -Message "Backend port/process state blocks LOCAL start. See launcher.log."
    }
}
$backendProcs = @(Find-LocalAceBackendProcesses -BackendRoot $BackendRoot)
$workerProcs = @(Find-LocalAceWorkerProcesses -BackendRoot $BackendRoot)
$frontendPorts = @(Find-AceFrontendVitePorts -FromPort $FrontendPortMin -ToPort $FrontendPortMax)

if ($frontendPorts.Count -gt 1) {
    Write-Warn ("Multiple ACE Vite servers detected (ports: {0}). None were stopped." -f ($frontendPorts -join ", "))
}

if ($ValidateOnly) {
    if ($script:RecoveryStartupGateResult) {
        Write-Host ("Recovery coordinator: " + [string]$script:RecoveryStartupGateResult.coordinatorOutcome)
    }
    Write-Info "--- ValidateOnly: component status ---"
    Write-Host ("Memurai PONG: " + $(if ($memuraiOk) { "OK" } else { "FAILED" }))
    Write-Host ("Backend port decision: " + $backendPortState.Action + " (" + $backendPortState.Reason + ")")
    Write-Host ("Backend ACE identity: " + $(if ($backendOk) { "OK" } else { "missing / not verified" }))
    Write-Host ("Backend app.py processes in repo path: " + $backendProcs.Count)
    Write-Host ("Worker worker_video.py: " + $(if ($workerProcs.Count -gt 0) { "running ($($workerProcs.Count))" } else { "not running" }))
    Write-Host ("Frontend ACE Vite: " + $(if ($frontendPorts.Count -gt 0) { "port $($frontendPorts[0])" } else { "not running" }))
    if ($frontendPorts.Count -gt 1) {
        Write-Warn "Two or more Vite ports matched ACE frontend fingerprint."
    }
    Write-Ok "ValidateOnly: env safety OK; nothing started."
    Stop-AceLocalLauncherTranscript
    if (-not $memuraiOk) {
        Write-Warn "ValidateOnly: Memurai not reachable (full start would stop here)."
        exit 1
    }
    exit 0
}

$started = @()

if ($backendOk) {
    Write-Ok ("Backend: already running (127.0.0.1:{0}, local dev verified)" -f $BackendPort)
    $managedRegistry.components.backend = $null
    Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $managedRegistry
    if ($ProgressUiEnabled) {
        Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'Backend'
    }
}
elseif ($backendPortState.Action -ne "Start") {
    Invoke-AceLocalFailure -Message "Backend: not started (port/process decision blocked start)."
}
else {
    $beforeBackendProcs = @(Find-LocalAceBackendProcesses -BackendRoot $BackendRoot)
    $backendLaunchWrapper = 0
    if ($ShowComponentWindows) {
        Write-Info "Backend: starting new window..."
        Start-ComponentWindow -Title "ACE Backend Local" -CmdPath $BackendStartCmd -WorkingDirectory $BackendRoot -ExtraEnv @{}
    }
    else {
        Write-Info "Backend: starting headless..."
        $backendLaunch = Start-ComponentSilent -ComponentName 'backend' -ScriptPath $BackendStartPs1 -WorkingDirectory $BackendRoot
        if ($backendLaunch) {
            $backendLaunchWrapper = [int]$backendLaunch.WrapperProcessId
        }
    }
    if (Wait-BackendLocalReady -Port $BackendPort -TimeoutSec 90) {
        Write-Ok "Backend: ready."
        $started += "Backend"
        $backendOk = $true
        if ($ProgressUiEnabled) {
            Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'Backend'
        }
        $discovered = Discover-AceLocalManagedBackendTree -WrapperPid $backendLaunchWrapper `
            -BeforeProcesses $beforeBackendProcs -BackendRoot $BackendRoot -ListenerPort $BackendPort
        if ($discovered) {
            $logPath = Join-Path $LogDir 'backend.stdout.log'
            if (-not (Register-ManagedComponentTree -Registry $managedRegistry -AceRoot $AceRoot -Name 'backend' -Discovery $discovered -LogPath $logPath)) {
                Write-Warn 'Backend: could not register managed process tree.'
            }
        }
        else {
            Write-Warn 'Backend: could not attribute managed process tree (not registered for stop).'
        }
    }
    else {
        $logHint = if ($SilentMode) { Join-Path $LogDir 'backend.stderr.log' } else { 'ACE Backend window' }
        Invoke-AceLocalFailure -Message ("Backend: timed out. Check log: {0}" -f $logHint)
    }
}

if ($workerProcs.Count -gt 0) {
    Write-Ok ("Worker: already running (PID {0})" -f $workerProcs[0].ProcId)
    $managedRegistry.components.worker = $null
    Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $managedRegistry
    if ($ProgressUiEnabled) {
        Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'Worker'
    }
}
else {
    $beforeWorkerProcs = @(Find-LocalAceWorkerProcesses -BackendRoot $BackendRoot)
    $workerLaunchWrapper = 0
    if ($ShowComponentWindows) {
        Write-Info "Worker: starting new window..."
        Start-ComponentWindow -Title "ACE Worker Local" -CmdPath $WorkerStartCmd -WorkingDirectory $BackendRoot -ExtraEnv @{}
    }
    else {
        Write-Info "Worker: starting headless..."
        $workerLaunch = Start-ComponentSilent -ComponentName 'worker' -ScriptPath $WorkerStartPs1 -WorkingDirectory $BackendRoot
        if ($workerLaunch) {
            $workerLaunchWrapper = [int]$workerLaunch.WrapperProcessId
        }
    }
    Start-Sleep -Seconds 3
    $workerProcs = @(Find-LocalAceWorkerProcesses -BackendRoot $BackendRoot)
    if ($workerProcs.Count -gt 0) {
        Write-Ok ("Worker: running (PID {0})" -f $workerProcs[0].ProcId)
        $started += "Worker"
        if ($ProgressUiEnabled) {
            Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'Worker'
        }
        $discovered = Discover-AceLocalManagedWorkerTree -WrapperPid $workerLaunchWrapper `
            -BeforeProcesses $beforeWorkerProcs -BackendRoot $BackendRoot
        if ($discovered) {
            $logPath = Join-Path $LogDir 'worker.stdout.log'
            if (-not (Register-ManagedComponentTree -Registry $managedRegistry -AceRoot $AceRoot -Name 'worker' -Discovery $discovered -LogPath $logPath)) {
                Write-Warn 'Worker: could not register managed process tree.'
            }
        }
        else {
            Write-Warn 'Worker: could not attribute managed process tree (not registered for stop).'
        }
    }
    else {
        $logHint = if ($SilentMode) { Join-Path $LogDir 'worker.stderr.log' } else { 'ACE Worker window' }
        Write-Warn ("Worker: not detected after start. Check log: {0}" -f $logHint)
    }
}

$frontendPort = $null
if ($frontendPorts.Count -ge 1) {
    $frontendPort = $frontendPorts[0]
    Write-Ok ("Frontend: already running (http://localhost:{0}/)" -f $frontendPort)
    $managedRegistry.components.frontend = $null
    Write-AceLocalManagedRegistry -AceRoot $AceRoot -RegistryObject $managedRegistry
    if ($ProgressUiEnabled) {
        Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'Frontend'
    }
}
else {
    $beforeFrontendProcs = @(Find-LocalAceFrontendViteProcesses -FrontendRoot $FrontendRoot)
    $frontendLaunchWrapper = 0
    if ($ShowComponentWindows) {
        Write-Info "Frontend: starting new window..."
        Start-ComponentWindow -Title "ACE Frontend Local" -CmdPath $FrontendStartCmd -WorkingDirectory $FrontendRoot -ExtraEnv @{ ACE_FRONTEND_LOCAL_NO_PAUSE = "1" }
    }
    else {
        Write-Info "Frontend: starting headless..."
        $frontendLaunch = Start-ComponentSilent -ComponentName 'frontend' -ScriptPath $FrontendStartPs1 -WorkingDirectory $FrontendRoot -ExtraEnv @{ ACE_FRONTEND_LOCAL_NO_PAUSE = '1' }
        if ($frontendLaunch) {
            $frontendLaunchWrapper = [int]$frontendLaunch.WrapperProcessId
        }
    }
    $frontendPort = Wait-AceFrontendVite -FromPort $FrontendPortMin -ToPort $FrontendPortMax -TimeoutSec 120
    if ($null -ne $frontendPort) {
        Write-Ok ("Frontend: ready (http://localhost:{0}/)" -f $frontendPort)
        $started += "Frontend"
        if ($ProgressUiEnabled) {
            Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'Frontend'
        }
        $discovered = Discover-AceLocalManagedFrontendTree -WrapperPid $frontendLaunchWrapper `
            -BeforeProcesses $beforeFrontendProcs -FrontendRoot $FrontendRoot
        if ($discovered) {
            $logPath = Join-Path $LogDir 'frontend.stdout.log'
            if (-not (Register-ManagedComponentTree -Registry $managedRegistry -AceRoot $AceRoot -Name 'frontend' -Discovery $discovered -LogPath $logPath)) {
                Write-Warn 'Frontend: could not register managed process tree.'
            }
        }
        else {
            Write-Warn 'Frontend: could not attribute managed process tree (not registered for stop).'
        }
    }
    else {
        $logHint = if ($SilentMode) { Join-Path $LogDir 'frontend.stderr.log' } else { 'Vite window' }
        Invoke-AceLocalFailure -Message ("Frontend: timed out. Check log: {0}" -f $logHint)
    }
}

if (-not $backendOk) {
    Invoke-AceLocalFailure -Message "Backend not verified. LOCAL did not start successfully."
}

$managedRegistry = Read-AceLocalManagedRegistry -AceRoot $AceRoot
if ($script:RecoveryStartupGateResult -and [string]$script:RecoveryStartupGateResult.coordinatorOutcome -eq 'LEGACY_RECOVERY_BOOTSTRAP') {
    try {
        $legacyBootstrap = Start-AceLocalRecoveryLegacyBootstrapBackground -AceRoot $AceRoot `
            -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot
        if ($legacyBootstrap.Started) {
            Write-Info 'Legacy recovery bootstrap: first full checkpoint running in background.'
        }
        elseif ([string]$legacyBootstrap.Reason -ne 'bootstrap_lock_held') {
            Write-Warn ("Legacy recovery bootstrap: not started ({0})." -f [string]$legacyBootstrap.Reason)
        }
    }
    catch {
        Write-Warn ("Legacy recovery bootstrap: start failed ({0})." -f $_.Exception.Message)
    }
}

try {
    $periodicSaveStart = Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart `
        -BackendRoot $BackendRoot -FrontendRoot $FrontendRoot -AceRoot $AceRoot `
        -BackendPort $BackendPort -GateResult $script:RecoveryStartupGateResult `
        -ManagedRegistry $managedRegistry -EngineBackendRoot $BackendRoot
    if ($periodicSaveStart.Started) {
        Write-Info 'Recovery periodic save: background service started (silent).'
    }
    elseif ([string]$periodicSaveStart.Reason -notin @('activation_denied', 'coordinator_blocked', 'already_running')) {
        Write-Warn ("Recovery periodic save: not started ({0})." -f [string]$periodicSaveStart.Reason)
    }
}
catch {
    Write-Warn ("Recovery periodic save: start skipped ({0})." -f $_.Exception.Message)
}

Start-AceLocalMemuraiWatchAsync -BackendRoot $BackendRoot -RedisUrl $ExpectedRedisUrl

if (-not $SkipLocalUndoShell) {
    Start-AceLocalUndoShellAsync -VitePort $frontendPort -FrontendRoot $FrontendRoot
    if ($ProgressUiEnabled) {
        Set-AceLocalProgressStep -StatePath $script:ProgressStatePath -Step 'LocalUi'
    }
}

$frontendUrl = "http://localhost:$frontendPort/"
Write-Info ("Opening browser: {0}" -f $frontendUrl)
Start-Process $frontendUrl | Out-Null
if ($ProgressUiEnabled) {
    Complete-AceLocalProgressState -StatePath $script:ProgressStatePath
    Start-Sleep -Milliseconds 250
}

Write-Host ""
Write-Ok "=== Local stack ready ==="
if ($started.Count -gt 0) {
    Write-Info ("Started now: {0}" -f ($started -join ", "))
}
else {
    Write-Info "All components were already running."
}
Write-Info ("Site: {0}" -f $frontendUrl)
Write-Info ("Backend health: http://127.0.0.1:{0}/health" -f $BackendPort)
if ($SilentMode) {
    Write-Info ("Logs: {0}" -f $LogDir)
    Write-Info "Stop managed stack: scripts\Stop-AceLocal.ps1"
}
else {
    Write-Warn "Keep Backend, Worker, and Frontend windows open."
    Write-Host ""
    Read-Host "Press Enter to close this launcher (servers keep running)"
}

Stop-AceLocalLauncherTranscript
exit 0
