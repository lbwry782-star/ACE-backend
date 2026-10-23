# Shared helpers for ACE local Backend/Worker startup scripts (Windows).
Set-StrictMode -Version Latest

function Write-Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}
function Write-Ok([string]$Message) {
    Write-Host $Message -ForegroundColor Green
}
function Write-Warn([string]$Message) {
    Write-Host $Message -ForegroundColor Yellow
}
function Write-Err([string]$Message) {
    Write-Host $Message -ForegroundColor Red
}

function Get-EnvString([string]$Name) {
    $item = Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
    if (-not $item) {
        return ""
    }
    return ([string]$item.Value).Trim()
}

function Import-DotEnvFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    Get-Content -LiteralPath $Path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }
        $eq = $line.IndexOf("=")
        if ($eq -lt 1) {
            return
        }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim()
        if ($value.StartsWith('"') -and $value.EndsWith('"') -and $value.Length -ge 2) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path "Env:$name" -Value $value
    }
    return $true
}

function Test-LoopbackHost([string]$HostName) {
    if ([string]::IsNullOrWhiteSpace($HostName)) {
        return $false
    }
    $h = $HostName.Trim().ToLowerInvariant()
    if ($h -in @("127.0.0.1", "localhost", "::1")) {
        return $true
    }
    if ($h.StartsWith("127.")) {
        return $true
    }
    return $false
}

function Get-RedisHostFromUrl([string]$Url) {
    if ([string]::IsNullOrWhiteSpace($Url)) {
        return ""
    }
    try {
        $uri = [Uri]$Url
        return $uri.Host
    }
    catch {
        return "__invalid__"
    }
}

function Test-ProductionPublicBaseUrl([string]$Url) {
    if ([string]::IsNullOrWhiteSpace($Url)) {
        return $false
    }
    try {
        $uri = [Uri]($(if ($Url -match "://") { $Url } else { "https://$Url" }))
        $pubHost = $uri.Host.ToLowerInvariant()
    }
    catch {
        return $true
    }
    $blocked = @(
        "ace-advertising.agency",
        "www.ace-advertising.agency",
        "onrender.com",
        "render.com"
    )
    foreach ($b in $blocked) {
        if ($pubHost -eq $b -or $pubHost.EndsWith(".$b")) {
            return $true
        }
    }
    return $false
}

function Resolve-MemuraiCliPath() {
    $candidates = @(
        "${env:ProgramFiles}\Memurai\memurai-cli.exe",
        "${env:ProgramFiles(x86)}\Memurai\memurai-cli.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return $path
        }
    }
    $cmd = Get-Command memurai-cli -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return ""
}

function Test-LocalMemuraiPing([string]$RedisUrl) {
    $cli = Resolve-MemuraiCliPath
    if (-not $cli) {
        return @{ Ok = $false; Error = "memurai_cli_not_found" }
    }
    $hostName = Get-RedisHostFromUrl $RedisUrl
    if (-not (Test-LoopbackHost $hostName)) {
        return @{ Ok = $false; Error = "redis_host_not_loopback" }
    }
    $port = 6379
    try {
        $uri = [Uri]$RedisUrl
        if ($uri.Port -gt 0) {
            $port = $uri.Port
        }
    }
    catch {
        return @{ Ok = $false; Error = "redis_url_invalid" }
    }
    try {
        $out = & $cli -h $hostName -p $port ping 2>&1
        $text = ($out | Out-String).Trim()
        if ($text -match "PONG") {
            return @{ Ok = $true; Error = "" }
        }
        return @{ Ok = $false; Error = "ping_no_pong:$text" }
    }
    catch {
        return @{ Ok = $false; Error = "ping_failed:$($_.Exception.Message)" }
    }
}

function Test-BackendHealth([int]$Port) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 5
        return ($resp.StatusCode -eq 200 -and ($resp.Content -match "ok"))
    }
    catch {
        return $false
    }
}

function Test-BackendLocalDevIdentity([int]$Port) {
    <#
      Beyond /health: confirm ACE local backend (security off + Redis configured on server).
      Does not prove Render isolation by itself; combined with script-side REDIS_URL checks.
    #>
    if (-not (Test-BackendHealth $port)) {
        return @{ Ok = $false; Error = "health_failed" }
    }
    try {
        $sec = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/security/config" -UseBasicParsing -TimeoutSec 5
        if ($sec.StatusCode -ne 200) {
            return @{ Ok = $false; Error = "security_config_http_$($sec.StatusCode)" }
        }
        $body = $sec.Content
        if ($body -notmatch '"securityEnabled"\s*:\s*false') {
            return @{ Ok = $false; Error = "security_not_local_off" }
        }
    }
    catch {
        return @{ Ok = $false; Error = "security_config_unreachable" }
    }
    try {
        $probe = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/api/video-status?jobId=ace-local-worker-env-probe" -UseBasicParsing -TimeoutSec 10
        if ($probe.StatusCode -eq 503) {
            return @{ Ok = $false; Error = "backend_redis_unconfigured" }
        }
        if ($probe.StatusCode -eq 404) {
            return @{ Ok = $true; Error = "" }
        }
        return @{ Ok = $false; Error = "video_status_unexpected_$($probe.StatusCode)" }
    }
    catch {
        $resp = $_.Exception.Response
        if ($resp) {
            $code = [int]$resp.StatusCode
            if ($code -eq 404) {
                return @{ Ok = $true; Error = "" }
            }
            if ($code -eq 503) {
                return @{ Ok = $false; Error = "backend_redis_unconfigured" }
            }
            return @{ Ok = $false; Error = "video_status_http_$code" }
        }
        return @{ Ok = $false; Error = "video_status_unreachable" }
    }
}

function Resolve-AceBackendRoot([string]$StartPath) {
    $dir = (Resolve-Path $StartPath).Path
    for ($i = 0; $i -lt 8; $i++) {
        $appPy = Join-Path $dir "app.py"
        $workerPy = Join-Path $dir "worker_video.py"
        if ((Test-Path -LiteralPath $appPy) -and (Test-Path -LiteralPath $workerPy)) {
            return $dir
        }
        $nested = Join-Path $dir "ACE-Backend"
        if ((Test-Path -LiteralPath (Join-Path $nested "app.py"))) {
            return (Resolve-Path $nested).Path
        }
        $parent = Split-Path $dir -Parent
        if (-not $parent -or $parent -eq $dir) {
            break
        }
        $dir = $parent
    }
    return ""
}

function Resolve-AceFrontendRoot([string]$BackendRoot) {
    if (-not $BackendRoot) {
        return ""
    }
    $sibling = Join-Path (Split-Path $BackendRoot -Parent) "ACE-Frontend"
    if ((Test-Path -LiteralPath (Join-Path $sibling "package.json"))) {
        return (Resolve-Path $sibling).Path
    }
    $dir = Split-Path $BackendRoot -Parent
    for ($i = 0; $i -lt 8; $i++) {
        $nested = Join-Path $dir "ACE-Frontend"
        if ((Test-Path -LiteralPath (Join-Path $nested "package.json"))) {
            return (Resolve-Path $nested).Path
        }
        $parent = Split-Path $dir -Parent
        if (-not $parent -or $parent -eq $dir) {
            break
        }
        $dir = $parent
    }
    return ""
}

function Get-AceNormalizedPath([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    try {
        return ([System.IO.Path]::GetFullPath($Path)).TrimEnd('\', '/').ToLowerInvariant()
    }
    catch {
        return $Path.Trim().ToLowerInvariant()
    }
}

function Test-Win32CommandLineIsAceBackendApp {
    param(
        [string]$CommandLine,
        [Parameter(Mandatory = $true)][string]$BackendRoot
    )
    if ([string]::IsNullOrWhiteSpace($CommandLine)) {
        return $false
    }
    if ($CommandLine -notmatch "app\.py") {
        return $false
    }
    $rootNorm = Get-AceNormalizedPath $BackendRoot
    $appNorm = Get-AceNormalizedPath (Join-Path $BackendRoot "app.py")
    $cmdNorm = $CommandLine.ToLowerInvariant()
    if ($rootNorm -and $cmdNorm.Contains($rootNorm)) {
        return $true
    }
    if ($appNorm -and $cmdNorm.Contains($appNorm)) {
        return $true
    }
    return $false
}

function Test-Win32ProcessIsAceBackendApp {
    param(
        [Parameter(Mandatory = $true)]$Win32Process,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [int]$ListenerPort = 0
    )
    $cmd = [string]$Win32Process.CommandLine
    if (Test-Win32CommandLineIsAceBackendApp -CommandLine $cmd -BackendRoot $BackendRoot) {
        return $true
    }
    if ($cmd -notmatch "app\.py") {
        return $false
    }
    $exe = [string]$Win32Process.ExecutablePath
    if (-not $exe) {
        return $false
    }
    $exeNorm = Get-AceNormalizedPath $exe
    $venvPyNorm = Get-AceNormalizedPath (Join-Path $BackendRoot ".venv312\Scripts\python.exe")
    if ($venvPyNorm -and ($exeNorm -eq $venvPyNorm)) {
        return $true
    }
    $rootNorm = Get-AceNormalizedPath $BackendRoot
    if ($rootNorm -and $exeNorm.Contains($rootNorm)) {
        return $true
    }
    if ($ListenerPort -gt 0 -and ($cmd -match "\sapp\.py(\s|$|"")")) {
        $listeners = @(Get-AceLoopbackListenerProcessIds -Port $ListenerPort)
        if ($listeners -contains [int]$Win32Process.ProcessId) {
            $identity = Test-BackendLocalDevIdentity -Port $ListenerPort
            if ($identity.Ok) {
                return $true
            }
        }
    }
    return $false
}

function Get-Win32ParentProcessId {
    param([Parameter(Mandatory = $true)][int]$ProcessId)
    if ($ProcessId -le 0) {
        return 0
    }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    if (-not $proc) {
        return 0
    }
    return [int]$proc.ParentProcessId
}

function Get-AceBackendIndependentTreeCount {
    param(
        [Parameter(Mandatory = $true)][array]$BackendProcesses,
        [hashtable]$ParentProcessIdByPid = $null
    )
    if ($null -eq $BackendProcesses -or $BackendProcesses.Count -eq 0) {
        return 0
    }
    $pidSet = @{}
    foreach ($p in $BackendProcesses) {
        if ($null -eq $p) { continue }
        $pidSet[[int]$p.ProcId] = $true
    }
    if ($pidSet.Count -eq 0) {
        return 0
    }

    function Get-ResolvedParentProcessId([int]$ProcId) {
        if ($ParentProcessIdByPid -and $ParentProcessIdByPid.ContainsKey($ProcId)) {
            return [int]$ParentProcessIdByPid[$ProcId]
        }
        return Get-Win32ParentProcessId -ProcessId $ProcId
    }

    $roots = @{}
    foreach ($p in $BackendProcesses) {
        if ($null -eq $p) { continue }
        $current = [int]$p.ProcId
        if (-not $pidSet.ContainsKey($current)) { continue }
        while ($true) {
            $parent = Get-ResolvedParentProcessId -ProcId $current
            if ($parent -gt 0 -and $pidSet.ContainsKey($parent)) {
                $current = $parent
                continue
            }
            break
        }
        $roots[$current] = $true
    }
    return $roots.Count
}

function Find-LocalAceBackendProcesses {
    param(
        [string]$BackendRoot,
        [int]$ListenerPort = 0
    )
    $found = @()
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        if (-not (Test-Win32ProcessIsAceBackendApp -Win32Process $_ -BackendRoot $BackendRoot -ListenerPort $ListenerPort)) {
            return
        }
        $found += [PSCustomObject]@{
            ProcId      = $_.ProcessId
            CommandLine = [string]$_.CommandLine
        }
    }
    return @($found)
}

function Test-TcpPortOpen([int]$Port) {
    try {
        $r = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -WarningAction SilentlyContinue
        return [bool]$r.TcpTestSucceeded
    }
    catch {
        return $false
    }
}

function Get-AceLoopbackListenerProcessIds([int]$Port) {
    $pids = @()
    try {
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
            $addr = ([string]$_.LocalAddress).Trim().ToLowerInvariant()
            if ($addr -in @("127.0.0.1", "::1", "0.0.0.0")) {
                if ($_.OwningProcess -gt 0) {
                    $pids += [int]$_.OwningProcess
                }
            }
        }
    }
    catch {
        return @()
    }
    return @($pids | Sort-Object -Unique)
}

function Resolve-AceBackendPythonExe([Parameter(Mandatory = $true)][string]$BackendRoot) {
    $venvPy = Join-Path $BackendRoot ".venv312\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPy) {
        return (Resolve-Path -LiteralPath $venvPy).Path
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return [string]$pythonCmd.Source
    }
    return ""
}

function Assert-BackendLocalEnvSafety {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][int]$BackendPort
    )
    if ((Get-EnvString "ACE_SECURITY_ENABLED").ToLowerInvariant() -notin @("false", "0", "no", "off")) {
        Write-Warn "ACE_SECURITY_ENABLED is not false; local dev expects security off."
        Write-Warn "Set ACE_SECURITY_ENABLED=false in .env.local for Ori local mode."
    }
    $redisUrl = Get-EnvString "REDIS_URL"
    if ($redisUrl) {
        $redisHost = Get-RedisHostFromUrl $redisUrl
        if ($redisHost -eq "__invalid__") {
            Write-Err "REDIS_URL is not a valid URL. Use loopback only, e.g. redis://127.0.0.1:6379/0"
            exit 1
        }
        if (-not (Test-LoopbackHost $redisHost)) {
            Write-Err "REDIS_URL must point to local Redis (127.0.0.1 / localhost), not production."
            exit 1
        }
    }
    if (Test-ProductionPublicBaseUrl (Get-EnvString "ACE_PUBLIC_BASE_URL")) {
        Write-Err "ACE_PUBLIC_BASE_URL looks like production (Render / public site)."
        Write-Err "Remove it or set http://127.0.0.1:$BackendPort for local use."
        exit 1
    }
}

function Get-AceBackendLocalPortState {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)][string]$BackendRoot,
        [hashtable]$Simulate = $null
    )
    if ($Simulate) {
        $simTreeCount = 0
        if ($Simulate.ContainsKey('AceBackendTreeCount')) {
            $simTreeCount = [int]$Simulate.AceBackendTreeCount
        }
        return [PSCustomObject]@{
            Port                     = $Port
            BackendRoot              = $BackendRoot
            HealthOk                 = [bool]$Simulate.HealthOk
            AceBackendProcessCount   = [int]$Simulate.AceBackendProcessCount
            AceBackendTreeCount      = $simTreeCount
            AceBackendProcesses      = @()
            ListenerPids             = @($Simulate.ListenerPids)
            AceListenerPids          = @($Simulate.AceListenerPids)
            ForeignListenerPids      = @($Simulate.ForeignListenerPids)
            UncertainListenerPids    = @($Simulate.UncertainListenerPids)
            Action                   = [string]$Simulate.Action
            Reason                   = [string]$Simulate.Reason
        }
    }

    $listenerPids = @(Get-AceLoopbackListenerProcessIds -Port $Port)
    $backendProcs = @(Find-LocalAceBackendProcesses -BackendRoot $BackendRoot -ListenerPort $Port)
    $aceListeners = @()
    $foreignListeners = @()
    $uncertainListeners = @()

    foreach ($procId in $listenerPids) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
        if (-not $proc) {
            $uncertainListeners += $procId
            continue
        }
        if (Test-Win32ProcessIsAceBackendApp -Win32Process $proc -BackendRoot $BackendRoot -ListenerPort $Port) {
            $aceListeners += $procId
        }
        elseif ([string]$proc.Name -match "python") {
            $foreignListeners += $procId
        }
        else {
            $foreignListeners += $procId
        }
    }

    $healthOk = Test-BackendHealth -Port $Port
    $backendTreeCount = Get-AceBackendIndependentTreeCount -BackendProcesses $backendProcs
    $action = "Start"
    $reason = "port_available"

    if ($backendTreeCount -gt 1) {
        $action = "BlockedDuplicate"
        $reason = "multiple_independent_backend_trees"
    }
    elseif ($aceListeners.Count -gt 1) {
        $action = "BlockedDuplicate"
        $reason = "multiple_ace_backend_listeners"
    }
    elseif ($foreignListeners.Count -gt 0) {
        $action = "BlockedForeign"
        $reason = "foreign_process_on_port"
    }
    elseif ($uncertainListeners.Count -gt 0) {
        $action = "BlockedUncertain"
        $reason = "uncertain_listener_process"
    }
    elseif ($aceListeners.Count -eq 1) {
        if ($healthOk) {
            $action = "AlreadyRunning"
            $reason = "ace_backend_listening_health_ok"
        }
        else {
            $action = "BlockedUnhealthy"
            $reason = "ace_backend_listening_health_failed"
        }
    }
    elseif ($listenerPids.Count -gt 0) {
        $action = "BlockedUncertain"
        $reason = "listeners_unclassified"
    }
    elseif ($healthOk) {
        $action = "BlockedUncertain"
        $reason = "health_ok_without_ace_listener"
    }
    elseif ((Test-TcpPortOpen -Port $Port) -and ($listenerPids.Count -eq 0)) {
        $action = "BlockedUncertain"
        $reason = "port_in_use_listener_unknown"
    }
    elseif ($backendTreeCount -eq 1 -and $backendProcs.Count -ge 1 -and $aceListeners.Count -eq 0 -and $listenerPids.Count -eq 0) {
        $action = "BlockedDuplicate"
        $reason = "orphan_ace_backend_process_port_free"
    }

    return [PSCustomObject]@{
        Port                   = $Port
        BackendRoot            = $BackendRoot
        HealthOk               = $healthOk
        AceBackendProcessCount = $backendProcs.Count
        AceBackendTreeCount    = $backendTreeCount
        AceBackendProcesses    = $backendProcs
        ListenerPids           = $listenerPids
        AceListenerPids        = $aceListeners
        ForeignListenerPids    = $foreignListeners
        UncertainListenerPids  = $uncertainListeners
        Action                 = $action
        Reason                 = $reason
    }
}

function Write-AceBackendLocalPortStateReport {
    param(
        [Parameter(Mandatory = $true)]$State
    )
    Write-Info ("Port {0}: listeners={1} ace={2} foreign={3} uncertain={4}" -f `
            $State.Port, `
            ($State.ListenerPids -join ","), `
            ($State.AceListenerPids -join ","), `
            ($State.ForeignListenerPids -join ","), `
            ($State.UncertainListenerPids -join ","))
    Write-Info ("ACE Backend app.py processes in repo: {0}" -f $State.AceBackendProcessCount)
    Write-Info ("Health /health: {0}" -f $(if ($State.HealthOk) { "OK" } else { "not OK" }))
    Write-Info ("Decision: {0} ({1})" -f $State.Action, $State.Reason)
}

function Test-AceFrontendViteOnPort([int]$Port) {
    $candidates = @(
        "http://127.0.0.1:$Port/src/services/backendUrl.js",
        "http://localhost:$Port/src/services/backendUrl.js"
    )
    foreach ($url in $candidates) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200 -and $resp.Content -match "LOCAL_BACKEND_URL" -and $resp.Content -match "127\.0\.0\.1:5000") {
                return $true
            }
        }
        catch {
            continue
        }
    }
    return $false
}

function Find-AceFrontendVitePorts([int]$FromPort = 3000, [int]$ToPort = 3010) {
    $found = @()
    for ($port = $FromPort; $port -le $ToPort; $port++) {
        if (Test-AceFrontendViteOnPort $port) {
            $found += $port
        }
    }
    return @($found)
}

function Wait-BackendLocalReady([int]$Port, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $check = Test-BackendLocalDevIdentity -Port $Port
        if ($check.Ok) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Wait-AceFrontendVite([int]$FromPort, [int]$ToPort, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $ports = @(Find-AceFrontendVitePorts -FromPort $FromPort -ToPort $ToPort)
        if ($ports.Count -gt 0) {
            return $ports[0]
        }
        Start-Sleep -Seconds 1
    }
    return $null
}

function Find-LocalAceWorkerProcesses([string]$BackendRoot = "") {
    $found = @()
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = $_.CommandLine
        if (-not ($cmd -and ($cmd -match "worker_video\.py"))) {
            return
        }
        if ($BackendRoot -and ($cmd -notmatch [regex]::Escape($BackendRoot))) {
            return
        }
        $found += [PSCustomObject]@{
            ProcId      = $_.ProcessId
            CommandLine = $cmd
        }
    }
    return @($found)
}

function Assert-LocalDevEnvFile([string]$RepoRoot) {
    $envFile = Join-Path $RepoRoot ".env.local"
    if (-not (Import-DotEnvFile $envFile)) {
        Write-Err "Missing file: .env.local"
        Write-Err "Copy .env.local.example to .env.local and configure local dev."
        exit 1
    }
    if ((Get-EnvString "ACE_LOCAL_DEV_ENABLED").ToLowerInvariant() -notin @("true", "1", "yes", "on")) {
        Write-Err "ACE_LOCAL_DEV_ENABLED must be true in .env.local"
        exit 1
    }
    if ((Get-EnvString "BUILDER1_PRODUCTION_MODE").ToLowerInvariant() -in @("true", "1", "yes", "on")) {
        Write-Err "BUILDER1_PRODUCTION_MODE must stay off for local Ori dev."
        exit 1
    }
}

function Assert-LocalRedisUrl([string]$ExpectedUrl) {
    $redisUrl = Get-EnvString "REDIS_URL"
    if (-not $redisUrl) {
        Write-Err "REDIS_URL is required for Worker (e.g. $ExpectedUrl)."
        exit 1
    }
    if ($redisUrl -ne $ExpectedUrl) {
        Write-Err "Worker local requires exact REDIS_URL=$ExpectedUrl"
        Write-Err "Current REDIS_URL=$redisUrl"
        exit 1
    }
    $redisHost = Get-RedisHostFromUrl $redisUrl
    if ($redisHost -eq "__invalid__" -or -not (Test-LoopbackHost $redisHost)) {
        Write-Err "REDIS_URL must use loopback host only (127.0.0.1)."
        exit 1
    }
}

function Assert-LocalPublicBaseUrl([string]$ExpectedBase, [int]$BackendPort) {
    $pub = Get-EnvString "ACE_PUBLIC_BASE_URL"
    if (-not $pub) {
        $pub = $ExpectedBase
    }
    if (Test-ProductionPublicBaseUrl $pub) {
        Write-Err "ACE_PUBLIC_BASE_URL must not point to Render or the public site."
        exit 1
    }
    try {
        $uri = [Uri]$pub
        $hostName = $uri.Host.ToLowerInvariant()
        if (-not (Test-LoopbackHost $hostName)) {
            Write-Err "ACE_PUBLIC_BASE_URL must use 127.0.0.1 for local Worker/Backend uploads."
            exit 1
        }
        if ($uri.Port -gt 0 -and $uri.Port -ne $BackendPort) {
            Write-Warn "ACE_PUBLIC_BASE_URL port $($uri.Port) differs from Backend PORT=$BackendPort."
        }
    }
    catch {
        Write-Err "ACE_PUBLIC_BASE_URL is not a valid URL: $pub"
        exit 1
    }
    Set-Item -Path "Env:ACE_PUBLIC_BASE_URL" -Value $pub.TrimEnd("/")
}
