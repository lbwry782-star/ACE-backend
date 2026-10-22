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

function Find-LocalAceBackendProcesses([string]$BackendRoot) {
    $found = @()
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = $_.CommandLine
        if (-not $cmd) {
            return
        }
        if ($cmd -notmatch "app\.py") {
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
