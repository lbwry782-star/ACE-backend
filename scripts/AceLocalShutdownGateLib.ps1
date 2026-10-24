# ACE LOCAL — Shutdown Gate PowerShell bridge (calls engine/ace_local_dev_shutdown_gate.py).
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-AceLocalShutdownGateBackendRoot {
    return (Resolve-Path (Join-Path $ScriptDir '..')).ProviderPath
}

function Resolve-AceLocalShutdownGatePython {
    param([Parameter(Mandatory = $true)][string]$BackendRoot)
    $candidates = @(
        (Join-Path $BackendRoot '.venv312\Scripts\python.exe'),
        (Join-Path $BackendRoot 'venv\Scripts\python.exe'),
        (Join-Path $BackendRoot '.venv\Scripts\python.exe')
    )
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) {
            return $path
        }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Test-AceLocalSafeQuitAceRootIsolated {
    param([Parameter(Mandatory = $true)][string]$AceRoot)
    if (-not (Test-Path -LiteralPath $AceRoot)) {
        return $false
    }
    $tempRoot = [System.IO.Path]::GetTempPath().TrimEnd('\', '/')
    try {
        $resolved = (Resolve-Path -LiteralPath $AceRoot).ProviderPath.TrimEnd('\')
    }
    catch {
        return $false
    }
    return $resolved.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase)
}

function Invoke-AceLocalShutdownGatePython {
    param([Parameter(Mandatory = $true)]$RequestObject)
    $backendRoot = Get-AceLocalShutdownGateBackendRoot
    $py = Resolve-AceLocalShutdownGatePython -BackendRoot $backendRoot
    if (-not $py) {
        throw 'SAFE_QUIT_GATE_PYTHON_MISSING'
    }
    $reqPath = Join-Path $env:TEMP ("ace-shutdown-gate-req-{0}.json" -f [guid]::NewGuid().ToString('N'))
    $runnerPath = Join-Path $env:TEMP ("ace-shutdown-gate-runner-{0}.py" -f [guid]::NewGuid().ToString('N'))
    $runnerPy = @'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
req = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
ace = Path(req["aceRoot"])
action = req["action"]

from engine.ace_local_dev_shutdown_gate import (
    GATE_SAFE_TO_STOP,
    GATE_SHUTDOWN_BLOCKED,
    GATE_SHUTDOWN_REQUESTED,
    GATE_WAITING_FOR_OPERATION,
    local_dev_operation_start_allowed,
    observe_active_local_dev_operations,
    read_shutdown_gate_state,
    refresh_local_shutdown_gate,
    request_local_shutdown,
)

out = {"ok": True, "action": action}
sessions_root = Path(req["sessionsRoot"]) if req.get("sessionsRoot") else None

if action == "request":
    out["state"] = request_local_shutdown(
        ace,
        launcher_session=str(req.get("launcherSession") or ""),
        reason=str(req.get("reason") or "safe_quit_isolated_test"),
    )
elif action == "observe":
    scan = observe_active_local_dev_operations(
        ace_root=ace,
        sessions_root=sessions_root,
        execute_busy=req.get("executeBusy"),
        backend_reachable=bool(req.get("backendReachable", True)),
        stage_timeout_seconds=req.get("stageTimeoutSeconds"),
        redis_reachable=req.get("redisReachable"),
        worker_reachable=req.get("workerReachable"),
        builder_activity_override=req.get("builderActivityOverride"),
    )
    out["scan"] = scan
elif action == "refresh":
    out["state"] = refresh_local_shutdown_gate(
        ace,
        launcher_session=req.get("launcherSession"),
        execute_busy=req.get("executeBusy"),
        backend_reachable=bool(req.get("backendReachable", True)),
        sessions_root=sessions_root,
        stage_timeout_seconds=req.get("stageTimeoutSeconds"),
        redis_reachable=req.get("redisReachable"),
        worker_reachable=req.get("workerReachable"),
        builder_activity_override=req.get("builderActivityOverride"),
    )
elif action == "read":
    out["state"] = read_shutdown_gate_state(ace)
elif action == "operation_start_allowed":
    allowed, reason = local_dev_operation_start_allowed(
        ace,
        expected_launcher_session=req.get("expectedLauncherSession"),
    )
    out["allowed"] = allowed
    out["reason"] = reason
else:
    out = {"ok": False, "error": "unknown_action:" + str(action)}

print(json.dumps(out, ensure_ascii=False))
'@
    try {
        $json = $RequestObject | ConvertTo-Json -Depth 12 -Compress
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($reqPath, $json, $utf8NoBom)
        [System.IO.File]::WriteAllText($runnerPath, $runnerPy, $utf8NoBom)
        $raw = & $py $runnerPath $backendRoot $reqPath 2>&1
        $text = ($raw | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw ('SAFE_QUIT_GATE_PYTHON_EXIT:' + $text)
        }
        if ($text -match '(?m)^Traceback ') {
            throw ('SAFE_QUIT_GATE_PYTHON_TRACEBACK:' + $text)
        }
        $line = ($raw | Out-String).Trim().Split("`n") | Where-Object { $_.Trim().StartsWith('{') } | Select-Object -Last 1
        if (-not $line) {
            throw 'SAFE_QUIT_GATE_PYTHON_EMPTY_OUTPUT'
        }
        return ($line | ConvertFrom-Json)
    }
    finally {
        Remove-Item -LiteralPath $reqPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $runnerPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-AceLocalSafeQuitShutdownGatePhase {
    param(
        [Parameter(Mandatory = $true)][string]$AceRoot,
        [Parameter(Mandatory = $true)][string]$LauncherSession,
        [int]$WaitSeconds = 30,
        [int]$PollMilliseconds = 300,
        [string]$SessionsRoot = '',
        [switch]$TestSimulateGateThrow
    )
    if (-not (Test-AceLocalSafeQuitAceRootIsolated -AceRoot $AceRoot)) {
        throw 'SAFE_QUIT_GATE_ACE_ROOT_NOT_ISOLATED'
    }
    $session = ($LauncherSession -replace '\s', '').Trim()
    if (-not $session) {
        throw 'SAFE_QUIT_GATE_LAUNCHER_SESSION_REQUIRED'
    }
    if ($TestSimulateGateThrow) {
        throw 'SAFE_QUIT_TEST_SIMULATED_GATE_FAILURE'
    }

    $sessionsArg = $SessionsRoot
    if (-not $sessionsArg) {
        $sessionsArg = Join-Path $AceRoot '_backups\sessions'
    }

    $requestPayload = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action          = 'request'
        aceRoot         = $AceRoot
        launcherSession = $session
        reason          = 'safe_quit_isolated_gate_test'
    }
    $observePayload = Invoke-AceLocalShutdownGatePython -RequestObject @{
        action           = 'observe'
        aceRoot          = $AceRoot
        sessionsRoot     = $sessionsArg
        backendReachable = $true
    }

    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    $lastState = $null
    $terminalStatus = $null
    do {
        $refreshPayload = Invoke-AceLocalShutdownGatePython -RequestObject @{
            action           = 'refresh'
            aceRoot          = $AceRoot
            launcherSession  = $session
            sessionsRoot     = $sessionsArg
            backendReachable = $true
        }
        $lastState = $refreshPayload.state
        $status = [string]$lastState.gateStatus
        if ($status -eq 'SAFE_TO_STOP') {
            $terminalStatus = 'SAFE_TO_STOP'
            break
        }
        if ($status -eq 'SHUTDOWN_BLOCKED') {
            $terminalStatus = 'SHUTDOWN_BLOCKED'
            break
        }
        if ($status -in @('SHUTDOWN_REQUESTED', 'WAITING_FOR_OPERATION')) {
            Start-Sleep -Milliseconds $PollMilliseconds
            continue
        }
        Start-Sleep -Milliseconds $PollMilliseconds
    } while ((Get-Date) -lt $deadline)

    if (-not $terminalStatus) {
        $status = if ($lastState) { [string]$lastState.gateStatus } else { 'UNKNOWN' }
        return @{
            Ok             = $false
            Outcome        = 'timeout'
            GateStatus     = $status
            RequestState   = $requestPayload.state
            ObserveScan    = $observePayload.scan
            FinalState     = $lastState
        }
    }

    $ok = ($terminalStatus -eq 'SAFE_TO_STOP')
    return @{
        Ok             = $ok
        Outcome        = if ($ok) { 'safe_to_stop' } else { 'blocked' }
        GateStatus     = $terminalStatus
        RequestState   = $requestPayload.state
        ObserveScan    = $observePayload.scan
        FinalState     = $lastState
    }
}
